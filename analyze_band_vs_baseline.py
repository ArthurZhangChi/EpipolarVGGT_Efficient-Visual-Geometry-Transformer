import os
import math
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================
# 0) 你在这里直接配置
# =========================
BASELINE_PATH = r"outputs/token_attention/scene1_DTU/add_epipolar_band/baseline/predictions.pt"
BAND_PATH     = r"outputs/token_attention/scene1_DTU/add_epipolar_band/layer_10_to_15_bw70/predictions.pt"
OUT_DIR       = r"outputs/token_attention/scene1_DTU/add_epipolar_band/layer_10_to_15_bw70"
TAG           = "layer_10_to_15_bw70"

# stride: 对 H,W 下采样的步长，1=不下采样；4=每隔4个像素取一个点（更快）
STRIDE = 4

# 是否使用 conf 作为权重（更稳健：关注模型更确信的区域）
USE_CONF_WEIGHT = True

# conf 的统计方式：用 mean 还是 median（median 对极端值更稳）
CONF_STAT = "mean"  # "mean" / "median"

# 柱形图是否按 view 画（推荐 True：能看出某个 view 是否被 band 影响更大）
BAR_PER_VIEW = True

# =========================
# 1) 工具函数
# =========================
def safe_get(d, k):
    return d[k] if (isinstance(d, dict) and k in d) else None


def downsample_hw(x, stride: int):
    """
    x: [B,S,H,W,*] 或 [B,S,H,W]
    """
    if x is None or stride <= 1:
        return x
    if x.ndim == 5:      # [B,S,H,W,C]
        return x[:, :, ::stride, ::stride, :]
    if x.ndim == 4:      # [B,S,H,W]
        return x[:, :, ::stride, ::stride]
    raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")


def masked_weighted_mean(err, weight=None, eps=1e-12):
    """
    err: tensor
    weight: 与 err broadcast-compatible 的权重（或 None）
    """
    if weight is None:
        return err.mean().item()
    w = torch.clamp(weight, min=0.0)
    num = (err * w).sum()
    den = w.sum().clamp(min=eps)
    return (num / den).item()


def masked_weighted_median(x, weight=None):
    """
    加权 median 实现较复杂；这里给你一个“实用版”：
    - 如果 weight=None：普通 median
    - 如果 weight!=None：用权重做采样近似（足够用于对比）
    """
    if weight is None:
        return x.median().item()

    # 采样近似：按权重归一化后做 multinomial
    # 为了稳，限制采样数量
    flat_x = x.reshape(-1)
    flat_w = torch.clamp(weight.reshape(-1), min=0.0)

    s = flat_w.sum()
    if s.item() <= 0:
        return flat_x.median().item()

    probs = (flat_w / s).to(torch.float32)
    num = min(200000, flat_x.numel())  # 上限 20 万
    idx = torch.multinomial(probs, num_samples=num, replacement=True)
    samp = flat_x[idx]
    return samp.median().item()


def stat_conf(conf, weight=None, mode="mean"):
    if conf is None:
        return np.nan
    if mode == "mean":
        if weight is None:
            return conf.mean().item()
        return masked_weighted_mean(conf, weight)
    if mode == "median":
        return masked_weighted_median(conf, weight)
    raise ValueError(f"Unknown CONF_STAT: {mode}")


def compute_depth_delta(d0, d1, w=None, eps=1e-6):
    """
    d0, d1: [B,S,H,W,1] 或 [B,S,H,W]
    返回 band 相对 baseline 的差距（越小越好）
    """
    if d0.ndim == 5 and d0.shape[-1] == 1:
        d0 = d0[..., 0]
    if d1.ndim == 5 and d1.shape[-1] == 1:
        d1 = d1[..., 0]

    diff = d1 - d0
    abs_err = diff.abs()
    sq_err = diff.pow(2)

    mae = masked_weighted_mean(abs_err, w)
    rmse = math.sqrt(masked_weighted_mean(sq_err, w))

    rel = abs_err / (d0.abs() + eps)
    rel_mae = masked_weighted_mean(rel, w)

    return {"depth_mae": mae, "depth_rmse": rmse, "depth_rel_mae": rel_mae}


def compute_points_delta(p0, p1, w=None, eps=1e-12):
    """
    p0, p1: [B,S,H,W,3]
    返回 band 相对 baseline 的差距（越小越好）
    """
    diff = p1 - p0
    l2 = torch.sqrt((diff * diff).sum(dim=-1) + eps)  # [B,S,H,W]
    l2_sq = l2.pow(2)

    mean_l2 = masked_weighted_mean(l2, w)
    rmse_l2 = math.sqrt(masked_weighted_mean(l2_sq, w))
    return {"pts_l2_mean": mean_l2, "pts_l2_rmse": rmse_l2}


def compute_poseenc_delta(pe0, pe1):
    """
    pe0, pe1: [B,S,9]
    仅衡量与 baseline 的差距（越小越好），不代表真实pose误差
    """
    diff = pe1 - pe0
    l2 = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # [B,S]
    l2_sq = l2.pow(2)

    mean_l2 = l2.mean().item()
    rmse_l2 = math.sqrt(l2_sq.mean().item())
    return {"pose_l2_mean": mean_l2, "pose_l2_rmse": rmse_l2}


def ensure_same_shape(a, b, name):
    if a is None or b is None:
        return
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"[错误] {name} 形状不一致：baseline={tuple(a.shape)}, band={tuple(b.shape)}")


# =========================
# 2) 主流程
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base = torch.load(BASELINE_PATH, map_location="cpu")
    band = torch.load(BAND_PATH, map_location="cpu")

    # ------- 取出 + 下采样 -------
    d0  = downsample_hw(safe_get(base, "depth"), STRIDE)
    dc0 = downsample_hw(safe_get(base, "depth_conf"), STRIDE)
    p0  = downsample_hw(safe_get(base, "world_points"), STRIDE)
    pc0 = downsample_hw(safe_get(base, "world_points_conf"), STRIDE)
    pe0 = safe_get(base, "pose_enc")

    d1  = downsample_hw(safe_get(band, "depth"), STRIDE)
    dc1 = downsample_hw(safe_get(band, "depth_conf"), STRIDE)
    p1  = downsample_hw(safe_get(band, "world_points"), STRIDE)
    pc1 = downsample_hw(safe_get(band, "world_points_conf"), STRIDE)
    pe1 = safe_get(band, "pose_enc")

    print("[Baseline]", BASELINE_PATH)
    print("[Band    ]", BAND_PATH)

    # ------- 形状一致性检查 -------
    ensure_same_shape(d0, d1, "depth")
    ensure_same_shape(dc0, dc1, "depth_conf")
    ensure_same_shape(p0, p1, "world_points")
    ensure_same_shape(pc0, pc1, "world_points_conf")
    ensure_same_shape(pe0, pe1, "pose_enc")

    # ------- conf 作为权重（可选）-------
    depth_w = None
    pts_w = None
    if USE_CONF_WEIGHT:
        # 用 band 的 conf 作为权重更符合“band 设置下模型的确信区域”
        depth_w = dc1 if dc1 is not None else dc0
        pts_w = pc1 if pc1 is not None else pc0

    # =========================
    # A) 全局统计（一个数）
    # =========================
    rows = []

    row = {"tag": TAG, "stride": STRIDE, "use_conf_weight": USE_CONF_WEIGHT, "conf_stat": CONF_STAT}

    # conf：越大越好（我们同时记录 baseline / band / delta）
    row["depth_conf_base"] = stat_conf(dc0, weight=None, mode=CONF_STAT)
    row["depth_conf_band"] = stat_conf(dc1, weight=None, mode=CONF_STAT)
    row["depth_conf_delta"] = row["depth_conf_band"] - row["depth_conf_base"]

    row["pts_conf_base"] = stat_conf(pc0, weight=None, mode=CONF_STAT)
    row["pts_conf_band"] = stat_conf(pc1, weight=None, mode=CONF_STAT)
    row["pts_conf_delta"] = row["pts_conf_band"] - row["pts_conf_base"]

    # depth/points：衡量与 baseline 差距（越小越好）
    if d0 is not None and d1 is not None:
        row.update(compute_depth_delta(d0, d1, w=depth_w))
    else:
        row.update({"depth_mae": np.nan, "depth_rmse": np.nan, "depth_rel_mae": np.nan})

    if p0 is not None and p1 is not None:
        row.update(compute_points_delta(p0, p1, w=pts_w))
    else:
        row.update({"pts_l2_mean": np.nan, "pts_l2_rmse": np.nan})

    if pe0 is not None and pe1 is not None:
        row.update(compute_poseenc_delta(pe0, pe1))
    else:
        row.update({"pose_l2_mean": np.nan, "pose_l2_rmse": np.nan})

    rows.append(row)

    # ------- 保存 CSV（全局）-------
    csv_path = os.path.join(OUT_DIR, f"eval_baseline_vs_{TAG}_S{STRIDE}_CONF.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[Saved] {csv_path}")

    print("\n===== 全局对比（band 相对 baseline）=====")
    print(f"depth_conf: base={row['depth_conf_base']:.6f}, band={row['depth_conf_band']:.6f}, delta={row['depth_conf_delta']:+.6f} (越大越好)")
    print(f"pts_conf  : base={row['pts_conf_base']:.6f}, band={row['pts_conf_band']:.6f}, delta={row['pts_conf_delta']:+.6f} (越大越好)")
    print(f"depth_mae : {row['depth_mae']:.6f} (越小越好)")
    print(f"pts_l2_mean: {row['pts_l2_mean']:.6f} (越小越好)")
    print(f"pose_l2_mean: {row['pose_l2_mean']:.6f} (越小越好，仅用于稳定性)")

    # =========================
    # B) 按 view 的柱形图（推荐）
    # =========================
    if BAR_PER_VIEW and (dc0 is not None or pc0 is not None):
        # 假设 B=1
        S = None
        if dc0 is not None:
            S = dc0.shape[1]
        elif pc0 is not None:
            S = pc0.shape[1]
        else:
            S = 0

        views = list(range(S))

        def per_view_stat(conf_tensor, mode):
            # conf_tensor: [B,S,H,W]
            if conf_tensor is None:
                return [np.nan] * S
            out = []
            for v in range(S):
                x = conf_tensor[0, v]  # [H,W]
                if mode == "mean":
                    out.append(x.mean().item())
                else:
                    out.append(x.median().item())
            return out

        depth_base = per_view_stat(dc0, CONF_STAT)
        depth_band = per_view_stat(dc1, CONF_STAT)
        pts_base = per_view_stat(pc0, CONF_STAT)
        pts_band = per_view_stat(pc1, CONF_STAT)

        # ---- 画 depth_conf 柱状图
        if dc0 is not None and dc1 is not None:
            x = np.arange(S)
            w = 0.35
            plt.figure(figsize=(10, 4))
            plt.bar(x - w/2, depth_base, width=w, label="baseline")
            plt.bar(x + w/2, depth_band, width=w, label="band(10-15)")
            plt.xticks(x, [f"v{v}" for v in views])
            plt.ylabel(f"depth_conf ({CONF_STAT})  ↑越大越好")
            plt.title(f"Depth Confidence per View | {TAG} | stride={STRIDE}")
            plt.grid(True, axis="y")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(OUT_DIR, f"bar_depth_conf_{TAG}_S{STRIDE}.png")
            plt.savefig(fig_path, dpi=200)
            print(f"[Saved] {fig_path}")

        # ---- 画 world_points_conf 柱状图
        if pc0 is not None and pc1 is not None:
            x = np.arange(S)
            w = 0.35
            plt.figure(figsize=(10, 4))
            plt.bar(x - w/2, pts_base, width=w, label="baseline")
            plt.bar(x + w/2, pts_band, width=w, label="band(10-15)")
            plt.xticks(x, [f"v{v}" for v in views])
            plt.ylabel(f"world_points_conf ({CONF_STAT})  ↑越大越好")
            plt.title(f"WorldPoints Confidence per View | {TAG} | stride={STRIDE}")
            plt.grid(True, axis="y")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(OUT_DIR, f"bar_world_points_conf_{TAG}_S{STRIDE}.png")
            plt.savefig(fig_path, dpi=200)
            print(f"[Saved] {fig_path}")

    print("\n分析完成。")


if __name__ == "__main__":
    main()
