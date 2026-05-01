# visualize_different_bandwidth.py
import os
import re
import math
import glob
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================
# 0) 你在这里直接配置（全部配置放最前面）
# =========================

# baseline predictions.pt 路径
BASELINE_PATH = r"outputs/token_attention/scene1_DTU/add_epipolar_band/baseline/predictions.pt"

# 你的 bandwidth 文件夹所在的父目录
# 例：里面有 layer_10_to_15_bw0.5 / layer_10_to_15_bw1 / layer_10_to_15_bw1.5 / ...
BAND_PARENT_DIR = r"outputs/token_attention/scene1_DTU/add_epipolar_band/bandwidth_compare"
# 子目录命名模式（用 glob）
BAND_DIR_GLOB = "layer_16_to_23_bw*"

# 每个 bandwidth 文件夹中 predictions 的文件名
PRED_FILENAME = "predictions.pt"

# 输出目录（通常和 BAND_PARENT_DIR 一致）
OUT_DIR = r"outputs/token_attention/scene1_DTU/add_epipolar_band/bandwidth_compare"

# 下采样步长：1=不下采样；4=每隔4个像素取一个点（更快）
STRIDE = 4

# 是否使用 conf 作为权重（更稳健：关注模型更确信的区域）
USE_CONF_WEIGHT = True

# conf 的统计方式（用于柱形图）：mean 或 median
CONF_STAT = "mean"  # "mean" / "median"

# 是否画“按 view 的 conf 柱形图”（你当前代码里未实现 per-view 图，这里保留开关以免你后续扩展）
CONF_BAR_PER_VIEW = False

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
    if weight is None:
        return err.mean().item()
    w = torch.clamp(weight, min=0.0)
    num = (err * w).sum()
    den = w.sum().clamp(min=eps)
    return (num / den).item()

def stat_conf(conf, mode="mean"):
    if conf is None:
        return np.nan
    if mode == "mean":
        return conf.mean().item()
    if mode == "median":
        return conf.median().item()
    raise ValueError(f"未知 CONF_STAT: {mode}")

def compute_depth_delta(d0, d1, w=None, eps=1e-6):
    """
    衡量 band 与 baseline 的差距（越小越好）
    d0,d1: [B,S,H,W,1] 或 [B,S,H,W]
    w: [B,S,H,W]（可选）
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
    p0,p1: [B,S,H,W,3]
    w: [B,S,H,W]（可选）
    """
    diff = p1 - p0
    l2 = torch.sqrt((diff * diff).sum(dim=-1) + eps)  # [B,S,H,W]
    l2_sq = l2.pow(2)

    mean_l2 = masked_weighted_mean(l2, w)
    rmse_l2 = math.sqrt(masked_weighted_mean(l2_sq, w))
    return {"pts_l2_mean": mean_l2, "pts_l2_rmse": rmse_l2}

def compute_poseenc_delta(pe0, pe1):
    """
    仅衡量与 baseline 的差距（越小越好），不代表真实pose误差
    pe0,pe1: [B,S,9]
    """
    diff = pe1 - pe0
    l2 = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # [B,S]
    l2_sq = l2.pow(2)

    mean_l2 = l2.mean().item()
    rmse_l2 = math.sqrt(l2_sq.mean().item())
    return {"pose_l2_mean": mean_l2, "pose_l2_rmse": rmse_l2}

def parse_bw_from_dirname(path: str):
    """
    从目录名里提取 bw，支持：
      layer_10_to_15_bw0.5
      layer_10_to_15_bw1
      layer_10_to_15_bw2.5
    """
    name = os.path.basename(os.path.normpath(path))
    m = re.search(r"bw(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None

def bw_to_label(bw: float) -> str:
    """把 1.0 显示成 '1'，0.5 显示成 '0.5'"""
    if float(bw).is_integer():
        return str(int(bw))
    return str(bw)

# =========================
# 2) 主流程
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------- 读 baseline -------
    base = torch.load(BASELINE_PATH, map_location="cpu")
    d0  = downsample_hw(safe_get(base, "depth"), STRIDE)
    dc0 = downsample_hw(safe_get(base, "depth_conf"), STRIDE)
    p0  = downsample_hw(safe_get(base, "world_points"), STRIDE)
    pc0 = downsample_hw(safe_get(base, "world_points_conf"), STRIDE)
    pe0 = safe_get(base, "pose_enc")

    print("[Baseline]", BASELINE_PATH)
    print("  has depth:", d0 is not None, "| has points:", p0 is not None, "| has pose:", pe0 is not None)

    # ------- 找所有 bandwidth 文件夹 -------
    band_dirs = sorted(glob.glob(os.path.join(BAND_PARENT_DIR, BAND_DIR_GLOB)))
    band_items = []
    for bd in band_dirs:
        bw = parse_bw_from_dirname(bd)
        if bw is None:
            continue
        pred_path = os.path.join(bd, PRED_FILENAME)
        if not os.path.isfile(pred_path):
            print(f"[Skip] 找到目录但缺少 predictions：{pred_path}")
            continue
        band_items.append((bw, bd, pred_path))

    band_items = sorted(band_items, key=lambda x: x[0])
    if len(band_items) == 0:
        raise FileNotFoundError(f"[错误] 没有匹配到 band 结果：{os.path.join(BAND_PARENT_DIR, BAND_DIR_GLOB)}")

    print("\n找到 bandwidth 结果：")
    for bw, bd, pp in band_items:
        print(f"  bw={bw_to_label(bw):<4s} | {pp}")

    # baseline 的 conf 统计
    base_depth_conf_stat = stat_conf(dc0, CONF_STAT)
    base_pts_conf_stat   = stat_conf(pc0, CONF_STAT)

    rows = []
    bws = []

    # 用于画图：整体 conf
    depth_conf_stats = []
    pts_conf_stats = []

    # 用于画图：差异曲线
    depth_mae_list = []
    depth_rel_list = []
    pts_l2_list = []
    pose_l2_list = []

    for bw, bd, pred_path in band_items:
        pred = torch.load(pred_path, map_location="cpu")
        d1  = downsample_hw(safe_get(pred, "depth"), STRIDE)
        dc1 = downsample_hw(safe_get(pred, "depth_conf"), STRIDE)
        p1  = downsample_hw(safe_get(pred, "world_points"), STRIDE)
        pc1 = downsample_hw(safe_get(pred, "world_points_conf"), STRIDE)
        pe1 = safe_get(pred, "pose_enc")

        # 权重（可选）：用 band 的 conf（更符合“band设置下模型确信区域”）
        depth_w = None
        pts_w = None
        if USE_CONF_WEIGHT:
            depth_w = dc1 if dc1 is not None else dc0
            pts_w = pc1 if pc1 is not None else pc0

        row = {"bw": bw, "pred_path": pred_path}

        # conf（越大越好）
        row["depth_conf"] = stat_conf(dc1, CONF_STAT)
        row["world_points_conf"] = stat_conf(pc1, CONF_STAT)

        # 与 baseline 的差距（越小越好）
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

        # 收集画图数据
        bws.append(bw)
        depth_conf_stats.append(row["depth_conf"])
        pts_conf_stats.append(row["world_points_conf"])

        depth_mae_list.append(row["depth_mae"])
        depth_rel_list.append(row["depth_rel_mae"])
        pts_l2_list.append(row["pts_l2_mean"])
        pose_l2_list.append(row["pose_l2_mean"])

        print(f"[Done] bw={bw_to_label(bw):<4s} | depth_conf={row['depth_conf']:.6f} | pts_conf={row['world_points_conf']:.6f} "
              f"| depth_mae={row['depth_mae']:.6f} | pts_l2={row['pts_l2_mean']:.6f} | pose_l2={row['pose_l2_mean']:.6f}")

    # ------- 保存 CSV -------
    csv_path = os.path.join(OUT_DIR, f"bandwidth_vs_baseline_S{STRIDE}_CONF_{USE_CONF_WEIGHT}.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[Saved] {csv_path}")

    # x 轴标签（确保 1.0 显示为 1）
    xlabels = [bw_to_label(x) for x in bws]

    # =========================================================
    # 图 1：conf 柱形图（baseline vs 不同 bw）
    # =========================================================
    plt.figure(figsize=(11.5, 4.5))

    # 子图1：depth_conf
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(xlabels, depth_conf_stats)
    ax1.axhline(base_depth_conf_stat, linestyle="--")
    ax1.set_title(f"Depth Conf ({CONF_STAT}): Band vs Baseline")
    ax1.set_xlabel("bandwidth (bw)")
    ax1.set_ylabel("depth_conf")
    ax1.grid(True, axis="y")

    # 子图2：world_points_conf
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(xlabels, pts_conf_stats)
    ax2.axhline(base_pts_conf_stat, linestyle="--")
    ax2.set_title(f"World Points Conf ({CONF_STAT}): Band vs Baseline")
    ax2.set_xlabel("bandwidth (bw)")
    ax2.set_ylabel("world_points_conf")
    ax2.grid(True, axis="y")

    plt.tight_layout()
    fig1_path = os.path.join(OUT_DIR, f"bar_conf_vs_baseline_{CONF_STAT}_S{STRIDE}.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[Saved] {fig1_path}")

    # =========================================================
    # 图 2：差异折线图（band 相对 baseline 差距，越小越好）
    # =========================================================
    plt.figure(figsize=(9.0, 6.0))

    # 用数值 bw 画折线，保证 x 是连续递增
    plt.plot(bws, depth_mae_list, marker="o", label="Depth MAE (vs baseline)")
    plt.plot(bws, depth_rel_list, marker="o", label="Depth Rel-MAE (vs baseline)")
    plt.plot(bws, pts_l2_list, marker="o", label="WorldPts L2 mean (vs baseline)")
    plt.plot(bws, pose_l2_list, marker="o", label="PoseEnc L2 mean (vs baseline)")

    plt.xlabel("bandwidth (bw)")
    plt.ylabel("Difference vs baseline")
    plt.title(f"Bandwidth Sweep (Global L10-15) vs Baseline | stride={STRIDE} | conf_weight={USE_CONF_WEIGHT}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, f"line_diff_vs_bw_S{STRIDE}_CONF_{USE_CONF_WEIGHT}.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[Saved] {fig2_path}")

    print("\n全部完成。输出目录：", OUT_DIR)

if __name__ == "__main__":
    main()
