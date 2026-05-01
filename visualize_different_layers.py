# visualize_different_layers.py
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

# 各种“不同层设置”的实验文件夹所在父目录
# 例：里面有 layer_10_to_15_bw42 / layer_12_to_17_bw42 / layer_14_to_19_bw42 / ...
EXP_PARENT_DIR = r"outputs/token_attention/scene1_DTU/add_epipolar_band/two_layers/soft_band/alpha_2"
# 子目录 glob（你可以用更宽的匹配，比如 "*layer*" 或 "*L*"
# 推荐先用 "*" 找到所有，再靠脚本解析过滤
EXP_DIR_GLOB = "*"

# 每个实验文件夹中 predictions 的文件名
PRED_FILENAME = "predictions.pt"

# 输出目录
OUT_DIR = r"outputs/token_attention/scene1_DTU/add_epipolar_band/two_layers"

# 下采样步长：1=不下采样；4=每隔4个像素取一个点（更快）
STRIDE = 4

# 是否使用 conf 作为权重（更稳健：关注模型更确信的区域）
USE_CONF_WEIGHT = True

# conf 的统计方式：mean 或 median
CONF_STAT = "mean"  # "mean" / "median"

# 只挑选“包含 predictions.pt 的文件夹”
ONLY_DIRS_WITH_PRED = True

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
    衡量与 baseline 的差距（越小越好）
    d0,d1: [B,S,H,W,1] 或 [B,S,H,W]
    w: [B,S,H,W]
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


def parse_layer_range(name: str):
    """
    尽量从文件夹名解析层范围，返回 (start, end) 或 None

    支持：
      - layer_10_to_15
      - L10_15
      - global10-15 / g10_15
      - 10to15
      - layer_12
      - L7
      - 7
      - anything_with_a_number (fallback: 取第一个数字当作 start=end)
    """
    s = name.replace("-", "_")
    # layer_10_to_15 / layer10to15
    m = re.search(r"layer[_\-]?(\d+)[_\-]?to[_\-]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    # L10_15
    m = re.search(r"\bL(\d+)[_\-](\d+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    # global10_15 / g10_15
    m = re.search(r"\b(global|g)[_\-]?(\d+)[_\-](\d+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(2)), int(m.group(3))
    # 10to15
    m = re.search(r"\b(\d+)[_\-]?to[_\-]?(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    # layer_12 / layer12
    m = re.search(r"\blayer[_\-]?(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return n, n
    # L7
    m = re.search(r"\bL(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return n, n
    # fallback：只要名字里含数字，取第一个数字
    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        return n, n
    return None


def make_label_from_dir(dir_path: str):
    """
    给图上显示的短标签：
    - 默认用目录名
    - 若能解析层范围，则优先显示 L{start}-{end}
    """
    name = os.path.basename(os.path.normpath(dir_path))
    lr = parse_layer_range(name)
    if lr is not None:
        a, b = lr
        return f"L{a}-{b}"
    return name


def natural_key(s: str):
    """
    自然排序 key：把字符串拆成 [文本, 数字, 文本, 数字...]，数字按 int 排序
    """
    parts = re.split(r"(\d+)", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

def sort_key(dir_path: str):
    name = os.path.basename(os.path.normpath(dir_path))
    lr = parse_layer_range(name)
    if lr is not None:
        # 能解析到层号：优先按 (start,end) 排
        return (0, lr[0], lr[1], natural_key(name))
    # 解析不到：也用自然排序
    return (1, 999, 999, natural_key(name))


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

    base_depth_conf = stat_conf(dc0, CONF_STAT)
    base_pts_conf   = stat_conf(pc0, CONF_STAT)

    print("[Baseline]", BASELINE_PATH)

    # ------- 找所有实验文件夹 -------
    cand_dirs = sorted(glob.glob(os.path.join(EXP_PARENT_DIR, EXP_DIR_GLOB)))
    exp_dirs = []
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        pred_path = os.path.join(d, PRED_FILENAME)
        if ONLY_DIRS_WITH_PRED and (not os.path.isfile(pred_path)):
            continue
        # 避免把 baseline 自己也扫进去（可选）
        if os.path.abspath(pred_path) == os.path.abspath(BASELINE_PATH):
            continue
        exp_dirs.append(d)

    exp_dirs = sorted(exp_dirs, key=sort_key)
    if len(exp_dirs) == 0:
        raise FileNotFoundError(f"[错误] 没有找到任何实验文件夹（包含 {PRED_FILENAME}）：{EXP_PARENT_DIR}")

    print("\n找到实验结果：")
    for d in exp_dirs:
        print("  ", os.path.join(d, PRED_FILENAME))

    labels = []
    rows = []

    # 用于画图
    depth_conf_list = []
    pts_conf_list = []
    depth_mae_list = []
    depth_rel_list = []
    pts_l2_list = []
    pose_l2_list = []

    for d in exp_dirs:
        pred_path = os.path.join(d, PRED_FILENAME)
        pred = torch.load(pred_path, map_location="cpu")

        d1  = downsample_hw(safe_get(pred, "depth"), STRIDE)
        dc1 = downsample_hw(safe_get(pred, "depth_conf"), STRIDE)
        p1  = downsample_hw(safe_get(pred, "world_points"), STRIDE)
        pc1 = downsample_hw(safe_get(pred, "world_points_conf"), STRIDE)
        pe1 = safe_get(pred, "pose_enc")

        # 权重（可选）
        depth_w = None
        pts_w = None
        if USE_CONF_WEIGHT:
            depth_w = dc1 if dc1 is not None else dc0
            pts_w   = pc1 if pc1 is not None else pc0

        label = make_label_from_dir(d)
        labels.append(label)

        row = {
            "setting": label,
            "folder": d,
            "pred_path": pred_path
        }

        # conf（越大越好）
        row["depth_conf"] = stat_conf(dc1, CONF_STAT)
        row["world_points_conf"] = stat_conf(pc1, CONF_STAT)

        # 差异（越小越好）
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

        depth_conf_list.append(row["depth_conf"])
        pts_conf_list.append(row["world_points_conf"])
        depth_mae_list.append(row["depth_mae"])
        depth_rel_list.append(row["depth_rel_mae"])
        pts_l2_list.append(row["pts_l2_mean"])
        pose_l2_list.append(row["pose_l2_mean"])

        print(f"[Done] {label:<10s} | depth_conf={row['depth_conf']:.6f} | pts_conf={row['world_points_conf']:.6f} "
              f"| depth_mae={row['depth_mae']:.6f} | pts_l2={row['pts_l2_mean']:.6f} | pose_l2={row['pose_l2_mean']:.6f}")

    # ------- 保存 CSV -------
    csv_path = os.path.join(OUT_DIR, f"layers_vs_baseline_S{STRIDE}_CONF_{USE_CONF_WEIGHT}_soft_band.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[Saved] {csv_path}")

    # =========================================================
    # 图 1：conf 柱形图（baseline 虚线 + 各设定柱子）
    # =========================================================
    plt.figure(figsize=(max(10, 0.6 * len(labels)), 4.8))

    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(labels, depth_conf_list)
    ax1.axhline(base_depth_conf, linestyle="--")
    ax1.set_title(f"Depth Conf({CONF_STAT}): 1 Layer vs Baseline")
    ax1.set_xlabel("Layer Setting")
    ax1.set_ylabel("depth_conf")
    ax1.grid(True, axis="y")
    ax1.tick_params(axis="x", rotation=35)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(labels, pts_conf_list)
    ax2.axhline(base_pts_conf, linestyle="--")
    ax2.set_title(f"World Points Conf({CONF_STAT}): 1 Layer vs Baseline")
    ax2.set_xlabel("Layer Setting")
    ax2.set_ylabel("world_points_conf")
    ax2.grid(True, axis="y")
    ax2.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    fig1_path = os.path.join(OUT_DIR, f"bar_conf_layers_vs_baseline_{CONF_STAT}_S{STRIDE}_soft_band.png")
    plt.savefig(fig1_path, dpi=200)
    print(f"[Saved] {fig1_path}")

    # =========================================================
    # 图 2：差异折线/点线图（横轴：层设定；纵轴：与 baseline 差异）
    # =========================================================
    plt.figure(figsize=(max(10, 0.8 * len(labels)), 6.0))
    x = np.arange(len(labels))

    plt.plot(x, depth_mae_list, marker="o", label="Depth MAE (vs baseline)")
    plt.plot(x, depth_rel_list, marker="o", label="Depth Rel-MAE (vs baseline)")
    plt.plot(x, pts_l2_list, marker="o", label="WorldPts L2 mean (vs baseline)")
    plt.plot(x, pose_l2_list, marker="o", label="PoseEnc L2 mean (vs baseline)")

    plt.xticks(x, labels, rotation=35)
    plt.xlabel("Layer Setting")
    plt.ylabel("Difference vs baseline")
    plt.title(f"Different Settings | stride={STRIDE} | conf_weight={USE_CONF_WEIGHT}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(OUT_DIR, f"line_diff_layers_vs_baseline_S{STRIDE}_CONF_{USE_CONF_WEIGHT}_soft_band.png")
    plt.savefig(fig2_path, dpi=200)
    print(f"[Saved] {fig2_path}")

    print("\n全部完成。输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()
