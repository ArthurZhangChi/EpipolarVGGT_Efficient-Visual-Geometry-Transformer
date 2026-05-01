import os
import math
import csv
import numpy as np
import torch


# =========================
# 0. 配置
# =========================
BASELINE_PATH = "outputs/room_baseline_predictions.pt"

POSE_DIR = "outputs/eval_sparse/L20"
RANDOM_DIRS = [
    "outputs/eval_sparse/L20_random_seed0",
    "outputs/eval_sparse/L20_random_seed1",
    "outputs/eval_sparse/L20_random_seed2",
    "outputs/eval_sparse/L20_random_seed3",
]

SEEDS = [0, 1, 2, 3]
K_MIN, K_MAX = 1, 7

STRIDE = 4
USE_CONF_WEIGHT = True

OUT_CSV = "outputs/eval_sparse/eval_random_stats_L20.csv"


# =========================
# 1. 基础工具函数
# =========================
def downsample_hw(x, stride):
    if x is None or stride <= 1:
        return x
    if x.ndim == 5:
        return x[:, :, ::stride, ::stride, :]
    if x.ndim == 4:
        return x[:, :, ::stride, ::stride]
    return x


def masked_mean(err, w=None, eps=1e-12):
    if w is None:
        return err.mean().item()
    w = torch.clamp(w, min=0)
    return (err * w).sum().div(w.sum().clamp(min=eps)).item()


def eval_vs_baseline(base, pred):
    d0 = downsample_hw(base["depth"], STRIDE)
    d1 = downsample_hw(pred["depth"], STRIDE)

    p0 = downsample_hw(base["world_points"], STRIDE)
    p1 = downsample_hw(pred["world_points"], STRIDE)

    pe0 = base["pose_enc"]
    pe1 = pred["pose_enc"]

    w_d = downsample_hw(pred.get("depth_conf"), STRIDE) if USE_CONF_WEIGHT else None
    w_p = downsample_hw(pred.get("world_points_conf"), STRIDE) if USE_CONF_WEIGHT else None

    out = {}

    # depth
    diff = (d1[..., 0] - d0[..., 0]).abs()
    out["depth_mae"] = masked_mean(diff, w_d)

    # points
    l2 = torch.norm(p1 - p0, dim=-1)
    out["pts_l2_mean"] = masked_mean(l2, w_p)

    # pose
    pose_l2 = torch.norm(pe1 - pe0, dim=-1)
    out["pose_l2_mean"] = pose_l2.mean().item()

    return out


# =========================
# 2. 主流程
# =========================
def main():
    baseline = torch.load(BASELINE_PATH, map_location="cpu")

    rows = []

    for K in range(K_MIN, K_MAX + 1):

        # ---------- pose ----------
        pose_path = os.path.join(
            POSE_DIR, f"room_sparse_L20_K{K}_predictions.pt"
        )
        pose_pred = torch.load(pose_path, map_location="cpu")
        pose_m = eval_vs_baseline(baseline, pose_pred)

        # ---------- random ----------
        rand_vals = {k: [] for k in pose_m.keys()}

        for seed, rdir in zip(SEEDS, RANDOM_DIRS):
            rand_path = os.path.join(
                rdir, f"room_sparse_L20_random_seed{seed}_K{K}_predictions.pt"
            )
            rand_pred = torch.load(rand_path, map_location="cpu")
            rm = eval_vs_baseline(baseline, rand_pred)
            for k in rand_vals:
                rand_vals[k].append(rm[k])

        # ---------- 聚合 ----------
        row = {"K": K}
        for m in rand_vals:
            arr = np.array(rand_vals[m])
            row[f"pose_{m}"] = pose_m[m]
            row[f"rand_mean_{m}"] = arr.mean()
            row[f"rand_std_{m}"] = arr.std()
            row[f"gap_rand_minus_pose_{m}"] = arr.mean() - pose_m[m]

        rows.append(row)

        print(
            f"[K={K}] "
            f"pose={pose_m['depth_mae']:.4f} | "
            f"rand={row['rand_mean_depth_mae']:.4f}±{row['rand_std_depth_mae']:.4f}"
        )

    # ---------- 保存 CSV ----------
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[Saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
