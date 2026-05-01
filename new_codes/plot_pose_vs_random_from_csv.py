import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "outputs/eval_sparse/eval_random_stats_L20.csv"
OUT_DIR = r"outputs/eval_sparse"
TAG = "L20"

METRICS = [
    "depth_mae",
    "pts_l2_mean",
    "pose_l2_mean",
]

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing columns: {missing}\nAvailable columns: {list(df.columns)}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # 基本检查
    require_cols(df, ["K"])
    Ks = df["K"].to_numpy()

    for m in METRICS:
        pose_col = f"pose_{m}"
        rmean_col = f"rand_mean_{m}"
        rstd_col  = f"rand_std_{m}"
        gap_col   = f"gap_rand_minus_pose_{m}"

        require_cols(df, [pose_col, rmean_col, rstd_col, gap_col])

        pose = df[pose_col].to_numpy()
        rmean = df[rmean_col].to_numpy()
        rstd  = df[rstd_col].to_numpy()
        gap   = df[gap_col].to_numpy()

        # -------------------------
        # 图1：Pose vs Random(mean ± std)
        # -------------------------
        plt.figure(figsize=(7, 5))
        plt.plot(Ks, pose, marker="o", label="Pose-TopK")
        plt.plot(Ks, rmean, marker="o", linestyle="--", label="Random-TopK (mean)")
        plt.fill_between(Ks, rmean - rstd, rmean + rstd, alpha=0.2, label="Random ±1 std")

        plt.xlabel("K (Top-K frames)")
        plt.ylabel(m)
        plt.title(f"{TAG} | Pose vs Random ({m})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out1 = os.path.join(OUT_DIR, f"{TAG}_pose_vs_random_{m}.png")
        plt.savefig(out1, dpi=200)
        plt.close()
        print(f"[Saved] {out1}")

        # -------------------------
        # 图2：Gap 曲线（你要的“差距”）
        # gap = rand_mean - pose
        # -------------------------
        plt.figure(figsize=(7, 5))
        plt.plot(Ks, gap, marker="o", label="Gap = Random(mean) - Pose")
        # 可选：用 random std 作为“波动参考带”
        plt.fill_between(Ks, gap - rstd, gap + rstd, alpha=0.2, label="Gap ± Random std")

        plt.axhline(0.0, linestyle="--")
        plt.xlabel("K (Top-K frames)")
        plt.ylabel(f"gap_{m}")
        plt.title(f"{TAG} | Gap Curve ({m})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out2 = os.path.join(OUT_DIR, f"{TAG}_gap_{m}.png")
        plt.savefig(out2, dpi=200)
        plt.close()
        print(f"[Saved] {out2}")

    print("\n[Done] All plots generated.")

if __name__ == "__main__":
    main()
