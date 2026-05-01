import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 配置区（只改这里）
# ============================================================
CSV_PATH = r"outputs/token_attention/scene1_DTU/add_epipolar_band/soft_band/eight_layers_pose_eval_summary.csv"   # or eight_layers_pose_eval_summary.csv
SAVE_PATH = r"outputs/token_attention/scene1_DTU/add_epipolar_band/soft_band/eight_layers_pose_AUC_soft_band.png"      # 设为 None 则不保存
FIGSIZE = (16, 8)
DPI = 200

COL_EXP = "exp_name"
AUC_COLS = ["auc_5", "auc_3", "auc_1"]
BASELINE_NAME = "baseline"

# 从 exp_name 提取：
# layer_0_to_5_soft_band_alpha_2 -> region=layer_0_to_5, alpha=2
REGION_REGEX = r"(layer_\d+_to_\d+)"
ALPHA_REGEX = r"alpha_(\d+)"
# ============================================================


def extract_region(exp_name: str) -> str:
    m = re.search(REGION_REGEX, exp_name)
    return m.group(1) if m else exp_name


def extract_alpha(exp_name: str):
    m = re.search(ALPHA_REGEX, exp_name)
    return int(m.group(1)) if m else None


def region_sort_key(region: str):
    m = re.match(r"layer_(\d+)_to_(\d+)", region)
    if not m:
        return (10**9, 10**9, region)
    return (int(m.group(1)), int(m.group(2)), region)


def main():
    df = pd.read_csv(CSV_PATH)

    # ---- baseline 行 ----
    base_df = df[df[COL_EXP] == BASELINE_NAME]
    if len(base_df) != 1:
        raise ValueError(f"需要且只能有一行 exp_name={BASELINE_NAME}，但现在是 {len(base_df)} 行。")
    baseline = base_df.iloc[0]

    # ---- 非 baseline 实验 ----
    df_exp = df[df[COL_EXP] != BASELINE_NAME].copy()
    if len(df_exp) == 0:
        raise ValueError("CSV里没有 baseline 之外的实验行，无法作图。")

    df_exp["region"] = df_exp[COL_EXP].astype(str).apply(extract_region)
    df_exp["alpha"] = df_exp[COL_EXP].astype(str).apply(extract_alpha)

    # 如果某些实验没写 alpha_?，这里直接丢掉（避免混乱）
    df_exp = df_exp[df_exp["alpha"].notna()].copy()
    df_exp["alpha"] = df_exp["alpha"].astype(int)

    # region 排序
    regions = sorted(df_exp["region"].unique().tolist(), key=region_sort_key)
    # alpha 排序（你目前是 2 和 4，但代码支持更多）
    alphas = sorted(df_exp["alpha"].unique().tolist())

    # 检查：每个 region 是否都包含所有 alpha
    # 不强制，但提示一下
    for r in regions:
        a_set = set(df_exp[df_exp["region"] == r]["alpha"].unique().tolist())
        miss = [a for a in alphas if a not in a_set]
        if miss:
            print(f"[Warn] region={r} 缺少 alpha={miss}，该 alpha 组不会画。")

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # 每个 region 是一个“大组”，大组内有 len(alphas) 个“alpha 小组”，每个 alpha 小组有 5 根柱
    n_metrics = len(AUC_COLS)
    n_alpha = len(alphas)

    # 大组宽度（0.8比较常用）
    region_width = 0.82
    # 每个 alpha 小组在 region 内的宽度（留一点 gap）
    alpha_group_gap = 0.06
    alpha_group_width = (region_width - alpha_group_gap*(n_alpha - 1)) / n_alpha
    # alpha 小组内每根柱宽度
    bar_w = alpha_group_width / n_metrics

    # AUC 5 种颜色固定（每个 alpha 小组内部复用同色系）
    metric_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

    x_centers = np.arange(len(regions))

    # 对每个 region 画多个 alpha 小组
    for ridx, region in enumerate(regions):
        # region 左边界（用于计算偏移）
        region_left = x_centers[ridx] - region_width / 2

        for aidx, alpha in enumerate(alphas):
            df_ra = df_exp[(df_exp["region"] == region) & (df_exp["alpha"] == alpha)]
            if len(df_ra) == 0:
                continue

            # 如果同一个 region+alpha 有多条（比如不同 layer 段命名重复），这里取第一条；你也可改 max/mean
            row = df_ra.iloc[0]
            vals = np.array([float(row[c]) for c in AUC_COLS], dtype=float)

            # alpha 小组起始位置
            alpha_left = region_left + aidx * (alpha_group_width + alpha_group_gap)

            for midx, col in enumerate(AUC_COLS):
                x_bar = alpha_left + (midx + 0.5) * bar_w
                ax.bar(
                    x_bar,
                    vals[midx],
                    width=bar_w * 0.92,
                    color=metric_colors[midx],
                    alpha=0.95,
                )

            # 在 alpha 小组的顶部标注 alpha 值（可选但很有用）
            ax.text(
                alpha_left + alpha_group_width / 2,
                1.01,
                f"α={alpha}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
                transform=ax.get_xaxis_transform(),  # y 用轴坐标，固定在顶部
            )

    # ---- baseline 5 条虚线（与 AUC 颜色对应）----
    for midx, col in enumerate(AUC_COLS):
        y = float(baseline[col])
        ax.axhline(y=y, linestyle="--", linewidth=1.8, color=metric_colors[midx], alpha=0.9)

    # ---- x 轴与样式 ----
    ax.set_xticks(x_centers)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Epipolar band applied global layers (region)")
    ax.set_title("DTU Pose Estimation: AUC vs Band Layer Region (grouped by alpha)\nBars: per-alpha AUCs | Dashed lines: baseline")

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # 做一个 legend：只放 AUC 五种（颜色含义）
    handles = []
    labels = []
    for midx, col in enumerate(AUC_COLS):
        h = plt.Rectangle((0, 0), 1, 1, color=metric_colors[midx])
        handles.append(h)
        labels.append(col.upper())
    ax.legend(handles, labels, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False)

    plt.tight_layout()

    if SAVE_PATH:
        plt.savefig(SAVE_PATH, bbox_inches="tight")
        print("[Saved]", SAVE_PATH)

    plt.show()


if __name__ == "__main__":
    main()