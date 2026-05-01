import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 配置区（只改这里）
# ============================================================
CSV_PATH  = r"outputs/token_attention/scene3_7Scenes/six_layers_recon_eval_summary_ICP_False.csv"
SAVE_PATH = r"outputs/token_attention/scene3_7Scenes/six_layers_recon_metrics_ICP_False.png"  # 设为 None 则不保存
FIGSIZE = (16, 8)
DPI = 200

COL_EXP = "exp_name"
BASELINE_NAME = "baseline"

# 你要画的4个柱（必须是CSV表头中的列名）
METRIC_COLS = ["accuracy_mean", "completeness_mean", "overall_mean"]

# 从 exp_name 提取 region：
# layer_0_to_5_soft_band_alpha_2 -> region=layer_0_to_5
REGION_REGEX = r"(layer_\d+_to_\d+)"

# 如果同一 region 有多条（例如不同 alpha），如何聚合
# 可选: "mean" / "max" / "first"
AGG_MODE = "mean"

# y 轴范围（这些指标一般是“越小越好”的距离/误差，按你实际数值改）
Y_LIM = None  # 例如 (0.0, 0.1)，不想固定就 None

# ============================================================


def extract_region(exp_name: str) -> str:
    m = re.search(REGION_REGEX, str(exp_name))
    return m.group(1) if m else str(exp_name)


def region_sort_key(region: str):
    m = re.match(r"layer_(\d+)_to_(\d+)", str(region))
    if not m:
        return (10**9, 10**9, str(region))
    return (int(m.group(1)), int(m.group(2)), str(region))


def aggregate_rows(df_region: pd.DataFrame, cols):
    """把同一 region 的多行聚合成一行"""
    if len(df_region) == 1 or AGG_MODE == "first":
        row = df_region.iloc[0]
        return {c: float(row[c]) for c in cols}

    if AGG_MODE == "mean":
        return {c: float(df_region[c].astype(float).mean()) for c in cols}
    if AGG_MODE == "max":
        return {c: float(df_region[c].astype(float).max()) for c in cols}

    raise ValueError(f"Unknown AGG_MODE={AGG_MODE}")


def main():
    df = pd.read_csv(CSV_PATH)

    # 检查列
    need_cols = [COL_EXP] + METRIC_COLS
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"CSV缺少列：{miss}\n你当前CSV列为：{df.columns.tolist()}")

    # ---- baseline 行 ----
    base_df = df[df[COL_EXP] == BASELINE_NAME]
    if len(base_df) != 1:
        raise ValueError(f"需要且只能有一行 exp_name={BASELINE_NAME}，但现在是 {len(base_df)} 行。")
    baseline = base_df.iloc[0]
    baseline_vals = {c: float(baseline[c]) for c in METRIC_COLS}

    # ---- 非 baseline 实验 ----
    df_exp = df[df[COL_EXP] != BASELINE_NAME].copy()
    if len(df_exp) == 0:
        raise ValueError("CSV里没有 baseline 之外的实验行，无法作图。")

    df_exp["region"] = df_exp[COL_EXP].astype(str).apply(extract_region)

    # region 排序
    regions = sorted(df_exp["region"].unique().tolist(), key=region_sort_key)

    # 对每个 region 聚合（解决同 region 多条，比如 alpha_2/alpha_4）
    rows = []
    for r in regions:
        dfr = df_exp[df_exp["region"] == r]
        agg = aggregate_rows(dfr, METRIC_COLS)
        rows.append({"region": r, **agg})
    df_plot = pd.DataFrame(rows)

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    n_metrics = len(METRIC_COLS)
    group_width = 0.82
    bar_w = group_width / n_metrics

    x_centers = np.arange(len(regions))

    # 柱形：每个 region 4 根
    for midx, col in enumerate(METRIC_COLS):
        vals = df_plot[col].astype(float).to_numpy()
        # 每根柱的中心偏移
        x_bar = x_centers - group_width/2 + (midx + 0.5) * bar_w
        ax.bar(x_bar, vals, width=bar_w * 0.92, alpha=0.95)

    # baseline 虚线：4条（颜色跟柱颜色一致：用当前颜色循环拿到的颜色）
    # trick：先取 legend handles 的颜色（或用 ax.get_children() 也行）
    # 这里简单做：画虚线时复用 matplotlib 默认颜色序列（与上面bar一致）
    # 先获取当前颜色循环
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for midx, col in enumerate(METRIC_COLS):
        y = baseline_vals[col]
        color = default_colors[midx % len(default_colors)] if default_colors else None
        ax.axhline(y=y, linestyle="--", linewidth=1.8, color=color, alpha=0.9)

    # x 轴
    ax.set_xticks(x_centers)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_xlabel("Epipolar band applied global layers (region)")
    ax.set_ylabel("Metric value")
    ax.set_title(
        "7Scenes 3D Reconstruction: metrics vs Band Layer Region\n"
        f"Bars: {', '.join(METRIC_COLS)} | Dashed lines: baseline"
    )

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if Y_LIM is not None:
        ax.set_ylim(Y_LIM)

    # legend：只放 4 个柱子的含义
    handles = []
    labels = []
    for midx, col in enumerate(METRIC_COLS):
        h = plt.Rectangle((0, 0), 1, 1, color=default_colors[midx % len(default_colors)] if default_colors else None)
        handles.append(h)
        labels.append(col)
    ax.legend(handles, labels, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False)

    plt.tight_layout()

    if SAVE_PATH:
        plt.savefig(SAVE_PATH, bbox_inches="tight")
        print("[Saved]", SAVE_PATH)

    plt.show()


if __name__ == "__main__":
    main()
