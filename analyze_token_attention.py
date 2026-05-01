import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# 一、【用户配置区域】——只改这里
# ============================================================

# attention 文件所在目录
ATTN_DIR = "outputs/token_attention/scene1_DTU"

# 要分析并画曲线的层
LAYERS = [0, 4, 8, 12, 16, 20, 23]

# 视图数
S = 7

# 每个 view 内 special token 数（camera + register）
PATCH_START_IDX = 5

# 保存 attention 时的参数
CACHE_MODE = "head_mean"
SAMPLE_Q = 0

# 输出目录
OUT_DIR = "outputs/token_attention/scene1_DTU/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 二、工具函数
# ============================================================

def build_type_indices(N: int, S: int, patch_start_idx: int):
    """
    假设 token 序列按 view 拼接：
      view: [special(0..patch_start_idx-1), patch(patch_start_idx..T-1)]
    """
    assert N % S == 0, f"N={N} 不能被 S={S} 整除"
    T = N // S
    assert patch_start_idx < T

    special_rows, patch_rows = [], []

    for v in range(S):
        base = v * T
        special_rows.extend(range(base, base + patch_start_idx))
        patch_rows.extend(range(base + patch_start_idx, base + T))

    special_rows = np.array(special_rows, dtype=np.int64)
    patch_rows = np.array(patch_rows, dtype=np.int64)

    special_mask = np.zeros(N, dtype=bool)
    patch_mask = np.zeros(N, dtype=bool)
    special_mask[special_rows] = True
    patch_mask[patch_rows] = True

    return T, special_rows, patch_rows, special_mask, patch_mask


def row_sum_stats(attn: np.ndarray):
    rs = attn.sum(axis=1)
    return float(rs.min()), float(rs.max()), float(rs.mean())


# ============================================================
# 三、主分析：逐层统计 + 逐层画 query 曲线
# ============================================================

records = []

for layer in LAYERS:
    print(f"\n===== 分析 Layer {layer:02d} =====")

    attn_path = os.path.join(
        ATTN_DIR,
        f"attn_layer{layer:02d}_{CACHE_MODE}_q{SAMPLE_Q}.pt"
    )
    if not os.path.exists(attn_path):
        raise FileNotFoundError(f"找不到 attention 文件：{attn_path}")

    # -------- 读取 attention --------
    attn_t = torch.load(attn_path, map_location="cpu")
    if not isinstance(attn_t, torch.Tensor):
        attn_t = attn_t[0]

    if attn_t.dim() == 3:
        attn = attn_t[0].float().numpy()
    elif attn_t.dim() == 2:
        attn = attn_t.float().numpy()
    else:
        raise ValueError(f"不支持的 attention 维度：{attn_t.shape}")

    N = attn.shape[0]
    assert attn.shape[1] == N

    # -------- 构造 mask --------
    T, special_rows, patch_rows, special_mask, patch_mask = build_type_indices(
        N=N, S=S, patch_start_idx=PATCH_START_IDX
    )

    # -------- sanity check --------
    rmin, rmax, rmean = row_sum_stats(attn)
    print(
        f"N={N}, T={T}, "
        f"specialQ={len(special_rows)}, patchQ={len(patch_rows)}, "
        f"row-sum mean={rmean:.6f}"
    )

    # ========================================================
    # 1) per-query attention mass
    # ========================================================
    m_to_special = attn[:, special_mask].sum(axis=1)
    m_to_patch = attn[:, patch_mask].sum(axis=1)

    max_err = float(np.max(np.abs(m_to_special + m_to_patch - 1.0)))
    if max_err > 1e-5:
        print(f"[警告] Row mass sum 最大误差={max_err:.2e}")

    # ========================================================
    # 2) 按 query 类型求平均（SS/SP/PS/PP）
    # ========================================================
    m_sp_q_to_special = float(m_to_special[special_rows].mean())
    m_sp_q_to_patch = float(m_to_patch[special_rows].mean())

    m_pa_q_to_special = float(m_to_special[patch_rows].mean())
    m_pa_q_to_patch = float(m_to_patch[patch_rows].mean())

    SS = m_sp_q_to_special
    SP = m_sp_q_to_patch
    PS = m_pa_q_to_special
    PP = m_pa_q_to_patch

    records.append({
        "layer": layer,
        "N": N,
        "S": S,
        "tokens_per_view": T,
        "patch_start_idx": PATCH_START_IDX,
        "num_special_queries": len(special_rows),
        "num_patch_queries": len(patch_rows),

        "specialQ_to_specialK_mean": m_sp_q_to_special,
        "specialQ_to_patchK_mean": m_sp_q_to_patch,
        "patchQ_to_specialK_mean": m_pa_q_to_special,
        "patchQ_to_patchK_mean": m_pa_q_to_patch,

        "SS": SS,
        "SP": SP,
        "PS": PS,
        "PP": PP,
    })

    # ========================================================
    # 3) 画当前 layer 的 query-level 曲线
    # ========================================================
    num_special = special_mask.sum()
    num_patch = patch_mask.sum()

    mean_to_special = m_to_special / num_special
    mean_to_patch = m_to_patch / num_patch

    x = np.arange(N)

    plt.figure(figsize=(12, 4.5))
    plt.plot(x, mean_to_special, linewidth=1.0, label=r"mean attention per SPECIAL key")
    plt.plot(x, mean_to_patch, linewidth=1.0, label=r"mean attention per PATCH key")

    plt.xlabel("Query index i")
    plt.ylabel("Attention mass")
    plt.title(f"Per-query Attention Mass (Layer {layer})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(
        OUT_DIR,
        f"query_level_mass_layer{layer:02d}.png"
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[保存] 曲线图：{fig_path}")


# ============================================================
# 四、保存 CSV（所有层）
# ============================================================

csv_path = os.path.join(OUT_DIR, "attention_mass_decomposition.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    for r in records:
        writer.writerow(r)

print(f"\n[保存] CSV：{csv_path}")
print("\n全部完成。")
