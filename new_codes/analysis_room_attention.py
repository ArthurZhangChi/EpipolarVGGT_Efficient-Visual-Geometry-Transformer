import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# =======================
# Config
# =======================
ATTN_PATH = "outputs/room_baseline_attn_layer0.pt"
PRED_PATH = "outputs/room_baseline_predictions.pt"

OUT_DIR = "outputs/attn_analysis_layer0_patch"
os.makedirs(OUT_DIR, exist_ok=True)

# per-frame token layout: 1 camera + 4 register = 5 special
N_SPECIAL_PER_FRAME = 5

# report / curve configs
K_REPORT = 3
TARGET_COVERAGE = 0.90               # for K* on cross-frame
PERCENTILE_CLIP = (5, 95)            # for heatmap color range

# if there are extra prefix tokens before per-frame blocks (unlikely in your case)
PREFIX = 0


# =======================
# Helpers: IO
# =======================
def load_S_from_predictions(pred_path: str) -> int:
    pred = torch.load(pred_path, map_location="cpu")
    if "images" not in pred:
        raise KeyError("predictions.pt missing key 'images' (need it to infer S).")
    # images: [B,S,3,H,W]
    return int(pred["images"].shape[1])


# =======================
# Helpers: math
# =======================
def row_normalize_np(M: np.ndarray) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)


def exclude_self_and_renorm(F: np.ndarray) -> np.ndarray:
    """
    Remove diagonal (self-frame), then renormalize each row.
    Output is a cross-frame distribution: diag=0 and row-sum=1.
    """
    F2 = F.copy()
    np.fill_diagonal(F2, 0.0)
    return row_normalize_np(F2)


def topk_from_row(M_row: np.ndarray, k: int, exclude_self: bool = True):
    """
    For each row i, pick indices of top-k columns by value.
    """
    S = M_row.shape[0]
    topk_list = []
    for i in range(S):
        scores = M_row[i].copy()
        if exclude_self:
            scores[i] = -np.inf
        idx = np.argsort(scores)[::-1][:k]
        topk_list.append(idx)
    return topk_list


def mass_at_k_curve(M_row: np.ndarray, exclude_self: bool = True, max_k: int | None = None):
    """
    M_row: [S,S] row-normalized distribution.
    mean mass@K: average over i of sum of topK entries in row i.
    """
    S = M_row.shape[0]
    if max_k is None:
        max_k = S
    max_k = min(max_k, S)

    Ks = np.arange(1, max_k + 1)
    mean_mass = []
    for k in Ks:
        topk = topk_from_row(M_row, k, exclude_self=exclude_self)
        masses = []
        for i in range(S):
            masses.append(float(M_row[i, topk[i]].sum()))
        mean_mass.append(np.mean(masses))
    return Ks, np.array(mean_mass)


def effective_num_frames_entropy(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Effective number of frames via entropy:
      N_eff = exp(H(p)), H(p) = -sum p log p
    P: [S,S] cross-frame distribution (diag=0, row-sum=1).
    Return: [S] N_eff per query frame.
    """
    P_ = np.clip(P, eps, 1.0)
    H = -(P_ * np.log(P_)).sum(axis=1)
    return np.exp(H)


def effective_num_frames_hhi(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Effective number of frames via HHI (Renyi-2):
      N_eff2 = 1 / sum(p^2)
    """
    denom = (P * P).sum(axis=1) + eps
    return 1.0 / denom


def topk_mass_and_gain(P: np.ndarray, K: int):
    """
    Top-K mass and gain vs uniform baseline.

    P: [S,S] cross-frame distribution (diag=0, row-sum=1).
    Uniform baseline on cross frames is 1/(S-1), so Top-K uniform mass = K/(S-1).

    Return:
      top_mass: [S] topK mass per query frame
      gain:     [S] (topK mass - K/(S-1)) per query frame
    """
    S = P.shape[0]
    if not (1 <= K <= S - 1):
        raise ValueError(f"K must be in [1, S-1], got K={K}, S={S}")

    # sort each row descending and take topK
    top_mass = np.sort(P, axis=1)[:, ::-1][:, :K].sum(axis=1)
    uniform_mass = K / (S - 1)
    gain = top_mass - uniform_mass
    return top_mass, gain


# =======================
# Helpers: plotting
# =======================
def plot_heatmap(M: np.ndarray, title: str, save_path: str,
                 cbar_label="Value", percentile_clip=PERCENTILE_CLIP):
    S = M.shape[0]
    lo, hi = percentile_clip
    vmin = np.percentile(M, lo)
    vmax = np.percentile(M, hi)

    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label=cbar_label)
    plt.xticks(range(S))
    plt.yticks(range(S))
    plt.xlabel("Key frame index j")
    plt.ylabel("Query frame index i")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def plot_curve(x, y, title: str, save_path: str, xlabel: str, ylabel: str):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def save_topk_report(M_row: np.ndarray, k: int, out_txt: str, exclude_self: bool = True):
    """
    Save per-row topK indices & values, and how often each frame appears in TopK.
    """
    S = M_row.shape[0]
    topk = topk_from_row(M_row, k, exclude_self=exclude_self)
    hit_count = np.zeros(S, dtype=int)

    lines = []
    lines.append(f"Top-{k} key frames per query frame (exclude_self={exclude_self})\n")
    for i in range(S):
        idx = topk[i]
        vals = M_row[i, idx]
        for j in idx:
            hit_count[j] += 1
        pairs = ", ".join([f"{int(j)}({vals[t]:.4f})" for t, j in enumerate(idx)])
        lines.append(f"Query frame {i}: {pairs}")

    lines.append("\nAnchor-like frequency (how many times a frame appears in Top-K):")
    for j in range(S):
        lines.append(f"Frame {j}: {hit_count[j]} / {S}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# =======================
# Core: Patch-only frame MASS
# =======================
def build_frame_mass_patch_only(attn_bhqq: torch.Tensor, S: int,
                                prefix: int = 0,
                                n_special_per_frame: int = 5):
    """
    Build frame-level MASS matrix using patch tokens only.

    attn_bhqq: [1,H,Q,Q]
    Assume after removing 'prefix', tokens are grouped by frame contiguously.
    Each frame has P_total tokens:
      first n_special_per_frame are special tokens (camera+register)
      remaining are patch tokens

    Return:
      F_mass_patch: [S,S]
      P_total, P_patch
    """
    assert attn_bhqq.ndim == 4
    B, H, Q, K = attn_bhqq.shape
    assert B == 1 and Q == K

    # head mean -> [Q,Q]
    A = attn_bhqq[0].mean(dim=0).float()

    # remove possible prefix
    Af = A[prefix:, prefix:]
    Qf = Af.shape[0]
    if Qf % S != 0:
        raise ValueError(f"(Q-prefix) must be divisible by S. Q={Q}, prefix={prefix}, S={S}, Qf={Qf}")

    P_total = Qf // S
    if n_special_per_frame >= P_total:
        raise ValueError(f"n_special_per_frame={n_special_per_frame} >= P_total={P_total}")

    # reshape to [S, P_total, S, P_total]
    M = Af.reshape(S, P_total, S, P_total)

    # patch slice
    q_slice = slice(n_special_per_frame, P_total)
    k_slice = slice(n_special_per_frame, P_total)
    M_patch = M[:, q_slice, :, k_slice]   # [S, P_patch, S, P_patch]
    P_patch = P_total - n_special_per_frame

    # MASS: sum over key tokens, mean over query tokens -> [S,S]
    F_mass_patch = M_patch.sum(dim=3).mean(dim=1).cpu().numpy()

    return F_mass_patch, P_total, P_patch


# =======================
# Main
# =======================
def main():
    attn = torch.load(ATTN_PATH, map_location="cpu")  # [1,16,8328,8328]
    S = load_S_from_predictions(PRED_PATH)
    print("[Load] attn shape:", tuple(attn.shape), "| S =", S)

    F_mass_patch, P_total, P_patch = build_frame_mass_patch_only(
        attn, S, prefix=PREFIX, n_special_per_frame=N_SPECIAL_PER_FRAME
    )
    print(f"[Info] PREFIX={PREFIX}, P_total={P_total}, special={N_SPECIAL_PER_FRAME}, P_patch={P_patch}")

    # Save raw matrices
    np.save(os.path.join(OUT_DIR, "frame_attn_mass_patch_raw.npy"), F_mass_patch)

    # Row-normalized (includes self)
    F_row = row_normalize_np(F_mass_patch)
    np.save(os.path.join(OUT_DIR, "frame_attn_mass_patch_row.npy"), F_row)

    # Cross-frame distribution (exclude self + renorm)
    F_cross = exclude_self_and_renorm(F_mass_patch)
    np.save(os.path.join(OUT_DIR, "frame_attn_mass_patch_cross.npy"), F_cross)

    # =======================
    # Heatmaps
    # =======================
    plot_heatmap(
        F_mass_patch,
        title="[PATCH] Frame attention MASS - raw",
        save_path=os.path.join(OUT_DIR, "heatmap_patch_mass_raw.png"),
        cbar_label="mass"
    )
    plot_heatmap(
        F_row,
        title="[PATCH] Frame attention MASS - row-normalized",
        save_path=os.path.join(OUT_DIR, "heatmap_patch_mass_row.png"),
        cbar_label="prob"
    )
    plot_heatmap(
        F_cross,
        title="[PATCH] Frame attention MASS - exclude self + renorm (CROSS-FRAME)",
        save_path=os.path.join(OUT_DIR, "heatmap_patch_mass_cross.png"),
        cbar_label="cross prob"
    )

    # =======================
    # Old stats (self vs cross on row-normalized)
    # =======================
    diag = np.diag(F_row)
    cross = 1.0 - diag
    print("\n[Stats] On row-normalized PATCH-MASS (includes self):")
    print("  Mean self-frame mass  :", float(diag.mean()))
    print("  Mean cross-frame mass :", float(cross.mean()))
    print("  Per-frame self mass   :", np.round(diag, 4).tolist())

    # =======================
    # Old metric: mass@K & K*
    # (computed on cross distribution)
    # =======================
    max_k = S - 1
    Ks, mean_mass = mass_at_k_curve(F_cross, exclude_self=False, max_k=max_k)
    plot_curve(
        Ks, mean_mass,
        title="[PATCH] Oracle mass@K on CROSS-FRAME distribution",
        save_path=os.path.join(OUT_DIR, "mass_at_k_cross_patch.png"),
        xlabel="K (number of selected other frames)",
        ylabel="Mean covered attention mass"
    )

    k_star = None
    for k, m in zip(Ks, mean_mass):
        if m >= TARGET_COVERAGE:
            k_star = int(k)
            break
    print(f"\n[Old Metric] Smallest K achieving >= {TARGET_COVERAGE*100:.0f}% cross-mass coverage: K* = {k_star}")

    # =======================
    # New metric 1: Effective number of frames
    # =======================
    Ne_ent = effective_num_frames_entropy(F_cross)
    Ne_hhi = effective_num_frames_hhi(F_cross)

    print("\n[New Metric] Effective number of frames (cross-frame distribution):")
    print("  N_eff(entropy) per query:", np.round(Ne_ent, 3).tolist())
    print("  N_eff(entropy) mean/std :", float(Ne_ent.mean()), float(Ne_ent.std()))
    print("  N_eff(HHI) per query    :", np.round(Ne_hhi, 3).tolist())
    print("  N_eff(HHI) mean/std     :", float(Ne_hhi.mean()), float(Ne_hhi.std()))

    # plot N_eff per query
    plt.figure()
    plt.plot(np.arange(S), Ne_ent, marker="o")
    plt.xlabel("Query frame i")
    plt.ylabel("N_eff = exp(H(p_i))")
    plt.title("[PATCH] Effective number of frames (entropy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Neff_entropy_per_query.png"), dpi=200)
    plt.show()

    # =======================
    # New metric 2: Top-K gain vs uniform
    # =======================
    # per-query and mean gain at selected K
    for K in [1, 2, 3, 4, 5, 6, 7]:
        if 1 <= K <= S - 1:
            top_mass, gain = topk_mass_and_gain(F_cross, K)
            print(f"\n[New Metric] Top-{K} mass & gain vs uniform (uniform={K}/{S-1}={K/(S-1):.4f}):")
            print("  topK mass per query:", np.round(top_mass, 4).tolist())
            print("  gain per query     :", np.round(gain, 4).tolist())
            print("  mean topK mass     :", float(top_mass.mean()))
            print("  mean gain          :", float(gain.mean()))

    # plot mean gain vs K
    Ks2 = np.arange(1, S)  # 1..S-1
    mean_gain = []
    mean_topmass = []
    for K in Ks2:
        tm, g = topk_mass_and_gain(F_cross, int(K))
        mean_topmass.append(tm.mean())
        mean_gain.append(g.mean())

    plt.figure()
    plt.plot(Ks2, mean_gain, marker="o")
    plt.xlabel("K (number of selected other frames)")
    plt.ylabel("Mean Top-K gain over uniform")
    plt.title("[PATCH] Mean Top-K gain vs K")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mean_topk_gain_vs_K.png"), dpi=200)
    plt.show()

    # =======================
    # Top-K report (based on cross distribution)
    # =======================
    report_path = os.path.join(OUT_DIR, f"top{K_REPORT}_report_cross_patch.txt")
    save_topk_report(F_cross, k=K_REPORT, out_txt=report_path, exclude_self=False)

    print("\nSaved all outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()




"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

ATTN_PATH = "outputs/room_baseline_attn_layer20.pt"
PRED_PATH = "outputs/room_baseline_predictions.pt"  # 用来自动读取 S（view 数）

OUT_DIR = "outputs/attn_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# 你想用哪个矩阵来定义“oracle topK”：
# - "mass": 用 F_mass（更符合“覆盖概率质量”的直觉）
# - "mean": 用 F_mean（更符合你原 heatmap 的“平均强度”）
TOPK_ORACLE_MODE = "mass"

# 打印/保存 topK 报告的 K
K_REPORT = 3

# 选 K 的覆盖率阈值
TARGET_COVERAGE = 0.90


# ============ 工具函数 ============

def load_S_from_predictions(pred_path: str) -> int:
    pred = torch.load(pred_path, map_location="cpu")
    if "images" not in pred:
        raise KeyError("predictions.pt 中找不到 key: 'images'，无法推断 S。")
    return int(pred["images"].shape[1])


def row_normalize(M: np.ndarray) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)


def plot_heatmap(M: np.ndarray, title: str, save_path: str, cbar_label="Value"):
    S = M.shape[0]
    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="viridis")
    plt.colorbar(label=cbar_label)
    plt.xticks(range(S))
    plt.yticks(range(S))
    plt.xlabel("Key frame index j")
    plt.ylabel("Query frame index i")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def topk_from_row(M_row: np.ndarray, k: int, exclude_self: bool = True):
    S = M_row.shape[0]
    topk_list = []
    for i in range(S):
        scores = M_row[i].copy()
        if exclude_self:
            scores[i] = -np.inf
        idx = np.argsort(scores)[::-1][:k]
        topk_list.append(idx)
    return topk_list


def mass_at_k_curve(M_row: np.ndarray, exclude_self: bool = True):
    S = M_row.shape[0]
    Ks = np.arange(1, S + 1)
    mean_mass = []

    for k in Ks:
        topk = topk_from_row(M_row, k, exclude_self=exclude_self)
        masses = []
        for i in range(S):
            masses.append(float(M_row[i, topk[i]].sum()))
        mean_mass.append(np.mean(masses))

    return Ks, np.array(mean_mass)


def plot_mass_at_k(Ks: np.ndarray, mean_mass: np.ndarray, title: str, save_path: str):
    plt.figure()
    plt.plot(Ks, mean_mass, marker="o")
    plt.xlabel("K (number of selected key frames)")
    plt.ylabel("Mean attention mass covered")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def save_topk_report(M_row: np.ndarray, k: int, out_txt: str):
    S = M_row.shape[0]
    topk = topk_from_row(M_row, k, exclude_self=True)
    hit_count = np.zeros(S, dtype=int)

    lines = []
    lines.append(f"Top-{k} key frames per query frame (exclude self)\n")
    for i in range(S):
        idx = topk[i]
        vals = M_row[i, idx]
        for j in idx:
            hit_count[j] += 1
        pairs = ", ".join([f"{int(j)}({vals[t]:.4f})" for t, j in enumerate(idx)])
        lines.append(f"Query frame {i}: {pairs}")

    lines.append("\nAnchor-like frequency (how many times a frame appears in Top-K):")
    for j in range(S):
        lines.append(f"Frame {j}: {hit_count[j]} / {S}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


# ============ 核心：同时构造 F_mean 与 F_mass ============

def build_frame_matrices(attn_bhqq: torch.Tensor, S: int):
    assert attn_bhqq.ndim == 4
    B, H, Q, K = attn_bhqq.shape
    assert B == 1 and Q == K
    assert Q % S == 0, f"Q={Q} not divisible by S={S}"

    # head mean -> token attention A: [Q,Q]
    A = attn_bhqq[0].mean(dim=0)  # [Q,Q]

    P_total = Q // S

    F_mean = torch.zeros(S, S)
    F_mass = torch.zeros(S, S)

    for i in range(S):
        qi = slice(i * P_total, (i + 1) * P_total)  # query tokens for frame i
        for j in range(S):
            kj = slice(j * P_total, (j + 1) * P_total)  # key tokens for frame j
            block = A[qi, kj]  # [P_total, P_total]

            # 1) mean-strength（你原来的 heatmap）
            F_mean[i, j] = block.mean()

            # 2) mass（概率质量）：对 key tokens 求和得到每个 query token落到该帧的质量，再对 query tokens 平均
            #    block.sum(dim=1): [P_total]  每个 query token -> key帧j 的总注意力
            #    mean(): 该帧所有 query token 的平均质量
            F_mass[i, j] = block.sum(dim=1).mean()

    return F_mean.numpy(), F_mass.numpy()


# ============ 主流程 ============

def main():
    attn = torch.load(ATTN_PATH, map_location="cpu")  # [1,H,SP,SP]
    S = load_S_from_predictions(PRED_PATH)
    print("Loaded attn:", tuple(attn.shape), "S =", S)

    # 构造两种 frame 矩阵
    F_mean, F_mass = build_frame_matrices(attn, S)

    # 保存矩阵
    np.save(os.path.join(OUT_DIR, "frame_attn_mean.npy"), F_mean)
    np.save(os.path.join(OUT_DIR, "frame_attn_mass.npy"), F_mass)

    # ===== 1) heatmap（保留你原来的两张图：基于 mean）=====
    F_mean_row = row_normalize(F_mean)

    plot_heatmap(
        F_mean,
        title="Frame-level attention (block-mean over heads) - raw",
        save_path=os.path.join(OUT_DIR, "heatmap_frame_attn_mean_raw.png"),
        cbar_label="Mean attention (block mean)"
    )

    plot_heatmap(
        F_mean_row,
        title="Frame-level attention (block-mean, row-normalized)",
        save_path=os.path.join(OUT_DIR, "heatmap_frame_attn_mean_row_norm.png"),
        cbar_label="Mean attention (row-normalized)"
    )

    # ===== 2) mass@K（关键：基于 F_mass）=====
    # F_mass 每一行本身应当近似和为 1（概率质量分布），但仍建议 row-normalize 一下增强鲁棒性
    F_mass_row = row_normalize(F_mass)

    Ks, mean_mass = mass_at_k_curve(F_mass_row, exclude_self=True)
    plot_mass_at_k(
        Ks, mean_mass,
        title="Oracle mass@K (based on frame attention MASS)",
        save_path=os.path.join(OUT_DIR, "oracle_mass_at_k_mass.png"),
    )

    # ===== 3) Top-K 报告：默认按 MASS 定义相关性（更贴近“覆盖质量”）=====
    if TOPK_ORACLE_MODE == "mass":
        report_mat = F_mass_row
        report_name = "mass"
    else:
        report_mat = F_mean_row
        report_name = "mean"

    save_topk_report(
        report_mat, k=K_REPORT,
        out_txt=os.path.join(OUT_DIR, f"top{K_REPORT}_report_{report_name}.txt")
    )

    # ===== 4) 建议的 K*（达到 TARGET_COVERAGE 的最小 K）=====
    k_star = None
    for k, m in zip(Ks, mean_mass):
        if m >= TARGET_COVERAGE:
            k_star = int(k)
            break

    print(f"\n[Suggestion] Smallest K achieving >= {TARGET_COVERAGE*100:.0f}% mean mass coverage (MASS): K = {k_star}")

    # ===== 5) 打印一下每行 sum，帮助你 sanity check（mass 是否像概率）=====
    row_sums = F_mass.sum(axis=1)
    print("\nSanity check: row sums of F_mass (should be ~1, before row-normalize):")
    print(row_sums)

    print("\nSaved all analysis results to:", OUT_DIR)


if __name__ == "__main__":
    main()
"""