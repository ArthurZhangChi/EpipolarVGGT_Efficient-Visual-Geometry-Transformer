# analyze_epipolar_attention_metrics.py
import os
import re
import glob
import math
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 0) 你只需要改这些配置
# ============================================================

# 你的 attention .pt 文件目录（里面应有 attn_layer00_*.pt 之类）
ATTN_DIR = r"outputs/token_attention/scene1_DTU"

# DTU 子数据集目录（含 rect_XXX_max.png 和 pos_XXX）
DATA_DIR = r"datasets/scene1_DTU"

# 输出目录
OUT_DIR = os.path.join(ATTN_DIR, "patch_attention_to_views/patch_500_1200/epipolar_attention_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# 需要分析的层
TARGET_LAYERS = [i for i in range(0, 24)]

# 视图数（必须与 attention 保存时一致）
NUM_VIEWS = 7

# 目标 view（0-based）
TARGET_VIEW = 2

# 你在 target rect 图上的点击点 (y, x)
QUERY_POS = (500, 1200)  # (y, x)

# 每个 view 的 special token 数：1 camera + 4 register = 5
NUM_SPECIAL_PER_VIEW = 5

# epipolar band 半宽：像素距离阈值（在 rect 图像像素坐标系下）
BAND_PX = 200

# top-k
TOPK = 200

# view_id -> pos_id 映射（按你实际拷贝到子集目录里的 pos 文件来写）
# 例：你有 pos_001 pos_003 pos_005 pos_007 pos_009 pos_011 pos_013
VIEW_ID_TO_POS_ID = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13}

# rect 图像模式：优先 rect_XXX_max，其次 rect_XXX
RECT_PATTERN_1 = "rect_{:03d}_max.*"
RECT_PATTERN_2 = "rect_{:03d}.*"

# ============================================================
# 1) 工具：读取 attention 文件
# ============================================================

def find_attn_file(layer: int) -> str:
    cands = sorted(glob.glob(os.path.join(ATTN_DIR, f"attn_layer{layer:02d}_*.pt")))
    if not cands:
        raise FileNotFoundError(f"[错误] 找不到 layer={layer} 的 attention：{ATTN_DIR}/attn_layer{layer:02d}_*.pt")
    return cands[0]

def load_attn_matrix(attn_path: str) -> np.ndarray:
    A = torch.load(attn_path, map_location="cpu")
    if isinstance(A, (list, tuple)):
        A = A[0]
    if A.ndim == 3:
        A = A[0]
    A = A.detach().cpu().numpy()
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"[错误] attention 不是方阵：{A.shape}")
    return A

# ============================================================
# 2) 工具：读取 rect 图与 P=pos_XXX（3x4 投影矩阵）
# ============================================================

def find_rect_image(pos_id: int) -> str:
    patt1 = os.path.join(DATA_DIR, RECT_PATTERN_1.format(pos_id))
    cands1 = sorted(glob.glob(patt1))
    if cands1:
        return cands1[0]
    patt2 = os.path.join(DATA_DIR, RECT_PATTERN_2.format(pos_id))
    cands2 = sorted(glob.glob(patt2))
    if cands2:
        return cands2[0]
    raise FileNotFoundError(f"[错误] 找不到 rect 图像：{patt1} 或 {patt2}")

def find_pos_file(pos_id: int) -> str:
    patt = os.path.join(DATA_DIR, f"pos_{pos_id:03d}*")
    cands = sorted(glob.glob(patt))
    if not cands:
        raise FileNotFoundError(f"[错误] 找不到 pos 文件：{patt}")
    return cands[0]

def load_projection_matrix(pos_path: str) -> np.ndarray:
    """
    兼容：
    - 3 行 4 列
    - 或 12 个数任意换行
    """
    with open(pos_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().strip()
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)]
    if len(nums) != 12:
        raise ValueError(f"[错误] {pos_path} 解析到 {len(nums)} 个数，期望 12 个（3x4）")
    return np.array(nums, dtype=np.float64).reshape(3, 4)

# ============================================================
# 3) 多视几何：由 P1,P2 得 F，使 l2 = F x1
# ============================================================

def camera_center_from_P(P: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(P)
    C = Vt[-1, :]
    if abs(C[-1]) > 1e-12:
        C = C / C[-1]
    return C

def skew(v: np.ndarray) -> np.ndarray:
    v = v.reshape(-1)
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ], dtype=np.float64)

def fundamental_from_projections(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    返回 F_21，使得 l2 = F_21 x1
    """
    C1 = camera_center_from_P(P1)      # 4D
    e2 = P2 @ C1                       # 3D
    if abs(e2[2]) > 1e-12:
        e2 = e2 / e2[2]

    P1_pinv = np.linalg.pinv(P1)       # 4x3
    M = P2 @ P1_pinv                   # 3x3
    F = skew(e2) @ M                   # 3x3

    n = np.linalg.norm(F)
    if n > 1e-12:
        F = F / n
    return F

# ============================================================
# 4) patch 网格与索引（与你 attention token 对齐）
# ============================================================

def infer_T_and_P_from_attn(A: np.ndarray, num_views: int, num_special_per_view: int):
    N = A.shape[0]
    if N % num_views != 0:
        raise ValueError(f"[错误] N={N} 不能被 NUM_VIEWS={num_views} 整除，检查 token 拼接假设")
    T = N // num_views
    P = T - num_special_per_view
    if P <= 0:
        raise ValueError(f"[错误] 推断 P=T-{num_special_per_view} 得到 P={P} 不合理")
    patch_start_idx = num_special_per_view
    return N, T, P, patch_start_idx

def factor_pairs(n):
    pairs = []
    for a in range(1, int(math.sqrt(n)) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
    return pairs

def infer_patch_hw(P: int, H: int, W: int):
    # 找 hp*wp=P，且 hp/wp 接近 H/W
    target_ratio = H / max(W, 1e-6)
    best_hw = None
    best_err = 1e9
    for hp, wp in factor_pairs(P):
        ratio = hp / max(wp, 1e-6)
        err = abs(math.log((ratio / target_ratio) + 1e-12))
        if err < best_err:
            best_err = err
            best_hw = (hp, wp)
    if best_hw is None:
        raise ValueError("[错误] 无法分解 P 得到合理网格")
    return best_hw

def pos_to_patch_index(y, x, H, W, Hp, Wp):
    y = float(np.clip(y, 0, H - 1))
    x = float(np.clip(x, 0, W - 1))
    py = int(np.clip((y / H) * Hp, 0, Hp - 1))
    px = int(np.clip((x / W) * Wp, 0, Wp - 1))
    return py, px, py * Wp + px

def global_index_of_view_patch(view_id, patch_idx, T, patch_start_idx):
    return view_id * T + patch_start_idx + patch_idx

def view_patch_range(view_id, T, patch_start_idx, P):
    start = view_id * T + patch_start_idx
    end = start + P
    return start, end

def patch_centers_xy(H, W, Hp, Wp):
    """
    返回每个 patch 的中心像素坐标 (x, y)，shape=(P,2)
    """
    xs = (np.arange(Wp) + 0.5) / Wp * W
    ys = (np.arange(Hp) + 0.5) / Hp * H
    grid_x, grid_y = np.meshgrid(xs, ys)  # Hp x Wp
    centers = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    return centers  # (P,2)

# ============================================================
# 5) 距离 epipolar line：l=[a,b,c], dist = |ax+by+c| / sqrt(a^2+b^2)
# ============================================================

def point_line_distance(l: np.ndarray, xy: np.ndarray) -> np.ndarray:
    a, b, c = l.astype(np.float64)
    denom = math.sqrt(a*a + b*b) + 1e-12
    x = xy[:, 0]
    y = xy[:, 1]
    return np.abs(a * x + b * y + c) / denom

# ============================================================
# 6) 主流程：三指标计算 + 曲线图
# ============================================================

def main():
    # ---- 读取 target rect，拿 H,W（用于 QUERY_POS 与 patch 网格映射）
    target_pos_id = VIEW_ID_TO_POS_ID[TARGET_VIEW]
    target_rect_path = find_rect_image(target_pos_id)
    img_t = np.asarray(Image.open(target_rect_path).convert("RGB"))
    H_img, W_img = img_t.shape[:2]
    print(f"[信息] target rect: {target_rect_path}")
    print(f"[信息] rect size H,W=({H_img},{W_img})")

    # ---- 用第一层 probe attention 推断 token 结构
    probe_path = find_attn_file(TARGET_LAYERS[0])
    A0 = load_attn_matrix(probe_path)
    N, T, P, patch_start_idx = infer_T_and_P_from_attn(A0, NUM_VIEWS, NUM_SPECIAL_PER_VIEW)
    print(f"[推断] N={N}, 每视图T={T}, patch_start_idx={patch_start_idx}, 每视图patch P={P}")

    # ---- 推断 patch 网格 Hp,Wp（与你之前 debug 一致应为 28x37）
    Hp, Wp = infer_patch_hw(P, H_img, W_img)
    print(f"[推断] patch grid Hp,Wp=({Hp},{Wp}) -> P={Hp*Wp}")

    # ---- 映射 QUERY_POS 到 patch idx（在 rect 像素坐标系）
    qy, qx = QUERY_POS
    py, px, patch_idx = pos_to_patch_index(qy, qx, H_img, W_img, Hp, Wp)
    q_global = global_index_of_view_patch(TARGET_VIEW, patch_idx, T, patch_start_idx)
    print(f"[query] QUERY_POS(y,x)={QUERY_POS} -> (py,px)=({py},{px}) patch_idx={patch_idx} global_q={q_global}")

    # ---- 预先计算 target 的投影矩阵
    P_t = load_projection_matrix(find_pos_file(target_pos_id))

    # ---- other views 列表
    other_views = [v for v in range(NUM_VIEWS) if v != TARGET_VIEW]

    # ---- patch centers（用于算每个 patch 到 epipolar line 的距离）
    centers = patch_centers_xy(H_img, W_img, Hp, Wp)  # (P,2)

    # ---- 记录
    records = []

    for layer in TARGET_LAYERS:
        attn_path = find_attn_file(layer)
        A = load_attn_matrix(attn_path)
        if A.shape[0] != N:
            raise ValueError(f"[错误] layer{layer} 的 N={A.shape[0]} 与 probe N={N} 不一致")

        # 取 query 行（注意：这是 query 给出的分布）
        row = A[q_global, :]  # (N,)

        for v in other_views:
            pos_id_v = VIEW_ID_TO_POS_ID[v]
            P_v = load_projection_matrix(find_pos_file(pos_id_v))

            # epipolar line on view v: l = F_{v<-t} x_t
            F_vt = fundamental_from_projections(P_t, P_v)
            x1 = np.array([qx, qy, 1.0], dtype=np.float64)
            l = F_vt @ x1  # (3,)

            # view v 的 patch key slice
            k0, k1 = view_patch_range(v, T, patch_start_idx, P)
            w = row[k0:k1].astype(np.float64)  # (P,)
            w = np.maximum(w, 0.0)

            # 注意：w 是“这个 query 对 view v 的 patch keys 的权重”
            # 但 row softmax 是对整个 N keys 的，所以 sum(w) 通常 < 1
            mass_total_to_view_patch = float(w.sum()) + 1e-12

            # 每个 patch center 到极线距离
            d = point_line_distance(l, centers)  # (P,)

            # band 内 mask
            band = (d <= BAND_PX)

            # 1) Epipolar line mass（在该 view 的 patch keys 内的质量占比）
            epi_mass = float(w[band].sum() / mass_total_to_view_patch)

            # 2) Expected distance（在该 view patch keys 内归一化后距离期望）
            p = w / mass_total_to_view_patch
            exp_dist = float((p * d).sum())

            # 3) Top-k hit ratio（在该 view patch keys 的 Topk）
            k = min(TOPK, P)
            topk_idx = np.argpartition(-w, k-1)[:k]
            hit = float(band[topk_idx].mean()) if k > 0 else 0.0

            records.append({
                "layer": layer,
                "target_view": TARGET_VIEW,
                "other_view": v,
                "target_pos_id": target_pos_id,
                "other_pos_id": pos_id_v,
                "query_pos_y": qy,
                "query_pos_x": qx,
                "query_patch_py": py,
                "query_patch_px": px,
                "Hp": Hp,
                "Wp": Wp,
                "P": P,
                "band_px": BAND_PX,
                "topk": TOPK,

                # 辅助：该 query 分给 view v patch 的总质量（越大说明跨视越强）
                "mass_total_to_view_patch": float(mass_total_to_view_patch),

                # 三指标
                "epipolar_line_mass": epi_mass,
                "expected_distance": exp_dist,
                "topk_hit_ratio": hit,
            })

        print(f"[完成] layer {layer:02d}")

    # ---- 保存 CSV
    csv_path = os.path.join(OUT_DIR, "epipolar_attention_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[保存] CSV: {csv_path}")

    # ============================================================
    # 7) 汇总并画曲线（对所有 other views 求平均）
    # ============================================================

    def avg_by_layer(key: str):
        out = []
        for layer in TARGET_LAYERS:
            vals = [r[key] for r in records if r["layer"] == layer]
            out.append(float(np.mean(vals)) if vals else 0.0)
        return out

    y_epi_mass = avg_by_layer("epipolar_line_mass")
    y_exp_dist = avg_by_layer("expected_distance")
    y_topk_hit = avg_by_layer("topk_hit_ratio")
    y_cross_mass = avg_by_layer("mass_total_to_view_patch")

    # 画 1：epipolar mass
    plt.figure(figsize=(8, 4.5))
    plt.plot(TARGET_LAYERS, y_epi_mass, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Epipolar line mass (within band)")
    plt.title(f"Epipolar line mass vs Layer (band={BAND_PX}px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1 = os.path.join(OUT_DIR, "curve_epipolar_line_mass.png")
    plt.savefig(fig1, dpi=300)
    plt.close()

    # 画 2：expected distance
    plt.figure(figsize=(8, 4.5))
    plt.plot(TARGET_LAYERS, y_exp_dist, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Expected distance to epipolar line (px)")
    plt.title("Expected distance vs Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2 = os.path.join(OUT_DIR, "curve_expected_distance.png")
    plt.savefig(fig2, dpi=300)
    plt.close()

    # 画 3：top-k hit ratio
    plt.figure(figsize=(8, 4.5))
    plt.plot(TARGET_LAYERS, y_topk_hit, marker="o")
    plt.xlabel("Layer")
    plt.ylabel(f"Top-{TOPK} hit ratio in band")
    plt.title(f"Top-{TOPK} hit ratio vs Layer (band={BAND_PX}px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3 = os.path.join(OUT_DIR, "curve_topk_hit_ratio.png")
    plt.savefig(fig3, dpi=300)
    plt.close()

    # 可选：画 4（辅助）：该 query 分给其它视图 patch 的总质量（跨视强度）
    plt.figure(figsize=(8, 4.5))
    plt.plot(TARGET_LAYERS, y_cross_mass, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Mass to other-view PATCH keys (sum)")
    plt.title("Cross-view attention mass to PATCH keys vs Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4 = os.path.join(OUT_DIR, "curve_mass_to_other_view_patch.png")
    plt.savefig(fig4, dpi=300)
    plt.close()

    print(f"[保存] 曲线图:\n- {fig1}\n- {fig2}\n- {fig3}\n- {fig4}")
    print("\n全部完成。输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()
