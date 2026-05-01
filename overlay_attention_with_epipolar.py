# overlay_rect_heatmap_epipolar_per_layer.py
import os
import re
import glob
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 0) 你只需要改这些配置
# ============================================================

# attention .pt 文件目录（里面应有 attn_layer00_*.pt 之类）
ATTN_DIR = r"outputs/token_attention/scene1_DTU/add_epipolar_band/bandwidth_compare/layer_16_to_23_bw3"

# DTU 子数据集目录（含 rect_XXX_max.png 和 pos_XXX）
DATA_DIR = r"datasets/scene1_DTU"

# 输出目录
OUT_DIR = ATTN_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# 需要处理的层（例如 0~23）
TARGET_LAYERS = [23]

# 视图数（必须与 attention 保存时一致）
NUM_VIEWS = 7

# 目标 view（0-based）
TARGET_VIEW = 0

# 你在 target rect 图上的点击点 (y, x) —— 注意是 (y,x)
QUERY_POS = (150, 200)

# 每个 view 的 special token 数：1 camera + 4 register = 5
NUM_SPECIAL_PER_VIEW = 5

# 只可视化 query->(其它帧 patch keys)，忽略 special keys
ONLY_PATCH_KEYS = True

# heatmap 叠加透明度
HEAT_ALPHA = 0.55

# 热力归一化模式："none" / "p99" / "log"
SCALE_MODE = "p99"
LOG_EPS = 1e-12

# view_id -> pos_id 映射（按你实际拷贝到子集目录里的 pos 文件来写）
# 例：pos_001 pos_003 pos_005 pos_007 pos_009 pos_011 pos_013
VIEW_ID_TO_POS_ID = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13}

# rect 图像模式：优先 rect_XXX_max，其次 rect_XXX
RECT_PATTERN_1 = "rect_{:03d}_max.*"
RECT_PATTERN_2 = "rect_{:03d}.*"

# 画线样式
LINE_COLOR = "red"
LINE_WIDTH = 6.0

# 是否保存 target view 的 query 星标图
SAVE_TARGET_STAR = True


# ============================================================
# 1) 工具：读 attention 文件
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
    c1 = sorted(glob.glob(patt1))
    if c1:
        return c1[0]
    patt2 = os.path.join(DATA_DIR, RECT_PATTERN_2.format(pos_id))
    c2 = sorted(glob.glob(patt2))
    if c2:
        return c2[0]
    raise FileNotFoundError(f"[错误] 找不到 rect 图像：{patt1} 或 {patt2}")

def find_pos_file(pos_id: int) -> str:
    patt = os.path.join(DATA_DIR, f"pos_{pos_id:03d}*")
    cands = sorted(glob.glob(patt))
    if not cands:
        raise FileNotFoundError(f"[错误] 找不到 pos 文件：{patt}")
    return cands[0]

def load_projection_matrix(pos_path: str) -> np.ndarray:
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


# ============================================================
# 5) heatmap 归一化 + 上采样（注意：最后用 extent 锁定坐标系）
# ============================================================

def normalize_heat(x, mode="p99"):
    x = x.astype(np.float32)
    x = np.maximum(x, 0.0)

    if mode == "none":
        mx = float(x.max())
        return x / (mx + 1e-12)

    if mode == "log":
        x = np.log(x + LOG_EPS)
        mn, mx = float(x.min()), float(x.max())
        return (x - mn) / (mx - mn + 1e-12)

    if mode == "p99":
        p = float(np.percentile(x, 99.0))
        p = max(p, 1e-12)
        return np.clip(x / p, 0.0, 1.0)

    raise ValueError(f"未知 SCALE_MODE: {mode}")

def upsample_heat_to_image(heat_hw, H, W):
    """
    heat_hw: (Hp,Wp) in [0,1]
    return:  (H,W)  in [0,1]
    """
    im = Image.fromarray((heat_hw * 255).astype(np.uint8))
    im = im.resize((W, H), resample=Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


# ============================================================
# 6) 极线 l=[a,b,c] 在图像边界内求两端点
# ============================================================

def line_segment_in_image(l, W, H):
    """
    l: [a,b,c] for a*x + b*y + c = 0
    return: (x1,y1,x2,y2) or None
    """
    a, b, c = float(l[0]), float(l[1]), float(l[2])
    pts = []

    # x = 0
    if abs(b) > 1e-12:
        y = -(a * 0 + c) / b
        if 0 <= y <= (H - 1):
            pts.append((0.0, y))

    # x = W-1
    if abs(b) > 1e-12:
        x = float(W - 1)
        y = -(a * x + c) / b
        if 0 <= y <= (H - 1):
            pts.append((x, y))

    # y = 0
    if abs(a) > 1e-12:
        x = -(b * 0 + c) / a
        if 0 <= x <= (W - 1):
            pts.append((x, 0.0))

    # y = H-1
    if abs(a) > 1e-12:
        y = float(H - 1)
        x = -(b * y + c) / a
        if 0 <= x <= (W - 1):
            pts.append((x, y))

    if len(pts) < 2:
        return None

    # 去重（避免同一点重复）
    uniq = []
    for p in pts:
        ok = True
        for q in uniq:
            if abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6:
                ok = False
                break
        if ok:
            uniq.append(p)
    if len(uniq) < 2:
        return None

    # 取最远的两点作为端点（更稳定）
    best = None
    best_d = -1
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            dx = uniq[i][0] - uniq[j][0]
            dy = uniq[i][1] - uniq[j][1]
            d = dx*dx + dy*dy
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])
    (x1, y1), (x2, y2) = best
    return x1, y1, x2, y2


# ============================================================
# 7) 主流程：每层、每视图输出 rect+heat+epipolar
# ============================================================

def main():
    # ---- target rect 读出来，用于推断 patch grid + query patch
    target_pos_id = VIEW_ID_TO_POS_ID[TARGET_VIEW]
    target_rect_path = find_rect_image(target_pos_id)
    target_img = np.asarray(Image.open(target_rect_path).convert("RGB"))
    Ht, Wt = target_img.shape[:2]
    print(f"[info] target rect: {target_rect_path} | H,W=({Ht},{Wt})")

    # ---- 用第一层 attention probe 推断 token 结构
    probe_path = find_attn_file(TARGET_LAYERS[0])
    A0 = load_attn_matrix(probe_path)
    N, T, P, patch_start_idx = infer_T_and_P_from_attn(A0, NUM_VIEWS, NUM_SPECIAL_PER_VIEW)
    print(f"[infer] N={N}, T(per-view)={T}, P(patch per-view)={P}, patch_start_idx={patch_start_idx}")

    # ---- 推断 patch 网格 Hp,Wp（与你之前一致应为 28x37）
    Hp, Wp = infer_patch_hw(P, Ht, Wt)
    print(f"[infer] patch grid Hp,Wp=({Hp},{Wp}) -> P={Hp*Wp}")

    # ---- query patch
    qy, qx = QUERY_POS
    py, px, patch_idx = pos_to_patch_index(qy, qx, Ht, Wt, Hp, Wp)
    q_global = global_index_of_view_patch(TARGET_VIEW, patch_idx, T, patch_start_idx)
    print(f"[query] (y,x)={QUERY_POS} -> patch(py,px)=({py},{px}), patch_idx={patch_idx}, q_global={q_global}")

    # ---- 投影矩阵
    P_t = load_projection_matrix(find_pos_file(target_pos_id))

    # ---- other views
    other_views = [v for v in range(NUM_VIEWS) if v != TARGET_VIEW]

    # ---- 可选：保存 target view 的星标图（每层都保存或只保存一次）
    if SAVE_TARGET_STAR:
        star_dir = os.path.join(OUT_DIR, "target_star")
        os.makedirs(star_dir, exist_ok=True)
        star_y = (py + 0.5) / Hp * Ht
        star_x = (px + 0.5) / Wp * Wt
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.imshow(target_img)
        ax.scatter([star_x], [star_y], s=120, marker="*", c="red")
        ax.set_title(f"Target view {TARGET_VIEW} (pos_{target_pos_id:03d}) | query patch *")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(star_dir, f"target_view{TARGET_VIEW}_pos{target_pos_id:03d}_query_star.png"), dpi=200)
        plt.close(fig)

    # ---- 逐层输出
    for layer in TARGET_LAYERS:
        attn_path = find_attn_file(layer)
        A = load_attn_matrix(attn_path)
        if A.shape[0] != N:
            raise ValueError(f"[错误] layer{layer:02d} 的 N={A.shape[0]} 与 probe N={N} 不一致")

        row = A[q_global, :]  # query -> keys

        layer_out = os.path.join(OUT_DIR, f"layer{layer:02d}")
        os.makedirs(layer_out, exist_ok=True)

        for v in other_views:
            other_pos_id = VIEW_ID_TO_POS_ID[v]

            # 读 other rect（用它的真实 H,W 来画）
            rect_path = find_rect_image(other_pos_id)
            rect_img = np.asarray(Image.open(rect_path).convert("RGB"))
            H, W = rect_img.shape[:2]

            # ---- 从 attention 取 view v 的 patch keys
            if ONLY_PATCH_KEYS:
                k0, k1 = view_patch_range(v, T, patch_start_idx, P)
                w = row[k0:k1].astype(np.float32)  # (P,)
                w = np.maximum(w, 0.0)
                # 关键：只在目标 view 的 patch keys 内做条件归一化
                w = w / (w.sum() + 1e-12)
                heat_hw = w.reshape(Hp, Wp)
            else:
                # 如果你要包括 special keys，就自己改；这里仍只画 patch 网格
                k0, k1 = v * T, (v + 1) * T
                w = row[k0:k1].astype(np.float32)
                heat_hw = w[patch_start_idx:].reshape(Hp, Wp)

            # ---- 归一化 + resize 到 rect 分辨率
            heat_hw = normalize_heat(heat_hw, SCALE_MODE)    # (Hp,Wp) -> [0,1]
            heat_up = upsample_heat_to_image(heat_hw, H, W)  # (H,W)  -> [0,1]

            # ---- 计算 epipolar line：l = F_{v<-t} * x_t
            P_v = load_projection_matrix(find_pos_file(other_pos_id))
            F_vt = fundamental_from_projections(P_t, P_v)
            x1 = np.array([qx, qy, 1.0], dtype=np.float64)   # 注意：这里用 rect 像素坐标
            l = F_vt @ x1

            seg = line_segment_in_image(l, W, H)

            # ---- 画图：全部在同一 ax、同一像素坐标系
            fig = plt.figure(figsize=(7.5, 5.5))
            ax = plt.gca()

            # 底图：rect 原图（像素坐标自然是 x∈[0,W), y∈[0,H)）
            ax.imshow(rect_img, extent=[0, W, H, 0])  # 固定坐标系：左上(0,0)

            # heatmap：同样用 extent=[0,W,H,0] 锁定坐标
            ax.imshow(
                heat_up,
                extent=[0, W, H, 0],
                alpha=HEAT_ALPHA,
            )

            # epipolar line
            if seg is not None:
                x1s, y1s, x2s, y2s = seg
                ax.plot([x1s, x2s], [y1s, y2s], color=LINE_COLOR, linewidth=LINE_WIDTH)
            else:
                # 极线可能完全不穿过图像范围（少见，但可能）
                ax.text(10, 30, "Epipolar line out of bounds", color="red", fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

            ax.set_title(f"Layer {layer:02d} | q=View{TARGET_VIEW} patch({py},{px}) -> View{v} (pos_{other_pos_id:03d})")
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()

            out_path = os.path.join(layer_out, f"layer{layer:02d}_qView{TARGET_VIEW}_to_view{v}_pos{other_pos_id:03d}_rect_heat_epi.png")
            fig.savefig(out_path, dpi=220)
            plt.close(fig)

        print(f"[done] layer {layer:02d} saved -> {layer_out}")

    print("\n全部完成。输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()
