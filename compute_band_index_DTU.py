import os
import re
import glob
import math
import numpy as np
import torch
from PIL import Image

# ============================================================
# 0) ======= 你只需要改这里 =======
# ============================================================

# DTU 7-view 子集目录（包含 rect_XXX(_max).* 和 pos_XXX*）
DATA_DIR = r"datasets/scene1_DTU"

# 输出索引文件（会自动在文件名里加 _bwPatchX）
OUT_PT_TEMPLATE = r"outputs/token_attention/scene1_DTU/add_epipolar_band/scene1_band_bw{bw}.pt"

# view / token 配置
NUM_VIEWS = 7
NUM_SPECIAL_PER_VIEW = 5     # 1 camera + 4 register

# >>> Epipolar Band 半宽（单位：patch） <<<
# 表示：距离极线 <= bw_patch 个 patch（半宽）
BAND_WIDTH_IN_PATCH_LIST = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# view_id -> pos_id 映射（与你 DTU 子集一致）
VIEW_ID_TO_POS_ID = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13}

# rect 文件名模式
RECT_PATTERN_1 = "rect_{:03d}_max.*"
RECT_PATTERN_2 = "rect_{:03d}.*"

# patch 网格（强烈建议直接写死，更稳）
PATCH_GRID_HW = (28, 37)  # 28*37=1036

# 如果 PATCH_GRID_HW=None，则必须给一个 attention probe 来推断 token 数
ATTN_PROBE = None

# 数值稳定
EPS = 1e-12

# ============================================================
# 1) IO 工具：rect / pos / attention probe
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

    raise FileNotFoundError(f"[错误] 找不到 rect 图像: pos_{pos_id:03d}")

def find_pos_file(pos_id: int) -> str:
    patt = os.path.join(DATA_DIR, f"pos_{pos_id:03d}*")
    cands = sorted(glob.glob(patt))
    if not cands:
        raise FileNotFoundError(f"[错误] 找不到 pos 文件: pos_{pos_id:03d}")
    return cands[0]

def load_projection_matrix(pos_path: str) -> np.ndarray:
    with open(pos_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().strip()
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)]
    if len(nums) != 12:
        raise ValueError(f"[错误] {pos_path} 解析到 {len(nums)} 个数（应为 12）")
    return np.array(nums, dtype=np.float64).reshape(3, 4)

def load_attn_probe(attn_path: str) -> np.ndarray:
    A = torch.load(attn_path, map_location="cpu")
    if isinstance(A, (list, tuple)):
        A = A[0]
    if A.ndim == 3:
        A = A[0]
    A = A.detach().cpu().numpy()
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"[错误] attention 不是方阵: {A.shape}")
    return A

# ============================================================
# 2) 多视几何：由投影矩阵计算 Fundamental Matrix
# ============================================================

def camera_center_from_P(P: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(P)
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

def fundamental_from_projections(P_src: np.ndarray, P_dst: np.ndarray) -> np.ndarray:
    C_src = camera_center_from_P(P_src)
    e_dst = P_dst @ C_src
    if abs(e_dst[2]) > 1e-12:
        e_dst = e_dst / e_dst[2]
    P_src_pinv = np.linalg.pinv(P_src)
    M = P_dst @ P_src_pinv
    F = skew(e_dst) @ M
    n = np.linalg.norm(F)
    if n > 1e-12:
        F = F / n
    return F

# ============================================================
# 3) patch 网格 + patch 中心点（像素坐标）
# ============================================================

def factor_pairs(n: int):
    pairs = []
    for a in range(1, int(math.sqrt(n)) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
    return pairs

def infer_patch_hw(P_patch: int, H: int, W: int):
    target_ratio = H / max(W, 1e-6)
    best_hw, best_err = None, 1e18
    for hp, wp in factor_pairs(P_patch):
        ratio = hp / max(wp, 1e-6)
        err = abs(math.log((ratio / target_ratio) + 1e-12))
        if err < best_err:
            best_err, best_hw = err, (hp, wp)
    if best_hw is None:
        raise ValueError("[错误] 无法推断 patch 网格")
    return best_hw

def build_patch_centers(H: int, W: int, Hp: int, Wp: int) -> np.ndarray:
    xs = (np.arange(Wp) + 0.5) / Wp * W
    ys = (np.arange(Hp) + 0.5) / Hp * H
    gx, gy = np.meshgrid(xs, ys)
    centers = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)  # (P,2) [x,y]
    return centers.astype(np.float64)

def patch_px_avg(H: int, W: int, Hp: int, Wp: int) -> float:
    # 一个 patch 的平均像素尺寸（用于把像素距离换算成 patch 距离）
    px_x = W / float(Wp)
    px_y = H / float(Hp)
    return 0.5 * (px_x + px_y)

# ============================================================
# 4) 主流程：计算 CSR band patch 索引（按 patch 单位）
# ============================================================

def main():
    # ---- 读取每个 view 的 rect 尺寸 和 投影矩阵
    rect_sizes = {}
    rect_paths = {}
    Proj = {}

    for v in range(NUM_VIEWS):
        pos_id = VIEW_ID_TO_POS_ID[v]
        rpath = find_rect_image(pos_id)
        img = Image.open(rpath)
        W, H = img.size
        rect_sizes[v] = (H, W)
        rect_paths[v] = rpath
        Proj[v] = load_projection_matrix(find_pos_file(pos_id))

    # ---- 推断 token / patch 结构
    if PATCH_GRID_HW is not None:
        Hp, Wp = PATCH_GRID_HW
        P_patch = Hp * Wp
        T = NUM_SPECIAL_PER_VIEW + P_patch
    else:
        if ATTN_PROBE is None:
            raise ValueError("[错误] PATCH_GRID_HW=None 时必须提供 ATTN_PROBE")
        A = load_attn_probe(ATTN_PROBE)
        N = A.shape[0]
        if N % NUM_VIEWS != 0:
            raise ValueError(f"[错误] N={N} 不能被 NUM_VIEWS={NUM_VIEWS} 整除")
        T = N // NUM_VIEWS
        P_patch = T - NUM_SPECIAL_PER_VIEW
        H0, W0 = rect_sizes[0]
        Hp, Wp = infer_patch_hw(P_patch, H0, W0)

    assert Hp * Wp == P_patch
    patch_start = NUM_SPECIAL_PER_VIEW

    print(f"[推断] T={T}, P_patch={P_patch}, Hp,Wp=({Hp},{Wp}), patch_start={patch_start}")
    print("[rect] example:", os.path.basename(rect_paths[0]), "size(H,W)=", rect_sizes[0])

    # ---- 计算每个 view 的 patch 中心点 + patch_px
    patch_centers = {}
    patch_px = {}
    for v in range(NUM_VIEWS):
        H, W = rect_sizes[v]
        patch_centers[v] = build_patch_centers(H, W, Hp, Wp)
        patch_px[v] = patch_px_avg(H, W, Hp, Wp)

    # 预计算 F
    Fmat = {}
    for src in range(NUM_VIEWS):
        for dst in range(NUM_VIEWS):
            if src == dst:
                continue
            Fmat[(src, dst)] = fundamental_from_projections(Proj[src], Proj[dst])

    # ---- 对每个 band 半宽（patch）输出一个 pt
    os.makedirs(os.path.dirname(OUT_PT_TEMPLATE), exist_ok=True)

    for bw_patch in BAND_WIDTH_IN_PATCH_LIST:
        bw_patch = float(bw_patch)

        pair_csr_patchid = {}

        print(f"\n[RUN] band_half_width = {bw_patch} patches")

        for src in range(NUM_VIEWS):
            src_xy = patch_centers[src]  # (P,2)
            for dst in range(NUM_VIEWS):
                if src == dst:
                    continue

                dst_xy = patch_centers[dst]  # (P,2)
                dst_x = dst_xy[:, 0]
                dst_y = dst_xy[:, 1]

                # 这对 (src,dst) 的像素阈值：bw_patch * (dst 的 patch 像素尺寸)
                bw_px_used = bw_patch * patch_px[dst]

                F = Fmat[(src, dst)]
                offsets = np.zeros((P_patch + 1,), dtype=np.int64)
                indices_list = []

                for q in range(P_patch):
                    xq, yq = src_xy[q]
                    l = F @ np.array([xq, yq, 1.0], dtype=np.float64)
                    a, b, c = float(l[0]), float(l[1]), float(l[2])

                    denom = math.sqrt(a * a + b * b) + EPS
                    d_px = np.abs(a * dst_x + b * dst_y + c) / denom  # (P,)

                    # 转换为 patch 单位：d_patch = d_px / patch_px_dst
                    d_patch = d_px / (patch_px[dst] + EPS)

                    keep_patch = np.nonzero(d_patch <= bw_patch)[0].astype(np.int64)
                    offsets[q + 1] = offsets[q] + keep_patch.shape[0]
                    indices_list.append(keep_patch)

                indices = np.concatenate(indices_list, axis=0).astype(np.int64)

                pair_csr_patchid[(src, dst)] = {
                    "offsets": torch.from_numpy(offsets),
                    "indices": torch.from_numpy(indices),
                }

                # 日志：每个 query 平均保留多少 patch
                avg_keep = indices.shape[0] / float(P_patch)
                print(f"[pair] bw={bw_patch:g}patch | {src}->{dst} | avg_keep_per_query={avg_keep:.2f}")

        # ---- 保存
        if float(bw_patch).is_integer():
            bw_tag = str(int(bw_patch))        # 1.0 -> "1"
        else:
            bw_tag = str(bw_patch)  
        out_pt = OUT_PT_TEMPLATE.format(bw=bw_tag)

        save_obj = {
            "meta": {
                "num_views": NUM_VIEWS,
                "num_special_per_view": NUM_SPECIAL_PER_VIEW,
                "T": T,
                "P_patch": P_patch,
                "HpWp": (Hp, Wp),
                "patch_start": patch_start,

                # 新：用 patch 单位定义 band
                "band_width_in_patch": bw_patch,
                "band_unit": "patch",

                # 兼容/可查：给出一个“参考 view(0)”的像素阈值
                "ref_rect": os.path.basename(rect_paths[0]),
                "ref_rect_size_hw": rect_sizes[0],
                "ref_patch_px_avg": float(patch_px[0]),
                "ref_band_width_px_used": float(bw_patch * patch_px[0]),

                "view_id_to_pos_id": VIEW_ID_TO_POS_ID,
                "note": "DTU subset epipolar band CSR in patch_id space; threshold defined in patch units; special tokens handled at runtime.",
            },
            "pair_csr_patchid": pair_csr_patchid,
        }

        torch.save(save_obj, out_pt)
        print(f"[完成] saved -> {out_pt}")

if __name__ == "__main__":
    main()
