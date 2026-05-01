import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ============================================================
# 0) 你只需要改这些
# ============================================================
DATA_DIR = r"datasets/scene1_DTU"  # 你的 DTU 子数据集目录（里面有 pos_XXX + rect_XXX_max.*）
OUT_DIR  = "outputs/token_attention/scene1_DTU/patch_attention_to_views/epipolar_lines"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_ID = 1          # 例如 1 对应 pos_001 和 rect_001_max.png
QUERY_POS = (600, 400) # (y, x) 像素坐标（注意：y在前）
DRAW_ALL_PAIRS = True  # True：target->所有其它；False：只画指定列表
OTHER_IDS = [3, 5, 7]  # 如果 DRAW_ALL_PAIRS=False，就用这个列表

# 可视化参数
LINE_WIDTH = 6
POINT_SIZE = 80
SAVE_DPI = 200

# 线 mask 保存参数
MASK_LINE_WIDTH = 6     # mask 里线宽（像素）
MASK_VALUE = 255        # 线像素强度（0-255）
# ============================================================


# ============================================================
# 1) 读 pos_XXX -> P (3x4)
# ============================================================
def load_projection_matrix(pos_path: str) -> np.ndarray:
    """
    兼容：
      - 3 行，每行 4 个数
      - 或者 12 个数（任意换行）
    """
    with open(pos_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().strip()
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)]
    if len(nums) != 12:
        raise ValueError(f"[错误] {pos_path} 解析到 {len(nums)} 个数，期望 12 个（3x4）")
    P = np.array(nums, dtype=np.float64).reshape(3, 4)
    return P


# ============================================================
# 2) 从 P 取相机中心 C（P 的右零空间）
# ============================================================
def camera_center_from_P(P: np.ndarray) -> np.ndarray:
    """
    C 是 4D 齐次坐标，满足 P C = 0
    用 SVD 求右零空间。
    """
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1, :]
    if abs(C[-1]) > 1e-12:
        C = C / C[-1]
    return C


# ============================================================
# 3) skew 矩阵 [e]_x
# ============================================================
def skew(v: np.ndarray) -> np.ndarray:
    v = v.reshape(-1)
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ], dtype=np.float64)


# ============================================================
# 4) 用两个投影矩阵算 Fundamental Matrix
#    F = [e']_x P' pinv(P)
# ============================================================
def fundamental_from_projections(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    给定 P1 (target), P2 (other)，返回 F_21，使得：
        l2 = F_21 x1
    """
    C1 = camera_center_from_P(P1)            # (4,)
    e2 = P2 @ C1                             # (3,)
    # 不强制除以 e2[2]，避免极点附近数值炸
    P1_pinv = np.linalg.pinv(P1)             # (4,3)
    M = P2 @ P1_pinv                         # (3,3)
    F = skew(e2) @ M                         # (3,3)

    # 归一化，便于稳定保存（不是必须，但推荐）
    n = np.linalg.norm(F)
    if n > 1e-12:
        F = F / n
    return F


# ============================================================
# 5) line ax+by+c=0 与边界求交，得到可画的两端点
# ============================================================
def line_border_intersections(l: np.ndarray, W: int, H: int):
    a, b, c = float(l[0]), float(l[1]), float(l[2])
    pts = []

    # x=0, x=W-1
    for x in [0.0, float(W - 1)]:
        if abs(b) > 1e-12:
            y = -(a * x + c) / b
            if 0 <= y <= H - 1:
                pts.append((x, y))

    # y=0, y=H-1
    for y in [0.0, float(H - 1)]:
        if abs(a) > 1e-12:
            x = -(b * y + c) / a
            if 0 <= x <= W - 1:
                pts.append((x, y))

    # 去重
    uniq = []
    for p in pts:
        if all((abs(p[0]-q[0]) > 1e-6 or abs(p[1]-q[1]) > 1e-6) for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return None

    # 若 >2，取最远的两个
    if len(uniq) > 2:
        best = None
        best_d = -1
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                dx = uniq[i][0] - uniq[j][0]
                dy = uniq[i][1] - uniq[j][1]
                d = dx*dx + dy*dy
                if d > best_d:
                    best_d = d
                    best = (uniq[i], uniq[j])
        return best

    return (uniq[0], uniq[1])


# ============================================================
# 6) 文件匹配：pos_001 / rect_001_max.png
# ============================================================
def find_pos_file(view_id: int) -> str:
    patt = os.path.join(DATA_DIR, f"pos_{view_id:03d}*")
    cands = sorted(glob.glob(patt))
    if not cands:
        raise FileNotFoundError(f"[错误] 找不到 {patt}")
    return cands[0]

def find_rect_image(view_id: int) -> str:
    patt = os.path.join(DATA_DIR, f"rect_{view_id:03d}_max.*")
    cands = sorted(glob.glob(patt))
    if cands:
        return cands[0]
    patt2 = os.path.join(DATA_DIR, f"rect_{view_id:03d}.*")
    cands2 = sorted(glob.glob(patt2))
    if not cands2:
        raise FileNotFoundError(f"[错误] 找不到 rect 图像：{patt} 或 {patt2}")
    return cands2[0]

def list_available_ids():
    pos_files = sorted(glob.glob(os.path.join(DATA_DIR, "pos_*")))
    ids = []
    for p in pos_files:
        m = re.search(r"pos_(\d+)", os.path.basename(p))
        if m:
            ids.append(int(m.group(1)))
    return sorted(list(set(ids)))


# ============================================================
# 7) 生成线 mask（单通道）
# ============================================================
def make_line_mask(W, H, pA, pB, line_width=6, value=255):
    """
    返回 uint8 (H,W) 的 mask，线位置为 value
    """
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.line([pA, pB], fill=int(value), width=int(line_width))
    return np.asarray(mask_img, dtype=np.uint8)


# ============================================================
# 8) 主流程
# ============================================================
def main():
    avail_ids = list_available_ids()
    if TARGET_ID not in avail_ids:
        raise ValueError(f"[错误] TARGET_ID={TARGET_ID} 不在可用 pos 列表中：{avail_ids}")

    if DRAW_ALL_PAIRS:
        ids = [i for i in avail_ids if i != TARGET_ID]
    else:
        ids = [i for i in OTHER_IDS if i != TARGET_ID]

    # 读 target 的 P 和图像
    pos_t = find_pos_file(TARGET_ID)
    img_t_path = find_rect_image(TARGET_ID)
    P_t = load_projection_matrix(pos_t)

    img_t = np.asarray(Image.open(img_t_path).convert("RGB"), dtype=np.float32) / 255.0
    Ht, Wt = img_t.shape[:2]

    y, x = QUERY_POS
    x1 = np.array([x, y, 1.0], dtype=np.float64)  # homogeneous

    # 保存 target 标点图
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.imshow(img_t)
    ax.scatter([x], [y], s=POINT_SIZE, c="red", marker="*")
    ax.set_title(f"Target rect_{TARGET_ID:03d} | point (x={x}, y={y})")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_t = os.path.join(OUT_DIR, f"target_{TARGET_ID:03d}_point.png")
    fig.savefig(out_t, dpi=SAVE_DPI)
    plt.close(fig)

    print(f"[保存] target 标点图：{out_t}")
    print(f"[信息] 将对 {len(ids)} 个其它视图绘制 epipolar line，并保存 mask + 数值。")

    for j in ids:
        pos_j = find_pos_file(j)
        img_j_path = find_rect_image(j)
        P_j = load_projection_matrix(pos_j)

        img_j = np.asarray(Image.open(img_j_path).convert("RGB"), dtype=np.float32) / 255.0
        Hj, Wj = img_j.shape[:2]

        # 计算 F_{j <- t}：l_j = F * x_t
        F_jt = fundamental_from_projections(P_t, P_j)
        l = F_jt @ x1  # (3,)  line in view j

        # 让 (a,b) 有单位长度，便于比较与保存
        ab = np.linalg.norm(l[:2])
        if ab > 1e-12:
            l = l / ab

        seg = line_border_intersections(l, Wj, Hj)
        if seg is None:
            print(f"[跳过] view {j:03d}：极线与图像边界无有效交点（可能点在极点附近/数值问题）。")
            continue

        (xA, yA), (xB, yB) = seg

        # 1) 保存 “线叠在图上” 的可视化（红线）
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.imshow(img_j)
        ax.plot([xA, xB], [yA, yB], linewidth=LINE_WIDTH, color="red")
        ax.set_title(f"Epipolar line on rect_{j:03d} (from rect_{TARGET_ID:03d} point)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()

        out_vis = os.path.join(OUT_DIR, f"epi_target{TARGET_ID:03d}_to_{j:03d}.png")
        fig.savefig(out_vis, dpi=SAVE_DPI)
        plt.close(fig)

        # 2) 保存 mask（单通道）
        mask = make_line_mask(Wj, Hj, (xA, yA), (xB, yB),
                              line_width=MASK_LINE_WIDTH, value=MASK_VALUE)
        out_mask = os.path.join(OUT_DIR, f"epi_target{TARGET_ID:03d}_to_{j:03d}_mask.png")
        Image.fromarray(mask).save(out_mask)

        # 3) 保存数值（npz）
        out_npz = os.path.join(OUT_DIR, f"epi_target{TARGET_ID:03d}_to_{j:03d}_line.npz")
        np.savez(
            out_npz,
            target_id=np.int32(TARGET_ID),
            other_id=np.int32(j),
            query_xy=np.array([x, y], dtype=np.float64),
            line_abc=l.astype(np.float64),               # ax+by+c=0
            pA=np.array([xA, yA], dtype=np.float64),
            pB=np.array([xB, yB], dtype=np.float64),
            W=np.int32(Wj),
            H=np.int32(Hj),
        )

        print(f"[保存] {out_vis}")
        print(f"[保存] {out_mask}")
        print(f"[保存] {out_npz}")

    print("\n全部完成。输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()
