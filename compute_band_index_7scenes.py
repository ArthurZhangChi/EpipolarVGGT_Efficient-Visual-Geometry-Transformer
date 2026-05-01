import os
import math
import numpy as np
import torch
from PIL import Image

# ============================================================
# 【全局配置区】——只需要改这里
# ============================================================

# 7Scenes 根目录（里面有 office / chess / ...）
DATA_ROOT = r"datasets/scene3_7Scenes"

# 你选的 10 个 view（用 frame id，不需要规律）
# 对应文件：frame-000000.color.png / frame-000000.pose.txt
# FRAME_IDS = [0, 52, 122, 200, 242, 270, 360, 410, 488, 538]
FRAME_IDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# 输出索引文件
OUT_PT = r"outputs/token_attention/scene3_7Scenes/scene3_band_bw42.pt"

# VGGT special token 配置（与你当前实现一致：1 camera + 4 register）
NUM_SPECIAL_PER_VIEW = 5

# patch size（VGGT 默认 14）
PATCH_SIZE = 14

# Epipolar Band 宽度（像素，基于“预处理后”的图像坐标系）
# 14px ≈ 1 patch；42px ≈ 3 patch
BAND_WIDTH_PX = 42.0

# 预处理模式：必须与你跑 VGGT 的输入一致（官方 quickstart 默认 crop）
# - crop: 宽固定 518，高按比例缩放并 round 到 14 的倍数
# - pad : 最大边缩放到 518，再 pad 成 518×518（若你用 pad，就选 pad）
PREPROCESS_MODE = "crop"  # "crop" or "pad"
TARGET_SIZE = 518         # 官方默认 518

# 7Scenes 原始 RGB 分辨率一般是 640×480（建议自动读取，不要手写）
# 但内参 RAW_K 需要你提供：这是“原始分辨率坐标系”的 K
# 注意：不同实现/来源可能略有差异，你务必用你工程里实际使用的那套 K。
RAW_K = np.array([
    [585.0,   0.0, 320.0],
    [  0.0, 585.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

# 是否打印更多 debug
VERBOSE = True

# ============================================================
# 工具函数：文件路径/读取 pose
# ============================================================

def frame_prefix(fid: int) -> str:
    return f"frame-{fid:06d}"

def color_path(fid: int) -> str:
    return os.path.join(DATA_ROOT, frame_prefix(fid) + ".color.png")

def pose_path(fid: int) -> str:
    return os.path.join(DATA_ROOT, frame_prefix(fid) + ".pose.txt")

def load_c2w_4x4(path: str) -> np.ndarray:
    """读取 7Scenes 的 camera-to-world 4×4"""
    M = np.loadtxt(path).astype(np.float64)
    M = M.reshape(4, 4)
    return M

def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """c2w -> w2c"""
    return np.linalg.inv(c2w)

# ============================================================
# 工具函数：复刻 VGGT 的 preprocess 尺寸变换，并把 K warp 到新坐标系
# ============================================================

def round_to_14(x: float) -> int:
    """四舍五入到 14 的倍数（与官方逻辑一致：round(...) * 14）"""
    return int(round(x / 14.0) * 14)

def preprocess_size_and_offsets(raw_w: int, raw_h: int, mode: str, target_size: int):
    """
    返回：
      new_w, new_h：resize 后（并已 round 到 14 倍数）的尺寸
      crop_top：若 crop 且 new_h > target_size，会做居中裁剪，这里返回裁剪的 top 偏移
      pad_left, pad_top：若 pad，会居中 padding 到 target_size×target_size，这里返回 padding 偏移
    """
    if mode not in ["crop", "pad"]:
        raise ValueError("PREPROCESS_MODE 必须是 crop 或 pad")

    if mode == "crop":
        new_w = target_size
        new_h_float = raw_h * (new_w / raw_w)
        new_h = round_to_14(new_h_float)

        crop_top = 0
        if new_h > target_size:
            crop_top = (new_h - target_size) // 2
            new_h = target_size  # 裁剪后高度就是 target_size

        return new_w, new_h, crop_top, 0, 0

    # pad 模式：最大边缩放到 518，再 pad 成 518×518
    if raw_w >= raw_h:
        new_w = target_size
        new_h = round_to_14(raw_h * (new_w / raw_w))
    else:
        new_h = target_size
        new_w = round_to_14(raw_w * (new_h / raw_h))

    pad_top = (target_size - new_h) // 2 if target_size > new_h else 0
    pad_left = (target_size - new_w) // 2 if target_size > new_w else 0

    # pad 后最终就是 target_size×target_size
    return target_size, target_size, 0, pad_left, pad_top

def warp_K_raw_to_processed(K_raw: np.ndarray, raw_w: int, raw_h: int, mode: str, target_size: int) -> np.ndarray:
    """
    把“原始分辨率坐标系”的 K 映射到“预处理后坐标系”的 K
    - resize：fx,fy 和 cx,cy 按缩放变
    - crop：cy 需要减去 crop_top
    - pad：cx,cy 需要加上 pad_left/pad_top
    """
    # 先得到 resize 后（round 到 14 倍数前后一致）的中间尺寸与偏移
    # 注意：crop 情况下我们在 preprocess_size_and_offsets 里把 new_h 改成 target_size（裁剪后）
    # 但 resize 的缩放因子应基于“裁剪前的 new_h_resize”
    if mode == "crop":
        new_w_resize = target_size
        new_h_resize = round_to_14(raw_h * (new_w_resize / raw_w))
        crop_top = 0
        if new_h_resize > target_size:
            crop_top = (new_h_resize - target_size) // 2
        sx = new_w_resize / raw_w
        sy = new_h_resize / raw_h

        K = K_raw.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        # crop：y 方向坐标整体上移 crop_top
        K[1, 2] -= crop_top

        # 最终坐标系尺寸是 (target_size, min(target_size, new_h_resize裁剪后))
        return K

    # pad：先缩放到 new_w_resize/new_h_resize，然后再加 padding 偏移
    # 这里最终输出坐标系是 518×518
    if raw_w >= raw_h:
        new_w_resize = target_size
        new_h_resize = round_to_14(raw_h * (new_w_resize / raw_w))
    else:
        new_h_resize = target_size
        new_w_resize = round_to_14(raw_w * (new_h_resize / raw_h))

    sx = new_w_resize / raw_w
    sy = new_h_resize / raw_h

    pad_top = (target_size - new_h_resize) // 2 if target_size > new_h_resize else 0
    pad_left = (target_size - new_w_resize) // 2 if target_size > new_w_resize else 0

    K = K_raw.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy

    # pad：坐标整体右移/下移
    K[0, 2] += pad_left
    K[1, 2] += pad_top
    return K

# ============================================================
# 多视几何：由 (R,t,K) 计算 Fundamental Matrix
# ============================================================

def skew(v: np.ndarray) -> np.ndarray:
    v = v.reshape(-1)
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ], dtype=np.float64)

def fundamental_from_w2c(w2c_i: np.ndarray, w2c_j: np.ndarray, K_i: np.ndarray, K_j: np.ndarray) -> np.ndarray:
    """
    输入：两个相机的 w2c（OpenCV 约定：x_cam = R X_world + t），以及各自内参 K
    输出：F，使得 l_j = F x_i
    """
    R_i = w2c_i[:3, :3]
    t_i = w2c_i[:3, 3]
    R_j = w2c_j[:3, :3]
    t_j = w2c_j[:3, 3]

    # 从相机 i 坐标到相机 j 坐标的相对变换：x_j = R_ji x_i + t_ji
    R_ji = R_j @ R_i.T
    t_ji = t_j - R_ji @ t_i

    E = skew(t_ji) @ R_ji
    F = np.linalg.inv(K_j).T @ E @ np.linalg.inv(K_i)

    n = np.linalg.norm(F)
    if n > 1e-12:
        F = F / n
    return F

# ============================================================
# patch 网格与 patch 中心（在“预处理后坐标系”里）
# ============================================================

def build_patch_centers(H: int, W: int, Hp: int, Wp: int) -> np.ndarray:
    """
    为每个 patch 计算中心点像素坐标 (x,y)，返回 (P,2)
    """
    xs = (np.arange(Wp) + 0.5) / Wp * W
    ys = (np.arange(Hp) + 0.5) / Hp * H
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

# ============================================================
# 主流程：计算 CSR band patch 索引并保存
# ============================================================

def main():
    # 1) 读一张 color 来拿原始分辨率（一般 640×480）
    p0 = color_path(FRAME_IDS[0])
    if not os.path.exists(p0):
        raise FileNotFoundError(f"找不到：{p0}")
    img0 = Image.open(p0)
    raw_w, raw_h = img0.size

    # 2) 计算预处理后的尺寸（必须与 VGGT 输入一致）
    proc_w, proc_h, crop_top, pad_left, pad_top = preprocess_size_and_offsets(
        raw_w, raw_h, PREPROCESS_MODE, TARGET_SIZE
    )

    # 3) 由最终输入尺寸决定 patch 网格（VGGT patch=14）
    #    注意：最终输入尺寸必须能被 14 整除（官方预处理就是为了这个）
    if (proc_w % PATCH_SIZE) != 0 or (proc_h % PATCH_SIZE) != 0:
        raise ValueError(f"预处理后尺寸 ({proc_w},{proc_h}) 不能被 patch={PATCH_SIZE} 整除")

    Wp = proc_w // PATCH_SIZE
    Hp = proc_h // PATCH_SIZE
    P_patch = Hp * Wp
    T = NUM_SPECIAL_PER_VIEW + P_patch

    if VERBOSE:
        print(f"[Raw size]  W,H = {raw_w},{raw_h}")
        print(f"[Proc size] W,H = {proc_w},{proc_h} | mode={PREPROCESS_MODE} | target={TARGET_SIZE}")
        if PREPROCESS_MODE == "crop":
            print(f"  crop_top = {crop_top}")
        else:
            print(f"  pad_left, pad_top = {pad_left}, {pad_top}")
        print(f"[Patch grid] Hp,Wp = {Hp},{Wp} | P_patch={P_patch} | T(per-view)={T}")
        print(f"[Band] width_px = {BAND_WIDTH_PX}")

    # 4) warp 内参到预处理坐标系（所有帧同分辨率时 K 一样）
    K_proc = warp_K_raw_to_processed(RAW_K, raw_w, raw_h, PREPROCESS_MODE, TARGET_SIZE)
    if VERBOSE:
        fx, fy, cx, cy = K_proc[0,0], K_proc[1,1], K_proc[0,2], K_proc[1,2]
        print(f"[K_proc] fx,fy,cx,cy = {fx:.3f}, {fy:.3f}, {cx:.3f}, {cy:.3f}")

    # 5) 读取每个 view 的 w2c（由 pose 的 c2w 取逆得到）
    NUM_VIEWS = len(FRAME_IDS)
    w2c_list = []
    for fid in FRAME_IDS:
        pp = pose_path(fid)
        if not os.path.exists(pp):
            raise FileNotFoundError(f"找不到：{pp}")
        c2w = load_c2w_4x4(pp)
        w2c = c2w_to_w2c(c2w)
        w2c_list.append(w2c)

    # 6) patch centers（在预处理后坐标系）
    patch_centers = build_patch_centers(proc_h, proc_w, Hp, Wp)  # (P_patch,2)

    # 7) 计算每对 view 的 CSR（只跨 view：src!=dst）
    pair_csr_patchid = {}

    for src in range(NUM_VIEWS):
        for dst in range(NUM_VIEWS):
            if src == dst:
                continue

            F = fundamental_from_w2c(w2c_list[src], w2c_list[dst], K_proc, K_proc)

            offsets = np.zeros((P_patch + 1,), dtype=np.int64)
            indices_list = []

            src_xy = patch_centers
            dst_xy = patch_centers

            for q in range(P_patch):
                xq, yq = src_xy[q]
                l = F @ np.array([xq, yq, 1.0], dtype=np.float64)
                a, b, c = float(l[0]), float(l[1]), float(l[2])

                denom = math.sqrt(a * a + b * b) + 1e-12
                d = np.abs(a * dst_xy[:, 0] + b * dst_xy[:, 1] + c) / denom

                keep_patch = np.nonzero(d <= BAND_WIDTH_PX)[0].astype(np.int64)
                offsets[q + 1] = offsets[q] + keep_patch.shape[0]
                indices_list.append(keep_patch)

            indices = np.concatenate(indices_list, axis=0).astype(np.int64)

            pair_csr_patchid[(src, dst)] = {
                "offsets": torch.from_numpy(offsets),
                "indices": torch.from_numpy(indices),
            }

            avg_keep = indices.shape[0] / float(P_patch)
            if VERBOSE:
                print(f"[pair] {src}->{dst} | nnz={indices.shape[0]} | avg_keep={avg_keep:.2f}")

    # 8) 保存（格式与你 DTU 那版一致）
    save_obj = {
        "meta": {
            "dataset": "7Scenes",
            "frame_ids": FRAME_IDS,
            "num_views": NUM_VIEWS,
            "num_special_per_view": NUM_SPECIAL_PER_VIEW,
            "patch_size": PATCH_SIZE,
            "T": T,
            "P_patch": P_patch,
            "HpWp": (Hp, Wp),
            "patch_start": NUM_SPECIAL_PER_VIEW,
            "band_width_px": BAND_WIDTH_PX,
            "band_unit": "pixel_in_processed_image",
            "preprocess_mode": PREPROCESS_MODE,
            "processed_size_wh": (proc_w, proc_h),
            "raw_size_wh": (raw_w, raw_h),
            "K_processed": K_proc,
        },
        "pair_csr_patchid": pair_csr_patchid,
    }

    os.makedirs(os.path.dirname(OUT_PT), exist_ok=True)
    torch.save(save_obj, OUT_PT)
    print("\n[完成] 已保存索引文件：", OUT_PT)

if __name__ == "__main__":
    main()
