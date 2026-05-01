# -*- coding: utf-8 -*-
"""
eval_epipolar_quality_per_layer_use_GT_K.py

用途：
- 在“proc(模型预处理后)坐标系”下评估每一层 pose_enc 产生的极线质量。
- 关键消融：用 GT 的 K（从 P_proc = K[R|t] 分解得到）替代预测 FoV 构造的 K，
  从而“只评 R,t 的影响”。

对比两个 CSV：
1) 你已有：用 K_pred(FoV) 的评测结果
2) 本脚本：用 K_gt(from P_proc) 的评测结果
如果(2)明显更好 => 主要问题在 FoV/K；否则主要问题在 R,t。

输入：
- DTU 子集目录：包含 rect_XXX(_max).* 以及 pos_XXX*
- 每层 pose_enc：pose_enc_by_layer.pt（你已生成）

输出：
- epipolar_quality_per_layer_aligned_proc_line_sampling_use_GT_K.csv

说明：
- 采样 GT 对应点：直接在 GT 极线上采样（稳定 4000/4000）。
- 误差单位：proc 尺度像素（518×392），不是 1600×1200。
"""

import os
import re
import glob
import math
import random
import numpy as np
import torch
import pandas as pd
from PIL import Image

# ============================================================
# 0) ======= 你只需要改这里（配置放最前面）=======
# ============================================================

# 随机种子（保证科研可复现）
SEED = 0

# DTU 子集目录（包含 rect_XXX(_max).* 与 pos_XXX*）
DATA_DIR = r"datasets/scene1_DTU"

# 每层 pose_enc（建议结构：{"pose_enc_by_layer": {layer:int -> tensor[1,S,9]}}）
POSE_PER_LAYER_PT = r"outputs/compute_epipolar_line/scene1_DTU/baseline/pose_per_layer/pose_enc_by_layer.pt"

# 输出目录
OUT_DIR = r"outputs/compute_epipolar_line/scene1_DTU/baseline/pose_per_layer/eval_use_GT_K"

# view 配置
NUM_VIEWS = 7
VIEW_ID_TO_POS_ID = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13}

# rect 文件名模式
RECT_PATTERN_1 = "rect_{:03d}_max.*"
RECT_PATTERN_2 = "rect_{:03d}.*"

# ===== 评测 pairs =====
USE_ALL_PAIRS = True
MAX_PAIRS = 200  # USE_ALL_PAIRS=False 时生效

# ===== GT 对应点采样（在 GT 极线上直接采）=====
SAMPLES_PER_PAIR = 4000
BORDER_MARGIN_PX = 1.0
MAX_LINE_SAMPLE_TRIES = 200000

# ====== 对齐：复刻 load_and_preprocess_images(mode="crop") ======
TARGET_SIZE = 518
PATCH_DIV = 14

# ====== 指标统计 ======
COVERAGE_Q = 0.95  # w@95

# 只评测部分层（None=全层）
EVAL_LAYER_IDS = None
# 例如：EVAL_LAYER_IDS = [0,4,8,12,16,18,19,20,21,22,23]


# ============================================================
# 1) 固定随机性（科研必备）
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# ============================================================
# 2) IO：找 rect / pos、读 P(3x4)
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

def load_projection_matrix_3x4(pos_path: str) -> np.ndarray:
    with open(pos_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().strip()
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)]
    if len(nums) != 12:
        raise ValueError(f"[错误] {pos_path} 解析到 {len(nums)} 个数（应为 12）")
    return np.array(nums, dtype=np.float64).reshape(3, 4)

# ============================================================
# 3) 复刻 preprocess：构建 H_raw->proc（crop 模式）
# ============================================================

def compute_new_hw_crop(raw_w: int, raw_h: int, target_size=TARGET_SIZE, div=PATCH_DIV):
    """
    完全复刻 load_and_preprocess_images(mode='crop') 的尺寸逻辑：
      new_w = 518
      new_h = round(raw_h * (new_w/raw_w) / 14) * 14
      if new_h > 518: center crop height 到 518（只裁 y）
    """
    new_w = target_size
    new_h = round(raw_h * (new_w / raw_w) / div) * div

    if new_h > target_size:
        proc_h = target_size
        start_y = (new_h - target_size) // 2
    else:
        proc_h = new_h
        start_y = 0

    proc_w = new_w
    return new_w, new_h, proc_w, proc_h, start_y

def build_H_raw_to_proc_crop(raw_w: int, raw_h: int, target_size=TARGET_SIZE, div=PATCH_DIV):
    """
    生成 3x3 齐次矩阵 H，使得：x_proc ~ H * x_raw
    """
    new_w, new_h, proc_w, proc_h, start_y = compute_new_hw_crop(raw_w, raw_h, target_size, div)

    sx = new_w / float(raw_w)
    sy = new_h / float(raw_h)

    # 1) resize
    S = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
    ], dtype=np.float64)

    # 2) center crop on y: v_proc = v_resize - start_y
    C = np.array([
        [1, 0, 0],
        [0, 1, -float(start_y)],
        [0, 0, 1],
    ], dtype=np.float64)

    H = C @ S
    meta = {
        "raw_w": raw_w, "raw_h": raw_h,
        "new_w": new_w, "new_h": new_h,
        "proc_w": proc_w, "proc_h": proc_h,
        "start_y": start_y,
        "sx": sx, "sy": sy,
    }
    return H, meta

def transform_P_raw_to_proc(P_raw_3x4: np.ndarray, H_raw_to_proc_3x3: np.ndarray) -> np.ndarray:
    # P_proc = H * P_raw
    return H_raw_to_proc_3x3 @ P_raw_3x4

# ============================================================
# 4) 多视几何：由投影矩阵计算 F（projective 稳定做法）
# ============================================================

def camera_center_from_P(P: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1, :]
    if abs(C[-1]) > 1e-12:
        C = C / C[-1]
    return C  # (4,)

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
# 5) 生成 GT 对应点：在 GT 极线上直接采样（稳定）
# ============================================================

def line_intersections_with_image(a, b, c, W, H, margin=0.0):
    pts = []

    x_min = margin
    y_min = margin
    x_max = (W - 1) - margin
    y_max = (H - 1) - margin

    if abs(b) > 1e-12:
        for x in (x_min, x_max):
            y = -(a * x + c) / b
            if y_min <= y <= y_max:
                pts.append((float(x), float(y)))

    if abs(a) > 1e-12:
        for y in (y_min, y_max):
            x = -(b * y + c) / a
            if x_min <= x <= x_max:
                pts.append((float(x), float(y)))

    uniq = []
    for p in pts:
        if all((abs(p[0]-q[0]) > 1e-6) or (abs(p[1]-q[1]) > 1e-6) for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return []
    if len(uniq) == 2:
        return uniq

    best = None
    best_d = -1.0
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            dx = uniq[i][0] - uniq[j][0]
            dy = uniq[i][1] - uniq[j][1]
            d = dx*dx + dy*dy
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])
    return [best[0], best[1]]

def sample_point_on_gt_epiline(Fgt, xi, Wj, Hj, margin=0.0):
    l = Fgt @ xi
    a, b, c = float(l[0]), float(l[1]), float(l[2])

    seg = line_intersections_with_image(a, b, c, Wj, Hj, margin=margin)
    if len(seg) != 2:
        return None

    (x1, y1), (x2, y2) = seg
    t = random.random()
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return np.array([x, y, 1.0], dtype=np.float64)

def synthesize_corr_on_epiline(Fgt, Wi, Hi, Wj, Hj, n_samples,
                               margin=BORDER_MARGIN_PX,
                               max_tries=MAX_LINE_SAMPLE_TRIES):
    xis, xjs = [], []
    tries = 0

    while len(xis) < n_samples and tries < max_tries:
        tries += 1

        ui = random.uniform(margin, (Wi - 1) - margin)
        vi = random.uniform(margin, (Hi - 1) - margin)
        xi = np.array([ui, vi, 1.0], dtype=np.float64)

        xj = sample_point_on_gt_epiline(Fgt, xi, Wj, Hj, margin=margin)
        if xj is None:
            continue

        xis.append(xi)
        xjs.append(xj)

    return xis, xjs, tries

# ============================================================
# 6) pose_enc -> R,t（只用 R,t；K 用 GT 分解）
# ============================================================

def quat_to_rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = torch.stack([
        1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R

def decode_pose_enc_BS9_only_Rt(pose_enc_BS9: torch.Tensor):
    pose = pose_enc_BS9.float()
    B, S, D = pose.shape
    assert B == 1 and D == 9
    t = pose[0, :, 0:3]          # [S,3]
    q = pose[0, :, 3:7]          # [S,4] xyzw
    R = quat_to_rotmat_xyzw(q)   # [S,3,3]
    return R, t

def relative_pose_world2cam(Ri, ti, Rj, tj):
    # 假设 R,t 都是 world->cam
    R_ji = Rj @ Ri.T
    t_ji = tj - R_ji @ ti
    return R_ji, t_ji

def essential_from_rt(R, t):
    return skew(t) @ R

def fundamental_from_EK(E, Ki, Kj):
    Ki_inv = np.linalg.inv(Ki)
    Kj_invT = np.linalg.inv(Kj).T
    F = Kj_invT @ E @ Ki_inv
    n = np.linalg.norm(F)
    if n > 1e-12:
        F = F / n
    return F

# ============================================================
# 7) 从 P_proc 分解出 GT 的 K（RQ 分解）
# ============================================================

def rq_decomposition_3x3(A: np.ndarray):
    """
    标准 RQ 分解（Hartley-Zisserman 写法）
    返回 R(上三角), Q(正交)，使 A = R @ Q
    """
    A = np.asarray(A, dtype=np.float64)
    J = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype=np.float64)

    # QR on reversed matrix
    A1 = J @ A.T @ J
    Q1, R1 = np.linalg.qr(A1)

    R = J @ R1.T @ J
    Q = J @ Q1.T @ J
    return R, Q


def decompose_P_to_KRt(P: np.ndarray):
    """
    输入 P(3x4)，输出 K(3x3), R(3x3), t(3,)
    约定：P = K [R|t]，并规范化 K：
      - K[2,2]=1
      - K 对角线为正
    """
    P = np.asarray(P, dtype=np.float64)
    M = P[:, :3]

    K, R = rq_decomposition_3x3(M)

    # 让 K 对角线为正
    T = np.diag(np.sign(np.diag(K) + 1e-12))
    K = K @ T
    R = T @ R

    # 归一化
    if abs(K[2, 2]) > 1e-12:
        K = K / K[2, 2]

    # t = K^{-1} p4
    t = np.linalg.inv(K) @ P[:, 3]
    return K, R, t.reshape(3,)

# ============================================================
# 8) 指标：点到极线距离（像素，proc 坐标系）
# ============================================================

def point_line_distance_px(F, xi, xj):
    l = F @ xi
    a, b, c = float(l[0]), float(l[1]), float(l[2])
    num = abs(float(xj.T @ (F @ xi)))
    den = math.sqrt(a*a + b*b) + 1e-12
    return num / den

# ============================================================
# 9) 主流程
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 读 raw 图尺寸（用 view0 的 rect 作为代表）
    r0 = find_rect_image(VIEW_ID_TO_POS_ID[0])
    W_raw, H_raw = Image.open(r0).size  # PIL: (W,H)

    H_raw_to_proc, meta = build_H_raw_to_proc_crop(W_raw, H_raw, TARGET_SIZE, PATCH_DIV)
    proc_w, proc_h = meta["proc_w"], meta["proc_h"]

    print("[DBG] raw_size =", (W_raw, H_raw))
    print("[DBG] proc_size =", (proc_w, proc_h))
    print("[DBG] resize_new_h =", meta["new_h"], "start_y =", meta["start_y"])
    print("[DBG] sx,sy =", meta["sx"], meta["sy"])

    # ---- 读取每个 view 的 P_raw，并映射到 P_proc；并从 P_proc 分解 GT K
    P_proc = {}
    K_gt = {}
    for v in range(NUM_VIEWS):
        pos_id = VIEW_ID_TO_POS_ID[v]
        P_raw = load_projection_matrix_3x4(find_pos_file(pos_id))
        Pp = transform_P_raw_to_proc(P_raw, H_raw_to_proc)
        P_proc[v] = Pp

        K, R, t = decompose_P_to_KRt(Pp)
        K_gt[v] = K

    # ---- pairs
    all_pairs = [(i, j) for i in range(NUM_VIEWS) for j in range(NUM_VIEWS) if i != j]
    pairs = all_pairs if USE_ALL_PAIRS else random.sample(all_pairs, min(MAX_PAIRS, len(all_pairs)))
    print(f"[INFO] num_views={NUM_VIEWS}, pairs_used={len(pairs)}")

    # ---- layers
    pack = torch.load(POSE_PER_LAYER_PT, map_location="cpu")
    if "pose_enc_by_layer" in pack:
        pose_dict = pack["pose_enc_by_layer"]
    else:
        pose_dict = pack

    layer_ids = sorted(list(pose_dict.keys()))
    if EVAL_LAYER_IDS is not None:
        s = set(EVAL_LAYER_IDS)
        layer_ids = [L for L in layer_ids if L in s]
    print("[INFO] layers_eval =", layer_ids)

    # ---- 预生成 GT 对应点：在 GT 极线上采样（稳定 4000/4000）
    corr = {}
    for (i, j) in pairs:
        Fgt = fundamental_from_projections(P_proc[i], P_proc[j])

        xis, xjs, tries = synthesize_corr_on_epiline(
            Fgt=Fgt,
            Wi=proc_w, Hi=proc_h,
            Wj=proc_w, Hj=proc_h,
            n_samples=SAMPLES_PER_PAIR,
            margin=BORDER_MARGIN_PX,
            max_tries=MAX_LINE_SAMPLE_TRIES
        )

        corr[(i, j)] = (xis, xjs, Fgt)
        print(f"[GT Corr] pair({i}->{j}) valid={len(xis)}/{SAMPLES_PER_PAIR} tries={tries}")

    # ---- GT lowerbound sanity（应接近 0）
    gt_dists = []
    for (i, j) in pairs:
        xis, xjs, Fgt = corr[(i, j)]
        for xi, xj in zip(xis, xjs):
            gt_dists.append(point_line_distance_px(Fgt, xi, xj))

    if len(gt_dists) == 0:
        raise RuntimeError("GT correspondences are empty. 请检查 pos/尺寸/投影矩阵是否正确。")

    gt_dists = np.array(gt_dists, dtype=np.float64)
    print(f"\n[GT LowerBound] median={np.median(gt_dists):.6f}px "
          f"p90={np.percentile(gt_dists,90):.6f}px "
          f"w@{int(COVERAGE_Q*100)}={np.quantile(gt_dists,COVERAGE_Q):.6f}px")

    # ---- 每层评测：F_pred 用 GT K；只看预测 R,t
    rows = []
    for L in layer_ids:
        pose_enc = pose_dict[L]
        R_t, t_t = decode_pose_enc_BS9_only_Rt(pose_enc)

        # Avoid Tensor.numpy() because some environments ship PyTorch without NumPy bridge support.
        R_pred = [np.asarray(R_t[v].detach().cpu().tolist(), dtype=np.float64) for v in range(NUM_VIEWS)]
        t_pred = [np.asarray(t_t[v].detach().cpu().tolist(), dtype=np.float64) for v in range(NUM_VIEWS)]

        dists = []
        for (i, j) in pairs:
            xis, xjs, _Fgt = corr[(i, j)]
            if len(xis) == 0:
                continue

            Ri, ti = R_pred[i], t_pred[i]
            Rj, tj = R_pred[j], t_pred[j]
            Rji, tji = relative_pose_world2cam(Ri, ti, Rj, tj)
            E = essential_from_rt(Rji, tji)

            Ki = K_gt[i]
            Kj = K_gt[j]
            F = fundamental_from_EK(E, Ki, Kj)

            for xi, xj in zip(xis, xjs):
                dists.append(point_line_distance_px(F, xi, xj))

        dists = np.array(dists, dtype=np.float64)
        med = float(np.median(dists))
        p90 = float(np.percentile(dists, 90))
        wq = float(np.quantile(dists, COVERAGE_Q))

        rows.append({
            "layer": L,
            "pairs": len(pairs),
            "points": int(len(dists)),
            "dist_median_px": med,
            "dist_p90_px": p90,
            f"w_at_{int(COVERAGE_Q*100)}_px": wq,
            "proc_w": proc_w,
            "proc_h": proc_h,
            "note": "F_pred uses GT K decomposed from P_proc; only R,t are from model pose_enc"
        })

        print(f"[L={L:02d}] median={med:.4f}px p90={p90:.4f}px w@{int(COVERAGE_Q*100)}={wq:.4f}px")

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    csv_path = os.path.join(OUT_DIR, "epipolar_quality_per_layer_aligned_proc_line_sampling_use_GT_K.csv")
    df.to_csv(csv_path, index=False)
    print("\n[SAVE]", csv_path)
    print("\nDone.")

if __name__ == "__main__":
    main()
