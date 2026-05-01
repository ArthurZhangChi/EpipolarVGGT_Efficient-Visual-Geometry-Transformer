# -*- coding: utf-8 -*-
"""
eval_epipolar_per_layer_dtu_realistic_vs_gtK.py

目标：
- 真正模拟“正常情景”：只用每层 pose_enc => (R,t, last2) 构造 F_pred，并和 GT 的 F_gt 比。
- 同时输出控制变量：用 GT-K 只看 R,t 的上限。

输出：
- epipolar_eval_compare_K_modes.csv
"""

import os, re, glob, math, random
import numpy as np
import torch
import pandas as pd
from PIL import Image

# ===================== 配置区 =====================
SEED = 0
DATA_DIR = r"datasets/scene1_DTU"
POSE_PER_LAYER_PT = r"outputs/compute_epipolar_line/scene1_DTU/baseline/pose_per_layer/pose_enc_by_layer.pt"
OUT_DIR = r"outputs/compute_epipolar_line/scene1_DTU/baseline/pose_per_layer/epipolar_eval_compare"

NUM_VIEWS = 7
VIEW_ID_TO_POS_ID = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13}
RECT_PATTERN_1 = "rect_{:03d}_max.*"
RECT_PATTERN_2 = "rect_{:03d}.*"

USE_ALL_PAIRS = True
MAX_PAIRS = 200
SAMPLES_PER_PAIR = 4000
BORDER_MARGIN_PX = 1.0
MAX_LINE_SAMPLE_TRIES = 200000

TARGET_SIZE = 518
PATCH_DIV = 14

COVERAGE_Q = 0.95
EVAL_LAYER_IDS = None  # None=all
# ================================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

set_seed(SEED)

# ---------------- IO ----------------
def find_rect_image(pos_id: int) -> str:
    patt1 = os.path.join(DATA_DIR, RECT_PATTERN_1.format(pos_id))
    c1 = sorted(glob.glob(patt1))
    if c1: return c1[0]
    patt2 = os.path.join(DATA_DIR, RECT_PATTERN_2.format(pos_id))
    c2 = sorted(glob.glob(patt2))
    if c2: return c2[0]
    raise FileNotFoundError(f"rect not found for pos_{pos_id:03d}")

def find_pos_file(pos_id: int) -> str:
    patt = os.path.join(DATA_DIR, f"pos_{pos_id:03d}*")
    cands = sorted(glob.glob(patt))
    if not cands:
        raise FileNotFoundError(f"pos file not found for pos_{pos_id:03d}")
    return cands[0]

def load_projection_matrix_3x4(pos_path: str) -> np.ndarray:
    with open(pos_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().strip()
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)]
    if len(nums) != 12:
        raise ValueError(f"{pos_path}: parsed {len(nums)} nums, expect 12")
    return np.array(nums, dtype=np.float64).reshape(3, 4)

# -------- preprocess alignment (crop) --------
def compute_new_hw_crop(raw_w, raw_h, target_size=TARGET_SIZE, div=PATCH_DIV):
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

def build_H_raw_to_proc_crop(raw_w, raw_h, target_size=TARGET_SIZE, div=PATCH_DIV):
    new_w, new_h, proc_w, proc_h, start_y = compute_new_hw_crop(raw_w, raw_h, target_size, div)
    sx = new_w / float(raw_w)
    sy = new_h / float(raw_h)
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0,  0, 1]], dtype=np.float64)
    C = np.array([[1, 0, 0],
                  [0, 1, -float(start_y)],
                  [0, 0, 1]], dtype=np.float64)
    H = C @ S
    meta = dict(raw_w=raw_w, raw_h=raw_h, new_w=new_w, new_h=new_h,
                proc_w=proc_w, proc_h=proc_h, start_y=start_y, sx=sx, sy=sy)
    return H, meta

def transform_P_raw_to_proc(P_raw, H_raw2proc):
    return H_raw2proc @ P_raw

# -------- geometry: F from P (GT) --------
def camera_center_from_P(P):
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1, :]
    if abs(C[-1]) > 1e-12: C = C / C[-1]
    return C

def skew(v):
    v = v.reshape(-1)
    return np.array([[0,    -v[2],  v[1]],
                     [v[2],  0,    -v[0]],
                     [-v[1], v[0],  0]], dtype=np.float64)

def fundamental_from_projections(P_src, P_dst):
    C_src = camera_center_from_P(P_src)
    e_dst = P_dst @ C_src
    if abs(e_dst[2]) > 1e-12: e_dst = e_dst / e_dst[2]
    P_src_pinv = np.linalg.pinv(P_src)
    M = P_dst @ P_src_pinv
    F = skew(e_dst) @ M
    n = np.linalg.norm(F)
    if n > 1e-12: F = F / n
    return F

# -------- RQ decompose P_proc -> K_gt (for control) --------
def rq_decomposition_3x3(A):
    A = np.asarray(A, dtype=np.float64)
    J = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.float64)
    A1 = J @ A.T @ J
    Q1, R1 = np.linalg.qr(A1)
    R = J @ R1.T @ J
    Q = J @ Q1.T @ J
    return R, Q  # R upper-tri, Q orth

def decompose_P_to_KRt(P):
    P = np.asarray(P, dtype=np.float64)
    M = P[:, :3]
    K, R = rq_decomposition_3x3(M)  # M = K @ R

    # normalize signs: make diag(K) positive
    D = np.diag(np.sign(np.diag(K) + 1e-12))
    K = K @ D
    R = D @ R

    # force det(R)=+1
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1
        R[2, :] *= -1

    # normalize K[2,2]=1
    if abs(K[2,2]) > 1e-12:
        K = K / K[2,2]

    t = np.linalg.inv(K) @ P[:, 3]
    return K, R, t.reshape(3,)

# -------- pose_enc decode --------
def quat_to_rotmat_xyzw(q):
    q = q / (np.linalg.norm(q) + 1e-12)
    x,y,z,w = q
    xx,yy,zz = x*x,y*y,z*z
    xy,xz,yz = x*y,x*z,y*z
    wx,wy,wz = w*x,w*y,w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
    ], dtype=np.float64)

def quat_to_rotmat_xyzw_torch(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R

def decode_pose_enc_BS9(pose_enc_BS9):
    pose = pose_enc_BS9.detach().float().cpu()
    assert pose.shape[0] == 1 and pose.shape[2] == 9
    t_t = pose[0, :, 0:3]
    q_t = pose[0, :, 3:7]  # xyzw
    last2_t = pose[0, :, 7:9]
    R_t = quat_to_rotmat_xyzw_torch(q_t)

    # Avoid Tensor.numpy() because some environments ship PyTorch without NumPy bridge support.
    t = np.asarray(t_t.tolist(), dtype=np.float64)
    last2 = np.asarray(last2_t.tolist(), dtype=np.float64)
    R = np.asarray(R_t.tolist(), dtype=np.float64)
    return R, t, last2

# -------- build K_pred from last2 (two hypotheses) --------
def K_from_fov(fovx, fovy, W, H):
    # fov in radians
    fx = (W/2.0) / math.tan(max(1e-6, float(fovx))/2.0)
    fy = (H/2.0) / math.tan(max(1e-6, float(fovy))/2.0)
    cx = W/2.0; cy = H/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

def K_from_fl_norm(fl_x, fl_y, W, H, mode="scale_wh"):
    """
    假设 last2 是 focal length 的某种归一化输出。
    两个常见解释：
      - mode="scale_wh": fx=fl_x*W, fy=fl_y*H   (fl 表示相对尺度)
      - mode="pixel":    fx=fl_x,   fy=fl_y     (fl 已经是像素单位)
    """
    if mode == "scale_wh":
        fx = float(fl_x) * W
        fy = float(fl_y) * H
    else:
        fx = float(fl_x)
        fy = float(fl_y)
    cx = W/2.0; cy = H/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

# -------- F from (R,t,K) --------
def relative_pose_world2cam(Ri, ti, Rj, tj):
    R_ji = Rj @ Ri.T
    t_ji = tj - R_ji @ ti
    return R_ji, t_ji

def essential_from_rt(R, t):
    return skew(t) @ R

def fundamental_from_EK(E, Ki, Kj):
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    n = np.linalg.norm(F)
    if n > 1e-12: F = F / n
    return F

# -------- sample correspondences on GT epiline --------
def line_intersections_with_image(a,b,c,W,H,margin=0.0):
    pts=[]
    x_min=margin; y_min=margin
    x_max=(W-1)-margin; y_max=(H-1)-margin
    if abs(b)>1e-12:
        for x in (x_min,x_max):
            y=-(a*x+c)/b
            if y_min<=y<=y_max: pts.append((float(x),float(y)))
    if abs(a)>1e-12:
        for y in (y_min,y_max):
            x=-(b*y+c)/a
            if x_min<=x<=x_max: pts.append((float(x),float(y)))
    uniq=[]
    for p in pts:
        if all((abs(p[0]-q[0])>1e-6) or (abs(p[1]-q[1])>1e-6) for q in uniq):
            uniq.append(p)
    if len(uniq)<2: return []
    if len(uniq)==2: return uniq
    best=None; best_d=-1
    for i in range(len(uniq)):
        for j in range(i+1,len(uniq)):
            dx=uniq[i][0]-uniq[j][0]; dy=uniq[i][1]-uniq[j][1]
            d=dx*dx+dy*dy
            if d>best_d: best_d=d; best=(uniq[i],uniq[j])
    return [best[0],best[1]]

def sample_point_on_gt_epiline(Fgt, xi, Wj, Hj, margin=0.0):
    l = Fgt @ xi
    a,b,c = float(l[0]), float(l[1]), float(l[2])
    seg = line_intersections_with_image(a,b,c,Wj,Hj,margin)
    if len(seg)!=2: return None
    (x1,y1),(x2,y2)=seg
    t = random.random()
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return np.array([x,y,1.0], dtype=np.float64)

def synthesize_corr_on_epiline(Fgt, Wi,Hi,Wj,Hj,n_samples,margin,max_tries):
    xis=[]; xjs=[]; tries=0
    while len(xis)<n_samples and tries<max_tries:
        tries += 1
        ui = random.uniform(margin,(Wi-1)-margin)
        vi = random.uniform(margin,(Hi-1)-margin)
        xi = np.array([ui,vi,1.0], dtype=np.float64)
        xj = sample_point_on_gt_epiline(Fgt, xi, Wj, Hj, margin)
        if xj is None: continue
        xis.append(xi); xjs.append(xj)
    return xis, xjs, tries

def point_line_distance_px(F, xi, xj):
    l = F @ xi
    a,b = float(l[0]), float(l[1])
    num = abs(float(xj.T @ (F @ xi)))
    den = math.sqrt(a*a+b*b)+1e-12
    return num/den

# ---------------- main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    r0 = find_rect_image(VIEW_ID_TO_POS_ID[0])
    W_raw, H_raw = Image.open(r0).size
    H_raw2proc, meta = build_H_raw_to_proc_crop(W_raw, H_raw, TARGET_SIZE, PATCH_DIV)
    proc_w, proc_h = meta["proc_w"], meta["proc_h"]
    print("[DBG] raw_size =", (W_raw, H_raw))
    print("[DBG] proc_size =", (proc_w, proc_h))

    # P_raw -> P_proc and GT-K
    P_proc = {}
    K_gt = {}
    for v in range(NUM_VIEWS):
        pos_id = VIEW_ID_TO_POS_ID[v]
        P_raw = load_projection_matrix_3x4(find_pos_file(pos_id))
        Pp = transform_P_raw_to_proc(P_raw, H_raw2proc)
        P_proc[v] = Pp
        K, R, t = decompose_P_to_KRt(Pp)
        K_gt[v] = K

    all_pairs = [(i,j) for i in range(NUM_VIEWS) for j in range(NUM_VIEWS) if i!=j]
    pairs = all_pairs if USE_ALL_PAIRS else random.sample(all_pairs, min(MAX_PAIRS, len(all_pairs)))
    print(f"[INFO] pairs_used={len(pairs)}")

    pack = torch.load(POSE_PER_LAYER_PT, map_location="cpu")
    pose_dict = pack["pose_enc_by_layer"] if "pose_enc_by_layer" in pack else pack
    layer_ids = sorted(list(pose_dict.keys()))
    if EVAL_LAYER_IDS is not None:
        s=set(EVAL_LAYER_IDS); layer_ids=[L for L in layer_ids if L in s]
    print("[INFO] layers_eval =", layer_ids)

    # GT correspondences
    corr={}
    for (i,j) in pairs:
        Fgt = fundamental_from_projections(P_proc[i], P_proc[j])
        xis,xjs,tries = synthesize_corr_on_epiline(
            Fgt, proc_w,proc_h, proc_w,proc_h, SAMPLES_PER_PAIR, BORDER_MARGIN_PX, MAX_LINE_SAMPLE_TRIES
        )
        corr[(i,j)] = (xis,xjs,Fgt)
        print(f"[GT Corr] {i}->{j} {len(xis)}/{SAMPLES_PER_PAIR}")

    # lowerbound
    gt_d=[]
    for (i,j) in pairs:
        xis,xjs,Fgt = corr[(i,j)]
        gt_d += [point_line_distance_px(Fgt, xi, xj) for xi,xj in zip(xis,xjs)]
    gt_d=np.array(gt_d)
    print(f"[GT LowerBound] median={np.median(gt_d):.6f}px w@95={np.quantile(gt_d,0.95):.6f}px")

    def summarize(arr):
        arr=np.array(arr, dtype=np.float64)
        return float(np.median(arr)), float(np.percentile(arr,90)), float(np.quantile(arr,COVERAGE_Q))

    rows=[]
    for L in layer_ids:
        R_pred, t_pred, last2 = decode_pose_enc_BS9(pose_dict[L])

        # pre-build K_pred for each view in two hypotheses
        K_pred_fov = [K_from_fov(last2[v,0], last2[v,1], proc_w, proc_h) for v in range(NUM_VIEWS)]
        K_pred_fl_scale = [K_from_fl_norm(last2[v,0], last2[v,1], proc_w, proc_h, mode="scale_wh") for v in range(NUM_VIEWS)]
        K_pred_fl_pixel = [K_from_fl_norm(last2[v,0], last2[v,1], proc_w, proc_h, mode="pixel") for v in range(NUM_VIEWS)]

        d_fov=[]; d_fl_s=[]; d_fl_p=[]; d_gtK=[]
        for (i,j) in pairs:
            xis,xjs,_ = corr[(i,j)]
            if len(xis)==0: continue

            Rji, tji = relative_pose_world2cam(R_pred[i], t_pred[i], R_pred[j], t_pred[j])
            E = essential_from_rt(Rji, tji)

            F_fov = fundamental_from_EK(E, K_pred_fov[i], K_pred_fov[j])
            F_fl_s = fundamental_from_EK(E, K_pred_fl_scale[i], K_pred_fl_scale[j])
            F_fl_p = fundamental_from_EK(E, K_pred_fl_pixel[i], K_pred_fl_pixel[j])
            F_gtK = fundamental_from_EK(E, K_gt[i], K_gt[j])

            for xi,xj in zip(xis,xjs):
                d_fov.append(point_line_distance_px(F_fov, xi, xj))
                d_fl_s.append(point_line_distance_px(F_fl_s, xi, xj))
                d_fl_p.append(point_line_distance_px(F_fl_p, xi, xj))
                d_gtK.append(point_line_distance_px(F_gtK, xi, xj))

        med,p90,wq = summarize(d_fov)
        med2,p902,wq2 = summarize(d_fl_s)
        med3,p903,wq3 = summarize(d_fl_p)
        med4,p904,wq4 = summarize(d_gtK)

        rows.append(dict(
            layer=L, points=len(d_gtK),
            predK_as_fov_median=med, predK_as_fov_p90=p90, predK_as_fov_w95=wq,
            predK_as_fl_scale_median=med2, predK_as_fl_scale_p90=p902, predK_as_fl_scale_w95=wq2,
            predK_as_fl_pixel_median=med3, predK_as_fl_pixel_p90=p903, predK_as_fl_pixel_w95=wq3,
            gtK_only_median=med4, gtK_only_p90=p904, gtK_only_w95=wq4,
            proc_w=proc_w, proc_h=proc_h
        ))
        print(f"[L={L:02d}] fov_w95={wq:.2f} | flS_w95={wq2:.2f} | flP_w95={wq3:.2f} | gtK_w95={wq4:.2f}")

    df=pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    out_csv=os.path.join(OUT_DIR, "epipolar_eval_compare_K_modes.csv")
    df.to_csv(out_csv, index=False)
    print("[SAVE]", out_csv)

if __name__ == "__main__":
    main()
