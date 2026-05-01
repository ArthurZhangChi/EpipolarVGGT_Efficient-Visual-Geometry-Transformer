import os
import math
import random
import numpy as np
import torch

# ============================================================
# 【全局配置区】——只需要改这里
# ============================================================

# 1) 模型输出 predictions.pt
PRED_PT = r"outputs/token_attention/scene1_DTU/add_epipolar_band/baseline/predictions.pt"

# 2) 你要对齐的 GT 点云（DTU 的 stl 点云）
GT_PLY = r"datasets/scene1_DTU/stl006_total.ply"
# 如果你用的是 scan6 的官方 ply，也可以填那个

# 3) 输出 ply
OUT_RAW_PLY  = r"outputs/token_attention/scene1_DTU/add_epipolar_band/3D_objects/baseline.ply"
OUT_SIM3_PLY = r"outputs/token_attention/scene1_DTU/add_epipolar_band/3D_objects/baseline_aligned.ply"

# 4) 图像分辨率（要与 predictions.pt 中 depth 的分辨率一致）
IMG_H = 392
IMG_W = 518

# 5) 采样与过滤（强烈建议先用较干净的点再对齐）
STRIDE = 2                  # 每隔多少像素采样（2/3/4）
KEEP_TOP_CONF = 0.20        # 每帧保留 depth_conf 的 top 比例（0.05~0.3 之间调）
MIN_DEPTH_MM = 1.0          # 太近的深度直接丢掉（单位 mm）

# 6) 深度尺度（VGGT 的 depth head 输出通常是“相对深度”，这里要转成 mm）
# 如果你已经确定 depth 的单位是“米”，那就设为 1000
DEPTH_SCALE_MM = 1000.0

# 7) Sim3 对齐设置（稳健）
ENABLE_SIM3_ALIGN = True
SIM3_SAMPLE_N = 50000       # 用多少点做 Umeyama（太大 SVD 容易不稳）
SIM3_TRIM_Q = 0.995         # 对齐前裁剪极端值（比如 99.5% 分位）
SIM3_MAX_TRIES = 5          # SVD 不收敛时重试次数
WITH_SCALE = True           # 是否估计尺度（RouteB 一般要 True）

# 8) 可选：ICP 微调（建议先关，先看 Sim3 是否正常）
ENABLE_ICP_REFINE = False
ICP_THRESHOLD_MM = 10.0

# 9) 可选：输出前简单清理
ENABLE_VOXEL_DOWN = True
VOXEL_SIZE_MM = 2.0

ENABLE_OUTLIER_REMOVAL = True
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

# 10) 是否打印调试信息
VERBOSE = True

# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def write_ply_xyz(path: str, xyz: np.ndarray):
    """写 ASCII PLY（只有 x y z）"""
    ensure_dir(path)
    xyz = np.asarray(xyz, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=-1)

def quantile_clip(xyz: np.ndarray, q=0.995):
    """按每个轴的分位裁剪极端点，提升 Umeyama 稳定性"""
    lo = np.quantile(xyz, 1.0 - q, axis=0)
    hi = np.quantile(xyz, q, axis=0)
    m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return xyz[m]

def random_sample(xyz: np.ndarray, n: int, seed=0):
    if xyz.shape[0] <= n:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=n, replace=False)
    return xyz[idx]

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale=True):
    """
    Umeyama 对齐：src -> dst
    输入：
      src, dst: (N,3)
    输出：
      s, R, t，使得： dst ≈ s * (R @ src.T).T + t
    """
    assert src.shape == dst.shape
    N = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 协方差
    Sigma = (dst_c.T @ src_c) / N  # (3,3)

    # SVD
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / N
        s = (D * np.diag(S)).sum() / (var_src + 1e-12)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return float(s), R.astype(np.float64), t.astype(np.float64)

def apply_sim3(xyz: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    """xyz: (N,3)"""
    return (s * (R @ xyz.T)).T + t.reshape(1, 3)

def load_gt_ply_points(gt_ply: str):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(gt_ply)
    pts = np.asarray(pcd.points, dtype=np.float64)
    return pts

def clean_points_open3d(xyz: np.ndarray):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))

    if ENABLE_VOXEL_DOWN and VOXEL_SIZE_MM > 0:
        pcd = pcd.voxel_down_sample(VOXEL_SIZE_MM)

    if ENABLE_OUTLIER_REMOVAL:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO
        )

    return np.asarray(pcd.points, dtype=np.float64)

def icp_refine(pred_xyz: np.ndarray, gt_xyz: np.ndarray, init_T: np.ndarray):
    """用 open3d ICP 在 Sim3 对齐后做一次微调（只返回 4x4）"""
    import open3d as o3d

    pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_xyz.astype(np.float64)))
    pcd_gt   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz.astype(np.float64)))

    threshold = float(ICP_THRESHOLD_MM)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_pred,
        pcd_gt,
        threshold,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg.transformation

# ============================================================
# Route B：depth + (w2c, K) -> world 点云
# ============================================================

def pose_enc_to_w2c_and_K(pose_enc: torch.Tensor, image_hw):
    """
    用你 repo 里的 pose_encoding_to_extri_intri 把 pose_enc 还原成 w2c 与 K
    pose_enc: [1,S,9]
    返回：
      w2c: (S,4,4) torch.float64
      K:   (S,3,3) torch.float64
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extri_3x4, intri_3x3 = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)
    extri_3x4 = extri_3x4[0].to(dtype=torch.float64)  # [S,3,4]
    intri_3x3 = intri_3x3[0].to(dtype=torch.float64)  # [S,3,3]

    S = extri_3x4.shape[0]
    w2c = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(S, 1, 1)
    w2c[:, :3, :4] = extri_3x4
    return w2c, intri_3x3

def unproject_depth_to_world(depth_mm: np.ndarray, conf: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """
    单帧：把深度图反投影成 world 点
    输入：
      depth_mm: (H,W) 深度（mm）
      conf:     (H,W) 置信度
      w2c:      (4,4) 世界到相机
      K:        (3,3)
    输出：
      world_xyz: (N,3)
    """
    H, W = depth_mm.shape
    us = np.arange(0, W, STRIDE)
    vs = np.arange(0, H, STRIDE)
    uu, vv = np.meshgrid(us, vs)

    z = depth_mm[::STRIDE, ::STRIDE].reshape(-1)
    c = conf[::STRIDE, ::STRIDE].reshape(-1)

    # 有效性过滤
    m = np.isfinite(z) & np.isfinite(c) & (z > MIN_DEPTH_MM)
    if m.sum() == 0:
        return None

    uu = uu.reshape(-1)[m]
    vv = vv.reshape(-1)[m]
    z  = z[m]
    c  = c[m]

    # top-conf 保留
    thr = np.quantile(c, 1.0 - KEEP_TOP_CONF)
    keep = c >= thr
    uu, vv, z = uu[keep], vv[keep], z[keep]

    # 反投影到相机系：X_cam = z * K^{-1} [u,v,1]^T
    Kinv = np.linalg.inv(K)
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=0)  # (3,N)
    rays = (Kinv @ pix)                                 # (3,N)
    X_cam = rays * z.reshape(1, -1)                      # (3,N)

    # cam -> world： Twc = inv(w2c)
    Twc = np.linalg.inv(w2c)
    Rwc = Twc[:3, :3]
    twc = Twc[:3, 3:4]
    X_world = (Rwc @ X_cam + twc).T                      # (N,3)
    return X_world.astype(np.float64)

def build_pred_pointcloud_routeB(pred_dict: dict):
    """
    从 predictions.pt 构建“自洽”的预测点云（world 坐标）
    """
    # 取 depth / conf
    depth = pred_dict["depth"][0, :, :, :, 0].cpu().numpy().astype(np.float64)       # (S,H,W)
    dconf = pred_dict["depth_conf"][0].cpu().numpy().astype(np.float64)              # (S,H,W)

    S, H, W = depth.shape
    assert (H, W) == (IMG_H, IMG_W), f"depth 分辨率 {H,W} != 配置 {(IMG_H,IMG_W)}"

    # pose_enc -> w2c & K
    pose_enc = pred_dict["pose_enc"]
    assert pose_enc.ndim == 3 and pose_enc.shape[0] == 1, "假设 batch=1"
    w2c_t, K_t = pose_enc_to_w2c_and_K(pose_enc, image_hw=(IMG_H, IMG_W))
    w2c = w2c_t.cpu().numpy()
    K   = K_t.cpu().numpy()

    all_pts = []
    for s in range(S):
        depth_mm = depth[s] * float(DEPTH_SCALE_MM)
        Xw = unproject_depth_to_world(depth_mm, dconf[s], w2c[s], K[s])
        if Xw is None:
            if VERBOSE:
                print(f"[frame {s}] 0 点（全部无效）")
            continue
        all_pts.append(Xw)
        if VERBOSE:
            print(f"[frame {s}] 保留 {Xw.shape[0]} 点")

    if len(all_pts) == 0:
        raise RuntimeError("没有生成任何有效点云，请检查 depth/conf/阈值。")

    xyz = np.concatenate(all_pts, axis=0)
    xyz = xyz[finite_mask(xyz)]
    return xyz

# ============================================================
# 主流程：生成 pred 点云 -> (可选清理) -> Sim3 对齐 -> (可选 ICP) -> 输出
# ============================================================

def main():
    # 1) 读取预测
    pred = torch.load(PRED_PT, map_location="cpu")

    # 2) Route B 生成点云（自洽世界系）
    pred_xyz = build_pred_pointcloud_routeB(pred)

    if VERBOSE:
        mn, mx = pred_xyz.min(0), pred_xyz.max(0)
        print(f"[Pred raw] N={pred_xyz.shape[0]} min={mn} max={mx} extent={mx-mn}")

    # 3) 可选：先做轻度清理（建议开）
    pred_xyz_clean = clean_points_open3d(pred_xyz) if (ENABLE_VOXEL_DOWN or ENABLE_OUTLIER_REMOVAL) else pred_xyz

    if VERBOSE:
        mn, mx = pred_xyz_clean.min(0), pred_xyz_clean.max(0)
        print(f"[Pred clean] N={pred_xyz_clean.shape[0]} extent={mx-mn}")

    # 输出未对齐的 ply（方便你自己看）
    write_ply_xyz(OUT_RAW_PLY, pred_xyz_clean.astype(np.float32))
    print(f"[Saved] raw ply -> {OUT_RAW_PLY}")

    if not ENABLE_SIM3_ALIGN:
        print("[Info] ENABLE_SIM3_ALIGN=False，结束。")
        return

    # 4) 读取 GT 点云
    gt_xyz = load_gt_ply_points(GT_PLY)
    gt_xyz = gt_xyz[finite_mask(gt_xyz)]
    if VERBOSE:
        mn, mx = gt_xyz.min(0), gt_xyz.max(0)
        print(f"[GT] N={gt_xyz.shape[0]} extent={mx-mn}")

    # 5) 为了 Umeyama 稳定：采样 + 裁剪极端值
    #    注意：Umeyama 需要一一对应点，这里我们用“随机对应”只是为了得到全局尺度/平移/旋转的粗对齐，
    #    实际上更稳的做法是用 ICP，但我们这里先做稳健 Sim3，然后可选 ICP refine。
    pred_s = pred_xyz_clean.copy()
    gt_s = gt_xyz.copy()

    # 裁剪极端值（避免少数飞点让 SVD 崩）
    pred_s = quantile_clip(pred_s, q=SIM3_TRIM_Q)
    gt_s   = quantile_clip(gt_s, q=SIM3_TRIM_Q)

    # 采样到相同数量
    n = min(SIM3_SAMPLE_N, pred_s.shape[0], gt_s.shape[0])
    pred_s = random_sample(pred_s, n, seed=0)
    gt_s   = random_sample(gt_s, n, seed=1)

    # 6) Umeyama 对齐（带重试）
    best = None
    for k_try in range(SIM3_MAX_TRIES):
        try:
            # 为了更稳，可以每次打乱对应关系（随机配对）
            np.random.shuffle(pred_s)
            np.random.shuffle(gt_s)

            s, R, t = umeyama_alignment(pred_s, gt_s, with_scale=WITH_SCALE)

            # 应用到全部 pred 点云
            pred_aligned = apply_sim3(pred_xyz_clean, s, R, t)

            # 简单打分：中心距离 + extent 比例（只用于选更合理的解）
            pred_center = pred_aligned.mean(0)
            gt_center   = gt_xyz.mean(0)
            center_dist = float(np.linalg.norm(pred_center - gt_center))

            pred_ext = pred_aligned.max(0) - pred_aligned.min(0)
            gt_ext   = gt_xyz.max(0) - gt_xyz.min(0)
            ext_ratio = float(np.mean(pred_ext / (gt_ext + 1e-9)))

            score = center_dist + 100.0 * abs(math.log(ext_ratio + 1e-12))  # 越小越好

            if VERBOSE:
                print(f"[Sim3 try {k_try}] s={s:.6f} center_dist={center_dist:.3f} ext_ratio={ext_ratio:.3f} score={score:.3f}")

            if (best is None) or (score < best["score"]):
                best = dict(s=s, R=R, t=t, pred_aligned=pred_aligned, score=score)

        except np.linalg.LinAlgError as e:
            print(f"[Warn] Umeyama SVD 不收敛（try={k_try}）：{e}")
            # 再随机采样一次重试
            pred_s = random_sample(pred_s, n, seed=10 + k_try)
            gt_s   = random_sample(gt_s, n, seed=20 + k_try)
            continue

    if best is None:
        raise RuntimeError("Sim3 对齐失败：多次尝试 SVD 仍不收敛。请检查点云是否大量 NaN/Inf 或极端退化。")

    pred_sim3 = best["pred_aligned"]

    print(f"[Sim3 Best] s={best['s']:.6f}")
    if VERBOSE:
        mn, mx = pred_sim3.min(0), pred_sim3.max(0)
        print(f"[Pred Sim3] extent={mx-mn}, center={pred_sim3.mean(0)}")

    # 7) 可选：ICP 微调（在 Sim3 结果上做 rigid refine）
    if ENABLE_ICP_REFINE:
        init_T = np.eye(4, dtype=np.float64)
        # 注意：ICP 只做刚体，不含尺度；我们已经通过 Sim3 把尺度对齐了
        T_icp = icp_refine(pred_sim3, gt_xyz, init_T)
        pred_sim3_h = np.concatenate([pred_sim3, np.ones((pred_sim3.shape[0], 1))], axis=1)
        pred_sim3 = (T_icp @ pred_sim3_h.T).T[:, :3]
        print("[ICP] refine done.")

    # 8) 输出对齐后的 ply
    write_ply_xyz(OUT_SIM3_PLY, pred_sim3.astype(np.float32))
    print(f"[Saved] sim3 ply -> {OUT_SIM3_PLY}")

    print("\n下一步：请用你的 DTU 评测脚本，评测 OUT_SIM3_PLY（对齐后的点云）")
    print("如果仍然出现 Accuracy/Completeness 全是 20mm：说明评测脚本阈值饱和，需打印 clamp 前距离分布。")

if __name__ == "__main__":
    main()