import numpy as np
import open3d as o3d
from scipy.io import loadmat
from scipy.spatial import cKDTree

# ============================================================
# 【全局配置区】——只需要改这里
# ============================================================

# 你的预测点云（scanX.ply）
PRED_PLY = "outputs/token_attention/scene1_DTU/add_epipolar_band/3D_objects/baseline_aligned.ply"

# DTU 官方 GT 点云
GT_PLY = "datasets/scene1_DTU/stl006_total.ply"

# DTU 官方可见性 Mask
OBS_MASK_MAT = "datasets/scene1_DTU/ObsMask6_10.mat"

# DTU 官方常用截断距离（毫米）
MAX_DIST_MM = 20.0

# 体素下采样大小（毫米，0 表示不下采样）
VOXEL_SIZE_MM = 1.0

# 是否随机限制点数（用于加速，0 表示不限制）
LIMIT_PRED_POINTS = 0    # 例如 500000
LIMIT_GT_POINTS   = 0    # 例如 500000

# 随机种子
RANDOM_SEED = 0


# -------------------------
# 读取 PLY 点云
# -------------------------
def read_ply_points(ply_path):
    """
    读取 ply 点云文件
    返回形状为 (N, 3) 的 numpy 数组
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points).astype(np.float64)

    # 去除 NaN / Inf 点
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


# -------------------------
# 体素下采样（加速）
# -------------------------
def voxel_downsample(points, voxel_size):
    """
    使用 Open3D 的 voxel downsample
    voxel_size 单位与点云一致（DTU 为毫米）
    """
    if voxel_size <= 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points)


# -------------------------
# 读取 ObsMask 文件
# -------------------------
def load_obsmask(mat_path):
    """
    读取 DTU 官方 ObsMask*.mat 文件
    返回：
      obs_mask : (X,Y,Z) bool 体素可见性
      bb       : (2,3) bounding box
      res      : 体素分辨率（毫米）
    """
    data = loadmat(mat_path)

    # 兼容不同 key 命名
    for k in ["ObsMask", "obsMask", "mask"]:
        if k in data:
            obs = data[k]
            break
    else:
        raise KeyError("ObsMask 文件中未找到 ObsMask 变量")

    for k in ["BB", "bb", "BBox"]:
        if k in data:
            bb = data[k]
            break
    else:
        bb = None

    for k in ["Res", "res"]:
        if k in data:
            res = float(data[k])
            break
    else:
        res = None

    obs = obs.astype(bool)

    # BB 统一 reshape 成 (2,3)
    if bb is not None:
        bb = np.array(bb).reshape(2, 3)

    return obs, bb, res


# -------------------------
# 使用 ObsMask 过滤 GT 点
# -------------------------
def filter_gt_by_obsmask(gt_points, obs, bb, res):
    """
    将 GT 点映射到 ObsMask 体素坐标
    只保留可见区域内的点
    """
    if bb is None or res is None:
        return gt_points

    bb_min = bb[0]

    # 计算体素索引
    idx = np.floor((gt_points - bb_min) / res).astype(np.int32)

    nx, ny, nz = obs.shape
    valid = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )

    idx = idx[valid]
    keep = obs[idx[:, 0], idx[:, 1], idx[:, 2]]

    return gt_points[valid][keep]


# -------------------------
# 计算 Accuracy / Completeness
# -------------------------
def compute_dtu_metrics(pred_points, gt_points, max_dist):
    """
    DTU 官方 Table-4 风格指标（Python 近似版）

    Accuracy      : pred → gt 最近邻距离均值
    Completeness  : gt → pred 最近邻距离均值
    Overall       : 二者平均
    """
    tree_gt = cKDTree(gt_points)
    d_pred2gt, _ = tree_gt.query(pred_points, k=1)

    tree_pred = cKDTree(pred_points)
    d_gt2pred, _ = tree_pred.query(gt_points, k=1)

    d_pred2gt = d_pred2gt[d_pred2gt < max_dist]
    d_gt2pred = d_gt2pred[d_gt2pred < max_dist]

    acc = d_pred2gt.mean() if len(d_pred2gt) > 0 else max_dist
    comp = d_gt2pred.mean() if len(d_gt2pred) > 0 else max_dist

    return acc, comp, 0.5 * (acc + comp)


# ============================================================
# 主流程
# ============================================================
def main():
    np.random.seed(RANDOM_SEED)

    # 读取点云
    pred_pts = read_ply_points(PRED_PLY)
    gt_pts = read_ply_points(GT_PLY)

    print(f"预测点数: {len(pred_pts)}")
    print(f"GT 点数: {len(gt_pts)}")

    # 随机裁剪点数（可选）
    if LIMIT_PRED_POINTS > 0 and len(pred_pts) > LIMIT_PRED_POINTS:
        idx = np.random.choice(len(pred_pts), LIMIT_PRED_POINTS, replace=False)
        pred_pts = pred_pts[idx]

    if LIMIT_GT_POINTS > 0 and len(gt_pts) > LIMIT_GT_POINTS:
        idx = np.random.choice(len(gt_pts), LIMIT_GT_POINTS, replace=False)
        gt_pts = gt_pts[idx]

    # 读取并应用 ObsMask
    obs, bb, res = load_obsmask(OBS_MASK_MAT)
    gt_pts = filter_gt_by_obsmask(gt_pts, obs, bb, res)
    print(f"ObsMask 过滤后 GT 点数: {len(gt_pts)}")

    # 体素下采样
    pred_pts = voxel_downsample(pred_pts, VOXEL_SIZE_MM)
    gt_pts = voxel_downsample(gt_pts, VOXEL_SIZE_MM)

    # 计算指标
    acc, comp, overall = compute_dtu_metrics(pred_pts, gt_pts, MAX_DIST_MM)

    print("\n========== DTU 3D Reconstruction 评测 ==========")
    print(f"Accuracy     : {acc:.4f} mm")
    print(f"Completeness : {comp:.4f} mm")
    print(f"Overall      : {overall:.4f} mm")
    print("================================================")


if __name__ == "__main__":
    main()