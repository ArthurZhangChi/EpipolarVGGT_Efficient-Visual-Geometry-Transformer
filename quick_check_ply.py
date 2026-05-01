import open3d as o3d
import numpy as np
import copy

# =========================
# 全局配置（只改这里）
# =========================
# PRED_PLY = r"outputs/token_attention/scene3_7Scenes/baseline/pred_aligned.ply"
PRED_PLY = r"server/outputs/7Scenes/office/pred_aligned.ply"
# GT_PLY   = r"outputs/token_attention/scene3_7Scenes/baseline/gt.ply"
GT_PLY   = r"server/outputs/7Scenes/office/gt.ply"

# 分位数裁剪范围（只用于“点很多”的情况）
P_LO = 1.0
P_HI = 99.0

# 体素下采样（单位：meters！）
# 7Scenes 室内建议 0.01~0.03（= 1~3cm）
VOXEL_M = 0.0      # 0=不下采样；比如 0.02 表示 2cm

# 小点云时不要裁剪（避免 N=10 时裁到 N=5）
CROP_ONLY_IF_N_GT = 5000

# 是否将 Pred 平移到 GT 的中心（仅用于可视化）
VIS_ALIGN_TRANSLATE = True

# 是否缩放（一般别开，除非你确认尺度不一致）
VIS_ALIGN_SCALE = False


# =========================
# 工具函数
# =========================
def robust_bbox_points(pts: np.ndarray, p_lo=1.0, p_hi=99.0):
    lo = np.percentile(pts, p_lo, axis=0)
    hi = np.percentile(pts, p_hi, axis=0)
    return lo, hi

def crop_by_bbox(pcd: o3d.geometry.PointCloud, lo, hi):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=lo, max_bound=hi)
    return pcd.crop(bbox)

def stats(name, pcd):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        print(f"{name}: EMPTY\n")
        return None, None, None
    mn, mx = pts.min(0), pts.max(0)
    center_mean = pts.mean(0)
    center_med = np.median(pts, axis=0)
    extent = mx - mn
    print(f"{name}: N={len(pts)}")
    print(f"  min={mn}")
    print(f"  max={mx}")
    print(f"  center(mean)={center_mean}")
    print(f"  center(median)={center_med}")
    print(f"  extent={extent}\n")
    return center_mean, center_med, extent

def downsample(pcd, voxel_m):
    if voxel_m is None or voxel_m <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_m)


# =========================
# 主流程
# =========================
pred = o3d.io.read_point_cloud(PRED_PLY)
gt   = o3d.io.read_point_cloud(GT_PLY)

# 1) 下采样（只影响显示）
pred = downsample(pred, VOXEL_M)
gt   = downsample(gt, VOXEL_M)

print("=== 原始统计（下采样后） ===")
stats("PRED", pred)
stats("GT", gt)

pred_pts = np.asarray(pred.points)
gt_pts   = np.asarray(gt.points)

# 2) 裁剪：用 “GT 的鲁棒 bbox” 统一裁剪 pred/gt（保证看的是同一空间）
pred_vis = pred
gt_vis = gt

if gt_pts.shape[0] >= CROP_ONLY_IF_N_GT and pred_pts.shape[0] >= CROP_ONLY_IF_N_GT:
    gt_lo, gt_hi = robust_bbox_points(gt_pts, P_LO, P_HI)
    pred_vis = crop_by_bbox(pred_vis, gt_lo, gt_hi)
    gt_vis   = crop_by_bbox(gt_vis,   gt_lo, gt_hi)

    print("=== 统一 bbox 裁剪后统计 ===")
    stats("PRED(crop_by_GT_bbox)", pred_vis)
    stats("GT(crop_by_GT_bbox)", gt_vis)
else:
    print(f"[Skip Crop] points too few (pred={pred_pts.shape[0]}, gt={gt_pts.shape[0]})")

# 3) 仅为可视化：平移到同中心
c_pred_mean, c_pred_med, e_pred = stats("PRED(final)", pred_vis)
c_gt_mean,   c_gt_med,   e_gt   = stats("GT(final)", gt_vis)

if VIS_ALIGN_TRANSLATE and (c_pred_med is not None) and (c_gt_med is not None):
    t = (c_gt_med - c_pred_med)
    pred_vis = pred_vis.translate(t, relative=True)

# 4) 可视化尺度缩放（一般别开）
if VIS_ALIGN_SCALE and (e_pred is not None) and (e_gt is not None):
    scale = np.max(e_gt / (e_pred + 1e-9))
    pred_vis = pred_vis.scale(float(scale), center=c_gt_med)

# 5) 上色并显示
pred_vis.paint_uniform_color([1, 0, 0])  # 红
gt_vis.paint_uniform_color([0, 1, 0])    # 绿

# 单独看 Pred

pred_only = copy.deepcopy(pred_vis)
gt_only   = copy.deepcopy(gt_vis)

pred_only.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pred_only], window_name="PRED ONLY")

# 单独看 GT
gt_only.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([gt_only], window_name="GT ONLY")

# 一起看（对比）
o3d.visualization.draw_geometries(
    [pred_only, gt_only],
    window_name="PRED vs GT"
)
