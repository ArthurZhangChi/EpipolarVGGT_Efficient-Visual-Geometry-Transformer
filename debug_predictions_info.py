"""
=============================================================
VGGT Prediction 字段 + 多View位姿约定诊断脚本
功能：
1) 打印 prediction 各字段基本信息
2) 解码 pose_enc = absT + quat + FoV
3) 计算每个 view 的 |t| 与旋转角
4) 对每个 view 做 world->cam / cam->world 两种约定自检
=============================================================
"""

import torch
import numpy as np
import random
import math

# ============================================================
# ======================= 全局配置区 =========================
# ============================================================

SEED = 0

SCENE = "inspect_sub_dataset"

PRED_PATH = "outputs/token_attention/scene1_DTU/predictions.pt"

SELF_CHECK_SAMPLES = 3000

PRINT_SAMPLE_NUM = 6

# ============================================================


# ==================== 基础工具 ====================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_tensor_info(name, x):
    print(f"\n[{name}] shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"  min={x.min().item():.4g} max={x.max().item():.4g} mean={x.mean().item():.4g}")
    flat = x.flatten()
    sample = flat[:min(PRINT_SAMPLE_NUM, flat.numel())].tolist()
    print(f"  sample={sample}")


# ==================== 四元数转旋转矩阵 ====================

def quat_to_rotmat(q):
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


def decode_pose(pose_enc):
    t = pose_enc[..., 0:3]
    quat_raw = pose_enc[..., 3:7]
    fov = pose_enc[..., 7:9]

    # 自动判断四元数顺序
    R1 = quat_to_rotmat(quat_raw)
    err1 = (R1.transpose(-1, -2) @ R1 - torch.eye(3)).abs().mean()

    quat2 = torch.stack([
        quat_raw[..., 1],
        quat_raw[..., 2],
        quat_raw[..., 3],
        quat_raw[..., 0],
    ], dim=-1)

    R2 = quat_to_rotmat(quat2)
    err2 = (R2.transpose(-1, -2) @ R2 - torch.eye(3)).abs().mean()

    if err1 <= err2:
        print("四元数顺序: xyzw")
        R = R1
    else:
        print("四元数顺序: wxyz")
        R = R2

    return t, R, fov


def rotation_angle_from_R(R):
    """
    从旋转矩阵计算旋转角（弧度）
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


def self_check(world_points, depth, R, t, view_id):
    """
    对单个 view 做两种位姿解释的自检
    """

    B, S, H, W, _ = world_points.shape
    device = world_points.device

    N = min(SELF_CHECK_SAMPLES, H * W)
    idx = torch.randint(0, H * W, (N,), device=device)
    ys = idx // W
    xs = idx % W

    Xw = world_points[0, view_id, ys, xs, :]
    Rv = R[0, view_id]
    tv = t[0, view_id]

    # A: world->cam
    Xc_A = (Rv @ Xw.T).T + tv
    zpos_A = (Xc_A[:, 2] > 0).float().mean().item()

    # B: cam->world
    Xc_B = (Rv.transpose(0, 1) @ (Xw - tv).T).T
    zpos_B = (Xc_B[:, 2] > 0).float().mean().item()

    if depth is not None:
        d = depth[0, view_id, ys, xs, 0]
        corr_A = torch.corrcoef(torch.stack([d, Xc_A[:, 2]]))[0, 1].item()
        corr_B = torch.corrcoef(torch.stack([d, Xc_B[:, 2]]))[0, 1].item()
    else:
        corr_A = corr_B = float("nan")

    return zpos_A, zpos_B, corr_A, corr_B


# ==================== 主流程 ====================

def main():

    set_seed(SEED)

    print("===================================================")
    print("调试场景:", SCENE)
    print("预测文件:", PRED_PATH)
    print("===================================================")

    pred = torch.load(PRED_PATH, map_location="cpu")
    print("Prediction keys:", list(pred.keys()))

    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            print_tensor_info(k, v)
        elif isinstance(v, list):
            print(f"\n[{k}] list len={len(v)}")
            if len(v) > 0:
                print_tensor_info(f"{k}[0]", v[0])
                print_tensor_info(f"{k}[-1]", v[-1])

    pose_enc = pred["pose_enc"].float()
    t, R, fov = decode_pose(pose_enc)

    print("\n================ 每个view的位姿统计 ================")

    depth = pred["depth"].float() if "depth" in pred else None
    world_points = pred["world_points"].float()

    B, S, _ = t.shape

    for s in range(S):

        t_norm = torch.norm(t[0, s]).item()
        rot_angle = rotation_angle_from_R(R[0, s]).item()

        zA, zB, corrA, corrB = self_check(
            world_points, depth, R, t, s
        )

        print(f"\nView {s}:")
        print(f"  |t| = {t_norm:.4f}")
        print(f"  rot_angle(rad) = {rot_angle:.4f}")
        print(f"  zpos_ratio world->cam = {zA:.4f}")
        print(f"  zpos_ratio cam->world = {zB:.4f}")
        print(f"  corr(depth,Z) world->cam = {corrA:.4f}")
        print(f"  corr(depth,Z) cam->world = {corrB:.4f}")

    print("\n================ 结束 ================")


if __name__ == "__main__":
    main()
