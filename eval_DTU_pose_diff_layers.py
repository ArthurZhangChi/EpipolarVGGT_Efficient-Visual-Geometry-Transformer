import os
import math
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 一、【用户可配置区域】——只需要改这里（配置放最前面）
# ============================================================

# 随机种子（保证科研可复现）
SEED = 0

# 输入：你生成的 pose_enc_by_layer.pt
POSE_PT = r"outputs/compute_epipolar_line/scene1_DTU/baseline/pose_per_layer/pose_enc_by_layer.pt"

# 输出目录
OUT_DIR = r"outputs/compute_epipolar_line/scene1_DTU/baseline/pose_per_layer"

# 参考层：默认用最大的 layer id（通常就是最后一层，比如 23）
REF_LAYER = "last"   # 或者直接写数字，比如 23

# 是否分析全部 view-pairs（S<=10 时建议 True）
USE_ALL_PAIRS = True

# 如果不想用全部 pairs，可随机采样 pairs 数
MAX_PAIRS = 200   # USE_ALL_PAIRS=False 时生效

# 平移方向角误差里，过滤掉太小的基线（避免除零/噪声）
MIN_BASELINE_NORM = 1e-3

# 画图与保存
SAVE_CSV = True
SAVE_PLOTS = True

# ============================================================
# 二、固定随机性（科研必备）
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


# ============================================================
# 三、数学工具：pose_enc -> (R,t)；相对位姿；误差
# ============================================================

def quat_to_rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """四元数(x,y,z,w) -> 旋转矩阵"""
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


def decode_pose_enc(pose_enc: torch.Tensor):
    """
    pose_enc: [B,S,9]
    返回:
      t: [B,S,3]
      R: [B,S,3,3]
      fov: [B,S,2]
    注意：你之前已经验证 quat 顺序是 xyzw，这里直接按 xyzw 解释
    """
    t = pose_enc[..., 0:3]
    q = pose_enc[..., 3:7]  # (x,y,z,w)
    fov = pose_enc[..., 7:9]
    R = quat_to_rotmat_xyzw(q)
    return t, R, fov


def relative_pose_world2cam(Ri, ti, Rj, tj):
    """
    输入：Ri,ti / Rj,tj 都是 world->cam
    输出：从 i 相机坐标到 j 相机坐标的相对位姿 (R_ji, t_ji)
      X_cj = R_ji X_ci + t_ji
    公式：
      R_ji = Rj * Ri^T
      t_ji = tj - R_ji * ti
    """
    R_ji = Rj @ Ri.transpose(-1, -2)
    t_ji = tj - (R_ji @ ti.unsqueeze(-1)).squeeze(-1)
    return R_ji, t_ji


def rot_geodesic_deg(Ra, Rb):
    """旋转误差：geodesic angle in degrees"""
    R = Ra.transpose(-1, -2) @ Rb
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)
    return ang * (180.0 / math.pi)


def angle_between_deg(a, b, eps=1e-8):
    """向量夹角（度）"""
    na = torch.norm(a, dim=-1)
    nb = torch.norm(b, dim=-1)
    mask = (na > eps) & (nb > eps)
    if mask.sum() == 0:
        return None, mask
    aa = a[mask] / na[mask].unsqueeze(-1)
    bb = b[mask] / nb[mask].unsqueeze(-1)
    cos = (aa * bb).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)
    return ang, mask


# ============================================================
# 四、主分析：每层 vs 参考层（最后层）
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = torch.load(POSE_PT, map_location="cpu")
    pose_dict = data["pose_enc_by_layer"]  # layer -> [B,S,9]
    layer_ids = sorted(list(pose_dict.keys()))
    num_layers = data.get("num_backbone_layers", None)
    print("[INFO] layers:", layer_ids)
    if num_layers is not None:
        print("[INFO] num_backbone_layers:", num_layers)

    # 选择参考层
    ref_layer = max(layer_ids) if REF_LAYER == "last" else int(REF_LAYER)
    assert ref_layer in pose_dict, f"REF_LAYER={ref_layer} not in pose_enc_by_layer keys"
    print("[INFO] ref_layer:", ref_layer)

    # 解码参考层
    t_ref, R_ref, _ = decode_pose_enc(pose_dict[ref_layer].float())  # [B,S,*]
    B, S, _ = t_ref.shape
    assert B == 1, "当前脚本默认 B=1（你的场景评测一般是这样）"

    # 构建 view pairs
    all_pairs = [(i, j) for i in range(S) for j in range(S) if i != j]
    if USE_ALL_PAIRS:
        pairs = all_pairs
    else:
        pairs = random.sample(all_pairs, min(MAX_PAIRS, len(all_pairs)))
    print(f"[INFO] num_views={S}, num_pairs_used={len(pairs)}")

    rows = []

    # 预先算参考层的所有相对位姿（减少重复）
    rel_ref = {}
    for (i, j) in pairs:
        Rji_ref, tji_ref = relative_pose_world2cam(R_ref[0, i], t_ref[0, i], R_ref[0, j], t_ref[0, j])
        rel_ref[(i, j)] = (Rji_ref, tji_ref)

    for L in layer_ids:
        pose_enc = pose_dict[L].float()
        tL, RL, _ = decode_pose_enc(pose_enc)

        # 统计：每个 view 的 |t| 与旋转角（仅作辅助观察）
        t_norm = torch.norm(tL[0], dim=-1)  # [S]
        # rotation angle from R: acos((tr-1)/2)
        trace = RL[0, :, 0, 0] + RL[0, :, 1, 1] + RL[0, :, 2, 2]
        rot_ang = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * (180.0 / math.pi)  # [S]

        # 相对位姿误差：对每个 pair
        rot_err_list = []
        tdir_err_list = []
        tscale_log_list = []

        for (i, j) in pairs:
            Rji_L, tji_L = relative_pose_world2cam(RL[0, i], tL[0, i], RL[0, j], tL[0, j])
            Rji_ref, tji_ref = rel_ref[(i, j)]

            # 旋转误差
            re = rot_geodesic_deg(Rji_ref, Rji_L).item()
            rot_err_list.append(re)

            # 平移方向误差（忽略尺度）
            ang, mask = angle_between_deg(tji_ref.unsqueeze(0), tji_L.unsqueeze(0), eps=MIN_BASELINE_NORM)
            if ang is not None:
                tdir_err_list.append(float(ang.item()))

            # 平移尺度误差（用 log 比值，避免尺度爆炸）
            nr = torch.norm(tji_ref).item()
            nl = torch.norm(tji_L).item()
            if nr > MIN_BASELINE_NORM and nl > MIN_BASELINE_NORM:
                tscale_log_list.append(abs(math.log(nl / nr)))

        def stat(x):
            if len(x) == 0:
                return dict(mean=np.nan, median=np.nan, p90=np.nan)
            x = np.array(x, dtype=np.float64)
            return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)))

        rot_stat = stat(rot_err_list)
        tdir_stat = stat(tdir_err_list)
        tscl_stat = stat(tscale_log_list)

        # 一个简单综合分数（越小越好）：你也可以后续自己调权重
        # 旋转(度) + 平移方向(度) + 10*log尺度误差
        score = rot_stat["median"] + tdir_stat["median"] + 10.0 * tscl_stat["median"]

        rows.append({
            "layer": L,
            "ref_layer": ref_layer,

            "rot_err_mean_deg": rot_stat["mean"],
            "rot_err_median_deg": rot_stat["median"],
            "rot_err_p90_deg": rot_stat["p90"],

            "tdir_err_mean_deg": tdir_stat["mean"],
            "tdir_err_median_deg": tdir_stat["median"],
            "tdir_err_p90_deg": tdir_stat["p90"],

            "tscale_log_mean": tscl_stat["mean"],
            "tscale_log_median": tscl_stat["median"],
            "tscale_log_p90": tscl_stat["p90"],

            "score_small_is_better": score,

            "t_norm_mean": float(t_norm.mean().item()),
            "t_norm_median": float(t_norm.median().item()),
            "rot_view_mean_deg": float(rot_ang.mean().item()),
            "rot_view_median_deg": float(rot_ang.median().item()),
        })

        print(f"[L={L:02d}] score={score:.4f} | rot_med={rot_stat['median']:.3f} "
              f"| tdir_med={tdir_stat['median']:.3f} | tscl_med={tscl_stat['median']:.3f}")

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)

    # 保存 CSV
    if SAVE_CSV:
        csv_path = os.path.join(OUT_DIR, "pose_per_layer_quality_vs_last.csv")
        df.to_csv(csv_path, index=False)
        print("[SAVE]", csv_path)

    # 找最优层（score 最小）
    best_row = df.loc[df["score_small_is_better"].idxmin()]
    print("\n[BEST] layer =", int(best_row["layer"]),
          "| score =", float(best_row["score_small_is_better"]))

    # 画图
    if SAVE_PLOTS:
        # 1) rot median
        plt.figure()
        plt.plot(df["layer"], df["rot_err_median_deg"])
        plt.xlabel("layer")
        plt.ylabel("median rot error vs last (deg)")
        plt.title("Rotation error per layer (median)")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, "rot_err_median_vs_last.png"), dpi=200)
        plt.close()

        # 2) t direction median
        plt.figure()
        plt.plot(df["layer"], df["tdir_err_median_deg"])
        plt.xlabel("layer")
        plt.ylabel("median t-direction error vs last (deg)")
        plt.title("Translation direction error per layer (median)")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, "tdir_err_median_vs_last.png"), dpi=200)
        plt.close()

        # 3) scale log median
        plt.figure()
        plt.plot(df["layer"], df["tscale_log_median"])
        plt.xlabel("layer")
        plt.ylabel("median |log(scale)| vs last")
        plt.title("Translation scale error per layer (median)")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, "tscale_log_median_vs_last.png"), dpi=200)
        plt.close()

        # 4) combined score
        plt.figure()
        plt.plot(df["layer"], df["score_small_is_better"])
        plt.xlabel("layer")
        plt.ylabel("score (smaller is better)")
        plt.title("Combined score per layer")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, "score_vs_last.png"), dpi=200)
        plt.close()

        print("[SAVE] plots ->", OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
