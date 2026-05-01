import os
import numpy as np
import torch
import cv2
import csv

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.rotation import mat_to_quat
from vggt.utils.geometry import closed_form_inverse_se3

# ============================================================
# 【全局配置区】——只需要改这里
# ============================================================

# 你的模型输出 predictions.pt（里面要有 pose_enc）
# PRED_PT = r"outputs/pipeline_epipolar_sparse_attention/scene1_DTU/baseline/predictions.pt"
PRED_PT = r"outputs/pipeline_epipolar_sparse_attention/scene1_DTU/scene1_DTU_staged_sparse/predictions.pt"

# DTU 子集目录（里面有 pos_001.txt / rect_xxx.png 等）
POS_DIR = r"datasets/scene1_DTU"

# 你这个场景实际用到的 pos 序号（要与输入图片顺序一一对应）
POS_IDS = [1, 3, 5, 7, 9, 11, 13]

# 你 forward 时图像尺寸（H, W），用于 pose_enc 解码
IMAGE_HW = (392, 518)

# AUC 阈值（与你截图表格一致：30/15/5；你也可以多算 3）
AUC_THRESHOLDS = [30, 15, 5, 3, 1]

# 平移方向误差是否处理 180° 二义性（litevggt: True）
TRANSLATION_AMBIGUITY = True

# ---- CSV logging ----
# EXP_NAME = "baseline"   #  csv名
EXP_NAME = "bw1"   #  csv名
CSV_PATH = r"outputs/pipeline_epipolar_sparse_attention/scene1_DTU/bandwidth/layer_22_to_23_pose_eval_summary.csv"
APPEND_CSV = True  # True=追加一行；False=覆盖重写（一般用 True）

# 调试
PRINT_GT_DEBUG = True
PRINT_PRED_DEBUG = True


# ============================================================
# liteVGGT 同款：pairs + relative pose error
# ============================================================

def build_pair_index(N, B=1):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    # liteVGGT: quaternion based
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    rel_rangle_deg = err_q * 180 / np.pi
    return rel_rangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, ambiguity=True):
    rel_tangle = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle * 180.0 / np.pi
    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    return rel_tangle_deg


def calculate_auc_np(r_error_deg, t_error_deg, max_threshold=30):
    # liteVGGT: e = max(r,t), histogram in [0..T)
    error_matrix = np.concatenate((r_error_deg[:, None], t_error_deg[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    if num_pairs == 0:
        return 0.0
    normalized_histogram = histogram.astype(float) / num_pairs
    return float(np.mean(np.cumsum(normalized_histogram)))


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    # NOTE: assumes w2c input (same as liteVGGT comment)
    pair_i1, pair_i2 = build_pair_index(num_frames)
    pair_i1 = pair_i1.to(pred_se3.device)
    pair_i2 = pair_i2.to(pred_se3.device)

    relative_pose_gt = gt_se3[pair_i1].bmm(closed_form_inverse_se3(gt_se3[pair_i2]))
    relative_pose_pred = pred_se3[pair_i1].bmm(closed_form_inverse_se3(pred_se3[pair_i2]))

    r_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    t_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3],
                              ambiguity=TRANSLATION_AMBIGUITY)
    return r_deg, t_deg


# ============================================================
# GT: pos_XXX.txt -> P(3x4) -> w2c(4x4)
# ============================================================

def load_pos_as_P_3x4(pos_path: str) -> np.ndarray:
    with open(pos_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    vals = []
    for ln in lines:
        parts = ln.replace(",", " ").split()
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                pass
    vals = np.array(vals, dtype=np.float64)
    if vals.size != 12:
        raise ValueError(f"无法解析 {pos_path}：期望 12 个数(3x4 P)，实际 {vals.size}")
    return vals.reshape(3, 4)


def decompose_P_to_w2c_opencv(P: np.ndarray):
    # OpenCV decomposition
    K, R, C_h = cv2.decomposeProjectionMatrix(P)[:3]
    if abs(K[2, 2]) > 1e-12:
        K = K / K[2, 2]

    C = (C_h[:3] / C_h[3]).reshape(3, 1)   # camera center in world
    t = (-R @ C).reshape(3,)

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c, K

def append_row_to_csv(csv_path: str, row: dict, header: list, append: bool = True):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)
    write_header = (not file_exists) or (not append)

    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 主流程
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load predictions.pt and decode predicted w2c
    pred = torch.load(PRED_PT, map_location=device)
    pose_enc = pred["pose_enc"]  # (1,S,9)
    assert pose_enc.ndim == 3 and pose_enc.shape[0] == 1, "假设 batch=1"

    with torch.no_grad():
        # 兼容不同版本参数名（image_size_hw / image_size）
        try:
            extri_pred, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=IMAGE_HW)
        except TypeError:
            extri_pred, _ = pose_encoding_to_extri_intri(pose_enc, image_size=IMAGE_HW)

        extri_pred = extri_pred[0].to(dtype=torch.float64)  # (S,3,4)

    S = extri_pred.shape[0]
    w2c_pred = torch.eye(4, device=device, dtype=torch.float64).unsqueeze(0).repeat(S, 1, 1)
    w2c_pred[:, :3, :4] = extri_pred

    if PRINT_PRED_DEBUG:
        det0 = torch.det(w2c_pred[0, :3, :3]).item()
        print(f"[Pred] S={S}, det(R0)={det0:.6f}, mean|t|={w2c_pred[:, :3, 3].norm(dim=1).mean().item():.6f}")

    # 2) Load GT w2c from pos_XXX.txt
    w2c_gt_list, K_list = [], []
    for pid in POS_IDS:
        pos_path = os.path.join(POS_DIR, f"pos_{pid:03d}.txt")
        P = load_pos_as_P_3x4(pos_path)
        w2c_np, K_np = decompose_P_to_w2c_opencv(P)
        w2c_gt_list.append(w2c_np)
        K_list.append(K_np)

    assert len(w2c_gt_list) == S, f"GT 数量({len(w2c_gt_list)}) 与预测帧数({S})不一致"
    w2c_gt = torch.from_numpy(np.stack(w2c_gt_list, axis=0)).to(device=device, dtype=torch.float64)

    if PRINT_GT_DEBUG:
        det0 = torch.det(w2c_gt[0, :3, :3]).item()
        print(f"[GT]  S={S}, det(R0)={det0:.6f}, mean|t|={w2c_gt[:, :3, 3].norm(dim=1).mean().item():.6f}")
        print("[GT] K(first) =\n", K_list[0])

    # 3) Relative pose error (liteVGGT)
    with torch.no_grad():
        r_deg, t_deg = se3_to_relative_pose_error(w2c_pred, w2c_gt, num_frames=S)

    r_np = r_deg.detach().cpu().numpy()
    t_np = t_deg.detach().cpu().numpy()
    e_np = np.maximum(r_np, t_np)

    # 4) AUCs
    aucs = {}
    for thr in AUC_THRESHOLDS:
        aucs[thr] = calculate_auc_np(r_np, t_np, max_threshold=int(thr))

    # 5) Print
    print("\n================ Pose Estimation (DTU, liteVGGT-style) ================")
    for thr in AUC_THRESHOLDS:
        print(f"AUC@{thr:>2d} = {aucs[thr]:.4f}")

    print("-----------------------------------------------------------------------")
    print(f"r_mean = {r_np.mean():.4f} deg | t_mean = {t_np.mean():.4f} deg | e_mean = {e_np.mean():.4f} deg")
    print("=======================================================================\n")

    # 6) ✅ Save CSV row (exact fields you requested)
    row = {
        "exp_name": EXP_NAME,
        "auc_30": float(aucs[30]),
        "auc_15": float(aucs[15]),
        "auc_5":  float(aucs[5]),
        "auc_3":  float(aucs[3]),
        "auc_1":  float(aucs[1]),
        "r_mean": float(r_np.mean()),
        "t_mean": float(t_np.mean()),
        "e_mean": float(e_np.mean()),
    }
    header = ["exp_name", "auc_30", "auc_15", "auc_5", "auc_3", "auc_1", "r_mean", "t_mean", "e_mean"]
    append_row_to_csv(CSV_PATH, row, header, append=APPEND_CSV)
    print(f"[CSV] saved -> {CSV_PATH}")


if __name__ == "__main__":
    main()
