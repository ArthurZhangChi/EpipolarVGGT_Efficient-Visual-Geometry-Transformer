import os
import re
import glob
import time
import numpy as np
import os.path as osp

import csv
import cv2
import torch
import open3d as o3d
from torch.utils.data._utils.collate import default_collate

# liteVGGT style utils/criterion (you copied into ./eval/)
from eval.utils import accuracy, completion, depthmap_to_absolute_camera_coordinates
from eval.criterion import Regr3D_t_ScaleShiftInv, L21

# your repo modules
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ============================================================
# 一、【用户可配置区域】——只需要改这里（配置放最前面）
# ============================================================

# ---- Paths ----
DATA_DIR   = r"datasets/scene3_7Scenes"
# CKPT_PATH  = r"outputs/token_attention/scene3_7Scenes/baseline/model_state_dict.pt"  # 你保存的 state_dict
CKPT_PATH  = r"outputs/pipeline_epipolar_sparse_attention/scene3_7Scenes/scene3_7Scenes_staged_sparse/model_state_dict.pt"  # 你保存的 state_dict
HF_MODEL   = "facebook/VGGT-1B"  # 用来初始化结构（保证 key 对齐）
OUT_DIR    = r"outputs/pipeline_epipolar_sparse_attention/scene3_7Scenes/"

# ---- CSV logging ----
EXP_NAME = "final_model"
CSV_PATH = osp.join(
    "outputs/pipeline_epipolar_sparse_attention/scene3_7Scenes",
    "final_3D_reconstruction.csv"
)

# ---- Device ----
DEVICE = "cuda:0"  # or "cpu"

# ---- Resolution (match liteVGGT args.size) ----
# size=518 -> (518,392), size=512 -> (512,384), size=224 -> (224,224)
SIZE = 518

# ---- 7Scenes intrinsics (SimpleRecon / liteVGGT raw intrinsics for 640x480) ----
FX, FY, CX, CY = 585.0, 585.0, 320.0, 240.0

# ---- liteVGGT eval behavior toggles ----
FORCE_CROP_224 = False          # mimic liteVGGT: center crop 224x224 before building pcd
USE_PROJ_UMEYAMA = False        # mimic liteVGGT --use_proj: Umeyama Sim3 align pred->gt before ICP
USE_ICP = False                 # ICP refine
ICP_THRESH = 0.10              # meters

# ---- confidence filter (optional) ----
CONF_THRESH = 0.0              # >0 to filter mask & (depth_conf > thresh)

# ---- speed/memory ----
MAX_POINTS = 999999            # cap point count for speed (random sampling)

# ---- saving ----
SAVE_PLY = True

# ---- IMPORTANT: dtype ----
# 默认关 AMP，强制 FP32，避免你之前 LayerNorm dtype 报错（Half/BFloat16）
USE_AMP = False


# ============================================================
# 二、工具：flat triplets
# ============================================================

def parse_frame_id(path: str) -> int:
    name = os.path.basename(path)
    m = re.match(r"frame-(\d+)\.", name)
    if m is None:
        raise ValueError(f"cannot parse frame id from: {name}")
    return int(m.group(1))


def build_triplets_flat(data_dir: str):
    colors = sorted(glob.glob(osp.join(data_dir, "frame-*.color.png")))
    depths = sorted(glob.glob(osp.join(data_dir, "frame-*.depth*.png")))  # depth.png / depth.proj.png
    poses  = sorted(glob.glob(osp.join(data_dir, "frame-*.pose.txt")))

    cmap = {parse_frame_id(p): p for p in colors}
    dmap = {parse_frame_id(p): p for p in depths}
    pmap = {parse_frame_id(p): p for p in poses}

    ids = sorted(set(cmap) & set(dmap) & set(pmap))
    if len(ids) == 0:
        raise RuntimeError(f"No complete triplets found in {data_dir}")
    return [(i, cmap[i], dmap[i], pmap[i]) for i in ids]


# ============================================================
# 三、预处理：resize by width + center crop height，并更新 intrinsics
# （与你当前 pipeline 的 crop 模式对齐）
# ============================================================

def preprocess_rgb_depth_and_K(rgb_bgr, depth_raw, K_raw, out_w, out_h, invalid_u16=65535):
    H0, W0 = rgb_bgr.shape[:2]

    # ensure depth aligns to rgb before resize
    if depth_raw.shape[:2] != (H0, W0):
        depth_raw = cv2.resize(depth_raw, (W0, H0), interpolation=cv2.INTER_NEAREST)

    scale = out_w / float(W0)
    new_w = out_w
    new_h = int(round(H0 * scale))

    rgb_r = cv2.resize(rgb_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dep_r = cv2.resize(depth_raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # pad if too short
    if new_h < out_h:
        pad = out_h - new_h
        top = pad // 2
        rgb_r = cv2.copyMakeBorder(rgb_r, top, pad - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        if dep_r.dtype == np.uint16:
            dep_r = cv2.copyMakeBorder(dep_r, top, pad - top, 0, 0, cv2.BORDER_CONSTANT, value=invalid_u16)
        else:
            dep_r = cv2.copyMakeBorder(dep_r, top, pad - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        new_h = out_h

    crop_top = (new_h - out_h) // 2
    rgb_c = rgb_r[crop_top:crop_top + out_h]
    dep_c = dep_r[crop_top:crop_top + out_h]

    fx, fy, cx, cy = K_raw[0, 0], K_raw[1, 1], K_raw[0, 2], K_raw[1, 2]
    fx2 = fx * scale
    fy2 = fy * scale
    cx2 = cx * scale
    cy2 = cy * scale - float(crop_top)

    K_new = np.array([[fx2, 0.0, cx2],
                      [0.0, fy2, cy2],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
    return rgb_c, dep_c, K_new


def center_crop_224(image, mask, pts):
    H, W = mask.shape[:2]
    cx = W // 2
    cy = H // 2
    l, t = cx - 112, cy - 112
    r, b = cx + 112, cy + 112
    image = image[t:b, l:r]
    mask = mask[t:b, l:r]
    pts = pts[t:b, l:r]
    return image, mask, pts


# ============================================================
# 四、unproject：depth + (w2c, K) -> world pointmap
# ============================================================

def unproject_depth_to_world(depth_m, K, w2c_3x4):
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth_m.astype(np.float32)
    x = (u - cx) * z / (fx + 1e-12)
    y = (v - cy) * z / (fy + 1e-12)

    Xc = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # HxWx4

    Tw2c = np.eye(4, dtype=np.float32)
    Tw2c[:3, :4] = w2c_3x4.astype(np.float32)
    Tc2w = np.linalg.inv(Tw2c).astype(np.float32)

    Xw = (Xc @ Tc2w.T)[..., :3]
    return Xw


# ============================================================
# 五、对齐：Umeyama (Sim3) + ICP
# ============================================================

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    assert src.shape == dst.shape
    N = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    Sigma = (dst_c.T @ src_c) / max(N, 1)
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_c ** 2) / max(N, 1)
        s = np.sum(D * np.diag(S)) / (var_src + 1e-12)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return float(s), R.astype(np.float32), t.astype(np.float32)


def apply_sim3(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    return (s * (R @ pts.T)).T + t.reshape(1, 3)


def random_cap_points(pts, cols, max_n: int):
    if pts.shape[0] <= max_n:
        return pts, cols
    idx = np.random.choice(pts.shape[0], max_n, replace=False)
    pts2 = pts[idx]
    cols2 = cols[idx] if cols is not None else None
    return pts2, cols2


def build_pcd(pts, cols=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if cols is not None:
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd


# ============================================================
# 六、主流程
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if SIZE == 518:
        out_w, out_h = 518, 392
    elif SIZE == 512:
        out_w, out_h = 512, 384
    elif SIZE == 224:
        out_w, out_h = 224, 224
    else:
        raise NotImplementedError(f"Unsupported SIZE={SIZE}")

    device = torch.device(DEVICE)

    # ---- data ----
    triplets = build_triplets_flat(DATA_DIR)
    ids = [t[0] for t in triplets]
    print(f"[Data] frames={len(triplets)} ids={ids}")

    K_raw = np.array([[FX, 0.0, CX],
                      [0.0, FY, CY],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

    views = []
    images_np = []

    for fid, cpath, dpath, ppath in triplets:
        rgb = cv2.imread(cpath, cv2.IMREAD_COLOR)
        dep = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None:
            raise RuntimeError(f"Failed to read {cpath} or {dpath}")

        rgb_c, dep_c, K = preprocess_rgb_depth_and_K(rgb, dep, K_raw, out_w, out_h)

        # depth: uint16 mm -> meters
        dep_c = dep_c.astype(np.float32)
        dep_c[dep_c == 65535] = 0.0
        depth_m = dep_c / 1000.0
        depth_m[depth_m > 10.0] = 0.0
        depth_m[depth_m < 1e-3] = 0.0

        # pose: cam2world
        T_c2w = np.loadtxt(ppath).astype(np.float32)

        # GT points (world) + valid mask (liteVGGT utils)
        Xw_gt, valid_mask = depthmap_to_absolute_camera_coordinates(depth_m, K, T_c2w)

        # image tensor [-1,1]
        rgb_rgb = cv2.cvtColor(rgb_c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.transpose(rgb_rgb, (2, 0, 1))).unsqueeze(0)
        img_t = img_t * 2.0 - 1.0

        views.append(dict(
            img=img_t,
            camera_pose=torch.from_numpy(T_c2w).unsqueeze(0),
            pts3d=torch.from_numpy(Xw_gt).unsqueeze(0),
            valid_mask=torch.from_numpy(valid_mask.astype(np.bool_)).unsqueeze(0),
            label=f"scene3_7Scenes/{fid:06d}",
            instance=cpath,
        ))
        images_np.append(rgb_rgb)

    # mimic liteVGGT: list[dict] then collate if you want
    _ = default_collate(views)  # not strictly needed, but keeps behavior closer

    # imgs_tensor [S,3,H,W] in [0,1]
    imgs_tensor = torch.cat([v["img"] for v in views], dim=0).to(device)      # [-1,1]
    imgs_tensor_01 = (imgs_tensor + 1.0) / 2.0                                # [0,1]

    # ---- model ----
    model = VGGT.from_pretrained(HF_MODEL).to(device).eval()

    if CKPT_PATH and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        if isinstance(ckpt, dict) and len(ckpt) > 0 and all(torch.is_tensor(v) for v in list(ckpt.values())[:10]):
            sd = ckpt
        elif isinstance(ckpt, dict):
            found = None
            for k in ["state_dict", "model", "net", "model_state_dict", "ema"]:
                if k in ckpt and isinstance(ckpt[k], dict):
                    found = ckpt[k]
                    break
            if found is None:
                raise KeyError(f"Cannot find dict-like state_dict in ckpt keys={list(ckpt.keys())}")
            sd = found
        else:
            raise TypeError(f"Unsupported ckpt type: {type(ckpt)}")

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[CKPT] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[CKPT] not found, using HF weights only.")

    # ---- inference ----
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        if USE_AMP and device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                preds_raw = model(imgs_tensor_01)
        else:
            preds_raw = model(imgs_tensor_01.float())

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    print(f"[Infer] {(t1 - t0) * 1000:.2f} ms")

    # ---- decode outputs ----
    depth_pred = preds_raw["depth"].squeeze(0)[..., 0]          # (S,H,W)
    depth_conf = preds_raw["depth_conf"].squeeze(0)             # (S,H,W)
    pose_enc = preds_raw["pose_enc"]                            # (1,S,9)

    extri, intri = pose_encoding_to_extri_intri(pose_enc, imgs_tensor_01.shape[-2:])
    w2c = extri.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (S,3,4)
    Kp  = intri.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (S,3,3)

    depth_pred_np = depth_pred.detach().cpu().numpy().astype(np.float32)

    point_map_world = []
    for s in range(depth_pred_np.shape[0]):
        Xw = unproject_depth_to_world(depth_pred_np[s], Kp[s], w2c[s])  # HxWx3 world (pred coord)

        # pred world -> pred cam1 using w2c of view0
        if s == 0:
            T_w2c0 = np.eye(4, dtype=np.float32)
            T_w2c0[:3, :4] = w2c[0]
            R0 = T_w2c0[:3, :3]
            t0 = T_w2c0[:3, 3]

        X_cam1 = (Xw @ R0.T) + t0  # HxWx3 in pred-cam1
        point_map_world.append(X_cam1)
    point_map_world = np.stack(point_map_world, axis=0)  # (S,H,W,3)

    # ---- wrap preds per view (liteVGGT style) ----
    preds_list = []
    for s in range(point_map_world.shape[0]):
        preds_list.append(dict(
            pts3d_in_other_view=torch.from_numpy(point_map_world[s]).unsqueeze(0).to(device),
            conf=depth_conf[s].unsqueeze(0),          # (1,H,W)
            camera_pose=pose_enc[:, s, :],            # keep for parity
        ))

    # ---- criterion (liteVGGT) ----
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    gts = []
    for v in views:
        gts.append(dict(
            camera_pose=v["camera_pose"].to(device),
            pts3d=v["pts3d"].to(device).float(),
            valid_mask=v["valid_mask"].to(device),
        ))

    gt_pts_list, pred_pts_list, gt_factor, pr_factor, masks_list, monitoring = criterion.get_all_pts3d_t(gts, preds_list)

    # ---- build point cloud (mimic liteVGGT eval) ----
    pts_all = []
    pts_gt_all = []
    cols_all = []

    paired_src = []   # ✅ pred配对点（Umeyama用）
    paired_dst = []   # ✅ gt配对点（Umeyama用）

    for i in range(len(gts)):
        pr = pred_pts_list[i][0].detach().cpu().numpy()  # HxWx3
        gt = gt_pts_list[i][0].detach().cpu().numpy()    # HxWx3
        mask = masks_list[i][0].detach().cpu().numpy().astype(bool)  # HxW
        img = images_np[i]  # HxWx3 RGB [0,1]

        if FORCE_CROP_224:
            img2, mask2, pr2 = center_crop_224(img, mask, pr)
            _,   _,    gt2 = center_crop_224(img, mask, gt)   # 用原始img/mask做同样bbox
            img, mask, pr, gt = img2, mask2, pr2, gt2

        if CONF_THRESH > 0:
            conf = depth_conf[i].detach().cpu().numpy()
            if FORCE_CROP_224:
                H, W = conf.shape
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                conf = conf[t:b, l:r]
            mask = mask & (conf > CONF_THRESH)

        pr_m = pr[mask]
        gt_m = gt[mask]
        col_m = img[mask]

        m1 = np.isfinite(pr_m).all(axis=1)
        m2 = np.isfinite(gt_m).all(axis=1)
        m = m1 & m2
        pr_m = pr_m[m]
        gt_m = gt_m[m]
        col_m = col_m[m]

        pts_all.append(pr_m.reshape(-1, 3))
        pts_gt_all.append(gt_m.reshape(-1, 3))
        cols_all.append(col_m.reshape(-1, 3))
        paired_src.append(pr_m.reshape(-1, 3))  # ✅ 配对点：pred
        paired_dst.append(gt_m.reshape(-1, 3))  # ✅ 配对点：gt

    pts_all = np.concatenate(pts_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    cols_all = np.concatenate(cols_all, axis=0)

    pts_all, cols_all = random_cap_points(pts_all, cols_all, MAX_POINTS)
    pts_gt_all, _ = random_cap_points(pts_gt_all, None, MAX_POINTS)

    # ---- Umeyama (Sim3) ----
    if USE_PROJ_UMEYAMA:
        src = np.concatenate(paired_src, axis=0)   # pred paired
        dst = np.concatenate(paired_dst, axis=0)   # gt paired

        # optional: cap for speed
        n = min(src.shape[0], 200000)
        if n < 2000:
            print("[Umeyama] skipped (too few paired points)")
        else:
            idx = np.random.choice(src.shape[0], n, replace=False)
            s, R, t = umeyama_alignment(src[idx], dst[idx], with_scale=True)

            # apply Sim3 to pred point cloud
            pts_all = apply_sim3(pts_all, s, R, t)
            print(f"[Umeyama] s={s:.6f}")

    # ---- ICP ----
    pcd = build_pcd(pts_all, cols_all)
    pcd_gt = build_pcd(pts_gt_all, None)

    if USE_ICP:
        reg = o3d.pipelines.registration.registration_icp(
            pcd, pcd_gt,
            ICP_THRESH,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd.transform(reg.transformation)
        print(f"[ICP] fitness={reg.fitness:.4f} rmse={reg.inlier_rmse:.4f}")

    # normals + metrics
    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    gt_normal = np.asarray(pcd_gt.normals)
    pr_normal = np.asarray(pcd.normals)

    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pr_normal)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pr_normal)

    nc_mean = 0.5 * (nc1 + nc2)
    nc_med  = 0.5 * (nc1_med + nc2_med)

    overall_mean = 0.5 * (acc + comp)
    overall_med  = 0.5 * (acc_med + comp_med)

    print("\n========== Flat 7Scenes Eval (liteVGGT-style) ==========")
    print(f"Frames used  : {len(views)} | ids={ids}")
    print(f"Acc          : {acc:.6f} m (median {acc_med:.6f})")
    print(f"Comp         : {comp:.6f} m (median {comp_med:.6f})")
    print(f"NC1/NC2      : {nc1:.6f}/{nc2:.6f} (median {nc1_med:.6f}/{nc2_med:.6f})")
    print(f"Overall      : {0.5*(acc+comp):.6f} m (median {0.5*(acc_med+comp_med):.6f})")
    print("=======================================================\n")

    # ---------------- CSV logging ----------------
    csv_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # header（只在第一次写）
        if not csv_exists:
            writer.writerow([
                "exp_name",
                "accuracy_mean", "accuracy_median",
                "completeness_mean", "completeness_median",
                "nc_mean", "nc_median",
                "overall_mean", "overall_median",
            ])

        writer.writerow([
            EXP_NAME,
            acc, acc_med,
            comp, comp_med,
            nc_mean, nc_med,
            overall_mean, overall_med,
        ])

    print(f"[CSV] appended results to {CSV_PATH}")

    if SAVE_PLY:
        o3d.io.write_point_cloud(osp.join(OUT_DIR, "pred_aligned.ply"), pcd)
        o3d.io.write_point_cloud(osp.join(OUT_DIR, "gt.ply"), pcd_gt)
        print("[Saved]", osp.join(OUT_DIR, "pred_aligned.ply"))
        print("[Saved]", osp.join(OUT_DIR, "gt.ply"))


if __name__ == "__main__":
    main()
