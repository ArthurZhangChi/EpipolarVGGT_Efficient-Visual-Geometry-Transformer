# pipeline_epipolar_sparse_attention.py
# ============================================================
# Online staged epipolar sparse attention pipeline
#
# Main idea:
#   - global 0~14 : dense
#   - global 15   : dense, then run camera_head
#   - use pose_15 to build sparse meta for 16,17
#   - use pose_17 to build sparse meta for 18,19
#   - ...
#
# Requirements:
#   - modified attention.py / block.py / aggregator.py / vggt.py
#   - sparse_config.py
#   - epipolar_sparse_builder.py
#   - epipolar_update_function.py
# ============================================================

import os
import json
import time
import random
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from sparse_config import (
    get_default_sparse_cfg,
    summarize_sparse_cfg,
)
from sparse_epipolar_builder import EpipolarSparseBuilder
from epipolar_update_function import EpipolarUpdateFunction


# ============================================================
# 一、【用户可配置区域】
# ============================================================

SEED = 0

# 场景信息
SCENE = "scene3_7Scenes/scene3_7Scenes_staged_sparse"
SCENE_DIR = "datasets/scene3_7Scenes"

# 输出目录
OUT_DIR = "outputs/pipeline_epipolar_sparse_attention"

# 模型
MODEL_NAME = "facebook/VGGT-1B"

# 是否保存完整 predictions
SAVE_PREDICTIONS = True

# 是否保存在线更新中间 pose
SAVE_ONLINE_POSE_UPDATES = False

# 是否保存运行配置
SAVE_CONFIG = True
SAVE_MODEL_STATE_DICT = True

# 是否打印更多 debug
VERBOSE = True


# ============================================================
# 二、固定随机性
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ============================================================
# 三、设备与精度
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


# ============================================================
# 四、工具函数
# ============================================================

def nested_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: nested_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nested_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(nested_to_cpu(v) for v in obj)
    return obj


def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 五、加载配置
# ============================================================

cfg = get_default_sparse_cfg()

if VERBOSE:
    print(summarize_sparse_cfg(cfg))


# ============================================================
# 六、读取场景图像
# ============================================================

image_names = sorted([
    os.path.join(SCENE_DIR, fname)
    for fname in os.listdir(SCENE_DIR)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"[INFO] 从场景目录加载 {len(image_names)} 张图像：")
for name in image_names:
    print("   ", name)

if len(image_names) == 0:
    raise RuntimeError(f"No images found in {SCENE_DIR}")


# ============================================================
# 七、加载并预处理图像
# ============================================================

images = load_and_preprocess_images(image_names).to(device)

# 当前 staged sparse builder 默认按 B=1 使用 pose_enc
# 因此这里推荐输入是 [S,3,H,W]，由 model.forward 内部自动 unsqueeze 成 [1,S,3,H,W]
if images.ndim != 4:
    raise RuntimeError(
        f"Expected load_and_preprocess_images to return [S,3,H,W], got shape={tuple(images.shape)}"
    )

proc_h, proc_w = int(images.shape[-2]), int(images.shape[-1])

print(f"[INFO] processed image size = (H={proc_h}, W={proc_w})")


# ============================================================
# 八、创建输出目录
# ============================================================

out_dir = os.path.join(OUT_DIR, SCENE)
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 九、加载模型
# ============================================================

model = VGGT.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ------------------------------------------------------------
# （关键）
# 开启后半段 sparse 逻辑
# 前半段是否用 fused SDPA / flash，取决于你 attention.py 里的 fused_attn 逻辑
# ------------------------------------------------------------
model.aggregator.enable_partial_sparse = True
model.aggregator.partial_sparse_start_layer = cfg["sparse_apply_start"]
model.aggregator.partial_sparse_end_layer = cfg["num_global_layers"] - 1

if VERBOSE:
    print("[INFO] aggregator.enable_partial_sparse =", model.aggregator.enable_partial_sparse)
    print("[INFO] partial_sparse_start_layer      =", model.aggregator.partial_sparse_start_layer)
    print("[INFO] partial_sparse_end_layer        =", model.aggregator.partial_sparse_end_layer)


# ============================================================
# 十、构造 builder 和 updater
# ============================================================

builder = EpipolarSparseBuilder(
    proc_h=proc_h,
    proc_w=proc_w,
    patch_size=model.aggregator.patch_size,
    patch_start_idx=model.aggregator.patch_start_idx,
    k_mode="fixed_prior",                 # 推荐第一版先用固定先验 K
    fixed_focal_ratio=0.9,
)

updater = EpipolarUpdateFunction(
    cfg=cfg,
    builder=builder,
)

if VERBOSE:
    print("[INFO] EpipolarSparseBuilder initialized.")
    print(f"[INFO] patch_size={model.aggregator.patch_size}, patch_start_idx={model.aggregator.patch_start_idx}")


# ============================================================
# 十一、保存配置快照
# ============================================================

if SAVE_CONFIG:
    cfg_dump = {
        "SEED": SEED,
        "SCENE": SCENE,
        "SCENE_DIR": SCENE_DIR,
        "OUT_DIR": OUT_DIR,
        "MODEL_NAME": MODEL_NAME,
        "SAVE_MODEL_STATE_DICT": SAVE_MODEL_STATE_DICT,
        "device": device,
        "dtype": str(dtype),
        "proc_h": proc_h,
        "proc_w": proc_w,
        "builder": {
            "patch_size": model.aggregator.patch_size,
            "patch_start_idx": model.aggregator.patch_start_idx,
            "k_mode": "fixed_prior",
            "fixed_focal_ratio": 0.9,
        },
        "staged_sparse_cfg": cfg,
    }
    save_json(os.path.join(out_dir, "cfg.json"), cfg_dump)
    print("[保存] cfg.json")

if SAVE_MODEL_STATE_DICT:
    model_state_path = os.path.join(out_dir, "model_state_dict.pt")
    torch.save(state_dict_to_cpu(model.state_dict()), model_state_path)
    print("[SAVE] model_state_dict ->", model_state_path)


# ============================================================
# 十二、前向推理（staged sparse 主流程）
# ============================================================

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

t0 = time.time()

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(
            images,
            staged_sparse=cfg["use_staged_sparse"],
            sparse_warmup_global_layer=cfg["warmup_dense_global_end"],
            sparse_update_interval=cfg["sparse_update_interval"],
            sparse_update_fn=updater,
            runtime_attn_bias_dict=None,               # true sparse 主线下通常不用
            return_pose_per_layer=False,               # 主线先不做全层 pose 分析
            cam_num_iterations=cfg["camera_num_iterations"],
        )

if torch.cuda.is_available():
    torch.cuda.synchronize()

t1 = time.time()
elapsed = t1 - t0

if torch.cuda.is_available():
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
else:
    peak_mem_mb = None

print(f"[运行] staged sparse 推理完成，用时 {elapsed:.3f} 秒")
if peak_mem_mb is not None:
    print(f"[运行] peak GPU memory = {peak_mem_mb:.2f} MB")


# ============================================================
# 十三、保存 predictions
# ============================================================

if SAVE_PREDICTIONS:
    pred_path = os.path.join(out_dir, "predictions.pt")
    torch.save(nested_to_cpu(predictions), pred_path)
    print("[保存] predictions ->", pred_path)


# ============================================================
# 十四、单独保存在线更新的 pose（推荐）
# ============================================================

if SAVE_ONLINE_POSE_UPDATES and ("online_pose_updates" in predictions):
    online_pose_path = os.path.join(out_dir, "online_pose_updates.pt")
    torch.save(nested_to_cpu(predictions["online_pose_updates"]), online_pose_path)
    print("[保存] online_pose_updates ->", online_pose_path)


# ============================================================
# 十五、保存运行摘要
# ============================================================

summary = {
    "elapsed_sec": elapsed,
    "peak_gpu_mem_mb": peak_mem_mb,
    "num_images": len(image_names),
    "proc_h": proc_h,
    "proc_w": proc_w,
    "staged_sparse": cfg["use_staged_sparse"],
    "warmup_dense_global_end": cfg["warmup_dense_global_end"],
    "sparse_apply_start": cfg["sparse_apply_start"],
    "sparse_update_interval": cfg["sparse_update_interval"],
    "update_mode": cfg["update_mode"],
    "pose_smoothing": cfg["pose_smoothing"],
    "bandwidth_by_anchor_layer": cfg["bandwidth_by_anchor_layer"],
    "apply_layers_by_anchor": cfg["apply_layers_by_anchor"],
}
save_json(os.path.join(out_dir, "run_summary.json"), summary)
print("[保存] run_summary.json")


# ============================================================
# 十六、简单 sanity check
# ============================================================

print("\n预测结果包含的 key：")
for k, v in predictions.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: type={type(v)}")

if "online_pose_updates" in predictions:
    anchor_layers = sorted(list(predictions["online_pose_updates"].keys()))
    print(f"\n[Sanity] online_pose_updates anchor layers = {anchor_layers}")
    if len(anchor_layers) > 0:
        L0 = anchor_layers[0]
        pe = predictions["online_pose_updates"][L0]["pose_enc"]
        print(f"[Sanity] online_pose_updates[{L0}]['pose_enc'] shape={tuple(pe.shape)} dtype={pe.dtype}")

print("\nStaged epipolar sparse attention pipeline 完成。")
