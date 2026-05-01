# pipeline_epipolar_sparse_attention_benchmark.py
# ============================================================
# Fair benchmarking pipeline for:
#   1) baseline dense VGGT
#   2) staged epipolar sparse attention VGGT
#
# Main goals:
#   - fair timing benchmark
#   - fair memory benchmark
#   - optional sparse metrics recording (layers after 15, default from 16)
#
# Notes:
#   - warmup runs are NOT included in final reported time
#   - saving predictions / sparse metrics is NOT included in benchmark time
#   - sparse metrics are collected in one extra forward pass after benchmark
# ============================================================

import os
import json
import time
import random
from copy import deepcopy

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from sparse_config import (
    get_default_sparse_cfg,
    summarize_sparse_cfg,
    get_bandwidth_for_anchor,
)
from sparse_epipolar_builder import EpipolarSparseBuilder
from epipolar_update_function import EpipolarUpdateFunction
from save_sparse_metrics import SparseMetricsRecorder


# ============================================================
# 一、【用户可配置区域】
# ============================================================

# True: baseline dense
# False: staged epipolar sparse
IS_BASELINE = True

SEED = 0
SCENE_DIR = "datasets/scene1_DTU"
MODEL_NAME = "facebook/VGGT-1B"

OUT_ROOT = "outputs/pipeline_epipolar_sparse_attention_benchmark/scene1_DTU"
RUN_NAME = "baseline" if IS_BASELINE else "staged_sparse"

# benchmark settings
WARMUP_ITERS = 5
BENCH_ITERS = 20

# output control
SAVE_CONFIG = True
SAVE_PREDICTIONS = True
SAVE_ONLINE_POSE_UPDATES = True
SAVE_SPARSE_METRICS = True

# optional analysis output
RETURN_POSE_PER_LAYER = False
POSE_LAYER_IDS = None
CAM_NUM_ITERATIONS = 4

# sparse metrics record range: after layer 15 -> from 16
MIN_LAYER_TO_RECORD = 16

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

if VERBOSE:
    print(f"[INFO] device = {device}")
    print(f"[INFO] dtype  = {dtype}")


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


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mean_std(x):
    x = np.array(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=0))


# ============================================================
# 五、读取图像
# ============================================================

image_names = sorted([
    os.path.join(SCENE_DIR, fname)
    for fname in os.listdir(SCENE_DIR)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
])

if len(image_names) == 0:
    raise RuntimeError(f"No images found in {SCENE_DIR}")

print(f"[INFO] 从场景目录加载 {len(image_names)} 张图像：")
for name in image_names:
    print("   ", name)


# ============================================================
# 六、加载并预处理图像
# ============================================================

images = load_and_preprocess_images(image_names).to(device)

# 期望 shape: [S, 3, H, W]
if images.ndim != 4:
    raise RuntimeError(
        f"Expected load_and_preprocess_images to return [S,3,H,W], got shape={tuple(images.shape)}"
    )

num_views = int(images.shape[0])
proc_h, proc_w = int(images.shape[-2]), int(images.shape[-1])

print(f"[INFO] processed image size = (H={proc_h}, W={proc_w})")
print(f"[INFO] num_views = {num_views}")


# ============================================================
# 七、创建输出目录
# ============================================================

out_dir = os.path.join(OUT_ROOT, RUN_NAME)
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 八、加载模型
# ============================================================

model = VGGT.from_pretrained(MODEL_NAME).to(device)
model.eval()


# ============================================================
# 九、配置 baseline / staged sparse
# ============================================================

cfg = None
builder = None
updater = None

if IS_BASELINE:
    # baseline：明确关闭 sparse
    model.aggregator.enable_partial_sparse = False

    if VERBOSE:
        print("[INFO] Running BASELINE benchmark")
        print("[INFO] aggregator.enable_partial_sparse = False")

else:
    cfg = get_default_sparse_cfg()
    cfg["debug_print"] = False

    # staged sparse：打开 sparse，总体层区间由 cfg 控制
    model.aggregator.enable_partial_sparse = True
    model.aggregator.partial_sparse_start_layer = cfg["sparse_apply_start"]
    model.aggregator.partial_sparse_end_layer = cfg["num_global_layers"] - 1

    builder = EpipolarSparseBuilder(
        proc_h=proc_h,
        proc_w=proc_w,
        patch_size=model.aggregator.patch_size,
        patch_start_idx=model.aggregator.patch_start_idx,
        k_mode="fixed_prior",
        fixed_focal_ratio=0.9,
    )

    updater = EpipolarUpdateFunction(
        cfg=cfg,
        builder=builder,
    )

    if VERBOSE:
        print("[INFO] Running STAGED SPARSE benchmark")
        print(summarize_sparse_cfg(cfg))
        print("[INFO] aggregator.enable_partial_sparse =", model.aggregator.enable_partial_sparse)
        print("[INFO] partial_sparse_start_layer      =", model.aggregator.partial_sparse_start_layer)
        print("[INFO] partial_sparse_end_layer        =", model.aggregator.partial_sparse_end_layer)
        print("[INFO] patch_size                      =", model.aggregator.patch_size)
        print("[INFO] patch_start_idx                 =", model.aggregator.patch_start_idx)


# ============================================================
# 十、保存配置
# ============================================================

if SAVE_CONFIG:
    cfg_dump = {
        "IS_BASELINE": IS_BASELINE,
        "SEED": SEED,
        "SCENE_DIR": SCENE_DIR,
        "OUT_ROOT": OUT_ROOT,
        "RUN_NAME": RUN_NAME,
        "MODEL_NAME": MODEL_NAME,
        "device": device,
        "dtype": str(dtype),
        "num_views": num_views,
        "proc_h": proc_h,
        "proc_w": proc_w,
        "WARMUP_ITERS": WARMUP_ITERS,
        "BENCH_ITERS": BENCH_ITERS,
        "RETURN_POSE_PER_LAYER": RETURN_POSE_PER_LAYER,
        "POSE_LAYER_IDS": POSE_LAYER_IDS,
        "CAM_NUM_ITERATIONS": CAM_NUM_ITERATIONS,
        "MIN_LAYER_TO_RECORD": MIN_LAYER_TO_RECORD,
        "SAVE_PREDICTIONS": SAVE_PREDICTIONS,
        "SAVE_ONLINE_POSE_UPDATES": SAVE_ONLINE_POSE_UPDATES,
        "SAVE_SPARSE_METRICS": SAVE_SPARSE_METRICS,
    }

    if not IS_BASELINE:
        cfg_dump["staged_sparse_cfg"] = deepcopy(cfg)
        cfg_dump["builder"] = {
            "patch_size": model.aggregator.patch_size,
            "patch_start_idx": model.aggregator.patch_start_idx,
            "k_mode": "fixed_prior",
            "fixed_focal_ratio": 0.9,
        }

    save_json(os.path.join(out_dir, "cfg.json"), cfg_dump)
    print("[保存] cfg.json")


# ============================================================
# 十一、单次 forward（不保存文件）
# ============================================================

def run_model_once(sparse_update_fn=None):
    """
    单次 forward。
    benchmark 阶段不做任何保存动作。
    """
    if IS_BASELINE:
        preds = model(
            images,
            staged_sparse=False,
            runtime_sparse_dict=None,
            runtime_attn_bias_dict=None,
            return_pose_per_layer=RETURN_POSE_PER_LAYER,
            pose_layer_ids=POSE_LAYER_IDS,
            cam_num_iterations=CAM_NUM_ITERATIONS,
        )
    else:
        preds = model(
            images,
            staged_sparse=True,
            sparse_warmup_global_layer=cfg["warmup_dense_global_end"],
            sparse_update_interval=cfg["sparse_update_interval"],
            sparse_update_fn=sparse_update_fn if sparse_update_fn is not None else updater,
            runtime_attn_bias_dict=None,
            return_pose_per_layer=RETURN_POSE_PER_LAYER,
            pose_layer_ids=POSE_LAYER_IDS,
            cam_num_iterations=cfg["camera_num_iterations"],
        )
    return preds


# ============================================================
# 十二、公平 benchmark
# ============================================================

def benchmark_model():
    # -------------------------
    # warmup
    # -------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            with torch.amp.autocast("cuda", dtype=dtype):
                _ = run_model_once()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------
    # formal benchmark
    # -------------------------
    event_times_ms = []
    wall_times_ms = []
    peak_alloc_mb_list = []
    peak_reserved_mb_list = []

    with torch.inference_mode():
        for _ in range(BENCH_ITERS):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
            else:
                starter = None
                ender = None

            t0 = time.perf_counter()

            if torch.cuda.is_available():
                starter.record()

            with torch.amp.autocast("cuda", dtype=dtype):
                preds = run_model_once()

            if torch.cuda.is_available():
                ender.record()
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            wall_times_ms.append((t1 - t0) * 1000.0)

            if torch.cuda.is_available():
                event_times_ms.append(float(starter.elapsed_time(ender)))
                peak_alloc_mb_list.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
                peak_reserved_mb_list.append(torch.cuda.max_memory_reserved() / (1024 ** 2))
            else:
                event_times_ms.append((t1 - t0) * 1000.0)
                peak_alloc_mb_list.append(None)
                peak_reserved_mb_list.append(None)

            del preds

    return {
        "event_times_ms": event_times_ms,
        "wall_times_ms": wall_times_ms,
        "peak_alloc_mb_list": peak_alloc_mb_list,
        "peak_reserved_mb_list": peak_reserved_mb_list,
    }


bench = benchmark_model()

event_mean, event_std = mean_std(bench["event_times_ms"])
wall_mean, wall_std = mean_std(bench["wall_times_ms"])

if torch.cuda.is_available():
    alloc_mean, alloc_std = mean_std(bench["peak_alloc_mb_list"])
    reserved_mean, reserved_std = mean_std(bench["peak_reserved_mb_list"])
else:
    alloc_mean = alloc_std = None
    reserved_mean = reserved_std = None

views_per_sec = num_views / (event_mean / 1000.0)

print("\n================ FAIR BENCHMARK SUMMARY ================")
print(f"Mode                  : {'BASELINE' if IS_BASELINE else 'STAGED_SPARSE'}")
print(f"Warmup iters          : {WARMUP_ITERS}")
print(f"Benchmark iters       : {BENCH_ITERS}")
print(f"CUDA event time       : {event_mean:.3f} ± {event_std:.3f} ms")
print(f"Wall clock time       : {wall_mean:.3f} ± {wall_std:.3f} ms")
print(f"Views / sec           : {views_per_sec:.3f}")
if torch.cuda.is_available():
    print(f"Peak alloc memory     : {alloc_mean:.2f} ± {alloc_std:.2f} MB")
    print(f"Peak reserved memory  : {reserved_mean:.2f} ± {reserved_std:.2f} MB")
print("========================================================\n")


# ============================================================
# 十三、额外跑一次 forward（不计入 benchmark）
# 用于保存 predictions / sparse metrics
# ============================================================

predictions = None
sparse_metrics_export = None

if SAVE_PREDICTIONS or (SAVE_ONLINE_POSE_UPDATES and not IS_BASELINE) or (SAVE_SPARSE_METRICS and not IS_BASELINE):
    if not IS_BASELINE:
        # ---------------------------------------------
        # 单独构造一个 recorder，只用于这一次非计时 forward
        # 避免 benchmark 重复记录、干扰时间
        # ---------------------------------------------
        recorder = SparseMetricsRecorder(
            proc_h=proc_h,
            proc_w=proc_w,
            patch_size=model.aggregator.patch_size,
            patch_start_idx=model.aggregator.patch_start_idx,
            num_views=num_views,
            min_layer_to_record=MIN_LAYER_TO_RECORD,
        )

        # 包装 sparse_update_fn：在真正 update 之后记录 sparse metrics
        def sparse_update_fn_with_record(anchor_global_idx, pose_enc, state, model):
            runtime_sparse_dict = updater(
                anchor_global_idx=anchor_global_idx,
                pose_enc=pose_enc,
                state=state,
                model=model,
            )

            bandwidth_patches = get_bandwidth_for_anchor(cfg, anchor_global_idx)
            recorder.record_runtime_sparse_dict(
                runtime_sparse_dict=runtime_sparse_dict,
                anchor_global_idx=anchor_global_idx,
                bandwidth_patches=bandwidth_patches,
            )
            return runtime_sparse_dict

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = run_model_once(sparse_update_fn=sparse_update_fn_with_record)

        sparse_metrics_export = recorder.export()

    else:
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = run_model_once()


# ============================================================
# 十四、保存 predictions
# ============================================================

if SAVE_PREDICTIONS and predictions is not None:
    pred_path = os.path.join(out_dir, "predictions.pt")
    torch.save(nested_to_cpu(predictions), pred_path)
    print("[保存] predictions ->", pred_path)


# ============================================================
# 十五、保存 online pose updates（仅 sparse）
# ============================================================

if (not IS_BASELINE) and SAVE_ONLINE_POSE_UPDATES and predictions is not None and ("online_pose_updates" in predictions):
    online_pose_path = os.path.join(out_dir, "online_pose_updates.pt")
    torch.save(nested_to_cpu(predictions["online_pose_updates"]), online_pose_path)
    print("[保存] online_pose_updates ->", online_pose_path)


# ============================================================
# 十六、保存 sparse metrics（仅 sparse）
# ============================================================

if (not IS_BASELINE) and SAVE_SPARSE_METRICS and sparse_metrics_export is not None:
    sparse_metrics_json = os.path.join(out_dir, "sparse_metrics.json")
    sparse_metrics_pt = os.path.join(out_dir, "sparse_metrics.pt")

    # json
    def to_jsonable(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {str(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return [to_jsonable(v) for v in obj]
        return obj

    save_json(sparse_metrics_json, to_jsonable(sparse_metrics_export))
    torch.save(sparse_metrics_export, sparse_metrics_pt)

    print("[保存] sparse_metrics.json ->", sparse_metrics_json)
    print("[保存] sparse_metrics.pt   ->", sparse_metrics_pt)

    # 简单打印 overall
    overall = sparse_metrics_export.get("overall_metrics", {})
    if len(overall) > 0:
        print("\n================ SPARSE METRICS SUMMARY ================")
        print(f"Recorded layers            : {overall.get('recorded_layer_ids', [])}")
        print(f"Dense keys / patch query   : {overall.get('dense_crossview_patch_keys_per_query', 'N/A')}")
        print(f"Overall avg kept keys      : {overall.get('overall_avg_kept_crossview_keys_per_query', 'N/A')}")
        print(f"Overall avg keep percent   : {overall.get('overall_avg_keep_percent', 'N/A'):.3f}%")
        print(f"Overall avg reduction      : {overall.get('overall_avg_reduction_percent', 'N/A'):.3f}%")
        print("========================================================\n")


# ============================================================
# 十七、保存 benchmark 摘要
# ============================================================

summary = {
    "mode": "baseline" if IS_BASELINE else "staged_sparse",
    "event_time_ms_mean": event_mean,
    "event_time_ms_std": event_std,
    "wall_time_ms_mean": wall_mean,
    "wall_time_ms_std": wall_std,
    "views_per_sec": views_per_sec,
    "num_views": num_views,
    "proc_h": proc_h,
    "proc_w": proc_w,
    "device": device,
    "dtype": str(dtype),
    "warmup_iters": WARMUP_ITERS,
    "bench_iters": BENCH_ITERS,
}

if torch.cuda.is_available():
    summary["peak_alloc_mb_mean"] = alloc_mean
    summary["peak_alloc_mb_std"] = alloc_std
    summary["peak_reserved_mb_mean"] = reserved_mean
    summary["peak_reserved_mb_std"] = reserved_std

if not IS_BASELINE:
    summary["warmup_dense_global_end"] = cfg["warmup_dense_global_end"]
    summary["sparse_apply_start"] = cfg["sparse_apply_start"]
    summary["sparse_update_interval"] = cfg["sparse_update_interval"]
    summary["update_mode"] = cfg["update_mode"]
    summary["pose_smoothing"] = cfg["pose_smoothing"]
    summary["bandwidth_by_anchor_layer"] = cfg["bandwidth_by_anchor_layer"]
    summary["apply_layers_by_anchor"] = cfg["apply_layers_by_anchor"]

    if sparse_metrics_export is not None:
        summary["sparse_overall_metrics"] = sparse_metrics_export.get("overall_metrics", {})

save_json(os.path.join(out_dir, "run_summary.json"), summary)
print("[保存] run_summary.json")


# ============================================================
# 十八、简单 sanity check
# ============================================================

if predictions is not None:
    print("\n预测结果包含的 key：")
    for k, v in predictions.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"{k}: type={type(v)}")

    if (not IS_BASELINE) and ("online_pose_updates" in predictions):
        anchor_layers = sorted(list(predictions["online_pose_updates"].keys()))
        print(f"\n[Sanity] online_pose_updates anchor layers = {anchor_layers}")
        if len(anchor_layers) > 0:
            L0 = anchor_layers[0]
            pe = predictions["online_pose_updates"][L0]["pose_enc"]
            print(f"[Sanity] online_pose_updates[{L0}]['pose_enc'] shape={tuple(pe.shape)} dtype={pe.dtype}")

print(f"\n{RUN_NAME} benchmark pipeline 完成。")