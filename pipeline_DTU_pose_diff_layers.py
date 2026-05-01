import os
import random
import json
import time
import numpy as np
import torch
import types

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# ============================================================
# 一、【用户可配置区域】——只需要改这里（配置放最前面）
# ============================================================

# 随机种子（保证科研可复现）
SEED = 0

# 场景名称（用于输出子目录）
SCENE = "pose_per_layer"

# 场景图像目录
SCENE_DIR = "datasets/scene1_DTU"

# 输出目录（保存 attention / predictions / pose_per_layer）
OUT_DIR = "outputs/compute_epipolar_line/scene1_DTU/baseline"

# 预训练模型名称
MODEL_NAME = "facebook/VGGT-1B"

# ====== 本次重点：输出每一层的相机结果（新增） ======
# 是否输出每一层 backbone tokens 对应的 pose
RETURN_POSE_PER_LAYER = True

# 要测试的层号列表：
# None 表示测试所有层（0..num_layers-1）
POSE_LAYER_IDS = None
# 例如只测前几层：POSE_LAYER_IDS = list(range(0, 8))
# 例如抽样测：POSE_LAYER_IDS = [0,1,2,3,5,7,11,15,23]

# CameraHead refinement 次数（默认 4，和 CameraHead.forward 一致）
CAM_NUM_ITERATIONS = 4

# ====== 可选：是否保存 attention（baseline 可关）======
SAVE_ATTENTION = False
TARGET_LAYERS = [23]
CACHE_MODE = "head_mean"
SAMPLE_Q = 0

# 是否保存 predictions（包含 pose_enc_by_layer）
SAVE_PREDICTIONS = True

# 是否保存配置快照（建议打开）
SAVE_CONFIG = True


# ============================================================
# 二、固定随机性（科研必备）
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ============================================================
# 三、设备与精度设置
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


# ============================================================
# 四、读取场景图像
# ============================================================

image_names = sorted([
    os.path.join(SCENE_DIR, fname)
    for fname in os.listdir(SCENE_DIR)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"从场景目录加载 {len(image_names)} 张图像：")
for name in image_names:
    print("  ", name)


# ============================================================
# 五、加载并预处理图像
# ============================================================

images = load_and_preprocess_images(image_names).to(device)


# ============================================================
# 六、创建输出目录
# ============================================================

out_dir = os.path.join(OUT_DIR, SCENE)
os.makedirs(out_dir, exist_ok=True)

if SAVE_CONFIG:
    cfg_path = os.path.join(out_dir, "cfg.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "SEED": SEED,
                "SCENE": SCENE,
                "SCENE_DIR": SCENE_DIR,
                "OUT_DIR": OUT_DIR,
                "MODEL_NAME": MODEL_NAME,
                "RETURN_POSE_PER_LAYER": RETURN_POSE_PER_LAYER,
                "POSE_LAYER_IDS": POSE_LAYER_IDS,
                "CAM_NUM_ITERATIONS": CAM_NUM_ITERATIONS,
                "SAVE_ATTENTION": SAVE_ATTENTION,
                "TARGET_LAYERS": TARGET_LAYERS,
                "CACHE_MODE": CACHE_MODE,
                "SAMPLE_Q": SAMPLE_Q,
                "SAVE_PREDICTIONS": SAVE_PREDICTIONS,
                "device": device,
                "dtype": str(dtype),
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    print("[保存] cfg.json")


# ============================================================
# 七、（可选）Attention 插桩函数：缓存注意力矩阵（保持原样）
# ============================================================

def patch_attention_forward(attn_module, *, cache_mode="full", sample_q=0):
    assert cache_mode in ("full", "head_mean")
    assert isinstance(sample_q, int) and sample_q >= 0

    def wrapped_forward(self, x, pos=None, token_mask=None, attn_bias=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # soft bias（baseline 一般不会传）
        if attn_bias is not None:
            if attn_bias.ndim == 2:
                attn_bias = attn_bias[None, None, :, :]
            elif attn_bias.ndim == 3:
                attn_bias = attn_bias[:, None, :, :]
            attn = attn + attn_bias

        if token_mask is not None:
            if token_mask.dtype != torch.bool:
                token_mask = token_mask.to(dtype=torch.bool)
            if token_mask.ndim == 3:
                token_mask = token_mask[:, None, :, :]
            neg = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(~token_mask, neg)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_to_cache = attn
        if sample_q > 0:
            attn_to_cache = attn_to_cache[:, :, :sample_q, :]
        if cache_mode == "head_mean":
            attn_to_cache = attn_to_cache.mean(dim=1)

        self._last_attn = attn_to_cache.detach().cpu()

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    attn_module.forward = types.MethodType(wrapped_forward, attn_module)


# ============================================================
# 八、加载 VGGT 模型（baseline：不启用 epipolar band）
# ============================================================

model = VGGT.from_pretrained(MODEL_NAME).to(device)
model.eval()

# baseline：明确关闭（即使你的 Aggregator 里有这些字段也不使用）
if hasattr(model, "aggregator") and hasattr(model.aggregator, "enable_epi_band"):
    model.aggregator.enable_epi_band = False
print("[Baseline] 不启用 Epipolar Band，仅输出每层 pose 供分析。")


# ============================================================
# 九、（可选）Hook attention 保存（baseline 可关）
# ============================================================

if SAVE_ATTENTION:
    for lid in TARGET_LAYERS:
        attn_mod = model.aggregator.global_blocks[lid].attn
        attn_mod.fused_attn = False
        patch_attention_forward(attn_mod, cache_mode=CACHE_MODE, sample_q=SAMPLE_Q)

    print("已 hook 的层：", TARGET_LAYERS)
    print("Attention 缓存模式：", CACHE_MODE)
    print("Query 抽样数：", SAMPLE_Q)


# ============================================================
# 十、前向推理（关键修改：传入 return_pose_per_layer / pose_layer_ids）
# ============================================================

t0 = time.time()
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(
            images,
            # ========================（新增）========================
            return_pose_per_layer=RETURN_POSE_PER_LAYER,   # ✅ 输出每层 pose
            pose_layer_ids=POSE_LAYER_IDS,                 # ✅ None 表示全层
            cam_num_iterations=CAM_NUM_ITERATIONS,         # ✅ refinement 次数
            # ======================================================
        )
t1 = time.time()
print(f"[运行] 推理完成，用时 {t1 - t0:.3f} 秒")


# ============================================================
# 十一、（可选）保存每一层 attention
# ============================================================

if SAVE_ATTENTION:
    for lid in TARGET_LAYERS:
        attn_cached = model.aggregator.global_blocks[lid].attn._last_attn
        save_path = os.path.join(out_dir, f"attn_layer{lid:02d}_{CACHE_MODE}_q{SAMPLE_Q}.pt")
        torch.save(attn_cached, save_path)
        print(f"[保存] Layer {lid:02d} attention，shape={tuple(attn_cached.shape)} -> {save_path}")


# ============================================================
# 十二、保存预测结果（包含 pose_enc_by_layer）
# ============================================================

if SAVE_PREDICTIONS:
    pred_path = os.path.join(out_dir, "predictions.pt")
    torch.save(predictions, pred_path)
    print("[保存] predictions ->", pred_path)


# ============================================================
# 十三、把每层 pose 单独导出成一个更轻量的文件（推荐）
# ============================================================

if RETURN_POSE_PER_LAYER and ("pose_enc_by_layer" in predictions):
    # 只存 pose，文件更小，便于你后处理脚本快速读
    pose_dump = {
        "pose_layer_ids": predictions.get("pose_layer_ids", None),
        "num_backbone_layers": predictions.get("num_backbone_layers", None),
        # layer -> pose_enc
        "pose_enc_by_layer": {L: v["pose_enc"].cpu() for L, v in predictions["pose_enc_by_layer"].items()},
    }
    pose_path = os.path.join(out_dir, "pose_enc_by_layer.pt")
    torch.save(pose_dump, pose_path)
    print("[保存] pose_enc_by_layer.pt ->", pose_path)


# ============================================================
# 十四、简单 sanity check
# ============================================================

print("\n预测结果包含的 key：")
for k, v in predictions.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: type={type(v)}")

if "pose_enc_by_layer" in predictions:
    some_L = sorted(predictions["pose_enc_by_layer"].keys())[0]
    pe = predictions["pose_enc_by_layer"][some_L]["pose_enc"]
    print(f"\n[Sanity] pose_enc_by_layer[{some_L}]['pose_enc'] shape={tuple(pe.shape)} dtype={pe.dtype}")

print("\nBaseline pose-per-layer pipeline 完成。")
