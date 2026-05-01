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
SCENE = "layer_16_to_23_bw42_soft_band_alpha_2"
# SCENE = "baseline"

# 只在 global attention 的某个区间层启用（0-based）
EPI_GLOBAL_LAYER_START = 16
EPI_GLOBAL_LAYER_END = 23

# 你的 7Scenes 场景目录（里面应包含 seq-XX 子目录）
# 例如：datasets/7scenes/office
SCENE_DIR = r"datasets/scene3_7Scenes"

# 你选的序列（比如 seq-01）
SEQ_NAME = "seq-01"

FRAME_IDS = ["000000", "000010", "000020", "000030", "000040", "000050", "000060", "000070", "000080", "000090"]

# 输出目录（保存 attention 和预测结果）
OUT_DIR = r"outputs/token_attention/scene3_7Scenes/eight_layers"

# 预训练模型名称
MODEL_NAME = "facebook/VGGT-1B"

# ========== Epipolar Band（固定索引）配置 ==========
ENABLE_EPI_BAND = True

# 是否使用 soft band（attn_bias）而非 hard band（token_mask）
USE_EPI_SOFT = True

# soft band 关键超参: 越大越接近 hard band(可以尝试 2/4/6/8)
EPI_BAND_ALPHA = 2.0

# 你预计算的 pt（compute_band_index.py 输出）
EPI_INDEX_PT = r"outputs/token_attention/scene3_7Scenes/scene3_band_bw42.pt"

# ========== 可选：是否继续保存 attention（用于对比 band 前后变化）==========
SAVE_STATE_DICT = True
SAVE_ATTENTION = False

# 若保存 attention，要 hook 的 global attention 层编号（0-based）
TARGET_LAYERS = [23]

# Attention 缓存模式：
#   "full"      : 缓存 [B, H, N, N]（最完整，最耗显存）
#   "head_mean" : 缓存 [B, N, N]（对 head 取平均）
CACHE_MODE = "head_mean"

# 是否只抽样部分 query（用于节省显存）
SAMPLE_Q = 0

# 是否保存 predictions
SAVE_PREDICTIONS = True

# 是否保存配置快照（建议打开，方便复现实验）
SAVE_CONFIG = True

# 预处理模式：crop/pad（和官方 quickstart 一致）
# crop：宽缩放到518，高按比例缩放后再中心裁剪到<=518（并对齐14）
PREPROCESS_MODE = "crop"


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
# 四、收集 10 张 color 图像路径（只读 *.color.png）
# ============================================================

scene_dir = SCENE_DIR
assert os.path.isdir(scene_dir), f"找不到 scene 目录：{scene_dir}"

image_names = []
for fid in FRAME_IDS:
    img_path = os.path.join(scene_dir, f"frame-{fid}.color.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到 color 图：{img_path}")
    image_names.append(img_path)

print(f"将使用 {len(image_names)} 张 RGB 图像进行推理：")
for p in image_names:
    print("  ", p)

# ============================================================
# 五、加载并预处理图像（只对 RGB）
# ============================================================

images = load_and_preprocess_images(image_names, mode=PREPROCESS_MODE).to(device)
# images: (S,3,H,W)
print(f"[DBG] 输入 images.shape = {tuple(images.shape)}")

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
                "SEQ_NAME": SEQ_NAME,
                "FRAME_IDS": FRAME_IDS,
                "OUT_DIR": OUT_DIR,
                "MODEL_NAME": MODEL_NAME,
                "PREPROCESS_MODE": PREPROCESS_MODE,
                "ENABLE_EPI_BAND": ENABLE_EPI_BAND,
                "EPI_INDEX_PT": EPI_INDEX_PT,
                "EPI_GLOBAL_LAYER_START": EPI_GLOBAL_LAYER_START,
                "EPI_GLOBAL_LAYER_END": EPI_GLOBAL_LAYER_END,
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
    print("[保存] cfg.json ->", cfg_path)

# ============================================================
# 七、（可选）Attention 插桩函数：缓存注意力矩阵
# ============================================================

def patch_attention_forward(attn_module, *, cache_mode="full", sample_q=0):
    """
    - cache_mode:
        "full"      : 缓存 [B, H, N, N]
        "head_mean" : 缓存 [B, N, N]（对 head 求平均）
    - sample_q:
        0     : 不抽样，保存所有 query
        >0    : 只保存前 sample_q 个 query
    注意：
      这里显式算 attention，因此会覆盖 fused kernel。
    """
    assert cache_mode in ("full", "head_mean")
    assert isinstance(sample_q, int) and sample_q >= 0

    def wrapped_forward(self, x, pos=None, token_mask=None, attn_bias=None):
        B, N, C = x.shape

        # 1) QKV
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # 2) RoPE
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # 3) attention logits
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B,H,N,N]

        # soft band：加到 logits
        if attn_bias is not None:
            if attn_bias.ndim == 2:
                attn_bias = attn_bias[None, None, :, :]
            elif attn_bias.ndim == 3:
                attn_bias = attn_bias[:, None, :, :]
            attn = attn + attn_bias

        # hard mask：mask 掉非法位置
        if token_mask is not None:
            if token_mask.dtype != torch.bool:
                token_mask = token_mask.to(dtype=torch.bool)
            if token_mask.ndim == 3:
                token_mask = token_mask[:, None, :, :]
            neg = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(~token_mask, neg)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4) 缓存
        attn_to_cache = attn
        if sample_q > 0:
            attn_to_cache = attn_to_cache[:, :, :sample_q, :]
        if cache_mode == "head_mean":
            attn_to_cache = attn_to_cache.mean(dim=1)

        self._last_attn = attn_to_cache.detach().cpu()

        # 5) 输出
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    attn_module.forward = types.MethodType(wrapped_forward, attn_module)

# ============================================================
# 八、加载 VGGT 模型
# ============================================================

model = VGGT.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ============================================================
# 九、启用固定 Epipolar Band（如果你 aggregator.py 已经加了对应字段）
# ============================================================

if ENABLE_EPI_BAND:
    model.aggregator.enable_epi_band = True
    model.aggregator.use_epi_soft = USE_EPI_SOFT
    model.aggregator.epi_band_alpha = EPI_BAND_ALPHA
    model.aggregator.epi_band_index_pt = EPI_INDEX_PT
    model.aggregator.epi_band_global_layer_start = EPI_GLOBAL_LAYER_START
    model.aggregator.epi_band_global_layer_end = EPI_GLOBAL_LAYER_END

    print("[EPI] 已启用 Epipolar Band")
    print(f"      index_pt = {EPI_INDEX_PT}")
    print(f"      global layers = [{EPI_GLOBAL_LAYER_START}, {EPI_GLOBAL_LAYER_END}]")
else:
    model.aggregator.enable_epi_band = False
    print("[EPI] 未启用 Epipolar Band（baseline）")

# ============================================================
# 十、（可选）Hook attention 保存
# ============================================================

if SAVE_ATTENTION:
    for lid in TARGET_LAYERS:
        attn_mod = model.aggregator.global_blocks[lid].attn
        attn_mod.fused_attn = False  # 强制不用 fused kernel
        patch_attention_forward(attn_mod, cache_mode=CACHE_MODE, sample_q=SAMPLE_Q)

    print("已 hook 的层：", TARGET_LAYERS)
    print("Attention 缓存模式：", CACHE_MODE)
    print("Query 抽样数：", SAMPLE_Q)

# ============================================================
# 十一、前向推理（10 张 view）
# ============================================================

t0 = time.time()
with torch.no_grad():
    # 这里 images 是 (S,3,H,W)，VGGT 接口支持这种输入
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images)
t1 = time.time()
print(f"[运行] 推理完成，用时 {t1 - t0:.3f} 秒")

# ============================================================
# 十二、（可选）保存每一层 attention
# ============================================================

if SAVE_ATTENTION:
    for lid in TARGET_LAYERS:
        attn_cached = model.aggregator.global_blocks[lid].attn._last_attn
        save_path = os.path.join(
            out_dir,
            f"attn_layer{lid:02d}_{CACHE_MODE}_q{SAMPLE_Q}.pt"
        )
        torch.save(attn_cached, save_path)
        print(f"[保存] Layer {lid:02d} attention，shape={tuple(attn_cached.shape)} -> {save_path}")

# ============================================================
# 十三、保存预测结果（predictions.pt）
# ============================================================
if SAVE_STATE_DICT:   # 或者你单独加一个 SAVE_MODEL
    ckpt_path = os.path.join(out_dir, "model_state_dict.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("[保存] model.state_dict ->", ckpt_path)

if SAVE_PREDICTIONS:
    pred_path = os.path.join(out_dir, "predictions.pt")
    torch.save(
        {
            **predictions,
            # 额外保存：你用的帧编号和路径，后面评测对齐用得上
            "frame_ids": FRAME_IDS,
            "image_paths": image_names,
            "scene_dir": SCENE_DIR,
            "seq_name": SEQ_NAME,
        },
        pred_path
    )
    print("[保存] predictions ->", pred_path)

# ============================================================
# 十四、简单 sanity check
# ============================================================

print("\n预测结果包含的 key：")
for k, v in predictions.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: type={type(v)}")

print("\n完成：10-view 推理并保存 predictions.pt")
