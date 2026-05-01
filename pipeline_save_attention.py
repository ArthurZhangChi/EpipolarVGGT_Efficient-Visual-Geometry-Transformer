import os
import random
import numpy as np
import torch
import types

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# ============================================================
# 一、【用户可配置区域】——只需要改这里
# ============================================================

# 随机种子（保证科研可复现）
SEED = 0

# 场景名称（用于输出子目录）
SCENE = "scene2_DTU"

# 场景图像目录
SCENE_DIR = "datasets/scene2_DTU"

# 输出目录（保存 attention 和预测结果）
OUT_DIR = "outputs/token_attention"

# 预训练模型名称
MODEL_NAME = "facebook/VGGT-1B"

# 要 hook 的 global attention 层编号（0-based）
# 建议：浅 / 中 / 深 / 最后层
TARGET_LAYERS = [0, 4, 8, 12, 16, 20, 23]

# Attention 缓存模式：
#   "full"      : 缓存 [B, H, N, N]（最完整，最耗显存）
#   "head_mean" : 缓存 [B, N, N]（对 head 取平均，阶段0强烈推荐）
CACHE_MODE = "head_mean"

# 是否只抽样部分 query（用于节省显存）
#   0    : 不抽样，保存所有 query
#   256  : 只保存前 256 个 query（做冗余分析已足够）
SAMPLE_Q = 0

# 是否保存 predictions（可选）
SAVE_PREDICTIONS = True

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

# Amp 自动精度选择
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
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 七、Attention 插桩函数（缓存注意力矩阵）
# ============================================================
def patch_attention_forward(attn_module, *, cache_mode="full", sample_q=0):
    """
    参数说明：
    - cache_mode:
        "full"      : 缓存 [B, H, N, N]
        "head_mean" : 缓存 [B, N, N]（对 head 求平均）
    - sample_q:
        0     : 不抽样，保存所有 query
        >0    : 只保存前 sample_q 个 query，显著降低显存
    """
    assert cache_mode in ("full", "head_mean")
    assert isinstance(sample_q, int) and sample_q >= 0

    def wrapped_forward(self, x, pos=None, token_mask=None):
        B, N, C = x.shape

        # 1. 计算 QKV
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # [B, H, N, d]

        q, k = self.q_norm(q), self.k_norm(k)

        # 2. RoPE（如果启用）
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # 3. 显式计算 attention（而不是 fused kernel）
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)   # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4. 缓存 attention（仅用于分析，不影响模型）
        attn_to_cache = attn

        if sample_q > 0:
            attn_to_cache = attn_to_cache[:, :, :sample_q, :]  # [B, H, sample_q, N]

        if cache_mode == "head_mean":
            attn_to_cache = attn_to_cache.mean(dim=1)          # [B, N, N] 或 [B, sample_q, N]

        # 放到 CPU，避免 GPU 显存爆炸
        self._last_attn = attn_to_cache.detach().cpu()

        # 5. 正常 forward 输出
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
# 九、Hook 指定的 global attention 层
# ============================================================
# 说明：
# fused_attn=True 会使用 PyTorch fused kernel，拿不到 attention matrix
# 因此阶段0分析时必须强制 fused_attn=False
for lid in TARGET_LAYERS:
    attn_mod = model.aggregator.global_blocks[lid].attn
    attn_mod.fused_attn = False
    patch_attention_forward(
        attn_mod,
        cache_mode=CACHE_MODE,
        sample_q=SAMPLE_Q
    )

print("已 hook 的层：", TARGET_LAYERS)
print("Attention 缓存模式：", CACHE_MODE)
print("Query 抽样数：", SAMPLE_Q)

# ============================================================
# 十、前向推理
# ============================================================
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images)

# ============================================================
# 十一、保存每一层的 attention
# ============================================================
for lid in TARGET_LAYERS:
    attn_cached = model.aggregator.global_blocks[lid].attn._last_attn
    save_path = os.path.join(
        OUT_DIR, SCENE, 
        f"attn_layer{lid:02d}_{CACHE_MODE}_q{SAMPLE_Q}.pt"
    )
    torch.save(attn_cached, save_path)
    print(f"[保存] Layer {lid:02d} attention，shape={tuple(attn_cached.shape)}")

# ============================================================
# 十二、保存预测结果（可选）
# ============================================================
if SAVE_PREDICTIONS:
    pred_path = os.path.join(OUT_DIR, SCENE, "predictions.pt")
    torch.save(predictions, pred_path)
    print("[保存] predictions")

# ============================================================
# 十三、简单 sanity check
# ============================================================
print("\n预测结果包含的 key：")
for k, v in predictions.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: type={type(v)}")

print("\n阶段0 baseline 推理完成。")
