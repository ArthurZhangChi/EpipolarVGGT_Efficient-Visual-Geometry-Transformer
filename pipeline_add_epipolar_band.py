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
SCENE = "layer_16_to_23_bw3"

# 只在 global attention 的某个区间层启用（0-based）
EPI_GLOBAL_LAYER_START = 16
EPI_GLOBAL_LAYER_END = 23

# 场景图像目录
SCENE_DIR = "datasets/scene1_DTU"

# 输出目录（保存 attention 和预测结果）
OUT_DIR = "outputs/token_attention/scene1_DTU/add_epipolar_band/bandwidth_compare"

# 预训练模型名称
MODEL_NAME = "facebook/VGGT-1B"

# ========== Epipolar Band（固定索引）配置 ==========
ENABLE_EPI_BAND = True

# 是否使用 soft band（attn_bias）而非 hard band（token_mask）
USE_EPI_SOFT = False

# soft band 关键超参: 越大越接近 hard band(可以尝试 2/4/6/8)
EPI_BAND_ALPHA = 4.0

# 你预计算的 pt（compute_band_index.py 输出）
EPI_INDEX_PT = r"outputs/token_attention/scene1_DTU/add_epipolar_band/scene1_band_bw3.pt"

# ========== 可选：是否继续保存 attention（用于对比 band 前后变化）==========
SAVE_ATTENTION = True

# 若保存 attention，要 hook 的 global attention 层编号（0-based）
TARGET_LAYERS = [23]

# Attention 缓存模式：
#   "full"      : 缓存 [B, H, N, N]（最完整，最耗显存）
#   "head_mean" : 缓存 [B, N, N]（对 head 取平均）
CACHE_MODE = "head_mean"

# 是否只抽样部分 query（用于节省显存）
SAMPLE_Q = 0

# 是否保存 predictions（你要的 SAVE_PREDICTIONS）
SAVE_PREDICTIONS = True

# 是否保存配置快照（建议打开，方便复现实验）
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
    print("[保存] cfg.json")

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
      如果你要“观察 band 后 attention”，建议在 band 启用的层上 hook。
    """
    assert cache_mode in ("full", "head_mean")
    assert isinstance(sample_q, int) and sample_q >= 0

    def wrapped_forward(self, x, pos=None, token_mask=None, attn_bias=None):
        if (attn_bias is not None) and (not hasattr(self, "_dbg_soft_once")):
            self._dbg_soft_once = True
            print("[DBG] attn_bias:", attn_bias.min().item(), attn_bias.max().item())

            # 用 “band 内/外” 的 mask 来算注意力总质量（需要你把 base_mask 也传进来或在外面拿到）

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

        # 3) 显式 attention（⚠️ 必须加 attn_bias）
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B,H,N,N]


        # ===== 关键：soft band 在这里生效 =====
        if attn_bias is not None:
            if attn_bias.ndim == 2:
                attn_bias = attn_bias[None, None, :, :]
            elif attn_bias.ndim == 3:
                attn_bias = attn_bias[:, None, :, :]

            attn = attn + attn_bias   # ✅ soft band 加到 logits
        # =====================================

        # hard mask（如果存在）
        if token_mask is not None:
            if token_mask.dtype != torch.bool:
                token_mask = token_mask.to(dtype=torch.bool)
            if token_mask.ndim == 3:
                token_mask = token_mask[:, None, :, :]
            neg = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(~token_mask, neg)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ================= mass_in_band 统计 =================
        if not hasattr(self, "_mass_logged"):
            self._mass_logged = True

            # attn: [B,H,N,N]
            attn_prob = attn.detach()

            # band mask: [1,1,N,N] -> broadcast
            m = band_mask_for_metric.to(attn_prob.device)

            # 只算一个 batch、head mean（避免显存爆炸）
            attn_mean = attn_prob.mean(dim=1, keepdim=True)  # [B,1,N,N]

            mass_in = (attn_mean * m).sum().item()
            mass_all = attn_mean.sum().item()

            print(
                f"[MASS_IN_BAND] layer={self.layer_id if hasattr(self,'layer_id') else 'unk'} "
                f"| ratio={mass_in / mass_all:.4f}"
            )
        # =====================================================

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
# 九、启用固定 Epipolar Band（核心）
# ============================================================

if ENABLE_EPI_BAND:
    # 这些字段需要你在 aggregator.py 里加过
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
# [新增] 为 mass_in_band 计算准备 band mask（pipeline 内）
# ============================================================

with torch.no_grad():
    agg = model.aggregator

    # 0) images 可能是 [S,3,H,W]，先补 batch 维
    if images.ndim == 4:
        images_ = images.unsqueeze(0)   # [1,S,3,H,W]
    else:
        images_ = images

    B, S, _, H, W = images_.shape
    patch_size = agg.patch_size

    # 1) 计算 P / N（与你 aggregator forward 里拼 token 的逻辑一致）
    Hp = H // patch_size
    Wp = W // patch_size
    P_patch = Hp * Wp
    P = agg.patch_start_idx + P_patch   # 每个 view 的 token 总数（special+patch）
    N = S * P

    # 2) 无论是否启用 band，都把 index 路径写进去（否则 baseline 分支没赋值会出错）
    agg.epi_band_index_pt = EPI_INDEX_PT

    # 3) 关键：baseline 时 enable=False 会导致 _load_epi_index_if_needed() 不加载
    #    所以这里临时打开一次 enable，加载完再恢复原状态
    old_enable = getattr(agg, "enable_epi_band", False)
    agg.enable_epi_band = True
    agg._load_epi_index_if_needed()
    agg.enable_epi_band = old_enable

    # 4) build mask（device 用 images_.device，别用字符串 "cuda"）
    base_mask = agg._build_global_token_mask(S=S, P=P, device=images_.device)  # [1,1,N,N] bool

    # 5) 你做 metric 用 batch=1 就够
    band_mask_for_metric = base_mask  # [1,1,N,N]

    print(f"[DBG] band_mask_for_metric: shape={tuple(band_mask_for_metric.shape)}, "
          f"true_ratio={band_mask_for_metric.float().mean().item():.4f}, N={N}")

# ============================================================
# 十、（可选）Hook attention 保存
# ============================================================

if SAVE_ATTENTION:
    for lid in TARGET_LAYERS:
        attn_mod = model.aggregator.global_blocks[lid].attn
        # 强制不用 fused kernel，保证能拿到 attention
        attn_mod.fused_attn = False
        patch_attention_forward(attn_mod, cache_mode=CACHE_MODE, sample_q=SAMPLE_Q)

    print("已 hook 的层：", TARGET_LAYERS)
    print("Attention 缓存模式：", CACHE_MODE)
    print("Query 抽样数：", SAMPLE_Q)

# ============================================================
# 十一、前向推理
# ============================================================

t0 = time.time()
with torch.no_grad():
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
# 十三、保存预测结果（SAVE_PREDICTIONS）
# ============================================================

if SAVE_PREDICTIONS:
    pred_path = os.path.join(out_dir, "predictions.pt")
    torch.save(predictions, pred_path)
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

print("\n固定 Epipolar Band 推理完成。")
