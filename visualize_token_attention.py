import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 一、【用户需要修改的配置区域】
# ============================================================

# 要可视化的层号
LAYER = 23

# attention 所在目录
ATTN_DIR = "outputs/token_attention/scene1_DTU"

# attention 文件名模板（与你保存时一致）
ATTN_FILENAME = f"attn_layer{LAYER:02d}_head_mean_q0.pt"

# attention 文件完整路径
ATTN_PATH = os.path.join(ATTN_DIR, ATTN_FILENAME)

# 保存可视化结果的目录
OUT_DIR = f"outputs/token_attention/scene1_DTU/visualizations/L{LAYER:02d}"
os.makedirs(OUT_DIR, exist_ok=True)

# 选择可视化参数
EPS = 1e-6          # log(attn + eps)
CMAP = "viridis"    # 论文常用 colormap

# special token 的数量（= patch_start_idx）
# ⚠️ 这个值你可以从 aggregator forward 里打印得到
# 例如：camera_token(1) + register_token(4) = 5
PATCH_START_IDX = 5

# zoom 区域大小
SPECIAL_ZOOM_SIZE = 300    # special 区域放大
PATCH_ZOOM_SIZE = 300     # patch 区域放大窗口

# ============================================================
# 二、加载 attention
# ============================================================

print("加载 attention:", ATTN_PATH)
attn = torch.load(ATTN_PATH)   # [1, N, N]
attn = attn[0].numpy()         # [N, N]

N = attn.shape[0]
print(f"Attention shape: {attn.shape}")

# ============================================================
# 三、log scale 处理（核心）
# ============================================================

log_attn = np.log1p(1e3 * attn)

# ============================================================
# 四、整体 attention heatmap
# ============================================================

plt.figure(figsize=(8, 8))
plt.imshow(attn, cmap=CMAP)
plt.title("Global Attention (log scale)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, f"layer{LAYER:02d}_global_attention_log.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"[保存] {out_path}")

# ============================================================
# 五、主对角线中部：special token zoom（严格版本）
# ============================================================

center = N // 2
half = SPECIAL_ZOOM_SIZE // 2

row_start = 900
row_end   = row_start + SPECIAL_ZOOM_SIZE
col_start = 900
col_end   = col_start + SPECIAL_ZOOM_SIZE

plt.figure(figsize=(4, 4))
plt.imshow(attn[row_start:row_end, col_start:col_end], cmap=CMAP)
plt.title(f"Layer {LAYER} | Special tokens (diagonal zoom)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, f"layer{LAYER:02d}_special_special_zoom.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"[保存] {out_path}")

# ============================================================
# 六、patch ↔ patch 区域（稀疏高亮）
# ============================================================

# 选取左下角一个 patch-only 区域
# 行：靠近底部（query 是 patch）
# 列：靠近左侧但避开 special（key 是 patch）
row_start = N - PATCH_ZOOM_SIZE
row_end   = N

col_start = PATCH_START_IDX * 10  # 避开 special ↔ patch
col_end   = PATCH_START_IDX * 10 + PATCH_ZOOM_SIZE

row_start = max(row_start, 0)
col_end   = min(col_end, N)

plt.figure(figsize=(4, 4))
plt.imshow(attn[row_start:row_end, col_start:col_end], cmap=CMAP)
plt.title(f"Layer {LAYER} | Patch ↔ Patch (zoom)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, f"layer{LAYER:02d}_patch_patch_zoom.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"[保存] {out_path}")

print("\n可视化完成。")
