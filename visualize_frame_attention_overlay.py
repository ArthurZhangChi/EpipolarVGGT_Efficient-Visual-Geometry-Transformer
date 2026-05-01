import os
import glob
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 0) 你只需要改这些变量
# ============================================================

# 需要可视化的层（循环所有层/多层）
TARGET_LAYERS = [23]

# attention 文件目录：里面应该有 attn_layerXX_*.pt
# ATTN_DIR = "outputs/token_attention/scene1_DTU/add_epipolar_band/single_layer/soft_band/layer_22_bw42_soft_band"   # 改成你的
# ATTN_DIR = "outputs/token_attention/scene1_DTU/epipolar_band/layer_10_to_15_bw42"
ATTN_DIR = "outputs/token_attention/scene3_7Scenes/two_layers/layer_22_to_23_bw42"

# 输入帧原图目录（按顺序读取前 NUM_VIEWS 张）
SCENE_DIR  = "datasets/scene3_7Scenes"                # 改成你的（放图片的目录）

# 7-Scenes 特定：要加载的 frame id 列表
FRAME_IDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# 输出目录
OUT_DIR = ATTN_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# 视图数（必须与你跑 attention 时的帧数一致）
NUM_VIEWS = 10

# 每帧 special token 数：1 camera + 4 register = 5（你要求固定为 5）
NUM_SPECIAL_PER_VIEW = 5

# patch_size（dinov2_vitl14 默认 14；如果你不是 14 就改）
PATCH_SIZE = 14

# 目标帧（0-based）
TARGET_VIEW = 0

# 你在目标帧上点一个位置（y,x），脚本会映射到对应 patch token
QUERY_POS = (900, 800)

# 叠加透明度
ALPHA = 0.55

# 热力的动态范围压缩："none" / "p99" / "log"
SCALE_MODE = "none"
LOG_EPS = 1e-12

# 如果你不想自动推断 patch 网格，就手动填（例如 28,37）
PATCH_H = None
PATCH_W = None

# 只可视化 query->(其它帧 patch keys)，忽略 special keys
ONLY_PATCH_KEYS = True

# ============================================================
# 1) 读图
# ============================================================

def load_images(scene_dir, num_views):
    img_paths = sorted([
        p for p in glob.glob(os.path.join(scene_dir, "*"))
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if len(img_paths) < num_views:
        raise ValueError(f"SCENE_DIR 图片数量 {len(img_paths)} < NUM_VIEWS={num_views}")
    imgs = []
    for p in img_paths[:num_views]:
        im = Image.open(p).convert("RGB")
        imgs.append(np.asarray(im).astype(np.float32) / 255.0)
    return imgs, img_paths[:num_views]

def load_7scenes_views(scene_dir, frame_ids):
    images = []
    depth_paths = []
    pose_paths = []

    for fid in frame_ids:
        name = f"frame-{fid:06d}"

        rgb_path   = os.path.join(scene_dir, f"{name}.color.png")
        depth_path = os.path.join(scene_dir, f"{name}.depth.png")
        pose_path  = os.path.join(scene_dir, f"{name}.pose.txt")

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(rgb_path)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(depth_path)
        if not os.path.exists(pose_path):
            raise FileNotFoundError(pose_path)

        # RGB
        img = Image.open(rgb_path).convert("RGB")
        img = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
        images.append(img)

        depth_paths.append(depth_path)
        pose_paths.append(pose_path)

    return images, depth_paths, pose_paths

# ============================================================
# 2) patch 网格推断
# ============================================================

def factor_pairs(n):
    pairs = []
    for a in range(1, int(math.sqrt(n)) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
    return pairs

def infer_patch_hw(P, H, W):
    # 选 hp*wp=P 且 hp/wp 接近 H/W 的因子对
    target_ratio = H / max(W, 1e-6)
    best = None
    best_err = 1e9
    for hp, wp in factor_pairs(P):
        ratio = hp / max(wp, 1e-6)
        err = abs(math.log((ratio / target_ratio) + 1e-12))
        if err < best_err:
            best_err = err
            best = (hp, wp)
    return best

def pos_to_patch_index(y, x, H, W, Hp, Wp):
    y0, x0 = y, x

    # 先 clamp
    y = float(np.clip(y, 0, H - 1))
    x = float(np.clip(x, 0, W - 1))

    # 计算连续坐标（不取整）
    fy = y / (H - 1) * Hp
    fx = x / (W - 1) * Wp

    py = int(np.floor(fy))
    px = int(np.floor(fx))

    py = int(np.clip(py, 0, Hp - 1))
    px = int(np.clip(px, 0, Wp - 1))

    patch_idx = py * Wp + px
    return patch_idx, (py, px)

# ============================================================
# 3) 归一化 + 上采样
# ============================================================

def normalize_heat(x, mode="p99"):
    x = x.astype(np.float32)
    x = np.maximum(x, 0.0)

    if mode == "none":
        mx = float(x.max())
        return x / (mx + 1e-12)

    if mode == "log":
        x = np.log(x + LOG_EPS)
        mn, mx = float(x.min()), float(x.max())
        return (x - mn) / (mx - mn + 1e-12)

    if mode == "p99":
        p = float(np.percentile(x, 99.0))
        p = max(p, 1e-12)
        return np.clip(x / p, 0.0, 1.0)

    raise ValueError(f"未知 SCALE_MODE: {mode}")

def upsample_heat(heat_hw, H, W):
    im = Image.fromarray((heat_hw * 255).astype(np.uint8))
    im = im.resize((W, H), resample=Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0

# ============================================================
# 4) token 索引：假设 tokens 按 view 拼接
# ============================================================

def view_patch_range(view_id, T, patch_start_idx, P):
    start = view_id * T + patch_start_idx
    end = start + P
    return start, end

def global_index_of_view_patch(view_id, patch_idx, T, patch_start_idx):
    return view_id * T + patch_start_idx + patch_idx

# ============================================================
# 5) 找 attention 文件
# ============================================================

def find_attn_file(layer):
    cands = sorted(glob.glob(os.path.join(ATTN_DIR, f"attn_layer{layer:02d}_*.pt")))
    if len(cands) == 0:
        raise FileNotFoundError(f"找不到 layer={layer} 的 attention 文件：{ATTN_DIR}")
    return cands[0]

# ============================================================
# 6) 主流程
# ============================================================

def main():
    # imgs, img_paths = load_images(SCENE_DIR, NUM_VIEWS)
    imgs, depth_paths, pose_paths = load_7scenes_views(SCENE_DIR, FRAME_IDS)
    H, W = imgs[0].shape[:2]
    
    print("H,W =", H, W, "QUERY_POS=", QUERY_POS)

    # 先读一层的 attention 来推断 N/T
    probe_path = find_attn_file(TARGET_LAYERS[0])
    A0 = torch.load(probe_path, map_location="cpu")
    if A0.ndim == 3:
        A0 = A0[0]
    N = int(A0.shape[0])

    if N % NUM_VIEWS != 0:
        raise ValueError(f"N={N} 不能被 NUM_VIEWS={NUM_VIEWS} 整除，检查 token 拼接假设")

    T = N // NUM_VIEWS

    # ============================================================
    # 【核心修改】不做启发式推断：强制每帧 special=5
    # ============================================================
    patch_start_idx = NUM_SPECIAL_PER_VIEW
    P = T - patch_start_idx
    if P <= 0:
        raise ValueError(f"T={T}, patch_start_idx={patch_start_idx} 导致 P={P} 不合理，请检查 NUM_SPECIAL_PER_VIEW 或 N/S")

    # 推断 patch 网格
    if PATCH_H is not None and PATCH_W is not None:
        Hp, Wp = PATCH_H, PATCH_W
        if Hp * Wp != P:
            raise ValueError(f"你手动设置 PATCH_H*PATCH_W={Hp*Wp} != 推得 P={P}，请修正")
    else:
        best_hw = infer_patch_hw(P, H, W)
        if best_hw is None:
            raise ValueError("无法因子分解 P 来推断 patch 网格，请手动设置 PATCH_H/PATCH_W")
        Hp, Wp = best_hw
        if Hp * Wp != P:
            raise ValueError(f"推断得到 Hp*Wp={Hp*Wp} != P={P}，请手动设置 PATCH_H/PATCH_W")

    print(f"[推断] N={N}, NUM_VIEWS={NUM_VIEWS}, 每帧T={T}")
    print(f"[设定] 每帧 special={NUM_SPECIAL_PER_VIEW} => patch_start_idx={patch_start_idx}")
    print(f"[推断] 每帧 patch tokens P={P}, patch_grid={Hp}x{Wp}")

    # 选 query patch
    y, x = QUERY_POS
    patch_idx, (py, px) = pos_to_patch_index(y, x, H, W, Hp, Wp)
    q_global = global_index_of_view_patch(TARGET_VIEW, patch_idx, T, patch_start_idx)

    # 每层一个输出子目录
    for layer in TARGET_LAYERS:
        layer_dir = os.path.join(OUT_DIR, f"layer{layer:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        attn_path = find_attn_file(layer)
        A = torch.load(attn_path, map_location="cpu")
        if A.ndim == 3:
            A = A[0]
        A = A.numpy()

        if A.shape[0] != N:
            raise ValueError(f"layer{layer} 的 N={A.shape[0]} 与 probe N={N} 不一致，检查输入帧/保存文件")

        # 目标帧标星图（每层都存一份，文件名带 layer）
        H_img, W_img = imgs[TARGET_VIEW].shape[:2]
        star_y = (py + 0.5) / Hp * H_img
        star_x = (px + 0.5) / Wp * W_img
        fig = plt.figure(figsize=(6, 4.5))
        ax = plt.gca()
        ax.imshow(imgs[TARGET_VIEW])
        ax.scatter([star_x], [star_y], s=120, marker="*", c="red")
        ax.set_title(f"Target view {TARGET_VIEW} | Layer {layer} | query patch *")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out_mark = os.path.join(layer_dir, f"target_view{TARGET_VIEW}_layer{layer:02d}_pos{QUERY_POS}.png")
        fig.savefig(out_mark, dpi=200)
        plt.close(fig)

        # 取 query 行：行 = query 给出的注意力分布（query -> keys）
        row = A[q_global, :]  # [N]

        for j in range(NUM_VIEWS):
            if j == TARGET_VIEW:
                continue

            if ONLY_PATCH_KEYS:
                k_start, k_end = view_patch_range(j, T, patch_start_idx, P)
                heat_1d = row[k_start:k_end]  # [P]
                heat_1d = row[k_start:k_end].astype(np.float32)
                heat_1d = heat_1d / (heat_1d.sum() + 1e-12)   # 关键：view内条件归一化
                heat_hw = heat_1d.reshape(Hp, Wp)
            else:
                # 看该帧全部 keys（含 special），但可视化仍只画 patch 网格
                k_start, k_end = j * T, (j + 1) * T
                heat_1d = row[k_start:k_end]
                heat_hw = heat_1d[patch_start_idx:].reshape(Hp, Wp)

            heat_hw = normalize_heat(heat_hw, SCALE_MODE)
            heat_up = upsample_heat(heat_hw, H, W)

            fig = plt.figure(figsize=(6, 4.5))
            ax = plt.gca()
            ax.imshow(imgs[j])
            ax.imshow(heat_up, alpha=ALPHA)
            ax.set_title(f"Layer {layer} | q=View{TARGET_VIEW} patch({py},{px}) -> View{j}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()

            out_path = os.path.join(
                layer_dir,
                f"layer{layer:02d}_qView{TARGET_VIEW}_to_view{j}_pos{QUERY_POS}.png"
            )
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        print(f"[完成] Layer {layer:02d} overlays 已输出到：{layer_dir}")

    print("\n全部层处理完成。")

if __name__ == "__main__":
    main()
