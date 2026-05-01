import os
import random
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# ---------------------------
# 1) 固定随机性
# ---------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)


# ---------------------------
# 2) 设备 & 精度
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


# ---------------------------
# 3) 配置：换场景只改这里
# ---------------------------
SCENE_DIR = "examples/kitchen/images"   # TODO: 改成你的新场景路径（包含 jpg/png）
SCENE_TAG = "kitchen"                  # TODO: 改成新场景名字（用于输出文件名）
OUT_DIR   = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------
# 4) 读取场景所有图像
# ---------------------------
image_names = sorted([
    os.path.join(SCENE_DIR, fname)
    for fname in os.listdir(SCENE_DIR)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
])

assert len(image_names) > 0, f"No images found in {SCENE_DIR}"

print(f"Loaded {len(image_names)} images from scene: {SCENE_TAG}")
for name in image_names:
    print("  ", name)


# ---------------------------
# 5) 加载并预处理图像
# ---------------------------
images = load_and_preprocess_images(image_names).to(device)


# ---------------------------
# 6) 加载 VGGT 模型（不做 attention hook）
# ---------------------------
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

# 如果你之前改过 sparse 的 aggregator，确保 baseline 时关闭 sparse（推荐显式写上）
if hasattr(model, "aggregator") and hasattr(model.aggregator, "enable_sparse_global"):
    model.aggregator.enable_sparse_global = False


# ---------------------------
# 7) Inference（baseline）
# ---------------------------
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images)


# ---------------------------
# 8) 简单 sanity check
# ---------------------------
print("\nPrediction keys:")
for k, v in predictions.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: type={type(v)}")


# ---------------------------
# 9) 保存 baseline 结果
# ---------------------------
out_path = os.path.join(OUT_DIR, f"{SCENE_TAG}_baseline_predictions.pt")
torch.save(predictions, out_path)
print(f"\n[Saved] {out_path}")
print("\nBaseline inference finished.")
