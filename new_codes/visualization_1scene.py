import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# 针对不同场景进行可视化
pred_path = "outputs/kitchen_baseline_predictions.pt"

pred = torch.load(pred_path, map_location="cpu")

print(pred.keys())

# 可视化一张 RGB（确认输入）
img = pred["images"][0, 0]   # 第 1 个 view
img = img.permute(1, 2, 0)   # C,H,W → H,W,C
plt.imshow(img)
plt.title("Input image (view 0)")
plt.axis("off")
plt.show()

# 可视化 depth（你一定要做的第一张图）
depth = pred["depth"][0, 0, :, :, 0]
plt.imshow(depth, cmap="inferno")
plt.colorbar()
plt.title("Predicted depth (view 0)")
plt.axis("off")
plt.show()

# 可视化 depth confidence
conf = pred["depth_conf"][0, 0]
plt.imshow(conf, cmap="viridis")
plt.colorbar()
plt.title("Depth confidence (view 0)")
plt.axis("off")
plt.show()

# 可视化 world points
"""
wp = pred["world_points"][0, 0]           # (H, W, 3)
conf = pred["world_points_conf"][0, 0]    # (H, W)

mask = conf > 0.5
pts = wp[mask].reshape(-1, 3).numpy()

def write_ply_xyz(path, xyz):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in xyz:
            f.write(f"{x} {y} {z}\n")

write_ply_xyz("room_world_points_view0.ply", pts)
print(f"Saved {pts.shape[0]} points to room_world_points_view0.ply")
"""

wp = pred["world_points"][0, 0]        # (H,W,3)
conf = pred["world_points_conf"][0, 0]
img = pred["images"][0, 0]             # (3,H,W)

img = img.permute(1, 2, 0)             # (H,W,3)

mask = conf > 0.5

pts = wp[mask].reshape(-1, 3).numpy()
cols = img[mask].reshape(-1, 3).numpy()

def write_ply_xyzrgb(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            r = int(max(0, min(255, r * 255)))
            g = int(max(0, min(255, g * 255)))
            b = int(max(0, min(255, b * 255)))
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

write_ply_xyzrgb("room_view0_rgb.ply", pts, cols)