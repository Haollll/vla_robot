# generate_fake_depth.py
import numpy as np
from PIL import Image

# 讀 RGB 圖
rgb = Image.open("test_photo.jpeg")

w, h = rgb.size

# 假設桌面距離 0.5m
depth = np.ones((h, w), dtype="float32") * 0.5

# 中央物體比較近
depth[h//3:2*h//3, w//3:2*w//3] = 0.4

np.save("test_depth.npy", depth)

print(f"Depth shape: {depth.shape}, saved.")