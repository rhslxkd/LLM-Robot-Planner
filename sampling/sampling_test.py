"""
downsample_test.py -- 실제 data/oracle_scene_D/oracle.png를 32x32로
다운샘플링(NEAREST/BILINEAR/LANCZOS 3가지 비교)만 수행. Neural A*는 안 돌림.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

MAP_SIZE = 32
ORACLE_PATH = "data/oracle_scene_D/oracle.png"
OUT_DIR = "data/oracle_scene_D/neural_A*"
os.makedirs(OUT_DIR, exist_ok=True)   # OUT_DIR 정의한 줄 바로 다음에 추가

img = Image.open(ORACLE_PATH).convert("RGB")
W, H = img.size
print(f"[oracle.png] 실제 해상도: {W}x{H}")

arr_full = np.array(img)
r, g, b = arr_full[...,0].astype(int), arr_full[...,1].astype(int), arr_full[...,2].astype(int)
red_mask_full = (r > 180) & (g < 120) & (b < 120)          # 벽(장애물) = 빨간 픽셀
obstacle_gray = Image.fromarray((~red_mask_full * 255).astype(np.uint8))  # 255=free, 0=장애물

methods = {"NEAREST": Image.NEAREST, "BILINEAR": Image.BILINEAR, "LANCZOS": Image.LANCZOS}
downsampled = {}
for name, method in methods.items():
    small = obstacle_gray.resize((MAP_SIZE, MAP_SIZE), method)
    downsampled[name] = small
    out_path = f"{OUT_DIR}/downsample_32_{name}.png"
    small.save(out_path)
    print(f"저장: {out_path}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, small) in zip(axes, downsampled.items()):
    ax.set_title(f"{name} (32x32)")
    ax.imshow(small, cmap="gray")
    ax.axis("off")
plt.tight_layout()
compare_path = f"{OUT_DIR}/downsample_comparison.png"
plt.savefig(compare_path, dpi=120)
print(f"저장: {compare_path}")