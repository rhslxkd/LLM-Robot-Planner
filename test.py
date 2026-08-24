import matplotlib.image as mpimg
import numpy as np

image_path = "data/oracle_scene_R001/neural_astar/overlay_solo.png"
arr = mpimg.imread(image_path)
if arr.dtype != np.uint8:
    arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
red_mask = (r > 180) & (g < 120) & (b < 120)

robot_pos = (421.0, 540.0); scale = 150.0
rx, ry = robot_pos
p_a = (3.97, 1.69)
p_b = (4.56, 1.35)

h, w = red_mask.shape
print("image size (w,h):", w, h)

dist = (((p_b[0]-p_a[0])*scale)**2 + ((p_b[1]-p_a[1])*scale)**2) ** 0.5
n = max(2, int(dist))
hits = []
for k in range(n + 1):
    t = k / n
    wx = p_a[0] + (p_b[0]-p_a[0]) * t
    wy = p_a[1] + (p_b[1]-p_a[1]) * t
    px, py = rx + wx*scale, ry - wy*scale
    xi, yi = int(round(px)), int(round(py))
    in_bounds = 0 <= yi < h and 0 <= xi < w
    if in_bounds and red_mask[yi, xi]:
        hits.append((xi, yi, tuple(int(v) for v in arr[yi, xi][:3])))
print("total samples:", n+1, "red hits:", len(hits))
print("sample hits (up to 15):", hits[:15])

# 비교용: 확실한 벽 픽셀 색상도 하나 찍어보기 (row540~541, col645 근방 -- 예전에 찾은 실제 벽)
print("known wall pixel color @ (645,540):", tuple(int(v) for v in arr[540, 645][:3]) if 540 < h and 645 < w else "out of bounds")