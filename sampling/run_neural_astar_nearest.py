"""
run_neural_astar_nearest.py
Step 1-2 최종본: oracle.png -> NEAREST 다운샘플(32x32) -> Neural A* -> 실제 이미지에 경로 오버레이
NEAREST만 사용, 파일 하나로 통합.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

# ---------------- Config ----------------
ORACLE_PATH = "data/oracle_scene_D/oracle.png"
OUT_DIR = "data/oracle_scene_D/neural_A*"
CKPT_PATH = "neural-astar/model/mazes_032_moore_c8/lightning_logs/version_0"

GRID = 32
ROBOT_PX = (420.0, 540.0)   # full-res, 확정값 (start_grid=(11,16)과 일치 확인됨)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 1. 실제 이미지 로드 + 장애물 마스크 ----------------
img = Image.open(ORACLE_PATH).convert("RGB")
FULL_W, FULL_H = img.size
arr = np.array(img)
r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
red_mask = (r > 180) & (g < 120) & (b < 120)  # 빨간 벽 픽셀

obs_img = Image.fromarray((red_mask * 255).astype(np.uint8))

# ---------------- 2. NEAREST 다운샘플 (32x32) ----------------
obs_32 = obs_img.resize((GRID, GRID), Image.NEAREST)
obs_32_arr = (np.array(obs_32) > 127).astype(np.float32)  # 1=obstacle

# 좌표 변환 스케일 (full-res <-> grid, x/y 비율 다름에 유의)
fx = GRID / FULL_W
fy = GRID / FULL_H
gx = FULL_W / GRID
gy = FULL_H / GRID

def full_to_grid(px, py):
    return px * fx, py * fy

def grid_to_full(gx_, gy_):
    return gx_ * gx, gy_ * gy

# ---------------- 3. start / goal 픽셀 (32x32 grid, 이전 실행에서 확인된 확정값) ----------------
start_grid = (11, 16)
goal_grid = (25, 16)

start_full = grid_to_full(*start_grid)
goal_full = grid_to_full(*goal_grid)

# ---------------- 4. Neural A* 입력 텐서 (encoder_input="m+") ----------------
start_map = np.zeros((GRID, GRID), dtype=np.float32)
goal_map = np.zeros((GRID, GRID), dtype=np.float32)
start_map[start_grid[1], start_grid[0]] = 1.0
goal_map[goal_grid[1], goal_grid[0]] = 1.0

map_tensor = torch.tensor(1.0 - obs_32_arr)[None, None].float()
start_tensor = torch.tensor(start_map)[None, None].float()
goal_tensor = torch.tensor(goal_map)[None, None].float()

# ---------------- 5. 모델 로드 (CPU 고정, sm_120 이슈 회피) ----------------
model = NeuralAstar(encoder_arch="CNN")
state_dict = load_from_ptl_checkpoint(CKPT_PATH)
model.load_state_dict(state_dict)
model.eval()
device = "cpu"
model.to(device)

with torch.no_grad():
    out = model(map_tensor.to(device), start_tensor.to(device), goal_tensor.to(device))

path_mask = out.paths[0, 0].cpu().numpy() > 0.5

# ---------------- 6. 경로 마스크 -> 순서있는 waypoint 시퀀스 ----------------
def extract_ordered_path(mask, start_px, goal_px):
    ys, xs = np.where(mask)
    pts = list(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return []
    pts_set = set(pts)
    current = min(pts_set, key=lambda p: (p[0]-start_px[0])**2 + (p[1]-start_px[1])**2)
    ordered = [current]
    visited = {current}
    for _ in range(len(pts_set) - 1):
        remaining = pts_set - visited
        if not remaining:
            break
        nxt = min(remaining, key=lambda p: (p[0]-current[0])**2 + (p[1]-current[1])**2)
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt
        if (current[0]-goal_px[0])**2 + (current[1]-goal_px[1])**2 <= 2:
            break
    return ordered

waypoints_grid = extract_ordered_path(path_mask, start_grid, goal_grid)
print(f"[NEAREST] waypoints (32x32 grid): {len(waypoints_grid)}")

# ---------------- 7. grid -> full-res 좌표 변환 (이미지 보간 아님, 좌표계산) ----------------
waypoints_full = [grid_to_full(x, y) for x, y in waypoints_grid]

# ---------------- 8. 시각화: 실제 oracle.png 위에 오버레이 + 32x32 원본 ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(obs_32_arr, cmap="gray_r")
if waypoints_grid:
    wx, wy = zip(*waypoints_grid)
    axes[0].plot(wx, wy, color="orange", linewidth=2)
axes[0].scatter(*start_grid, c="green", s=40, label="start")
axes[0].scatter(*goal_grid, c="red", s=40, label="goal")
axes[0].set_title("32x32 raw (NEAREST)")
axes[0].legend()

axes[1].imshow(img)
if waypoints_full:
    fx_, fy_ = zip(*waypoints_full)
    axes[1].plot(fx_, fy_, color="orange", linewidth=2)
axes[1].scatter(*start_full, c="green", s=40)
axes[1].scatter(*goal_full, c="red", s=40)
axes[1].set_title("Overlay on real oracle.png")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "nearest_path_result.png")
plt.savefig(out_path, dpi=150)
# 오버레이 패널만 따로 저장 (VLM 입력용, 축 레이블 포함)
fig2, ax2 = plt.subplots(figsize=(10, 8.5))
ax2.imshow(img)
if waypoints_full:
    fx_, fy_ = zip(*waypoints_full)
    ax2.plot(fx_, fy_, color="orange", linewidth=2)
ax2.scatter(*start_full, c="green", s=60)
ax2.scatter(*goal_full, c="red", s=60)
ax2.axis("off")
solo_path = os.path.join(OUT_DIR, "nearest_path_overlay_solo.png")
plt.savefig(solo_path, dpi=150, bbox_inches="tight")
print(f"saved (VLM용 단독 패널): {solo_path}")

# waypoints도 저장 (Step 3에서 재사용 가능)
import json
with open(os.path.join(OUT_DIR, "nearest_waypoints_full_px.json"), "w") as f:
    json.dump(waypoints_full, f)