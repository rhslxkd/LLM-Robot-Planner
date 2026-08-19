"""
run_neural_astar_gratio_sweep_E.py
Scene E(3 baffle slalom)에서 g_ratio 스윕. ROBOT_PX/PPM은 렌더러 고정값 재사용.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

# ---------------- Config ----------------
ORACLE_PATH = "data/oracle_scene_R/oracle.png"
OUT_DIR = "data/oracle_scene_R/neural_A*"
CKPT_PATH = "neural-astar/model/mazes_032_moore_c8/lightning_logs/version_0"

GRID = 32
ROBOT_PX = (421.0, 540.0)   # 렌더러 고정값 (D와 동일)
PPM = 90.0                  # 렌더러 고정값 (D와 동일)
GOAL_WORLD_XY = (5.0, 0.0)  # Scene E 시나리오 목표

G_RATIOS = [0.1, 0.5, 0.9]
COLORS = {0.1: "orange", 0.5: "cyan", 0.9: "magenta"}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 1. 이미지 로드 + 장애물 마스크 ----------------
img = Image.open(ORACLE_PATH).convert("RGB")
FULL_W, FULL_H = img.size
arr = np.array(img)
r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
red_mask = (r > 180) & (g < 120) & (b < 120)

cell_h = FULL_H / GRID
cell_w = FULL_W / GRID
obs_32_arr = np.zeros((GRID, GRID), dtype=np.float32)
for gy_ in range(GRID):
    y0, y1 = int(gy_ * cell_h), int((gy_ + 1) * cell_h)
    for gx_ in range(GRID):
        x0, x1 = int(gx_ * cell_w), int((gx_ + 1) * cell_w)
        obs_32_arr[gy_, gx_] = 1.0 if red_mask[y0:y1, x0:x1].any() else 0.0

fx, fy = GRID / FULL_W, GRID / FULL_H
gx, gy = FULL_W / GRID, FULL_H / GRID
def full_to_grid(px, py):
    return px * fx, py * fy
def grid_to_full(gx_, gy_):
    return gx_ * gx, gy_ * gy

goal_full = (ROBOT_PX[0] + GOAL_WORLD_XY[0] * PPM, ROBOT_PX[1] - GOAL_WORLD_XY[1] * PPM)
start_full = ROBOT_PX

start_grid = tuple(int(round(v)) for v in full_to_grid(*start_full))
goal_grid = tuple(int(round(v)) for v in full_to_grid(*goal_full))
start_grid = (min(max(start_grid[0], 0), GRID-1), min(max(start_grid[1], 0), GRID-1))
goal_grid = (min(max(goal_grid[0], 0), GRID-1), min(max(goal_grid[1], 0), GRID-1))
print(f"start_grid={start_grid}, goal_grid={goal_grid}")

start_map = np.zeros((GRID, GRID), dtype=np.float32)
goal_map = np.zeros((GRID, GRID), dtype=np.float32)
start_map[start_grid[1], start_grid[0]] = 1.0
goal_map[goal_grid[1], goal_grid[0]] = 1.0

map_tensor = torch.tensor(1.0 - obs_32_arr)[None, None].float()
start_tensor = torch.tensor(start_map)[None, None].float()
goal_tensor = torch.tensor(goal_map)[None, None].float()

def extract_ordered_path(mask, start_px, goal_px):
    ys, xs = np.where(mask)
    pts = list(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return []
    pts_set = set(pts)
    goal_px = tuple(goal_px)
    current = min(pts_set, key=lambda p: (p[0]-start_px[0])**2 + (p[1]-start_px[1])**2)
    ordered = [current]
    visited = {current}
    for _ in range(len(pts_set) - 1):
        if current == goal_px:
            break
        remaining = pts_set - visited
        if not remaining:
            break
        nxt = min(remaining, key=lambda p: (p[0]-current[0])**2 + (p[1]-current[1])**2)
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt
    if ordered[-1] != goal_px and goal_px in pts_set:
        ordered.append(goal_px)
    return ordered

# ---------------- 2. g_ratio별로 Neural A* 실행 ----------------
device = "cpu"
results = {}
for gr in G_RATIOS:
    model = NeuralAstar(g_ratio=gr, encoder_arch="CNN")
    state_dict = load_from_ptl_checkpoint(CKPT_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        out = model(map_tensor.to(device), start_tensor.to(device), goal_tensor.to(device))

    path_mask = out.paths[0, 0].cpu().numpy() > 0.5
    waypoints_grid = extract_ordered_path(path_mask, start_grid, goal_grid)
    goal_blocked = bool(obs_32_arr[goal_grid[1], goal_grid[0]])
    reaches_goal = bool(path_mask[goal_grid[1], goal_grid[0]])
    if waypoints_grid:
        last = waypoints_grid[-1]
        dist = ((last[0]-goal_grid[0])**2 + (last[1]-goal_grid[1])**2) ** 0.5
        print(f"  [g_ratio={gr}] goal 셀 막힘={goal_blocked}, path가 goal 포함={reaches_goal}, 마지막 지점 {last} -> goal 거리 {dist:.1f}칸")
    waypoints_full = [grid_to_full(x, y) for x, y in waypoints_grid]
    results[gr] = {"grid": waypoints_grid, "full": waypoints_full}
    print(f"[g_ratio={gr}] waypoints: {len(waypoints_grid)}")
    print("=== 32x32 occupancy grid (1=장애물) ===")
    for row in obs_32_arr.astype(int):
        print("".join(str(v) for v in row))

# ---------------- 3. 시각화 ----------------
fig, ax = plt.subplots(figsize=(12, 8.5))
ax.imshow(img)
for gr in G_RATIOS:
    wf = results[gr]["full"]
    if wf:
        fx_, fy_ = zip(*wf)
        ax.plot(fx_, fy_, color=COLORS[gr], linewidth=2, label=f"g_ratio={gr}", alpha=0.8)
ax.scatter(*start_full, c="green", s=60, zorder=5)
ax.scatter(*goal_full, c="red", s=60, zorder=5)
ax.legend()
ax.set_title("Neural A* paths across g_ratio values (Scene E)")

out_path = os.path.join(OUT_DIR, "gratio_sweep_overlay_E.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"saved: {out_path}")

sets = {gr: set(results[gr]["grid"]) for gr in G_RATIOS}
for i, gr1 in enumerate(G_RATIOS):
    for gr2 in G_RATIOS[i+1:]:
        overlap = len(sets[gr1] & sets[gr2]) / max(len(sets[gr1] | sets[gr2]), 1)
        print(f"  g_ratio {gr1} vs {gr2}: IoU = {overlap:.2f}")