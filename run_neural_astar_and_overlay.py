"""
run_neural_astar_and_overlay.py -- 1,2단계: Neural A* 실행 + 실제 oracle 이미지 위에
경로 오버레이. (3단계 Coordinate 프롬프트 수정은 별도, 아직 미포함)
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines"))
from occupancy_grid import GridSpec  # 실제 oracle 이미지 좌표계 (150px/m, 기존 코드 재사용)

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

MAP_SIZE = 32
CKPT_DIR = "neural-astar/model/mazes_032_moore_c8/lightning_logs/"
ORACLE_IMAGE_PATH = "data/oracle_scene_D/oracle.png"  # 없으면 경로 맞게 수정
OUT_PATH = "neural_astar_path_on_oracle.png"

# ---- 1단계: 우리 planning grid (스크린샷에서 역산한 real-scene 기하) ----
X_MIN, X_MAX, Y_MIN, Y_MAX = -1.0, 7.0, -4.0, 4.0
PPM = MAP_SIZE / (X_MAX - X_MIN)
ROBOT_PX = (-X_MIN * PPM, Y_MAX * PPM)

def world_to_grid_px(wx, wy):
    rpx, rpy = ROBOT_PX
    return rpx + wx * PPM, rpy - wy * PPM

def grid_px_to_world(u, v):
    rpx, rpy = ROBOT_PX
    return (u - rpx) / PPM, (rpy - v) / PPM

def rasterize_line(grid, p0, p1, thickness_px=1):
    u0, v0 = p0; u1, v1 = p1
    h, w = grid.shape
    length = max(int(round(np.hypot(u1 - u0, v1 - v0))), 1)
    n = length * 4
    us = np.linspace(u0, u1, n); vs = np.linspace(v0, v1, n)
    half = thickness_px / 2.0
    for du in (np.arange(-half, half + 1) if half > 0 else [0]):
        for dv in (np.arange(-half, half + 1) if half > 0 else [0]):
            ui = np.clip((us + du).round().astype(int), 0, w - 1)
            vi = np.clip((vs + dv).round().astype(int), 0, h - 1)
            grid[vi, ui] = True

grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
rasterize_line(grid, world_to_grid_px(0.79, 0.44), world_to_grid_px(6.04, 0.44), thickness_px=1)
rasterize_line(grid, world_to_grid_px(0.79, -0.43), world_to_grid_px(6.04, -0.43), thickness_px=1)
map_design = (1.0 - grid.astype(np.float32))

sx, sy = (int(round(c)) for c in world_to_grid_px(0.0, 0.0))
gx, gy = (int(round(c)) for c in world_to_grid_px(6.5, 0.0))
assert map_design[sy, sx] == 1.0, "start 막힘"
assert map_design[gy, gx] == 1.0, "goal 막힘"

start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32); start_map[sy, sx] = 1.0
goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32); goal_map[gy, gx] = 1.0

map_designs = torch.tensor(map_design).float().unsqueeze(0).unsqueeze(0)
start_maps = torch.tensor(start_map).float().unsqueeze(0).unsqueeze(0)
goal_maps = torch.tensor(goal_map).float().unsqueeze(0).unsqueeze(0)

device = "cpu"  # torch==1.12.1이 RTX 50계열 커널 없음
neural_astar = NeuralAstar(encoder_arch="CNN").to(device)
neural_astar.load_state_dict(load_from_ptl_checkpoint(CKPT_DIR))
neural_astar.eval()

with torch.no_grad():
    na_out = neural_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device))

# ---- 2단계 준비: path mask를 순서 있는 waypoint 시퀀스로 변환 ----
path_mask = na_out.paths[0, 0].cpu().numpy() > 0.5
print(f"[neural_astar] path mask 픽셀 수: {path_mask.sum()}")

def extract_ordered_path(mask, start_px, goal_px):
    h, w = mask.shape
    visited = {start_px}
    path = [start_px]
    current = start_px
    neigh8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while current != goal_px and len(path) <= h * w:
        cx, cy = current
        cands = [(cx+dx, cy+dy) for dx, dy in neigh8
                  if 0 <= cx+dx < w and 0 <= cy+dy < h
                  and (cx+dx, cy+dy) not in visited and mask[cy+dy, cx+dx]]
        if not cands:
            print("[extract_ordered_path] ⚠️ 경로가 끊김 (dead end) -- goal 도달 실패")
            break
        cands.sort(key=lambda p: (p[0]-goal_px[0])**2 + (p[1]-goal_px[1])**2)
        current = cands[0]
        visited.add(current)
        path.append(current)
    return path

ordered_path_px = extract_ordered_path(path_mask, (sx, sy), (gx, gy))
print(f"[neural_astar] 순서 있는 waypoint {len(ordered_path_px)}개 추출, "
      f"goal 도달: {ordered_path_px[-1] == (gx, gy)}")

# world 좌표로도 변환 (나중에 courtroom/DIAL-MPC에 바로 쓸 수 있게)
ordered_path_world = [grid_px_to_world(u, v) for u, v in ordered_path_px]
for i, (wx, wy) in enumerate(ordered_path_world[::3]):  # 3개마다 하나씩만 출력
    print(f"  wp{i*3}: ({wx:.2f}, {wy:.2f})")

# ---- 2단계: 실제 oracle 이미지 위에 오버레이 ----
assert os.path.exists(ORACLE_IMAGE_PATH), (
    f"oracle 이미지 없음: {ORACLE_IMAGE_PATH} -- 경로를 실제 위치로 수정할 것 "
    f"(Table 4 실험 때 만들어진 data/oracle_scene_D/oracle.png 등)"
)
oracle_spec = GridSpec.matching_oracle_image()  # 기존 코드 재사용, 150px/m
img = Image.open(ORACLE_IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

oracle_px_path = [oracle_spec.world_to_pixel(wx, wy) for wx, wy in ordered_path_world]
for i in range(len(oracle_px_path) - 1):
    draw.line([oracle_px_path[i], oracle_px_path[i+1]], fill=(230, 25, 25), width=4)
for u, v in oracle_px_path[::2]:  # 2개마다 점 하나 (너무 빽빽하지 않게)
    r = 6
    draw.ellipse([u-r, v-r, u+r, v+r], fill=(255, 230, 0))

img.save(OUT_PATH)
print(f"✅ 저장: {OUT_PATH}")
