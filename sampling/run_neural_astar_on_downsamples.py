"""
run_neural_astar_on_downsamples.py -- 아까 저장한 3개의 32x32 다운샘플 이미지
(NEAREST/BILINEAR/LANCZOS)에 각각 Neural A*를 돌려서 path 뽑고,
원본 oracle.png 위에 겹쳐서 비교.
"""
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

MAP_SIZE = 32
CKPT_DIR = "neural-astar/model/mazes_032_moore_c8/lightning_logs/"
ORACLE_PATH = "data/oracle_scene_D/oracle.png"
DS_DIR = "data/oracle_scene_D/neural_A*"
METHODS = ["NEAREST", "BILINEAR", "LANCZOS"]

orig = Image.open(ORACLE_PATH).convert("RGB")
W, H = orig.size
print(f"[oracle.png] 원본 해상도: {W}x{H}")

ROBOT_PX = (420.0, 540.0)
PPM = 90.0

def world_to_full_px(wx, wy):
    rpx, rpy = ROBOT_PX
    return rpx + wx * PPM, rpy - wy * PPM

def full_px_to_grid32(px, py):
    return px / W * MAP_SIZE, py / H * MAP_SIZE

def grid32_to_full_px(gx, gy):
    return gx / MAP_SIZE * W, gy / MAP_SIZE * H

sx_f, sy_f = full_px_to_grid32(*world_to_full_px(0.0, 0.0))
gx_f, gy_f = full_px_to_grid32(*world_to_full_px(6.5, 0.0))
sx, sy, gx, gy = int(round(sx_f)), int(round(sy_f)), int(round(gx_f)), int(round(gy_f))
print(f"[32x32] start px=({sx},{sy}) goal px=({gx},{gy})")

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
            break
        cands.sort(key=lambda p: (p[0]-goal_px[0])**2 + (p[1]-goal_px[1])**2)
        current = cands[0]
        visited.add(current)
        path.append(current)
    return path

device = "cpu"
neural_astar = NeuralAstar(encoder_arch="CNN").to(device)
neural_astar.load_state_dict(load_from_ptl_checkpoint(CKPT_DIR))
neural_astar.eval()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for col, name in enumerate(METHODS):
    ax_up = axes[0, col]    # 업스케일링 후 (원본 위)
    ax_raw = axes[1, col]   # 업스케일링 전 (32x32 그대로)

    ds_path = f"{DS_DIR}/downsample_32_{name}.png"
    small = Image.open(ds_path).convert("L")
    arr = np.array(small).astype(np.float32) / 255.0
    map_design = (arr > 0.5).astype(np.float32)

    start_ok = map_design[sy, sx] == 1.0
    goal_ok = map_design[gy, gx] == 1.0
    print(f"[{name}] start 통행가능={start_ok}, goal 통행가능={goal_ok}")

    ax_up.set_title(f"{name} (업스케일링 후)")
    ax_up.imshow(orig)
    ax_raw.set_title(f"{name} (32x32 원본)")
    ax_raw.imshow(small, cmap="gray")

    if not (start_ok and goal_ok):
        print(f"  -> {name}: 통로 소실로 탐색 불가")
        ax_up.text(W/2, H/2, "탐색 불가", color="yellow", ha="center", fontsize=14)
        ax_up.axis("off"); ax_raw.axis("off")
        continue

    map_designs = torch.tensor(map_design).float().unsqueeze(0).unsqueeze(0)
    start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32); start_map[sy, sx] = 1.0
    goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32); goal_map[gy, gx] = 1.0
    start_maps = torch.tensor(start_map).float().unsqueeze(0).unsqueeze(0)
    goal_maps = torch.tensor(goal_map).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = neural_astar(map_designs, start_maps, goal_maps)
    path_mask = out.paths[0, 0].numpy() > 0.5

    ordered = extract_ordered_path(path_mask, (sx, sy), (gx, gy))
    reached = ordered[-1] == (gx, gy)
    print(f"  -> {name}: waypoint {len(ordered)}개, goal 도달={reached}")

    # 업스케일링 버전 (원본 해상도 좌표로 변환)
    full_pts = [grid32_to_full_px(px, py) for px, py in ordered]
    xs_up, ys_up = zip(*full_pts)
    ax_up.plot(xs_up, ys_up, color="lime", linewidth=2)
    ax_up.axis("off")

    # 업스케일링 전 버전 (32x32 픽셀 좌표 그대로)
    xs_raw, ys_raw = zip(*ordered)
    ax_raw.plot(xs_raw, ys_raw, color="lime", linewidth=1.5)
    ax_raw.axis("off")

plt.tight_layout()
out_compare = f"{DS_DIR}/path_comparison.png"
plt.savefig(out_compare, dpi=120)
print(f"저장: {out_compare}")