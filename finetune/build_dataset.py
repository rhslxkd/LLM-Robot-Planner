"""
100개 랜덤배치 씬 -> Neural A* fine-tuning용 (map, start, goal, opt_traj) npz 캐시 빌드.
run_neural_astar_step.py의 좌표변환(full_to_grid)/그리드 로직을 그대로 재사용해서
추론 파이프라인과 좌표계가 절대 어긋나지 않도록 함.
"""
import sys, os, csv, json
from collections import Counter
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import run_neural_astar_step as nas  # GRID, ROBOT_PX, PPM, full_to_grid, CKPT_PATH 재사용

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "data/random_batch_manifest_v2.csv")
VARIANT = "waypoint_gen_v1"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache/dataset.npz")

GRID = nas.GRID
ROBOT_PX = nas.ROBOT_PX
PPM = nas.PPM
full_to_grid = nas.full_to_grid


def build_map_design(oracle_path):
    img = Image.open(oracle_path).convert("RGB")
    w, h = img.size
    arr = np.array(img)
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    red_mask = (r > 180) & (g < 120) & (b < 120)
    cell_h, cell_w = h / GRID, w / GRID
    obs = np.zeros((GRID, GRID), dtype=np.float32)
    for gy in range(GRID):
        y0, y1 = int(gy * cell_h), int((gy + 1) * cell_h)
        for gx in range(GRID):
            x0, x1 = int(gx * cell_w), int((gx + 1) * cell_w)
            obs[gy, gx] = 1.0 if red_mask[y0:y1, x0:x1].any() else 0.0
    return (1.0 - obs).astype(np.float32), w, h  # 1=통행가능 (Neural A* 컨벤션)


def world_to_grid(x, y, w, h):
    px = ROBOT_PX[0] + x * PPM
    py = ROBOT_PX[1] - y * PPM
    gx, gy = full_to_grid(px, py, w, h)
    gx = min(max(int(round(gx)), 0), GRID - 1)
    gy = min(max(int(round(gy)), 0), GRID - 1)
    return gx, gy


def onehot(gx, gy):
    m = np.zeros((GRID, GRID), dtype=np.float32)
    m[gy, gx] = 1.0
    return m


def bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


def rasterize_path(grid_pts):
    mask = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(len(grid_pts) - 1):
        for gx, gy in bresenham(*grid_pts[i], *grid_pts[i + 1]):
            mask[gy, gx] = 1.0
    if grid_pts:
        gx, gy = grid_pts[-1]
        mask[gy, gx] = 1.0
    return mask


def main():
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    included, skipped = [], []
    map_list, start_list, goal_list, traj_list, scene_list = [], [], [], [], []

    for row in rows:
        scene = row["scene"]
        if row.get("generator_passed") != "True" or row.get("dial_mpc_ok") != "True":
            skipped.append((scene, "gate_failed"))
            continue
        oracle_path = os.path.join(REPO_ROOT, f"data/{scene}/oracle.png")
        path_file = os.path.join(REPO_ROOT, f"data/{scene}/{VARIANT}/last_judged_path.json")
        if not (os.path.exists(oracle_path) and os.path.exists(path_file)):
            skipped.append((scene, "missing_file"))
            continue

        map_design, w, h = build_map_design(oracle_path)
        with open(path_file) as f:
            coords = json.load(f)
        if len(coords) < 2:
            skipped.append((scene, "path_too_short"))
            continue

        goal_x, goal_y = float(row["goal_x"]), float(row["goal_y"])
        start_gx, start_gy = world_to_grid(0.0, 0.0, w, h)
        goal_gx, goal_gy = world_to_grid(goal_x, goal_y, w, h)
        grid_pts = [world_to_grid(c["x"], c["y"], w, h) for c in coords]
        opt_traj = rasterize_path(grid_pts)

        map_list.append(map_design)
        start_list.append(onehot(start_gx, start_gy))
        goal_list.append(onehot(goal_gx, goal_gy))
        traj_list.append(opt_traj)
        scene_list.append(scene)
        included.append(scene)

    print(f"포함: {len(included)}개, 스킵: {len(skipped)}개")
    if skipped:
        print(f"  스킵 사유: {dict(Counter(r for _, r in skipped))}")
    if not included:
        print("ERROR: 포함된 샘플이 0개 - manifest/경로 확인 필요")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        map_designs=np.stack(map_list)[:, None, :, :],
        start_maps=np.stack(start_list)[:, None, :, :],
        goal_maps=np.stack(goal_list)[:, None, :, :],
        opt_trajs=np.stack(traj_list)[:, None, :, :],
        scenes=np.array(scene_list),
    )
    print(f"저장 완료: {OUT_PATH} ({len(included)} samples, shape={map_list[0].shape})")


if __name__ == "__main__":
    main()
