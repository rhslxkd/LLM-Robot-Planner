"""
run_neural_astar_step.py
neural-astar env 전용. run_random_batch.py(vlm_court env)에서
`conda run -n neural-astar`로 서브프로세스 호출됨.

사용: python run_neural_astar_step.py --scene oracle_scene_R000 --goal-x 5.3 --goal-y -1.2
성공 시 data/<scene>/neural_astar/overlay_solo.png 를 만들고 exit 0.
실패(목표 미도달) 시 exit 1.
"""
import os, sys, argparse, json as _json
import numpy as np
from PIL import Image
import torch

GRID = 32
ROBOT_PX = (421.0, 540.0)
PPM = 150.0
CKPT_PATH = "neural-astar/model/mazes_032_moore_c8/lightning_logs/version_0"

def full_to_grid(px, py, w, h): return px * GRID / w, py * GRID / h
def grid_to_full(gx, gy, w, h): return gx * w / GRID, gy * h / GRID

def extract_ordered_path(mask, start_px, goal_px):
    ys, xs = np.where(mask)
    pts = list(zip(xs.tolist(), ys.tolist()))
    if not pts: return []
    pts_set = set(pts); goal_px = tuple(goal_px)
    current = min(pts_set, key=lambda p: (p[0]-start_px[0])**2 + (p[1]-start_px[1])**2)
    ordered = [current]; visited = {current}
    for _ in range(len(pts_set) - 1):
        if current == goal_px: break
        remaining = pts_set - visited
        if not remaining: break
        nxt = min(remaining, key=lambda p: (p[0]-current[0])**2 + (p[1]-current[1])**2)
        ordered.append(nxt); visited.add(nxt); current = nxt
    if ordered[-1] != goal_px and goal_px in pts_set:
        ordered.append(goal_px)
    return ordered

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    args = parser.parse_args()
    scene, goal_x, goal_y = args.scene, args.goal_x, args.goal_y

    oracle_path = f"data/{scene}/oracle.png"
    out_dir = f"data/{scene}/neural_astar"
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(oracle_path).convert("RGB")
    w, h = img.size
    arr = np.array(img)
    r, g, b = arr[...,0].astype(int), arr[...,1].astype(int), arr[...,2].astype(int)
    red_mask = (r > 180) & (g < 120) & (b < 120)

    cell_h, cell_w = h / GRID, w / GRID
    obs = np.zeros((GRID, GRID), dtype=np.float32)
    for gy in range(GRID):
        y0, y1 = int(gy*cell_h), int((gy+1)*cell_h)
        for gx in range(GRID):
            x0, x1 = int(gx*cell_w), int((gx+1)*cell_w)
            obs[gy, gx] = 1.0 if red_mask[y0:y1, x0:x1].any() else 0.0

    start_full = ROBOT_PX
    goal_full = (ROBOT_PX[0] + goal_x*PPM, ROBOT_PX[1] - goal_y*PPM)
    start_grid = tuple(int(round(v)) for v in full_to_grid(*start_full, w, h))
    goal_grid = tuple(int(round(v)) for v in full_to_grid(*goal_full, w, h))
    start_grid = (min(max(start_grid[0],0),GRID-1), min(max(start_grid[1],0),GRID-1))
    goal_grid = (min(max(goal_grid[0],0),GRID-1), min(max(goal_grid[1],0),GRID-1))

    start_map = np.zeros((GRID,GRID),dtype=np.float32); start_map[start_grid[1],start_grid[0]]=1.0
    goal_map = np.zeros((GRID,GRID),dtype=np.float32); goal_map[goal_grid[1],goal_grid[0]]=1.0
    map_tensor = torch.tensor(1.0-obs)[None,None].float()
    start_tensor = torch.tensor(start_map)[None,None].float()
    goal_tensor = torch.tensor(goal_map)[None,None].float()

    from neural_astar.planner import NeuralAstar
    from neural_astar.utils.training import load_from_ptl_checkpoint

    model = NeuralAstar(g_ratio=0.5, encoder_arch="CNN")
    model.load_state_dict(load_from_ptl_checkpoint(CKPT_PATH))
    model.eval()

    def dump_diagnostics():
        print(f"  start_grid={start_grid} free={obs[start_grid[1],start_grid[0]]==0}")
        print(f"  goal_grid={goal_grid} free={obs[goal_grid[1],goal_grid[0]]==0}")
        print("  32x32 grid ('#'=obstacle, 'S'=start, 'G'=goal):")
        for gy in range(GRID):
            row = ""
            for gx in range(GRID):
                if (gx, gy) == start_grid:
                    row += "S"
                elif (gx, gy) == goal_grid:
                    row += "G"
                else:
                    row += "#" if obs[gy, gx] else "."
            print("  " + row)

    try:
        with torch.no_grad():
            out = model(map_tensor, start_tensor, goal_tensor)
        path_mask = out.paths[0,0].numpy() > 0.5
    except Exception as e:
        print(f"FAIL: model forward crashed ({type(e).__name__}: {e})")
        dump_diagnostics()
        sys.exit(1)

    wp_grid = extract_ordered_path(path_mask, start_grid, goal_grid)
    if not wp_grid or wp_grid[-1] != goal_grid:
        print(f"FAIL: goal not reached (wp_grid last={wp_grid[-1] if wp_grid else None}, goal_grid={goal_grid})")
        dump_diagnostics()
        sys.exit(1)

    wp_full = [grid_to_full(x,y,w,h) for x,y in wp_grid]

    path_length_px = sum(
        ((wp_full[i][0]-wp_full[i-1][0])**2 + (wp_full[i][1]-wp_full[i-1][1])**2)**0.5
        for i in range(1, len(wp_full))
    )
    path_length_m = path_length_px / PPM
    with open(os.path.join(out_dir, "path_info.json"), "w") as f:
        _json.dump({"path_length_m": path_length_m}, f)
    print(f"PATH_LENGTH_M: {path_length_m:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,8.5))
    ax.imshow(img)
    fx_, fy_ = zip(*wp_full)
    ax.plot(fx_, fy_, color="orange", linewidth=2)
    ax.scatter(*start_full, c="green", s=60, zorder=5)
    ax.scatter(*goal_full, c="red", s=60, zorder=5)
    ax.axis("off")
    overlay_path = os.path.join(out_dir, "overlay_solo.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"OK: {overlay_path}")
    sys.exit(0)

if __name__ == "__main__":
    main()