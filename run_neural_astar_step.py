"""
run_neural_astar_step.py
neural-astar env 전용. run_random_batch.py(vlm_court env)에서
`conda run -n neural-astar`로 서브프로세스 호출됨.

Neural A* raw grid path -> world 좌표 변환 -> line-of-sight 경로 단순화
(visibility-aware shortcutting) -> 목표 waypoint 개수로 보정 -> 각 점의
실제 통로 폭(clearance_m)을 perpendicular ray-casting으로 결정론적 계산
-> courtroom에 넘길 coordinate_proposal.json + 오버레이 이미지 저장.

사용: python run_neural_astar_step.py --scene oracle_scene_R000 --goal-x 5.3 --goal-y -1.2
성공 시 data/<scene>/neural_astar/{overlay_solo.png, coordinate_proposal.json, path_info.json} 생성, exit 0.
실패(목표 미도달) 시 exit 1.
"""
import os, sys, argparse, json as _json
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_dilation

GRID = 32
ROBOT_PX = (421.0, 540.0)
PPM = 150.0
CKPT_PATH = "neural-astar/model/mazes_032_moore_c8/lightning_logs/version_0"
TARGET_STEP_M = 0.65      # 로코모션 RECOMMENDED 스텝 길이(0.6~0.7m)에 맞춤
MIN_WAYPOINTS = 6
MAX_WAYPOINTS = 25
WALL_DILATE_PX = 3        # line-of-sight 충돌체크용 마스크 팽창 폭(px). anti-aliasing으로 인한
                           # 1~2px짜리 렌더링 드롭아웃(false gap)을 막기 위함. clearance 계산에는
                           # 쓰지 않음(원본 red_mask로 정확한 값 유지).
MIN_STEP_M = 0.4
MAX_STEP_M = 1.0
MIN_WALL_DIST_M = 0.4    # 2026-08-31: 통로 전체 폭(clearance_m)과 별개로,
                         # 한쪽 벽에 치우쳐 지나가는 걸 막는 최소 편측 이격거리 기준.
CORRECTION_MAX_SHIFT_M = 0.3  # 2026-08-31: 중앙 정렬 보정 시 이동량 상한. 이게 없으면 한쪽
                         # 벽이 아주 멀 때(반대쪽이 열린 공간) shift가 무한정 커져서 원래
                         # 위치에서 엉뚱하게 먼 곳으로 waypoint가 튀는 문제가 있었음 (실측 확인).

def full_to_grid(px, py, w, h): return px * GRID / w, py * GRID / h
def grid_to_full(gx, gy, w, h): return gx * w / GRID, gy * h / GRID
def full_to_world(px, py): return (px - ROBOT_PX[0]) / PPM, (ROBOT_PX[1] - py) / PPM

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

# ---------- Stage 2: line-of-sight path simplification (visibility-aware shortcutting) ----------

def segment_is_clear(p0, p1, red_mask):
    h, w = red_mask.shape
    dist = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) ** 0.5
    n = max(2, int(dist))  # 픽셀 거리에 비례한 샘플 수 -> 한 픽셀도 안 건너뜀
    for k in range(n + 1):
        t = k / n
        x = p0[0] + (p1[0]-p0[0]) * t
        y = p0[1] + (p1[1]-p0[1]) * t
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and red_mask[yi, xi]:
            return False
    return True

def simplify_line_of_sight(points, red_mask):
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    i = 0
    n = len(points)
    while i < n - 1:
        farthest = i + 1
        for j in range(n - 1, i, -1):
            if segment_is_clear(points[i], points[j], red_mask):
                farthest = j
                break
        simplified.append(points[farthest])
        i = farthest
    return simplified

# ---------- Stage 3: 목표 waypoint 개수로 보정 (이미 안전한 segment 내부에서만 중간점 추가) ----------

def upsample_to_target(points, target_n):
    pts = list(points)
    while len(pts) < target_n:
        idx = max(range(len(pts) - 1),
                   key=lambda i: (pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)
        mx = (pts[idx][0] + pts[idx+1][0]) / 2
        my = (pts[idx][1] + pts[idx+1][1]) / 2
        pts.insert(idx + 1, (mx, my))
    return pts

def enforce_step_constraints(points, red_mask, ppm, min_step_m=MIN_STEP_M,
                              max_step_m=MAX_STEP_M, max_passes=6):
    """MIN/MAX 보폭 제약을 결정론적으로 강제. VLM 판사가 이 문제로 REJECTED를
    내리는 걸 애초에 막기 위한 후처리 (Stage 3.5)."""
    pts = list(points)
    for _ in range(max_passes):
        changed = False

        # (a) 너무 긴 구간 -> 중간점 삽입
        i = 0
        while i < len(pts) - 1:
            d = ((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2) ** 0.5 / ppm
            if d > max_step_m:
                mx = (pts[i][0] + pts[i+1][0]) / 2
                my = (pts[i][1] + pts[i+1][1]) / 2
                pts.insert(i + 1, (mx, my))
                changed = True
            i += 1

        # (b) 너무 짧은 구간 -> 뒤쪽 점 제거 (line-of-sight 안전할 때만, 시작/끝점 보존)
        i = 0
        while i < len(pts) - 1:
            d = ((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2) ** 0.5 / ppm
            remove_idx = i + 1
            if d < min_step_m and len(pts) > 2 and remove_idx != len(pts) - 1:
                prev_p, next_p = pts[remove_idx - 1], pts[remove_idx + 1]
                if segment_is_clear(prev_p, next_p, red_mask):
                    pts.pop(remove_idx)
                    changed = True
                    continue  # 인덱스 유지, 새로 당겨진 다음 구간 재검사
            i += 1

        if not changed:
            break
    return pts

def path_length_m(points, ppm):
    px_len = sum(((points[i][0]-points[i-1][0])**2 + (points[i][1]-points[i-1][1])**2) ** 0.5
                 for i in range(1, len(points)))
    return px_len / ppm

# ---------- Stage 4: 각 점의 실제 통로 폭(clearance_m) - perpendicular ray-casting ----------

def measure_corridor_width_m(points, idx, red_mask, ppm):
    """clearance_m 계산 + (0.8m 미만일 때) 통로 중앙으로 정렬한 좌표까지 함께 반환.
    2026-08-29: Prosecutor가 attempt 1부터 정확한 교정 좌표를 받도록, courtroom.py의
    compute_correction_suggestions()와 동일한 계산을 Neural A* 직후에 선제적으로 수행."""
    h, w = red_mask.shape
    if idx == 0:
        tx, ty = points[1][0]-points[0][0], points[1][1]-points[0][1]
    elif idx == len(points) - 1:
        tx, ty = points[idx][0]-points[idx-1][0], points[idx][1]-points[idx-1][1]
    else:
        tx, ty = points[idx+1][0]-points[idx-1][0], points[idx+1][1]-points[idx-1][1]
    norm = (tx**2 + ty**2) ** 0.5
    if norm < 1e-6:
        return 99.0, points[idx][0], points[idx][1]
    perp_x, perp_y = -ty/norm, tx/norm  # 경로 진행방향에 수직인 단위벡터
    x0, y0 = points[idx]
    max_range = max(h, w)

    def cast(dx, dy):
        for r in range(1, max_range):
            xi, yi = int(round(x0+dx*r)), int(round(y0+dy*r))
            if not (0 <= xi < w and 0 <= yi < h):
                return r  # 이미지 경계 벗어남 = 그쪽은 열려있다고 간주
            if red_mask[yi, xi]:
                return r
        return max_range

    d_pos = cast(perp_x, perp_y)
    d_neg = cast(-perp_x, -perp_y)
    width_m = (d_pos + d_neg) / ppm
    width_m = 99.0 if width_m > 8.0 else round(width_m, 2)  # 너무 넓으면 "개방구역"으로 표기
    near_m = round(min(d_pos, d_neg) / ppm, 2)   # 가까운 쪽 벽까지 거리 (편향 감지 + 설명용)
    far_m = round(max(d_pos, d_neg) / ppm, 2)    # 먼 쪽 벽까지 거리 (좌/우 라벨 없이 크기만)

    cap_px = CORRECTION_MAX_SHIFT_M * ppm
    shift_px = max(-cap_px, min(cap_px, (d_pos - d_neg) / 2.0))
    center_x = x0 + perp_x * shift_px
    center_y = y0 + perp_y * shift_px
    return width_m, center_x, center_y, near_m, far_m

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
    red_mask = (r > 180) & (g < 120) & (b < 120)  # 고해상도 원본 마스크(다운샘플 전)
                                                    # -> clearance 계산은 이 원본으로 정확하게
    # line-of-sight 충돌체크 전용: anti-aliasing 등으로 생기는 1~2px짜리 렌더링 드롭아웃이
    # 벽에 없는 "바늘구멍"을 만들어 직선 단순화가 벽을 관통하는 걸로 오판하는 걸 방지.
    red_mask_dilated = binary_dilation(red_mask, iterations=WALL_DILATE_PX)

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
                if (gx, gy) == start_grid: row += "S"
                elif (gx, gy) == goal_grid: row += "G"
                else: row += "#" if obs[gy, gx] else "."
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

    # ---- Stage 1: grid -> full-res pixel (원시, 촘촘한 경로) ----
    wp_full_raw = [grid_to_full(x,y,w,h) for x,y in wp_grid]

    # ---- Stage 2: line-of-sight 단순화 (팽창된 마스크 기준 -> 렌더링 노이즈로 인한 오탐 방지) ----
    wp_simplified = simplify_line_of_sight(wp_full_raw, red_mask_dilated)

    # ---- Stage 3: 목표 waypoint 개수로 보정 (안전한 segment 내부에서만 중간점 삽입) ----
    length_m = path_length_m(wp_simplified, PPM)
    target_n = max(MIN_WAYPOINTS, min(MAX_WAYPOINTS, round(length_m / TARGET_STEP_M)))
    wp_final = upsample_to_target(wp_simplified, target_n)
    wp_final = enforce_step_constraints(wp_final, red_mask_dilated, PPM)
    final_length_m = path_length_m(wp_final, PPM)

    # ---- Stage 4: 각 점의 실제 통로 폭(clearance_m), perpendicular ray-casting (원본 red_mask 사용) ----
    coordinates = []
    for idx, (px, py) in enumerate(wp_final):
        wx, wy = full_to_world(px, py)
        clearance, cx_px, cy_px, near_m, far_m = measure_corridor_width_m(wp_final, idx, red_mask, PPM)
        entry = {"x": round(wx, 2), "y": round(wy, 2), "clearance_m": clearance}
        if clearance < 0.8 or near_m < MIN_WALL_DIST_M:
            cwx, cwy = full_to_world(cx_px, cy_px)
            entry["suggested_x"] = round(cwx, 2)
            entry["suggested_y"] = round(cwy, 2)
            entry["near_wall_m"] = near_m
            entry["far_wall_m"] = far_m
        coordinates.append(entry)

    with open(os.path.join(out_dir, "coordinate_proposal.json"), "w") as f:
        _json.dump(coordinates, f, indent=2)
    with open(os.path.join(out_dir, "path_info.json"), "w") as f:
        _json.dump({"path_length_m": final_length_m, "num_waypoints": len(wp_final),
                     "raw_points": len(wp_full_raw), "simplified_points": len(wp_simplified)}, f)
    print(f"PATH_LENGTH_M: {final_length_m:.2f}  waypoints: {len(wp_final)} (raw {len(wp_full_raw)} -> simplified {len(wp_simplified)} -> final {len(wp_final)})")

    # ---- 오버레이 이미지: 단순화+보정된 최종 경로를 그림 (원본 픽셀 좌표계 그대로 유지, 크롭 없음) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # 여백 0, 이미지 전체를 꽉 채움
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # 이미지 좌표계(y축 아래로 증가)와 일치
    fx_, fy_ = zip(*wp_final)
    ax.plot(fx_, fy_, color="orange", linewidth=2)
    ax.scatter(fx_, fy_, c="orange", s=15, zorder=4)
    ax.scatter(*start_full, c="green", s=60, zorder=5)
    ax.scatter(*goal_full, c="red", s=60, zorder=5)
    ax.axis("off")
    overlay_path = os.path.join(out_dir, "overlay_solo.png")
    plt.savefig(overlay_path, dpi=dpi)  # bbox_inches="tight" 제거 -> 원본과 동일한 (w,h) 픽셀 크기 보장
    plt.close(fig)
    print(f"OK: {overlay_path}")

if __name__ == "__main__":
    main()