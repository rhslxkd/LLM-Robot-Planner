"""
끝판왕 waypoint 생성기 (config-space / distance-field 기반).
VLM 없이, courtroom의 ray-casting 대신 distance transform 하나로 통일.

핵심 아이디어 (고전 config-space [Lozano-Perez 1983] + 최신 distance-field
모션플래닝, cf. "Safe Bubble Cover for Motion Planning on Distance Fields",
arXiv:2408.13377):
  1. 씬 이미지에서 벽 마스크를 한 번 만든다.
  2. 그 마스크의 Euclidean distance transform(EDT)을 한 번 계산한다:
     dist_field[y,x] = (x,y)에서 가장 가까운 벽 픽셀까지의 거리(px).
  3. 이후 모든 clearance 계산은 O(1) 배열 조회. ray-casting 반복 없음.
  4. 세그먼트(waypoint 사이 직선) 위의 매 픽셀에서 clearance를 조회하면
     corner-cutting이 별도 로직 없이 자동으로 잡힌다 (courtroom의
     _segment_is_clear가 이미 픽셀 단위로 촘촘히 샘플링하던 것과 동일한
     방식이되, 이번엔 이진 마스크가 아니라 연속값 거리장을 사용).
  5. 위반 waypoint 보정 = distance field에서 전방향(원형 탐색)으로 가장
     가까운 안전점을 찾음 (기존 perpendicular-only shift보다 일반적).

두 단계 마진:
  HARD_RADIUS_M = 0.4  (= MIN_CLEARANCE_M/2 = 0.8m 전체 통로폭, DIAL-MPC 실측 검증된 하드플로어)
  SOFT_RADIUS_M = 0.6  (2026-09-02 배치 스윕: 0.3->0.4->0.5->0.6 순으로 튜닝,
                        13개 중 10개 DIAL-MPC 생존 확인된 경험적 선호 마진)
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

HARD_RADIUS_M = 0.4
SOFT_RADIUS_M = 0.6
MIN_STEP_M = 0.4
MAX_STEP_M = 1.0
GOAL_MARKER_EXCLUDE_PX = 30
STUCK_EPS_PX = 3.0
MAX_ITERS = 30
CORRECTION_MAX_SHIFT_M = 0.3   # waypoint 자체를 살짝 안전한 쪽으로 미세조정할 때의 최대 이동거리.
CORNER_BEND_SEARCH_M = 0.6     # 코너를 "우회"할 굴절점을 찾을 때의 탐색 반경. 코너를 완전히
                                # 벗어나려면 단순 미세조정보다 더 큰 이동이 필요하므로 별도로 더
                                # 넉넉하게 잡는다 (우리 씬의 벽 두께/기둥 폭 스케일 기준 0.6m).
CORNER_FIX_IMPROVE_EPS = 0.01  # 굴절점을 넣었을 때 min_clearance가 이만큼도 안 좋아지면
                                # "진짜 병목이라 못 고침"으로 판단하고 포기한다.
MAX_CORNER_FIX_ATTEMPTS = 5    # 같은 종류의 corner-cut 위반에 대해 굴절점 삽입을 시도하는
                                # 최대 횟수. 이게 없으면 진짜 병목(R012 같은 케이스)에서
                                # 개선 없는 삽입을 max_iters까지 무한 반복하게 된다.


def build_wall_mask(image_path, exclude_center_px=None, exclude_radius_px=0):
    import matplotlib.image as mpimg
    arr = mpimg.imread(image_path)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    mask = (r > 180) & (g < 120) & (b < 120)
    if exclude_center_px is not None and exclude_radius_px > 0:
        h, w = mask.shape
        cx, cy = exclude_center_px
        yy, xx = np.ogrid[:h, :w]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= exclude_radius_px ** 2
        mask = mask & ~disk
    return mask


def build_distance_field(wall_mask):
    """dist_field[y,x] = 가장 가까운 벽 픽셀까지의 유클리드 거리(px)."""
    free = ~wall_mask
    return distance_transform_edt(free)


def clearance_m_at(dist_field, px, py, ppm):
    h, w = dist_field.shape
    xi, yi = int(round(px)), int(round(py))
    if not (0 <= xi < w and 0 <= yi < h):
        return 0.0
    return float(dist_field[yi, xi]) / ppm


def segment_min_clearance_m(dist_field, p0, p1, ppm):
    """세그먼트를 픽셀 단위로 촘촘히 샘플링해 최소 clearance 반환 (corner-cutting 자동 검출)."""
    dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    n = max(2, int(dist))
    min_c = float("inf")
    for k in range(n + 1):
        t = k / n
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        c = clearance_m_at(dist_field, x, y, ppm)
        if c < min_c:
            min_c = c
    return min_c


def find_nearest_safe_point(dist_field, px, py, target_radius_px, search_radius_px=200):
    """(px,py) 근처에서 dist_field >= target_radius_px인 가장 가까운 점을 전방향 탐색.
    못 찾으면 None (target_radius_px를 만족하는 점이 이 반경 내에 없음 -> 구조적 병목 후보)."""
    h, w = dist_field.shape
    x0, y0 = int(round(px)), int(round(py))
    if 0 <= x0 < w and 0 <= y0 < h and dist_field[y0, x0] >= target_radius_px:
        return px, py
    best = None
    best_d2 = None
    for rad in range(1, int(search_radius_px), 2):
        n_samples = max(8, int(2 * np.pi * rad / 3))
        for i in range(n_samples):
            theta = 2 * np.pi * i / n_samples
            xi = int(round(x0 + rad * np.cos(theta)))
            yi = int(round(y0 + rad * np.sin(theta)))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            if dist_field[yi, xi] >= target_radius_px:
                d2 = (xi - x0) ** 2 + (yi - y0) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = (float(xi), float(yi))
        if best is not None:
            return best
    return None


def check_step_lengths(coords, min_step_m=MIN_STEP_M, max_step_m=MAX_STEP_M):
    violations = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1]["x"] - coords[i]["x"]
        dy = coords[i + 1]["y"] - coords[i]["y"]
        d = (dx ** 2 + dy ** 2) ** 0.5
        if d > max_step_m:
            violations.append((i, i + 1, round(d, 3), "too_long"))
        elif d < min_step_m:
            violations.append((i, i + 1, round(d, 3), "too_short"))
    return violations


def generate_waypoints(image_path, raw_coords, robot_px, ppm,
                        hard_radius_m=HARD_RADIUS_M, soft_radius_m=SOFT_RADIUS_M,
                        min_step_m=MIN_STEP_M, max_step_m=MAX_STEP_M,
                        max_iters=MAX_ITERS):
    """Neural A*의 raw 좌표를 받아 distance-field 기반으로 안전 마진을 만족하는
    최종 좌표를 반환한다. 시작점/goal(coords[0], coords[-1])은 이동하지 않는다.
    반환: (final_coords, passed: bool, log: list[str])
    """
    coords = [dict(c) for c in raw_coords]
    rx, ry = robot_px
    log = []

    def to_px(c):
        return (rx + c["x"] * ppm, ry - c["y"] * ppm)

    def to_world(px, py):
        return (px - rx) / ppm, (ry - py) / ppm

    hard_px = hard_radius_m * ppm
    soft_px = soft_radius_m * ppm
    max_shift_px = CORRECTION_MAX_SHIFT_M * ppm
    corner_bend_px = CORNER_BEND_SEARCH_M * ppm
    min_step_px = min_step_m * ppm

    goal_px = to_px(coords[-1])
    wall_mask = build_wall_mask(image_path, exclude_center_px=goal_px,
                                 exclude_radius_px=GOAL_MARKER_EXCLUDE_PX)
    dist_field = build_distance_field(wall_mask)

    corner_fix_attempts = 0
    corner_fix_exhausted = False

    for it in range(1, max_iters + 1):
        changed = False
        points_px = [to_px(c) for c in coords]

        # 좁은 반경(CORRECTION_MAX_SHIFT_M)으로 못 찾으면, 코너 탈출 규모의 넓은
        # 반경(CORNER_BEND_SEARCH_M)으로 재시도한다. 안 그러면 idx는 영원히
        # "hard floor도 만족 불가"로 방치되고, corner-cut 굴절점 삽입은 이 idx를
        # 직접 못 옮긴 채 옆에 점만 끼워넣으려다 계속 실패한다.
        # 넓은 반경 후보는 이웃 waypoint와 최소 보폭(min_step_px) 이상 떨어진
        # 것만 채택해 다음 iteration에 "보폭 미달"로 제거->재삽입되는 오실레이션을 막는다.
        for i in range(1, len(coords) - 1):
            px, py = points_px[i]
            c_m = clearance_m_at(dist_field, px, py, ppm)
            if c_m < soft_radius_m:
                target = find_nearest_safe_point(dist_field, px, py, soft_px, search_radius_px=max_shift_px)
                degraded = False
                widened = False
                if target is None:
                    target = find_nearest_safe_point(dist_field, px, py, hard_px, search_radius_px=max_shift_px)
                    degraded = True
                if target is None:
                    prev_px = points_px[i - 1]
                    next_px = points_px[i + 1] if i + 1 < len(points_px) else None
                    for r_px, is_hard in ((soft_px, False), (hard_px, True)):
                        cand = find_nearest_safe_point(dist_field, px, py, r_px, search_radius_px=corner_bend_px)
                        if cand is None:
                            continue
                        d_prev = ((cand[0] - prev_px[0]) ** 2 + (cand[1] - prev_px[1]) ** 2) ** 0.5
                        d_next = (((cand[0] - next_px[0]) ** 2 + (cand[1] - next_px[1]) ** 2) ** 0.5
                                  if next_px is not None else float("inf"))
                        if d_prev >= min_step_px and d_next >= min_step_px:
                            target = cand
                            degraded = is_hard
                            widened = True
                            break
                if target is None:
                    log.append(f"[iter {it}] idx {i}: hard floor({hard_radius_m}m)도 만족 불가 (광역탐색 {CORNER_BEND_SEARCH_M}m 포함) -> 구조적 병목")
                    continue
                moved = ((target[0] - px) ** 2 + (target[1] - py) ** 2) ** 0.5
                if moved < STUCK_EPS_PX:
                    continue
                wx, wy = to_world(*target)
                coords[i]["x"], coords[i]["y"] = round(wx, 3), round(wy, 3)
                tag = "hard floor만 만족" if degraded else "soft margin 만족"
                if widened:
                    tag += f" / {CORNER_BEND_SEARCH_M}m 광역탐색"
                log.append(f"[iter {it}] idx {i}: clearance {c_m:.2f}m -> ({wx:.2f},{wy:.2f}) 이동 ({tag})")
                changed = True
        if changed:
            continue

        if not corner_fix_exhausted:
            points_px = [to_px(c) for c in coords]
            bad_segment = None
            for i in range(len(coords) - 1):
                min_c = segment_min_clearance_m(dist_field, points_px[i], points_px[i + 1], ppm)
                if min_c < hard_radius_m:
                    bad_segment = (i, i + 1, min_c)
                    break
            if bad_segment is not None:
                i, j, min_c = bad_segment
                mx = (points_px[i][0] + points_px[j][0]) / 2
                my = (points_px[i][1] + points_px[j][1]) / 2
                target = find_nearest_safe_point(dist_field, mx, my, soft_px, search_radius_px=corner_bend_px)
                if target is None:
                    target = find_nearest_safe_point(dist_field, mx, my, hard_px, search_radius_px=corner_bend_px)
                accepted = False
                if target is not None:
                    d_i_target = ((target[0]-points_px[i][0])**2 + (target[1]-points_px[i][1])**2) ** 0.5
                    d_target_j = ((target[0]-points_px[j][0])**2 + (target[1]-points_px[j][1])**2) ** 0.5
                    if d_i_target >= min_step_px and d_target_j >= min_step_px:
                        new_min = min(
                            segment_min_clearance_m(dist_field, points_px[i], target, ppm),
                            segment_min_clearance_m(dist_field, target, points_px[j], ppm),
                        )
                        if new_min > min_c + CORNER_FIX_IMPROVE_EPS:
                            wx, wy = to_world(*target)
                            coords.insert(j, {"x": round(wx, 3), "y": round(wy, 3)})
                            log.append(f"[iter {it}] segment ({i},{j}) corner-cut(min_clearance={min_c:.2f}m) -> 굴절점 삽입 (개선: {new_min:.2f}m)")
                            changed = True
                            accepted = True
                if not accepted:
                    corner_fix_attempts += 1
                    log.append(f"[iter {it}] segment ({i},{j}) corner-cut(min_clearance={min_c:.2f}m) -> 굴절점 삽입 실패/무의미 (시도 {corner_fix_attempts}/{MAX_CORNER_FIX_ATTEMPTS})")
                if corner_fix_attempts >= MAX_CORNER_FIX_ATTEMPTS:
                    corner_fix_exhausted = True
                    log.append(f"[iter {it}] corner-cut 보정 시도 한도 초과 -> 구조적 병목으로 판단, 이후 시도 중단")
        if changed:
            continue

        step_violations = check_step_lengths(coords, min_step_m, max_step_m)
        if step_violations:
            i, j, d, kind = step_violations[0]
            if kind == "too_long":
                mx = (coords[i]["x"] + coords[j]["x"]) / 2
                my = (coords[i]["y"] + coords[j]["y"]) / 2
                coords.insert(j, {"x": round(mx, 3), "y": round(my, 3)})
                log.append(f"[iter {it}] segment ({i},{j}) 보폭 초과({d}m) -> 중간점 삽입")
                changed = True
            elif 0 < j < len(coords) - 1:
                coords.pop(j)
                log.append(f"[iter {it}] segment ({i},{j}) 보폭 미달({d}m) -> idx {j} 제거")
                changed = True
        if changed:
            continue

        points_px = [to_px(c) for c in coords]
        final_ok = all(
            segment_min_clearance_m(dist_field, points_px[k], points_px[k + 1], ppm) >= hard_radius_m
            for k in range(len(coords) - 1)
        )
        step_ok = not check_step_lengths(coords, min_step_m, max_step_m)
        passed = final_ok and step_ok
        log.append(f"[iter {it}] 수렴. passed={passed}")
        return coords, passed, log

    log.append(f"max_iters({max_iters}) 초과 -- 미수렴")
    return coords, False, log
