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
# scipy가 제공하는 "정확한 유클리드 거리 변환" 함수. 이진 마스크를 입력하면,
    # 마스크가 False(0)인 각 픽셀에 대해 "가장 가까운 True(1) 픽셀까지의 유클리드
    # 거리"를 계산해서 같은 크기의 실수 배열로 돌려줌. 내부적으로 효율적인 알고리즘
    # (Felzenszwalb & Huttenlocher 방식 등)을 써서 각 픽셀마다 브루트포스로 모든
    # 벽 픽셀과의 거리를 재는 것보다 훨씬 빠름.


# ══════════════════════════════════════════════════════════════════════
# 섹션 1. 튜닝 상수
# ══════════════════════════════════════════════════════════════════════
 
HARD_RADIUS_M = 0.4
# "하드 플로어" -- 이보다 가까우면 무조건 안전 위반으로 간주하는 절대 최소 여유거리.
    # 이 값의 2배(0.8m)가 로봇이 지나갈 수 있는 통로의 최소 전체 폭이 됨
    # (로봇이 통로 중앙을 지난다고 가정하면, 양쪽 벽까지 각각 0.4m씩 여유가 있어야
    #  전체 폭 0.8m). 이 값은 DIAL-MPC로 실측 검증된 하드 하한선.
SOFT_RADIUS_M = 0.6
# "소프트 마진" -- 이보다 가까우면 "가능하면 더 멀리 밀어내려고 시도"하는
    # 권장 여유거리. 2026-09-02에 0.3 -> 0.4 -> 0.5 -> 0.6 순서로 배치 스윕(실측
    # 반복 실험)을 통해 튜닝된 값: 13개 테스트 씬 중 10개가 DIAL-MPC에서 실제로
    # 살아남은(성공한) 경험적으로 검증된 마진
MIN_STEP_M = 0.4
MAX_STEP_M = 1.0
# waypoint 사이 보폭의 하한/상한 (run_neural_astar_step.py의 동일 상수와 같은
    # 값 -- 로코모션 컨트롤러의 실측 안정 보폭 범위).
GOAL_MARKER_EXCLUDE_PX = 30
# 목표 지점 주변 30픽셀 반경은 "벽 마스크에서 제외"하는 특수 처리. 이유: 씬
    # 이미지에 목표 지점을 표시하는 마커(원, 십자 등)가 그려져 있을 수 있는데, 그
    # 마커 자체의 색이 우연히 "빨간색"(벽 판정 기준)과 겹치면 목표 지점 바로 그
    # 자리가 "벽"으로 오판되어 목표에 도달하는 게 원천적으로 불가능해지는 모순이
    # 생길 수 있음. 이를 막기 위해 목표 주변 반경은 애초에 벽 마스크 계산에서
    # 제외시켜버림.
STUCK_EPS_PX = 3.0
# 보정 이동량이 이 값(3픽셀)보다 작으면 "사실상 안 움직인 것"으로 간주하고
    # 그 이동을 적용하지 않음. 부동소수점 반올림 등으로 인해 미세하게 위치가
    # 흔들리기만 하고 실질적인 개선 없이 반복(iteration)이 낭비되는 것을 방지.
MAX_ITERS = 30
# generate_waypoints()의 전체 수렴 루프가 최대 몇 번까지 반복될 수 있는지의 상한.
    # 이 안에 수렴 못 하면 실패(passed=False)로 처리.
CORRECTION_MAX_SHIFT_M = 0.3   
# waypoint 하나를 "제자리에서 살짝" 안전한 곳으로 옮길 때 허용하는 최대 이동
    # 반경. 이 반경 안에서 안전한 지점을 못 찾으면 "이 정도 미세조정으로는 안 되는
    # 케이스"로 보고 다음 단계(더 넓은 범위 탐색)로 넘어감.
CORNER_BEND_SEARCH_M = 0.6     # 코너를 "우회"할 굴절점을 찾을 때의 탐색 반경. 코너를 완전히
                                # 벗어나려면 단순 미세조정보다 더 큰 이동이 필요하므로 별도로 더
                                # 넉넉하게 잡는다 (우리 씬의 벽 두께/기둥 폭 스케일 기준 0.6m).
CORNER_FIX_IMPROVE_EPS = 0.01  
# 굴절점을 넣었을 때 min_clearance(구간의 최소 안전거리)가 최소 이 값(1cm)
    # 이상 개선되지 않으면 "의미 없는 수정"으로 보고 채택하지 않음. 이게 없으면
    # 거의 개선이 안 되는(혹은 오히려 미세하게 나빠지는) 굴절점을 계속 넣었다 뺐다
    # 하며 진동(oscillation)할 위험이 있음.
MAX_CORNER_FIX_ATTEMPTS = 5    
# 같은 종류의 코너-컷(corner-cut) 위반에 대해 굴절점 삽입을 시도하는 최대
    # 횟수. 이 제한이 없으면, 진짜로 물리적으로 좁아서 못 고치는 구조적 병목
    # (예: R012 케이스, 실제로 발견됐던 사례)에서 "고쳐지지도 않는데 계속 시도만
    # 하다가" MAX_ITERS를 전부 허비하게 됨. 5번 시도해도 안 되면 포기하고 다음
    # 검사(보폭 제약 등)로 넘어감.


def build_wall_mask(image_path, exclude_center_px=None, exclude_radius_px=0):
    """
    [함수 역할] oracle.png(또는 그 경로)를 읽어서, "빨간색=벽"인 이진 마스크를 만든다.
    옵션으로 특정 중심점 주변 원형 영역을 마스크에서 제외할 수 있다(목표 마커 제외용).
    """
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
    # distance_transform_edt는 "0인 픽셀에서 가장 가까운 1인 픽셀까지의 거리"가
        # 아니라 정확히는 "0이 아닌(non-zero) 픽셀까지의 거리"를 계산하는 함수라서,
        # "벽이 아닌 곳(free)"을 True로 표시한 배열을 넘겨줘야 함 -- 함수 자체는
        # "각 픽셀에서 가장 가까운 False(0) 픽셀까지의 거리"를 계산하므로, wall_mask를
        # 그대로 넘기면 반대 결과가 나옴. 따라서 반전(~)해서 넘겨야 함.
        # (참고: scipy 공식 동작은 "0이 아닌 요소에서 가장 가까운 0인 요소까지의
        #  거리"를 계산 -- free가 True(1)인 곳에서 wall(False=0)까지의 거리를 구하는
        #  것이므로 이 free 배열을 넣는 게 맞음)
    return distance_transform_edt(free)
    # 결과: free 배열과 같은 크기의 float 배열. 벽에서 멀수록 값이 크고, 벽
        # 픽셀 자체에서는 0에 가까움. 이 배열 하나가 이 파일 전체 알고리즘의 기반.


def clearance_m_at(dist_field, px, py, ppm):
    """[함수 역할] 픽셀 좌표 (px,py)에서의 clearance(안전 여유거리)를 미터 단위로 조회."""
    h, w = dist_field.shape
    xi, yi = int(round(px)), int(round(py))
    if not (0 <= xi < w and 0 <= yi < h):
        return 0.0
    return float(dist_field[yi, xi]) / ppm


def segment_min_clearance_m(dist_field, p0, p1, ppm):
    """
    [함수 역할] p0-p1 직선 구간을 촘촘히 샘플링하면서 각 지점의 clearance를 조회해,
    그 중 "가장 작은"(가장 위험한) 값을 반환한다.
 
    [왜 이게 corner-cutting을 자동으로 잡아내는가]
    "corner-cutting"이란, 두 waypoint 각각은 벽에서 충분히 떨어져 있는데
    그 "사이를 잇는 직선"이 코너 모서리를 아슬아슬하게 스치듯 지나가는 문제.
    양 끝점만 검사하면 이걸 놓치지만, 구간 전체를 촘촘히 샘플링해서 매 지점의
    clearance를 확인하면 중간에 숨어있는 위반 지점도 반드시 걸러짐.
    """
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
    """
    [함수 역할] (px,py) 근처에서 "clearance >= target_radius_px를 만족하는 가장
    가까운 지점"을 전방향(원형/방사형)으로 탐색해서 찾는다.
 
    [알고리즘] 링(고리) 확장 탐색: 반경 1픽셀부터 시작해서 점점 반경을 넓혀가며,
    각 반경의 원둘레 위에서 여러 각도(theta)로 샘플링한다. 어느 반경에서든 조건을
    만족하는 점을 찾으면 즉시 반환 -- 이렇게 하면 자동으로 "가장 가까운" 안전점을
    찾게 됨 (더 작은 반경부터 검사하니까).
    """
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
    """[함수 역할] 좌표 리스트에서 보폭 제약(min/max)을 위반하는 구간들을 모두 찾아
    (i, j, 거리, 위반종류) 튜플의 리스트로 반환한다."""
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
    """
    [함수 역할 -- 이 파일의 핵심]
    Neural A*의 raw 좌표(world 단위)를 받아서, distance-field 기반으로 안전
    마진을 만족하는 최종 좌표로 반복적으로(iteratively) 보정한다.
    시작점(coords[0])과 목표점(coords[-1])은 절대 이동시키지 않는다
    (로봇의 실제 시작 위치와 요청받은 목표는 임의로 바꿀 수 없으니까).
 
    반환값: (final_coords, passed: bool, log: list[str])
      - final_coords: 보정된 최종 waypoint 리스트
      - passed: 모든 안전/보폭 제약을 만족했는지 여부
      - log: 각 반복(iteration)에서 무슨 일이 있었는지 사람이 읽을 수 있는 로그 목록
             (verdict.png 옆에 log.txt로 같이 저장되어 나중에 디버깅에 사용됨)
    """
    coords = [dict(c) for c in raw_coords]
    # 입력을 그대로 참조/변경하지 않도록 각 딕셔너리를 얕은 복사(dict(c))해서
        # 새 리스트를 만듦 -- 호출자가 넘긴 원본 raw_coords가 이 함수에 의해
        # 의도치 않게 변경되는 부작용(side effect)을 방지하는 방어적 프로그래밍.
    rx, ry = robot_px
    log = []

    def to_px(c):
        return (rx + c["x"] * ppm, ry - c["y"] * ppm)
    # world -> pixel 변환 (world_to_pixel과 동일한 공식을 인라인 클로저로 정의).

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
        # ════════════════════════════════════════════════════════════
        # 매 반복(iteration)마다 아래 우선순위 순서로 딱 "한 종류의 문제만" 고치고
        # 즉시 다음 iteration으로 넘어간다 (continue). 이렇게 "한 번에 하나씩"
        # 고치는 이유는, 여러 문제를 동시에 고치면 서로 간섭해서(한쪽을 고치다가
        # 다른 쪽을 다시 깨뜨리는 식) 수렴이 불안정해질 수 있기 때문. 우선순위:
        #   (1) 개별 점의 clearance 위반 (가장 기본적인 안전 문제)
        #   (2) 구간(segment)의 corner-cutting 위반 (점은 괜찮은데 사이 직선이 위험)
        #   (3) 보폭(step length) 제약 위반
        #   (4) 위 셋 다 없으면 수렴 완료
        # ════════════════════════════════════════════════════════════
        changed = False
        points_px = [to_px(c) for c in coords]

        # 좁은 반경(CORRECTION_MAX_SHIFT_M)으로 못 찾으면, 코너 탈출 규모의 넓은
        # 반경(CORNER_BEND_SEARCH_M)으로 재시도한다. 안 그러면 idx는 영원히
        # "hard floor도 만족 불가"로 방치되고, corner-cut 굴절점 삽입은 이 idx를
        # 직접 못 옮긴 채 옆에 점만 끼워넣으려다 계속 실패한다.
        # 넓은 반경 후보는 이웃 waypoint와 최소 보폭(min_step_px) 이상 떨어진
        # 것만 채택해 다음 iteration에 "보폭 미달"로 제거->재삽입되는 오실레이션을 막는다.
        # ── (1) 개별 점(waypoint)의 clearance 위반 검사 및 보정 ──
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

        # ── (2) 구간(segment)의 corner-cutting 위반 검사 및 보정 ──
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

        # ── (3) 보폭(step length) 제약 위반 검사 및 보정 ──
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

         # ── (4) 위 (1),(2),(3) 어느 것도 안 바뀌었으면: 수렴 완료, 최종 판정 ──
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
