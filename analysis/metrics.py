"""
metrics.py ─ 실험 C(물리적 타당성 검증)용 정량 지표

각 함수는 load_rollout() 이 반환하는 배열(numpy)을 그대로 받는다.
정의를 바꿀 때는 여기만 고치면 되고, 논문 실험 C 섹션 문구와 반드시
동기화해서 유지할 것 (여기 숫자 = 논문 Table의 숫자여야 함).
"""
import math
import numpy as np

# Unitree Go2 공식 스펙 기준 관절 최대 연속 토크(N·m). 실험 A/B/C 어디서든
# 이 상수 하나만 바꾸면 전체 분석에 일괄 반영된다. 값 재검증 필요 시 갱신.
GO2_TORQUE_LIMIT_NM = 45.0  # Unitree 공식 스펙: 최대 관절 토크 (largest joint motor 기준)

# go2.xml 실측 trunk+hip envelope (ROBOT_PHYSICAL_CONSTRAINTS 프롬프트 블록과 동일 수치).
ROBOT_HALF_LENGTH_M = 0.195  # 전장 0.39m / 2
ROBOT_HALF_WIDTH_M = 0.175   # 전폭 0.35m / 2


def torque_limit_violation_rate(ctrl: np.ndarray, limit: float = GO2_TORQUE_LIMIT_NM) -> float:
    """명령 토크(ctrl)가 로봇 스펙 한계를 넘긴 비율. 0에 가까울수록 좋음."""
    return float((np.abs(ctrl) > limit).mean())


def cmd_vs_actual_error(ctrl: np.ndarray, qfrc_actuator_joints: np.ndarray) -> np.ndarray:
    """
    명령 토크 vs 실제 인가 토크의 관절별 평균 절대오차.
    ctrl: (T, 12), qfrc_actuator_joints: (T, 12) — 반드시 관절 12개로 슬라이싱해서 전달
          (qfrc_actuator 전체(18)를 넣으면 freejoint 6개가 섞여 오차 계산이 깨짐)
    반환: (12,) 관절별 평균오차. 논문에는 .mean() 한 스칼라도 같이 보고할 것.
    """
    assert ctrl.shape == qfrc_actuator_joints.shape, (
        f"shape 불일치: ctrl={ctrl.shape}, qfrc_actuator_joints={qfrc_actuator_joints.shape}. "
        "qfrc_actuator[:, 6:18] 을 전달했는지 확인."
    )
    return np.abs(ctrl - qfrc_actuator_joints).mean(axis=0)


def trajectory_smoothness(qpos: np.ndarray, dt: float = 0.02) -> float:
    """
    저크(jerk) 기반 궤적 부드러움 지표. 값이 작을수록 부드러운(안정적인) 궤적.
    qpos 전체(19열, freejoint 포함)를 넣어도 되지만, 관절각만 보고 싶으면
    qpos[:, 7:19] 를 잘라서 전달할 것.
    """
    qacc = np.diff(qpos, n=2, axis=0) / dt**2
    jerk = np.diff(qacc, axis=0) / dt
    return float(np.abs(jerk).mean())


def base_stability(qpos: np.ndarray) -> dict:
    """
    qpos: (T, 19) 전체 배열, freejoint(7) + 관절(12).
    freejoint: qpos[:,0:3]=base xyz, qpos[:,3:7]=쿼터니언(w,x,y,z).
    낙상/전복처럼 균형을 잃는 실패를 잡아낸다 (충돌 자체는 못 잡음 -> obstacle_clearance 참고).
    """
    base_z = qpos[:, 2]
    qx, qy = qpos[:, 4], qpos[:, 5]
    cos_tilt = 1 - 2 * (qx**2 + qy**2)
    tilt_deg = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
    return {
        "base_height_min": float(base_z.min()),
        "base_height_mean": float(base_z.mean()),
        "base_tilt_max_deg": float(tilt_deg.max()),
        "base_tilt_mean_deg": float(tilt_deg.mean()),
    }


def _quat_to_yaw(qw, qx, qy, qz) -> float:
    """MuJoCo 쿼터니언(w,x,y,z) -> z축 기준 yaw(rad)."""
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def _robot_corners(cx, cy, yaw, half_length, half_width):
    """yaw로 회전된 로봇 몸통 사각형의 world-frame 꼭짓점 4개."""
    ux, uy = math.cos(yaw), math.sin(yaw)
    vx, vy = -math.sin(yaw), math.cos(yaw)
    corners = []
    for sl in (1, -1):
        for sw in (1, -1):
            corners.append((cx + sl * half_length * ux + sw * half_width * vx,
                             cy + sl * half_length * uy + sw * half_width * vy))
    return corners


def _point_to_box_signed_distance(px, py, box_cx, box_cy, box_hx, box_hy) -> float:
    """점 -> 축정렬 박스까지의 부호 있는 거리. 음수=박스 내부(침투 깊이), 양수=박스 밖 거리."""
    dx = abs(px - box_cx) - box_hx
    dy = abs(py - box_cy) - box_hy
    if dx <= 0 and dy <= 0:
        return max(dx, dy)
    dx_c, dy_c = max(dx, 0.0), max(dy, 0.0)
    return math.hypot(dx_c, dy_c)


def _point_to_obb_distance(px, py, cx, cy, half_length, half_width, yaw) -> float:
    """점 -> yaw로 회전된 로봇 사각형까지의 거리 (로봇 로컬 좌표계로 역회전시켜서 계산)."""
    dx, dy = px - cx, py - cy
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    lx = dx * cos_y - dy * sin_y
    ly = dx * sin_y + dy * cos_y
    return _point_to_box_signed_distance(lx, ly, 0.0, 0.0, half_length, half_width)


def _rect_vs_aabb_signed_distance(cx, cy, yaw, half_length, half_width,
                                   ocx, ocy, ohx, ohy) -> float:
    """
    로봇(yaw 회전 사각형) - 장애물(축정렬 박스) 사이의 부호 있는 최소 거리(m).
    음수면 실제로 두 사각형이 겹침(=충돌, 값은 침투 깊이).

    로봇 꼭짓점 4개 -> 박스 거리만 재면 안 된다: 얇은 벽(Scene D 벽 두께 0.3m)이
    로봇 옆면의 "모서리와 모서리 사이"를 뚫고 지나가면 꼭짓점은 전부 벽 밖에 있어서
    충돌을 놓친다 (단위테스트로 실제 재현/확인된 케이스). 그래서 4축 SAT로 먼저
    겹침 여부와 침투 깊이를 정확히 구하고, 안 겹치는 경우에만 꼭짓점-박스 거리로
    최소거리를 구한다 (분리된 두 볼록다각형의 최단거리는 항상 한쪽의 꼭짓점에서
    나온다는 성질 이용 — 양방향 다 확인해야 정확함).
    """
    ux, uy = math.cos(yaw), math.sin(yaw)
    vx, vy = -math.sin(yaw), math.cos(yaw)
    axes = ((1.0, 0.0), (0.0, 1.0), (ux, uy), (vx, vy))
    dcx, dcy = cx - ocx, cy - ocy
    overlaps = []
    for ax, ay in axes:
        r_robot = half_length * abs(ax * ux + ay * uy) + half_width * abs(ax * vx + ay * vy)
        r_obs = ohx * abs(ax) + ohy * abs(ay)
        overlaps.append((r_robot + r_obs) - abs(dcx * ax + dcy * ay))

    if min(overlaps) > 0:
        return -min(overlaps)  # 겹침: 침투 깊이(음수), SAT 최소축(MTV) 기준

    robot_corners = _robot_corners(cx, cy, yaw, half_length, half_width)
    d1 = min(_point_to_box_signed_distance(rx, ry, ocx, ocy, ohx, ohy) for rx, ry in robot_corners)
    obs_corners = [(ocx + sx * ohx, ocy + sy * ohy) for sx in (1, -1) for sy in (1, -1)]
    d2 = min(_point_to_obb_distance(ox_, oy_, cx, cy, half_length, half_width, yaw)
             for ox_, oy_ in obs_corners)
    return min(d1, d2)


def obstacle_clearance(qpos: np.ndarray, obstacles: list) -> dict:
    """
    실제 실행 궤적(qpos, freejoint 포함 전체 배열)과 실제 장애물 기하 사이의
    clearance를 매 스텝 계산한다. 로봇을 점이 아니라 매 스텝 실제 yaw로 회전한
    몸통 사각형(go2.xml 실측 0.39m x 0.35m)으로 취급하고, 장애물과의 최소거리를
    정확한 사각형-사각형(OBB-AABB) 거리로 구한다 (단순 corner-to-box 거리는
    얇은 벽 관통 케이스를 놓치는 버그가 있어서 안 씀 — 위 _rect_vs_aabb_signed_distance
    docstring 참고).

    obstacles: occupancy_grid.obstacles_from_mjmodel()이 반환하는 리스트
               (box 타입만 지원 -- Scene A/D/E 장애물은 전부 box).
    반환값의 min_clearance_m이 음수면 실제 충돌(몸통-장애물 침투)이 발생한 것.
    """
    box_obstacles = [o for o in obstacles if o["kind"] == "box"]
    if not box_obstacles:
        return {"min_clearance_m": float("nan"), "mean_clearance_m": float("nan"),
                "pct_steps_colliding": float("nan")}

    T = qpos.shape[0]
    per_step_min = np.empty(T)

    for t in range(T):
        px, py = qpos[t, 0], qpos[t, 1]
        qw, qx, qy, qz = qpos[t, 3], qpos[t, 4], qpos[t, 5], qpos[t, 6]
        yaw = _quat_to_yaw(qw, qx, qy, qz)

        step_min = float("inf")
        for obs in box_obstacles:
            ocx, ocy = obs["pos"]
            ohx, ohy = obs["size"][0], obs["size"][1]
            d = _rect_vs_aabb_signed_distance(px, py, yaw, ROBOT_HALF_LENGTH_M, ROBOT_HALF_WIDTH_M,
                                               ocx, ocy, ohx, ohy)
            step_min = min(step_min, d)
        per_step_min[t] = step_min

    return {
        "min_clearance_m": float(per_step_min.min()),
        "mean_clearance_m": float(per_step_min.mean()),
        "pct_steps_colliding": float((per_step_min < 0).mean()),
    }


def summarize(data: dict, obstacles: list = None) -> dict:
    """
    load_rollout() 결과 dict 하나를 받아 실험 C 지표를 한 번에 계산.
    obstacles가 주어지면(해당 Scene의 실제 장애물 목록, obstacles_from_mjmodel() 결과)
    obstacle_clearance()도 함께 계산해서 합친다.
    """
    ctrl = data["ctrl"]
    qfrc_joints = data["qfrc_actuator"][:, 6:18]  # freejoint 6개 제외, 관절 12개만
    result = {
        "torque_violation_rate": torque_limit_violation_rate(ctrl),
        "cmd_actual_error_mean": float(cmd_vs_actual_error(ctrl, qfrc_joints).mean()),
        "smoothness": trajectory_smoothness(data["qpos"]),
    }
    result.update(base_stability(data["qpos"]))
    if obstacles is not None:
        result.update(obstacle_clearance(data["qpos"], obstacles))
    return result


def waypoint_tracking_error(qpos_xy: np.ndarray, judged_path_xy: np.ndarray) -> dict:
    """
    로봇의 실제 궤적(qpos_xy, world 기준)이 courtroom이 정한 waypoint(judged_path_xy,
    world 기준으로 이미 변환된 값)에 얼마나 가깝게 접근했는지 측정.
    (문헌 용어: condition matching / waypoint reaching error)

    qpos_xy: (T, 2) 로봇 실제 xy 궤적
    judged_path_xy: (N, 2) courtroom이 정한 waypoint들의 world 좌표
    """
    closest_dists = []
    for wp in judged_path_xy:
        d = np.linalg.norm(qpos_xy - wp[None, :], axis=1)
        closest_dists.append(d.min())
    closest_dists = np.array(closest_dists)

    final_goal_error = np.linalg.norm(qpos_xy[-1] - judged_path_xy[-1])

    return {
        "waypoint_mean_closest_dist": float(closest_dists.mean()),
        "waypoint_max_closest_dist": float(closest_dists.max()),
        "final_goal_error": float(final_goal_error),
    }