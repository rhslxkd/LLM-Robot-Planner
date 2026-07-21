"""
metrics.py ─ 실험 C(물리적 타당성 검증)용 정량 지표

각 함수는 load_rollout() 이 반환하는 배열(numpy)을 그대로 받는다.
정의를 바꿀 때는 여기만 고치면 되고, 논문 실험 C 섹션 문구와 반드시
동기화해서 유지할 것 (여기 숫자 = 논문 Table의 숫자여야 함).
"""
import numpy as np

# Unitree Go2 공식 스펙 기준 관절 최대 연속 토크(N·m). 실험 A/B/C 어디서든
# 이 상수 하나만 바꾸면 전체 분석에 일괄 반영된다. 값 재검증 필요 시 갱신.
GO2_TORQUE_LIMIT_NM = 45.0  # Unitree 공식 스펙: 최대 관절 토크 (largest joint motor 기준)


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


def summarize(data: dict) -> dict:
    """load_rollout() 결과 dict 하나를 받아 실험 C 지표를 한 번에 계산."""
    ctrl = data["ctrl"]
    qfrc_joints = data["qfrc_actuator"][:, 6:18]  # freejoint 6개 제외, 관절 12개만
    return {
        "torque_violation_rate": torque_limit_violation_rate(ctrl),
        "cmd_actual_error_mean": float(cmd_vs_actual_error(ctrl, qfrc_joints).mean()),
        "smoothness": trajectory_smoothness(data["qpos"]),
    }

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
