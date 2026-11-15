"""
episode_metrics.py ─ 표 ①·② 공통 '실행 지표' (MuJoCo rollout 하나 → 지표 dict 하나).

정의는 여기 한 곳에만 둔다. 논문 표의 숫자 = 이 파일의 숫자.

| 지표                 | 정의                                                          | 근거 (Go2 + MuJoCo)              |
|----------------------|---------------------------------------------------------------|----------------------------------|
| success              | reached ∧ ¬fell ∧ ¬collided                                   | LEAP, Footholds, DDAT, RoboGauge |
| reached              | 몸통 xy가 goal 반경(GOAL_RADIUS_M) 안에 한 번이라도 들어옴      |                                  |
| time_to_goal_s       | 처음 goal 반경에 들어온 시각 (도달 못 하면 NaN)                | RoboGauge(통과 시간)             |
| fell                 | 몸통 z < 0.18 m 또는 기울기 > 90° (check_fall.py와 동일)       | DDAT, RoboGauge                  |
| tilt_max/mean_deg    | 몸통 +z축과 world +z축 사이 각도                              | RoboGauge(자세 안정성)           |
| collided / steps     | go2.xml 전신 모델 재생 시 장애물 침투 timestep (collision_replay) | Shield-Loco                    |
| track_mean/max_m     | 각 waypoint에 대한 실제 궤적 최근접 거리                      | DIAL-MPC, Adaptive MPC           |
| path_dev_mean_m      | 실제 몸통 궤적 → 계획 polyline까지 거리의 평균                 |                                  |
| torque_sat_*         | |ctrl| ≥ 0.99·한계인 비율 (관절 유형별)                        | RoboGauge(DoF limits)            |
| torque_violation     | |ctrl| > 한계 (ctrlrange 복구 후에는 0이어야 정상 → 검증용)     |                                  |
| torque_smooth        | 평균 |Δτ| (N·m/step)                                          | RoboGauge(torque smoothness)     |
"""
from __future__ import annotations

import numpy as np

from analysis.collision_replay import count_collisions

DT = 0.02
GOAL_RADIUS_M = 0.3
FALL_MIN_Z_M = 0.18

# Go2 관절별 토크 한계 (actuator 순서: FR, FL, RR, RL × hip(abduction), thigh, calf)
GO2_TORQUE_LIMITS = np.array([23.7, 23.7, 45.43] * 4)
JOINT_TYPES = ("abduction", "thigh", "calf")

# states.npy 80열 스키마 (dial_core.py 저장 순서)
SCHEMA = (("step", 1), ("qpos", 19), ("qvel", 18), ("ctrl", 12), ("qfrc_actuator", 18), ("feet_xyz", 12))


def split_states(states: np.ndarray) -> dict:
    out, i = {}, 0
    for name, w in SCHEMA:
        out[name] = states[:, i:i + w]
        i += w
    assert i == states.shape[1], f"states 열 수 {states.shape[1]} != 스키마 {i}"
    return out


def _tilt_deg(qpos: np.ndarray) -> np.ndarray:
    qx, qy = qpos[:, 4], qpos[:, 5]
    return np.degrees(np.arccos(np.clip(1 - 2 * (qx ** 2 + qy ** 2), -1.0, 1.0)))


def _point_to_polyline(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """p: (T,2), poly: (N,2) → 각 점에서 polyline까지 최소 거리 (T,)."""
    if len(poly) == 1:
        return np.linalg.norm(p - poly[0], axis=1)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    ap = p[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - proj, axis=-1).min(1)


def evaluate_rollout(states: np.ndarray, scene_xml: str, planned_xy: np.ndarray, goal_xy,
                     goal_radius: float = GOAL_RADIUS_M, dt: float = DT) -> dict:
    """
    states: (T, 80) states.npy
    planned_xy: (N, 2) 계획 waypoint (로봇 시작점 기준 상대좌표 = last_judged_path.json 그대로)
    goal_xy: 목표점 (시작점 기준 상대좌표)
    """
    s = split_states(states)
    qpos, ctrl = s["qpos"], s["ctrl"]
    origin = qpos[0, :2]                                  # env의 vlm_origin_xy와 동일 (초기 몸통 xy)
    xy = qpos[:, :2] - origin
    planned_xy = np.asarray(planned_xy, float).reshape(-1, 2)
    goal_xy = np.asarray(goal_xy, float)

    # 도달
    dist_goal = np.linalg.norm(xy - goal_xy, axis=1)
    inside = dist_goal <= goal_radius
    reached = bool(inside.any())
    t_goal = float(np.argmax(inside) * dt) if reached else float("nan")

    # 낙상 / 기울기
    tilt = _tilt_deg(qpos)
    fell_mask = (qpos[:, 2] < FALL_MIN_Z_M) | (tilt > 90.0)
    fell = bool(fell_mask.any())

    # 충돌 (전신 모델 재생)
    col = count_collisions(scene_xml, qpos)

    # 추종
    wp_d = np.array([np.linalg.norm(xy - w, axis=1).min() for w in planned_xy])
    dev = _point_to_polyline(xy, planned_xy)

    # 토크
    ratio = np.abs(ctrl) / GO2_TORQUE_LIMITS
    sat = ratio >= 0.99
    out = {
        "success": bool(reached and not fell and not col["collided"]),
        "reached": reached,
        "time_to_goal_s": t_goal,
        "final_goal_dist_m": float(dist_goal[-1]),
        "fell": fell,
        "first_fall_step": int(np.argmax(fell_mask)) if fell else -1,
        "tilt_max_deg": float(tilt.max()),
        "tilt_mean_deg": float(tilt.mean()),
        "collided": col["collided"],
        "collision_steps": col["collision_steps"],
        "collision_frac": col["collision_frac"],
        "max_penetration_m": col["max_penetration_m"],
        "collision_parts": col["parts"],
        "track_mean_m": float(wp_d.mean()),
        "track_max_m": float(wp_d.max()),
        "path_dev_mean_m": float(dev.mean()),
        "torque_sat_rate": float(sat.mean()),
        "torque_violation_rate": float((ratio > 1.0 + 1e-6).mean()),
        "torque_smooth_nm": float(np.abs(np.diff(ctrl, axis=0)).mean()) if len(ctrl) > 1 else 0.0,
        "n_steps": int(states.shape[0]),
    }
    for k, name in enumerate(JOINT_TYPES):
        out[f"torque_sat_{name}"] = float(sat[:, k::3].mean())
        out[f"torque_peak_ratio_{name}"] = float(ratio[:, k::3].max())
    return out
