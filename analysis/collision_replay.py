"""
collision_replay.py ─ 저장된 rollout(qpos)을 '전신 충돌 모델'로 재생해서 로봇–장애물 충돌을 센다.

왜 재생인가
-----------
DIAL-MPC 시뮬레이션에 쓰는 mjx_go2_robot_only.xml은 충돌 geom이 발끝 구 4개뿐이다
(몸통·고관절·허벅지·종아리는 장애물을 통과). 그래서 시뮬레이션 안의 contact만 세면
몸통이 벽을 뚫고 지나가도 충돌 0으로 나온다.

여기서는 같은 관절 구조(nq=19, 관절 이름 동일)에 충돌 geom 23개가 있는 go2.xml로
씬을 다시 조립하고, 매 timestep의 qpos를 그대로 넣어 mj_forward → 접촉(dist <= 0)을 센다.
기구학적 재생이라 물리를 다시 풀지 않으며, 이미 돌린 rollout에도 그대로 적용된다.

논문 문구(제안): "Collisions are evaluated by replaying each executed trajectory with the
full collision model of Go2 (23 collision primitives) and counting timesteps in which any
robot geometry penetrates a static obstacle (Shield-Loco, 2026)."
"""
from __future__ import annotations

import os
import re
import uuid
from functools import lru_cache

import mujoco
import numpy as np

SIM_ROBOT_XML = "mjx_go2_robot_only.xml"
REPLAY_ROBOT_XML = "go2.xml"


def _scene_text_with_full_collision(scene_xml_path: str) -> str:
    txt = open(scene_xml_path).read()
    pat = re.compile(r'<include\s+file="' + re.escape(SIM_ROBOT_XML) + r'"\s*/>')
    if not pat.search(txt):
        raise ValueError(f"{scene_xml_path}: '{SIM_ROBOT_XML}' include를 찾을 수 없음")
    return pat.sub(f'<include file="{REPLAY_ROBOT_XML}"/>', txt)


@lru_cache(maxsize=32)
def build_replay_model(scene_xml_path: str) -> mujoco.MjModel:
    """씬 XML의 로봇 include만 go2.xml로 바꾼 replay 모델. include/asset 상대경로 때문에 같은 폴더에 임시 저장."""
    scene_xml_path = os.path.abspath(scene_xml_path)
    tmp = os.path.join(os.path.dirname(scene_xml_path), f"_replay_{uuid.uuid4().hex[:8]}.xml")
    try:
        with open(tmp, "w") as f:
            f.write(_scene_text_with_full_collision(scene_xml_path))
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _robot_root(m: mujoco.MjModel) -> int:
    for b in range(1, m.nbody):
        if m.body_jntnum[b] > 0 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
            return b
    raise ValueError("free joint를 가진 로봇 root body가 없음")


def geom_partition(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """(robot_geoms, obstacle_geoms). 장애물 = 로봇이 아닌, 바닥(plane)이 아닌, 충돌 가능한 geom."""
    root = _robot_root(m)

    def is_robot(body: int) -> bool:
        while body != 0:
            if body == root:
                return True
            body = m.body_parentid[body]
        return False

    collidable = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    robot = np.array([g for g in range(m.ngeom) if collidable[g] and is_robot(m.geom_bodyid[g])], dtype=int)
    obstacle = np.array([g for g in range(m.ngeom) if collidable[g] and not is_robot(m.geom_bodyid[g])
                         and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE], dtype=int)
    return robot, obstacle


def count_collisions(scene_xml_path: str, qpos: np.ndarray, penetration_tol: float = 0.0) -> dict:
    """
    qpos: (T, 19) rollout의 qpos 그대로 (world 좌표, freejoint 포함).
    반환:
      collision_steps   충돌이 하나라도 있었던 timestep 수
      collision_frac    collision_steps / T
      collided          collision_steps > 0
      first_collision_step  처음 충돌한 step (없으면 -1)
      max_penetration_m 최대 침투 깊이
      parts             충돌에 관여한 로봇 body 이름 → step 수
    """
    m = build_replay_model(os.path.abspath(scene_xml_path))
    d = mujoco.MjData(m)
    robot, obstacle = geom_partition(m)
    if qpos.shape[1] != m.nq:
        raise ValueError(f"qpos 열 수 {qpos.shape[1]} != model.nq {m.nq}")
    is_robot = np.zeros(m.ngeom, bool); is_robot[robot] = True
    is_obs = np.zeros(m.ngeom, bool); is_obs[obstacle] = True

    steps, first, max_pen = 0, -1, 0.0
    parts: dict[str, int] = {}
    for t in range(qpos.shape[0]):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        hit = False
        hit_bodies = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if c.dist > penetration_tol:
                continue
            if (is_robot[g1] and is_obs[g2]) or (is_robot[g2] and is_obs[g1]):
                hit = True
                max_pen = max(max_pen, -float(c.dist))
                rg = g1 if is_robot[g1] else g2
                hit_bodies.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[rg]))
        if hit:
            steps += 1
            first = t if first < 0 else first
            for b in hit_bodies:
                parts[b] = parts.get(b, 0) + 1
    T = qpos.shape[0]
    return {"collision_steps": steps, "collision_frac": steps / max(T, 1), "collided": steps > 0,
            "first_collision_step": first, "max_penetration_m": max_pen, "parts": parts}
