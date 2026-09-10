"""
================================================================================
 [STUDY VERSION] check_fall.py -- 함수/줄 단위 상세 해설판
================================================================================
 원본: core/check_fall.py (독립 실행형 유틸리티 + 2026-09-10부터 다른 스크립트가
       import해서 쓰는 공용 안전판정 함수(check_scene_safety)도 겸함)

 [역할]
   DIAL-MPC 시뮬레이션이 끝난 뒤 저장되는 로봇 상태 궤적(*_states.npy)을 읽어서
   낙상 여부 + (씬 XML을 찾을 수 있으면) 장애물 충돌 여부를 판정한다.

   2026-09-10: run_random_batch_v2.py의 dial_mpc_ok가 지금까지 states.npy 파일
   생성 여부만으로 판정되고 있었던 게 확인되어(낙상/충돌 전혀 미검증), 이 파일의
   판정 로직을 check_scene_safety() 함수로 뽑아내 배치 스크립트에서도 재사용
   가능하게 함. CLI 동작(__main__)은 기존과 동일.

 [사용법] python check_fall.py <path/to/xxx_states.npy>
          또는: from check_fall import check_scene_safety
================================================================================
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_scene_safety(states_path, verbose=True):
    """
    낙상 판정 + 장애물 충돌 판정(가능한 경우)을 모두 수행해서 dict로 반환.
    반환: {"fell": bool, "fall_step": int|None,
           "collision_detected": bool|None,  # None=판정 불가(씬 XML 없음/에러)
           "min_clearance_m": float|None,
           "safe": bool}  # fell=False AND collision_detected != True 일 때만 True
    """
    data = np.load(states_path, allow_pickle=True)
    t = data[:, 0]
    qpos = data[:, 1:20]
    z = qpos[:, 2]
    quat = qpos[:, 3:7]
    w, x, y, yq = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    R22 = 1 - 2 * (x**2 + y**2)
    tilt_done = R22 < 0
    height_done = z < 0.18
    done = tilt_done | height_done

    if verbose:
        print(f"total steps: {len(t)}")
        print(f"z-height: min={z.min():.3f}, max={z.max():.3f}, final={z[-1]:.3f}")
        print(f"R22 (tilt, up-dot): min={R22.min():.3f}, final={R22[-1]:.3f}")

    fell = bool(done.any())
    fall_step = None
    if fell:
        first_fall_idx = int(np.argmax(done))
        fall_step = int(t[first_fall_idx])
        if verbose:
            reason = []
            if tilt_done[first_fall_idx]:
                reason.append("tilt(전복)")
            if height_done[first_fall_idx]:
                reason.append("height(주저앉음)")
            print(f"\n>>> FALL DETECTED at step {fall_step} (reason: {', '.join(reason)})")
            print(f"    z={z[first_fall_idx]:.3f}, R22={R22[first_fall_idx]:.3f}")
    elif verbose:
        print("\n>>> NO FALL (done never triggered) — 로봇이 끝까지 서서 걸었음")

    scene = os.path.basename(os.path.dirname(os.path.abspath(states_path)))
    scene_xml = os.path.join(_REPO_ROOT, "dial_mpc", "dial_mpc", "models", "unitree_go2", f"{scene}.xml")

    collision_detected = None
    min_clearance_m = None
    if verbose:
        print(f"\n[collision check] scene='{scene}', xml='{scene_xml}'")
    if not os.path.exists(scene_xml):
        if verbose:
            print(f">>> SKIP: 씬 XML을 찾을 수 없음 ({scene_xml}) — 충돌 판정 생략")
    else:
        try:
            import mujoco
            from baselines.occupancy_grid import obstacles_from_mjmodel
            from analysis.metrics import obstacle_clearance

            model = mujoco.MjModel.from_xml_path(scene_xml)
            obstacles = obstacles_from_mjmodel(model)
            box_obstacles = [o for o in obstacles if o["kind"] == "box"]
            result = obstacle_clearance(qpos, obstacles)
            min_clearance_m = result["min_clearance_m"]
            collision_detected = min_clearance_m < 0
            if verbose:
                print(f"    obstacles found: {len(obstacles)} (box: {len(box_obstacles)})")
                print(f"    min_clearance_m={result['min_clearance_m']:.4f}  "
                      f"mean_clearance_m={result['mean_clearance_m']:.4f}  "
                      f"pct_steps_colliding={result['pct_steps_colliding']*100:.1f}%")
                if collision_detected:
                    print(f"\n>>> COLLISION DETECTED: 로봇 몸체가 장애물에 최대 "
                          f"{-min_clearance_m:.4f}m 침투함 (전체 스텝의 "
                          f"{result['pct_steps_colliding']*100:.1f}%에서 발생)")
                else:
                    print("\n>>> NO COLLISION (min_clearance_m >= 0) — 장애물과 실제로 겹친 적 없음")
        except Exception as e:
            if verbose:
                print(f">>> 충돌 판정 실패 (에러): {e!r} — 위의 낙상 판정 결과만 신뢰할 것")

    safe = (not fell) and (collision_detected is not True)

    if verbose:
        if fell:
            print("\n=== FINAL: FAIL (낙상) ===")
        elif collision_detected is True:
            print("\n=== FINAL: FAIL (충돌, 낙상은 아님) ===")
        elif collision_detected is False:
            print("\n=== FINAL: PASS (낙상 없음 + 충돌 없음) ===")
        else:
            print("\n=== FINAL: PASS (낙상 없음, 충돌 판정 불가/생략됨 — 주의해서 해석할 것) ===")

    return {"fell": fell, "fall_step": fall_step,
            "collision_detected": collision_detected,
            "min_clearance_m": min_clearance_m, "safe": safe}


if __name__ == "__main__":
    check_scene_safety(sys.argv[1], verbose=True)
