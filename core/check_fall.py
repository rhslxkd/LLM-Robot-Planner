"""
================================================================================
 [STUDY VERSION] check_fall.py -- 함수/줄 단위 상세 해설판
================================================================================
 원본: core/check_fall.py (독립 실행형 유틸리티 -- run_random_batch_v2.py 등
       다른 스크립트가 import해서 쓰는 게 아니라, DIAL-MPC 결과 states.npy 파일을
       사람이 직접 커맨드라인에서 돌려보는 "수동 진단 도구")
 
 [역할]
   DIAL-MPC 시뮬레이션이 끝난 뒤 저장되는 로봇 상태 궤적(*_states.npy)을 읽어서,
   로봇이 시뮬레이션 도중에 "넘어졌는지"(fall) 여부와, 넘어졌다면 몇 스텝째에
   어떤 이유(전복/주저앉음)로 넘어졌는지를 판정하는 스크립트.
 
 [사용법] python check_fall.py <path/to/xxx_states.npy>
================================================================================
"""
import sys
import numpy as np

states_path = sys.argv[1]
data = np.load(states_path, allow_pickle=True)  # (T, 80)

t = data[:, 0]
qpos = data[:, 1:20]      # 19
z = qpos[:, 2]            # torso height
quat = qpos[:, 3:7]       # w,x,y,z (mujoco convention)
w, x, y, yq = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

# rotate world-up [0,0,1] by quat, take dot with up = R[2,2]
R22 = 1 - 2 * (x**2 + y**2)

tilt_done = R22 < 0          # 90도 이상 전복
height_done = z < 0.18       # 로봇 몸통이 바닥에 닿을 정도로 낮음
done = tilt_done | height_done

print(f"total steps: {len(t)}")
print(f"z-height: min={z.min():.3f}, max={z.max():.3f}, final={z[-1]:.3f}")
print(f"R22 (tilt, up-dot): min={R22.min():.3f}, final={R22[-1]:.3f}")

if done.any():
    first_fall_idx = int(np.argmax(done))
    reason = []
    if tilt_done[first_fall_idx]:
        reason.append("tilt(전복)")
    if height_done[first_fall_idx]:
        reason.append("height(주저앉음)")
    print(f"\n>>> FALL DETECTED at step {int(t[first_fall_idx])} (reason: {', '.join(reason)})")
    print(f"    z={z[first_fall_idx]:.3f}, R22={R22[first_fall_idx]:.3f}")
else:
    print("\n>>> NO FALL (done never triggered) — 로봇이 끝까지 서서 걸었음")
