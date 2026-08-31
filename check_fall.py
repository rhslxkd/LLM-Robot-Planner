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
