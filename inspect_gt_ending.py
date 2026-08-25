import numpy as np

arr = np.load("data/oracle_scene_R001/dial_gt/20260825-140143_states.npy")
step, x, y, z, qw = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

GOAL = (4.56, 1.35)
dist = np.hypot(x - GOAL[0], y - GOAL[1])

print("목표에 처음 0.5m 이내로 들어온 시점:")
close_idx = np.argmax(dist < 0.5) if np.any(dist < 0.5) else -1
if close_idx >= 0:
    print(f"  step {int(step[close_idx])}, pos=({x[close_idx]:.2f},{y[close_idx]:.2f}), dist={dist[close_idx]:.3f}")

print("\nstep 900부터 끝까지 50스텝 간격 x,y,z,qw,dist:")
for i in range(900, len(step), 50):
    print(f"  step {int(step[i]):4d}  x={x[i]:6.2f} y={y[i]:6.2f} z={z[i]:6.3f} qw={qw[i]:6.3f} dist={dist[i]:5.2f}")
print(f"  step {int(step[-1]):4d}  x={x[-1]:6.2f} y={y[-1]:6.2f} z={z[-1]:6.3f} qw={qw[-1]:6.3f} dist={dist[-1]:5.2f}  (마지막)")

print("\nz>0.15 & qw>0.7 (정상 자세) 마지막으로 유지된 시점:")
ok = (z > 0.15) & (qw > 0.7)
ok_idx = np.where(ok)[0]
if len(ok_idx):
    last_ok = ok_idx[-1]
    tail = "-- 이후 끝까지 비정상" if last_ok < len(step) - 1 else "-- 끝까지 정상"
    print(f"  step {int(step[last_ok])}, pos=({x[last_ok]:.2f},{y[last_ok]:.2f}), dist={dist[last_ok]:.3f}  {tail}")
