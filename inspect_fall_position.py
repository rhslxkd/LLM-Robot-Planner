import sys
import glob
import numpy as np

scene = sys.argv[1]
goal_x = float(sys.argv[2])
goal_y = float(sys.argv[3])

path = sorted(glob.glob(f"data/{scene}/batch/*_states.npy"))[-1]
data = np.load(path, allow_pickle=True)

t = data[:, 0]
qpos = data[:, 1:20]
x, y, z = qpos[:, 0], qpos[:, 1], qpos[:, 2]
quat = qpos[:, 3:7]
qx, qy = quat[:, 1], quat[:, 2]
R22 = 1 - 2 * (qx ** 2 + qy ** 2)
tilt_done = R22 < 0
height_done = z < 0.18
done = tilt_done | height_done

idx = int(np.argmax(done)) if done.any() else len(t) - 1
dist_to_goal = ((x[idx] - goal_x) ** 2 + (y[idx] - goal_y) ** 2) ** 0.5

print(f"scene={scene}")
print(f"fall/end step={int(t[idx])} / total={len(t)}")
print(f"robot pos at that step: x={x[idx]:.2f}, y={y[idx]:.2f}, z={z[idx]:.3f}")
print(f"goal: x={goal_x:.2f}, y={goal_y:.2f}")
print(f"dist_to_goal at fall = {dist_to_goal:.2f} m")
print(f"start pos: x={x[0]:.2f}, y={y[0]:.2f}")
print(f"total path straight-line dist (start->goal) = {((x[0]-goal_x)**2+(y[0]-goal_y)**2)**0.5:.2f} m")

lo = max(0, idx - 20)
print(f"\nz trajectory steps [{lo}:{idx+1}]:")
print(np.round(z[lo:idx+1], 3))
