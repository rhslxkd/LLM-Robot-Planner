import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOAL = (4.56, 1.35)

FILES = {
    "GT": "data/oracle_scene_R001/dial_gt/20260825-140143_states.npy",
    "NA": "data/oracle_scene_R001/dial_na/20260825-141155_states.npy",
}

Z_FALL_THRESH = 0.15   # 이 높이 밑으로 떨어지면 넘어졌다고 판단 (초기 z~0.27)
QW_TILT_THRESH = 0.7   # 이 밑으로 떨어지면 90도 이상 기울었다고 판단

plt.figure(figsize=(7, 7))

for label, path in FILES.items():
    arr = np.load(path)
    step = arr[:, 0]
    x, y, z = arr[:, 1], arr[:, 2], arr[:, 3]
    qw = arr[:, 4]

    dist_to_goal = np.hypot(x - GOAL[0], y - GOAL[1])
    closest_idx = int(np.argmin(dist_to_goal))
    closest_dist = dist_to_goal[closest_idx]

    fall_idx = np.argmax(z < Z_FALL_THRESH) if np.any(z < Z_FALL_THRESH) else -1
    tilt_idx = np.argmax(qw < QW_TILT_THRESH) if np.any(qw < QW_TILT_THRESH) else -1

    print(f"\n=== {label} ===")
    print(f"  총 스텝: {len(step)}")
    print(f"  목표까지 최단 거리: {closest_dist:.3f}m (step {closest_idx}, pos=({x[closest_idx]:.2f},{y[closest_idx]:.2f}))")
    print(f"  최종 위치: ({x[-1]:.2f},{y[-1]:.2f}), 최종 목표거리: {dist_to_goal[-1]:.3f}m")
    print(f"  z<{Z_FALL_THRESH}m(넘어짐 의심) 최초 시점: {'없음' if fall_idx==-1 else f'step {int(step[fall_idx])} (pos=({x[fall_idx]:.2f},{y[fall_idx]:.2f}))'}")
    print(f"  qw<{QW_TILT_THRESH}(90도+ 기울어짐) 최초 시점: {'없음' if tilt_idx==-1 else f'step {int(step[tilt_idx])} (pos=({x[tilt_idx]:.2f},{y[tilt_idx]:.2f}))'}")

    plt.plot(x, y, label=f"{label} trajectory", linewidth=1.5)
    plt.scatter([x[0]], [y[0]], marker='o', s=40)
    plt.scatter([x[-1]], [y[-1]], marker='x', s=60)

plt.scatter([GOAL[0]], [GOAL[1]], marker='*', s=200, color='red', label='goal', zorder=5)
plt.scatter([0], [0], marker='s', s=60, color='black', label='start', zorder=5)
plt.gca().set_aspect('equal')
plt.legend()
plt.title("DIAL-MPC actual walked trajectory: GT vs Neural A* original")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("data/oracle_scene_R001/dial_trajectory_comparison.png", dpi=150)
print("\n저장됨: data/oracle_scene_R001/dial_trajectory_comparison.png")
