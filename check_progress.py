import sys
import numpy as np

states_path = sys.argv[1]
data = np.load(states_path, allow_pickle=True)  # (T, 80)

t = data[:, 0]
qpos = data[:, 1:20]
x, y = qpos[:, 0], qpos[:, 1]

dx = np.diff(x)
dy = np.diff(y)
step_dist = np.sqrt(dx**2 + dy**2)
cum_dist = np.concatenate([[0], np.cumsum(step_dist)])

print(f"total steps: {len(t)}")
print(f"start xy=({x[0]:.3f},{y[0]:.3f})  end xy=({x[-1]:.3f},{y[-1]:.3f})")
print(f"직선거리(start->end) = {np.hypot(x[-1]-x[0], y[-1]-y[0]):.3f}m")
print(f"누적 이동거리(실제 걸은 거리) = {cum_dist[-1]:.3f}m")

window = 100
print("\n[100-step 구간별 이동거리]")
for i in range(0, len(t) - 1, window):
    j = min(i + window, len(t) - 1)
    seg = cum_dist[j] - cum_dist[i]
    print(f"  step {i:4d}-{j:4d}: {seg:.3f}m  (누적 {cum_dist[j]:.3f}m)")

last_seg = cum_dist[-1] - cum_dist[max(0, len(t) - 1 - window)]
mid_idx = len(t) // 2
mid_seg = cum_dist[mid_idx + window] - cum_dist[mid_idx] if mid_idx + window < len(t) else None
print(f"\n마지막 {window}스텝 이동거리: {last_seg:.3f}m" + (f" (중반부 {window}스텝: {mid_seg:.3f}m)" if mid_seg else ""))
if mid_seg and last_seg < mid_seg * 0.3:
    print(">>> 후반부에 거의 안 움직임 -> 일찍 도착해서 제자리걸음했을 가능성 높음. n_steps 줄여도 될 듯.")
else:
    print(">>> 끝까지 꾸준히 이동 중이었음 -> n_steps 줄이면 목표 못 미칠 수도 있음.")
