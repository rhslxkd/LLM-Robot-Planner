import numpy as np
from analysis.load_rollout import load_latest_rollout

data = load_latest_rollout("oracle_scene_D/becoy_skeleton")
qpos = data["qpos"]
base_z = qpos[:, 2]
qx, qy = qpos[:, 4], qpos[:, 5]
tilt_deg = np.degrees(np.arccos(np.clip(1 - 2*(qx**2 + qy**2), -1.0, 1.0)))

t_min_height = int(np.argmin(base_z))
t_max_tilt = int(np.argmax(tilt_deg))
T = len(base_z)
print(f"total steps: {T}")
print(f"min height {base_z[t_min_height]:.3f}m at step {t_min_height} ({100*t_min_height/T:.1f}% of run)")
print(f"max tilt {tilt_deg[t_max_tilt]:.1f}deg at step {t_max_tilt} ({100*t_max_tilt/T:.1f}% of run)")
