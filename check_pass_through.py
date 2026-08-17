import numpy as np
from analysis.load_rollout import load_latest_rollout
from analysis.metrics import summarize

data = load_latest_rollout("oracle_scene_D/gemini_wp12")
y = data["qpos"][:, 1]
print(f"T={data['qpos'].shape[0]} steps, y range: [{y.min():.3f}, {y.max():.3f}]")
m = summarize(data)
for k, v in m.items():
    print(f"  {k}: {v:.4f}")
