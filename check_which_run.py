import numpy as np
from analysis.load_rollout import list_rollout_files, load_rollout

files = list_rollout_files("oracle_scene_D/gemini_wp15")
for f in files:
    data = load_rollout(f)
    y = data["qpos"][:, 1]
    print(f"{f}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]  (통과=0 근처, 우회=2m대)")
