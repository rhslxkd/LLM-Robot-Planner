import numpy as np
from analysis.load_rollout import load_rollout
from analysis.metrics import summarize

files = {
    "detour_v1_collision_reported": "data/oracle_scene_D/gemini_wp15/20260811-180450_states.npy",
    "detour_v2_avoid_fixed": "data/oracle_scene_D/gemini_wp15/20260811-182526_states.npy",
}

for label, path in files.items():
    data = load_rollout(path)
    m = summarize(data)
    print(f"{label} (T={data['qpos'].shape[0]} steps):")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
    print()
