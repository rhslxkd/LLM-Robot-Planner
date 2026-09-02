import os
import sys
import json

sys.path.insert(0, ".")
from vlm_courtroom.court.courtroom import deterministic_correct

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
NEW_MIN_WALL_DIST_M = 0.6

SCENES = ["R000", "R001", "R002", "R003", "R004", "R006", "R008",
          "R009", "R010", "R011", "R012", "R013", "R015"]

for s in SCENES:
    scene = f"oracle_scene_{s}"
    with open(f"data/{scene}/neural_astar/coordinate_proposal.json") as f:
        coords = json.load(f)

    image_path = f"data/{scene}/oracle.png"
    final_coords, passed, log = deterministic_correct(
        image_path, coords, ROBOT_PX, PPM,
        min_wall_dist_m=NEW_MIN_WALL_DIST_M,
    )

    out_dir = f"data/{scene}/no_vlm_06"
    os.makedirs(out_dir, exist_ok=True)

    dial_path = [{"x": c["x"], "y": c["y"]} for c in final_coords]
    with open(f"{out_dir}/last_judged_path.json", "w") as f:
        json.dump(dial_path, f, indent=2)

    with open(f"{out_dir}/log.txt", "w") as f:
        f.write(f"PASSED: {passed}\nmin_wall_dist_m={NEW_MIN_WALL_DIST_M}\n\n")
        f.write("\n".join(log))

    print(f"{scene}: PASSED={passed}, waypoints={len(final_coords)} -> {out_dir}/last_judged_path.json")
