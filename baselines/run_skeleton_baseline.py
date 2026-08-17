"""
run_skeleton_baseline.py -- runs the Becoy et al. (2025) morphological-
skeleton baseline (adapted for point-to-point navigation -- see
skeleton_planner.py's module docstring for exactly what was kept/dropped
from the original coverage-planning algorithm) against a real
oracle_scene_{A,D,E}.xml, and writes a waypoint JSON in the SAME schema as
the courtroom's own last_judged_path.json (list of {"x": float, "y": float}
dicts, world-frame, robot-origin-relative) -- directly comparable to
run_astar_baseline.py's output for the same scene/start/goal/clearance, and
feedable into DIAL-MPC unchanged.

*** MUST RUN ON THE LAB SERVER *** (needs `mujoco` + the real oracle_scene_*
XMLs, same as run_astar_baseline.py). Also needs scikit-image:
    pip install scikit-image --break-system-packages

Usage (on the lab server, from repo root) -- mirror run_astar_baseline.py's
start/goal/clearance exactly for an apples-to-apples table row:
    python baselines/run_skeleton_baseline.py \
        dial_mpc/dial_mpc/models/unitree_go2/oracle_scene_D.xml \
        --start 0 0 --goal 7 0 --clearance 0.8 \
        --out data/oracle_scene_D/becoy_skeleton/last_judged_path.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occupancy_grid import GridSpec, obstacles_from_mjmodel, build_occupancy_grid, inflate_grid
from astar_planner import simplify_path, path_to_world_waypoints
from skeleton_planner import plan_skeleton_path

EFFECTIVE_CLEARANCE_M = 0.8  # keep in sync with run_astar_baseline.py / paper/main.tex Table 1
MARGIN_M = 1.0               # extra grid margin around start/goal so the skeleton isn't clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_xml", help="path to oracle_scene_X.xml")
    ap.add_argument("--start", nargs=2, type=float, default=[0.0, 0.0], metavar=("X", "Y"))
    ap.add_argument("--goal", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--clearance", type=float, default=EFFECTIVE_CLEARANCE_M)
    ap.add_argument("--epsilon-px", type=float, default=3.0, help="RDP simplification tolerance")
    ap.add_argument("--out", required=True, help="output path for last_judged_path.json-style JSON")
    args = ap.parse_args()

    import mujoco  # only needed here, not in occupancy_grid/astar_planner/skeleton_planner themselves

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    obstacles = obstacles_from_mjmodel(model)
    print(f"[run_skeleton_baseline] loaded {len(obstacles)} static obstacles from {args.scene_xml}")
    for o in obstacles:
        print(f"    - {o['name']}: pos={o['pos']} kind={o['kind']} size={o['size']}")

    sx, sy = args.start
    gx, gy = args.goal
    spec = GridSpec(
        x_min=min(sx, gx) - MARGIN_M, x_max=max(sx, gx) + MARGIN_M,
        y_min=min(sy, gy) - MARGIN_M - 2.0, y_max=max(sy, gy) + MARGIN_M + 2.0,
    )

    grid = build_occupancy_grid(spec, obstacles)
    inflated = inflate_grid(grid, clearance_m=args.clearance, ppm=spec.ppm)

    start_px = spec.world_to_pixel(sx, sy)
    goal_px = spec.world_to_pixel(gx, gy)
    raw_path_px, skel_meta = plan_skeleton_path(inflated, start_px, goal_px)
    print(f"[run_skeleton_baseline] skeleton meta: {skel_meta}")

    if raw_path_px is None:
        print(f"[run_skeleton_baseline] NO PATH FOUND ({skel_meta.get('reason')}) at {args.clearance}m "
              f"clearance. This is itself a valid data point for the comparison table (skeleton method "
              f"fails where the courtroom may still find a route, same as the A* baseline).")
        result = {"scene_xml": args.scene_xml, "start": args.start, "goal": args.goal,
                  "clearance_m": args.clearance, "path_found": False, "waypoints": [],
                  "skeleton_meta": skel_meta}
    else:
        simplified = simplify_path(raw_path_px, epsilon_px=args.epsilon_px)
        wps = path_to_world_waypoints(simplified, spec)
        print(f"[run_skeleton_baseline] path found: {len(raw_path_px)} px-steps -> {len(wps)} waypoints")
        result = {"scene_xml": args.scene_xml, "start": args.start, "goal": args.goal,
                  "clearance_m": args.clearance, "path_found": True,
                  "waypoints": [{"x": round(x, 4), "y": round(y, 4)} for x, y in wps],
                  "skeleton_meta": skel_meta}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        # last_judged_path.json schema is just the waypoint list; the extra
        # metadata (path_found, clearance_m, skeleton_meta, etc.) is written
        # to a sibling file for comparison-table bookkeeping instead of
        # polluting the DIAL-MPC-consumed file.
        json.dump(result["waypoints"], f, indent=2)
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_skeleton_baseline] wrote waypoints -> {args.out}")
    print(f"[run_skeleton_baseline] wrote metadata  -> {meta_path}")


if __name__ == "__main__":
    main()
