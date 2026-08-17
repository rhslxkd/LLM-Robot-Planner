"""
run_astar_baseline.py -- runs the A* baseline (paper/main.tex Sec. 1.2/3:
"single deterministic path" contrast against the courtroom) against a real
oracle_scene_{A,D,E}.xml, and writes a waypoint JSON in the SAME schema as
the courtroom's own vlm_courtroom/outputs/last_judged_path.json
(list of {"x": float, "y": float} dicts, world-frame, robot-origin-relative)
so it can be fed into DIAL-MPC unchanged for a like-for-like comparison.

*** MUST RUN ON THE LAB SERVER *** (needs `mujoco` + the actual scene XMLs,
neither of which exist in this drafting environment). This script was
written and unit-tested against SYNTHETIC obstacles only (see
occupancy_grid.py / astar_planner.py __main__ blocks) -- verify the
assumptions below against your current code before trusting the output:

  1. Effective clearance = 0.8m, minimum passable gap = 1.6m: taken from
     paper/main.tex Table 1 (ROBOT_PHYSICAL_CONSTRAINTS). Your local mounted
     repo's vlm_courtroom/agents/specific_agents.py still has an OLDER
     0.5m-margin/0.8m-gap constraint (and your branch was showing "21 commits
     behind origin/feature-oracle_gen" in your VS Code terminal) -- pull
     first and confirm which constraint values are actually live before
     trusting a head-to-head comparison against the courtroom.
  2. last_judged_path.json schema: read directly from
     vlm_courtroom/court/courtroom.py (list of {"x","y"} dicts). If your
     current (post-pull) pipeline saves to a different path/schema
     (e.g. a data/<scene>/<variant>/ layout), adjust OUT_PATH below.
  3. Obstacles (including corridor/room walls) are assumed to be static
     MuJoCo bodies with box/cylinder/sphere geoms, same as oracle_gen.py's
     _report_obstacles() assumes -- i.e. no separate wall-plane geometry
     type. If Scene D/E model walls differently, extend
     occupancy_grid.obstacles_from_mjmodel() accordingly.

Usage (on the lab server, from repo root):
    python baselines/run_astar_baseline.py \
        dial_mpc/dial_mpc/models/unitree_go2/oracle_scene_A.xml \
        --start 0 0 --goal 5 0 --clearance 0.8 \
        --out data/A/astar/last_judged_path.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occupancy_grid import GridSpec, obstacles_from_mjmodel, build_occupancy_grid, inflate_grid
from astar_planner import astar_grid, simplify_path, path_to_world_waypoints

EFFECTIVE_CLEARANCE_M = 0.8  # paper/main.tex Table 1 -- keep in sync manually
MARGIN_M = 1.0               # extra grid margin around start/goal so A* isn't clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_xml", help="path to oracle_scene_X.xml")
    ap.add_argument("--start", nargs=2, type=float, default=[0.0, 0.0], metavar=("X", "Y"))
    ap.add_argument("--goal", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--clearance", type=float, default=EFFECTIVE_CLEARANCE_M)
    ap.add_argument("--epsilon-px", type=float, default=3.0, help="RDP simplification tolerance")
    ap.add_argument("--out", required=True, help="output path for last_judged_path.json-style JSON")
    args = ap.parse_args()

    import mujoco  # only needed here, not in occupancy_grid/astar_planner themselves

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    obstacles = obstacles_from_mjmodel(model)
    print(f"[run_astar_baseline] loaded {len(obstacles)} static obstacles from {args.scene_xml}")
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
    raw_path = astar_grid(inflated, start_px, goal_px)

    if raw_path is None:
        print(f"[run_astar_baseline] NO PATH FOUND (start or goal blocked, or no passable route "
              f"at {args.clearance}m clearance). This is itself a valid data point for the "
              f"comparison table (A* fails where the courtroom may still find a route).")
        result = {"scene_xml": args.scene_xml, "start": args.start, "goal": args.goal,
                  "clearance_m": args.clearance, "path_found": False, "waypoints": []}
    else:
        simplified = simplify_path(raw_path, epsilon_px=args.epsilon_px)
        wps = path_to_world_waypoints(simplified, spec)
        print(f"[run_astar_baseline] path found: {len(raw_path)} px-steps -> {len(wps)} waypoints")
        result = {"scene_xml": args.scene_xml, "start": args.start, "goal": args.goal,
                  "clearance_m": args.clearance, "path_found": True,
                  "waypoints": [{"x": round(x, 4), "y": round(y, 4)} for x, y in wps]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        # last_judged_path.json schema is just the waypoint list; the extra
        # metadata (path_found, clearance_m, etc.) is written to a sibling
        # file for the comparison-table bookkeeping instead of polluting the
        # DIAL-MPC-consumed file.
        json.dump(result["waypoints"], f, indent=2)
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_astar_baseline] wrote waypoints -> {args.out}")
    print(f"[run_astar_baseline] wrote metadata  -> {meta_path}")


if __name__ == "__main__":
    main()
