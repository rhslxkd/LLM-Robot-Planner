"""
astar_planner.py -- grid-based A* baseline #1 for VLM2Torque's
diverse-path-generation comparison (paper/main.tex Sec. 1.2 Related Work /
Sec. 3 baselines): a classical, deterministic single-path planner to
contrast against the courtroom's ability to generate multiple valid paths.

Takes the boolean occupancy grid from occupancy_grid.py (already inflated by
the robot's effective clearance) and returns ONE shortest waypoint sequence
from start to goal, simplified down to a small number of waypoints and
expressed in WORLD coordinates -- same format the courtroom's Judge agent
outputs -- so it can be fed to DIAL-MPC unchanged for a like-for-like
physical-execution comparison.
"""
import heapq
import numpy as np

# 8-connected moves: (du, dv, step_cost)
_MOVES = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (-1, -1, 2 ** 0.5),
]


def astar_grid(grid, start_px, goal_px):
    """
    grid: (H, W) bool array, True = occupied/blocked.
    start_px, goal_px: (u, v) pixel coordinates (rounded to nearest cell).
    Returns a list of (u, v) pixel waypoints from start to goal (inclusive),
    or None if no path exists (e.g. a corridor narrower than the courtroom's
    own 1.6m minimum-passable-width threshold, once inflated).
    """
    h, w = grid.shape
    start = (int(round(start_px[0])), int(round(start_px[1])))
    goal = (int(round(goal_px[0])), int(round(goal_px[1])))

    def blocked(u, v):
        return not (0 <= u < w and 0 <= v < h) or grid[v, u]

    if blocked(*start) or blocked(*goal):
        return None

    def heuristic(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for du, dv, cost in _MOVES:
            nxt = (current[0] + du, current[1] + dv)
            if nxt in visited or blocked(*nxt):
                continue
            ng = g + cost
            if ng < g_score.get(nxt, float("inf")):
                g_score[nxt] = ng
                came_from[nxt] = current
                heapq.heappush(open_heap, (ng + heuristic(nxt, goal), ng, nxt))
    return None


def simplify_path(path_px, epsilon_px=3.0):
    """
    Ramer-Douglas-Peucker simplification so the raw per-grid-cell path
    (hundreds of points) collapses to a small waypoint sequence comparable
    in count to the courtroom's (10-15 waypoints), rather than one point per
    pixel step.
    """
    if len(path_px) <= 2:
        return list(path_px)

    def _rdp(pts):
        if len(pts) < 3:
            return pts
        start, end = np.array(pts[0], dtype=float), np.array(pts[-1], dtype=float)
        line = end - start
        line_len = np.linalg.norm(line)
        if line_len == 0:
            dists = [float(np.linalg.norm(np.array(p) - start)) for p in pts[1:-1]]
        else:
            # 2D cross product magnitude (point-to-line distance numerator);
            # computed manually since np.cross on 2D vectors is deprecated
            # as of NumPy 2.0.
            dists = [
                float(abs(line[0] * (p[1] - start[1]) - line[1] * (p[0] - start[0])) / line_len)
                for p in pts[1:-1]
            ]
        if not dists:
            return [pts[0], pts[-1]]
        idx = int(np.argmax(dists)) + 1
        if dists[idx - 1] > epsilon_px:
            left = _rdp(pts[: idx + 1])
            right = _rdp(pts[idx:])
            return left[:-1] + right
        return [pts[0], pts[-1]]

    return _rdp(list(path_px))


def path_to_world_waypoints(path_px, spec):
    """spec: occupancy_grid.GridSpec (has .pixel_to_world)."""
    return [spec.pixel_to_world(u, v) for u, v in path_px]


if __name__ == "__main__":
    from occupancy_grid import GridSpec, build_occupancy_grid, inflate_grid

    print("[Scene A-like] A* should route around the obstacle, not through it")
    spec_a = GridSpec(x_min=-1.0, x_max=6.0, y_min=-3.0, y_max=3.0)
    obs_a = [{"name": "obs1", "pos": (2.0, 0.0), "kind": "cylinder", "size": (0.3, 0.5, 0.0)}]
    grid_a = inflate_grid(build_occupancy_grid(spec_a, obs_a), clearance_m=0.8, ppm=spec_a.ppm)
    start_px = spec_a.world_to_pixel(0.0, 0.0)
    goal_px = spec_a.world_to_pixel(5.0, 0.0)
    raw = astar_grid(grid_a, start_px, goal_px)
    assert raw is not None, "A* failed to find a path around a single small obstacle"
    simplified = simplify_path(raw, epsilon_px=3.0)
    wps = path_to_world_waypoints(simplified, spec_a)
    print(f"  raw path length: {len(raw)} px-steps -> simplified to {len(wps)} waypoints")
    print("  waypoints (world m):", [f"({x:.2f},{y:.2f})" for x, y in wps])
    max_abs_y = max(abs(y) for _, y in wps)
    print(f"  max |y| deviation: {max_abs_y:.2f} m (should be > 0: it must detour around the obstacle)")

    print("\n[Scene D-like, too narrow] A* should report NO path (corridor < 1.6m min)")
    spec_d = GridSpec(x_min=-1.0, x_max=8.0, y_min=-2.0, y_max=2.0)
    walls_narrow = [((-1.0, 0.75), (8.0, 0.75)), ((-1.0, -0.75), (8.0, -0.75))]
    grid_d_narrow = inflate_grid(build_occupancy_grid(spec_d, [], walls=walls_narrow), clearance_m=0.8, ppm=spec_d.ppm)
    p = astar_grid(grid_d_narrow, spec_d.world_to_pixel(0.0, 0.0), spec_d.world_to_pixel(7.0, 0.0))
    print(f"  result: {'None (correctly impassable)' if p is None else f'{len(p)} px-steps (UNEXPECTED)'}")

    print("\n[Scene D-like, OK width] A* should find exactly one straight-ish path")
    walls_ok = [((-1.0, 0.85), (8.0, 0.85)), ((-1.0, -0.85), (8.0, -0.85))]
    grid_d_ok = inflate_grid(build_occupancy_grid(spec_d, [], walls=walls_ok), clearance_m=0.8, ppm=spec_d.ppm)
    p_ok = astar_grid(grid_d_ok, spec_d.world_to_pixel(0.0, 0.0), spec_d.world_to_pixel(7.0, 0.0))
    assert p_ok is not None
    wps_ok = path_to_world_waypoints(simplify_path(p_ok, epsilon_px=3.0), spec_d)
    print(f"  waypoints: {len(wps_ok)}  max |y| deviation: {max(abs(y) for _, y in wps_ok):.3f} m (should be ~0: straight down centerline)")
