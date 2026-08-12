"""
skeleton_planner.py -- morphological-skeleton path-planning baseline,
adapted for point-to-point navigation, based on:

  Becoy AJ, Khomenko K, Peternel L, Rajan RT (2025). "Autonomous navigation of
  quadrupeds using coverage path planning with morphological skeleton maps."
  Frontiers in Robotics and AI 12:1601862. doi:10.3389/frobt.2025.1601862

Reimplements the core technique from the paper's Section 3.1 (Map reader --
reduce free space to its 1-px-wide morphological/topological skeleton) and
Section 3.2 (Path planning -- build a graph from the skeleton and search it),
WITHOUT the coverage-specific parts:

  - The paper's Algorithm 1 (map reader) additionally does SLAM-map-specific
    cleanup (Gaussian smoothing of sensor noise, largest-contour extraction
    to discard unreachable regions) that doesn't apply here -- our occupancy
    grid is built directly and noise-free from known MuJoCo obstacle
    geometry (occupancy_grid.py), so those steps are skipped. The erosion
    step IS kept, via occupancy_grid.inflate_grid (same clearance-radius
    semantics used by the A* baseline and the courtroom's own safety rule,
    see paper/main.tex Table 1 / vlm_courtroom/agents/specific_agents.py) --
    this plays the same structural role as the paper's Erode step, just
    parameterized in meters instead of pixels.
  - The paper's Algorithm 2 solves *coverage*: visit every leaf/dead-end of
    the skeleton graph in sequence (repeated nearest-unvisited-leaf +
    Dijkstra), because the goal is to scan the whole room. Our comparison
    task is *point-to-point* navigation (same start/goal the courtroom and
    astar_planner.py are given), so we keep the paper's central idea --
    search a graph built from the topological skeleton via Dijkstra -- but
    reduce the objective to a single shortest path from the skeleton node
    nearest `start` to the skeleton node nearest `goal`. This preserves
    exactly what's under test (a deterministic, single fixed path per map)
    while matching this repo's point-to-point comparison protocol used by
    astar_planner.py / run_astar_baseline.py.

Caveat worth keeping in mind when reading results: skimage.morphology.
skeletonize treats the array boundary itself as part of the free-space
shape, so for open scenes (e.g. Scene A, no walls) the skeleton's overall
branch structure is somewhat sensitive to how large occupancy_grid.GridSpec's
bounding box is (there's no real wall there to anchor it) -- this matches
Scene D/E well (real walls define the region) but is a soft caveat for
open/unbounded scenes worth noting in the comparison writeup.

Needs scikit-image (not required by occupancy_grid.py / astar_planner.py):
    pip install scikit-image --break-system-packages
"""
import heapq
import math

import numpy as np
from skimage.morphology import skeletonize


def skeletonize_free_space(inflated_grid: np.ndarray) -> np.ndarray:
    """
    inflated_grid: (H, W) bool array, True = occupied/too-close-to-obstacle
                   -- this is occupancy_grid.inflate_grid's output, i.e. the
                   clearance-radius erosion step already applied (plays the
                   role of the paper's Algorithm 1 Erode step).
    Returns: (H, W) bool array, True = skeleton pixel (paper's Algorithm 1
             Skeletonize step -- a 1-pixel-wide topological skeleton of the
             free space).
    """
    free = ~inflated_grid
    return skeletonize(free)


def _skeleton_neighbors(skeleton: np.ndarray, u: int, v: int):
    """
    8-connected (u, v) neighbors of (u, v) that are also skeleton pixels.
    skeleton is indexed [v, u] (row=v, col=u), matching occupancy_grid.py's
    grid[v, u] convention (and astar_planner.astar_grid's blocked(u, v)).
    """
    h, w = skeleton.shape
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            if du == 0 and dv == 0:
                continue
            nu, nv = u + du, v + dv
            if 0 <= nu < w and 0 <= nv < h and skeleton[nv, nu]:
                yield nu, nv, math.hypot(du, dv)


def nearest_skeleton_pixel(skeleton: np.ndarray, uv: tuple):
    """
    Find the skeleton pixel closest (Euclidean, pixel space) to uv=(u, v),
    same (u, v) = (col, row) pixel convention as astar_planner.astar_grid's
    start_px/goal_px. Returns (u, v) of the nearest skeleton pixel, or None
    if the skeleton is empty (e.g. clearance so large the whole grid is
    blocked -- itself a valid, reportable "impassable" outcome).
    """
    vs, us = np.nonzero(skeleton)
    if len(us) == 0:
        return None
    u, v = uv
    d2 = (us - u) ** 2 + (vs - v) ** 2
    idx = int(np.argmin(d2))
    return int(us[idx]), int(vs[idx])


def dijkstra_on_skeleton(skeleton: np.ndarray, start_uv: tuple, goal_uv: tuple):
    """
    Dijkstra shortest path between two skeleton pixels, using only skeleton
    pixels as graph nodes/edges (paper's Section 3.2 graph G, built from
    8-connected skeleton-pixel adjacency -- "Each vertex is then connected
    to other vertices if they are neighbors around the map's resolution").

    Returns a list of (u, v) pixel coords from start_uv to goal_uv inclusive,
    or None if they lie on disconnected skeleton components -- itself a
    valid, reportable outcome (same convention as astar_grid returning None
    when no path exists).
    """
    if start_uv == goal_uv:
        return [start_uv]

    dist = {start_uv: 0.0}
    prev = {}
    visited = set()
    heap = [(0.0, start_uv)]

    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal_uv:
            break
        u, v = node
        for nu, nv, step in _skeleton_neighbors(skeleton, u, v):
            neighbor = (nu, nv)
            if neighbor in visited:
                continue
            nd = d + step
            if nd < dist.get(neighbor, float("inf")):
                dist[neighbor] = nd
                prev[neighbor] = node
                heapq.heappush(heap, (nd, neighbor))

    if goal_uv not in visited:
        return None

    path = [goal_uv]
    while path[-1] != start_uv:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def plan_skeleton_path(inflated_grid: np.ndarray, start_px: tuple, goal_px: tuple):
    """
    Full pipeline: skeletonize free space -> snap start/goal to nearest
    skeleton pixel -> Dijkstra shortest path along the skeleton graph.

    start_px / goal_px: (u, v) pixel coords, same convention as
    astar_planner.astar_grid's start_px/goal_px.

    Returns (path_px, meta):
      - path_px: list of (u, v) pixel coords (same convention as
        astar_grid's return value, so it drops straight into
        astar_planner.simplify_path() / path_to_world_waypoints()
        unchanged), or None if no path found.
      - meta: dict with diagnostic info (skeleton pixel count, snap
        distances, failure reason) useful for the comparison table / sanity
        checking.

    IMPORTANT: if start_px or goal_px themselves land on an occupied cell in
    `inflated_grid` (e.g. a corridor narrower than the clearance threshold,
    so the centerline itself is blocked), this returns None immediately --
    same semantics as astar_planner.astar_grid's own
    `if blocked(*start) or blocked(*goal): return None`. Without this check,
    nearest_skeleton_pixel() would silently snap to whatever skeleton
    fragment happens to be closest (possibly meters away, on the far side of
    a wall), which would produce a false "path found" for a scenario that is
    actually impassable at the requested clearance -- caught via unit test
    below (the "too narrow" case must return None, not a route that
    silently teleports past the blockage).
    """
    h, w = inflated_grid.shape

    def _blocked(uv):
        u, v = int(round(uv[0])), int(round(uv[1]))
        return not (0 <= u < w and 0 <= v < h) or bool(inflated_grid[v, u])

    if _blocked(start_px) or _blocked(goal_px):
        return None, {"reason": "start_or_goal_blocked_in_inflated_grid"}

    skeleton = skeletonize_free_space(inflated_grid)
    n_skel_px = int(skeleton.sum())
    if n_skel_px == 0:
        return None, {"reason": "empty_skeleton", "n_skeleton_px": 0}

    start_uv = nearest_skeleton_pixel(skeleton, start_px)
    goal_uv = nearest_skeleton_pixel(skeleton, goal_px)

    start_snap_dist = math.hypot(start_uv[0] - start_px[0], start_uv[1] - start_px[1])
    goal_snap_dist = math.hypot(goal_uv[0] - goal_px[0], goal_uv[1] - goal_px[1])

    path_px = dijkstra_on_skeleton(skeleton, start_uv, goal_uv)
    meta = {
        "n_skeleton_px": n_skel_px,
        "start_snap_px": start_uv,
        "goal_snap_px": goal_uv,
        "start_snap_dist_px": round(start_snap_dist, 2),
        "goal_snap_dist_px": round(goal_snap_dist, 2),
    }
    if path_px is None:
        meta["reason"] = "disconnected_skeleton_components"
        return None, meta

    return path_px, meta


if __name__ == "__main__":
    from occupancy_grid import GridSpec, build_occupancy_grid, inflate_grid
    from astar_planner import simplify_path, path_to_world_waypoints

    print("[Scene A-like] skeleton path should route around the obstacle, not through it")
    spec_a = GridSpec(x_min=-1.0, x_max=6.0, y_min=-3.0, y_max=3.0)
    obs_a = [{"name": "obs1", "pos": (2.0, 0.0), "kind": "cylinder", "size": (0.3, 0.5, 0.0)}]
    grid_a = inflate_grid(build_occupancy_grid(spec_a, obs_a), clearance_m=0.8, ppm=spec_a.ppm)
    start_px = spec_a.world_to_pixel(0.0, 0.0)
    goal_px = spec_a.world_to_pixel(5.0, 0.0)
    raw, meta = plan_skeleton_path(grid_a, start_px, goal_px)
    assert raw is not None, f"skeleton planner failed to find a path around a single small obstacle: {meta}"
    simplified = simplify_path(raw, epsilon_px=3.0)
    wps = path_to_world_waypoints(simplified, spec_a)
    print(f"  meta: {meta}")
    print(f"  raw path length: {len(raw)} px-steps -> simplified to {len(wps)} waypoints")
    print("  waypoints (world m):", [f"({x:.2f},{y:.2f})" for x, y in wps])
    max_abs_y = max(abs(y) for _, y in wps)
    print(f"  max |y| deviation: {max_abs_y:.2f} m (should be > 0: it must detour around the obstacle)")

    print("\n[Scene D-like, too narrow] skeleton planner should report NO path (corridor < 1.6m min)")
    spec_d = GridSpec(x_min=-1.0, x_max=8.0, y_min=-2.0, y_max=2.0)
    walls_narrow = [((-1.0, 0.75), (8.0, 0.75)), ((-1.0, -0.75), (8.0, -0.75))]
    grid_d_narrow = inflate_grid(build_occupancy_grid(spec_d, [], walls=walls_narrow), clearance_m=0.8, ppm=spec_d.ppm)
    p, meta = plan_skeleton_path(grid_d_narrow, spec_d.world_to_pixel(0.0, 0.0), spec_d.world_to_pixel(7.0, 0.0))
    print(f"  result: {'None (correctly impassable)' if p is None else f'{len(p)} px-steps (UNEXPECTED)'}  meta={meta}")

    print("\n[Scene D-like, OK width] skeleton planner should find exactly one straight-ish path")
    walls_ok = [((-1.0, 0.85), (8.0, 0.85)), ((-1.0, -0.85), (8.0, -0.85))]
    grid_d_ok = inflate_grid(build_occupancy_grid(spec_d, [], walls=walls_ok), clearance_m=0.8, ppm=spec_d.ppm)
    p_ok, meta_ok = plan_skeleton_path(grid_d_ok, spec_d.world_to_pixel(0.0, 0.0), spec_d.world_to_pixel(7.0, 0.0))
    assert p_ok is not None, f"meta={meta_ok}"
    wps_ok = path_to_world_waypoints(simplify_path(p_ok, epsilon_px=3.0), spec_d)
    print(f"  meta: {meta_ok}")
    print(f"  waypoints: {len(wps_ok)}  max |y| deviation: {max(abs(y) for _, y in wps_ok):.3f} m (should be ~0: straight down centerline)")
