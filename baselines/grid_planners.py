"""
grid_planners.py ─ 표 ② 고전 경로계획 베이스라인 (점유 지도 → waypoint 경로)

  astar     8-연결 격자 A* (octile 휴리스틱)
  dijkstra  같은 격자, 휴리스틱 0 (균일 격자에서는 A*와 경로 비용이 같다 → 계산량만 다름)
  rrtstar   연속 공간 RRT* (seed마다 다른 경로)

공통 전처리: 장애물을 로봇 반경만큼 팽창(inflate). 기본 반경 = Go2 footprint 외접원 0.391 m + 여유 0.05 m.
→ 고전 알고리즘이 반드시 받아야 하는 '수작업 파라미터'이므로 meta에 기록한다.
공통 후처리: line-of-sight 단축(string pulling) → 꺾인점은 유지하고 긴 구간을 나눠 N개 waypoint로 맞춤
(VLM 방법과 같은 개수를 DIAL-MPC에 넘기기 위해).
"""
from __future__ import annotations

import heapq
import time

import numpy as np
from scipy import ndimage

from analysis.plan_metrics import PIVOT_RADIUS, OccGrid

DEFAULT_INFLATE_M = PIVOT_RADIUS + 0.05


def inflate(grid: OccGrid, radius_m: float) -> OccGrid:
    k = int(np.ceil(radius_m / grid.res))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = (xx ** 2 + yy ** 2) * grid.res ** 2 <= radius_m ** 2
    return OccGrid(ndimage.binary_dilation(grid.occ, structure=disk), grid.res, grid.origin)


def _nearest_free(occ: np.ndarray, rc: tuple[int, int], max_r: int = 10):
    """시작/목표 셀이 팽창 영역에 걸리면 가장 가까운 빈 셀로 옮긴다 (없으면 None)."""
    r0, c0 = rc
    H, W = occ.shape
    if 0 <= r0 < H and 0 <= c0 < W and not occ[r0, c0]:
        return rc
    best = None
    for dr in range(-max_r, max_r + 1):
        for dc in range(-max_r, max_r + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < H and 0 <= c < W and not occ[r, c]:
                d = dr * dr + dc * dc
                if best is None or d < best[0]:
                    best = (d, (r, c))
    return best[1] if best else None


_NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def grid_search(occ: np.ndarray, start: tuple, goal: tuple, heuristic: bool = True):
    """8-연결 A*/Dijkstra. 반환: (셀 경로 또는 None, 확장한 노드 수)."""
    H, W = occ.shape
    g = np.full((H, W), np.inf)
    parent = -np.ones((H, W, 2), int)
    closed = np.zeros((H, W), bool)
    gr, gc = goal

    def h(r, c):
        if not heuristic:
            return 0.0
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (np.sqrt(2) - 2) * min(dr, dc)

    g[start] = 0.0
    pq = [(h(*start), 0.0, start)]
    expanded = 0
    while pq:
        _, gcur, (r, c) = heapq.heappop(pq)
        if closed[r, c]:
            continue
        closed[r, c] = True
        expanded += 1
        if (r, c) == goal:
            path = [(r, c)]
            while parent[r, c, 0] >= 0:
                r, c = parent[r, c]
                path.append((int(r), int(c)))
            return path[::-1], expanded
        for dr, dc in _NBR:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or occ[nr, nc] or closed[nr, nc]:
                continue
            if dr and dc and (occ[r, nc] or occ[nr, c]):        # 대각선 모서리 끼기 금지
                continue
            ng = gcur + (np.sqrt(2) if dr and dc else 1.0)
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = (r, c)
                heapq.heappush(pq, (ng + h(nr, nc), ng, (nr, nc)))
    return None, expanded


def _segment_free(grid: OccGrid, a: np.ndarray, b: np.ndarray) -> bool:
    n = max(int(np.ceil(np.linalg.norm(b - a) / (grid.res / 2))), 1)
    pts = a + np.linspace(0, 1, n + 1)[:, None] * (b - a)
    return not grid.occupied(pts).any()


def shortcut(grid: OccGrid, poly: np.ndarray) -> np.ndarray:
    """string pulling: 앞에서부터 직선으로 갈 수 있는 가장 먼 점으로 건너뛴다."""
    out = [poly[0]]
    i = 0
    while i < len(poly) - 1:
        j = len(poly) - 1
        while j > i + 1 and not _segment_free(grid, poly[i], poly[j]):
            j -= 1
        out.append(poly[j])
        i = j
    return np.array(out)


def to_n_waypoints(poly: np.ndarray, n: int) -> np.ndarray:
    """꺾인점은 유지하고, 가장 긴 구간을 반으로 나누는 것을 반복해 n개로 맞춘다 (n보다 많으면 그대로)."""
    pts = [p for p in poly]
    while len(pts) < n:
        seg = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        i = int(np.argmax(seg))
        pts.insert(i + 1, (pts[i] + pts[i + 1]) / 2)
    return np.array(pts)


def plan_grid(grid: OccGrid, goal_xy, n_waypoints: int, method: str = "astar",
              inflate_m: float = DEFAULT_INFLATE_M) -> dict:
    t0 = time.perf_counter()
    inf = inflate(grid, inflate_m)
    s = _nearest_free(inf.occ, tuple(int(v) for v in inf.to_cell(np.array([0.0, 0.0]))))
    g = _nearest_free(inf.occ, tuple(int(v) for v in inf.to_cell(np.asarray(goal_xy, float))))
    if s is None or g is None:
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "reason": "start/goal blocked after inflation"}
    cells, expanded = grid_search(inf.occ, s, g, heuristic=(method == "astar"))
    if cells is None:
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "expanded": expanded, "reason": "no path"}
    poly = inf.to_world(*np.array(cells).T)
    poly[0], poly[-1] = [0.0, 0.0], np.asarray(goal_xy, float)
    poly = to_n_waypoints(shortcut(inf, poly), n_waypoints)
    return {"path": poly, "plan_time_s": time.perf_counter() - t0, "expanded": expanded,
            "inflate_m": inflate_m}


def plan_rrtstar(grid: OccGrid, goal_xy, n_waypoints: int, seed: int, inflate_m: float = DEFAULT_INFLATE_M,
                 max_iter: int = 4000, step: float = 0.5, goal_bias: float = 0.1, rewire_r: float = 1.0,
                 goal_tol: float = 0.3, do_shortcut: bool = True) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    inf = inflate(grid, inflate_m)
    H, W = inf.occ.shape
    lo = np.array(inf.origin); hi = lo + np.array([W - 1, H - 1]) * inf.res
    goal = np.asarray(goal_xy, float)
    nodes = [np.zeros(2)]; parent = [-1]; cost = [0.0]
    best = None
    for _ in range(max_iter):
        x = goal if rng.random() < goal_bias else rng.uniform(lo, hi)
        P = np.array(nodes)
        i_near = int(np.argmin(np.linalg.norm(P - x, axis=1)))
        d = x - P[i_near]
        dist = np.linalg.norm(d)
        new = x if dist <= step else P[i_near] + d / dist * step
        if inf.occupied(new[None])[0] or not _segment_free(inf, P[i_near], new):
            continue
        near = np.where(np.linalg.norm(P - new, axis=1) <= rewire_r)[0]
        bp, bc = i_near, cost[i_near] + np.linalg.norm(new - P[i_near])
        for j in near:
            cj = cost[j] + np.linalg.norm(new - P[j])
            if cj < bc and _segment_free(inf, P[j], new):
                bp, bc = int(j), cj
        nodes.append(new); parent.append(bp); cost.append(bc)
        k = len(nodes) - 1
        for j in near:                                          # rewire
            cj = bc + np.linalg.norm(P[j] - new)
            if cj < cost[j] and _segment_free(inf, new, P[j]):
                parent[j], cost[j] = k, cj
        if np.linalg.norm(new - goal) <= goal_tol and (best is None or bc < cost[best]):
            best = k
    if best is None:
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "reason": "no path", "n_nodes": len(nodes)}
    seq, k = [], best
    while k >= 0:
        seq.append(nodes[k]); k = parent[k]
    poly = np.array(seq[::-1])
    poly[-1] = goal
    if do_shortcut:
        poly = shortcut(inf, poly)
    poly = to_n_waypoints(poly, n_waypoints)
    return {"path": poly, "plan_time_s": time.perf_counter() - t0, "n_nodes": len(nodes),
            "inflate_m": inflate_m, "seed": seed}
