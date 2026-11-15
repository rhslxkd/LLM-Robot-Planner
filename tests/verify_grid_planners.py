"""
verify_grid_planners.py ─ 표 ② 고전 베이스라인의 정확성 검증 (논문 부록 / 리뷰 대응용)

  python tests/verify_grid_planners.py            # 기본 500개 무작위 지도
  python tests/verify_grid_planners.py --n 2000

검증 항목
  1. A*·Dijkstra 최적성: 독립 구현(networkx Dijkstra, 같은 8-연결·모서리 끼기 금지 그래프)과 최단 비용이 일치
  2. A* 경로 유효성: 연속 셀이 인접, 전부 빈 셀, 대각선 이동 시 모서리 끼기 없음
  3. A* 효율: A* 확장 노드 수 ≤ Dijkstra 확장 노드 수 (휴리스틱이 일관적이면 성립)
  4. 선분 셀 순회(경로 단축의 충돌 검사): 아주 촘촘한 샘플링(셀의 1/200 간격)이 찾는 셀을 하나도 놓치지 않음
  5. 단축 후 경로: 모든 구간이 팽창 지도에서 충돌 없음
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import networkx as nx
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.plan_metrics import OccGrid  # noqa: E402
from baselines.grid_planners import _segment_free, grid_search, segment_cells, shortcut  # noqa: E402

SQ2 = np.sqrt(2)


def random_map(rng, H, W, n_rect):
    occ = np.zeros((H, W), bool)
    occ[0], occ[-1], occ[:, 0], occ[:, -1] = True, True, True, True
    for _ in range(n_rect):
        h, w = rng.integers(1, max(H // 3, 2)), rng.integers(1, max(W // 3, 2))
        r, c = rng.integers(0, H - h), rng.integers(0, W - w)
        occ[r:r + h, c:c + w] = True
    return occ


def nx_graph(occ):
    H, W = occ.shape
    G = nx.Graph()
    for r in range(H):
        for c in range(W):
            if occ[r, c]:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or occ[nr, nc]:
                    continue
                if dr and dc and (occ[r, nc] or occ[nr, c]):
                    continue
                G.add_edge((r, c), (nr, nc), weight=SQ2 if dr and dc else 1.0)
    return G


def path_cost(path):
    return sum(SQ2 if (a[0] != b[0] and a[1] != b[1]) else 1.0 for a, b in zip(path[:-1], path[1:]))


def path_valid(occ, path):
    for a, b in zip(path[:-1], path[1:]):
        dr, dc = b[0] - a[0], b[1] - a[1]
        if max(abs(dr), abs(dc)) != 1 or occ[b]:
            return False
        if dr and dc and (occ[a[0], b[1]] or occ[b[0], a[1]]):
            return False
    return not occ[path[0]]


def check_search(n, seed):
    rng = np.random.default_rng(seed)
    stats = {"cases": 0, "cost_mismatch_astar": 0, "cost_mismatch_dijkstra": 0, "invalid_path": 0,
             "reachability_mismatch": 0, "astar_more_expansions": 0, "exp_ratio": []}
    while stats["cases"] < n:
        H, W = rng.integers(15, 60), rng.integers(15, 60)
        occ = random_map(rng, H, W, rng.integers(2, 12))
        free = np.argwhere(~occ)
        if len(free) < 2:
            continue
        s, g = (tuple(int(v) for v in free[i]) for i in rng.choice(len(free), 2, replace=False))
        G = nx_graph(occ)
        try:
            ref = nx.dijkstra_path_length(G, s, g)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            ref = None
        pa, ea = grid_search(occ, s, g, heuristic=True)
        pd, ed = grid_search(occ, s, g, heuristic=False)
        stats["cases"] += 1
        if (ref is None) != (pa is None) or (ref is None) != (pd is None):
            stats["reachability_mismatch"] += 1
            continue
        if ref is None:
            continue
        if abs(path_cost(pa) - ref) > 1e-9:
            stats["cost_mismatch_astar"] += 1
        if abs(path_cost(pd) - ref) > 1e-9:
            stats["cost_mismatch_dijkstra"] += 1
        if not (path_valid(occ, pa) and path_valid(occ, pd)):
            stats["invalid_path"] += 1
        if ea > ed:
            stats["astar_more_expansions"] += 1
        stats["exp_ratio"].append(ea / max(ed, 1))
    stats["exp_ratio"] = float(np.mean(stats["exp_ratio"]))
    return stats


def check_segment_cells(n, seed):
    rng = np.random.default_rng(seed)
    grid = OccGrid(np.zeros((40, 40), bool), 0.05, (0.0, 0.0))
    missed = 0
    for _ in range(n):
        a, b = rng.uniform(-0.02, 1.95, 2), rng.uniform(-0.02, 1.95, 2)
        if rng.random() < 0.2:                                   # 축에 평행하거나 꼭짓점을 지나는 특수 경우
            b = a + np.array([rng.integers(-10, 10), rng.integers(-10, 10)]) * 0.05
        exact = set(segment_cells(grid, a, b))
        t = np.linspace(0, 1, int(np.linalg.norm(b - a) / (0.05 / 200)) + 2)
        pts = a + t[:, None] * (b - a)
        r, c = grid.to_cell(pts)
        dense = set(zip(r.tolist(), c.tolist()))
        missed += len(dense - exact)
    return {"segments": n, "cells_missed_vs_dense_sampling": missed}


def check_shortcut(n, seed):
    rng = np.random.default_rng(seed)
    bad, cases = 0, 0
    while cases < n:
        occ = random_map(rng, 50, 50, 8)
        free = np.argwhere(~occ)
        s, g = (tuple(int(v) for v in free[i]) for i in rng.choice(len(free), 2, replace=False))
        p, _ = grid_search(occ, s, g)
        if p is None:
            continue
        grid = OccGrid(occ, 0.05, (0.0, 0.0))
        poly = shortcut(grid, grid.to_world(*np.array(p).T))
        cases += 1
        if not all(_segment_free(grid, a, b) for a, b in zip(poly[:-1], poly[1:])):
            bad += 1
    return {"paths": n, "shortcut_segments_in_collision": bad}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    s1 = check_search(a.n, a.seed)
    s2 = check_segment_cells(a.n * 4, a.seed + 1)
    s3 = check_shortcut(a.n // 2, a.seed + 2)
    print("1-3) A*/Dijkstra vs networkx:", s1)
    print("4)   segment traversal:", s2)
    print("5)   shortcut validity:", s3)
    ok = (s1["cost_mismatch_astar"] == s1["cost_mismatch_dijkstra"] == s1["invalid_path"] == s1["reachability_mismatch"]
          == s1["astar_more_expansions"] == 0 and s2["cells_missed_vs_dense_sampling"] == 0
          and s3["shortcut_segments_in_collision"] == 0)
    print(f"{'✅ PASS' if ok else '❌ FAIL'} ({time.time() - t0:.0f}s)")
    sys.exit(0 if ok else 1)
