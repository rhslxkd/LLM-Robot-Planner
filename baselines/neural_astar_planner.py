"""
neural_astar_planner.py ─ Neural A* (Yonetani et al., ICML 2021; omron-sinicx/neural-astar)를 실내 정답 지도에 연결

흐름 (A*·Dijkstra와 같은 전처리·후처리, 탐색만 Neural A*):
  1. 정답 점유 지도(0.05 m)를 로봇 반경 0.441 m만큼 팽창          ← grid_planners.inflate
  2. 출발점과 목표점을 둘러싼 창(window)을 잘라 cell 크기로 내림      (any-pooling: 칸 안에 하나라도 막히면 막힘)
  3. Neural A*: CNN 인코더가 칸마다 탐색 비용을 예측하고 A*가 그 비용으로 탐색
       mode="pq"   : 우선순위 큐 A* (큰 지도용, 공식 구현의 use_differentiable_astar=False)
       mode="diff" : 학습 때 쓰는 미분 가능 A* (32×32 같은 작은 지도용)
     장애물 칸은 탐색에서 제외되므로, 결과 경로는 팽창한 지도 위에서 충돌이 없다.
  4. 경로 칸을 출발점부터 순서대로 잇고 월드 좌표로 변환
  5. line-of-sight 단축(팽창 지도 0.05 m 기준) → 꺾인 점 유지한 채 n개로    ← grid_planners.shortcut / to_n_waypoints

체크포인트: 공식 사전학습(mazes_032_moore_c8) 또는 feature/neural-astar-finetune의 finetune/checkpoints/<tag>
            (PyTorch Lightning .ckpt 폴더. pytorch_lightning 없이 state_dict만 읽는다.)
neural_astar 패키지: 설치돼 있으면 그대로, 아니면 NEURAL_ASTAR_SRC 또는 <repo>/neural-astar/src 를 sys.path에 추가.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
from collections import deque

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.plan_metrics import OccGrid  # noqa: E402
from baselines.grid_planners import DEFAULT_INFLATE_M, _nearest_free, inflate, shortcut, to_n_waypoints  # noqa: E402

DEFAULT_CKPT = os.path.join(REPO, "neural-astar", "model", "mazes_032_moore_c8")
_MODEL_CACHE: dict = {}


def _import_neural_astar():
    try:
        from neural_astar.planner import NeuralAstar  # noqa: F401
    except ImportError:
        src = os.environ.get("NEURAL_ASTAR_SRC", os.path.join(REPO, "neural-astar", "src"))
        sys.path.insert(0, src)
    from neural_astar.planner import NeuralAstar
    return NeuralAstar


def load_model(ckpt_dir: str, mode: str = "pq", g_ratio: float = 0.5):
    """PyTorch Lightning 체크포인트 폴더에서 planner 가중치만 읽는다 (공식 load_from_ptl_checkpoint와 같은 규칙)."""
    key = (os.path.abspath(ckpt_dir), mode, g_ratio)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch
    NeuralAstar = _import_neural_astar()
    files = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"{ckpt_dir} 아래에 .ckpt 없음")
    ckpt = files[-1]
    try:                                     # torch >= 1.13: weights_only 인자 필요(2.6부터 기본 True)
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:                        # 옛 torch: 인자 없음
        raw = torch.load(ckpt, map_location="cpu")
    sd = raw["state_dict"]
    sd = {re.split("planner.", k)[-1]: v for k, v in sd.items() if "planner" in k}
    model = NeuralAstar(g_ratio=g_ratio, encoder_arch="CNN", use_differentiable_astar=(mode == "diff"))
    model.load_state_dict(sd)
    model.eval()
    _MODEL_CACHE[key] = (model, ckpt)
    return model, ckpt


def _window(grid: OccGrid, goal_xy, pad_m: float, cell_m: float):
    """출발점(0,0)과 목표를 포함하고 pad_m 여유를 둔 창. 창 크기는 cell 배수로 맞춘다."""
    pts = np.array([[0.0, 0.0], goal_xy], float)
    lo, hi = pts.min(0) - pad_m, pts.max(0) + pad_m
    k = max(1, int(round(cell_m / grid.res)))
    r0, c0 = grid.to_cell(lo)
    r1, c1 = grid.to_cell(hi)
    H, W = grid.occ.shape
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    r1, c1 = min(H, int(r1) + 1), min(W, int(c1) + 1)
    r1 = r0 + ((r1 - r0) // k) * k
    c1 = c0 + ((c1 - c0) // k) * k
    return r0, r1, c0, c1, k


def _ordered_cells(mask: np.ndarray, s, g):
    """경로 마스크(8-연결) 위에서 출발 칸 → 목표 칸 BFS. 마스크가 끊겨 있으면 None."""
    H, W = mask.shape
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == g:
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (cur[0] + dr, cur[1] + dc)
                if (dr or dc) and 0 <= n[0] < H and 0 <= n[1] < W and mask[n] and n not in prev:
                    prev[n] = cur
                    q.append(n)
    if g not in prev:
        return None
    out, cur = [], g
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    return out[::-1]


def plan_neural_astar(grid: OccGrid, goal_xy, n_waypoints: int, ckpt_dir: str = DEFAULT_CKPT, mode: str = "pq",
                      cell_m: float = 0.10, pad_m: float = 1.5, inflate_m: float = DEFAULT_INFLATE_M,
                      g_ratio: float = 0.5) -> dict:
    import torch
    t0 = time.perf_counter()
    goal_xy = np.asarray(goal_xy, float)
    inf = inflate(grid, inflate_m)
    r0, r1, c0, c1, k = _window(inf, goal_xy, pad_m, cell_m)
    sub = inf.occ[r0:r1, c0:c1]
    h, w = sub.shape[0] // k, sub.shape[1] // k
    coarse = sub.reshape(h, k, w, k).any(axis=(1, 3))                  # 칸 안에 하나라도 막히면 막힘 (보수적)
    to_coarse = lambda xy: ((np.asarray(inf.to_cell(np.asarray(xy, float))) - [r0, c0]) // k).astype(int)
    s = _nearest_free(coarse, tuple(to_coarse([0.0, 0.0])), max_r=3)
    g = _nearest_free(coarse, tuple(to_coarse(goal_xy)), max_r=3)
    base = {"inflate_m": inflate_m, "cell_m": k * inf.res, "window_cells": [int(h), int(w)], "mode": mode}
    if s is None or g is None:
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "reason": "start/goal blocked after inflation", **base}
    model, ckpt = load_model(ckpt_dir, mode, g_ratio)
    free = torch.tensor(~coarse, dtype=torch.float32)[None, None]
    sm = torch.zeros_like(free); sm[0, 0, s[0], s[1]] = 1.0
    gm = torch.zeros_like(free); gm[0, 0, g[0], g[1]] = 1.0
    with torch.no_grad():
        out = model(free, sm, gm)
    mask = out.paths[0, 0].numpy() > 0.5
    expanded = int((out.histories[0, 0].numpy() > 0.5).sum())
    base.update(ckpt=os.path.relpath(ckpt, REPO) if ckpt.startswith(REPO) else ckpt, expanded=expanded)
    cells = _ordered_cells(mask, s, g)
    if cells is None:
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "reason": "no path", **base}
    rc = np.array(cells, float) * k + (k - 1) / 2.0 + [r0, c0]          # 큰 칸 중심 → 0.05 m 격자 좌표
    poly = inf.to_world(rc[:, 0], rc[:, 1])
    poly[0], poly[-1] = [0.0, 0.0], goal_xy
    raw_len = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
    poly = to_n_waypoints(shortcut(inf, poly), n_waypoints)
    return {"path": poly, "plan_time_s": time.perf_counter() - t0, "raw_grid_len_m": raw_len,
            "n_path_cells": len(cells), **base}
