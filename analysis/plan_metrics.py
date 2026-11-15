"""
plan_metrics.py ─ '계획 지표' (시뮬레이션 전, waypoint 경로만 보고 계산).

모든 방법을 같은 잣대로 채점하기 위해 BEV 이미지가 아니라 **GT 점유 지도**(OccGrid)로 계산한다.

좌표 규약 (courtroom / DIAL-MPC와 동일)
  - 로봇 시작점 = (0, 0), 전방 = +x, 왼쪽 = +y, 단위 m
  - OccGrid.occ[r, c] = True 이면 장애물. 셀 (r, c) 중심 = (x0 + c*res, y0 + r*res)

지표
  output_ok        waypoint가 2개 이상, 숫자형, 유한값                 (파이프라인 강건성)
  reaches_goal     마지막 waypoint가 goal 반경 안
  bev_collision    직선 보간 경로를 따라 Go2 footprint(0.70×0.35 m 직사각형, 진행방향 정렬)를
                   쓸었을 때 점유 셀과 겹침. 회전반경 < 0.5 m인 꺾임점에서는 제자리 회전으로 보고
                   외접원(반경 0.391 m)으로 검사                              (교수님 요구 지표)
  min_clearance_m  경로 중심선에서 가장 가까운 장애물까지 거리
  valid            output_ok ∧ reaches_goal ∧ ¬bev_collision
  homotopy         장애물별 ray 교차 word (Bhattacharya et al. 2012의 H-signature, 축약형)
  diversity        같은 (씬, 방법)에서 나온 유효 경로들의 homotopy class 수, 평균 쌍별 거리
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

GO2_FOOTPRINT_L = 0.70     # 다리 포함 전장 (Unitree 사양)
GO2_FOOTPRINT_W = 0.35     # 고관절 하우징 포함 폭 (go2.xml 충돌 geom 실측)
PIVOT_RADIUS = float(np.hypot(GO2_FOOTPRINT_L / 2, GO2_FOOTPRINT_W / 2))   # 0.391 m
MIN_TURN_RADIUS = 0.5      # courtroom 프롬프트의 Locomotion constraint와 동일
GOAL_RADIUS_M = 0.3
SAMPLE_STEP_M = 0.05


@dataclass
class OccGrid:
    occ: np.ndarray          # (H, W) bool
    res: float               # m / cell
    origin: tuple            # 셀 (0,0) 중심의 world (x, y)

    @classmethod
    def load(cls, path: str) -> "OccGrid":
        z = np.load(path)
        return cls(z["occ"].astype(bool), float(z["res"]), tuple(z["origin"].tolist()))

    def save(self, path: str) -> None:
        np.savez_compressed(path, occ=self.occ, res=self.res, origin=np.array(self.origin))

    def to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.round((xy[..., 0] - self.origin[0]) / self.res).astype(int)
        r = np.round((xy[..., 1] - self.origin[1]) / self.res).astype(int)
        return r, c

    def to_world(self, r, c) -> np.ndarray:
        return np.stack([self.origin[0] + np.asarray(c) * self.res, self.origin[1] + np.asarray(r) * self.res], -1)

    def occupied(self, xy: np.ndarray) -> np.ndarray:
        """지도 밖은 점유(=알 수 없음)로 본다 → 보수적."""
        r, c = self.to_cell(xy)
        H, W = self.occ.shape
        inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        out = np.ones(r.shape, bool)
        out[inside] = self.occ[r[inside], c[inside]]
        return out

    def clearance_field(self) -> np.ndarray:
        """각 셀에서 가장 가까운 점유 셀까지 거리 (m)."""
        return ndimage.distance_transform_edt(~self.occ) * self.res


# ─── 경로 유틸 ─────────────────────────────────────────────────────────
def parse_path(path) -> np.ndarray | None:
    """last_judged_path.json 형식([{x, y}, ...]) 또는 (N,2) → (N,2) float. 형식 불량이면 None."""
    try:
        if isinstance(path, (list, tuple)) and path and isinstance(path[0], dict):
            arr = np.array([[float(p["x"]), float(p["y"])] for p in path])
        else:
            arr = np.asarray(path, float).reshape(-1, 2)
    except (KeyError, TypeError, ValueError):
        return None
    if len(arr) < 2 or not np.isfinite(arr).all():
        return None
    return arr


def densify(poly: np.ndarray, step: float = SAMPLE_STEP_M) -> tuple[np.ndarray, np.ndarray]:
    """polyline → (점들, 각 점의 진행방향 heading). env와 같은 직선 보간."""
    pts, heads = [], []
    for a, b in zip(poly[:-1], poly[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(int(np.ceil(L / step)), 1)
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
        heads.append(np.full(n, np.arctan2(b[1] - a[1], b[0] - a[0]) if L > 1e-9 else 0.0))
    pts.append(poly[-1:]); heads.append(heads[-1][-1:] if heads else np.zeros(1))
    return np.concatenate(pts), np.concatenate(heads)


def sharp_turn_points(poly: np.ndarray, r_min: float = MIN_TURN_RADIUS) -> np.ndarray:
    """회전반경 < r_min인 꺾임 waypoint. 두 선분 중점에 접하는 원의 반경 = (min(l1,l2)/2)/tan(θ/2)."""
    idx = []
    for i in range(1, len(poly) - 1):
        v1, v2 = poly[i] - poly[i - 1], poly[i + 1] - poly[i]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        th = np.arccos(np.clip(v1 @ v2 / (l1 * l2), -1, 1))
        if th < 1e-6:
            continue
        r = (min(l1, l2) / 2) / np.tan(th / 2)
        if r < r_min:
            idx.append(i)
    return np.array(idx, int)


def _footprint_offsets(res: float, L=GO2_FOOTPRINT_L, W=GO2_FOOTPRINT_W) -> np.ndarray:
    step = res / 2
    xs = np.arange(-L / 2, L / 2 + 1e-9, step)
    ys = np.arange(-W / 2, W / 2 + 1e-9, step)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def _disk_offsets(res: float, R=PIVOT_RADIUS) -> np.ndarray:
    o = _footprint_offsets(res, 2 * R, 2 * R)
    return o[np.linalg.norm(o, axis=1) <= R]


def evaluate_path(path, grid: OccGrid, goal_xy, goal_radius: float = GOAL_RADIUS_M) -> dict:
    poly = parse_path(path)
    if poly is None:
        return {"output_ok": False, "reaches_goal": False, "bev_collision": True, "valid": False,
                "min_clearance_m": float("nan"), "n_waypoints": 0, "path_len_m": float("nan"),
                "sharp_turns": 0, "collision_arclen_m": float("nan")}
    goal_xy = np.asarray(goal_xy, float)
    pts, heads = densify(poly)
    fp = _footprint_offsets(grid.res)
    c, s = np.cos(heads), np.sin(heads)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)        # (S, 2, 2)
    swept = pts[:, None, :] + np.einsum("sij,kj->ski", rot, fp)              # (S, K, 2)
    hit_rect = grid.occupied(swept).any(1)
    sharp = sharp_turn_points(poly)
    hit_pivot = np.zeros(len(sharp), bool)
    if len(sharp):
        disk = _disk_offsets(grid.res)
        hit_pivot = grid.occupied(poly[sharp][:, None, :] + disk[None]).any(1)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate([[0], np.cumsum(seg)])
    clear = grid.clearance_field()
    r, cc = grid.to_cell(pts)
    H, W = grid.occ.shape
    ok = (r >= 0) & (r < H) & (cc >= 0) & (cc < W)
    min_clear = float(clear[r[ok], cc[ok]].min()) if ok.any() else 0.0

    collision = bool(hit_rect.any() or hit_pivot.any())
    reaches = bool(np.linalg.norm(poly[-1] - goal_xy) <= goal_radius)
    return {
        "output_ok": True,
        "reaches_goal": reaches,
        "bev_collision": collision,
        "valid": bool(reaches and not collision),
        "min_clearance_m": min_clear,
        "n_waypoints": int(len(poly)),
        "path_len_m": float(arclen[-1]),
        "sharp_turns": int(len(sharp)),
        "collision_arclen_m": float(arclen[np.argmax(hit_rect)]) if hit_rect.any() else float("nan"),
    }


# ─── homotopy / 다양성 ──────────────────────────────────────────────────
@dataclass
class HomotopyRays:
    points: np.ndarray       # (K, 2) 장애물 대표점 (world)

    @classmethod
    def from_grid(cls, grid: OccGrid, min_area_m2: float = 0.04) -> "HomotopyRays":
        """지도 테두리에 닿지 않는 장애물 섬(연결 성분)마다 대표점 1개 = 성분 내부에서 경계로부터 가장 먼 셀."""
        lab, n = ndimage.label(grid.occ, structure=np.ones((3, 3)))
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        inner = ndimage.distance_transform_edt(grid.occ)
        pts = []
        min_cells = min_area_m2 / grid.res ** 2
        for k in range(1, n + 1):
            if k in border:
                continue
            mask = lab == k
            if mask.sum() < min_cells:
                continue
            rr, cc = np.unravel_index(np.argmax(np.where(mask, inner, -1)), mask.shape)
            pts.append(grid.to_world(rr, cc))
        pts = np.array(pts).reshape(-1, 2)
        # 같은 x에 대표점 두 개가 있으면 +y ray가 겹치므로 아주 작게 흔들어 준다
        pts[:, 0] += np.arange(len(pts)) * 1e-6
        return cls(pts)

    def signature(self, path, start=(0.0, 0.0), goal=None) -> tuple:
        """경로를 start/goal로 닫은 뒤 +y 방향 ray 교차를 (장애물 id, ±1) word로 기록 → 인접 상쇄 축약."""
        poly = parse_path(path)
        if poly is None:
            return ("invalid",)
        seq = [np.asarray(start, float)] + list(poly) + ([np.asarray(goal, float)] if goal is not None else [])
        poly = np.array(seq)
        word: list[tuple[int, int]] = []
        for a, b in zip(poly[:-1], poly[1:]):
            hits = []
            for k, (px, py) in enumerate(self.points):
                if (a[0] - px) * (b[0] - px) < 0:                     # 선분이 x = px를 가로지름
                    t = (px - a[0]) / (b[0] - a[0])
                    y = a[1] + t * (b[1] - a[1])
                    if y > py:
                        hits.append((t, k, 1 if b[0] > a[0] else -1))
            for _, k, sgn in sorted(hits):
                if word and word[-1] == (k, -sgn):
                    word.pop()
                else:
                    word.append((k, sgn))
        return tuple(word)


def resample(poly: np.ndarray, n: int = 50) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(poly[:1], n, 0)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], -1)


def avg_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return float((d.min(1).mean() + d.min(0).mean()) / 2)


def diversity(paths: list, grid: OccGrid, goal_xy, rays: HomotopyRays | None = None) -> dict:
    """같은 (씬, 방법)의 여러 시행 경로 → 유효율, homotopy class 수, 평균 쌍별 거리."""
    rays = rays or HomotopyRays.from_grid(grid)
    evals = [evaluate_path(p, grid, goal_xy) for p in paths]
    valid = [parse_path(p) for p, e in zip(paths, evals) if e["valid"]]
    sigs = {rays.signature(p, goal=goal_xy) for p in valid}
    rs = [resample(p) for p in valid]
    pair = [avg_hausdorff(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs))]
    return {
        "n_paths": len(paths),
        "n_valid": len(valid),
        "valid_rate": len(valid) / max(len(paths), 1),
        "n_homotopy": len(sigs),
        "pairwise_dist_m": float(np.mean(pair)) if pair else 0.0,
        "n_obstacle_islands": int(len(rays.points)),
    }
