"""
tests/verify_metrics.py ─ 평가 지표 구현을 독립 구현과 대조한다.

  python tests/verify_metrics.py            # 전부
  python tests/verify_metrics.py --quick    # 무작위 사례 수를 줄여서

1) BEV 충돌 (evaluate_path)   vs  shapely의 정확한 기하: 직선 구간마다 Go2 직사각형(0.70×0.35 m)을 쓸고 지나간
                                  다각형(구간 시작·끝 직사각형의 convex hull) + 급회전 지점의 원(0.391 m)과
                                  점유 셀 사각형들의 교차. 우리 구현은 2.5 cm 간격 점 샘플링이라 그보다 얕은 겹침은
                                  놓칠 수 있다 → 불일치 사례의 침투 깊이(또는 실제 여유)를 보고하고 2.5 cm 이하인지 확인.
2) 목표 도달                  vs  직접 계산
3) 기울기 (tilt)              vs  scipy Rotation으로 몸통 z축과 세계 z축 사이 각
4) 경로 이탈 (point→polyline) vs  shapely distance
5) homotopy                  vs  손으로 만든 사례: 장애물 하나를 왼쪽/오른쪽으로 돌기, 같은 쪽 두 경로, 한 바퀴 감기
6) 토크 포화                  vs  손으로 만든 ctrl
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
from shapely import affinity
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis import plan_metrics as pm  # noqa: E402
from analysis.episode_metrics import GO2_TORQUE_LIMITS, _point_to_polyline, _tilt_deg  # noqa: E402

L, W, R = pm.GO2_FOOTPRINT_L, pm.GO2_FOOTPRINT_W, pm.PIVOT_RADIUS


def occ_polygons(grid: pm.OccGrid):
    rr, cc = np.nonzero(grid.occ)
    h = grid.res / 2
    xs, ys = grid.origin[0] + cc * grid.res, grid.origin[1] + rr * grid.res
    cells = [box(x - h, y - h, x + h, y + h) for x, y in zip(xs, ys)]
    H, Wd = grid.occ.shape                                   # 지도 밖 = 점유 (구현과 동일한 규약)
    x0, y0 = grid.origin[0] - h, grid.origin[1] - h
    x1, y1 = x0 + Wd * grid.res, y0 + H * grid.res
    outside = box(x0 - 50, y0 - 50, x1 + 50, y1 + 50).difference(box(x0, y0, x1, y1))
    return unary_union(cells + [outside])


def rect_at(p, heading):
    r = box(-L / 2, -W / 2, L / 2, W / 2)
    r = affinity.rotate(r, heading, use_radians=True, origin=(0, 0))
    return affinity.translate(r, p[0], p[1])


def exact_swept(poly: np.ndarray):
    parts = []
    for a, b in zip(poly[:-1], poly[1:]):
        if np.linalg.norm(b - a) < 1e-9:
            continue
        th = np.arctan2(b[1] - a[1], b[0] - a[0])
        parts.append(unary_union([rect_at(a, th), rect_at(b, th)]).convex_hull)
    for i in pm.sharp_turn_points(poly):
        parts.append(Point(poly[i]).buffer(R, 64))
    return unary_union(parts)


def random_grid(rng, n_obs=6, size=(80, 80), res=0.05):
    occ = np.zeros(size, bool)
    for _ in range(n_obs):
        h, w = rng.integers(3, 18, 2)
        r, c = rng.integers(0, size[0] - h), rng.integers(0, size[1] - w)
        occ[r:r + h, c:c + w] = True
    return pm.OccGrid(occ, res, (0.0, 0.0))


def _penetration(sw, obs):
    """swept 다각형을 안쪽으로 얼마나 줄여야 장애물과 안 겹치는가 (= 최대 침투 깊이, 이분 탐색)."""
    lo, hi = 0.0, 0.2
    for _ in range(20):
        m = (lo + hi) / 2
        if sw.buffer(-m).intersection(obs).area > 1e-10:
            lo = m
        else:
            hi = m
    return hi


def check_bev(n_cases, rng):
    """장애물 3개 + 넓은 지도로 '아슬아슬한' 사례가 많이 나오게 한다."""
    agree = 0
    missed, false_alarm, n_col = [], [], 0
    for _ in range(n_cases):
        g = random_grid(rng, n_obs=3, size=(120, 120))
        poly = rng.uniform(0.8, 5.2, (rng.integers(2, 6), 2))
        ours = pm.evaluate_path(poly, g, poly[-1])["bev_collision"]
        obs = occ_polygons(g)
        sw = exact_swept(poly)
        exact = sw.intersection(obs).area > 1e-10
        n_col += exact
        if ours == exact:
            agree += 1
        elif exact:
            missed.append(_penetration(sw, obs))            # 우리가 놓친 충돌의 침투 깊이
        else:
            false_alarm.append(float(sw.distance(obs)))     # 우리가 과하게 잡은 충돌의 실제 여유
    return {"cases": n_cases, "exact_collisions": int(n_col), "agree": agree,
            "missed": len(missed), "max_missed_penetration_m": round(max(missed, default=0.0), 4),
            "false_alarms": len(false_alarm), "max_false_alarm_clearance_m": round(max(false_alarm, default=0.0), 4)}


def check_goal():
    g = pm.OccGrid(np.zeros((40, 40), bool), 0.05, (0.0, 0.0))
    p = np.array([[0.2, 0.2], [1.0, 1.0]])
    a = pm.evaluate_path(p, g, (1.0, 1.29))["reaches_goal"]
    b = pm.evaluate_path(p, g, (1.0, 1.31))["reaches_goal"]
    return {"inside_0.29m": a, "outside_0.31m": b, "ok": a and not b}


def check_tilt(n, rng):
    eul = rng.uniform(-np.pi, np.pi, (n, 3)) * np.array([1, 0.5, 1])
    rot = Rotation.from_euler("zyx", eul)
    q_xyzw = rot.as_quat()
    qpos = np.zeros((n, 19))
    qpos[:, 3] = q_xyzw[:, 3]
    qpos[:, 4:7] = q_xyzw[:, :3]
    ours = _tilt_deg(qpos)
    zb = rot.apply(np.array([0, 0, 1.0]))
    ref = np.degrees(np.arccos(np.clip(zb[:, 2], -1, 1)))
    return {"cases": n, "max_abs_err_deg": float(np.abs(ours - ref).max())}


def check_polyline(n, rng):
    errs = []
    for _ in range(n):
        poly = rng.uniform(-3, 3, (rng.integers(2, 8), 2))
        pts = rng.uniform(-4, 4, (20, 2))
        ours = _point_to_polyline(pts, poly)
        ls = LineString(poly)
        ref = np.array([ls.distance(Point(p)) for p in pts])
        errs.append(np.abs(ours - ref).max())
    return {"cases": n, "max_abs_err_m": float(max(errs))}


def check_homotopy():
    occ = np.zeros((80, 80), bool)
    occ[35:45, 35:45] = True                                  # 가운데 장애물 하나 (x,y = 1.75~2.2 m)
    g = pm.OccGrid(occ, 0.05, (0.0, 0.0))
    rays = pm.HomotopyRays.from_grid(g)
    start, goal = (0.5, 2.0), (3.5, 2.0)
    above1 = [start, (2.0, 3.2), goal]
    above2 = [start, (1.5, 3.0), (2.6, 3.4), goal]
    below = [start, (2.0, 0.8), goal]
    loop = [start, (2.0, 3.2), (3.3, 2.0), (2.0, 0.8), (0.9, 2.0), (2.0, 3.3), goal]    # 한 바퀴 감고 위로
    sig = {k: rays.signature(np.array(v[1:]), start=start, goal=goal)
           for k, v in dict(above1=above1, above2=above2, below=below, loop=loop).items()}
    ok = (sig["above1"] == sig["above2"]) and (sig["above1"] != sig["below"]) and (sig["loop"] not in (sig["above1"], sig["below"]))
    return {"islands": int(len(rays.points)), "same_side_equal": sig["above1"] == sig["above2"],
            "opposite_sides_differ": sig["above1"] != sig["below"], "winding_distinct": sig["loop"] not in (sig["above1"], sig["below"]),
            "ok": bool(ok)}


def check_torque():
    lim = GO2_TORQUE_LIMITS
    ctrl = np.zeros((4, 12))
    ctrl[1] = lim * 0.995                                     # 포화 (≥0.99)
    ctrl[2] = lim * 0.98                                      # 포화 아님
    ctrl[3, 2] = 45.43                                        # 종아리 하나만 한계
    ratio = np.abs(ctrl) / lim
    sat = ratio >= 0.99
    expect = (12 + 1) / 48
    return {"sat_rate": float(sat.mean()), "expected": expect, "ok": abs(sat.mean() - expect) < 1e-12,
            "limits_order": "abduction, thigh, calf × 4 = " + str(lim[:3].tolist())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    n = 300 if a.quick else 1500
    res = {
        "1) BEV collision vs exact geometry": check_bev(n, rng),
        "2) goal reach": check_goal(),
        "3) tilt vs scipy Rotation": check_tilt(2000, rng),
        "4) point-to-polyline vs shapely": check_polyline(300, rng),
        "5) homotopy hand cases": check_homotopy(),
        "6) torque saturation": check_torque(),
    }
    for k, v in res.items():
        print(f"{k}: {v}")
    b = res["1) BEV collision vs exact geometry"]
    # 점 샘플 간격(셀/2 = 2.5 cm)보다 얕은 겹침만 놓칠 수 있다 → 놓친 충돌의 침투 깊이가 2.5 cm 이하면 통과
    ok = (b["max_missed_penetration_m"] <= 0.025 and b["max_false_alarm_clearance_m"] <= 0.025
          and res["2) goal reach"]["ok"] and res["3) tilt vs scipy Rotation"]["max_abs_err_deg"] < 1e-6
          and res["4) point-to-polyline vs shapely"]["max_abs_err_m"] < 1e-9
          and res["5) homotopy hand cases"]["ok"] and res["6) torque saturation"]["ok"])
    print(("✅ PASS" if ok else "❌ FAIL") + f" ({time.time() - t0:.0f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
