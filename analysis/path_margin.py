"""
path_margin.py ─ VLM 경로의 중심선이 장애물에 얼마나 붙거나 들어갔는지 측정.

목적: 프롬프트의 '인식 여유' 값을 추측이 아니라 측정값으로 정하기 위함.
  signed distance  = 중심선 위 점에서 가장 가까운 장애물 경계까지 거리 (밖 +, 안 −)
  min_sd           = 경로 전체에서 가장 작은 signed distance
  shortfall        = 몸통 반폭(0.175 m)을 확보하려면 경로를 얼마나 더 떨어뜨려야 했는지 = max(0, 0.175 − min_sd)

  python analysis/path_margin.py --scenes replica_office_4_p0 replica_frl_apartment_1_p0
  → 시행별 표 + 규칙(1.6 m 보관본 / 현재)별 분포, results/path_margin.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.plan_metrics import GO2_FOOTPRINT_W, OccGrid, densify, evaluate_path, parse_path  # noqa: E402

HALF_W = GO2_FOOTPRINT_W / 2


def signed_field(grid: OccGrid) -> np.ndarray:
    out = ndimage.distance_transform_edt(~grid.occ) * grid.res
    inside = ndimage.distance_transform_edt(grid.occ) * grid.res
    return np.where(grid.occ, -inside, out)


def measure(path, grid, sdf, goal):
    poly = parse_path(path)
    if poly is None or len(poly) < 2:
        return None
    poly = np.vstack([[0.0, 0.0], poly]) if np.linalg.norm(poly[0]) > 1e-6 else poly
    pts, _ = densify(poly)
    r, c = grid.to_cell(pts)
    H, W = grid.occ.shape
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    sd = np.full(len(pts), -1.0)                 # 지도 밖 = 장애물 취급
    sd[ok] = sdf[r[ok], c[ok]]
    m = float(sd.min())
    return {"min_sd_m": m, "shortfall_m": max(0.0, HALF_W - m),
            "valid": evaluate_path(path, grid, goal)["valid"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "path_margin.csv"))
    a = ap.parse_args()
    rows = []
    for sc in a.scenes:
        sd_dir = os.path.join(REPO, "data", sc)
        grid = OccGrid.load(os.path.join(sd_dir, "occupancy.npz"))
        goal = json.load(open(os.path.join(sd_dir, "scene_meta.json")))["goal"]
        sdf = signed_field(grid)
        for rule, pat in (("1.6m", "_pilot_1.6m/C*/trial_*"), ("current", "C*/trial_*")):
            for d in sorted(glob.glob(os.path.join(sd_dir, pat))):
                pj = os.path.join(d, "last_judged_path.json")
                if not os.path.exists(pj):
                    continue
                m = measure(json.load(open(pj)), grid, sdf, goal)
                if m:
                    rows.append({"scene": sc, "rule": rule, "method": os.path.basename(os.path.dirname(d)),
                                 "trial": os.path.basename(d), **m})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\n규칙별 분포 (m)")
    for rule, g in df.groupby("rule"):
        sf = g["shortfall_m"]
        print(f"  {rule:8s} n={len(g):2d}  유효 {g.valid.sum()}/{len(g)}  "
              f"min_sd 중앙값 {g.min_sd_m.median():+.2f}  "
              f"부족분 중앙값 {sf.median():.2f} / p75 {sf.quantile(.75):.2f} / p90 {sf.quantile(.9):.2f} / 최대 {sf.max():.2f}")
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
