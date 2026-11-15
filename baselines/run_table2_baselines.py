"""
run_table2_baselines.py ─ 표 ② 고전 베이스라인 실행 (계획 단계만. DIAL-MPC는 experiments/run_dial.py)

방법 (표의 행 이름 = 출력 폴더 이름)
  astar_gt      A*       + GT 점유 지도
  astar_seg     A*       + 이미지 분할 지도 (baselines/image_occupancy.py)
  dijkstra_gt   Dijkstra + GT 점유 지도
  rrtstar_gt    RRT*     + GT 점유 지도   (seed = trial 번호 → 시행마다 다른 경로)
  rrtstar_seg   RRT*     + 이미지 분할 지도

결정적 방법(A*, Dijkstra)은 시행을 반복해도 경로가 같으므로 --det-trials(기본 1)만큼만 돌린다.

출력 (VLM 구성과 같은 형식): data/<scene>/<method>/trial_<i>/{last_judged_path.json, meta.json}
  + data/<scene>/seg_occupancy.npz, seg_check.png (분할 결과 확인용)

  python baselines/run_table2_baselines.py --scenes indoor_000 indoor_001 --trials 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.plan_metrics import OccGrid  # noqa: E402
from baselines.grid_planners import DEFAULT_INFLATE_M, plan_grid, plan_rrtstar  # noqa: E402
from baselines.image_occupancy import occupancy_from_image  # noqa: E402

METHODS = ("astar_gt", "astar_seg", "dijkstra_gt", "rrtstar_gt", "rrtstar_seg")
DETERMINISTIC = {"astar_gt", "astar_seg", "dijkstra_gt"}


def seg_grid(scene_dir: str, sm: dict, gt: OccGrid, tol: float) -> OccGrid:
    path = os.path.join(scene_dir, "seg_occupancy.npz")
    bev = np.array(Image.open(os.path.join(scene_dir, "bev_raw.png")).convert("RGB"))
    seg = occupancy_from_image(bev, sm["ppm"], sm["robot_px"], gt, tol=tol)
    seg.save(path)
    # 확인 이미지: 분할상 장애물(파랑)과 GT 장애물(빨강) 비교. 보라 = 둘 다
    H, W = bev.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    xy = np.stack([(uu - sm["robot_px"][0]) / sm["ppm"], (sm["robot_px"][1] - vv) / sm["ppm"]], -1)
    a, b = seg.occupied(xy), gt.occupied(xy)
    out = bev.astype(float)
    out[a & ~b] = 0.5 * out[a & ~b] + 0.5 * np.array([40, 90, 255])
    out[b & ~a] = 0.5 * out[b & ~a] + 0.5 * np.array([255, 30, 30])
    out[a & b] = 0.5 * out[a & b] + 0.5 * np.array([170, 40, 200])
    Image.fromarray(out.astype(np.uint8)).save(os.path.join(scene_dir, "seg_check.png"))
    return seg


def seg_agreement(seg: OccGrid, gt: OccGrid, margin_m: float = 0.3) -> float:
    """로봇이 실제로 갈 수 있는 영역(GT에서 시작점과 연결된 빈 공간, margin만큼 확장) 안에서만 일치율.
    방 바깥(어차피 못 가는 곳)은 분할과 GT의 정의가 달라서 제외한다."""
    from scipy import ndimage
    lab, _ = ndimage.label(~gt.occ)
    r, c = (int(v) for v in gt.to_cell(np.array([0.0, 0.0])))
    reach = lab == lab[r, c]
    k = int(np.ceil(margin_m / gt.res))
    reach = ndimage.binary_dilation(reach, iterations=k)
    return float((seg.occ[reach] == gt.occ[reach]).mean())


def run_scene(scene: str, methods, trials: int, det_trials: int, inflate_m: float, seg_tol: float) -> list[str]:
    scene_dir = os.path.join(REPO, "data", scene)
    sm = json.load(open(os.path.join(scene_dir, "scene_meta.json")))
    gt = OccGrid.load(os.path.join(scene_dir, "occupancy.npz"))
    seg = seg_grid(scene_dir, sm, gt, seg_tol) if any(m.endswith("_seg") for m in methods) else None
    agree = seg_agreement(seg, gt) if seg is not None else None
    goal, n = sm["goal"], sm["num_waypoints"]
    logs = []
    for method in methods:
        algo, src = method.split("_")
        grid = gt if src == "gt" else seg
        for t in range(det_trials if method in DETERMINISTIC else trials):
            if algo in ("astar", "dijkstra"):
                r = plan_grid(grid, goal, n, method=algo, inflate_m=inflate_m)
            else:
                r = plan_rrtstar(grid, goal, n, seed=t, inflate_m=inflate_m)
            out = os.path.join(scene_dir, method, f"trial_{t:02d}")
            os.makedirs(out, exist_ok=True)
            pj = os.path.join(out, "last_judged_path.json")
            if os.path.exists(pj):
                os.remove(pj)
            if r["path"] is not None:
                json.dump([{"x": float(x), "y": float(y)} for x, y in r["path"]], open(pj, "w"), indent=2)
            meta = {"scene": scene, "config": method, "trial": t, "algorithm": algo, "map_source": src,
                    "parse_ok": r["path"] is not None, "n_points": 0 if r["path"] is None else len(r["path"]),
                    "n_requested": n, "plan_time_s": round(r["plan_time_s"], 4), "n_calls": 0, "api_error": False,
                    "inflate_m": inflate_m, "seg_tol": seg_tol if src == "seg" else None,
                    "seg_cell_agreement": agree if src == "seg" else None,
                    **{k: v for k, v in r.items() if k in ("expanded", "n_nodes", "reason", "seed")}}
            json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
            logs.append(f"{scene}/{method}/trial_{t:02d}: {'ok' if r['path'] is not None else '✗ ' + r.get('reason', '')}"
                        f" ({r['plan_time_s'] * 1000:.0f} ms)")
    return logs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--trials", type=int, default=5, help="확률적 방법(RRT*)의 시행 수")
    ap.add_argument("--det-trials", type=int, default=1, help="결정적 방법(A*, Dijkstra)의 시행 수")
    ap.add_argument("--inflate", type=float, default=DEFAULT_INFLATE_M, help="장애물 팽창 반경 (m)")
    ap.add_argument("--seg-tol", type=float, default=28.0, help="이미지 분할 색 거리 허용치")
    a = ap.parse_args()
    for s in a.scenes:
        for line in run_scene(s, a.methods, a.trials, a.det_trials, a.inflate, a.seg_tol):
            print(line)


if __name__ == "__main__":
    main()
