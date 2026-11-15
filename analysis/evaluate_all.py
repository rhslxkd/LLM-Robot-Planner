"""
evaluate_all.py ─ 모든 시행의 계획·실행 지표를 계산해 CSV로 저장 (표 생성의 유일한 입력)

  python analysis/evaluate_all.py --scenes indoor_000 indoor_001 --out results/

출력
  results/trials.csv   시행 1개 = 1행 (계획 지표 + 실행 지표 + 비용)
  results/groups.csv   (씬, 방법) 1개 = 1행 (유효율, homotopy class 수, 쌍별 거리)

제외 규칙: meta.json의 api_error=True (할당량·네트워크 오류)인 시행은 모델 실패가 아니므로 제외하고
excluded 열에 표시한다. run_ablation.py를 다시 돌리면 그런 시행만 재실행된다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.episode_metrics import evaluate_rollout  # noqa: E402
from analysis.plan_metrics import HomotopyRays, OccGrid, diversity, evaluate_path  # noqa: E402

MODELS_DIR = os.path.join(REPO, "dial_mpc", "dial_mpc", "models", "unitree_go2")


def evaluate_scene(scene: str) -> tuple[list[dict], list[dict]]:
    sd = os.path.join(REPO, "data", scene)
    sm = json.load(open(os.path.join(sd, "scene_meta.json")))
    grid = OccGrid.load(os.path.join(sd, "occupancy.npz"))
    rays = HomotopyRays.from_grid(grid)
    goal = sm["goal"]
    scene_xml = os.path.join(MODELS_DIR, f"{scene}.xml")
    rows, by_method = [], {}
    for d in sorted(glob.glob(os.path.join(sd, "*", "trial_*"))):
        method = os.path.basename(os.path.dirname(d))
        meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
        pj = os.path.join(d, "last_judged_path.json")
        path = json.load(open(pj)) if os.path.exists(pj) else None
        row = {"scene": scene, "method": method, "trial": int(d[-2:]),
               "excluded": bool(meta.get("api_error", False)),
               "n_calls": meta.get("n_calls"), "prompt_tokens": meta.get("prompt_tokens"),
               "output_tokens": meta.get("output_tokens"),
               "plan_time_s": meta.get("plan_time_s", meta.get("vlm_latency_s")),
               "count_ok": bool(path is not None and len(path) == meta.get("n_requested", len(path)))}
        pm = evaluate_path(path, grid, goal) if path is not None else evaluate_path(None, grid, goal)
        row.update({f"plan_{k}": v for k, v in pm.items()})
        row["homotopy"] = str(rays.signature(path, goal=goal)) if pm["valid"] else ""
        states = sorted(glob.glob(os.path.join(d, "*_states.npy")))
        row["executed"] = bool(states)
        if states:
            ex = evaluate_rollout(np.load(states[-1]), scene_xml,
                                  np.array([[p["x"], p["y"]] for p in path]), goal)
            ex["collision_parts"] = json.dumps(ex["collision_parts"])
            row.update({f"exec_{k}": v for k, v in ex.items()})
        else:
            row["exec_success"] = False
        rows.append(row)
        if not row["excluded"]:
            by_method.setdefault(method, []).append(path)
    groups = []
    for method, paths in by_method.items():
        g = diversity([p for p in paths], grid, goal, rays) if paths else {}
        groups.append({"scene": scene, "method": method, **g})
    return rows, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, groups = [], []
    for s in a.scenes:
        r, g = evaluate_scene(s)
        rows += r; groups += g
        print(f"✅ {s}: 시행 {len(r)}개 (실행됨 {sum(x['executed'] for x in r)}, 제외 {sum(x['excluded'] for x in r)})")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "trials.csv"), index=False)
    pd.DataFrame(groups).to_csv(os.path.join(a.out, "groups.csv"), index=False)
    print(f"→ {a.out}/trials.csv, groups.csv")


if __name__ == "__main__":
    main()
