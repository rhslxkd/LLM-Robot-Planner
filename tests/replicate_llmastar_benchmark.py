"""
replicate_llmastar_benchmark.py ─ LLM을 Gemini로 바꾼 LLM-A*가 원 논문의 결과 경향을 재현하는지 확인 (서버에서 실행)

원 논문 벤치마크(51×31 지도)에서 원본 A*(llmastar.pather.AStar)와 LLM-A*(원본 탐색 + Gemini 2.5 Pro, temperature 0)를
같은 질의로 돌리고, 논문 표 1과 같은 지표를 계산한다.
  Operation / Storage Ratio : LLM-A* ÷ A* 의 기하평균 (%) ↓
  Relative Path Length      : 경로 길이 비의 기하평균 (%) ↓
  Valid Path Ratio          : 목표까지 경로를 찾은 비율

원 논문 표 1 (GPT-3.5, few-shot): Operation 57.39 / Storage 74.96 / Path 102.44 / Valid 100
→ 비슷한 경향(연산·저장 감소, 경로 길이 ≈ 102%)이 나오면 "LLM을 바꿔도 방법이 의도대로 작동한다"는 근거가 된다.

  python tests/replicate_llmastar_benchmark.py --dataset /tmp/llm-astar/dataset/environment_50_30.json --n 100
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from baselines.llmastar_adapter import GeminiAsk, _import_llmastar, make_llmastar  # noqa: E402

PAPER = {"operation": 57.39, "storage": 74.96, "length": 102.44, "valid": 100.0}


def gmean_pct(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean()) * 100) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=100, help="앞에서부터 쓸 질의 수 (전체 1,000)")
    ap.add_argument("--gemini-model", default="gemini-2.5-pro")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "llmastar_replication.json"))
    a = ap.parse_args()

    from vlm_courtroom.config import init_vertex_ai
    init_vertex_ai()
    _import_llmastar()
    from llmastar.pather.a_star.a_star import AStar

    envs = json.load(open(a.dataset))
    queries = [{"start": sg[0], "goal": sg[1], "size": [e["range_x"][1], e["range_y"][1]],
                "horizontal_barriers": e["horizontal_barriers"], "vertical_barriers": e["vertical_barriers"],
                "range_x": e["range_x"], "range_y": e["range_y"]} for e in envs for sg in e["start_goal"]][:a.n]
    tmp = tempfile.mkdtemp()
    rows = []
    for i, q in enumerate(queries):
        base = AStar().searching(query=copy.deepcopy(q), filepath=os.path.join(tmp, "a.png"))
        client = GeminiAsk(a.gemini_model, temperature=0.0)
        planner = make_llmastar(client, "standard")
        try:
            res = planner.searching(query=copy.deepcopy(q))
            ok = planner.path_ is not None and tuple(planner.path_[-1]) == tuple(q["goal"])
        except Exception as ex:
            res, ok = {"error": f"{type(ex).__name__}: {ex}"}, False
        rows.append({"i": i, "astar": base, "llmastar": res, "valid": ok,
                     "llm_text": client.calls[-1].get("text") if client.calls else None})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)}", flush=True)

    v = [r for r in rows if r["valid"]]
    ours = {
        "operation": gmean_pct([r["llmastar"]["operation"] / r["astar"]["operation"] for r in v]),
        "storage": gmean_pct([r["llmastar"]["storage"] / r["astar"]["storage"] for r in v]),
        "length": gmean_pct([r["llmastar"]["length"] / r["astar"]["length"] for r in v]),
        "valid": 100.0 * len(v) / max(len(rows), 1),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"n": len(rows), "model": a.gemini_model, "ours": ours, "paper_gpt35_fewshot": PAPER, "rows": rows},
              open(a.out, "w"), indent=2, default=str)
    print(f"\n{'':22s}{'Operation':>11s}{'Storage':>10s}{'Path len':>10s}{'Valid':>8s}")
    print(f"{'Paper (GPT-3.5)':22s}{PAPER['operation']:11.2f}{PAPER['storage']:10.2f}{PAPER['length']:10.2f}{PAPER['valid']:8.1f}")
    print(f"{'Ours (' + a.gemini_model + ')':22s}{ours['operation']:11.2f}{ours['storage']:10.2f}{ours['length']:10.2f}{ours['valid']:8.1f}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
