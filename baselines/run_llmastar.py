"""
run_llmastar.py ─ 표 ② LLM-A* 행 실행 (Gemini 2.5 Pro, temperature 0)

  python baselines/run_llmastar.py --scenes indoor_000 indoor_001 --maps gt seg --trials 3

출력: data/<scene>/llmastar_<gt|seg>/trial_<i>/{last_judged_path.json, meta.json, llm_response.txt}
seg 지도는 run_table2_baselines.py가 만든 data/<scene>/seg_occupancy.npz를 쓴다 (없으면 먼저 그걸 실행).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.plan_metrics import OccGrid  # noqa: E402
from baselines.grid_planners import DEFAULT_INFLATE_M  # noqa: E402
from baselines.llmastar_adapter import LATTICE_UNIT_M, GeminiAsk, plan_llmastar  # noqa: E402
from experiments.trial_io import clear_execution  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--maps", nargs="+", default=["gt", "seg"], choices=["gt", "seg"])
    ap.add_argument("--trials", type=int, default=3, help="temperature 0이라 거의 같지만 API 변동 확인용")
    ap.add_argument("--gemini-model", default="gemini-2.5-pro")
    ap.add_argument("--inflate", type=float, default=DEFAULT_INFLATE_M)
    ap.add_argument("--lattice-unit", type=float, default=LATTICE_UNIT_M, help="LLM에게 줄 격자 간격 (m)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from vlm_courtroom.config import init_vertex_ai
    init_vertex_ai()

    for scene in a.scenes:
        sd = os.path.join(REPO, "data", scene)
        sm = json.load(open(os.path.join(sd, "scene_meta.json")))
        for src in a.maps:
            f = os.path.join(sd, "occupancy.npz" if src == "gt" else "seg_occupancy.npz")
            if not os.path.exists(f):
                print(f"⚠️  {scene}: {os.path.basename(f)} 없음 → run_table2_baselines.py 먼저 실행"); continue
            grid = OccGrid.load(f)
            method = f"llmastar_{src}"
            for t in range(a.trials):
                out = os.path.join(sd, method, f"trial_{t:02d}")
                mp = os.path.join(out, "meta.json")
                if os.path.exists(mp) and not a.force and not json.load(open(mp)).get("api_error"):
                    print(f"{scene}/{method}/trial_{t:02d}: 이미 있음 (건너뜀)"); continue
                client = GeminiAsk(a.gemini_model, temperature=0.0)
                r = plan_llmastar(grid, sm["goal"], sm["num_waypoints"], client, inflate_m=a.inflate,
                                  unit_m=a.lattice_unit)
                os.makedirs(out, exist_ok=True)
                clear_execution(out)
                pj = os.path.join(out, "last_judged_path.json")
                if os.path.exists(pj):
                    os.remove(pj)
                if r["path"] is not None:
                    json.dump([{"x": float(x), "y": float(y)} for x, y in r["path"]], open(pj, "w"), indent=2)
                texts = [c.get("text", "") for c in client.calls]
                open(os.path.join(out, "llm_response.txt"), "w").write("\n\n-----\n\n".join(texts))
                tok = lambda k: sum((c.get(k) or 0) for c in client.calls)
                meta = {"scene": scene, "config": method, "trial": t, "algorithm": "llm-astar", "map_source": src,
                        "model": a.gemini_model, "temperature": 0.0, "prompt": "standard (5-shot, original)",
                        "parse_ok": r["path"] is not None, "n_points": 0 if r["path"] is None else len(r["path"]),
                        "n_requested": sm["num_waypoints"], "plan_time_s": round(r["plan_time_s"], 3),
                        "n_calls": len(client.calls), "prompt_tokens": tok("prompt_tokens"),
                        "output_tokens": tok("output_tokens"), "thinking_tokens": tok("thinking_tokens"),
                        "api_error": bool(r.get("api_error", False)),
                        **{k: v for k, v in r.items() if k not in ("path", "plan_time_s", "api_error")}}
                json.dump(meta, open(mp, "w"), indent=2, default=str)
                print(f"{scene}/{method}/trial_{t:02d}: {'ok' if r['path'] is not None else '✗ ' + str(r.get('reason'))}"
                      f" (barriers {r.get('n_barriers')}, lattice {r.get('lattice_size')})", flush=True)


if __name__ == "__main__":
    main()
