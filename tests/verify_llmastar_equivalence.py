"""
verify_llmastar_equivalence.py ─ "LLM-A*를 원본 그대로 썼다"를 보이는 검증

원본 llmastar.LLMAStar(무수정, 그림 저장까지 그대로)와 우리 래퍼(baselines/llmastar_adapter.make_llmastar)에
같은 LLM 응답을 주고, 원 논문 벤치마크 1,000개 질의(지도 100개 × 시작/목표 10쌍)에서 아래가 전부 같은지 비교한다.
  - LLM에 보낸 프롬프트 문자열
  - 탐색 결과: operation(확장 노드 수), storage, length(경로 길이), llm_output(필터 후 LLM target)
  - 최종 경로 (격자점 순서열)

LLM 응답은 실제 API 대신 질의마다 결정적으로 만든 가짜 응답을 쓴다 (정상 좌표 / 장애물·범위 밖 좌표 /
파싱 불가 문장 / 시작·목표만). 래퍼가 바꾼 것은 LLM 호출 객체뿐이므로, 같은 응답이면 결과가 같아야 한다.

  git clone --depth 1 https://github.com/SilinMeng0510/llm-astar /tmp/llm-astar
  python tests/verify_llmastar_equivalence.py --dataset /tmp/llm-astar/dataset/environment_50_30.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from baselines.llmastar_adapter import _import_llmastar, make_llmastar  # noqa: E402


class FakeLLM:
    """프롬프트 마지막 질의의 시작/목표로 결정적인 응답을 만든다. 보낸 프롬프트를 기록한다."""

    def __init__(self):
        self.prompts = []

    def ask(self, prompt, stop=None, max_tokens=None):
        self.prompts.append(prompt)
        last = prompt.split("Start Point:")[-1]
        st = json.loads(re.search(r"\[[^\]]*\]", last).group(0))
        go = json.loads(re.search(r"Goal Point: (\[[^\]]*\])", last).group(1))
        seed = (st[0] * 7919 + st[1] * 104729 + go[0] * 1299709 + go[1] * 15485863) % (2 ** 32)
        rng = np.random.default_rng(seed)
        kind = seed % 4
        if kind == 2:
            return "I would go around the obstacles on the left side."
        if kind == 3:
            return f"Generated Path: {json.dumps([st, go])}"
        mids = [[int(rng.integers(-3, 54)), int(rng.integers(-3, 34))] for _ in range(rng.integers(1, 5))]
        if kind == 0:
            mids = [[min(max(x, 2), 48), min(max(y, 2), 28)] for x, y in mids]
        return f"Generated Path: {json.dumps([st] + mids + [go])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    LLMAStar, _ = _import_llmastar()
    envs = json.load(open(a.dataset))
    queries = []
    for e in envs:
        for sg in e["start_goal"]:
            queries.append({"start": sg[0], "goal": sg[1], "size": [e["range_x"][1], e["range_y"][1]],
                            "horizontal_barriers": e["horizontal_barriers"], "vertical_barriers": e["vertical_barriers"],
                            "range_x": e["range_x"], "range_y": e["range_y"]})
    if a.limit:
        queries = queries[:a.limit]
    t0 = time.time()
    diff = {"prompt": 0, "result": 0, "path": 0, "exception_mismatch": 0}
    both_failed = 0
    tmp = tempfile.mkdtemp()
    for i, q in enumerate(queries):
        f_orig, f_ours = FakeLLM(), FakeLLM()
        orig = LLMAStar(llm="gpt", prompt="standard")           # 원본 그대로 (그림 저장 포함)
        orig.model = f_orig                                    # 원본도 self.model.ask(...)로 LLM을 부른다
        ours = make_llmastar(f_ours, "standard")
        out = {}
        for name, planner, fp in (("orig", orig, os.path.join(tmp, "o.png")), ("ours", ours, None)):
            try:
                res = planner.searching(query=copy.deepcopy(q), **({"filepath": fp} if fp else {}))
                path = planner.extract_path(planner.PARENT) if name == "orig" else planner.path_
                out[name] = ("ok", res, path)
            except Exception as ex:
                out[name] = ("err", type(ex).__name__, None)
        if out["orig"][0] != out["ours"][0]:
            diff["exception_mismatch"] += 1
            continue
        if out["orig"][0] == "err":
            both_failed += 1
            if out["orig"][1] != out["ours"][1]:
                diff["exception_mismatch"] += 1
            continue
        diff["prompt"] += f_orig.prompts != f_ours.prompts
        diff["result"] += out["orig"][1] != out["ours"][1]
        diff["path"] += out["orig"][2] != out["ours"][2]
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(queries)} ...", flush=True)
    ok = sum(diff.values()) == 0
    print(f"queries={len(queries)}  differences={diff}  both_failed_identically={both_failed}")
    print(f"{'✅ PASS: 원본과 래퍼의 프롬프트·탐색 결과·경로가 모두 동일' if ok else '❌ FAIL'} ({time.time() - t0:.0f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
