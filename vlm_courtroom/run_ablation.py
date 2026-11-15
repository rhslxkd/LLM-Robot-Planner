"""
run_ablation.py ─ 표 ① Agent ablation 실행기 (VLM 단계만. DIAL-MPC는 experiments/run_dial.py)

구성 (모든 구성이 같은 프롬프트 본문·물리 제약·이미지를 쓴다)
  C1    Coordinate 1회                                          호출 1
  C1xk  Coordinate k회 → 메도이드 경로 선택 (호출 수 대조군)     호출 k (기본 4 = C4와 동일)
  C2    Coordinate → Prosecutor → Coordinate 수정               호출 3
  C3    Coordinate → Prosecutor → Defense → Coordinate 수정     호출 4
  C4    Coordinate → Prosecutor → Defense → Judge (기존 courtroom 그대로)   호출 4

메도이드: k개 경로를 50점으로 재표본화하고, 다른 경로들과의 평균 Hausdorff 거리 합이 최소인 경로.
판사 없이도 "여러 번 뽑아 가장 대표적인 것"을 고르는 self-consistency 방식이라 정답 정보를 쓰지 않는다.

이미지 접근 (--image-mode)
  coord (기본) : 기존 시스템과 동일. Coordinate(와 수정 단계)만 이미지를 본다.
  all          : 검사·변호·판사도 이미지를 본다.

출력: data/<scene>/<config>/trial_<i>/
  last_judged_path.json   최종 경로 (DIAL-MPC 입력, 파싱 실패 시 없음)
  transcript.json         단계별 응답 원문 + 호출 로그
  meta.json               구성, 호출 수, 토큰, 시간, 파싱 결과, API 오류 여부
  candidates.json         (C1xk만) k개 후보와 선택 인덱스

예시
  python vlm_courtroom/run_ablation.py --scenes indoor_000 indoor_001 --configs C1 C1xk C2 C3 C4 \
      --trials 5 --gemini-model gemini-2.5-pro
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CONFIGS = ("C1", "C1xk", "C2", "C3", "C4")


# ─── 경로 추출 / 메도이드 ───────────────────────────────────────────────
_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_ARRAY = re.compile(r"\[\s*\{[^\[\]]*?\}(?:\s*,\s*\{[^\[\]]*?\})*\s*\]", re.DOTALL)


def _try_parse(s: str):
    s = re.sub(r"'([a-zA-Z0-9_]+)'\s*:", r'"\1":', s)
    s = re.sub(r":\s*'([^']*)'", r': "\1"', s)
    try:
        arr = json.loads(s)
        pts = [{"x": float(p["x"]), "y": float(p["y"])} for p in arr]
        return pts if len(pts) >= 2 else None
    except (ValueError, TypeError, KeyError):
        return None


def extract_path(text: str):
    """응답에서 마지막 좌표 JSON 배열을 꺼낸다 (```json 블록 우선, 없으면 본문의 [{...}, ...]).
    프롬프트 예시를 그대로 따라 쓴 경우를 피하려고 '마지막' 것을 쓴다."""
    if not isinstance(text, str) or text.startswith("Error"):
        return None
    for cands in (_FENCE.findall(text), _ARRAY.findall(text)):
        for c in reversed(cands):
            p = _try_parse(c)
            if p is not None:
                return p
    return None


def _resample(poly: np.ndarray, n: int = 50) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(poly[:1], n, 0)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], -1)


def medoid_index(paths: list) -> int:
    rs = [_resample(np.array([[p["x"], p["y"]] for p in path])) for path in paths]
    n = len(rs)
    if n <= 2:
        return 0
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(rs[i][:, None] - rs[j][None], axis=-1)
            D[i, j] = D[j, i] = (d.min(1).mean() + d.min(0).mean()) / 2
    return int(np.argmin(D.sum(1)))


# ─── 한 시행 ────────────────────────────────────────────────────────────
def make_agents(agent_kwargs: dict):
    from vlm_courtroom.agents.specific_agents import (CoordinateAgent, DefenseAttorneyAgent, JudgeAgent,
                                                      ProsecutorAgent)
    return (CoordinateAgent(**agent_kwargs), ProsecutorAgent(**agent_kwargs),
            DefenseAttorneyAgent(**agent_kwargs), JudgeAgent(**agent_kwargs))


def run_trial(config: str, scenario: str, image_path: str, num_waypoints: int, agent_kwargs: dict,
              image_mode: str = "coord", k: int = 4) -> dict:
    coord, pros, dfn, judge = make_agents(agent_kwargs)
    critic_img = image_path if image_mode == "all" else None
    steps, extra = [], {}

    def log(step, msg):
        steps.append({"step": step, "agent": msg.sender, "role": msg.role, "text": msg.content})
        return msg

    base = {"image_description": scenario, "image_path": image_path, "num_waypoints": num_waypoints}
    if config == "C1":
        final = log("coordinate", coord.process(base))
    elif config == "C1xk":
        cands, texts = [], []
        for i in range(k):
            m = log(f"coordinate_{i}", coord.process(base))
            texts.append(m.content)
            cands.append(extract_path(m.content))
        parsed = [(i, p) for i, p in enumerate(cands) if p is not None]
        if parsed:
            sel = parsed[medoid_index([p for _, p in parsed])][0]
            final_text = texts[sel]
        else:
            sel, final_text = -1, texts[-1]
        extra = {"candidates": cands, "selected_index": sel, "n_parsed_candidates": len(parsed)}
        final = type("M", (), {"content": final_text})()
    else:
        p0 = log("coordinate", coord.process(base))
        p1 = log("prosecutor", pros.process({"last_message_content": p0.content, "image_path": critic_img}))
        if config == "C2":
            final = log("revise", coord.revise({**base, "original_proposal": p0.content,
                                                "prosecution_argument": p1.content}))
        else:
            p2 = log("defense", dfn.process({"last_message_content": p0.content,
                                             "prosecution_argument": p1.content, "image_path": critic_img}))
            if config == "C3":
                final = log("revise", coord.revise({**base, "original_proposal": p0.content,
                                                    "prosecution_argument": p1.content,
                                                    "defense_argument": p2.content}))
            elif config == "C4":
                final = log("judge", judge.process({"original_proposal": p0.content,
                                                    "prosecution_argument": p1.content,
                                                    "defense_argument": p2.content,
                                                    "num_waypoints": num_waypoints, "image_path": critic_img}))
            else:
                raise ValueError(config)

    calls = sorted(coord.call_log + pros.call_log + dfn.call_log + judge.call_log, key=lambda c: c["t_start"])
    path = extract_path(final.content)
    tok = lambda key: sum((c.get(key) or 0) for c in calls)
    return {
        "path": path,
        "steps": steps,
        "calls": calls,
        "meta": {
            "n_calls": len(calls),
            "prompt_tokens": tok("prompt_tokens"),
            "output_tokens": tok("output_tokens"),
            "thinking_tokens": tok("thinking_tokens"),
            "vlm_latency_s": round(sum(c["latency_s"] for c in calls), 2),
            "api_error": any(c["api_error"] for c in calls),
            "parse_ok": path is not None,
            "n_points": len(path) if path else 0,
            "n_requested": num_waypoints,
            **{k_: v for k_, v in extra.items() if k_ != "candidates"},
        },
        "candidates": extra.get("candidates"),
    }


# ─── 씬 / 실행 ──────────────────────────────────────────────────────────
def load_scene(scene: str) -> dict:
    meta_path = os.path.join(REPO, "data", scene, "scene_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} 없음 — indoor/build_scene.py로 먼저 씬을 만들 것")
    meta = json.load(open(meta_path))
    meta["image_path"] = os.path.join(REPO, "data", scene, "oracle.png")
    return meta


def trial_dir(scene: str, config: str, trial: int) -> str:
    return os.path.join(REPO, "data", scene, config, f"trial_{trial:02d}")


def save_trial(out: str, scene: str, config: str, trial: int, res: dict, run_info: dict) -> None:
    os.makedirs(out, exist_ok=True)
    stale = os.path.join(out, "last_judged_path.json")
    if os.path.exists(stale):
        os.remove(stale)
    if res["path"] is not None:
        json.dump(res["path"], open(stale, "w"), indent=2)
    json.dump({"steps": res["steps"], "calls": res["calls"]},
              open(os.path.join(out, "transcript.json"), "w"), indent=2, ensure_ascii=False)
    if res.get("candidates") is not None:
        json.dump({"candidates": res["candidates"], "selected_index": res["meta"].get("selected_index")},
                  open(os.path.join(out, "candidates.json"), "w"), indent=2)
    meta = {"scene": scene, "config": config, "trial": trial, **run_info, **res["meta"]}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, ensure_ascii=False)


def needs_run(out: str, rerun_api_errors: bool = True) -> bool:
    mp = os.path.join(out, "meta.json")
    if not os.path.exists(mp):
        return True
    return rerun_api_errors and json.load(open(mp)).get("api_error", False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--k", type=int, default=4, help="C1xk의 반복 횟수 (기본 4 = C4 호출 수)")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "ollama", "openai"])
    ap.add_argument("--gemini-model", default="gemini-2.5-pro")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--openai-model", default=None)
    ap.add_argument("--image-mode", default="coord", choices=["coord", "all"])
    ap.add_argument("--workers", type=int, default=1, help="동시에 돌릴 시행 수 (API 할당량 주의)")
    ap.add_argument("--no-rerun-api-errors", action="store_true")
    args = ap.parse_args()

    agent_kwargs = {"backend": args.backend}
    model_name = {"gemini": args.gemini_model, "ollama": args.ollama_model, "openai": args.openai_model}[args.backend]
    agent_kwargs[f"{args.backend}_model"] = model_name
    if args.backend == "gemini":
        from vlm_courtroom.config import init_vertex_ai
        init_vertex_ai()
    run_info = {"backend": args.backend, "model": model_name, "image_mode": args.image_mode,
                "k": args.k, "generation_config": "SDK default"}

    jobs = []
    for scene in args.scenes:
        sm = load_scene(scene)
        for config in args.configs:
            for t in range(args.trials):
                out = trial_dir(scene, config, t)
                if needs_run(out, not args.no_rerun_api_errors):
                    jobs.append((scene, sm, config, t, out))
    print(f"🧪 실행할 시행 {len(jobs)}개 (씬 {len(args.scenes)} × 구성 {len(args.configs)} × {args.trials}회 중 미완료)")

    def work(job):
        scene, sm, config, t, out = job
        t0 = time.perf_counter()
        try:
            res = run_trial(config, sm["scenario"].format(num_waypoints=sm["num_waypoints"]), sm["image_path"],
                            sm["num_waypoints"], agent_kwargs, args.image_mode, args.k)
            res["meta"]["wall_s"] = round(time.perf_counter() - t0, 2)
            save_trial(out, scene, config, t, res, run_info)
            m = res["meta"]
            return f"{scene}/{config}/trial_{t:02d}: parse={m['parse_ok']} calls={m['n_calls']} api_error={m['api_error']}"
        except Exception as e:
            os.makedirs(out, exist_ok=True)
            json.dump({"scene": scene, "config": config, "trial": t, **run_info, "api_error": True,
                       "parse_ok": False, "exception": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()}, open(os.path.join(out, "meta.json"), "w"), indent=2)
            return f"{scene}/{config}/trial_{t:02d}: ❌ {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            print(f"[{i}/{len(jobs)}] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
