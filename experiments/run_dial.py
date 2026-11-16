"""
run_dial.py ─ 계획이 끝난 모든 시행을 DIAL-MPC로 실행 (표 ①·② 공통, 연구실 서버 GPU에서 실행)

data/<scene>/<method>/trial_<i>/last_judged_path.json 이 있고 *_states.npy가 아직 없는 시행을 찾아
  python dial_mpc/dial_mpc/core/dial_core.py --example <scene> --vlm-path-json ... --output-dir ... --n-steps N
를 순서대로 실행한다. n_steps = 경로 길이 / 순항속도(0.8 m/s) / dt + 여유 150 step.

  python experiments/run_dial.py --scenes indoor_000 indoor_001            # 모든 방법
  python experiments/run_dial.py --scenes indoor_000 --methods C4 astar_gt  # 일부만
  python experiments/run_dial.py --scenes indoor_000 --dry-run              # 실행할 목록만 출력
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from indoor.build_scene import n_steps_for  # noqa: E402

JIT_OVERHEAD_S = 150          # 첫 컴파일 포함 여유
SEC_PER_STEP = 0.2            # 실측: oracle A 400 step ≈ 38 s (RTX 5060 Ti) → 넉넉히


def path_length(path: list) -> float:
    p = np.array([[0.0, 0.0]] + [[q["x"], q["y"]] for q in path])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def pending(scenes, methods, force=False):
    jobs = []
    for scene in scenes:
        for d in sorted(glob.glob(os.path.join(REPO, "data", scene, "*", "trial_*"))):
            method = os.path.basename(os.path.dirname(d))
            if methods and method not in methods:
                continue
            pj = os.path.join(d, "last_judged_path.json")
            if not os.path.exists(pj):
                continue
            if glob.glob(os.path.join(d, "*_states.npy")) and not force:
                continue
            jobs.append((scene, method, d, pj))
    return jobs


def run_one(scene, d, pj, python=sys.executable):
    path = json.load(open(pj))
    n_steps = n_steps_for(path_length(path))
    timeout = JIT_OVERHEAD_S + int(SEC_PER_STEP * n_steps)
    cmd = [python, "dial_mpc/dial_mpc/core/dial_core.py", "--example", scene,
           "--vlm-path-json", os.path.abspath(pj), "--output-dir", os.path.abspath(d), "--n-steps", str(n_steps),
           "--no-serve"]
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    t0 = time.perf_counter()
    with open(os.path.join(d, "dial.log"), "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                rc = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); rc = -9
    states = sorted(glob.glob(os.path.join(d, "*_states.npy")))
    info = {"n_steps": n_steps, "timeout_s": timeout, "rc": rc, "wall_s": round(time.perf_counter() - t0, 1),
            "states": os.path.basename(states[-1]) if states else None}
    json.dump(info, open(os.path.join(d, "dial_meta.json"), "w"), indent=2)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="states.npy가 있어도 다시 실행")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    jobs = pending(a.scenes, a.methods, a.force)
    print(f"🦿 DIAL-MPC 실행 대상 {len(jobs)}개")
    for i, (scene, method, d, pj) in enumerate(jobs, 1):
        rel = os.path.relpath(d, REPO)
        if a.dry_run:
            print(f"  [{i}] {rel}  n_steps={n_steps_for(path_length(json.load(open(pj))))}")
            continue
        info = run_one(scene, d, pj)
        ok = "✅" if info["states"] else "❌"
        print(f"[{i}/{len(jobs)}] {ok} {rel}  n_steps={info['n_steps']} rc={info['rc']} {info['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
