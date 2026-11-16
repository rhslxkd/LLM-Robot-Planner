"""
decide_controller.py ─ 본 실험 전에 제어기 설정 두 가지를 개발용 씬에서 결정하기 위한 실험.

  python experiments/decide_controller.py --scenes replica_office_4_p0 replica_frl_apartment_1_p0
  → results/controller_decision.md

실험 A (yaw 목표):  같은 경로를 yaw 보상 목표 2가지로 실행해서 비교
    fixed   = 출발 방향 유지 (기존 보상, 회전을 억제)
    segment = 현재 구간의 진행 방향 (회전을 돕는다)
    대상: 씬마다 astar_gt/trial_00 + 계획이 유효했던 C4 시행 1개 (있으면)
실험 B (반복 실행 필요성):  A* 경로 하나를 제어기 seed 0~4로 실행해서 결과가 얼마나 흔들리는지

평가: analysis/episode_metrics.evaluate_rollout + 옆걸음 지표
    crab_mean_deg / crab_p90_deg = 몸통 방향과 실제 이동 방향의 차이 (속도 0.2 m/s 이상인 step)
결과 폴더: data/<scene>/<method>/trial_xx/decide/<variant>/  (본 실험 결과와 섞이지 않음)
개발용 씬에서만 돌릴 것: 여기서 정한 설정을 평가 씬에 쓰기 때문.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.episode_metrics import evaluate_rollout, split_states  # noqa: E402
from analysis.plan_metrics import OccGrid, evaluate_path  # noqa: E402
from experiments.run_dial import path_length  # noqa: E402
from indoor.build_scene import n_steps_for  # noqa: E402

MODELS = os.path.join(REPO, "dial_mpc", "dial_mpc", "models", "unitree_go2")


def crab_angles(states: np.ndarray, dt: float = 0.02) -> np.ndarray:
    s = split_states(states)
    q = s["qpos"]
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    v = np.diff(q[:, :2], axis=0) / dt
    speed = np.linalg.norm(v, axis=1)
    mv = np.arctan2(v[:, 1], v[:, 0])
    d = np.degrees(np.abs(np.arctan2(np.sin(mv - yaw[1:]), np.cos(mv - yaw[1:]))))
    return d[speed > 0.2]


def run_variant(scene, trial_dir, variant, seed, yaw_mode):
    pj = os.path.join(trial_dir, "last_judged_path.json")
    out = os.path.join(trial_dir, "decide", variant)
    os.makedirs(out, exist_ok=True)
    if glob.glob(os.path.join(out, "*_states.npy")):
        return out, "cached"
    n_steps = n_steps_for(path_length(json.load(open(pj))))
    cmd = [sys.executable, "dial_mpc/dial_mpc/core/dial_core.py", "--example", scene, "--vlm-path-json", pj,
           "--output-dir", out, "--n-steps", str(n_steps), "--seed", str(seed), "--yaw-target-mode", yaw_mode]
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "PYTHONPATH": os.path.join(REPO, "dial_mpc") + os.pathsep + REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}
    t0 = time.perf_counter()
    with open(os.path.join(out, "dial.log"), "w") as log:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    return out, f"rc={rc} {time.perf_counter() - t0:.0f}s"


def evaluate(scene, trial_dir, out):
    st = sorted(glob.glob(os.path.join(out, "*_states.npy")))
    if not st:
        return None
    sm = json.load(open(os.path.join(REPO, "data", scene, "scene_meta.json")))
    path = json.load(open(os.path.join(trial_dir, "last_judged_path.json")))
    states = np.load(st[-1])
    ev = evaluate_rollout(states, os.path.join(MODELS, f"{scene}.xml"),
                          np.array([[p["x"], p["y"]] for p in path]), sm["goal"])
    ca = crab_angles(states)
    ev["crab_mean_deg"] = float(ca.mean()) if len(ca) else float("nan")
    ev["crab_p90_deg"] = float(np.percentile(ca, 90)) if len(ca) else float("nan")
    return ev


def valid_c4_trial(scene):
    sd = os.path.join(REPO, "data", scene)
    grid = OccGrid.load(os.path.join(sd, "occupancy.npz"))
    goal = json.load(open(os.path.join(sd, "scene_meta.json")))["goal"]
    for d in sorted(glob.glob(os.path.join(sd, "C4", "trial_*"))):
        pj = os.path.join(d, "last_judged_path.json")
        if os.path.exists(pj) and evaluate_path(json.load(open(pj)), grid, goal)["valid"]:
            return d
    return None


def fmt(ev, keys):
    if ev is None:
        return ["실행 실패"] + [""] * (len(keys) - 1)
    out = []
    for k in keys:
        v = ev[k]
        out.append(("O" if v else "X") if isinstance(v, (bool, np.bool_)) else f"{v:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    keys = ["success", "reached", "fell", "collided", "time_to_goal_s", "track_mean_m", "tilt_max_deg",
            "crab_mean_deg", "crab_p90_deg"]
    lines = ["# 제어기 설정 결정 실험 (개발용 씬)", ""]

    if not a.skip_a:
        lines += ["## A. yaw 보상 목표: fixed(출발 방향 유지) vs segment(구간 진행 방향)", "",
                  "| 씬 | 경로 | yaw 목표 | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 3)]
        for sc in a.scenes:
            trials = [os.path.join(REPO, "data", sc, "astar_gt", "trial_00")]
            c4 = valid_c4_trial(sc)
            if c4:
                trials.append(c4)
            for td in trials:
                if not os.path.exists(os.path.join(td, "last_judged_path.json")):
                    continue
                label = os.path.relpath(td, os.path.join(REPO, "data", sc))
                for mode in ("fixed", "segment"):
                    out, msg = run_variant(sc, td, f"yaw_{mode}_seed0", 0, mode)
                    print(f"[A] {sc}/{label} yaw={mode}: {msg}", flush=True)
                    lines.append(f"| {sc} | {label} | {mode} | " + " | ".join(fmt(evaluate(sc, td, out), keys)) + " |")
        lines.append("")

    if not a.skip_b:
        lines += ["## B. 같은 A* 경로를 제어기 seed만 바꿔 실행 (yaw 목표 = fixed)", "",
                  "| 씬 | seed | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
        for sc in a.scenes:
            td = os.path.join(REPO, "data", sc, "astar_gt", "trial_00")
            if not os.path.exists(os.path.join(td, "last_judged_path.json")):
                continue
            evs = []
            for seed in a.seeds:
                out, msg = run_variant(sc, td, f"yaw_fixed_seed{seed}", seed, "fixed")
                print(f"[B] {sc} seed={seed}: {msg}", flush=True)
                ev = evaluate(sc, td, out)
                evs.append(ev)
                lines.append(f"| {sc} | {seed} | " + " | ".join(fmt(ev, keys)) + " |")
            ok = [e for e in evs if e]
            if ok:
                sr = np.mean([e["success"] for e in ok])
                tt = [e["time_to_goal_s"] for e in ok if e["reached"]]
                lines.append(f"| {sc} | 요약 | 성공 {sr * 100:.0f}% | 도달 시간 {np.mean(tt):.2f} ± {np.std(tt):.2f} s"
                             if tt else f"| {sc} | 요약 | 성공 {sr * 100:.0f}% |")
        lines.append("")

    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    p = os.path.join(REPO, "results", "controller_decision.md")
    open(p, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n→ {p}")


if __name__ == "__main__":
    main()
