"""
random_batch_manifest_v3.csv에서 generator_passed=True인데 dial_mpc_ok=False인 씬들만
DIAL-MPC 재시도. JAX GPU 메모리 preallocate를 끄고 필요한 만큼만 쓰게 해서
labmate 프로세스와의 OOM 경합을 줄임. 결과는 별도 파일에 기록.
"""
import os, sys, csv, glob, subprocess, time

REPO = "/home/user/hyeonsoo/LLM-Robot-Planner"
os.chdir(REPO)
MANIFEST = "data/random_batch_manifest_v3.csv"
RETRY_LOG = "data/random_batch_manifest_v3_retry.csv"
N_STEPS = 800

rows = list(csv.DictReader(open(MANIFEST)))
targets = [r["scene"] for r in rows if r["generator_passed"] == "True" and r["dial_mpc_ok"] == "False"]
print(f"재시도 대상: {len(targets)}개")

env = {**os.environ,
       "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3"}

results = []
with open(RETRY_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scene", "dial_mpc_ok", "info", "elapsed_s"])
    for i, scene in enumerate(targets):
        t0 = time.time()
        out_dir = f"data/{scene}"
        timeout = int(N_STEPS * 0.5) + 120
        cmd = ["timeout", "-s", "SIGINT", "--kill-after=15s", f"{timeout}s",
               sys.executable, "dial_mpc/dial_mpc/core/dial_core.py",
               "--example", scene, "--vlm-path-json", f"{out_dir}/last_judged_path.json",
               "--output-dir", out_dir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30, env=env)
        except subprocess.TimeoutExpired:
            writer.writerow([scene, False, "hard timeout", time.time() - t0]); f.flush()
            print(f"[{i+1}/{len(targets)}] {scene}: timeout")
            continue
        produced = glob.glob(f"{out_dir}/*_states.npy")
        ok = len(produced) > 0
        info = "ok" if ok else proc.stderr[-300:]
        writer.writerow([scene, ok, info.replace("\n", " "), time.time() - t0]); f.flush()
        print(f"[{i+1}/{len(targets)}] {scene}: {'성공' if ok else '실패'} ({time.time()-t0:.0f}s)")

print(f"\n완료: {RETRY_LOG} 확인")
