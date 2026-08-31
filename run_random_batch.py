"""
run_random_batch.py  (vlm_court env에서 실행)
파일럿: python run_random_batch.py --n-scenes 8 --start-seed 0
DIAL-MPC까지 돌리려면: python run_random_batch.py --n-scenes 8 --start-seed 0 --run-dial
"""
import os, sys, csv, time, subprocess, argparse, shutil, glob, json

REPO = "/home/user/hyeonsoo/LLM-Robot-Planner"
sys.path.insert(0, REPO)
os.chdir(REPO)

from vlm_courtroom.court.courtroom import VLMCourt
from generate_random_baffle_maze import generate as generate_maze

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
NEURAL_ASTAR_ENV = "neural-astar"
VARIANT = "batch"  # courtroom 산출물이 저장되는 data/<scene>/<VARIANT>/ 하위 폴더명

YAML_TEMPLATE = """# DIAL-MPC (auto-generated)
seed: 0
output_dir: data/{scene}
n_steps: {n_steps}
env_name: unitree_go2_walk
Nsample: 2048
Hsample: 16
Hnode: 4
Ndiffuse: 2
Ndiffuse_init: 10
temp_sample: 0.05
horizon_diffuse_factor: 0.9
traj_diffuse_factor: 0.5
update_method: mppi
dt: 0.02
timestep: 0.02
leg_control: torque
action_scale: 1.0
default_vx: 0.8
default_vy: 0.0
default_vyaw: 0.0
ramp_up_time: 1.0
gait: trot
scene_xml: {scene}.xml
vlm_path_json: data/{scene}/last_judged_path.json
"""


def run_neural_astar_step(scene, goal_x, goal_y, timeout=120):
    """neural-astar env를 서브프로세스로 호출. 성공 시 (overlay_path, coordinate_proposal)
    튜플 리턴, 실패 시 (None, None). 전체 stdout/stderr는 항상 로그 파일로 저장(터미널
    출력은 잘려도 파일은 안 잘림)."""
    cmd = ["conda", "run", "-n", NEURAL_ASTAR_ENV, "--no-capture-output",
           "python", "run_neural_astar_step.py",
           "--scene", scene, "--goal-x", str(goal_x), "--goal-y", str(goal_y)]
    try:
        proc = subprocess.run(cmd, cwd=REPO, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print("  ⚠️ neural-astar 서브프로세스 timeout")
        return None, None

    log_dir = f"data/{scene}/neural_astar"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/na_stdout.log", "w") as f:
        f.write(proc.stdout)
    with open(f"{log_dir}/na_stderr.log", "w") as f:
        f.write(proc.stderr)

    overlay_path = f"{log_dir}/overlay_solo.png"
    proposal_path = f"{log_dir}/coordinate_proposal.json"
    if proc.returncode != 0 or not os.path.exists(overlay_path) or not os.path.exists(proposal_path):
        print(f"  ⚠️ neural-astar 실패 (returncode={proc.returncode}) -- 전체 로그: {log_dir}/na_stdout.log")
        return None, None

    with open(proposal_path) as f:
        coordinate_proposal = json.load(f)
    return overlay_path, coordinate_proposal


def run_dial_mpc(scene, timeout=300):
    out_dir = f"data/{scene}"
    cmd = ["timeout", "-s", "SIGINT", "--kill-after=15s", f"{timeout}s",
           sys.executable, "dial_mpc/dial_mpc/core/dial_core.py",
           "--example", scene, "--vlm-path-json", f"{out_dir}/last_judged_path.json",
           "--output-dir", out_dir]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+30)
    except subprocess.TimeoutExpired:
        return False, "hard timeout"
    produced = glob.glob(f"{out_dir}/*_states.npy")
    return len(produced) > 0, (proc.stderr[-500:] if proc.returncode != 0 else "ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenes", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--run-dial", action="store_true",
                         help="지정하면 courtroom 통과 후 DIAL-MPC까지 실행. 기본은 미실행"
                              " (courtroom까지만 파일럿 검증).")
    parser.add_argument("--n-steps", type=int, default=400,
                         help="생성되는 YAML의 n_steps(DIAL-MPC 시뮬레이션 스텝 수). 기본 400.")
    args = parser.parse_args()

    manifest_path = "data/random_batch_manifest.csv"
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                done.add(row["scene"])

    is_new = not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0
    with open(manifest_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if is_new:
            writer.writerow(["scene", "goal_x", "goal_y", "n_baffles", "neural_astar_ok",
                              "courtroom_verdict", "verified_clear", "dial_mpc_ok", "elapsed_s"])

        for i in range(args.n_scenes):
            seed = args.start_seed + i
            scene = f"oracle_scene_R{seed:03d}"
            if scene in done:
                print(f"[{scene}] 이미 완료, 스킵"); continue

            t0 = time.time()
            print(f"\n=== [{scene}] 시작 ===")

            xml, meta = generate_maze(seed, scene)
            goal_x, goal_y = meta["goal_x"], meta["goal_y"]
            with open(f"dial_mpc/dial_mpc/models/unitree_go2/{scene}.xml", "w") as f:
                f.write(xml)
            with open(f"dial_mpc/dial_mpc/examples/{scene}.yaml", "w") as f:
                f.write(YAML_TEMPLATE.format(scene=scene, n_steps=args.n_steps))
            print(f"  미로 생성 완료 goal=({goal_x:.2f},{goal_y:.2f}) n_baffles={meta['n_baffles']}")

            env = {**os.environ, "MUJOCO_GL": "egl"}
            render_proc = subprocess.run(["python", "oracle_gen.py", f"{scene}.xml"],
                                          cwd=REPO, capture_output=True, text=True, timeout=60, env=env)
            if not os.path.exists(f"data/{scene}/oracle.png"):
                print(f"  ⚠️ 렌더링 실패: {render_proc.stderr[-300:]}")
                writer.writerow([scene, goal_x, goal_y, meta["n_baffles"], "render_fail", "", "", "", time.time()-t0]); mf.flush()
                continue

            overlay_path, coordinate_proposal = run_neural_astar_step(scene, goal_x, goal_y)
            if overlay_path is None:
                writer.writerow([scene, goal_x, goal_y, meta["n_baffles"], False, "", "", "", time.time()-t0]); mf.flush()
                continue
            print(f"  Neural A* 완료 ({len(coordinate_proposal)} waypoints)")

            court = VLMCourt(backend="gemini", gemini_model="gemini-2.5-flash")
            scenario = (f"로봇(go2)은 (0,0)에서 시작해서 ({goal_x:.1f},{goal_y:.1f})까지 이동해야 해. "
                        f"이미 계산된 경로를 검토해줘.")
            judge_msg, coords = court.run_case(
                scenario, image_path=f"data/{scene}/oracle.png", robot_pos=ROBOT_PX, scale=PPM,
                scene_name=scene, coordinate_proposal=coordinate_proposal, variant=VARIANT
            )
            if judge_msg is None or judge_msg.content.strip().startswith("Error generating response"):
                verdict, verified_clear = "ERROR", False
            else:
                is_final_reject = "[SYSTEM]" in judge_msg.content and "REJECTED" in judge_msg.content
                verdict = "REJECTED" if is_final_reject else "ACCEPTED"
                verified_clear = (verdict == "ACCEPTED")
            print(f"  Courtroom: {verdict} (verified_clear={verified_clear})")

            if verdict in ("REJECTED", "ERROR") or not coords:
                writer.writerow([scene, goal_x, goal_y, meta["n_baffles"], True, verdict, verified_clear, "", time.time()-t0]); mf.flush()
                continue

            src = f"data/{scene}/{VARIANT}/last_judged_path.json"
            dst = f"data/{scene}/last_judged_path.json"
            if os.path.exists(src):
                shutil.copy(src, dst)

            if not args.run_dial:
                print("  DIAL-MPC: 스킵 (--run-dial 없이 실행됨 -- courtroom까지만 파일럿)")
                writer.writerow([scene, goal_x, goal_y, meta["n_baffles"], True, verdict, verified_clear, "skipped", time.time()-t0]); mf.flush()
                continue

            ok, info = run_dial_mpc(scene)
            print(f"  DIAL-MPC: {'성공' if ok else '실패: ' + str(info)}")
            writer.writerow([scene, goal_x, goal_y, meta["n_baffles"], True, verdict, verified_clear, ok, time.time()-t0]); mf.flush()

    print(f"\n완료. manifest: {manifest_path}")


if __name__ == "__main__":
    main()