"""
run_random_batch_v2.py  (vlm_court env에서 실행)
2026-09-02: VLMCourt(Gemini VLM 판사) 제거, waypoint_generator.py(distance-field 기반
결정론적 보정)로 교체. 씬 레이아웃도 매번 슬라롬/박스 중 랜덤 선택(다양성 확대,
Neural A* 재학습용 데이터셋 목적). scene 접두사는 기존 R000~R015(수동 검증된 세트,
특히 R004)와 절대 안 겹치게 B로 분리. manifest도 컬럼 구조가 달라져서 v2 파일로 분리.

파일럿: python run_random_batch_v2.py --n-scenes 5 --start-seed 0
DIAL-MPC까지 돌리려면: python run_random_batch_v2.py --n-scenes 5 --start-seed 0 --run-dial
"""
import os, sys, csv, time, subprocess, argparse, shutil, glob, json, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (파일 위치 기준 자동 계산, 컴퓨터마다 안전)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "core"))
os.chdir(REPO)

from waypoint_generator import generate_waypoints
from generate_random_baffle_maze import generate as generate_slalom, generate_boxes

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
NEURAL_ASTAR_ENV = "neural-astar"
VARIANT = "waypoint_gen_v1"

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
    cmd = ["conda", "run", "-n", NEURAL_ASTAR_ENV, "--no-capture-output",
           "python", "core/run_neural_astar_step.py",
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


def save_verdict_png(scene, image_path, out_dir, final_coords, passed):
    rx, ry = ROBOT_PX
    xs = [rx + c["x"] * PPM for c in final_coords]
    ys = [ry - c["y"] * PPM for c in final_coords]
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.plot(xs, ys, "r-", linewidth=2)
    ax.scatter(xs, ys, c="yellow", s=50, zorder=5)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(i), (x, y), color="white", fontsize=11, fontweight="bold")
    ax.plot(rx, ry, "bo", markersize=10)
    ax.set_title(f"{scene} -- PASSED: {passed}")
    plt.savefig(os.path.join(out_dir, "verdict.png"))
    plt.close()


def run_dial_mpc(scene, n_steps, timeout=None):
    if timeout is None:
        timeout = int(n_steps * 0.5) + 120
    out_dir = f"data/{scene}"
    cmd = ["timeout", "-s", "SIGINT", "--kill-after=15s", f"{timeout}s",
           sys.executable, "dial_mpc/dial_mpc/core/dial_core.py",
           "--example", scene, "--vlm-path-json", f"{out_dir}/last_judged_path.json",
           "--output-dir", out_dir]
    # GPU 공유 환경에서 JAX가 메모리를 욕심껏 미리 할당(75~90%)하면서 다른 프로세스와
    # 경합 시 CUDA_ERROR_OUT_OF_MEMORY로 즉시 죽는 문제 방지 (2026-09 200개 배치에서
    # 실측 확인: 이 옵션으로 74/74 재시도 성공).
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30, env=env)
    except subprocess.TimeoutExpired:
        return False, "hard timeout"
    produced = glob.glob(f"{out_dir}/*_states.npy")
    return len(produced) > 0, (proc.stderr[-500:] if proc.returncode != 0 else "ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenes", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--box-prob", type=float, default=0.5)
    parser.add_argument("--run-dial", action="store_true")
    parser.add_argument("--prefix", type=str, default="B")
    parser.add_argument("--manifest", type=str, default="data/random_batch_manifest_v2.csv")
    parser.add_argument("--purpose", type=str, default="unlabeled",
                         help="이 배치의 용도 태그 (예: finetune_data, diversity_pilot, smoke_test)")
    args = parser.parse_args()

    date_str = time.strftime("%Y%m%d")
    run_dir = f"data/runs/{date_str}_{args.purpose}_{args.prefix}"
    os.makedirs(run_dir, exist_ok=True)
    run_meta = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": args.purpose,
        "prefix": args.prefix,
        "start_seed": args.start_seed,
        "n_scenes": args.n_scenes,
        "n_steps": args.n_steps,
        "box_prob": args.box_prob,
        "run_dial": args.run_dial,
        "manifest": args.manifest,
    }
    with open(f"{run_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    scenes_txt_path = f"{run_dir}/scenes.txt"
    print(f"런 인덱스: {run_dir}/ (run_meta.json, scenes.txt)")

    manifest_path = args.manifest
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                done.add(row["scene"])

    is_new = not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0
    with open(manifest_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if is_new:
            writer.writerow(["scene", "layout", "goal_x", "goal_y", "n_obstacles",
                              "neural_astar_ok", "generator_passed", "n_waypoints",
                              "dial_mpc_ok", "elapsed_s"])

        for i in range(args.n_scenes):
            seed = args.start_seed + i
            scene = f"oracle_scene_{args.prefix}{seed:04d}"
            if scene in done:
                print(f"[{scene}] 이미 완료, 스킵"); continue

            t0 = time.time()
            print(f"\n=== [{scene}] 시작 ===")
            with open(scenes_txt_path, "a") as sf:
                sf.write(scene + "\n")

            layout_rng = random.Random(seed * 7919 + 13)
            use_boxes = layout_rng.random() < args.box_prob
            if use_boxes:
                room_half_y = layout_rng.uniform(2.0, 4.0)
                goal_x_hi = 4.4
            else:
                room_half_y = layout_rng.uniform(1.4, 2.4)
                goal_x_hi = 4.6
            goal_y_margin = room_half_y * 0.3
            goal_y_bound = room_half_y - goal_y_margin
            goal_x_range = (1.0, goal_x_hi)
            max_dist = (goal_x_hi ** 2 + goal_y_bound ** 2) ** 0.5
            min_dist_from_start = layout_rng.uniform(0.3, 0.85) * max_dist
            diversity_kwargs = dict(
                goal_x_range=goal_x_range,
                room_half_y=room_half_y,
                goal_y_margin=goal_y_margin,
                min_dist_from_start=min_dist_from_start,
            )
            if use_boxes:
                xml, meta = generate_boxes(seed, scene, **diversity_kwargs)
            else:
                xml, meta = generate_slalom(seed, scene, **diversity_kwargs)
            layout = meta.get("layout", "slalom")
            n_obstacles = meta.get("n_boxes", meta.get("n_baffles"))
            goal_x, goal_y = meta["goal_x"], meta["goal_y"]

            with open(f"dial_mpc/dial_mpc/models/unitree_go2/{scene}.xml", "w") as f:
                f.write(xml)
            with open(f"dial_mpc/dial_mpc/examples/{scene}.yaml", "w") as f:
                f.write(YAML_TEMPLATE.format(scene=scene, n_steps=args.n_steps))
            print(f"  씬 생성 완료 layout={layout} goal=({goal_x:.2f},{goal_y:.2f}) n_obstacles={n_obstacles}")

            env = {**os.environ, "MUJOCO_GL": "egl"}
            render_proc = subprocess.run(["python", "core/oracle_gen.py", f"{scene}.xml"],
                                          cwd=REPO, capture_output=True, text=True, timeout=60, env=env)
            if not os.path.exists(f"data/{scene}/oracle.png"):
                print(f"  ⚠️ 렌더링 실패: {render_proc.stderr[-300:]}")
                writer.writerow([scene, layout, goal_x, goal_y, n_obstacles, "render_fail", "", "", "", time.time()-t0]); mf.flush()
                continue

            overlay_path, coordinate_proposal = run_neural_astar_step(scene, goal_x, goal_y)
            if overlay_path is None:
                writer.writerow([scene, layout, goal_x, goal_y, n_obstacles, False, "", "", "", time.time()-t0]); mf.flush()
                continue
            print(f"  Neural A* 완료 ({len(coordinate_proposal)} waypoints)")

            image_path = f"data/{scene}/oracle.png"
            final_coords, passed, gen_log = generate_waypoints(image_path, coordinate_proposal, ROBOT_PX, PPM)
            print(f"  waypoint_generator: passed={passed} ({len(final_coords)} waypoints)")

            out_dir = f"data/{scene}/{VARIANT}"
            os.makedirs(out_dir, exist_ok=True)
            dial_path = [{"x": c["x"], "y": c["y"]} for c in final_coords]
            with open(f"{out_dir}/last_judged_path.json", "w") as f:
                json.dump(dial_path, f, indent=2)
            with open(f"{out_dir}/log.txt", "w") as f:
                f.write(f"PASSED: {passed}\n\n")
                f.write("\n".join(gen_log))
            save_verdict_png(scene, image_path, out_dir, final_coords, passed)

            if not passed:
                writer.writerow([scene, layout, goal_x, goal_y, n_obstacles, True, False, len(final_coords), "", time.time()-t0]); mf.flush()
                continue

            dst = f"data/{scene}/last_judged_path.json"
            shutil.copy(f"{out_dir}/last_judged_path.json", dst)

            if not args.run_dial:
                print("  DIAL-MPC: 스킵 (--run-dial 없이 실행됨 -- 경로 생성까지만 파일럿)")
                writer.writerow([scene, layout, goal_x, goal_y, n_obstacles, True, True, len(final_coords), "skipped", time.time()-t0]); mf.flush()
                continue

            ok, info = run_dial_mpc(scene, args.n_steps)
            print(f"  DIAL-MPC: {'성공' if ok else '실패: ' + str(info)}")
            writer.writerow([scene, layout, goal_x, goal_y, n_obstacles, True, True, len(final_coords), ok, time.time()-t0]); mf.flush()

    print(f"\n완료. manifest: {manifest_path}")


if __name__ == "__main__":
    main()
