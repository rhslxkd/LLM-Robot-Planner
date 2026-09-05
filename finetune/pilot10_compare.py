"""
학습에 전혀 안 쓰인 새 씬 10개를 생성해서, 우리 production 파이프라인 그대로
(Neural A* -> waypoint_generator 보정) pretrained/fine-tuned 체크포인트 각각으로
raw proposal을 뽑고, GT(=pretrained proposal을 보정한 안전경로)와 3-way 비교.
vlm_court env에서 실행 (run_random_batch_v2.py와 동일 환경).
"""
import os, sys, subprocess, json, random, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (파일 위치 기준 자동 계산, 컴퓨터마다 안전)
sys.path.insert(0, REPO)
os.chdir(REPO)
from waypoint_generator import generate_waypoints
from generate_random_baffle_maze import generate as generate_slalom, generate_boxes

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
NEURAL_ASTAR_ENV = "neural-astar"
FINETUNED_CKPT_DIR = os.path.join(REPO, "finetune/checkpoints")
OUT_DIR = os.path.join(REPO, "finetune/verify")
N_SCENES = 10
SEED_BASE = 90000  # 기존 배치(0~299)와 절대 안 겹치는 범위


def run_neural_astar_step(scene, goal_x, goal_y, ckpt_override=None, timeout=120):
    env = {**os.environ}
    if ckpt_override:
        env["NEURAL_ASTAR_CKPT_OVERRIDE"] = ckpt_override
    cmd = ["conda", "run", "-n", NEURAL_ASTAR_ENV, "--no-capture-output",
           "python", "core/run_neural_astar_step.py",
           "--scene", scene, "--goal-x", str(goal_x), "--goal-y", str(goal_y)]
    try:
        proc = subprocess.run(cmd, cwd=REPO, timeout=timeout, capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        print("  ⚠️ timeout")
        return None
    proposal_path = f"data/{scene}/neural_astar/coordinate_proposal.json"
    if proc.returncode != 0 or not os.path.exists(proposal_path):
        print(f"  ⚠️ 실패 (returncode={proc.returncode}): {proc.stdout[-300:]}")
        return None
    with open(proposal_path) as f:
        return json.load(f)


def to_px(coords):
    rx, ry = ROBOT_PX
    xs = [rx + c["x"] * PPM for c in coords]
    ys = [ry - c["y"] * PPM for c in coords]
    return xs, ys


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for i in range(N_SCENES):
        seed = SEED_BASE + i
        scene = f"oracle_scene_PILOT{i:02d}"
        print(f"\n=== [{scene}] (seed={seed}) ===")
        layout_rng = random.Random(seed * 7919 + 13)
        use_boxes = layout_rng.random() < 0.5
        xml, meta = (generate_boxes(seed, scene) if use_boxes else generate_slalom(seed, scene))
        goal_x, goal_y = meta["goal_x"], meta["goal_y"]
        with open(f"dial_mpc/dial_mpc/models/unitree_go2/{scene}.xml", "w") as f:
            f.write(xml)

        env = {**os.environ, "MUJOCO_GL": "egl"}
        render_proc = subprocess.run(["python", "core/oracle_gen.py", f"{scene}.xml"],
                                      cwd=REPO, capture_output=True, text=True, timeout=60, env=env)
        image_path = f"data/{scene}/oracle.png"
        if not os.path.exists(image_path):
            print(f"  ⚠️ 렌더링 실패: {render_proc.stderr[-300:]}")
            continue

        pretrained_proposal = run_neural_astar_step(scene, goal_x, goal_y)
        if pretrained_proposal is None:
            print("  ⚠️ pretrained 추론 실패, 스킵"); continue

        finetuned_proposal = run_neural_astar_step(scene, goal_x, goal_y, ckpt_override=FINETUNED_CKPT_DIR)
        if finetuned_proposal is None:
            print("  ⚠️ fine-tuned 추론 실패, 스킵"); continue

        final_coords, passed, _ = generate_waypoints(image_path, pretrained_proposal, ROBOT_PX, PPM)
        print(f"  GT(waypoint_generator) passed={passed}")

        gt_xs, gt_ys = to_px(final_coords)
        pre_xs, pre_ys = to_px(pretrained_proposal)
        ft_xs, ft_ys = to_px(finetuned_proposal)

        img = mpimg.imread(image_path)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.plot(gt_xs, gt_ys, color="lime", linewidth=3, linestyle="--", label="GT (safety-corrected)", zorder=4)
        ax.plot(pre_xs, pre_ys, color="blue", linewidth=2, label="pretrained (before)", zorder=5)
        ax.plot(ft_xs, ft_ys, color="red", linewidth=2, label="fine-tuned (after)", zorder=6)
        ax.legend(loc="upper right")
        ax.set_title(f"{scene}  passed={passed}")
        out_path = os.path.join(OUT_DIR, f"{scene}_pilot.png")
        plt.savefig(out_path)
        plt.close()
        print(f"  저장: {out_path}")
        results.append(scene)

    print(f"\n완료: {len(results)}/{N_SCENES}개 -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
