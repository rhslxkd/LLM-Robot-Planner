import sys
import os
import json
import argparse

# Add the project root to sys.path to ensure imports work if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from vlm_courtroom.config import init_vertex_ai
from vlm_courtroom.court.courtroom import VLMCourt


def main():
    parser = argparse.ArgumentParser(
        description="단일 씬 딥다이브용 courtroom 실행기. Neural A*가 미리 계산한 "
                    "coordinate_proposal.json을 courtroom(4-agent: Coordinate/Prosecutor/"
                    "Judge/Verifier)에 넣고 프롬프트를 반복 튜닝하기 위한 스크립트. "
                    "(2026-08-28: num_waypoints/SCENARIO_TEMPLATES 기반 구식 API 제거 -- "
                    "지금 courtroom.py는 coordinate_proposal만 받는다.)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="씬 이름 (예: oracle_scene_R004). data/<scene>/oracle.png와 "
             "data/<scene>/neural_astar/coordinate_proposal.json이 있어야 함.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        choices=["gemini", "ollama", "openai"],
        help="VLM backend to use for all four courtroom agents (default: gemini).",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model tag to use when --backend=ollama "
             "(e.g., qwen2.5vl:7b, llava-llama3, qwen3-vl, minicpm-v). Required if --backend=ollama.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=None,
        help="Gemini model to use uniformly for all four agents when --backend=gemini "
             "(e.g., gemini-2.5-flash, gemini-2.5-pro). If omitted, falls back to the "
             "role-based mix (Judge=pro, others=flash) -- NOT a fair single-model comparison.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=None,
        help="OpenAI model tag to use when --backend=openai (e.g., gpt-4o, gpt-4o-mini). "
             "Required if --backend=openai.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="이미지 경로를 강제로 지정하고 싶을 때만 사용. 지정 안 하면 "
             "data/<scene>/oracle.png (경로가 그려지지 않은 원본 씬 이미지)를 쓴다. "
             "CRITICAL: overlay_solo.png(Neural A* 경로가 그려진 이미지)는 넘기지 말 것 -- "
             "그 경로 선이 결정론적 벽-충돌 검사(_build_red_mask)에 벽으로 오탐될 수 있고, "
             "VLM이 원래 경로 모양에 시각적으로 낚일 수 있다.",
    )
    args = parser.parse_args()

    if args.backend == "ollama" and not args.ollama_model:
        parser.error("--ollama-model is required when --backend=ollama")
    if args.backend == "openai" and not args.openai_model:
        parser.error("--openai-model is required when --backend=openai")

    try:
        if args.backend == "gemini":
            init_vertex_ai()

        court = VLMCourt(
            backend=args.backend,
            ollama_model=args.ollama_model,
            gemini_model=args.gemini_model,
            openai_model=args.openai_model,
        )

        data_dir = os.path.join(project_root, "data", args.scene)
        image_path = args.image_path if args.image_path else os.path.join(data_dir, "oracle.png")

        coord_json_path = os.path.join(data_dir, "neural_astar", "coordinate_proposal.json")
        if not os.path.exists(coord_json_path):
            raise FileNotFoundError(
                f"{coord_json_path} 없음 -- 먼저 이 씬에 대해 Neural A*를 돌려서 "
                f"coordinate_proposal.json을 생성해야 함."
            )
        with open(coord_json_path, "r") as f:
            coordinate_proposal = json.load(f)

        goal = coordinate_proposal[-1]
        image_description = (
            f"로봇(go2)이 (0,0)에서 시작해 목표 지점 근처 ({goal['x']}, {goal['y']})까지 "
            f"이동하는 미로 씬. 장애물은 이미지의 붉은 벽/기둥으로 표시됨."
        )

        print(f"📸 Scene: {args.scene}")
        print(f"📸 Image: {image_path}")
        print(f"📍 Coordinate proposal: {len(coordinate_proposal)} waypoints from {coord_json_path}")

        # 씬별로 카메라 시야를 넓힌 경우, oracle_gen.py 의 SCENE_PPM 과 반드시 동일값 유지
        SCENE_SCALE = {"oracle_scene_D": 90.0, "oracle_scene_E": 90.0}
        robot_pos = (421, 540)  # oracle_gen.py 의 ROBOT_PX(IMG_W/3, IMG_H/2) 와 반드시 동일값 유지
        scale = SCENE_SCALE.get(args.scene, 150.0)

        # variant: 백엔드/모델 식별자
        if args.backend == "gemini":
            if args.gemini_model:
                safe_tag = args.gemini_model.replace(":", "_").replace(".", "_").replace("-", "_")
                variant = f"gemini_{safe_tag}"
            else:
                variant = "gemini"
        elif args.backend == "ollama":
            safe_tag = args.ollama_model.replace(":", "_").replace(".", "_")
            variant = f"ollama_{safe_tag}"
        elif args.backend == "openai":
            safe_tag = args.openai_model.replace(":", "_").replace(".", "_").replace("-", "_")
            variant = f"openai_{safe_tag}"

        print(f"🗂️  Output directory: data/{args.scene}/{variant}/")

        court.run_case(
            image_description,
            image_path=image_path,
            robot_pos=robot_pos,
            scale=scale,
            scene_name=args.scene,
            coordinate_proposal=coordinate_proposal,
            variant=variant,
        )

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
