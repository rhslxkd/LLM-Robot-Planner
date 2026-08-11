import sys
import os
import argparse

# Add the project root to sys.path to ensure imports work if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from vlm_courtroom.config import init_vertex_ai
from vlm_courtroom.court.courtroom import VLMCourt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name; writes outputs to <repo_root>/data/<scene>/ instead of the default vlm_courtroom/inputs|outputs/",
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
        "--num-waypoints",
        type=int,
        default=None,
        help="Override the number of waypoints requested from the courtroom "
             "(default: per-scene hardcoded value, e.g., 10 for most scenes, 15 for E). "
             "No upper bound enforced -- e.g. --num-waypoints 100 or 200 is allowed for "
             "the waypoint-count experiment, though very large values may reduce JSON "
             "parse success rate for weaker/local models (this is itself a metric of interest).",
    )
    args = parser.parse_args()

    if args.backend == "ollama" and not args.ollama_model:
        parser.error("--ollama-model is required when --backend=ollama")
    if args.backend == "openai" and not args.openai_model:
        parser.error("--openai-model is required when --backend=openai")
    if args.num_waypoints is not None and args.num_waypoints < 2:
        parser.error("--num-waypoints must be at least 2 (need a start and an end point)")

    try:
        # Vertex AI 초기화는 gemini 백엔드를 쓸 때만 필요함
        # (ollama/openai 실험은 서비스 계정 키가 없는 환경에서도 돌아갈 수 있어야 함)
        if args.backend == "gemini":
            init_vertex_ai()

        # Initialize Court with the selected backend
        court = VLMCourt(
            backend=args.backend,
            ollama_model=args.ollama_model,
            gemini_model=args.gemini_model,
            openai_model=args.openai_model,
        )

        # [Configuration]
        if args.scene:
            data_dir = os.path.join(project_root, "data", args.scene)
            image_path = os.path.join(data_dir, "oracle.png")
        else:
            IMAGE_DIR = "/home/user/hyeonsoo/LLM-Robot-Planner/vlm_courtroom/inputs/"
            image_filename = "brax (1).png"
            if image_filename:
                image_path = (image_filename if os.path.isabs(image_filename)
                              else os.path.join(IMAGE_DIR, image_filename))
            else:
                image_path = None
        
        # 웨이포인트 개수 계산을 시나리오 텍스트 선택보다 먼저 한다 --
        # 시나리오 텍스트 안의 "N개의 좌표" 문구도 이 값으로 채워야
        # Coordinate 프롬프트의 "EXACTLY {num_waypoints}" 지시와 모순이 안 생긴다.
        NUM_WAYPOINTS = {"oracle_scene_C": 14, "oracle_scene_E": 15}
        if args.num_waypoints is not None:
            num_waypoints = args.num_waypoints
            print(f"⚙️  num_waypoints overridden via CLI: {num_waypoints}")
        else:
            num_waypoints = NUM_WAYPOINTS.get(args.scene, 10)

        # Example Scenario Description (Used if image_path is None or as context)
        # {num_waypoints} 자리는 아래에서 실제 num_waypoints 값으로 .format() 처리된다 --
        # 숫자를 하드코딩해두면 "N개 내놔"(시나리오 텍스트)와 "EXACTLY M개"(Coordinate
        # 프롬프트 자체 지시)가 --num-waypoints 오버라이드 시 서로 모순되는 지시가 된다.
        SCENARIO_TEMPLATES = {
            "oracle_scene_A": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                그 상황속 사진에 보이듯이, 앞에 빨간 상자 장애물이 하나 있어(크기는 작은 편).
                이 장애물을 피해서 앞으로 5m 이동할 수 있도록 {num_waypoints}개의 좌표를 제시해줘.
                장애물로부터 최소 안전마진 0.5m만 확보하면 충분해 - 상자 자체가 작으니
                필요 이상으로 크게 돌아갈 필요가 전혀 없어. 최단 경로에 가깝게, 살짝만 옆으로 틀어서
                효율적으로 피해가.
                반드시 상자를 피해가되, 불필요한 과잉 우회는 하지 마.
                """,
            "oracle_scene_B": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                그 상황속 사진에 보이듯이, 앞에 빨간 상자 장애물이 두 개 있어.
                첫 번째 상자(가까운 것, 로봇 기준 위쪽에 위치)는 아래쪽으로 살짝만 틀어서 피하고,
                두 번째 상자(먼 것, 로봇 기준 아래쪽에 위치)는 위쪽으로 살짝만 틀어서 피해서,
                지그재그(S자)로 최단 경로에 가깝게 앞으로 6.5m 이동할 수 있도록 {num_waypoints}개의 좌표를 제시해줘.
                각 상자로부터 최소 안전마진 0.5m는 반드시 확보하되, 불필요하게 크게 우회하지 말고
                최대한 직선에 가까운 효율적인 경로로 움직여.
                반드시 두 상자를 모두 피해가야해.
                """,
            "oracle_scene_C": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                그 상황속 사진에 보이듯이, 앞에 상자 형태의 장애물이 세 개 있어(위-아래-위 순서로 지그재그 배치, 크기도 서로 다름).
                세 장애물을 순서대로 피하면서 지그재그(슬랄롬) 형태로 앞으로 7m 이동할 수 있도록
                {num_waypoints}개의 좌표를 제시해줘.

                가장 중요한 것은 각 장애물을 정확한 위치에서 안전마진만큼 피해가는 것이야 - 이걸 절대 희생하지 마.
                그 다음으로, waypoint 사이의 급격한 방향전환(90도에 가까운 꺾임)만 피해서
                각 지점을 좀 더 촘촘하고 매끄럽게 연결해줘.
                안전 회피가 우선이고 부드러움은 그 다음이야 - 부드러움을 위해 장애물과의 거리를 희생하면 안 돼.
                각 상자로부터 최소 안전마진을 확보하되 불필요하게 크게 우회하지는 마.
                반드시 세 상자를 모두 정확히 피해가야해.
                """,
            "oracle_scene_D": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                이 통로를 통과하지 않고 우회해서 앞으로 7m 이동할 수 있는 {num_waypoints}개의 좌표를 제시해줘.
                통로를 통과하지않고 장애물을 회피, 우회해서 가도록 waypoints를 설계해줘.
                """,
            "oracle_scene_E": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                로봇은 사방이 벽으로 둘러싸인 방 안에 있고, 그 안에 기둥 형태의 장애물이 세 개 있어.

                ⚠️ 매우 중요: 이 기둥들은 작은 점 장애물이 아니라, 방 높이의 대부분을 막는 긴 벽이야.
                중심점에서 조금만 벗어나면 되는 게 아니라, 벽의 가장자리(끝)를 완전히 지나쳐야 해.
                각 기둥이 정확히 어디를 막고 있는지, 그리고 로봇이 반드시 도달해야 하는 y값은 다음과 같아
                (이 숫자들은 정확히 계산된 값이니 그대로 따라야 해):

                1. 첫 번째 기둥(x=1.5 부근): y=-0.4부터 위쪽 전부를 막고 있어. 로봇은 이 기둥을 지날 때
                   반드시 y가 -1.2 이하가 되어야 해 (즉 -1.2보다 더 아래로, 예: -1.3, -1.5 등).
                2. 두 번째 기둥(x=3.5 부근): y=0.4부터 아래쪽 전부를 막고 있어. 로봇은 이 기둥을 지날 때
                   반드시 y가 +1.2 이상이 되어야 해 (즉 1.2보다 더 위로, 예: 1.3, 1.5 등).
                3. 세 번째 기둥(x=5.5 부근): y=-0.4부터 위쪽 전부를 막고 있어. 로봇은 이 기둥을 지날 때
                   반드시 y가 -1.2 이하가 되어야 해.

                이 세 지점(x≈1.5일 때 y≤-1.2, x≈3.5일 때 y≥1.2, x≈5.5일 때 y≤-1.2)을
                반드시 통과하도록 {num_waypoints}개의 좌표를 만들어줘. 시작점(0,0)에서 목표지점(6.5, 0)까지
                이동하되, 위 세 지점을 절대 놓치지 말고 정확한 y값(또는 그보다 더 안전한 값)으로
                통과해야 해. 애매하게 y=±0.5 정도로만 살짝 틀면 절대 안전하지 않아 - 반드시
                위에서 요구한 y값(±1.2 이상)을 달성해야 해.
                각 지점 사이는 부드러운 곡선으로 이어지게 하고, 웨이포인트 간 거리는 0.4m~1.0m를 유지해.
                """,
            
        }

        DEFAULT_SCENARIO = SCENARIO_TEMPLATES["oracle_scene_A"]
        scenario_template = SCENARIO_TEMPLATES.get(args.scene, DEFAULT_SCENARIO)
        scenario = scenario_template.format(num_waypoints=num_waypoints)
        
        if image_path:
            print(f"📸 Analying Image: {image_path}")
            # Image size: 1263x1080. Robot is perfectly centered.
            # New calibrated robot_pos: (631, 540)
            # New scale: 150.0 (Making 1m represent fewer pixels, thus AI plans longer jumps)
            # 씬별로 카메라 시야를 넓힌 경우, oracle_gen.py 의 SCENE_PPM 과 반드시 동일값 유지
            SCENE_SCALE = {"oracle_scene_C": 90.0, "oracle_scene_E": 90.0, "oracle_scene_D": 90.0}
            robot_pos = (421, 540)  # oracle_gen.py 의 ROBOT_PX(IMG_W/3, IMG_H/2) 와 반드시 동일값 유지
            scale = SCENE_SCALE.get(args.scene, 150.0)
        else:
            print(f"Scenario Description: {scenario}")
            robot_pos = None
            scale = None
        
        # variant: 백엔드/모델 식별자. scene_name(원래 씬 이름, 입력 이미지 위치)은 그대로 두고,
        # 출력만 data/<scene>/<variant>/ 하위 폴더로 분리한다.
        #   Gemini는 역할별로 다른 모델(judge=2.5-pro, 나머지=2.5-flash)을 쓰므로
        #   variant에는 개별 모델명 대신 "gemini"로만 표시한다.
        #   --gemini-model로 단일 모델을 지정한 경우(공정 비교용)엔 그 모델명을 그대로 반영해서
        #   기존 "gemini"(혼합) variant랑 겹치지 않게 한다.
        if args.backend == "gemini":
            if args.gemini_model:
                safe_gemini_tag = args.gemini_model.replace(":", "_").replace(".", "_").replace("-", "_")
                variant = f"gemini_{safe_gemini_tag}"
            else:
                variant = "gemini"
        elif args.backend == "ollama":
            safe_model_tag = args.ollama_model.replace(":", "_").replace(".", "_")
            variant = f"ollama_{safe_model_tag}"
        elif args.backend == "openai":
            safe_openai_tag = args.openai_model.replace(":", "_").replace(".", "_").replace("-", "_")
            variant = f"openai_{safe_openai_tag}"

        # 웨이포인트 개수를 CLI로 오버라이드한 경우, variant에 접미사를 붙여서
        # 같은 씬/백엔드로 개수만 다르게 돌린 실험 결과들이 서로 안 덮어쓰게 한다
        # (기본값 그대로 쓴 경우엔 접미사 없음 -> 기존 결과 위치와 호환 유지).
        if args.num_waypoints is not None:
            variant = f"{variant}_wp{num_waypoints}"

        if args.scene:
            print(f"🗂️  Output directory: data/{args.scene}/{variant}/")

        court.run_case(scenario, image_path=image_path, robot_pos=robot_pos, scale=scale,
                       scene_name=args.scene, num_waypoints=num_waypoints, variant=variant)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()