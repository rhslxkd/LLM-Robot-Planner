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
    parser.add_argument(
        "--prompt-level",
        type=str,
        default=None,
        choices=["rich", "medium", "minimal"],
        help="Task #5/#6 프롬프트 정보량 ablation: rich(VLN급으로 절차를 거의 알려줌) / "
             "medium(기존 SCENARIO_TEMPLATES 수준 - 장애물 존재는 알려주되 경로는 안 줌) / "
             "minimal(장애물 설명 없이 start/goal/물리제약만, 이미지만 보고 판단). "
             "--scene이 지정되면 필수.",
    )
    args = parser.parse_args()

    if args.backend == "ollama" and not args.ollama_model:
        parser.error("--ollama-model is required when --backend=ollama")
    if args.backend == "openai" and not args.openai_model:
        parser.error("--openai-model is required when --backend=openai")
    if args.num_waypoints is not None and args.num_waypoints < 2:
        parser.error("--num-waypoints must be at least 2 (need a start and an end point)")
    if args.scene is not None and args.prompt_level is None:
        parser.error("--prompt-level is required when --scene is given "
                      "(choose rich/medium/minimal for the prompt-richness ablation)")

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
        # Task #5/#6 ablation: 프롬프트 레벨(rich/medium/minimal) 간에는 num_waypoints를
        # 고정해서 "정보량"만 변수가 되게 한다 (씬별 기존 사용값 그대로 유지: D=12, E=15, A=기본10).
        NUM_WAYPOINTS = {"oracle_scene_D": 12, "oracle_scene_E": 15}
        if args.num_waypoints is not None:
            num_waypoints = args.num_waypoints
            print(f"⚙️  num_waypoints overridden via CLI: {num_waypoints}")
        else:
            num_waypoints = NUM_WAYPOINTS.get(args.scene, 10)

        # Task #5/#6: 프롬프트 정보량 3단계 ablation.
        #   rich   = 관련연구(LM-Nav 등)처럼 방향/거리를 거의 다 알려줌 (절차를 미리 풀어줌)
        #   medium = 기존에 쓰던 SCENARIO_TEMPLATES 수준 (장애물 존재/위치는 알려주되 정확한
        #            경로는 안 줌, courtroom이 스스로 판단)
        #   minimal = 장애물 설명 자체를 다 빼고 start/goal/물리제약(ROBOT_PHYSICAL_CONSTRAINTS,
        #             CoordinateAgent가 별도로 항상 주입)만 -- 이미지만 보고 전부 판단하게 함
        # {num_waypoints} 자리는 아래에서 실제 num_waypoints 값으로 .format() 처리된다 --
        # 숫자를 하드코딩해두면 "N개 내놔"(시나리오 텍스트)와 "EXACTLY M개"(Coordinate
        # 프롬프트 자체 지시)가 --num-waypoints 오버라이드 시 서로 모순되는 지시가 된다.
        SCENARIO_TEMPLATES = {
            "oracle_scene_A": {
                "rich": """
                    로봇(go2)이 전방의 작은 빨간 장애물을 피해 (5,0)까지 이동해야 해.
                    대략 1.5m 전진한 후 오른쪽(-y 방향)으로 약 0.5m 틀어서 장애물을 통과하고,
                    이후 다시 왼쪽(+y 방향)으로 중앙선(y=0)에 복귀하며 남은 거리를 전진해.
                    이 절차를 따라 {num_waypoints}개의 좌표를 제시해줘.
                    """,
                "medium": """
                    중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                    그 상황속 사진에 보이듯이, 앞에 빨간 상자 장애물이 하나 있어(크기는 작은 편).
                    이 장애물을 피해서 앞으로 5m 이동할 수 있도록 {num_waypoints}개의 좌표를 제시해줘.
                    장애물로부터 최소 안전마진 0.5m만 확보하면 충분해 - 상자 자체가 작으니
                    필요 이상으로 크게 돌아갈 필요가 전혀 없어. 최단 경로에 가깝게, 살짝만 옆으로 틀어서
                    효율적으로 피해가.
                    반드시 상자를 피해가되, 불필요한 과잉 우회는 하지 마.
                    """,
                "minimal": """
                    로봇(go2)은 (0,0)에서 시작해서 (5,0)까지 이동해야 해.
                    이미지를 보고 안전하게 도달할 수 있는 {num_waypoints}개의 좌표를 제시해줘.
                    """,
            },
            "oracle_scene_D": {
                "rich": """
                    로봇(go2)이 전방의 좁은 통로를 따라 (7,0)까지 이동해야 해.
                    통로 중앙(y=0)을 유지하며 곧장 전진하면 돼 - 별도의 방향 전환 없이
                    0m 지점부터 7m 지점까지 y=0 직선을 따라 {num_waypoints}개의 좌표를 균등하게 배치해줘.
                    """,
                "medium": """
                    중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                    그 상황속 사진에 보이듯이, 로봇의 양옆으로 붉은 벽이 길게 이어져
                    좁은 통로를 이루고 있어.
                    이 통로를 따라 앞으로 7m 이동할 수 있는 {num_waypoints}개의 좌표를 제시해줘.
                    통로의 폭이 로봇이 안전하게 통과하기에 충분한지는 스스로
                    물리적 제약 조건(유효 클리어런스, 최소 통과 폭)을 근거로 판단해서 결정해.
                    충분하다고 판단되면 중앙선을 유지하며 직진하는 경로를,
                    불충분하다고 판단되면 그 판단과 근거를 명확히 설명해.
                    """,
                "minimal": """
                    로봇(go2)은 (0,0)에서 시작해서 (7,0)까지 이동해야 해.
                    이미지를 보고 안전하게 도달할 수 있는 {num_waypoints}개의 좌표를 제시해줘.
                    """,
            },
            "oracle_scene_E": {
                "rich": """
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
                "medium": """
                    중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                    로봇은 사방이 벽으로 둘러싸인 방 안에 있고, 그 안에 기둥 형태의 장애물이 세 개 있어.
                    이 기둥들은 작은 점 장애물이 아니라 방 높이의 대부분을 막는 긴 벽이라서,
                    중심점에서 살짝 벗어나는 정도로는 부족하고 벽의 끝을 완전히 지나쳐야 해.
                    사진을 보고 각 기둥의 위치와 막고 있는 범위를 스스로 판단해서,
                    시작점(0,0)에서 목표지점(6.5,0)까지 세 기둥을 모두 안전하게 피해가는
                    {num_waypoints}개의 좌표를 제시해줘. 각 지점 사이는 부드러운 곡선으로 연결하고
                    웨이포인트 간 거리는 0.4m~1.0m를 유지해.
                    """,
                "minimal": """
                    로봇(go2)은 (0,0)에서 시작해서 (6.5,0)까지 이동해야 해.
                    이미지를 보고 안전하게 도달할 수 있는 {num_waypoints}개의 좌표를 제시해줘.
                    """,
            },
        }

        # ollama 백엔드(Qwen2.5-VL, LLaVA 등)는 영어 중심으로 학습된 모델이 대부분이라,
        # 한글 시나리오를 그대로 주면 지시 이해도가 떨어지는 문제가 실제로 관찰됨(Task #7).
        # 그래서 ollama 백엔드에는 동일한 내용의 영어 템플릿을 대신 사용한다.
        # 좌표/수치/물리제약 문구는 한글판과 1:1로 정확히 대응시켰음 (특히 Scene E의
        # y=±1.2, ±0.4 임계값은 숫자 하나도 다르지 않게 유지).
        SCENARIO_TEMPLATES_EN = {
            "oracle_scene_A": {
                "rich": """
                    The robot (go2) must move to (5,0), avoiding the small red obstacle ahead.
                    Move forward about 1.5m, then shift right (-y direction) by about 0.5m to pass the obstacle,
                    then shift back left (+y direction) to return to the centerline (y=0) and continue forward
                    for the remaining distance.
                    Following this procedure, provide {num_waypoints} coordinates.
                    """,
                "medium": """
                    The robot (go2), centered in the frame, needs to move forward.
                    As shown in the image, there is one red box obstacle ahead (relatively small in size).
                    Provide {num_waypoints} coordinates that let the robot move forward 5m while avoiding this obstacle.
                    A minimum safety margin of 0.5m from the obstacle is enough - since the box itself is small,
                    there is no need to detour more than necessary. Stay close to the shortest path, veering only
                    slightly to the side to avoid it efficiently.
                    Be sure to avoid the box, but do not make an unnecessarily large detour.
                    """,
                "minimal": """
                    The robot (go2) starts at (0,0) and must move to (5,0).
                    Looking at the image, provide {num_waypoints} coordinates that safely reach the goal.
                    """,
            },
            "oracle_scene_D": {
                "rich": """
                    The robot (go2) must move to (7,0) along the narrow corridor ahead.
                    Simply go straight forward while keeping to the corridor centerline (y=0) - with no direction
                    changes needed, evenly place {num_waypoints} coordinates along the straight line y=0 from the
                    0m point to the 7m point.
                    """,
                "medium": """
                    The robot (go2), centered in the frame, needs to move forward.
                    As shown in the image, long red walls run along both sides of the robot,
                    forming a narrow corridor.
                    Provide {num_waypoints} coordinates that let the robot move forward 7m along this corridor.
                    Decide for yourself, based on the physical constraints (effective clearance, minimum passable
                    width), whether the corridor is wide enough for the robot to pass through safely.
                    If you judge it to be wide enough, provide a straight path along the centerline;
                    if you judge it to be insufficient, clearly explain that judgment and your reasoning.
                    """,
                "minimal": """
                    The robot (go2) starts at (0,0) and must move to (7,0).
                    Looking at the image, provide {num_waypoints} coordinates that safely reach the goal.
                    """,
            },
            "oracle_scene_E": {
                "rich": """
                    The robot (go2), centered in the frame, needs to move forward.
                    The robot is in a room enclosed by walls on all sides, containing three pillar-shaped obstacles.

                    ⚠️ Very important: these pillars are not small point obstacles - they are long walls that
                    block most of the room's height.
                    It is not enough to deviate slightly from the center point; the robot must completely clear
                    the edge (end) of each wall.
                    Exactly where each pillar blocks, and the y-value the robot must reach, are as follows
                    (these numbers are precisely calculated, so follow them exactly):

                    1. First pillar (around x=1.5): blocks everything from y=-0.4 upward. When passing this pillar,
                       the robot must have y at or below -1.2 (i.e., further below -1.2, e.g. -1.3, -1.5, etc.).
                    2. Second pillar (around x=3.5): blocks everything from y=0.4 downward. When passing this
                       pillar, the robot must have y at or above +1.2 (i.e., further above 1.2, e.g. 1.3, 1.5, etc.).
                    3. Third pillar (around x=5.5): blocks everything from y=-0.4 upward. When passing this pillar,
                       the robot must have y at or below -1.2.

                    Create {num_waypoints} coordinates that are guaranteed to pass through these three points
                    (y<=-1.2 at x=1.5, y>=1.2 at x=3.5, y<=-1.2 at x=5.5). Move from the start point (0,0) to the
                    goal point (6.5, 0), never missing these three points, and passing them with exactly the
                    required y-value (or an even safer value).
                    Vaguely shifting by only about y=+-0.5 is not safe at all - the robot must
                    achieve the y-values required above (+-1.2 or beyond).
                    Connect each point with a smooth curve, and keep the distance between waypoints between
                    0.4m and 1.0m.
                    """,
                "medium": """
                    The robot (go2), centered in the frame, needs to move forward.
                    The robot is in a room enclosed by walls on all sides, containing three pillar-shaped obstacles.
                    These pillars are not small point obstacles - they are long walls that block most of the
                    room's height, so deviating only slightly from the center point is not enough; the robot must
                    completely clear the end of each wall.
                    Look at the image and judge for yourself the position and blocked range of each pillar, then
                    provide {num_waypoints} coordinates that safely avoid all three pillars while moving
                    from the start point (0,0) to the goal point (6.5,0). Connect each point with a smooth curve
                    and keep the distance between waypoints between 0.4m and 1.0m.
                    """,
                "minimal": """
                    The robot (go2) starts at (0,0) and must move to (6.5,0).
                    Looking at the image, provide {num_waypoints} coordinates that safely reach the goal.
                    """,
            },
        }

        # ollama 백엔드는 영어 템플릿, 그 외(gemini/openai)는 기존 한글 템플릿을 그대로 사용.
        # (gemini/openai는 다국어 성능이 검증돼 있어 한글 유지 -- 언어를 바꾸는 게 아니라
        # "영어 중심으로 학습된 소형 오픈소스 모델"이라는 원인에 대한 대응이라는 점을 명확히 함)
        active_templates = SCENARIO_TEMPLATES_EN if args.backend == "ollama" else SCENARIO_TEMPLATES
        prompt_lang = "English" if args.backend == "ollama" else "Korean"

        DEFAULT_SCENARIO = active_templates["oracle_scene_A"]["medium"]
        scenario_template = active_templates.get(args.scene, {}).get(args.prompt_level, DEFAULT_SCENARIO)
        scenario = scenario_template.format(num_waypoints=num_waypoints)
        print(f"🌐 Prompt language: {prompt_lang} (backend={args.backend})")

        if image_path:
            print(f"📸 Analying Image: {image_path}")
            # Image size: 1263x1080. Robot is perfectly centered.
            # New calibrated robot_pos: (631, 540)
            # New scale: 150.0 (Making 1m represent fewer pixels, thus AI plans longer jumps)
            # 씬별로 카메라 시야를 넓힌 경우, oracle_gen.py 의 SCENE_PPM 과 반드시 동일값 유지
            SCENE_SCALE = {"oracle_scene_D": 90.0, "oracle_scene_E": 90.0}
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

        # Task #5/#6 ablation: 프롬프트 레벨을 variant에 반영해서 같은 씬/백엔드라도
        # rich/medium/minimal 결과가 서로 안 덮어쓰게 한다.
        if args.prompt_level is not None:
            variant = f"{variant}_ablation_{args.prompt_level}"

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