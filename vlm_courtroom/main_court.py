
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
    args = parser.parse_args()

    try:
        # Initialize Vertex AI connection
        init_vertex_ai()

        # Initialize Court with Database Reset (clears previous tests)
        court = VLMCourt(reset_db=True)

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
        
        # Example Scenario Description (Used if image_path is None or as context)
        SCENARIO_TEMPLATES = {
            "oracle_scene_A": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                그 상황속 사진에 보이듯이, 앞에 빨간 상자 장애물이 하나 있어(크기는 작은 편).
                이 장애물을 피해서 앞으로 5m 이동할 수 있도록 10개의 좌표를 제시해줘.
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
                지그재그(S자)로 최단 경로에 가깝게 앞으로 6.5m 이동할 수 있도록 10개의 좌표를 제시해줘.
                각 상자로부터 최소 안전마진 0.5m는 반드시 확보하되, 불필요하게 크게 우회하지 말고
                최대한 직선에 가까운 효율적인 경로로 움직여.
                반드시 두 상자를 모두 피해가야해.
                """,
            "oracle_scene_C": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                그 상황속 사진에 보이듯이, 앞에 상자 형태의 장애물이 세 개 있어(위-아래-위 순서로 지그재그 배치, 크기도 서로 다름).
                세 장애물을 순서대로 피하면서 지그재그(슬랄롬) 형태로 앞으로 7m 이동할 수 있도록
                14개의 좌표를 제시해줘.

                가장 중요한 것은 각 장애물을 정확한 위치에서 안전마진만큼 피해가는 것이야 - 이걸 절대 희생하지 마.
                그 다음으로, waypoint 사이의 급격한 방향전환(90도에 가까운 꺾임)만 피해서
                각 지점을 좀 더 촘촘하고 매끄럽게 연결해줘.
                안전 회피가 우선이고 부드러움은 그 다음이야 - 부드러움을 위해 장애물과의 거리를 희생하면 안 돼.
                각 상자로부터 최소 안전마진을 확보하되 불필요하게 크게 우회하지는 마.
                반드시 세 상자를 모두 정확히 피해가야해.
                """,
            "oracle_scene_D": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                그 상황속 사진에 보이듯이, 로봇의 양옆으로 붉은 벽이 길고 좁은 통로를 이루고 있어.
                통로 안쪽 폭은 로봇 중심선 기준 좌우 약 0.85m로 상당히 빠듯하지만,
                로봇의 동적 클리어런스(0.5m)와 안전마진(0.3m)을 합한 필요 폭(0.8m)은 만족해.
                좌우로 크게 틀 필요는 없으니, 통로 중앙선(y=0)을 유지하며 곧게 직진해서
                앞으로 7m 이동할 수 있도록 12개의 좌표를 제시해줘.
                모든 좌표의 y값은 0에 최대한 가깝게(±0.1m 이내) 유지해야 해.
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
                반드시 통과하도록 15개의 좌표를 만들어줘. 시작점(0,0)에서 목표지점(6.5, 0)까지
                이동하되, 위 세 지점을 절대 놓치지 말고 정확한 y값(또는 그보다 더 안전한 값)으로
                통과해야 해. 애매하게 y=±0.5 정도로만 살짝 틀면 절대 안전하지 않아 - 반드시
                위에서 요구한 y값(±1.2 이상)을 달성해야 해.
                각 지점 사이는 부드러운 곡선으로 이어지게 하고, 웨이포인트 간 거리는 0.4m~1.0m를 유지해.
                """,
            
        }

        DEFAULT_SCENARIO = SCENARIO_TEMPLATES["oracle_scene_A"]
        scenario = SCENARIO_TEMPLATES.get(args.scene, DEFAULT_SCENARIO)
        
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
        
        NUM_WAYPOINTS = {"oracle_scene_C": 14, "oracle_scene_E": 15}
        num_waypoints = NUM_WAYPOINTS.get(args.scene, 10)
        court.run_case(scenario, image_path=image_path, robot_pos=robot_pos, scale=scale, scene_name=args.scene, num_waypoints=num_waypoints)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
