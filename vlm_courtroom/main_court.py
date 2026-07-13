
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
                그 상황속 사진에 보이듯이, 앞에 빨간 상자인 장애물이 있어,
                이 사진속 장애물을 피해서 앞으로 5m 이동할 수 있도록 10개의 좌표를 제시해줘.
                반드시 상자를 피해가야해.
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
                세 장애물을 순서대로 피하면서 지그재그(슬랄롬) 형태로 앞으로 7m 이동할 수 있도록 10개의 좌표를 제시해줘.
                각 상자로부터 최소 안전마진 0.5m를 확보하되 불필요하게 크게 우회하지 말고 효율적으로 움직여.
                반드시 세 상자를 모두 피해가야해.
                """,
            "oracle_scene_D": """
                중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야.
                그 상황속 사진에 보이듯이, 로봇의 양옆으로 붉은 벽이 통로를 이루고 있어.
                통로 안쪽 폭은 로봇 중심선 기준 좌우 약 1.1m로, 로봇의 동적 클리어런스(0.5m)와
                안전마진(0.5m)을 합한 필요 폭(1.0m)을 각 방향에서 충분히 만족해.
                다만 좌우로 크게 틀 필요는 없으니, 통로 중앙선(y=0)을 유지하며 곧게 직진해서
                앞으로 4.2m 이동할 수 있도록 10개의 좌표를 제시해줘.
                모든 좌표의 y값은 0에 최대한 가깝게(±0.1m 이내) 유지해야 해.
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
            SCENE_SCALE = {"oracle_scene_C": 90.0}
            robot_pos = (421, 540)  # oracle_gen.py 의 ROBOT_PX(IMG_W/3, IMG_H/2) 와 반드시 동일값 유지
            scale = SCENE_SCALE.get(args.scene, 150.0)
        else:
            print(f"Scenario Description: {scenario}")
            robot_pos = None
            scale = None
        
        court.run_case(scenario, image_path=image_path, robot_pos=robot_pos, scale=scale, scene_name=args.scene)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
