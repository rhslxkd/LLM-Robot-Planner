
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
                그 상황속 사진에 보이듯이, 로봇의 양옆으로 붉은 벽이 통로를 이루고 있어.
                통로 안쪽 폭은 로봇 중심선 기준 좌우 약 1.1m로, 로봇의 동적 클리어런스(0.5m)와
                안전마진(0.5m)을 합한 필요 폭(1.0m)을 각 방향에서 충분히 만족해.
                다만 좌우로 크게 틀 필요는 없으니, 통로 중앙선(y=0)을 유지하며 곧게 직진해서
                앞으로 4.2m 이동할 수 있도록 10개의 좌표를 제시해줘.
                모든 좌표의 y값은 0에 최대한 가깝게(±0.1m 이내) 유지해야 해.
                """,
            "oracle_scene_E": """
                중앙에 있는 로봇(go2)이 미로 형태의 공간을 통과해서 목표지점까지 가야 하는 상황이야.
                이 미로는 복잡하므로, 네가 직접 장애물 위치를 다시 측정하지 말고
                아래에 이미 계산되어 주어진 체크포인트를 반드시 그대로, 순서대로 지나가는 경로를 만들어줘.

                ⚠️ 매우 중요: 아래 체크포인트들은 장애물의 위치나 경계선이 절대 아니야.
                각 체크포인트는 "로봇이 안전하게 서 있어도 되는 지점"이며, 안전마진까지 이미 계산이
                끝난 좌표야. 즉 체크포인트 좌표 자체를 지나가는 것은 안전하고, 오히려 반드시 그 지점을
                실제로 통과해야 해. 체크포인트를 피해서 돌아가면 안 되고, 체크포인트가 장애물이라고
                착각해서 그 근처를 우회하려 하면 오히려 실제 장애물(기둥, 가로 판)과 충돌하게 돼.
                장애물은 체크포인트 사이 구간에 있는 것이지, 체크포인트 위에 있는 게 아니야.

                반드시 아래 순서대로 통과해야 하는 체크포인트 (로봇 원점 기준 상대좌표):
                1. (1.2, -0.8)  - 첫 번째 기둥(위에서 내려옴)을 남쪽(아래)으로 통과
                2. (2.6, 0.8)   - 두 번째 기둥(아래에서 올라옴)을 북쪽(위)으로 통과
                3. (5.5, 0.8)   - 첫 번째 가로 판의 오른쪽 바깥쪽(안전한 지점)까지 동쪽으로 이동
                4. (5.8, 1.25)  - 급격한 수직 꺾임을 피하기 위한 대각선 라운딩 지점 (3번과 5번 사이를 부드러운 곡선으로 연결)
                5. (5.5, 1.7)   - 첫 번째 가로 판보다 위쪽의 안전한 지점 (상승 완료)
                6. (2.0, 1.7)   - 두 번째 가로 판의 왼쪽 바깥쪽(안전한 지점)까지 서쪽으로 이동
                7. (2.0, 2.9)   - 두 번째 가로 판보다 위쪽의 안전한 지점 (상승 완료)
                8. (7.0, 2.9)   - 최종 목표지점까지 동쪽으로 이동

                특히 3번과 5번 사이(4번 라운딩 지점을 지나는 구간)는 절대 급격한 90도 꺾임 없이,
                4번 지점을 통해 완만한 곡선으로 방향을 틀어야 해. 로봇이 몸을 크게 회전시켜야 하는
                급격한 코너를 최대한 피하는 게 이번 경로의 핵심 목표야.

                ⚠️ 특히 주의: 시작점(0,0)에서 첫 번째 체크포인트(1.2, -0.8)로 이동하는 구간에서
                로봇이 첫 번째 기둥(x=1.2 지점의 세로 벽)을 완전히 피하지 못하고 스치거나 관통하는
                경로가 자주 나왔어. 시작 직후의 waypoint 1~2번이 x=1.2 지점을 지날 때는
                반드시 y가 -0.4보다 충분히 작은(더 음수인) 값이어야 기둥 아래쪽으로 안전하게 피해갈 수 있어.
                이 시작 구간을 특히 신경 써서, 절대 벽을 뚫고 지나가지 않도록 확인해줘.

                위 8개 체크포인트에 시작점 (0.0, 0.0)을 더해서 총 9개 지점을 지나야 하는데,
                이 9개 지점을 그대로 다 쓰거나, 그 사이사이에 자연스러운 중간 지점을 추가해서
                총 15개의 좌표로 만들어줘 (체크포인트 자체는 절대 순서를 바꾸거나 생략하면 안 돼).
                각 지점 사이는 부드러운 곡선으로 이어지게 하고, 웨이포인트 간 거리는 0.4m~1.0m를 유지해.
                이 체크포인트들은 이미 안전마진(0.3m)을 확보해서 계산된 좌표이니, 그대로 신뢰하고 사용해도 돼.
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
            SCENE_SCALE = {"oracle_scene_C": 90.0, "oracle_scene_E": 90.0}
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
