
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
        scenario = """
        중앙에 있는 로봇(go2)이 앞으로 가야하는 상황이야. 
        그 상황속 사진에 보이듯이, 앞에 빨간 상자인 장애물이 있어, 
        이 사진속 장애물을 피해서 앞으로 5m 이동할 수 있도록 10개의 좌표를 제시해줘.
        반드시 상자를 피해가야해.
        """
        
        if image_path:
            print(f"📸 Analying Image: {image_path}")
            # Image size: 1263x1080. Robot is perfectly centered.
            # New calibrated robot_pos: (631, 540)
            # New scale: 150.0 (Making 1m represent fewer pixels, thus AI plans longer jumps)
            robot_pos = (631, 540) 
            scale = 150.0 # 스케일을 낮춰서 AI가 더 시원시원한 경로(m)를 짜게 유도한다.
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
