
import sys
import os

# Add the project root to sys.path to ensure imports work if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from vlm_courtroom.config import init_vertex_ai
from vlm_courtroom.court.courtroom import VLMCourt

def main():
    try:
        # Initialize Vertex AI connection
        init_vertex_ai()
        
        # Initialize Court with Database Reset (clears previous tests)
        court = VLMCourt(reset_db=True)

        # [Configuration]
        # Windows Path Example in WSL: "/mnt/d/Users/Downloads"
        # Project Input Path: os.path.join(current_dir, "vlm_courtroom", "inputs")
        IMAGE_DIR = "/mnt/d/Datasets/HELM/Input_images/go2/" 
        
        # [User Input] Image Filename or Full Path
        image_filename = "brax (1).png"
        
        if image_filename:
            # Check if it's a full path, otherwise join with IMAGE_DIR
            if os.path.isabs(image_filename):
                image_path = image_filename
            else:
                image_path = os.path.join(IMAGE_DIR, image_filename)
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
        
        court.run_case(scenario, image_path=image_path, robot_pos=robot_pos, scale=scale)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
