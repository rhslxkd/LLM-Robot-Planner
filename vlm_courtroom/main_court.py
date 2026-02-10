
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
        
        court = VLMCourt()
        
        # Example Scenario Description
        # In a real system, this would come from an VLM analyzing an image file
        scenario = """
        A robotic dog (Go2) is facing a path.
        To the immediate left, there is a deep puddle (water hazard).
        To the right, there is a road with occasional car traffic.
        Directly ahead, there are scattered rocks but a clear path exists in the center.
        Target is 5 meters ahead.
        """
        
        print(f"Scenario Description: {scenario}")
        
        court.run_case(scenario)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
