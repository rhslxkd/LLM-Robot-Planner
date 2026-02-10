
from vlm_courtroom.agents.specific_agents import CoordinateAgent, ProsecutorAgent, DefenseAttorneyAgent, JudgeAgent
import chromadb
import os

class VLMCourt:
    def __init__(self, reset_db: bool = False):
        print("initializing VLMCourt...")
        self.coordinate_agent = CoordinateAgent(reset_db=reset_db)
        self.prosecutor_agent = ProsecutorAgent()
        self.defense_agent = DefenseAttorneyAgent()
        self.judge_agent = JudgeAgent()
        print("Agents initialized.")

    def run_case(self, image_description: str, image_path: str = None, robot_pos: tuple = None, scale: float = None):
        print("\n=== 🏛️ VLM Courtroom Simulation Started 🏛️ ===\n")
        
        # 1. Coordinate Agent
        print("--- [Step 1] Coordinate Agent (Analyzing & Mapping) ---")
        coord_msg = self.coordinate_agent.process({
            'image_description': image_description,
            'image_path': image_path
        })
        print(f"📍 Proposal:\n{coord_msg.content}\n")

        # 2. Prosecutor Agent
        print("--- [Step 2] Prosecutor Agent (Critique) ---")
        pros_msg = self.prosecutor_agent.process({'last_message_content': coord_msg.content})
        print(f"⚖️ Prosecution:\n{pros_msg.content}\n")

        # 3. Defense Agent
        print("--- [Step 3] Defense Agent (Rebuttal) ---")
        def_msg = self.defense_agent.process({
            'last_message_content': coord_msg.content, 
            'prosecution_argument': pros_msg.content
        })
        print(f"🛡️ Defense:\n{def_msg.content}\n")

        # 4. Judge Agent
        print("--- [Step 4] Judge Agent (Final Verdict) ---")
        judge_msg = self.judge_agent.process({
            'original_proposal': coord_msg.content,
            'prosecution_argument': pros_msg.content,
            'defense_argument': def_msg.content
        })
        print(f"👨‍⚖️ Verdict:\n{judge_msg.content}\n")
        
        # 5. Visualization (if image_path is provided)
        if image_path:
            self.visualize_path(image_path, judge_msg.content, robot_pos, scale)

        print("=== 🏛️ Case Closed 🏛️ ===")
        return judge_msg

    def visualize_path(self, image_path: str, verdict_text: str, robot_pos: tuple = None, scale: float = None):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            import json
            import re

            # Extract JSON coordinates from verdict
            json_match = re.search(r'\[.*\]', verdict_text, re.DOTALL)
            if not json_match:
                print("⚠️ Could not find coordinate JSON in verdict for visualization.")
                return

            coordinates = json.loads(json_match.group(0))
            
            # Load image
            img = mpimg.imread(image_path)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img)
            
            # Default values if not provided (fallback to previous simple scaling)
            img_h, img_w = img.shape[:2]
            
            if robot_pos and scale:
                # Robot-centric transformation
                # Assumes:
                # - Robot Pos (rx, ry) is the origin (0,0) in logical coords
                # - Logical X is Forward (Right in image for this specific scenario?) 
                #   WAIT. In the image (brax.png), the robot is facing RIGHT/FORWARD.
                #   Let's assume X is forward (Right) and Y is Left (Up).
                #   We need to verify the coordinate system convention. 
                #   Usually X=Forward.
                rx, ry = robot_pos
                
                plot_xs = []
                plot_ys = []
                for c in coordinates:
                    # Transform logical (x, y) to image (px, py)
                    # Image X = Robot X + Logical X * Scale (if X is right/forward)
                    # Image Y = Robot Y - Logical Y * Scale (if Y is left/up)
                    # Note: We need to see if LLM generates x,y in a specific frame. 
                    # Usually: x=forward, y=lateral.
                    
                    # For brax.png, robot faces RIGHT. 
                    # So x+ is Image Right. y+ is Image Up (Left of robot).
                    
                    px = rx + (c['x'] * scale)
                    py = ry - (c['y'] * scale) 
                    plot_xs.append(px)
                    plot_ys.append(py)
                
                # Plot robot position
                ax.plot(rx, ry, 'bo', markersize=10, label='Go2 Robot (Origin)')
                
            else:
                # Previous Scale-to-Fit mode
                scale_x = img_w / 5.0 
                scale_y = img_h / 5.0 
                plot_xs = [c['x'] * scale_x for c in coordinates]
                plot_ys = [c['y'] * scale_y for c in coordinates]

            ax.plot(plot_xs, plot_ys, 'r-', linewidth=2, label='Judge Path')
            ax.scatter(plot_xs, plot_ys, c='yellow', s=50, zorder=5)
            
            # Add labels
            for i, (x, y) in enumerate(zip(plot_xs, plot_ys)):
                ax.annotate(f"{i}", (x, y), color='white', fontsize=12, fontweight='bold')

            plt.title("Judge's Final Verdict Path")
            plt.legend()
            
            # Imports for saving
            import os
            import shutil
            from datetime import datetime

            input_filename = os.path.basename(image_path)
            filename_no_ext, ext = os.path.splitext(input_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Manage Input File (Copy to project inputs directory)
            project_input_dir = os.path.join(os.getcwd(), "vlm_courtroom", "inputs")
            os.makedirs(project_input_dir, exist_ok=True)
            
            target_input_path = os.path.join(project_input_dir, input_filename)
            
            # Only copy if source and destination are different
            if os.path.abspath(image_path) != os.path.abspath(target_input_path):
                shutil.copy2(image_path, target_input_path)
                print(f"📂 Copied input image to: {target_input_path}")
            else:
                print(f"📂 Input image is already in project inputs: {target_input_path}")
            
            # 2. Save Result to Project Outputs Directory (ONLY)
            project_output_dir = os.path.join(os.getcwd(), "vlm_courtroom", "outputs")
            os.makedirs(project_output_dir, exist_ok=True)
            
            output_filename = f"{filename_no_ext}_verdict_{timestamp}{ext}"
            output_path_project = os.path.join(project_output_dir, output_filename)
            
            plt.savefig(output_path_project)
            print(f"🖼️ Saved verdict to Project Outputs: {output_path_project}")
            
            plt.close()

        except Exception as e:
            print(f"❌ Visualization failed: {e}")
