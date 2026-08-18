from vlm_courtroom.agents.specific_agents import CoordinateAgent, ProsecutorAgent, DefenseAttorneyAgent, JudgeAgent
import os

class VLMCourt:
    def __init__(self, backend: str = "gemini", ollama_model: str = None,
                 gemini_model: str = None, openai_model: str = None):
        """
        backend: "gemini"(기본값) / "ollama" / "openai"
        ollama_model: backend="ollama"일 때 4개 에이전트 전부에 적용할 모델 태그
                      (예: "qwen2.5vl:7b", "llava-llama3", "qwen3-vl", "minicpm-v")
        gemini_model: backend="gemini"일 때 지정하면 4개 에이전트 전부 이 모델 하나로
                      통일(예: "gemini-2.5-flash", "gemini-2.5-pro"). 지정 안 하면
                      기존처럼 역할별 혼합 매핑(Judge=pro, 나머지=flash)을 씀 --
                      이건 공정한 백본 비교용이 아니라는 점 주의.
        openai_model: backend="openai"일 때 4개 에이전트 전부에 적용할 모델명
                      (예: "gpt-4o", "gpt-4o-mini").
        -- VLM 백본 비교 실험은 4개 에이전트 모두 동일 모델로 통일해서 돌리는 게 원칙.
        """
        label = f"backend={backend}"
        if backend == "ollama":
            label += f", ollama_model={ollama_model}"
        elif backend == "gemini":
            label += f", gemini_model={gemini_model or '(역할별 혼합: pro/flash)'}"
        elif backend == "openai":
            label += f", openai_model={openai_model}"
        print(f"initializing VLMCourt... ({label})")

        agent_kwargs = {"backend": backend}
        if backend == "ollama":
            agent_kwargs["ollama_model"] = ollama_model
        elif backend == "gemini":
            agent_kwargs["gemini_model"] = gemini_model
        elif backend == "openai":
            agent_kwargs["openai_model"] = openai_model

        self.coordinate_agent = CoordinateAgent(**agent_kwargs)
        self.prosecutor_agent = ProsecutorAgent(**agent_kwargs)
        self.defense_agent = DefenseAttorneyAgent(**agent_kwargs)
        self.judge_agent = JudgeAgent(**agent_kwargs)
        print("Agents initialized.")

    def run_case(self, image_description: str, image_path: str = None, robot_pos: tuple = None, scale: float = None, scene_name: str = None, num_waypoints: int = 10, variant: str = None):
        """
        scene_name: 씬 이름 (예: "oracle_scene_A") -- 입력 이미지가 있는 위치 (data/<scene_name>/oracle.png)
        variant: 백엔드/모델 식별자 (예: "gemini", "ollama_qwen2_5vl_7b") -- 지정하면
                 출력이 data/<scene_name>/<variant>/ 하위 폴더에 저장되어, 같은 씬을
                 여러 백엔드/모델로 돌려도 서로 덮어쓰지 않는다. None이면 기존처럼
                 data/<scene_name>/ 바로 아래에 저장 (하위 호환).
        """
        print("\n=== 🏛️ VLM Courtroom Simulation Started 🏛️ ===\n")
        
        # 1. Coordinate Agent
        print("--- [Step 1] Coordinate Agent (Analyzing & Mapping) ---")
        coord_msg = self.coordinate_agent.process({
            'image_description': image_description,
            'image_path': image_path,
            'num_waypoints': num_waypoints
        })
        print(f"📍 Proposal:\n{coord_msg.content}\n")

        # 2. Prosecutor Agent
        print("--- [Step 2] Prosecutor Agent (Critique) ---")
        pros_msg = self.prosecutor_agent.process({
            'last_message_content': coord_msg.content,
            'num_waypoints': num_waypoints
            })
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
            'defense_argument': def_msg.content,
            'num_waypoints': num_waypoints
        })
        print(f"👨‍⚖️ Verdict:\n{judge_msg.content}\n")
        
        # 5. Visualization (if image_path is provided)
        coordinates = []
        if image_path:
            coordinates = self.visualize_path(image_path, judge_msg.content, robot_pos, scale, scene_name, variant)

        print("=== 🏛️ Case Closed 🏛️ ===")
        return judge_msg, coordinates

    def visualize_path(self, image_path: str, verdict_text: str, robot_pos: tuple = None, scale: float = None, scene_name: str = None, variant: str = None):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            import json
            import re

            json_match = re.search(r'```json\s*(\[.*?\])\s*```', verdict_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]', verdict_text, re.DOTALL)

            if not json_match:
                print(f"⚠️ Could not find coordinate JSON in verdict. Raw verdict:\n{verdict_text}")
                return []

            json_str = json_match.group(1) if json_match.groups() else json_match.group(0)
            json_str = re.sub(r"'([a-zA-Z0-9_]+)'\s*:", r'"\1":', json_str)
            json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
            
            try:
                coordinates = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parsing failed: {e}")
                print(f"Raw extracted string: {json_str}")
                return []
            
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            repo_root = os.path.dirname(project_root)

            if scene_name:
                base_dir = os.path.join(repo_root, "data", scene_name)
                project_output_dir = os.path.join(base_dir, variant) if variant else base_dir
            else:
                project_output_dir = os.path.join(project_root, "outputs")
            os.makedirs(project_output_dir, exist_ok=True)

            automation_json_path = os.path.join(project_output_dir, "last_judged_path.json")
            with open(automation_json_path, 'w') as f:
                json.dump(coordinates, f, indent=2)
            print(f"📄 Saved coordinates for automation to: {automation_json_path}")

            img = mpimg.imread(image_path)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img)
            
            img_h, img_w = img.shape[:2]
            
            if robot_pos and scale:
                rx, ry = robot_pos
                plot_xs = []
                plot_ys = []
                for c in coordinates:
                    px = rx + (c['x'] * scale)
                    py = ry - (c['y'] * scale) 
                    plot_xs.append(px)
                    plot_ys.append(py)
                ax.plot(rx, ry, 'bo', markersize=10, label='Go2 Robot (Origin)')
            else:
                scale_x = img_w / 5.0 
                scale_y = img_h / 5.0 
                plot_xs = [c['x'] * scale_x for c in coordinates]
                plot_ys = [c['y'] * scale_y for c in coordinates]

            ax.plot(plot_xs, plot_ys, 'r-', linewidth=2, label='Judge Path')
            ax.scatter(plot_xs, plot_ys, c='yellow', s=50, zorder=5)
            
            for i, (x, y) in enumerate(zip(plot_xs, plot_ys)):
                ax.annotate(f"{i}", (x, y), color='white', fontsize=12, fontweight='bold')

            plt.title("Judge's Final Verdict Path")
            plt.legend()
            
            from datetime import datetime
            import shutil

            input_filename = os.path.basename(image_path)
            filename_no_ext, ext = os.path.splitext(input_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            repo_root = os.path.dirname(project_root)

            if scene_name:
                project_input_dir = os.path.join(repo_root, "data", scene_name)
            else:
                project_input_dir = os.path.join(project_root, "inputs")
            os.makedirs(project_input_dir, exist_ok=True)

            target_input_path = os.path.join(project_input_dir, input_filename)

            if os.path.abspath(image_path) != os.path.abspath(target_input_path):
                shutil.copy2(image_path, target_input_path)
                print(f"📂 Copied input image to: {target_input_path}")
            else:
                print(f"📂 Input image is already in project inputs: {target_input_path}")

            if scene_name:
                base_dir = os.path.join(repo_root, "data", scene_name)
                project_output_dir = os.path.join(base_dir, variant) if variant else base_dir
            else:
                project_output_dir = os.path.join(project_root, "outputs")
            os.makedirs(project_output_dir, exist_ok=True)
            
            output_filename = f"{filename_no_ext}_verdict_{timestamp}{ext}"
            output_path_project = os.path.join(project_output_dir, output_filename)
            
            plt.savefig(output_path_project)
            print(f"🖼️ Saved verdict to Project Outputs: {output_path_project}")
            
            plt.close()
            return coordinates

        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return []