
from vlm_courtroom.agents.specific_agents import CoordinateAgent, ProsecutorAgent, DefenseAttorneyAgent, JudgeAgent
import chromadb
import os

class VLMCourt:
    def __init__(self):
        print("initializing VLMCourt...")
        self.coordinate_agent = CoordinateAgent()
        self.prosecutor_agent = ProsecutorAgent()
        self.defense_agent = DefenseAttorneyAgent()
        self.judge_agent = JudgeAgent()
        print("Agents initialized.")

    def run_case(self, image_description: str):
        print("\n=== 🏛️ VLM Courtroom Simulation Started 🏛️ ===\n")
        
        # 1. Coordinate Agent
        print("--- [Step 1] Coordinate Agent (Analyzing & Mapping) ---")
        coord_msg = self.coordinate_agent.process({'image_description': image_description})
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
        
        print("=== 🏛️ Case Closed 🏛️ ===")
        return judge_msg
