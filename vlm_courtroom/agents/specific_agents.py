
import json
import chromadb
from typing import List, Dict, Any
from vlm_courtroom.agents.base_agent import VLMAgent, Message

class CoordinateAgent(VLMAgent):
    def __init__(self, name="CoordinateAgent"):
        super().__init__(name, "Coordinate Generator", model_role="COORDINATE")
        # Initialize ChromaDB (Persistent storage)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="scene_coordinates")

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Analyzing image and generating coordinates...")
        
        # In a real scenario, we'd pass the image to the model.
        # For now, we simulate the prompt or use a text description if no image is loaded.
        image_description = context.get('image_description', 'A scene with obstacles.')
        
        prompt = f"""
        You are a robot navigation assistant. 
        Analyze the scene: {image_description}
        Identify obstacles (e.g., puddles, cars).
        Generate 10 sequential (x, y) coordinates for a valid path avoiding obstacles.
        Return ONLY a JSON list of dictionaries with 'x' and 'y' keys.
        Example: [{{"x": 1, "y": 2}}, {{"x": 3, "y": 4}}]
        """
        
        response_text = self.generate_response(prompt)
        
        # Parse JSON from response (simple cleanup)
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                coordinates = json.loads(response_text[start_idx:end_idx])
                
                # Save to ChromaDB
                self.collection.add(
                    documents=[json.dumps(coordinates)],
                    metadatas=[{"context": image_description, "sender": self.name}],
                    ids=[f"path_{len(self.collection.get()['ids']) + 1}"]
                )
                print(f"[{self.name}] Coordinates saved to ChromaDB.")
            else:
                print(f"[{self.name}] ⚠️ Could not parse coordinates from output.")
                coordinates = []
        except Exception as e:
            print(f"[{self.name}] ⚠️ Error parsing JSON: {e}")
            coordinates = []

        return Message(self.name, response_text, "coordinate_proposal")


class ProsecutorAgent(VLMAgent):
    def __init__(self, name="ProsecutorAgent"):
        super().__init__(name, "Prosecutor", model_role="PROSECUTOR")

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Reviewing proposal...")
        previous_proposal = context.get('last_message_content', '')
        
        prompt = f"""
        You are a Prosecutor in a navigation court.
        Review the proposed path: {previous_proposal}
        Your goal is to find faults or risks (e.g., too close to obstacles, slipping risk).
        Provide:
        1. One strong Opinion (Critical).
        2. Three Reasons (Evidence based on safety).
        Format clearly.
        """
        
        response_text = self.generate_response(prompt)
        return Message(self.name, response_text, "argument_prosecution")


class DefenseAttorneyAgent(VLMAgent):
    def __init__(self, name="DefenseAttorneyAgent"):
        super().__init__(name, "Defense Attorney", model_role="DEFENSE")

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Defending proposal...")
        previous_proposal = context.get('last_message_content', '')
        prosecution_arg = context.get('prosecution_argument', '')
        
        prompt = f"""
        You are a Defense Attorney in a navigation court.
        Defend the proposed path: {previous_proposal}
        Counter the prosecution's argument: {prosecution_arg}
        Provide:
        1. One strong Opinion (Supportive).
        2. Three Reasons (Efficiency, feasibility).
        Format clearly.
        """
        
        response_text = self.generate_response(prompt)
        return Message(self.name, response_text, "argument_defense")


class JudgeAgent(VLMAgent):
    def __init__(self, name="JudgeAgent"):
        super().__init__(name, "Judge", model_role="JUDGE")

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Deliberating...")
        proposal = context.get('original_proposal', '')
        prosecution = context.get('prosecution_argument', '')
        defense = context.get('defense_argument', '')
        
        prompt = f"""
        You are the Chief Judge.
        Evaluate the Original Path: {proposal}
        Prosecution Argument: {prosecution}
        Defense Argument: {defense}
        
        Decide on the FINAL path. You can accept the original or modify it.
        1. State your Verdict and Logic.
        2. Provide the FINAL list of 10 coordinates (x, y) for the robot.
        3. Explain how these points should be connected (mention Spline).
        """
        
        response_text = self.generate_response(prompt)
        return Message(self.name, response_text, "verdict")
