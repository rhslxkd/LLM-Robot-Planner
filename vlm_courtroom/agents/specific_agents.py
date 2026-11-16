import json
from typing import List, Dict, Any
from vlm_courtroom.agents.base_agent import VLMAgent, Message


# 4개 agent가 공유하는 물리 제약 블록. 씬 파라미터(clearance, margin 등)가 바뀌면
# 여기 한 곳만 고치면 Coordinate/Prosecutor/Defense/Judge 전부에 일괄 반영된다.
ROBOT_PHYSICAL_CONSTRAINTS = """
        ### [CRITICAL: Robot Physical Constraints - Unitree Go2]
        You MUST adhere to the following physical constraints for path planning:

        1. **Physical Footprint** (measured from the simulated collision geometry,
           dial_mpc/dial_mpc/models/unitree_go2/go2.xml):
           - Trunk + hip-joint envelope: 0.39m (Length, hip-to-hip) x 0.35m (Width,
             including hip joint housings). Note: this is narrower than the
             commonly-cited Unitree marketing spec (0.7m x 0.31m), which includes
             fully outstretched legs; the values above reflect the actual simulated
             collision shapes.
           - Do NOT derive your own required corridor width from this footprint
             (e.g. "radius r, so a gap of 2r is needed"). The only authoritative
             minimum passable gap is the 0.8m figure in Section 2 below, which was
             measured by executing the robot in simulation.

        2. **Safety Protocol** (measured with DIAL-MPC, Unitree Go2 trot gait):
           - In a corridor-width stress test on a straight centered path, each foot
             swings up to ~0.28m sideways from the path centerline. Feet brush the
             walls but the robot stays upright down to a 0.40-0.55m gap, and it
             falls at a 0.35m gap. A corridor of 0.63-0.98m combined with a sharp
             turn inside it also caused a fall.
           - Therefore: if a gap between obstacles is narrower than **0.8m**, it is
             UNPASSABLE. Do not attempt to go through.
           - This 0.8m already includes the safety buffer. Do NOT add another margin
             on top of it: a gap of 0.8m or wider is PASSABLE, and a gap that is only
             slightly wider than 0.8m is not grounds for rejection.
           - Keep the path centered in any gap, and avoid sharp turns inside narrow
             gaps.

        3. **Locomotion Constraints**:
           - Sequential Waypoint Distance (Step Length):
             - MIN: 0.4m (to prevent gait instability)
             - MAX: 1.0m (to prevent excessive acceleration)
             - RECOMMENDED: 0.6m - 0.7m
           - Turning Radius: Avoid sharp 90-degree turns. Use smooth arcs with a
             radius of at least **0.5m**.

        4. **Coordinate Mapping Strategy**:
           - Use the robot's current position as (0, 0).
           - Forward progress must be along the **+X axis**.
           - Side-to-side movement is along the **Y axis**.
"""


class CoordinateAgent(VLMAgent):
    def __init__(self, name="CoordinateAgent", **kwargs):
        # **kwargs: backend="gemini"/"ollama", ollama_model="..." 등이
        # VLMCourt로부터 그대로 흘러들어와 VLMAgent.__init__까지 전달된다.
        super().__init__(name, "Coordinate Generator", model_role="COORDINATE", **kwargs)
        # NOTE: ChromaDB(VectorDB) 저장 로직 제거됨.
        # 과거엔 self.collection.add()로 제안 좌표를 저장했으나, 어디서도 query()로
        # 조회하지 않아 실질적으로 아무 기능이 없는 오버헤드였음(임베딩 계산 + 디스크 I/O
        # + case가 쌓일수록 느려지는 collection.get() 전체 스캔). 4-agent 간 정보 공유는
        # courtroom.py의 순차적 컨텍스트 전달(dict)로 이미 이루어지고 있으므로 제거.
        # 최종 판결 좌표는 courtroom.py의 visualize_path()에서 last_judged_path.json으로
        # 영속 저장된다.

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Analyzing image and generating coordinates...")
        
        image_path = context.get('image_path')
        image_description = context.get('image_description', 'A scene with obstacles.')
        num_waypoints = context.get('num_waypoints', 10)
        
        prompt = f"""
        You are a robot navigation assistant. 
        Analyze the scene (image or description provided): {image_description}
        Identify obstacles (e.g., puddles, cars).
        {ROBOT_PHYSICAL_CONSTRAINTS}
        Task:
        1. Analyze the scene and Explain your path planning logic.
        2. Generate EXACTLY {num_waypoints} sequential (x, y) coordinates for a valid path avoiding obstacles.
           This number ({num_waypoints}) is not optional -- the output list MUST contain exactly {num_waypoints} points.
        
        Output Format:
        ## Scene Analysis
        (Explain obstacles and safe zones)
        
        ## Path Strategy
        (Explain why you chose this path)
        
        ## Coordinates
        (Return the JSON list here)
        Example: [{{"x": 1, "y": 2}}, {{"x": 3, "y": 4}}]

        Please respond in Korean.
        """
        
        # Pass image_path if available
        response_text = self.generate_response(prompt, image_path=image_path)
        
        # Parse JSON from response (simple cleanup)
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                coordinates = json.loads(response_text[start_idx:end_idx])
            else:
                print(f"[{self.name}] ⚠️ Could not parse coordinates from output.")
                coordinates = []
        except Exception as e:
            print(f"[{self.name}] ⚠️ Error parsing JSON: {e}")
            coordinates = []

        return Message(self.name, response_text, "coordinate_proposal")

    def revise(self, context: Dict[str, Any]) -> Message:
        """
        표 ① ablation 전용 (C2: +검사, C3: +검사+변호). 판사가 없는 구성에서는 제안자 자신이
        비판(과 변론)을 읽고 최종 경로를 정한다. 기존 4-agent 흐름(C4)에서는 호출되지 않는다.
        """
        print(f"[{self.name}] Revising proposal...")
        image_path = context.get('image_path')
        image_description = context.get('image_description', 'A scene with obstacles.')
        num_waypoints = context.get('num_waypoints', 10)
        proposal = context.get('original_proposal', '')
        prosecution = context.get('prosecution_argument', '')
        defense = context.get('defense_argument')
        defense_block = f"""
        Defense Rebuttal: {defense}""" if defense else ""

        prompt = f"""
        You are a robot navigation assistant revising your own path proposal.
        Analyze the scene (image or description provided): {image_description}
        {ROBOT_PHYSICAL_CONSTRAINTS}
        Your Previous Proposal: {proposal}
        Prosecution Critique: {prosecution}{defense_block}

        Task:
        1. Check each concern against the image and the constraints above, and decide which are valid.
        2. Revise the path only where a valid concern requires it.
        3. Generate EXACTLY {num_waypoints} sequential (x, y) coordinates for the FINAL path.
           This number ({num_waypoints}) is not optional -- the output list MUST contain exactly {num_waypoints} points.

        Output Format:
        ## Revision Logic
        (Which concerns you accepted or rejected, and why)

        ## Coordinates
        (Return the JSON list here)
        ```json
        [{{"x": 1.0, "y": 2.0}}, {{"x": 3.5, "y": 4.2}}]
        ```

        Please respond in Korean, but keep the JSON strictly in English/Numeric format.
        """
        response_text = self.generate_response(prompt, image_path=image_path)
        return Message(self.name, response_text, "revised_proposal")


class ProsecutorAgent(VLMAgent):
    def __init__(self, name="ProsecutorAgent", **kwargs):
        super().__init__(name, "Prosecutor", model_role="PROSECUTOR", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Reviewing proposal...")
        previous_proposal = context.get('last_message_content', '')
        
        prompt = f"""
        You are a Prosecutor in a navigation court.
        Review the proposed path: {previous_proposal}
        {ROBOT_PHYSICAL_CONSTRAINTS}
        Your goal is to find faults or risks using the constraints above as evidence
        (e.g., "the gap near waypoint 3 is only 0.6m wide, which violates the 0.8m
        minimum passable width" or "the turn between waypoint 5 and 6 exceeds the
        safe turning radius"). Do not rely on vague impressions -- cite specific
        waypoints and, where possible, estimated distances against the constraints.
        Provide:
        1. One strong Opinion (Critical).
        2. Three Reasons (Evidence based on the physical constraints and safety).
        Format clearly.

        Please respond in Korean.
        """
        
        response_text = self.generate_response(prompt, image_path=context.get('image_path'))
        return Message(self.name, response_text, "argument_prosecution")


class DefenseAttorneyAgent(VLMAgent):
    def __init__(self, name="DefenseAttorneyAgent", **kwargs):
        super().__init__(name, "Defense Attorney", model_role="DEFENSE", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Defending proposal...")
        previous_proposal = context.get('last_message_content', '')
        prosecution_arg = context.get('prosecution_argument', '')
        
        prompt = f"""
        You are a Defense Attorney in a navigation court.
        Defend the proposed path: {previous_proposal}
        Counter the prosecution's argument: {prosecution_arg}
        {ROBOT_PHYSICAL_CONSTRAINTS}
        When countering, use the constraints above to show the path actually satisfies
        them (e.g., "the flagged gap is in fact 1.0m wide, above the 0.8m minimum") --
        do not simply assert efficiency without checking the prosecution's specific claim
        against these numbers first. If the prosecution's safety concern is valid, concede
        it and argue for a minimal correction instead of blanket disagreement.
        Provide:
        1. One strong Opinion (Supportive).
        2. Three Reasons (Efficiency, feasibility, and constraint compliance).
        Format clearly.

        Please respond in Korean.
        """
        
        response_text = self.generate_response(prompt, image_path=context.get('image_path'))
        return Message(self.name, response_text, "argument_defense")


class JudgeAgent(VLMAgent):
    def __init__(self, name="JudgeAgent", **kwargs):
        super().__init__(name, "Judge", model_role="JUDGE", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Deliberating...")
        proposal = context.get('original_proposal', '')
        prosecution = context.get('prosecution_argument', '')
        defense = context.get('defense_argument', '')
        num_waypoints = context.get('num_waypoints', 10)
        
        prompt = f"""
        You are the Chief Judge.
        Evaluate the Original Path: {proposal}
        Prosecution Argument: {prosecution}
        Defense Argument: {defense}
        {ROBOT_PHYSICAL_CONSTRAINTS}
        Decide on the FINAL path. You can accept the original or modify it.
        1. State your Verdict and Logic.
        2. Provide the FINAL list of EXACTLY {num_waypoints} coordinates (x, y) for the robot.
           This number ({num_waypoints}) is not optional -- your final JSON array MUST contain exactly {num_waypoints} points,
           even if you modify or reject the original proposal.
        3. Explain how these points should be connected (mention Spline).

        Important: The coordinates MUST be provided as a JSON array at the end of your response.
        Example format:
        ```json
        [{{ "x": 1.0, "y": 2.0 }}, {{ "x": 3.5, "y": 4.2 }}]
        ```

        Please respond in Korean, but keep the JSON strictly in English/Numeric format.
        """
        
        response_text = self.generate_response(prompt, image_path=context.get('image_path'))
        return Message(self.name, response_text, "verdict")