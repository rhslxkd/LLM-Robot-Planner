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
           - Dynamic Clearance: Consider the robot as a cylinder with a **0.5m radius**.
             This fully circumscribes the measured footprint even at 45-degree yaw
             (worst-case half-diagonal is 0.26m) and adds margin for gait sway during
             trotting, which is not captured by the static collision geometry above.

        2. **Safety Protocol**:
           - Maintain a minimum **Safety Margin of 0.3m** from any detected obstacle
             (puddles, objects, curbs).
             (Note: this margin is ON TOP OF the 0.5m dynamic clearance radius above,
             which already accounts for the robot's physical footprint and gait sway.
             The margin itself only needs to cover residual uncertainty, so do not
             treat it as an additional large buffer.)
           - If a gap between obstacles is narrower than **1.6m** (2 x effective
             clearance radius of 0.8m), it is considered UNPASSABLE. Do not attempt
             to go through.

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
            super().__init__(name, "Coordinate Generator", model_role="COORDINATE", **kwargs)
            # NOTE: ChromaDB(VectorDB) 저장 로직 제거됨.
            # 과거엔 self.collection.add()로 제안 좌표를 저장했으나, 어디서도 query()로
            # 조회하지 않아 실질적으로 아무 기능이 없는 오버헤드였음(임베딩 계산 + 디스크 I/O
            # + case가 쌓일수록 느려지는 collection.get() 전체 스캔). 4-agent 간 정보 공유는
            # courtroom.py의 순차적 컨텍스트 전달(dict)로 이미 이루어지고 있으므로 제거.
            # 최종 판결 좌표는 courtroom.py의 visualize_path()에서 last_judged_path.json으로
            # 영속 저장된다.

        def process(self, context: Dict[str, Any]) -> Message:
            print(f"[{self.name}] Reading pre-planned path from image...")

            image_path = context.get('image_path')
            image_description = context.get('image_description', 'A scene with obstacles.')
            num_waypoints = context.get('num_waypoints', 10)

            prompt = f"""
            You are a robot navigation assistant.
            The provided image shows a top-down scene with a world-coordinate axis
            grid (labeled in meters, e.g. "+1.0m", "-2.0m") and a path ALREADY
            DRAWN on it: an orange line from the START point (green dot) to the
            GOAL point (red dot). This path was computed by a validated search
            algorithm -- it already avoids all obstacles.

            Your job is NOT to invent a new path. Your job is to READ the drawn
            orange path off the image and convert it into the robot's world
            coordinate frame, using the visible axis labels as your reference.
            Scene context: {image_description}
            {ROBOT_PHYSICAL_CONSTRAINTS}

            Task:
            1. Describe where the drawn orange path goes relative to the axis
            labels and obstacles (e.g., "corridor를 y=0 근처로 통과").
            2. Sample EXACTLY {num_waypoints} points evenly spaced along the
            drawn orange path (not invented) and convert each to the robot's
            world frame: start point = (0, 0), +X = forward.
            This number ({num_waypoints}) is not optional -- the output list
            MUST contain exactly {num_waypoints} points, ordered from start
            to goal.
            3. Do NOT deviate from the drawn path's shape. If part of it appears
            to violate the physical constraints above, note the deviation
            explicitly instead of silently redrawing it.

            Output Format:
            ## Path Reading
            (Describe the drawn path's shape and how it relates to obstacles/axis)

            ## Coordinates
            (Return the JSON list here)
            Example: [{{"x": 1, "y": 2}}, {{"x": 3, "y": 4}}]

            Please respond in Korean.
            """

            response_text = self.generate_response(prompt, image_path=image_path)

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


class ProsecutorAgent(VLMAgent):
    def __init__(self, name="ProsecutorAgent", **kwargs):
        super().__init__(name, "Prosecutor", model_role="PROSECUTOR", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Checking safety margin (clearance)...")
        previous_proposal = context.get('last_message_content', '')
        num_waypoints = context.get('num_waypoints', 10)

        prompt = f"""
        You are the Safety Officer (Prosecutor) in a navigation court.
        Review the proposed path below (waypoint coordinates plus any
        geometry/distance estimates already stated):
        {previous_proposal}
        {ROBOT_PHYSICAL_CONSTRAINTS}

        Your ONLY job is to verify SAFETY MARGIN compliance -- clearance from
        obstacles (0.8m effective radius) and minimum passable width (1.6m).
        Do NOT second-guess the overall route choice (which side of an
        obstacle it goes around, general direction) -- that routing decision
        was already produced by a validated search process and is out of
        scope for you. Focus only on margins.

        Determine which ONE of these two cases applies, and state it clearly
        as the FIRST line of your response:

        "VERDICT: LOCALLY_FIXABLE" -- violations exist only at specific
        waypoints, and can be resolved by nudging those waypoints away from
        the nearest obstacle while keeping the same overall route.

        "VERDICT: STRUCTURALLY_INFEASIBLE" -- the path passes through a gap
        or corridor narrower than 1.6m over an extended stretch (not just a
        single point), so no amount of nudging individual waypoints can
        achieve the required clearance. State the estimated actual width of
        the offending gap and over what distance it stays that narrow.

        If LOCALLY_FIXABLE: output a corrected coordinate list with EXACTLY
        {num_waypoints} points (same count, same start/goal), nudging only
        the violating waypoints. Cite which waypoints you changed and why.

        If STRUCTURALLY_INFEASIBLE: do NOT attempt to output a corrected
        list -- it cannot satisfy the constraint by local correction. State
        the violation clearly with evidence (measured width, affected
        waypoint range) so the Judge can decide whether to reject the route
        entirely.

        Output Format:
        VERDICT: <LOCALLY_FIXABLE or STRUCTURALLY_INFEASIBLE>

        ## Evidence
        (Cite specific waypoints/segments and estimated distances against
        the constraints)

        ## Corrected Coordinates (only if LOCALLY_FIXABLE)
        [{{"x": 1, "y": 2}}, {{"x": 3, "y": 4}}]

        Please respond in Korean, but keep the "VERDICT:" line and any JSON
        in English/Numeric format so they can be parsed programmatically.
        """

        response_text = self.generate_response(prompt)
        return Message(self.name, response_text, "argument_prosecution")


class DefenseAttorneyAgent(VLMAgent):
    def __init__(self, name="DefenseAttorneyAgent", **kwargs):
        super().__init__(name, "Defense Attorney", model_role="DEFENSE", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Reviewing for semantic improvements...")
        previous_proposal = context.get('last_message_content', '')
        prosecution_arg = context.get('prosecution_argument', '')

        prompt = f"""
        You are the Defense Attorney in a navigation court -- but your role is
        NOT to blindly defend the original proposal. Your role is to check
        whether a SEMANTICALLY-INFORMED local improvement exists, using
        context the Prosecutor's pure clearance-based check cannot see
        (task priorities, ambiguous risk areas, scene-specific context
        mentioned in the description).

        Original proposal: {previous_proposal}
        Prosecutor's safety review: {prosecution_arg}
        {ROBOT_PHYSICAL_CONSTRAINTS}

        First, check the Prosecutor's VERDICT line.

        If VERDICT: STRUCTURALLY_INFEASIBLE -- this is a measured physical
        fact (gap width vs 1.6m requirement), NOT a matter of opinion or
        planning philosophy. You are FORBIDDEN from arguing that the path
        is "conceptually optimal" or blaming the environment as a way to
        soften the verdict -- that is not a valid defense, it is evasion.
        You have exactly two allowed moves:
        (a) If the description/context plausibly indicates a genuinely
            different route exists that avoids the offending corridor
            entirely, describe it concretely.
        (b) Otherwise, concede plainly and output the exact line
            "RECOMMEND REJECTION" followed by one sentence why.
        Do not output a coordinate list in this case.

        If VERDICT: LOCALLY_FIXABLE -- the Prosecutor already nudged specific
        waypoints for safety. Check ONLY those corrected waypoints (and
        immediate neighbors) for a semantically better choice than a
        minimal safety nudge. Any change must be justified with concrete
        evidence tied to the image/description -- vague appeals to
        "efficiency" are not sufficient.
        If you cannot find clear justification, output the exact line
        "ADOPT PROSECUTOR'S WAYPOINTS UNCHANGED" and explain why briefly.
        Do not alter waypoints the Prosecutor did not flag.

        Your response MUST contain exactly one of these three exact lines,
        verbatim, so it can be parsed programmatically:
        "RECOMMEND REJECTION"
        "ADOPT PROSECUTOR'S WAYPOINTS UNCHANGED"
        "PROPOSED CHANGES"
        (use "PROPOSED CHANGES" only when providing a justified revised list
        under the LOCALLY_FIXABLE case)

        Output Format:
        ## Assessment
        (State which VERDICT case applies and your reasoning)

        ## Recommendation
        <one of the three exact lines above> -- (brief justification)

        ## Coordinates (only if PROPOSED CHANGES)
        [{{"x": 1, "y": 2}}, {{"x": 3, "y": 4}}]

        Please respond in Korean, but keep the three exact recommendation
        lines and any JSON in English/Numeric format.
        """

        response_text = self.generate_response(prompt)
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
        You are the Chief Judge in a navigation court.
        Original proposal (from a validated search algorithm, transcribed
        into coordinates): {proposal}
        Prosecutor's safety review: {prosecution}
        Defense's semantic review: {defense}
        {ROBOT_PHYSICAL_CONSTRAINTS}

        First, check the Prosecutor's VERDICT line:

        CASE 1 - VERDICT: STRUCTURALLY_INFEASIBLE
        This means the path passes through a gap/corridor narrower than the
        required 1.6m over an extended stretch -- a physical fact, not a
        matter of opinion. No local waypoint adjustment can fix this.
        - If the Defense identified a genuinely different alternative route
          that avoids the offending corridor entirely, evaluate whether
          that alternative route is itself safe (re-check it against the
          physical constraints above), and if so, adopt it as the final
          path.
        - Otherwise, you MUST REJECT the task. Do NOT force a coordinate
          list that violates the safety margin just to produce an output.
          State clearly: "FINAL VERDICT: REJECTED" and explain that the
          scene's geometry itself does not permit a safe path under the
          current physical constraints (cite the measured gap width vs the
          1.6m requirement). Do NOT output a JSON coordinate list in this
          case.

        CASE 2 - VERDICT: LOCALLY_FIXABLE
        Compare the Prosecutor's corrected waypoints against the Defense's
        recommendation:
        - If Defense said "ADOPT PROSECUTOR'S WAYPOINTS UNCHANGED", finalize
          the Prosecutor's corrected list.
        - If Defense proposed justified changes, re-check those changes
          yourself against the physical constraints before adopting them.
          If they introduce a new violation, fall back to the Prosecutor's
          corrected list instead.
        State "FINAL VERDICT: ACCEPTED" (or "ACCEPTED WITH CORRECTIONS"),
        explain your reasoning, and provide the FINAL list of EXACTLY
        {num_waypoints} coordinates (x, y). This number ({num_waypoints})
        is not optional -- the JSON array MUST contain exactly
        {num_waypoints} points. Briefly explain how the points should be
        connected (mention Spline).

        Output Format:
        ## Verdict
        FINAL VERDICT: <ACCEPTED / ACCEPTED WITH CORRECTIONS / REJECTED>
        (reasoning)

        ## Coordinates (omit entirely if REJECTED)
```json
        [{{ "x": 1.0, "y": 2.0 }}, {{ "x": 3.5, "y": 4.2 }}]
```

        Please respond in Korean, but keep "FINAL VERDICT:" and any JSON
        strictly in English/Numeric format.
        """

        response_text = self.generate_response(prompt)
        return Message(self.name, response_text, "verdict")