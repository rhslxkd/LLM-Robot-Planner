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
           - Do NOT independently derive a required corridor width from this
             footprint (e.g. do not reason "radius r, so diameter 2r must fit
             in the gap"). That naive geometric estimate is WRONG and has been
             superseded by real DIAL-MPC empirical testing. The ONLY authoritative
             minimum-passable-gap number is the 0.6m figure in Section 2 below --
             use that number exclusively, never recompute your own.

        2. **Safety Protocol**:
           - Empirically measured via DIAL-MPC corridor-width stress test (centered
             straight-line path, Unitree Go2 trot gait): foot lateral excursion from
             the path centerline reaches up to ~0.28m per foot under normal gait.
             Degraded-but-stable behavior (feet brushing the boundary, no fall)
             observed down to a 0.40-0.55m gap. Catastrophic failure (loss of
             balance) confirmed at a 0.35m gap.
           - If a gap between obstacles is narrower than **0.6m**, it is considered
             UNPASSABLE. Do not attempt to go through. (This threshold keeps a
             margin above the empirically observed degradation zone while staying
             well clear of the confirmed 0.35m failure point.)

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

        def process(self, context: Dict[str, Any]) -> Message:
            print(f"[{self.name}] Narrating pre-computed path (no coordinate math)...")

            image_path = context.get('image_path')
            image_description = context.get('image_description', 'A scene with obstacles.')
            coordinate_proposal = context.get('coordinate_proposal')

            if not coordinate_proposal:
                print(f"[{self.name}] ⚠️ coordinate_proposal이 context에 없음 -- 빈 응답 반환")
                return Message(self.name, "Error: no coordinate_proposal provided", "coordinate_proposal")

            proposal_json = json.dumps(coordinate_proposal, ensure_ascii=False)
            num_waypoints = len(coordinate_proposal)

            prompt = f"""
            You are a robot navigation assistant.
            A path has ALREADY been computed by a validated search algorithm AND
            precisely measured by a deterministic geometry module -- every
            coordinate and every "clearance_m" value below is EXACT, computed
            directly from the occupancy map, not an estimate.

            Exact waypoints (world frame, start=(0,0), +X=forward), {num_waypoints} points:
            {proposal_json}

            Scene context: {image_description}
            The provided image shows this same path drawn on the scene for your
            reference.

            Your ONLY job is to describe, in Korean, how this given path relates
            to the obstacles/corridors visible in the image (e.g., which gap it
            threads through, any point where clearance_m looks tight relative to
            the 0.6m minimum). You MUST NOT alter, recompute, round differently,
            or re-estimate ANY of the given x/y/clearance_m values -- copy them
            through EXACTLY as given. Do not add or remove points.

            Output Format:
            ## Path Reading
            (Your description here, in Korean)

            ## Coordinates
            (Return the EXACT list given above, unchanged, as JSON)
            """

            response_text = self.generate_response(prompt, image_path=image_path)
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
        obstacles (0.3m effective radius) and minimum passable width (0.6m).
        Do NOT second-guess the overall route choice (which side of an
        obstacle it goes around, general direction) -- that routing decision
        was already produced by a validated search process and is out of
        scope for you. Focus only on margins.

        IMPORTANT: You do NOT see the image yourself -- you can ONLY use the
        "clearance_m" value the Coordinate agent already estimated for each
        waypoint (given in the proposal above). Do NOT invent, guess, or
        re-estimate your own width/distance numbers -- you have no way to
        verify them independently, and fabricated numbers make your verdict
        meaningless. Simply compare each waypoint's stated clearance_m against
        the 0.6m threshold. If clearance_m is missing for a waypoint, treat it
        as unknown and do not flag it as a violation on that basis alone.

        Determine which ONE of these two cases applies, and state it clearly
        as the FIRST line of your response:

        "VERDICT: LOCALLY_FIXABLE" -- violations exist only at specific
        waypoints, and can be resolved by nudging those waypoints away from
        the nearest obstacle while keeping the same overall route.

        "VERDICT: STRUCTURALLY_INFEASIBLE" -- the path passes through a gap
        or corridor narrower than 0.6m over an extended stretch (not just a
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
        (Cite specific waypoints/segments and their stated clearance_m values
        against the constraints -- do not introduce new numbers)

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
        fact (gap width vs 0.6m requirement), NOT a matter of opinion or
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
        required 0.6m over an extended stretch -- a physical fact, not a
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
          0.6m requirement). Do NOT output a JSON coordinate list in this
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