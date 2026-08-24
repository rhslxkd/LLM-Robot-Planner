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
           - **CRITICAL: Do NOT stack your own additional safety margin on top of
             this 0.6m number.** It is not a bare physical minimum -- it was
             derived directly from the empirical degradation/failure data above
             and already includes adequate margin. If a waypoint's clearance_m is
             >= 0.6m -- even if only marginally, e.g. 0.61m-0.65m -- you MUST
             treat that segment as PASSABLE without qualification. Do not describe
             a passing value as "barely enough," "risky," "too tight for comfort,"
             or use closeness to the threshold (or the length of the segment) as
             grounds for STRUCTURALLY_INFEASIBLE or REJECTED. The ONLY valid
             clearance-based rejection criterion is clearance_m < 0.6m. Nothing
             else about a passing value is grounds for rejection.

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

        - **CRITICAL: The ONLY verified geometric ground truth is the numeric
             {x, y, clearance_m} data given in the "## Coordinates" list.** Any
             prose description elsewhere (e.g. a "Scene Analysis" or "Path
             Description" section describing wall shapes, positions, or extents)
             is an unverified visual interpretation and MAY BE WRONG -- it is not
             measured data. If such prose seems to claim a waypoint crosses or is
             blocked by a wall, but that exact waypoint's own clearance_m is
             >= 0.6m, TRUST THE NUMBER, not the prose. A clearance_m >= 0.6m at a
             specific (x,y) is a direct ray-cast measurement proving that point is
             in open space with at least 0.6m of room -- it cannot simultaneously
             be "inside a wall." Do NOT issue a STRUCTURALLY_INFEASIBLE or
             REJECTED verdict based on a prose claim about wall geometry when the
             numeric clearance_m data for the relevant waypoints contradicts it.
        - **How to evaluate this**: the final path is spline-interpolated
             across ALL waypoints (see Judge's instructions), not followed as
             sharp straight segments. Do NOT compute a circumradius from 3
             consecutive waypoints and reject the path if that number is below
             0.5m -- with closely-spaced waypoints (which is normal and good,
             since MIN step length is only 0.4m), any 3-point circumradius will
             look artificially small even for a path that is, after spline
             interpolation, a perfectly smooth curve. Only flag a turning-radius
             concern if the path makes a large, sustained direction change (turn
             angle > ~90 degrees between consecutive segment directions) AND the
             segments involved are close to the MAX step length (~1.0m) -- that
             combination is what actually produces a sharp corner the spline
             cannot smooth out. Do NOT issue STRUCTURALLY_INFEASIBLE based on
             circumradius math alone.
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

        A path has ALREADY been computed by a validated search algorithm (Neural A*)
        AND precisely measured by a deterministic geometry module. Every coordinate
        and every "clearance_m" value below is EXACT -- it was NOT estimated visually,
        it was computed from pixel-level collision geometry.

        Scene: {image_description}

        Exact waypoints (world frame, robot start = (0,0), +X = forward direction,
        +Y = left), {num_waypoints} points in order:
        {proposal_json}

        Each point's "clearance_m" is the full corridor width (wall-to-wall) measured
        perpendicular to the path's direction of travel at that exact point.

        Task:
        1. Look at the image. Describe in Korean, in GENERAL/QUALITATIVE terms only,
           which obstacles/corridors/walls this given path passes through or around
           (e.g., "1~4번 지점은 남쪽 통로를 통해 벽을 우회한다"). Do NOT state specific
           numeric coordinate ranges or extents for where a wall begins/ends (e.g., do
           NOT write things like "벽이 X축 1.0m~2.0m에 걸쳐 있다" or "Y축 -2.0m 지점에
           틈새 없는 벽이 있다") -- you cannot reliably verify exact wall boundaries from
           the image, and an invented number here can be mistaken for verified data by
           the Prosecutor/Judge later, causing a false rejection. The ONLY verified
           geometric ground truth is the given x/y/clearance_m data below -- everything
           else you write is descriptive framing, not a geometric fact.
        2. Note any waypoint where clearance_m looks unusually tight relative to the
           robot's footprint, purely as an observation (you are not deciding pass/fail --
           that's the Prosecutor/Judge's job).

        CRITICAL CONSTRAINT:
        You MUST NOT alter, recompute, round differently, add, remove, or re-estimate
        ANY of the given x/y/clearance_m values. Copy them through EXACTLY as given in
        the "## Coordinates" section below -- your role is narration/interpretation only,
        never coordinate generation.

        Output Format:
        ## Scene Analysis
        (obstacles/corridors visible in the image, in Korean)

        ## Path Description
        (how the given path relates to those obstacles, waypoint by waypoint, in Korean)

        ## Coordinates
        {proposal_json}

        Please respond in Korean, except the JSON in ## Coordinates which must be copied
        through exactly as given above.
        """

        response_text = self.generate_response(prompt, image_path=image_path)
        return Message(self.name, response_text, "coordinate_proposal")


class ProsecutorAgent(VLMAgent):
    def __init__(self, name="ProsecutorAgent", **kwargs):
        super().__init__(name, "Prosecutor", model_role="PROSECUTOR", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Checking safety margin (clearance)...")
        previous_proposal = context.get('last_message_content', '')
        retry_feedback = context.get('retry_feedback', '')

        retry_block = ""
        if retry_feedback:
            retry_block = f"""
        ### PREVIOUS ATTEMPT FAILED VERIFICATION
        Your last proposed path failed independent checks (visual inspection and/or
        deterministic geometric recomputation -- these recompute exact numbers from
        the actual image pixels, they are not estimates). Report:
        {retry_feedback}

        CRITICAL: If the report above includes a "suggested_x"/"suggested_y" value
        for a specific waypoint index, that value was computed by deterministic
        ray-casting against the actual wall geometry (finding the true center of
        the corridor at that point) -- it is NOT a guess. USE THAT EXACT VALUE for
        that waypoint. Do NOT invent your own direction/magnitude of correction
        (e.g. "move it 0.08m this way") for any waypoint that already has a
        suggested_x/suggested_y given -- your own visual guesses at this have
        repeatedly failed verification in prior attempts. Only reason freely about
        waypoints that do NOT have a suggested value provided.

        If the report distinguishes between two suggestion types -- "(a) 기존
        waypoint 이동 제안" (with a plain "idx") and "(b) 새 굴절점 삽입 제안" (with
        "insert_after_idx") -- treat them differently:
        - An "idx" suggestion means: REPLACE that existing waypoint's x/y with the
          given suggested_x/suggested_y. Waypoint count stays the same.
        - An "insert_after_idx" suggestion means: INSERT a brand-new waypoint with
          the given suggested_x/suggested_y immediately after that index, keeping
          all existing waypoints unchanged. Waypoint count increases by one. This
          happens when a straight line between two existing waypoints cuts a wall
          corner (e.g. because a previous edit deleted the point that used to
          route around that corner) -- moving either endpoint alone cannot fix it,
          a new bend point is required. Do NOT try to solve an "insert_after_idx"
          case by moving an existing waypoint instead; insert the new point.
        """

        prompt = f"""
        You are a Prosecutor (safety officer) in a navigation court.
        Review the proposed path: {previous_proposal}
        {ROBOT_PHYSICAL_CONSTRAINTS}
        {retry_block}
        Your job:
        1. Check every waypoint's "clearance_m" (>= 0.6m required) and, if present,
           "dist_to_wall_m" (distance to the NEAREST wall -- a small value relative
           to clearance_m means the waypoint sits close to one side of the corridor
           even though the corridor itself has more total room).
        2. If you find a waypoint that is unsafe or poorly positioned (hugging a
           wall), you are AUTHORIZED to propose a corrected (x, y) for that
           waypoint directly -- do not just flag the problem, fix it. Reason from
           the given numeric data and neighboring waypoints' coordinates: if a
           point sits close to one wall, move it toward the corridor's estimated
           center, roughly perpendicular to the local path direction. Keep any
           correction small and local (do not redesign the whole path) and keep
           it consistent with neighboring waypoints (no large jumps).
        3. If a segment is fundamentally impassable (clearance_m < 0.6m across a
           wide stretch with no viable local fix), say so clearly.

        VERDICT: [STRUCTURALLY_INFEASIBLE / LOCALLY_FIXABLE / NO_ISSUE]

        ## Evidence
        (cite specific waypoint indices and numbers)

        ## Corrected Coordinates
        (the FULL waypoint list, in order, with your corrections applied -- copy
        every field through unchanged for points you didn't touch, and give new
        x/y for points you changed. The count does not need to exactly match the
        input if you removed a redundant point or split an overlong segment, but
        avoid unnecessary changes.)

        Please respond in Korean, keep the JSON list in English/Numeric format.
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
        You are a Defense Attorney in a navigation court.
        Original proposed path: {previous_proposal}
        Prosecutor's argument (may include a corrected coordinate list):
        {prosecution_arg}
        {ROBOT_PHYSICAL_CONSTRAINTS}

        Your job:
        1. Check whether the Prosecutor's corrections (if any) are actually
           necessary and correctly reasoned -- use the physical constraints
           above, not vague impressions.
        2. If the Prosecutor over-corrected (e.g. changed/rejected a waypoint
           that was already safe: clearance_m >= 0.6m and not meaningfully
           wall-hugging), push back and restore the original value for that
           point.
        3. If the Prosecutor missed something (e.g. a wall-hugging waypoint it
           didn't flag, or a step-length constraint violation), propose your
           own additional correction.
        4. You are authorized to directly propose corrected (x, y) values, same
           rules as the Prosecutor: keep corrections small/local, stay
           consistent with neighboring waypoints.

        ## Assessment
        (agree/disagree with the Prosecutor's specific claims, with reasons)

        ## Recommendation
        (ADOPT PROSECUTOR'S COORDINATES / ADOPT MY CORRECTIONS / ADOPT ORIGINAL)

        ## Coordinates
        (the FULL waypoint list you recommend as final, in order)

        Please respond in Korean, keep the JSON list in English/Numeric format.
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

        prompt = f"""
        You are the Chief Judge.
        Evaluate the Original Path: {proposal}
        Prosecution Argument: {prosecution}
        Defense Argument: {defense}
        {ROBOT_PHYSICAL_CONSTRAINTS}

        Decide on the FINAL path. Choose between the Prosecution's coordinates,
        the Defense's coordinates, or the original -- or merge specific point
        corrections from either side, whichever set of coordinates you determine
        is actually safe and correct per the physical constraints above.

        The final waypoint count does NOT need to exactly match any of the
        inputs -- if a correction legitimately required adding or removing a
        point, that is fine. Do not reject an otherwise-valid safety correction
        merely because it changed the waypoint count.

        1. State your Verdict and Logic.
        2. Provide the FINAL list of coordinates (x, y) for the robot, in order.
        3. Explain how these points should be connected (mention Spline).

        Important: The coordinates MUST be provided as a JSON array at the end
        of your response.
        Example format:
```json
        [{{ "x": 1.0, "y": 2.0 }}, {{ "x": 3.5, "y": 4.2 }}]
```

        Please respond in Korean, but keep the JSON strictly in English/Numeric format.
        """

        response_text = self.generate_response(prompt)
        return Message(self.name, response_text, "verdict")


class VerifierAgent(VLMAgent):
    """Judge의 최종 경로가 그려진 이미지를 보고 '선이 벽과 겹치는가'만 이진 판단.
    좌표 계산이나 clearance 판단은 하지 않음 -- 순수 시각적 충돌 검사 전용.
    (거리/clearance 재검증은 VLM 눈대중이 아니라 courtroom.py의 결정론적
    ray-casting이 별도로 담당한다.)"""
    def __init__(self, name="VerifierAgent", **kwargs):
        super().__init__(name, "Visual Verifier", model_role="VERIFIER", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Visually inspecting rendered path for collisions...")
        image_path = context.get('image_path')

        prompt = """
        You are a visual collision inspector for a robot navigation system.

        Look ONLY at the provided image. It shows a top-down map with red
        obstacles/walls and a drawn line representing a planned robot path,
        with numbered waypoint markers along it.

        Your ONLY job: determine whether the drawn path line visually crosses,
        overlaps, or passes through any red obstacle/wall pixel anywhere along
        its length. Do NOT evaluate clearance distances, safety margins, or any
        other physical constraint -- purely: does the drawn line touch red?

        Answer in exactly this format:

        COLLISION: [YES or NO]
        DETAILS: (in Korean -- if YES, name the approximate waypoint index or
        segment where the line crosses a wall, e.g. "waypoint 7과 8 사이 구간이
        벽을 관통함"; if NO, briefly confirm the path stays clear of all walls)
        """

        response_text = self.generate_response(prompt, image_path=image_path)
        return Message(self.name, response_text, "verification")