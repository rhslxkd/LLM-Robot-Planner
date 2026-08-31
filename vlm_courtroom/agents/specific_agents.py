import json
from typing import List, Dict, Any
from vlm_courtroom.agents.base_agent import VLMAgent, Message

# Prosecutor/Judge가 공유하는 물리 제약 블록 (Defense 폐지로 2개 agent만 참조).
# 씬 파라미터(clearance, margin 등)가 바뀌면 여기 한 곳만 고치면 된다.
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
             minimum-passable-gap number is the 0.8m figure in Section 2 below --
             use that number exclusively, never recompute your own.
        2. **Safety Protocol**:
           - Empirically measured via DIAL-MPC corridor-width stress test (centered
             straight-line path, Unitree Go2 trot gait): foot lateral excursion from
             the path centerline reaches up to ~0.28m per foot under normal gait.
             Degraded-but-stable behavior (feet brushing the boundary, no fall)
             observed down to a 0.40-0.55m gap. Catastrophic failure (loss of
             balance) confirmed at a 0.35m gap.
           - A separate full-trajectory DIAL-MPC run (oracle_scene_R001, 2026-08-25)
             additionally showed that a real corridor with clearance_m in the
             0.63-0.98m range -- which passed the OLD 0.6m threshold -- still
             caused the robot to fall while executing a sharp direction change
             inside that corridor. A tight gap combined with a turn is harder
             than a tight gap alone.
           - Because of that empirical failure, the passability threshold has
             been raised: if a gap between obstacles is narrower than **0.8m**,
             it is considered UNPASSABLE. Do not attempt to go through.
           - **CRITICAL: Do NOT stack your own additional safety margin on top of
             this 0.8m number.** It already includes the extra buffer described
             above. If a waypoint's clearance_m is >= 0.8m -- even if only
             marginally, e.g. 0.81m-0.85m -- you MUST treat that segment as
             PASSABLE without qualification. Do not describe a passing value as
             "barely enough," "risky," "too tight for comfort," or use closeness
             to the threshold (or the length of the segment) as grounds for
             STRUCTURALLY_INFEASIBLE or REJECTED. The ONLY valid clearance-based
             rejection criterion is clearance_m < 0.8m. Nothing else about a
             passing value is grounds for rejection.
        3. **Locomotion Constraints**:
           - Sequential Waypoint Distance (Step Length):
             - MIN: 0.4m (to prevent gait instability)
             - MAX: 1.0m (to prevent excessive acceleration)
             - RECOMMENDED: 0.6m - 0.7m
           - Turning Radius: Avoid sharp 90-degree turns. Use smooth arcs with a
             radius of at least **0.5m**. Pay particular attention to any
             waypoint where a sharp turn AND a clearance_m close to 0.8m occur
             together -- that specific combination is the observed R001 failure
             mode above, and is the highest-priority case to fix.
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
             >= 0.8m, TRUST THE NUMBER, not the prose. A clearance_m >= 0.8m at a
             specific (x,y) is a direct ray-cast measurement proving that point is
             in open space with at least 0.8m of room -- it cannot simultaneously
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
    """씬(장애물/통로 구조) 서술 전담. 좌표 생성/수정은 하지 않는다.
    2026-08-28: 입력 이미지는 반드시 경로가 안 그려진 원본 씬 이미지여야 한다."""

    def __init__(self, name="CoordinateAgent", **kwargs):
        super().__init__(name, "Scene Describer", model_role="COORDINATE", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Describing raw scene (no coordinate math)...")
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
        IMPORTANT: The image you are given shows ONLY the static scene (obstacles
        and walls, top-down view). It does NOT have any path, line, or waypoint
        markers drawn on it -- any path shape must come from the numeric
        coordinates below, not from something you see drawn in the image.

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
        the "## Coordinates" section below -- your role is scene description only,
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
    """좌표 수정 전담. Coordinate의 씬 서술 + Neural A* 좌표 + (재시도 시) 결정론적
    위반/제안 리포트를 보고 waypoint를 직접 고친다."""

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
        1. Check every waypoint's "clearance_m" (>= 0.8m required). If a waypoint
           already includes "suggested_x"/"suggested_y" fields in the input data,
           that means it violates the safety margin (either total corridor width
           OR is biased too close to one side) and a deterministically ray-cast
           -computed safe alternative (the corridor's true center at that point)
           is already provided -- ADOPT THAT EXACT VALUE for this waypoint. Do not
           compute your own direction/magnitude for it; your own visual guess is
           not more accurate than the ray-casting that produced this number.
           If "near_wall_m"/"far_wall_m" are also present, they tell you how close
           this waypoint currently is to the nearest wall on each side of the path
           -- use them ONLY to explain WHY the correction is needed in your
           reasoning text (e.g., "this point sits only {{near_wall_m}}m from one
           wall while the other side has {{far_wall_m}}m free"). Do not use them to
           compute your own alternative coordinate; suggested_x/suggested_y is
           already that computation.
        2. For any OTHER waypoint that still looks unsafe or poorly positioned
           and does NOT have a suggested_x/suggested_y given, you are AUTHORIZED
           to propose a corrected (x, y) yourself -- reason from the given numeric
           data and neighboring waypoints' coordinates, move it toward the
           corridor's estimated center, keep corrections small/local and
           consistent with neighboring waypoints (no large jumps).
        3. If a segment is fundamentally impassable (clearance_m < 0.8m across a
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


class JudgeAgent(VLMAgent):
    """최종 판단 + 시각화 트리거. Defense가 없으므로 Prosecutor의 근거(인용한 clearance_m,
    거리 계산 등)를 원본 데이터와 대조해 직접 재검증하는 역할까지 겸한다."""

    def __init__(self, name="JudgeAgent", **kwargs):
        super().__init__(name, "Judge", model_role="JUDGE", **kwargs)

    def process(self, context: Dict[str, Any]) -> Message:
        print(f"[{self.name}] Deliberating...")
        proposal = context.get('original_proposal', '')
        prosecution = context.get('prosecution_argument', '')
        prompt = f"""
        You are the Chief Judge.
        Evaluate the Original Path: {proposal}
        Prosecutor's Argument (scene-grounded critique + proposed corrections): {prosecution}
        {ROBOT_PHYSICAL_CONSTRAINTS}

        There is no Defense counter-argument in this court -- you are the sole
        check on the Prosecutor's reasoning. Before adopting any Prosecutor
        correction, verify their cited numbers (clearance_m, step distances,
        VERDICT category) against the original waypoint data yourself. If the
        Prosecutor over-corrected a waypoint that was already safe (clearance_m
        >= 0.8m and not meaningfully wall-hugging), restore the original value
        for that point. If the Prosecutor missed a real violation, fix it
        yourself using the same rules (small, local correction; consistent with
        neighboring waypoints).

        Decide on the FINAL path: adopt the Prosecutor's corrected coordinates,
        the original coordinates, or a merge of specific point corrections --
        whichever set of coordinates you determine is actually safe and correct
        per the physical constraints above.
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
