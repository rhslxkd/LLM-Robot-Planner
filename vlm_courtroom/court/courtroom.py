from vlm_courtroom.agents.specific_agents import (
    CoordinateAgent, ProsecutorAgent, JudgeAgent, VerifierAgent
)
from vlm_courtroom.agents.base_agent import Message
import os
import re
import numpy as np

MAX_VERIFY_RETRIES = 3  # Prosecutor->Judge->시각화->Verifier 루프 최대 반복 횟수 (Defense 폐지)
MIN_CLEARANCE_M = 0.8   # ROBOT_PHYSICAL_CONSTRAINTS와 동일한 임계값. 2026-08-28 0.6->0.8 상향:
                         # oracle_scene_R001 실측 DIAL-MPC에서 clearance_m 0.63~0.98m 구간(구
                         # 기준으로는 통과)에서 로봇이 급회전 중 실제로 넘어지는 것을 확인함.
MIN_WALL_DIST_M = 0.4   # 2026-08-31: 통로 전체 폭(clearance_m)과 별개로, 한쪽 벽에
                         # 치우쳐 지나가는 걸 막는 최소 편측 이격거리 기준.
GOAL_MARKER_EXCLUDE_PX = 20  # goal 마커 오탐 방지용 제외 반경 (실측: goal 근처에 순수
                              # (255,0,0) 클러스터가 있고 이는 실제 벽 렌더링 색과 다름)


def _build_red_mask(image_path, exclude_center_px=None, exclude_radius_px=0):
    """이미지에서 붉은 장애물 픽셀 마스크 생성. exclude_center_px가 주어지면 그 주변
    exclude_radius_px 반경은 마스크에서 제외한다 (goal 마커가 벽과 같은 색 계열이라
    오탐을 일으키는 문제를 막기 위함).
    CRITICAL: image_path는 반드시 "원본" 씬 이미지(경로가 안 그려진, 장애물만 있는 이미지)
    여야 한다. Neural A*가 자체적으로 그린 경로 오버레이를 넘기면, 그 경로 선이 빨간
    계열 색으로 그려져 있을 경우 이 마스크가 그 선까지 "벽"으로 오탐할 수 있다."""
    import matplotlib.image as mpimg
    arr = mpimg.imread(image_path)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    mask = (r > 180) & (g < 120) & (b < 120)
    if exclude_center_px is not None and exclude_radius_px > 0:
        h, w = mask.shape
        cx, cy = exclude_center_px
        yy, xx = np.ogrid[:h, :w]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= exclude_radius_px ** 2
        mask = mask & ~disk
    return mask


def _cast_ray(x0, y0, dx, dy, red_mask):
    h, w = red_mask.shape
    max_range = max(h, w)
    for r in range(1, max_range):
        xi, yi = int(round(x0 + dx * r)), int(round(y0 + dy * r))
        if not (0 <= xi < w and 0 <= yi < h):
            return r
        if red_mask[yi, xi]:
            return r
    return max_range


def _perp_dir(points_px, idx):
    if idx == 0:
        tx, ty = points_px[1][0] - points_px[0][0], points_px[1][1] - points_px[0][1]
    elif idx == len(points_px) - 1:
        tx, ty = points_px[idx][0] - points_px[idx - 1][0], points_px[idx][1] - points_px[idx - 1][1]
    else:
        tx, ty = points_px[idx + 1][0] - points_px[idx - 1][0], points_px[idx + 1][1] - points_px[idx - 1][1]
    n = (tx ** 2 + ty ** 2) ** 0.5
    if n < 1e-6:
        return 0.0, 0.0
    return -ty / n, tx / n


def verify_clearance_deterministic(image_path, coordinates, robot_pos, scale,
                                    min_clearance_m=MIN_CLEARANCE_M, min_wall_dist_m=MIN_WALL_DIST_M):
    """Judge가 확정한 최종 좌표를 실제 이미지 픽셀 기준으로 재검증. VLM이 눈대중으로
    거리를 추정하게 하지 않고, Neural A* 단계와 동일한 ray-casting을 그대로 재실행해서
    각 waypoint의 실제 clearance_m을 다시 계산한다. 2026-08-31: 통로 전체 폭(width_m)뿐
    아니라 한쪽 벽까지의 최소거리(min(d1,d2))도 확인 -- 통로가 넓어도 한쪽에 치우쳐
    지나가는 경우를 잡는다. 위반 시 중앙 정렬 suggested_x/y + 참고용 near/far 거리를 반환.
    마지막 waypoint(goal)는 goal 마커 오탐을 피하기 위해 red_mask에서 그 주변을 제외한다.
    반환: (모두 통과했는가: bool, [{"idx","width_m","near_wall_m","far_wall_m",
                                    "suggested_x","suggested_y"}, ...])"""
    if not robot_pos or not scale or not coordinates:
        return True, []
    rx, ry = robot_pos
    points_px = [(rx + c['x'] * scale, ry - c['y'] * scale) for c in coordinates]
    goal_px = points_px[-1]
    red_mask = _build_red_mask(image_path, exclude_center_px=goal_px, exclude_radius_px=GOAL_MARKER_EXCLUDE_PX)
    violations = []
    for idx, (px, py) in enumerate(points_px):
        perp_x, perp_y = _perp_dir(points_px, idx)
        if perp_x == 0 and perp_y == 0:
            continue
        d1 = _cast_ray(px, py, perp_x, perp_y, red_mask)
        d2 = _cast_ray(px, py, -perp_x, -perp_y, red_mask)
        width_m = (d1 + d2) / scale
        min_side_m = min(d1, d2) / scale
        if width_m < min_clearance_m or min_side_m < min_wall_dist_m:
            shift_px = (d1 - d2) / 2.0
            cand_x, cand_y = px + perp_x * shift_px, py + perp_y * shift_px
            violations.append({
                "idx": idx,
                "width_m": round(width_m, 2),
                "near_wall_m": round(min(d1, d2) / scale, 2),
                "far_wall_m": round(max(d1, d2) / scale, 2),
                "suggested_x": round((cand_x - rx) / scale, 2),
                "suggested_y": round((ry - cand_y) / scale, 2),
            })
    return (len(violations) == 0), violations


def _segment_is_clear(p0, p1, red_mask):
    h, w = red_mask.shape
    dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    n = max(2, int(dist))
    for k in range(n + 1):
        t = k / n
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and red_mask[yi, xi]:
            return False
    return True


def verify_segments_deterministic(image_path, coordinates, robot_pos, scale):
    """연속된 waypoint 사이 직선이 실제로 벽을 관통/스치는지 결정론적으로 확인.
    개별 점의 clearance_m은 괜찮아도 그 사이 직선이 코너를 스칠 수 있음 (line-of-sight
    문제 -- Neural A* 단계에서 겪은 것과 동일한 종류). Verifier의 시각 판단을 구조화된
    정확한 인덱스로 보완한다. goal 근처는 마커 오탐 방지를 위해 제외한다.
    반환: (모두 통과했는가: bool, [(i, i+1), ...] 충돌하는 segment의 (시작,끝) 인덱스 목록)"""
    if not robot_pos or not scale or not coordinates or len(coordinates) < 2:
        return True, []
    rx, ry = robot_pos
    points_px = [(rx + c['x'] * scale, ry - c['y'] * scale) for c in coordinates]
    goal_px = points_px[-1]
    red_mask = _build_red_mask(image_path, exclude_center_px=goal_px, exclude_radius_px=GOAL_MARKER_EXCLUDE_PX)
    bad_segments = []
    for i in range(len(points_px) - 1):
        if not _segment_is_clear(points_px[i], points_px[i + 1], red_mask):
            bad_segments.append((i, i + 1))
    return (len(bad_segments) == 0), bad_segments


def compute_correction_suggestions(image_path, coordinates, robot_pos, scale, indices,
                                    center_max_shift_m=0.3):
    """지정된 waypoint 인덱스들에 대해, 그 지점의 통로 단면 중앙으로 이동시킨 안전한
    대안 좌표를 ray-casting으로 정확히 계산 (Prosecutor가 방향/거리를 눈대중으로
    추측하지 않고 이 값을 그대로 채택할 수 있게 함). 이동 후 이웃 점과의 직선도
    안전한지 확인하고, 안전할 때만 제안에 포함. (기존 waypoint를 "옮기는" 제안만
    다룸 -- 굴절점을 새로 "삽입"해야 하는 코너-컷 케이스는 compute_bend_suggestions 참고.)"""
    if not robot_pos or not scale or not coordinates:
        return []
    rx, ry = robot_pos
    points_px = [(rx + c['x'] * scale, ry - c['y'] * scale) for c in coordinates]
    goal_px = points_px[-1]
    red_mask = _build_red_mask(image_path, exclude_center_px=goal_px, exclude_radius_px=GOAL_MARKER_EXCLUDE_PX)
    n = len(points_px)
    suggestions = []
    cap = center_max_shift_m * scale
    for idx in set(indices):
        if idx <= 0 or idx >= n - 1:
            continue  # 시작/끝점은 건드리지 않음
        px, py = points_px[idx]
        perp_x, perp_y = _perp_dir(points_px, idx)
        if perp_x == 0 and perp_y == 0:
            continue
        d1 = _cast_ray(px, py, perp_x, perp_y, red_mask)
        d2 = _cast_ray(px, py, -perp_x, -perp_y, red_mask)
        shift_px = max(-cap, min(cap, (d1 - d2) / 2.0))
        cand_x, cand_y = px + perp_x * shift_px, py + perp_y * shift_px
        if _segment_is_clear(points_px[idx - 1], (cand_x, cand_y), red_mask) and \
           _segment_is_clear((cand_x, cand_y), points_px[idx + 1], red_mask):
            suggestions.append({
                "idx": idx,
                "current_x": coordinates[idx]['x'], "current_y": coordinates[idx]['y'],
                "suggested_x": round((cand_x - rx) / scale, 2),
                "suggested_y": round((ry - cand_y) / scale, 2),
            })
    return suggestions


def compute_bend_suggestions(image_path, coordinates, robot_pos, scale, bad_segments,
                              max_shift_m=0.6, num_shift_steps=6):
    """point-move 제안으로 안 풀리는 코너-컷 segment(예: Prosecutor가 중간 waypoint를
    삭제해서 생긴 직선이 모서리를 스치는 경우)에 대해, 두 waypoint 사이에 새로 삽입할
    굴절점(bend point)을 결정론적으로 탐색한다. segment 중점에서 시작해 그 segment에
    수직인 방향으로 조금씩 밀어보며(최대 max_shift_m까지 num_shift_steps 단계),
    양쪽 서브세그먼트가 모두 안전해지는 첫 후보를 채택한다."""
    if not robot_pos or not scale or not coordinates:
        return []
    rx, ry = robot_pos
    points_px = [(rx + c['x'] * scale, ry - c['y'] * scale) for c in coordinates]
    goal_px = points_px[-1]
    red_mask = _build_red_mask(image_path, exclude_center_px=goal_px, exclude_radius_px=GOAL_MARKER_EXCLUDE_PX)
    suggestions = []
    cap = max_shift_m * scale
    for (i, j) in bad_segments:
        p0, p1 = points_px[i], points_px[j]
        mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = (dx ** 2 + dy ** 2) ** 0.5
        if n < 1e-6:
            continue
        perp_x, perp_y = -dy / n, dx / n
        found = None
        for step in range(1, num_shift_steps + 1):
            shift = cap * step / num_shift_steps
            for sign in (1, -1):
                cx, cy = mx + perp_x * shift * sign, my + perp_y * shift * sign
                if _segment_is_clear(p0, (cx, cy), red_mask) and \
                   _segment_is_clear((cx, cy), p1, red_mask):
                    found = (cx, cy)
                    break
            if found:
                break
        if found:
            cx, cy = found
            suggestions.append({
                "insert_after_idx": i,
                "suggested_x": round((cx - rx) / scale, 2),
                "suggested_y": round((ry - cy) / scale, 2),
            })
    return suggestions


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
        (2026-08-28: Defense Attorney 폐지. 모든 transcript에서 예외 없이 Prosecutor에
        동의만 해서 실질적 교차검증 기능이 없었음 -- 대신 Judge가 Prosecutor의 근거를
        직접 재검증하는 역할까지 겸함.)
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
        self.judge_agent = JudgeAgent(**agent_kwargs)
        self.verifier_agent = VerifierAgent(**agent_kwargs)
        print("Agents initialized.")

    def run_case(self, image_description: str, image_path: str = None, robot_pos: tuple = None,
                 scale: float = None, scene_name: str = None, coordinate_proposal: list = None,
                 variant: str = None):
        """
        image_path: 반드시 Neural A* 입력으로 쓰인 "원본" 씬 이미지(장애물만 있고 경로가
                    그려지지 않은 이미지)를 넘길 것 -- Neural A*가 자체적으로 그린 경로
                    오버레이 이미지를 넘기면 안 됨. (1) 그 경로 선이 벽과 비슷한 색이면
                    _build_red_mask가 벽으로 오탐할 수 있고, (2) CoordinateAgent/Judge/
                    Verifier가 Neural A*의 원래 경로 모양에 시각적으로 낚여서 좌표 숫자를
                    독립적으로 판단하지 못하게 된다. (2026-08-28 재설계 핵심 변경.)
        coordinate_proposal: run_neural_astar_step.py가 미리 계산한 좌표+clearance_m 리스트
                              (data/<scene>/neural_astar/coordinate_proposal.json).
        Prosecutor가 좌표 수정을 제안하면 Judge가 최종 판단 및 시각화를 하고, 그 결과를
        (1) VerifierAgent의 시각 판단, (2) 결정론적 clearance 재계산, (3) 결정론적
        segment(직선) 충돌 재계산, (4) Judge 자신의 REJECTED/STRUCTURALLY_INFEASIBLE
        판결 여부까지 4중으로 재검사한다. 문제가 있으면 (segment 충돌의 경우 ray-casting
        으로 계산된 안전한 대안 좌표 -- 기존 점 이동안 + 굴절점 삽입안 -- 까지 포함해서)
        Prosecutor에게 피드백을 주고 최대 MAX_VERIFY_RETRIES번 재시도, 그래도 안 되면
        최종 REJECTED로 마킹.
        """
        print("\n=== 🏛️ VLM Courtroom Simulation Started 🏛️ ===\n")
        transcript_sections = []

        # 1. Coordinate Agent (원본 씬 서술 -- 재시도해도 바뀌지 않으므로 한 번만)
        print("--- [Step 1] Coordinate Agent (Scene description) ---")
        coord_msg = self.coordinate_agent.process({
            'image_description': image_description,
            'image_path': image_path,
            'coordinate_proposal': coordinate_proposal
        })
        print(f"📍 Proposal:\n{coord_msg.content}\n")
        transcript_sections.append(("Coordinate", coord_msg.content))

        retry_feedback = None
        final_coords = []
        verdict_image_path = None
        judge_msg = None
        verified_clear = False

        for attempt in range(1, MAX_VERIFY_RETRIES + 1):
            print(f"--- Attempt {attempt}/{MAX_VERIFY_RETRIES} ---")

            # 2. Prosecutor Agent (좌표 수정 담당)
            print("--- [Step 2] Prosecutor Agent (Correction) ---")
            pros_msg = self.prosecutor_agent.process({
                'last_message_content': coord_msg.content,
                'retry_feedback': retry_feedback
            })
            print(f"⚖️ Prosecution:\n{pros_msg.content}\n")
            transcript_sections.append((f"Prosecutor (attempt {attempt})", pros_msg.content))

            # 3. Judge Agent (최종 판단 -- Defense 없으므로 Prosecutor 근거를 Judge가 직접 재검증)
            print("--- [Step 3] Judge Agent (Final Verdict) ---")
            judge_msg = self.judge_agent.process({
                'original_proposal': coord_msg.content,
                'prosecution_argument': pros_msg.content,
            })
            print(f"👨‍⚖️ Verdict:\n{judge_msg.content}\n")
            transcript_sections.append((f"Judge (attempt {attempt})", judge_msg.content))

            # 4. 시각화 (Judge 최종 경로를 원본 이미지 위에 그림)
            if not image_path:
                final_coords = []
                break
            final_coords, verdict_image_path = self.visualize_path(
                image_path, judge_msg.content, robot_pos, scale, scene_name, variant
            )
            if not final_coords or not verdict_image_path:
                print("⚠️ Judge 응답에서 좌표를 파싱하지 못함 -- 재시도해도 의미 없어 종료")
                break

            # 5. Verifier Agent (시각적으로 "선이 벽에 닿는가"만 판단 -- 거리 추정은 안 함)
            print("--- [Step 5] Verifier Agent (Visual line-collision check) ---")
            verify_msg = self.verifier_agent.process({'image_path': verdict_image_path})
            print(f"🔍 Verification:\n{verify_msg.content}\n")
            transcript_sections.append((f"Verifier (attempt {attempt})", verify_msg.content))
            m = re.search(r'COLLISION:\s*(YES|NO)', verify_msg.content, re.IGNORECASE)
            line_collision = (m.group(1).upper() == "YES") if m else True  # 파싱 실패시 보수적으로 충돌 간주
            # 5.5 결정론적 clearance 재검증 (VLM 눈대중 아님, Neural A*와 동일한 ray-casting)
            clear_ok, violations = verify_clearance_deterministic(image_path, final_coords, robot_pos, scale)
            det_report = ""
            if not clear_ok:
                det_report = (
                f"[결정론적 재검증] 다음 waypoint는 안전기준 위반입니다 (통로 폭 < {MIN_CLEARANCE_M}m "
                f"또는 한쪽 벽까지 거리 < {MIN_WALL_DIST_M}m). 방향/거리를 추측하지 말고 "
                f"suggested_x/suggested_y를 그대로 채택하세요: {violations}"
            )
                print(f"📐 {det_report}\n")
                transcript_sections.append((f"Deterministic clearance check (attempt {attempt})", det_report))

            # 5.6 결정론적 segment(직선 구간) 충돌 재검증
            seg_ok, bad_segments = verify_segments_deterministic(image_path, final_coords, robot_pos, scale)
            seg_report = ""
            if not seg_ok:
                bad_indices = sorted({i for pair in bad_segments for i in pair})
                move_suggestions = compute_correction_suggestions(image_path, final_coords, robot_pos, scale, bad_indices)
                bend_suggestions = compute_bend_suggestions(image_path, final_coords, robot_pos, scale, bad_segments)
                seg_report = (
                    f"[결정론적 segment 재검증] 다음 구간이 실제로 벽을 관통/스칩니다: {bad_segments}. "
                    f"두 종류의 제안이 있습니다 -- 방향/거리를 추측하지 말고 그대로 채택하세요:\n"
                    f"(a) 기존 waypoint 이동 제안 (idx의 x,y를 suggested_x,suggested_y로 교체): {move_suggestions}\n"
                    f"(b) 새 굴절점 삽입 제안 (insert_after_idx 뒤에 suggested_x,suggested_y를 새 waypoint로 삽입, "
                    f"기존 점들은 그대로 유지): {bend_suggestions}"
                )
                print(f"📏 {seg_report}\n")
                transcript_sections.append((f"Deterministic segment check (attempt {attempt})", seg_report))

            # 5.7 Judge 자신의 판결도 확인
            judge_rejected = bool(re.search(r'\b(REJECTED|STRUCTURALLY_INFEASIBLE)\b', judge_msg.content, re.IGNORECASE))

            # 2026-08-31: line_collision(VLM 시각판단) 단독으로는 재시도를 강제하지 않음.
            # 이 환경은 장애물이 전부 red_mask 기반이라 Verifier가 ray-casting보다 더 많은
            # 정보를 가질 수 없음 -- R001 실측에서 clearance_m=2.99m인 지점을 line_collision=True로
            # 오탐하여 Prosecutor가 멀쩡한 waypoint를 삭제하는 부작용 확인.
            collision = (not clear_ok) or (not seg_ok) or judge_rejected
            if not collision:
                print(f"✅ Attempt {attempt}: 시각+결정론적 검증+Judge 판결 모두 통과")
                verified_clear = True
                break
            else:
                print(f"❌ Attempt {attempt}: 검증 실패 (line_collision={line_collision}, "
                      f"clearance_ok={clear_ok}, segment_ok={seg_ok}, judge_rejected={judge_rejected})")
                feedback_parts = []
                if not seg_ok:
                    feedback_parts.append(seg_report)
                # elif line_collision:  # 2026-08-31: line_collision은 이제 판정에 안 쓰므로 피드백에서도 제외 (Verifier는 로깅 전용)
                #     feedback_parts.append(verify_msg.content)
                if not clear_ok:
                    feedback_parts.append(det_report)
                if judge_rejected:
                    feedback_parts.append(
                        "[Judge 자체 판결] Judge가 이 경로를 REJECTED/STRUCTURALLY_INFEASIBLE로 "
                        "판결하고 수정된 좌표를 내지 못했습니다. 판결 이유를 실제로 해결하는 구체적인 "
                        "좌표 수정을 시도하세요 (단순히 '불가능하다'고 다시 반복하지 마세요):\n"
                        + judge_msg.content[:1000]
                    )
                retry_feedback = "\n".join(feedback_parts)
                if attempt == MAX_VERIFY_RETRIES:
                    print(f"⚠️ 최대 재시도({MAX_VERIFY_RETRIES}회) 초과 -- 최종 REJECTED 처리")
                    judge_msg = Message(
                        self.judge_agent.name,
                        judge_msg.content + (
                            f"\n\n[SYSTEM] 시각 검증 {MAX_VERIFY_RETRIES}회 재시도 후에도 "
                            f"경로가 벽과 충돌하는 것으로 확인됨. 최종 REJECTED 처리."
                        ),
                        "verdict"
                    )

        # 전체 대화 로그 저장 (모든 시도 포함)
        if scene_name:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            repo_root = os.path.dirname(project_root)
            base_dir = os.path.join(repo_root, "data", scene_name)
            log_dir = os.path.join(base_dir, variant) if variant else base_dir
            os.makedirs(log_dir, exist_ok=True)
            transcript_path = os.path.join(log_dir, "transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                for title, content in transcript_sections:
                    f.write(f"=== {title} ===\n{content}\n\n")
                f.write(f"=== Final: verified_clear={verified_clear} ===\n")
            print(f"📝 Saved full transcript to: {transcript_path}")

        print("=== 🏛️ Case Closed 🏛️ ===")
        return judge_msg, final_coords

    def visualize_path(self, image_path: str, verdict_text: str, robot_pos: tuple = None,
                        scale: float = None, scene_name: str = None, variant: str = None):
        """반환값: (coordinates, saved_image_path). 실패 시 ([], None)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            import json
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', verdict_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]', verdict_text, re.DOTALL)
            if not json_match:
                print(f"⚠️ Could not find coordinate JSON in verdict. Raw verdict:\n{verdict_text}")
                return [], None
            json_str = json_match.group(1) if json_match.groups() else json_match.group(0)
            json_str = re.sub(r"'([a-zA-Z0-9_]+)'\s*:", r'"\1":', json_str)
            json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
            try:
                coordinates = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parsing failed: {e}")
                print(f"Raw extracted string: {json_str}")
                return [], None
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
            return coordinates, output_path_project
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return [], None
