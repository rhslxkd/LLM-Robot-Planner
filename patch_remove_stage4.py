with open("run_neural_astar_step.py") as f:
    content = f.read()

# 1. docstring 정리 (courtroom 언급 제거, 파일명 오탈자 수정)
old_doc = '''"""
run_neural_astar_step.py
neural-astar env 전용. run_random_batch.py(vlm_court env)에서
`conda run -n neural-astar`로 서브프로세스 호출됨.

Neural A* raw grid path -> world 좌표 변환 -> line-of-sight 경로 단순화
(visibility-aware shortcutting) -> 목표 waypoint 개수로 보정 -> 각 점의
실제 통로 폭(clearance_m)을 perpendicular ray-casting으로 결정론적 계산
-> courtroom에 넘길 coordinate_proposal.json + 오버레이 이미지 저장.

사용: python run_neural_astar_step.py --scene oracle_scene_R000 --goal-x 5.3 --goal-y -1.2
성공 시 data/<scene>/neural_astar/{overlay_solo.png, coordinate_proposal.json, path_info.json} 생성, exit 0.
실패(목표 미도달) 시 exit 1.
"""'''

new_doc = '''"""
run_neural_astar_step.py
neural-astar env 전용. run_random_batch_v2.py(vlm_court env)에서
`conda run -n neural-astar`로 서브프로세스 호출됨.

Neural A* raw grid path -> world 좌표 변환 -> line-of-sight 경로 단순화
(visibility-aware shortcutting) -> 목표 waypoint 개수로 보정
-> waypoint_generator.py에 넘길 coordinate_proposal.json(초기 제안 경로,
안전 마진 검증은 waypoint_generator.py가 별도로 수행) + 오버레이 이미지 저장.

사용: python run_neural_astar_step.py --scene oracle_scene_R000 --goal-x 5.3 --goal-y -1.2
성공 시 data/<scene>/neural_astar/{overlay_solo.png, coordinate_proposal.json, path_info.json} 생성, exit 0.
실패(목표 미도달) 시 exit 1.
"""'''

assert content.count(old_doc) == 1, f"old_doc occurrence = {content.count(old_doc)}"
content = content.replace(old_doc, new_doc)

# 2. 죽은 상수 제거
old_const = '''MIN_STEP_M = 0.4
MAX_STEP_M = 1.0
MIN_WALL_DIST_M = 0.4    # 2026-08-31: 통로 전체 폭(clearance_m)과 별개로,
                         # 한쪽 벽에 치우쳐 지나가는 걸 막는 최소 편측 이격거리 기준.
CORRECTION_MAX_SHIFT_M = 0.3  # 2026-08-31: 중앙 정렬 보정 시 이동량 상한. 이게 없으면 한쪽
                         # 벽이 아주 멀 때(반대쪽이 열린 공간) shift가 무한정 커져서 원래
                         # 위치에서 엉뚱하게 먼 곳으로 waypoint가 튀는 문제가 있었음 (실측 확인).'''

new_const = '''MIN_STEP_M = 0.4
MAX_STEP_M = 1.0'''

assert content.count(old_const) == 1, f"old_const occurrence = {content.count(old_const)}"
content = content.replace(old_const, new_const)

# 3. measure_corridor_width_m() 함수 전체 삭제 (Stage 4 comment 헤더부터 def main() 직전까지)
import re
pattern = re.compile(
    r"# ---------- Stage 4:.*?\ndef main\(\):",
    re.DOTALL
)
matches = pattern.findall(content)
assert len(matches) == 1, f"stage4 block matches = {len(matches)}"
content = pattern.sub("def main():", content)

# 4. main() 안의 Stage 4 사용부 -> 단순 x/y 저장으로 교체
old_usage = '''    # ---- Stage 4: 각 점의 실제 통로 폭(clearance_m), perpendicular ray-casting (원본 red_mask 사용) ----
    coordinates = []
    for idx, (px, py) in enumerate(wp_final):
        wx, wy = full_to_world(px, py)
        clearance, cx_px, cy_px, near_m, far_m = measure_corridor_width_m(wp_final, idx, red_mask, PPM)
        entry = {"x": round(wx, 2), "y": round(wy, 2), "clearance_m": clearance}
        if clearance < 0.8 or near_m < MIN_WALL_DIST_M:
            cwx, cwy = full_to_world(cx_px, cy_px)
            entry["suggested_x"] = round(cwx, 2)
            entry["suggested_y"] = round(cwy, 2)
            entry["near_wall_m"] = near_m
            entry["far_wall_m"] = far_m
        coordinates.append(entry)'''

new_usage = '''    # ---- 좌표 변환: world 좌표로만 저장. 안전 마진 검증/보정은
    # waypoint_generator.py의 EDT 기반 로직이 별도로 담당한다 (이 스크립트는
    # 초기 제안 경로만 만든다). ----
    coordinates = []
    for px, py in wp_final:
        wx, wy = full_to_world(px, py)
        coordinates.append({"x": round(wx, 2), "y": round(wy, 2)})'''

assert content.count(old_usage) == 1, f"old_usage occurrence = {content.count(old_usage)}"
content = content.replace(old_usage, new_usage)

with open("run_neural_astar_step.py", "w") as f:
    f.write(content)

print("패치 완료")
