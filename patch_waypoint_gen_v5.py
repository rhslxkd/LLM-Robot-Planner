import pathlib

path = pathlib.Path("waypoint_generator.py")
content = path.read_text()

old = '''        for i in range(1, len(coords) - 1):
            px, py = points_px[i]
            c_m = clearance_m_at(dist_field, px, py, ppm)
            if c_m < soft_radius_m:
                target = find_nearest_safe_point(dist_field, px, py, soft_px, search_radius_px=max_shift_px)
                degraded = False
                if target is None:
                    target = find_nearest_safe_point(dist_field, px, py, hard_px, search_radius_px=max_shift_px)
                    degraded = True
                if target is None:
                    log.append(f"[iter {it}] idx {i}: hard floor({hard_radius_m}m)도 만족 불가 -> 구조적 병목")
                    continue
                moved = ((target[0] - px) ** 2 + (target[1] - py) ** 2) ** 0.5
                if moved < STUCK_EPS_PX:
                    continue
                wx, wy = to_world(*target)
                coords[i]["x"], coords[i]["y"] = round(wx, 3), round(wy, 3)
                tag = "hard floor만 만족" if degraded else "soft margin 만족"
                log.append(f"[iter {it}] idx {i}: clearance {c_m:.2f}m -> ({wx:.2f},{wy:.2f}) 이동 ({tag})")
                changed = True
        if changed:
            continue'''

n = content.count(old)
assert n == 1, f"old block occurrence count = {n} (expected 1)"

new = '''        # 좁은 반경(CORRECTION_MAX_SHIFT_M)으로 못 찾으면, 코너 탈출 규모의 넓은
        # 반경(CORNER_BEND_SEARCH_M)으로 재시도한다. 안 그러면 idx는 영원히
        # "hard floor도 만족 불가"로 방치되고, corner-cut 굴절점 삽입은 이 idx를
        # 직접 못 옮긴 채 옆에 점만 끼워넣으려다 계속 실패한다.
        # 넓은 반경 후보는 이웃 waypoint와 최소 보폭(min_step_px) 이상 떨어진
        # 것만 채택해 다음 iteration에 "보폭 미달"로 제거->재삽입되는 오실레이션을 막는다.
        for i in range(1, len(coords) - 1):
            px, py = points_px[i]
            c_m = clearance_m_at(dist_field, px, py, ppm)
            if c_m < soft_radius_m:
                target = find_nearest_safe_point(dist_field, px, py, soft_px, search_radius_px=max_shift_px)
                degraded = False
                widened = False
                if target is None:
                    target = find_nearest_safe_point(dist_field, px, py, hard_px, search_radius_px=max_shift_px)
                    degraded = True
                if target is None:
                    prev_px = points_px[i - 1]
                    next_px = points_px[i + 1] if i + 1 < len(points_px) else None
                    for r_px, is_hard in ((soft_px, False), (hard_px, True)):
                        cand = find_nearest_safe_point(dist_field, px, py, r_px, search_radius_px=corner_bend_px)
                        if cand is None:
                            continue
                        d_prev = ((cand[0] - prev_px[0]) ** 2 + (cand[1] - prev_px[1]) ** 2) ** 0.5
                        d_next = (((cand[0] - next_px[0]) ** 2 + (cand[1] - next_px[1]) ** 2) ** 0.5
                                  if next_px is not None else float("inf"))
                        if d_prev >= min_step_px and d_next >= min_step_px:
                            target = cand
                            degraded = is_hard
                            widened = True
                            break
                if target is None:
                    log.append(f"[iter {it}] idx {i}: hard floor({hard_radius_m}m)도 만족 불가 (광역탐색 {CORNER_BEND_SEARCH_M}m 포함) -> 구조적 병목")
                    continue
                moved = ((target[0] - px) ** 2 + (target[1] - py) ** 2) ** 0.5
                if moved < STUCK_EPS_PX:
                    continue
                wx, wy = to_world(*target)
                coords[i]["x"], coords[i]["y"] = round(wx, 3), round(wy, 3)
                tag = "hard floor만 만족" if degraded else "soft margin 만족"
                if widened:
                    tag += f" / {CORNER_BEND_SEARCH_M}m 광역탐색"
                log.append(f"[iter {it}] idx {i}: clearance {c_m:.2f}m -> ({wx:.2f},{wy:.2f}) 이동 ({tag})")
                changed = True
        if changed:
            continue'''

content = content.replace(old, new)
path.write_text(content)
print("OK: waypoint_generator.py 1단계 광역탐색 폴백 패치 적용 완료")
