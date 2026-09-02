import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
content = path.read_text()

anchor = 'if __name__ == "__main__":'
n = content.count(anchor)
assert n == 1, f"anchor occurrence count = {n} (expected 1)"

new_func = '''def _reachable_with_margin(room_x_min, room_x_max, room_half_y, boxes, start_xy, goal_xy,
                            hard_radius_m, grid_res_m=0.02):
    """boxes를 hard_radius_m 마진으로 침식시킨 free space에서 start->goal이 연결되는지 확인.
    단순 BFS 연결성만 보면 박스 사이 틈이 실제로는 로봇이 못 지나갈 만큼 좁아도 통과로
    잘못 판정할 수 있어서, EDT(waypoint_generator.py와 동일 방식)로 clearance를 계산한
    뒤 hard_radius_m 이상인 셀만 free로 취급한다."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt, label

    nx = max(2, int(round((room_x_max - room_x_min) / grid_res_m)) + 1)
    ny = max(2, int(round((2 * room_half_y) / grid_res_m)) + 1)
    occ = np.zeros((ny, nx), dtype=bool)

    def world_to_grid(x, y):
        gx = (x - room_x_min) / grid_res_m
        gy = (y + room_half_y) / grid_res_m
        return gx, gy

    def mark_rect(cx, cy, hx, hy):
        x0, y0 = world_to_grid(cx - hx, cy - hy)
        x1, y1 = world_to_grid(cx + hx, cy + hy)
        xi0, xi1 = int(np.floor(min(x0, x1))), int(np.ceil(max(x0, x1)))
        yi0, yi1 = int(np.floor(min(y0, y1))), int(np.ceil(max(y0, y1)))
        xi0, yi0 = max(0, xi0), max(0, yi0)
        xi1, yi1 = min(nx, xi1), min(ny, yi1)
        if xi1 > xi0 and yi1 > yi0:
            occ[yi0:yi1, xi0:xi1] = True

    wall_t = 0.15  # 외벽 두께(기존 TEMPLATE 0.075*2 근사)
    mark_rect((room_x_min + room_x_max) / 2, -room_half_y, (room_x_max - room_x_min) / 2 + wall_t, wall_t)
    mark_rect((room_x_min + room_x_max) / 2, room_half_y, (room_x_max - room_x_min) / 2 + wall_t, wall_t)
    mark_rect(room_x_min, 0, wall_t, room_half_y + wall_t)
    mark_rect(room_x_max, 0, wall_t, room_half_y + wall_t)
    for (cx, cy, hx, hy) in boxes:
        mark_rect(cx, cy, hx, hy)

    dist_px = distance_transform_edt(~occ)
    dist_m = dist_px * grid_res_m
    free_safe = dist_m >= hard_radius_m

    sx, sy = world_to_grid(*start_xy)
    gx, gy = world_to_grid(*goal_xy)
    sxi, syi = int(round(sx)), int(round(sy))
    gxi, gyi = int(round(gx)), int(round(gy))
    if not (0 <= sxi < nx and 0 <= syi < ny and 0 <= gxi < nx and 0 <= gyi < ny):
        return False
    if not free_safe[syi, sxi] or not free_safe[gyi, gxi]:
        return False

    labeled, _ = label(free_safe)
    return labeled[syi, sxi] == labeled[gyi, gxi] != 0


def generate_boxes(seed, name, goal_x_range=(4.0, 8.0), room_half_y=3.0,
                    n_boxes_range=(3, 7), box_half_size_range=(0.15, 1.0),
                    room_margin=0.5, min_dist_from_start=3.0, goal_y_margin=1.0,
                    hard_radius_m=0.4, start_clear_r=0.6, goal_clear_r=0.6,
                    max_layout_attempts=300):
    """Scene E 슬라롬 대신, 방 안에 크기가 제각각인 박스 N개를 랜덤 배치하는 레이아웃.
    구조적으로 통과 가능성이 보장되지 않으므로, 배치 직후 _reachable_with_margin()으로
    hard_radius_m(로봇 안전마진) 기준 연결성을 실제로 검증하고, 실패하면 재배치를
    반복한다. max_layout_attempts를 다 써도 못 찾으면 항상 통과 가능한 generate()
    (슬라롬)로 안전하게 폴백한다."""
    rng = random.Random(seed)
    goal_y_bound = room_half_y - goal_y_margin
    for _ in range(100):
        goal_x = rng.uniform(*goal_x_range)
        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)
        if (goal_x ** 2 + goal_y ** 2) ** 0.5 >= min_dist_from_start:
            break
    else:
        goal_x, goal_y = goal_x_range[1], 0.0

    room_x_min, room_x_max = -room_margin, goal_x + room_margin
    room_cx = (room_x_min + room_x_max) / 2
    half_len = (room_x_max - room_x_min) / 2
    west_x, east_x = room_x_min, room_x_max
    start_xy, goal_xy = (0.0, 0.0), (goal_x, goal_y)

    def boxes_overlap(a, b, pad=0.05):
        acx, acy, ahx, ahy = a
        bcx, bcy, bhx, bhy = b
        return (abs(acx - bcx) < ahx + bhx + pad) and (abs(acy - bcy) < ahy + bhy + pad)

    def clear_of_point(cx, cy, hx, hy, px, py, r):
        return abs(cx - px) > hx + r or abs(cy - py) > hy + r

    found_boxes = None
    for attempt in range(max_layout_attempts):
        n_boxes = rng.randint(*n_boxes_range)
        boxes = []
        for _ in range(n_boxes):
            for _try in range(30):
                hx = rng.uniform(*box_half_size_range)
                hy = rng.uniform(*box_half_size_range)
                cx = rng.uniform(room_x_min + hx + 0.1, room_x_max - hx - 0.1)
                cy = rng.uniform(-room_half_y + hy + 0.1, room_half_y - hy - 0.1)
                if not clear_of_point(cx, cy, hx, hy, *start_xy, start_clear_r):
                    continue
                if not clear_of_point(cx, cy, hx, hy, *goal_xy, goal_clear_r):
                    continue
                if any(boxes_overlap((cx, cy, hx, hy), b) for b in boxes):
                    continue
                boxes.append((cx, cy, hx, hy))
                break
        if not boxes:
            continue
        if _reachable_with_margin(room_x_min, room_x_max, room_half_y, boxes,
                                   start_xy, goal_xy, hard_radius_m):
            found_boxes = boxes
            break

    if found_boxes is None:
        xml, meta = generate(seed, name, goal_x_range=goal_x_range, room_half_y=room_half_y,
                              room_margin=room_margin, min_dist_from_start=min_dist_from_start,
                              goal_y_margin=goal_y_margin)
        meta["layout"] = "boxes_fallback_to_slalom"
        return xml, meta

    box_bodies = []
    for i, (cx, cy, hx, hy) in enumerate(found_boxes):
        box_bodies.append(
            f'    <body name="box{i}" pos="{cx:.3f} {cy:.3f} 0.3">\\n'
            f'      <geom type="box" size="{hx:.3f} {hy:.3f} 0.3" material="wall_material"/>\\n    </body>'
        )
    xml = TEMPLATE.format(
        name=name, seed=seed, room_cx=f"{room_cx:.3f}", half_len=f"{half_len:.3f}",
        neg_half_y_wall=f"{-room_half_y - 0.075:.3f}", half_y_wall=f"{room_half_y + 0.075:.3f}",
        west_x=f"{west_x - 0.075:.3f}", east_x=f"{east_x + 0.075:.3f}", half_y=f"{room_half_y + 0.15:.3f}",
        baffles="\\n".join(box_bodies),
    )
    meta = {
        "seed": seed, "layout": "boxes", "n_boxes": len(found_boxes),
        "goal_x": goal_x, "goal_y": goal_y,
        "boxes": [{"cx": round(cx, 3), "cy": round(cy, 3), "hx": round(hx, 3), "hy": round(hy, 3)}
                  for (cx, cy, hx, hy) in found_boxes],
        "any_infeasible": False,
    }
    return xml, meta


'''

content = content.replace(anchor, new_func + anchor)
path.write_text(content)
print("OK: generate_boxes() 추가 완료")
