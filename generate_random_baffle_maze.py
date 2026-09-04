"""
generate_random_baffle_maze.py
Scene E 패턴(강제 슬라롬)을 랜덤 파라미터로 확장한 미로 생성기.
방을 벽으로 완전히 감싸고, N개의 배플을 랜덤 배치.
- 매 배플마다 정확히 한쪽만 열려있어 구조상 항상 기하학적으로 통과 가능(BFS 불필요)
- open_width를 1.6~2.6m로 랜덤화 -> courtroom REJECT 기준(1.6m) 미만 없음, 항상 통과가능
- 목표 지점(goal_x, goal_y)도 랜덤화, 시작점(0,0)과 최소 거리 보장
- 쉬운 버전: 배플 수 축소, 방향전환 확률 축소(급슬라롬 방지), 배플 간 최소 간격 보장
"""
import os
import random

TEMPLATE = """<mujoco model="go2 oracle scene {name} - random baffle slalom (seed={seed})">
  <include file="mjx_go2_robot_only.xml"/>
  <statistic center="0 0 0.1" extent="0.8"/>
  <visual>
    <map zfar="100"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1263" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="wall_material" rgba="0.85 0.1 0.1 1" shininess="0.1" specular="0.5" roughness="0.3"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true" diffuse="1 1 1" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <body name="outer_south" pos="{room_cx} {neg_half_y_wall} 0.3"><geom type="box" size="{half_len} 0.075 0.3" material="wall_material"/></body>
    <body name="outer_north" pos="{room_cx} {half_y_wall} 0.3"><geom type="box" size="{half_len} 0.075 0.3" material="wall_material"/></body>
    <body name="outer_west" pos="{west_x} 0 0.3"><geom type="box" size="0.075 {half_y} 0.3" material="wall_material"/></body>
    <body name="outer_east" pos="{east_x} 0 0.3"><geom type="box" size="0.075 {half_y} 0.3" material="wall_material"/></body>
{baffles}
  </worldbody>
</mujoco>
"""

def generate(seed, name, goal_x_range=(2.5, 4.6), room_half_y=1.4, n_baffles=None,
             open_width_range=(1.6, 2.6), room_margin=0.5,
             min_dist_from_start=3.0, goal_y_margin=1.0,
             switch_prob=0.5, min_baffle_spacing=0.7):
    rng = random.Random(seed)

    goal_y_bound = room_half_y - goal_y_margin
    for _ in range(100):
        goal_x = rng.uniform(*goal_x_range)
        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)
        dist_ok = (goal_x ** 2 + goal_y ** 2) ** 0.5 >= min_dist_from_start
        camera_ok = goal_x + room_margin <= CAMERA_VISIBLE_X_MAX_M
        if dist_ok and camera_ok:
            break
    else:
        goal_x = min(goal_x_range[1], CAMERA_VISIBLE_X_MAX_M - room_margin)
        goal_y = 0.0

    if n_baffles is None:
        n_baffles = rng.randint(2, 3)

    room_x_min, room_x_max = -room_margin, goal_x + room_margin
    room_cx = (room_x_min + room_x_max) / 2
    half_len = (room_x_max - room_x_min) / 2
    west_x, east_x = room_x_min, room_x_max

    best = None
    for _layout_attempt in range(60):
        lo, hi = goal_x * 0.15, goal_x * 0.85
        baffle_xs = []
        for _ in range(200):
            cand = sorted(rng.uniform(lo, hi) for _ in range(n_baffles))
            if all(b2 - b1 >= min_baffle_spacing for b1, b2 in zip(cand, cand[1:])):
                baffle_xs = cand
                break
        if not baffle_xs:
            baffle_xs = sorted(rng.uniform(lo, hi) for _ in range(n_baffles))

        baffles, open_sides, open_widths, boxes_for_check = [], [], [], []
        prev_side = None
        for i, bx in enumerate(baffle_xs):
            if prev_side is None:
                side = rng.choice(["north", "south"])
            else:
                side = ("south" if prev_side == "north" else "north") if rng.random() < switch_prob else prev_side
            prev_side = side

            open_width = rng.uniform(*open_width_range)
            open_widths.append(open_width)
            open_sides.append(side)

            half_y = room_half_y - open_width / 2
            center_y = open_width / 2 if side == "south" else -open_width / 2

            baffles.append(f'    <body name="baffle{i}" pos="{bx:.3f} {center_y:.3f} 0.3">\n'
                            f'      <geom type="box" size="0.075 {half_y:.3f} 0.3" material="wall_material"/>\n    </body>')
            boxes_for_check.append((bx, center_y, 0.075, half_y))

        if best is None:
            best = (baffles, open_sides, open_widths)
        if _reachable_with_margin(room_x_min, room_x_max, room_half_y, boxes_for_check,
                                   (0.0, 0.0), (goal_x, goal_y), hard_radius_m=0.4):
            best = (baffles, open_sides, open_widths)
            break
    baffles, open_sides, open_widths = best

    xml = TEMPLATE.format(
        name=name, seed=seed, room_cx=f"{room_cx:.3f}", half_len=f"{half_len:.3f}",
        neg_half_y_wall=f"{-room_half_y - 0.075:.3f}", half_y_wall=f"{room_half_y + 0.075:.3f}",
        west_x=f"{west_x - 0.075:.3f}", east_x=f"{east_x + 0.075:.3f}", half_y=f"{room_half_y + 0.15:.3f}",
        baffles="\n".join(baffles),
    )
    meta = {
        "seed": seed, "n_baffles": n_baffles,
        "goal_x": goal_x, "goal_y": goal_y,
        "open_sides": open_sides, "open_widths": [round(w, 2) for w in open_widths],
        "min_open_width": round(min(open_widths), 2) if open_widths else None,
        "any_infeasible": any(w < 1.6 for w in open_widths),
    }
    return xml, meta

def _reachable_with_margin(room_x_min, room_x_max, room_half_y, boxes, start_xy, goal_xy,
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

    wall_t = 0.075  # 외벽 half-thickness (TEMPLATE의 geom size="0.075 ..."와 동일해야 함.
                    # 이전 버전은 0.15를 half-width로 써서 벽을 실제보다 2배 두껍게
                    # 마킹하는 버그가 있었음 -- start가 항상 마진 미달로 걸려 100% fallback됨)
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


# 카메라(ROBOT_PX=(421,540), PPM=150, 이미지 1263x1080)가 로봇 중심 기준 비대칭이라
# 동쪽으로 보이는 실제 가시 한계는 (1263-421)/150 ~= 5.61m. 이걸 넘는 방 크기는
# oracle.png에 벽이 안 찍혀서(화면 밖) Neural A*/waypoint_generator가 "벽 없음"으로
# 착각하는 위험한 버그가 됨. 0.3m 여유를 두고 5.3m를 하드 리밋으로 잡는다.
CAMERA_VISIBLE_X_MAX_M = 5.3


def generate_boxes(seed, name, goal_x_range=(2.5, 4.4), room_half_y=3.0,
                    n_boxes_range=(3, 7), box_half_size_range=(0.15, 1.0),
                    room_margin=0.7, min_dist_from_start=3.0, goal_y_margin=1.0,
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
        if ((goal_x ** 2 + goal_y ** 2) ** 0.5 >= min_dist_from_start
                and goal_x + room_margin <= CAMERA_VISIBLE_X_MAX_M):
            break
    else:
        goal_x = min(goal_x_range[1], CAMERA_VISIBLE_X_MAX_M - room_margin)
        goal_y = 0.0

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
            f'    <body name="box{i}" pos="{cx:.3f} {cy:.3f} 0.3">\n'
            f'      <geom type="box" size="{hx:.3f} {hy:.3f} 0.3" material="wall_material"/>\n    </body>'
        )
    xml = TEMPLATE.format(
        name=name, seed=seed, room_cx=f"{room_cx:.3f}", half_len=f"{half_len:.3f}",
        neg_half_y_wall=f"{-room_half_y - 0.075:.3f}", half_y_wall=f"{room_half_y + 0.075:.3f}",
        west_x=f"{west_x - 0.075:.3f}", east_x=f"{east_x + 0.075:.3f}", half_y=f"{room_half_y + 0.15:.3f}",
        baffles="\n".join(box_bodies),
    )
    meta = {
        "seed": seed, "layout": "boxes", "n_boxes": len(found_boxes),
        "goal_x": goal_x, "goal_y": goal_y,
        "boxes": [{"cx": round(cx, 3), "cy": round(cy, 3), "hx": round(hx, 3), "hy": round(hy, 3)}
                  for (cx, cy, hx, hy) in found_boxes],
        "any_infeasible": False,
    }
    return xml, meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenes", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="dial_mpc/dial_mpc/models/unitree_go2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n_scenes):
        seed = args.start_seed + i
        name = f"oracle_scene_R{i:03d}"
        xml, meta = generate(seed, name)
        with open(os.path.join(args.out_dir, f"{name}.xml"), "w") as f:
            f.write(xml)
        tag = "⚠️INFEASIBLE예상" if meta["any_infeasible"] else "통과가능"
        print(f"[{name}] goal=({meta['goal_x']:.2f},{meta['goal_y']:.2f}) n_baffles={meta['n_baffles']} open_widths={meta['open_widths']} {tag}")
