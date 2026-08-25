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

def generate(seed, name, goal_x_range=(4.0, 8.0), room_half_y=3.0, n_baffles=None,
             open_width_range=(1.6, 2.6), room_margin=0.5,
             min_dist_from_start=3.0, goal_y_margin=1.0,
             switch_prob=0.5, min_baffle_spacing=1.2):
    rng = random.Random(seed)

    goal_y_bound = room_half_y - goal_y_margin
    for _ in range(100):
        goal_x = rng.uniform(*goal_x_range)
        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)
        if (goal_x**2 + goal_y**2) ** 0.5 >= min_dist_from_start:
            break
    else:
        goal_x, goal_y = goal_x_range[1], 0.0

    if n_baffles is None:
        n_baffles = rng.randint(2, 4)

    room_x_min, room_x_max = -room_margin, goal_x + room_margin
    room_cx = (room_x_min + room_x_max) / 2
    half_len = (room_x_max - room_x_min) / 2
    west_x, east_x = room_x_min, room_x_max

    # 최소 간격을 보장하며 배플 x좌표 샘플링 (좁은 구간 중첩 방지)
    lo, hi = goal_x * 0.15, goal_x * 0.85
    baffle_xs = []
    for _ in range(200):
        cand = sorted(rng.uniform(lo, hi) for _ in range(n_baffles))
        if all(b2 - b1 >= min_baffle_spacing for b1, b2 in zip(cand, cand[1:])):
            baffle_xs = cand
            break
    if not baffle_xs:
        baffle_xs = sorted(rng.uniform(lo, hi) for _ in range(n_baffles))  # 폴백

    baffles, open_sides, open_widths = [], [], []
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
