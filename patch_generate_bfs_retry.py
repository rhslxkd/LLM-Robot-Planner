import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
lines = path.read_text().splitlines(keepends=True)

idx_start = 59
assert lines[idx_start] == '    if n_baffles is None:\n', f"60번 줄 불일치: {lines[idx_start]!r}"

idx_end = 96
assert 'wall_material"/>\\n    </body>\')' in lines[idx_end - 1], f"96번 줄 불일치: {lines[idx_end-1]!r}"

new_block = '''    if n_baffles is None:
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

            baffles.append(f'    <body name="baffle{i}" pos="{bx:.3f} {center_y:.3f} 0.3">\\n'
                            f'      <geom type="box" size="0.075 {half_y:.3f} 0.3" material="wall_material"/>\\n    </body>')
            boxes_for_check.append((bx, center_y, 0.075, half_y))

        if best is None:
            best = (baffles, open_sides, open_widths)
        if _reachable_with_margin(room_x_min, room_x_max, room_half_y, boxes_for_check,
                                   (0.0, 0.0), (goal_x, goal_y), hard_radius_m=0.4):
            best = (baffles, open_sides, open_widths)
            break
    baffles, open_sides, open_widths = best
'''

lines[idx_start:idx_end] = [new_block]
path.write_text(''.join(lines))
print("OK: generate()에 EDT 기반 재배치 재시도 루프 추가")
