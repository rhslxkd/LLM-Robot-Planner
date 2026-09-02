import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
lines = path.read_text().splitlines(keepends=True)

idx_sig = 41
expected_sig = 'def generate(seed, name, goal_x_range=(4.0, 8.0), room_half_y=3.0, n_baffles=None,\n'
assert lines[idx_sig] == expected_sig, f"42번 줄 불일치: {lines[idx_sig]!r}"
lines[idx_sig] = 'def generate(seed, name, goal_x_range=(2.5, 4.6), room_half_y=3.0, n_baffles=None,\n'

idx_start, idx_end = 48, 55
expected_block = [
    '    for _ in range(100):\n',
    '        goal_x = rng.uniform(*goal_x_range)\n',
    '        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)\n',
    '        if (goal_x**2 + goal_y**2) ** 0.5 >= min_dist_from_start:\n',
    '            break\n',
    '    else:\n',
    '        goal_x, goal_y = goal_x_range[1], 0.0\n',
]
actual_block = lines[idx_start:idx_end]
assert actual_block == expected_block, f"49~55번 줄 불일치:\n{actual_block!r}"

new_block = [
    '    for _ in range(100):\n',
    '        goal_x = rng.uniform(*goal_x_range)\n',
    '        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)\n',
    '        dist_ok = (goal_x ** 2 + goal_y ** 2) ** 0.5 >= min_dist_from_start\n',
    '        camera_ok = goal_x + room_margin <= CAMERA_VISIBLE_X_MAX_M\n',
    '        if dist_ok and camera_ok:\n',
    '            break\n',
    '    else:\n',
    '        goal_x = min(goal_x_range[1], CAMERA_VISIBLE_X_MAX_M - room_margin)\n',
    '        goal_y = 0.0\n',
]
lines[idx_start:idx_end] = new_block

path.write_text(''.join(lines))
print("OK: generate() 라인 기반 패치 적용 완료")
