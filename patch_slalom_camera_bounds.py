import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
content = path.read_text()

old_sig = '''def generate(seed, name, goal_x_range=(4.0, 8.0), room_half_y=3.0, n_baffles=None,
             open_width_range=(1.6, 2.6), room_margin=0.5,
             min_dist_from_start=3.0, goal_y_margin=1.0,
             switch_prob=0.5, min_baffle_spacing=1.2):'''
already_patched = 'goal_x_range=(2.5, 4.6)' in content
if not already_patched:
    n1 = content.count(old_sig)
    assert n1 == 1, f"old_sig occurrence count = {n1} (expected 1)"
    new_sig = '''def generate(seed, name, goal_x_range=(2.5, 4.6), room_half_y=3.0, n_baffles=None,
             open_width_range=(1.6, 2.6), room_margin=0.5,
             min_dist_from_start=3.0, goal_y_margin=1.0,
             switch_prob=0.5, min_baffle_spacing=1.2):'''
    content = content.replace(old_sig, new_sig)

    old_loop = '''    rng = random.Random(seed)
    goal_y_bound = room_half_y - goal_y_margin
    for _ in range(100):
        goal_x = rng.uniform(*goal_x_range)
        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)
        if (goal_x**2 + goal_y**2) ** 0.5 >= min_dist_from_start:
            break
    else:
        goal_x, goal_y = goal_x_range[1], 0.0'''
    n2 = content.count(old_loop)
    assert n2 == 1, f"old_loop occurrence count = {n2} (expected 1)"
    new_loop = '''    rng = random.Random(seed)
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
        goal_y = 0.0'''
    content = content.replace(old_loop, new_loop)
    path.write_text(content)
    print("OK: generate() 패치 적용")
else:
    print("OK: generate()는 이미 패치되어 있음 (스킵)")
