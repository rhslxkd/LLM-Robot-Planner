#!/bin/bash
set -e

echo "=== 1) generate() 카메라 범위 패치 ==="
cat > patch_slalom_camera_bounds.py << 'PYEOF'
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
PYEOF
python3 patch_slalom_camera_bounds.py
python3 -c "import generate_random_baffle_maze; print('generate_random_baffle_maze import OK')"

echo "=== 2) run_random_batch_v2.py scene 이름 패치 ==="
cat > patch_scene_naming.py << 'PYEOF'
import pathlib

path = pathlib.Path("run_random_batch_v2.py")
content = path.read_text()

if 'oracle_scene_B{seed:04d}' in content:
    print("OK: 이미 patched (스킵)")
else:
    old = '''            seed = args.start_seed + i
            scene = f"oracle_scene_B{i:03d}"'''
    n = content.count(old)
    assert n == 1, f"old occurrence count = {n} (expected 1)"
    new = '''            seed = args.start_seed + i
            scene = f"oracle_scene_B{seed:04d}"'''
    content = content.replace(old, new)
    path.write_text(content)
    print("OK: scene 이름 패치 적용")
PYEOF
python3 patch_scene_naming.py
python3 -c "import ast; ast.parse(open('run_random_batch_v2.py').read()); print('run_random_batch_v2.py 문법 OK')"

echo "=== 3) 이전 pilot 잔여물 정리 ==="
rm -f data/random_batch_manifest_v2.csv
rm -rf data/oracle_scene_B000 data/oracle_scene_B001 data/oracle_scene_B002 data/oracle_scene_B003 data/oracle_scene_B004
rm -rf data/oracle_scene_B0100 data/oracle_scene_B0101 data/oracle_scene_B0102 data/oracle_scene_B0103 data/oracle_scene_B0104
echo "정리 완료"

echo "=== 4) pilot 실행 (5개, DIAL 포함) ==="
python3 run_random_batch_v2.py --n-scenes 5 --start-seed 100 --n-steps 800 --run-dial

echo "=== 5) 결과 ==="
cat data/random_batch_manifest_v2.csv
