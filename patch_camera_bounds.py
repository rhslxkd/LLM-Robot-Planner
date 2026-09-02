import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
content = path.read_text()

old = '''def generate_boxes(seed, name, goal_x_range=(4.0, 8.0), room_half_y=3.0,
                    n_boxes_range=(3, 7), box_half_size_range=(0.15, 1.0),
                    room_margin=1.0, min_dist_from_start=3.0, goal_y_margin=1.0,
                    hard_radius_m=0.4, start_clear_r=0.6, goal_clear_r=0.6,
                    max_layout_attempts=300):'''
n = content.count(old)
assert n == 1, f"old occurrence count = {n} (expected 1)"

new = '''# 카메라(ROBOT_PX=(421,540), PPM=150, 이미지 1263x1080)가 로봇 중심 기준 비대칭이라
# 동쪽으로 보이는 실제 가시 한계는 (1263-421)/150 ~= 5.61m. 이걸 넘는 방 크기는
# oracle.png에 벽이 안 찍혀서(화면 밖) Neural A*/waypoint_generator가 "벽 없음"으로
# 착각하는 위험한 버그가 됨. 0.3m 여유를 두고 5.3m를 하드 리밋으로 잡는다.
CAMERA_VISIBLE_X_MAX_M = 5.3


def generate_boxes(seed, name, goal_x_range=(2.5, 4.4), room_half_y=3.0,
                    n_boxes_range=(3, 7), box_half_size_range=(0.15, 1.0),
                    room_margin=0.7, min_dist_from_start=3.0, goal_y_margin=1.0,
                    hard_radius_m=0.4, start_clear_r=0.6, goal_clear_r=0.6,
                    max_layout_attempts=300):'''
content = content.replace(old, new)

old2 = '''    rng = random.Random(seed)
    goal_y_bound = room_half_y - goal_y_margin
    for _ in range(100):
        goal_x = rng.uniform(*goal_x_range)
        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)
        if (goal_x ** 2 + goal_y ** 2) ** 0.5 >= min_dist_from_start:
            break
    else:
        goal_x, goal_y = goal_x_range[1], 0.0

    room_x_min, room_x_max = -room_margin, goal_x + room_margin'''
n2 = content.count(old2)
assert n2 == 1, f"old2 occurrence count = {n2} (expected 1)"
new2 = '''    rng = random.Random(seed)
    goal_y_bound = room_half_y - goal_y_margin
    for _ in range(100):
        goal_x = rng.uniform(*goal_x_range)
        goal_y = rng.uniform(-goal_y_bound, goal_y_bound)
        if (goal_x ** 2 + goal_y ** 2) ** 0.5 >= min_dist_from_start and \\\\
           goal_x + room_margin <= CAMERA_VISIBLE_X_MAX_M:
            break
    else:
        goal_x = min(goal_x_range[1], CAMERA_VISIBLE_X_MAX_M - room_margin)
        goal_y = 0.0

    room_x_min, room_x_max = -room_margin, goal_x + room_margin'''
content = content.replace(old2, new2)

path.write_text(content)
print("OK: 카메라 가시범위 체크 추가 + goal_x_range/room_margin 기본값 축소")
