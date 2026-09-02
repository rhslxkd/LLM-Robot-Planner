import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
content = path.read_text()

old1 = '''    wall_t = 0.15  # 외벽 두께(기존 TEMPLATE 0.075*2 근사)'''
n1 = content.count(old1)
assert n1 == 1, f"old1 occurrence count = {n1} (expected 1)"
new1 = '''    wall_t = 0.075  # 외벽 half-thickness (TEMPLATE의 geom size="0.075 ..."와 동일해야 함.
                    # 이전 버전은 0.15를 half-width로 써서 벽을 실제보다 2배 두껍게
                    # 마킹하는 버그가 있었음 -- start가 항상 마진 미달로 걸려 100% fallback됨)'''
content = content.replace(old1, new1)

old2 = '''def generate_boxes(seed, name, goal_x_range=(4.0, 8.0), room_half_y=3.0,
                    n_boxes_range=(3, 7), box_half_size_range=(0.15, 1.0),
                    room_margin=0.5, min_dist_from_start=3.0, goal_y_margin=1.0,'''
n2 = content.count(old2)
assert n2 == 1, f"old2 occurrence count = {n2} (expected 1)"
new2 = '''def generate_boxes(seed, name, goal_x_range=(4.0, 8.0), room_half_y=3.0,
                    n_boxes_range=(3, 7), box_half_size_range=(0.15, 1.0),
                    room_margin=1.0, min_dist_from_start=3.0, goal_y_margin=1.0,'''
content = content.replace(old2, new2)

path.write_text(content)
print("OK: wall_t 버그 수정 + room_margin 0.5->1.0")
