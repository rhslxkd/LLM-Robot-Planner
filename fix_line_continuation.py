import pathlib
import re

path = pathlib.Path("generate_random_baffle_maze.py")
content = path.read_text()

pattern = re.compile(
    r"if \(goal_x \*\* 2 \+ goal_y \*\* 2\) \*\* 0\.5 >= min_dist_from_start and [\\ \t]*\n"
    r"\s*goal_x \+ room_margin <= CAMERA_VISIBLE_X_MAX_M:"
)
replacement = (
    "if ((goal_x ** 2 + goal_y ** 2) ** 0.5 >= min_dist_from_start\n"
    "                and goal_x + room_margin <= CAMERA_VISIBLE_X_MAX_M):"
)

new_content, n = pattern.subn(replacement, content)
assert n == 1, f"match count = {n} (expected 1)"
path.write_text(new_content)
print("OK: line-continuation 버그 수정 (backslash 제거, 괄호로 대체)")
