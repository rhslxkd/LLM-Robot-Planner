import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
lines = path.read_text().splitlines(keepends=True)

idx = 41
expected = 'def generate(seed, name, goal_x_range=(2.5, 4.6), room_half_y=3.0, n_baffles=None,\n'
assert lines[idx] == expected, f"42번 줄 불일치: {lines[idx]!r}"
lines[idx] = 'def generate(seed, name, goal_x_range=(2.5, 4.6), room_half_y=1.4, n_baffles=None,\n'

path.write_text(''.join(lines))
print("OK: room_half_y 3.0 -> 1.4")
