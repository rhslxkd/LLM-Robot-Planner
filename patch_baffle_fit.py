import pathlib

path = pathlib.Path("generate_random_baffle_maze.py")
lines = path.read_text().splitlines(keepends=True)

idx = 44
expected = '             switch_prob=0.5, min_baffle_spacing=1.2):\n'
assert lines[idx] == expected, f"45번 줄 불일치: {lines[idx]!r}"
lines[idx] = '             switch_prob=0.5, min_baffle_spacing=0.7):\n'

idx2 = 60
expected2 = '        n_baffles = rng.randint(2, 4)\n'
assert lines[idx2] == expected2, f"61번 줄 불일치: {lines[idx2]!r}"
lines[idx2] = '        n_baffles = rng.randint(2, 3)\n'

path.write_text(''.join(lines))
print("OK: min_baffle_spacing 1.2->0.7, n_baffles 상한 4->3")
