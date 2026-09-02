import pathlib

path = pathlib.Path("run_random_batch_v2.py")
content = path.read_text()

old = '''            seed = args.start_seed + i
            scene = f"oracle_scene_B{i:03d}"'''
n = content.count(old)
assert n == 1, f"old occurrence count = {n} (expected 1)"
new = '''            seed = args.start_seed + i
            scene = f"oracle_scene_B{seed:04d}"'''
content = content.replace(old, new)

path.write_text(content)
print("OK: scene 이름을 i 대신 seed 기준으로 변경")
