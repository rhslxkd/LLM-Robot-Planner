import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from vlm_courtroom.court.courtroom import deterministic_correct

scene = "oracle_scene_R012"
VARIANT = "no_vlm"  # VLM 버전의 "batch" 폴더에 대응하는, VLM 없는 결정론적 버전 결과 폴더

with open(f"data/{scene}/neural_astar/coordinate_proposal.json") as f:
    coords = json.load(f)

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
image_path = f"data/{scene}/oracle.png"

final_coords, passed, log = deterministic_correct(image_path, coords, ROBOT_PX, PPM)
print("PASSED:", passed)
print("\n".join(log))
print("\nfinal waypoints:", len(final_coords))

out_dir = f"data/{scene}/{VARIANT}"
os.makedirs(out_dir, exist_ok=True)

# 1) 좌표 JSON 저장 (DIAL-MPC 입력 포맷: x,y만 있는 순수 리스트)
dial_path = [{"x": c["x"], "y": c["y"]} for c in final_coords]
with open(f"{out_dir}/last_judged_path.json", "w") as f:
    json.dump(dial_path, f, indent=2)
print(f"OK: {out_dir}/last_judged_path.json (DIAL-MPC 포맷)")

# 2) 로그 저장
with open(f"{out_dir}/log.txt", "w") as f:
    f.write(f"PASSED: {passed}\n\n")
    f.write("\n".join(log))
print(f"OK: {out_dir}/log.txt")

# 3) 시각화 이미지 저장
img = mpimg.imread(image_path)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
rx, ry = ROBOT_PX
xs = [rx + c['x'] * PPM for c in final_coords]
ys = [ry - c['y'] * PPM for c in final_coords]
ax.plot(xs, ys, 'r-', linewidth=2)
ax.scatter(xs, ys, c='yellow', s=50, zorder=5)
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.annotate(str(i), (x, y), color='white', fontsize=11, fontweight='bold')
ax.plot(rx, ry, 'bo', markersize=10)
ax.set_title(f"deterministic_correct() (no VLM) -- PASSED={passed}")
out_path = f"{out_dir}/verdict.png"
plt.savefig(out_path)
plt.close()
print(f"OK: {out_path}")
