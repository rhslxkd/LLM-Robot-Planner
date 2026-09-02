import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from waypoint_generator import generate_waypoints

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
scene = "oracle_scene_R004"

with open(f"data/{scene}/neural_astar/coordinate_proposal.json") as f:
    raw_coords = json.load(f)

image_path = f"data/{scene}/oracle.png"
final_coords, passed, log = generate_waypoints(image_path, raw_coords, ROBOT_PX, PPM)

print("PASSED:", passed)
print("\n".join(log))
print("\nfinal waypoints:", len(final_coords))

out_dir = f"data/{scene}/waypoint_gen_v1"
os.makedirs(out_dir, exist_ok=True)

dial_path = [{"x": c["x"], "y": c["y"]} for c in final_coords]
with open(f"{out_dir}/last_judged_path.json", "w") as f:
    json.dump(dial_path, f, indent=2)

with open(f"{out_dir}/log.txt", "w") as f:
    f.write(f"PASSED: {passed}\n\n")
    f.write("\n".join(log))

img = mpimg.imread(image_path)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
rx, ry = ROBOT_PX
xs = [rx + c["x"] * PPM for c in final_coords]
ys = [ry - c["y"] * PPM for c in final_coords]
ax.plot(xs, ys, "r-", linewidth=2)
ax.scatter(xs, ys, c="yellow", s=50, zorder=5)
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.annotate(str(i), (x, y), color="white", fontsize=11, fontweight="bold")
ax.plot(rx, ry, "bo", markersize=10)
ax.set_title(f"{scene} / waypoint_generator v1 -- PASSED={passed}")
plt.savefig(f"{out_dir}/verdict.png")
plt.close()
print(f"OK: {out_dir}/verdict.png")
