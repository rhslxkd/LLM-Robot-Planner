import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from waypoint_generator import build_wall_mask, build_distance_field, clearance_m_at

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
scene = "oracle_scene_R004"

with open(f"data/{scene}/waypoint_gen_v1/last_judged_path.json") as f:
    coords = json.load(f)

rx, ry = ROBOT_PX
points_px = [(rx + c["x"] * PPM, ry - c["y"] * PPM) for c in coords]
goal_px = points_px[-1]

wall_mask = build_wall_mask(f"data/{scene}/oracle.png",
                             exclude_center_px=goal_px, exclude_radius_px=30)
dist_field = build_distance_field(wall_mask)

px, py = points_px[4]
c_m = clearance_m_at(dist_field, px, py, PPM)
print(f"idx4 world=({coords[4]['x']},{coords[4]['y']}) pixel=({px:.1f},{py:.1f})")
print(f"clearance_m at idx4 = {c_m:.3f}m (= {c_m*PPM:.1f}px)")
print(f"wall_mask[idx4 pixel] (벽 위에 있는지) = {wall_mask[int(round(py)), int(round(px))]}")

xi, yi = int(round(px)), int(round(py))
crop = wall_mask[max(0,yi-40):yi+40, max(0,xi-40):xi+40]
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(crop, cmap="gray_r")
ax.scatter([40], [40], c="red", s=100, marker="x", label="idx4")
ax.set_title(f"idx4 주변 확대 (clearance={c_m:.3f}m)")
ax.legend()
plt.savefig(f"data/{scene}/waypoint_gen_v1/idx4_zoom.png")
print(f"OK: data/{scene}/waypoint_gen_v1/idx4_zoom.png")
