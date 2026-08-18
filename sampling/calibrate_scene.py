"""
calibrate_scene.py
클릭 순서: 1,2) 실제 거리를 아는 기준 두 점(눈금 등)  3) 로봇/시작  4) 목표
사용법: python calibrate_scene.py data/oracle_scene_D/oracle.png
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

GRID = 32
img_path = sys.argv[1] if len(sys.argv) > 1 else "data/oracle_scene_D/oracle.png"
img = np.array(Image.open(img_path).convert("RGB"))
FULL_H, FULL_W = img.shape[:2]

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
ax.set_title("1,2: 기준 두 점(실거리 아는 지점)  3: 로봇  4: 목표")

clicks, labels = [], ["ref1", "ref2", "robot(start)", "goal"]

def onclick(event):
    if event.xdata is None or len(clicks) >= 4:
        return
    px, py = event.xdata, event.ydata
    clicks.append((px, py))
    label = labels[len(clicks) - 1]
    print(f"[{label}] pixel=({px:.1f}, {py:.1f})")
    ax.plot(px, py, "r+", markersize=14)
    ax.annotate(label, (px, py), color="red")
    fig.canvas.draw()
    if len(clicks) == 4:
        plt.close(fig)

fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()

if len(clicks) < 4:
    print("4개 클릭 안 채워짐, 종료"); sys.exit(1)

ref1, ref2, robot_px, goal_px = clicks
real_dist_m = float(input("ref1-ref2 실제 거리(m) 입력: "))
ppm = np.hypot(ref2[0]-ref1[0], ref2[1]-ref1[1]) / real_dist_m

fx, fy = GRID / FULL_W, GRID / FULL_H
robot_grid = (round(robot_px[0]*fx), round(robot_px[1]*fy))
goal_grid = (round(goal_px[0]*fx), round(goal_px[1]*fy))

print("\n--- 파이프라인 스크립트 Config에 붙여넣기 ---")
print(f"ROBOT_PX = ({robot_px[0]:.1f}, {robot_px[1]:.1f})")
print(f"PPM = {ppm:.2f}")
print(f"start_grid = {robot_grid}")
print(f"goal_grid = {goal_grid}")