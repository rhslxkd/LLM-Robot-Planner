"""
auto_detect_positions.py
씬 스크린샷마다 robot_px를 자동 검출 (world origin/PPM은 이 렌더러 고정값 재사용).
사용법: python auto_detect_positions.py <image_path>
"""
import sys
import numpy as np
from PIL import Image

img_path = sys.argv[1]
img = np.array(Image.open(img_path).convert("RGB"))
h, w = img.shape[:2]
r, g, b = img[...,0].astype(int), img[...,1].astype(int), img[...,2].astype(int)

# world origin/PPM: 이 렌더러에서 고정값 (여러 씬에서 확인됨)
ORIGIN_PX = (426, 540)
PPM = 90.0

# 로봇 아이콘: 흰색 계열 (체커보드=파랑, 벽=빨강, 라벨=노란-초록이라 안 겹침)
robot_mask = (r > 200) & (g > 200) & (b > 200)
ys, xs = np.where(robot_mask)
if len(xs) == 0:
    print("로봇 흰색 마스크로 못 찾음 -- 임계값 조정 필요")
else:
    robot_px = (float(xs.mean()), float(ys.mean()))
    world_x = (robot_px[0] - ORIGIN_PX[0]) / PPM
    world_y = (ORIGIN_PX[1] - robot_px[1]) / PPM
    print(f"robot_px = {robot_px}  world = ({world_x:.2f}, {world_y:.2f})")

# 목표(goal) 마커: 색깔 확인 필요 -- 아래 채워넣기
# goal_mask = (r < 80) & (g > 150) & (b < 80)   # 예시: 초록이면
# ys2, xs2 = np.where(goal_mask)
# if len(xs2): print("goal_px =", (xs2.mean(), ys2.mean()))