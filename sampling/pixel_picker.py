"""
pixel_picker.py
이미지 위 클릭 -> 픽셀좌표 콘솔 출력. 로컬(디스플레이 있는) 환경에서 실행.
사용법: python pixel_picker.py data/oracle_scene_D/oracle.png
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = sys.argv[1] if len(sys.argv) > 1 else "data/oracle_scene_D/oracle.png"
img = np.array(Image.open(img_path).convert("RGB"))

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
ax.set_title("클릭 -> 콘솔에 픽셀좌표 출력 (창 닫으면 종료)")

clicks = []

def onclick(event):
    if event.xdata is None:
        return
    px, py = event.xdata, event.ydata
    clicks.append((px, py))
    print(f"[{len(clicks)}] pixel=({px:.1f}, {py:.1f})")
    ax.plot(px, py, "r+", markersize=12)
    ax.annotate(str(len(clicks)), (px, py), color="red")
    fig.canvas.draw()

fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()