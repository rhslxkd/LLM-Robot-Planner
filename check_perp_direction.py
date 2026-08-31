"""
perp_x, perp_y 부호가 실제 이미지 좌표계에서 올바른 쪽(빨간 벽이 없는 쪽)을
가리키는지 한 번에 시각적으로 검증하는 스크립트.
사용법: (neural-astar env, 리포 루트에서)
  python check_perp_direction.py --scene oracle_scene_R001
narrow(clearance<2.0m)한 waypoint 몇 개를 골라 cast(+perp)/cast(-perp) 방향을
화살표로 그려서 overlay로 저장 -> 어느 화살표가 실제로 가까운 벽을 향하는지
눈으로 1회 확인하면 부호 컨벤션이 확정됨.
"""
import argparse, json, numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()

    coord_path = f"data/{args.scene}/neural_astar/coordinate_proposal.json"
    img_path = f"data/{args.scene}/oracle.png"
    with open(coord_path) as f:
        coords = json.load(f)

    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    r, g, b = arr[...,0].astype(int), arr[...,1].astype(int), arr[...,2].astype(int)
    red_mask = (r > 180) & (g < 120) & (b < 120)
    h, w = red_mask.shape

    ROBOT_PX = (421.0, 540.0)
    PPM = 150.0
    def world_to_full(wx, wy):
        return ROBOT_PX[0] + wx*PPM, ROBOT_PX[1] - wy*PPM

    pts_full = [world_to_full(c["x"], c["y"]) for c in coords]

    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(img)
    fx, fy = zip(*pts_full)
    ax.plot(fx, fy, color="orange", linewidth=1.5)

    max_range = max(h, w)
    def cast(x0, y0, dx, dy):
        for rr in range(1, max_range):
            xi, yi = int(round(x0+dx*rr)), int(round(y0+dy*rr))
            if not (0 <= xi < w and 0 <= yi < h):
                return rr, xi, yi
            if red_mask[yi, xi]:
                return rr, xi, yi
        return max_range, x0, y0

    narrow_idxs = [i for i, c in enumerate(coords) if c["clearance_m"] < 2.0]
    if not narrow_idxs:
        narrow_idxs = list(range(len(coords)))

    for idx in narrow_idxs:
        if idx == 0:
            tx, ty = pts_full[1][0]-pts_full[0][0], pts_full[1][1]-pts_full[0][1]
        elif idx == len(pts_full)-1:
            tx, ty = pts_full[idx][0]-pts_full[idx-1][0], pts_full[idx][1]-pts_full[idx-1][1]
        else:
            tx, ty = pts_full[idx+1][0]-pts_full[idx-1][0], pts_full[idx+1][1]-pts_full[idx-1][1]
        norm = (tx**2+ty**2)**0.5
        if norm < 1e-6: continue
        perp_x, perp_y = -ty/norm, tx/norm
        x0, y0 = pts_full[idx]

        r_pos, xp, yp = cast(x0, y0, perp_x, perp_y)
        r_neg, xn, yn = cast(x0, y0, -perp_x, -perp_y)

        ax.annotate("", xy=(xp, yp), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="lime", lw=2))
        ax.annotate("", xy=(xn, yn), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="cyan", lw=2))
        ax.text(x0, y0, f"{idx}", color="white", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.6, pad=1))

    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    out_path = f"data/{args.scene}/neural_astar/perp_check.png"
    plt.savefig(out_path, dpi=100, bbox_inches=None)
    print(f"OK: {out_path}  (초록 화살표=+perp(d_pos 방향), 청록 화살표=-perp(d_neg 방향))")
    print(f"narrow waypoints checked: {narrow_idxs}")

if __name__ == "__main__":
    main()
