"""
plot_paths.py ─ 씬별로 모든 방법의 경로(와 실행 궤적)를 BEV 위에 겹쳐 그린 그림.

  python analysis/plot_paths.py --scenes replica_office_4_p0 replica_frl_apartment_1_p0
  → results/figs/<scene>_paths.png      (방법마다 한 칸: 계획 경로 = 실선, 실제 몸통 궤적 = 점선)
  → results/figs/overview_paths.png     (위 그림들을 한 장으로)

색: 유효(목표 도달·BEV 충돌 없음) = 초록, 무효 = 빨강, 실행 궤적 = 하늘색. 회색 = GT 점유.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.plan_metrics import OccGrid, evaluate_path, parse_path  # noqa: E402

ORDER = ["C1", "C1xk", "C2", "C3", "C4", "C2_coord", "C3_coord", "C4_coord",
         "astar_gt", "dijkstra_gt", "rrtstar_gt", "llmastar_gt", "astar_seg", "rrtstar_seg", "llmastar_seg", "astarraw_gt", "neuralastar_gt"]
LABEL = {"C1": "C1 Coordinate", "C1xk": "C1x4 (medoid)", "C2": "C2 +Prosecutor", "C3": "C3 +Defense",
         "C4": "C4 Courtroom (ours)", "C2_coord": "C2 (img: Coord only)", "C3_coord": "C3 (img: Coord only)",
         "C4_coord": "C4 (img: Coord only)", "astar_gt": "A* (GT map)", "dijkstra_gt": "Dijkstra (GT map)",
         "rrtstar_gt": "RRT* (GT map)", "llmastar_gt": "LLM-A* (GT map)", "astar_seg": "A* (segmented)",
         "rrtstar_seg": "RRT* (segmented)", "llmastar_seg": "LLM-A* (segmented)", "astarraw_gt": "A* raw (GT map)",
         "neuralastar_gt": "Neural A* (GT map)"}


def _font(size):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _torso_xy(d):
    st = sorted(glob.glob(os.path.join(d, "*_states.npy")))
    if not st:
        return None
    s = np.load(st[-1])
    return s[:, 1:3]                                     # qpos x, y


def scene_figure(scene: str, tile_w: int = 520) -> Image.Image | None:
    sd = os.path.join(REPO, "data", scene)
    meta = json.load(open(os.path.join(sd, "scene_meta.json")))
    grid = OccGrid.load(os.path.join(sd, "occupancy.npz"))
    goal = meta["goal"]
    ppm, (rx, ry) = float(meta["ppm"]), meta["robot_px"]
    base = Image.open(os.path.join(sd, "bev_raw.png")).convert("RGB")
    W, H = base.size
    # 경로가 있는 영역만 잘라서 크게 보이게: 시작·목표와 모든 경로를 포함하는 상자 + 1 m
    methods = [m for m in ORDER if glob.glob(os.path.join(sd, m, "trial_*", "meta.json"))]
    if not methods:
        return None
    pts = [np.array([[0, 0], goal])]
    for m in methods:
        for pj in glob.glob(os.path.join(sd, m, "trial_*", "last_judged_path.json")):
            p = parse_path(json.load(open(pj)))
            if p is not None:
                pts.append(np.clip(p, -50, 50))
    allp = np.concatenate(pts)
    to_px = lambda p: np.stack([rx + p[:, 0] * ppm, ry - p[:, 1] * ppm], 1)
    pp = to_px(allp)
    u0, v0 = np.clip(pp.min(0) - ppm, 0, None).astype(int)
    u1, v1 = np.minimum(pp.max(0) + ppm, [W, H]).astype(int)
    box = (int(u0), int(v0), int(max(u1, u0 + 2 * ppm)), int(max(v1, v0 + 2 * ppm)))
    s = tile_w / (box[2] - box[0])
    th = int((box[3] - box[1]) * s)
    occ = np.flipud(grid.occ)                           # 행 0 = y_max (이미지 위쪽)
    f, fs = _font(18), _font(14)
    tiles = []
    for m in methods:
        im = base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        # GT 점유를 옅게
        oy, ox = grid.origin[1] + (grid.occ.shape[0] - 1) * grid.res, grid.origin[0]
        ou, ov = rx + ox * ppm, ry - oy * ppm
        cell = grid.res * ppm
        mask = Image.fromarray((occ * 90).astype(np.uint8)).resize(
            (int(round(occ.shape[1] * cell)), int(round(occ.shape[0] * cell))), Image.NEAREST)
        gray = Image.new("RGBA", mask.size, (60, 60, 60, 0))
        gray.putalpha(mask)
        im.paste(gray, (int(round(ou - cell / 2)), int(round(ov - cell / 2))), gray)
        d = ImageDraw.Draw(im, "RGBA")
        n_valid = n_all = 0
        for td in sorted(glob.glob(os.path.join(sd, m, "trial_*"))):
            tr = _torso_xy(td)
            if tr is not None and len(tr) > 1:
                d.line([tuple(q) for q in to_px(tr)], fill=(0, 190, 255, 230), width=4)
            pj = os.path.join(td, "last_judged_path.json")
            n_all += 1
            if not os.path.exists(pj):
                continue
            path = json.load(open(pj))
            ev = evaluate_path(path, grid, goal)
            ok = bool(ev.get("valid"))
            n_valid += ok
            p = parse_path(path)
            if p is None:
                continue
            col = (40, 200, 70, 255) if ok else (235, 40, 40, 255)
            q = [tuple(x) for x in to_px(p)]
            d.line(q, fill=col, width=5)
            for x, y in q:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)
        if n_valid == 0:
            reasons = set()
            for mp in glob.glob(os.path.join(sd, m, "trial_*", "meta.json")):
                mm = json.load(open(mp))
                reasons.add(mm.get("reason") or ("parse failed" if not mm.get("parse_ok", True) else "invalid path"))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            msg = "NO VALID PATH: " + ", ".join(sorted(str(r) for r in reasons))[:60]
            d.rectangle([cx - 330, cy - 26, cx + 330, cy + 26], fill=(0, 0, 0, 170))
            d.text((cx - 320, cy - 14), msg, fill=(255, 90, 90), font=_font(int(22 / s)))
        for (x, y), c in zip(to_px(np.array([[0, 0], goal])), [(40, 90, 255), (30, 220, 60)]):
            d.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 255, 255), width=4, fill=c)
        t = im.crop(box).resize((tile_w, th))
        dt = ImageDraw.Draw(t)
        dt.rectangle([0, 0, tile_w, 30], fill=(0, 0, 0))
        dt.text((8, 6), f"{LABEL.get(m, m)}   valid {n_valid}/{n_all}", fill=(255, 255, 255), font=f)
        tiles.append(t)
    cols = 4
    rows = -(-len(tiles) // cols)
    head = 40
    fig = Image.new("RGB", (cols * tile_w, head + rows * th), (255, 255, 255))
    ImageDraw.Draw(fig).text((10, 10), f"{scene}   (green = valid plan, red = invalid plan, "
                                       f"cyan = executed torso trajectory, gray = GT occupancy)",
                             fill=(0, 0, 0), font=fs)
    for i, t in enumerate(tiles):
        fig.paste(t, ((i % cols) * tile_w, head + (i // cols) * th))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "figs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figs = []
    for sc in a.scenes:
        fig = scene_figure(sc)
        if fig is None:
            print(f"⚠️  {sc}: 결과 없음")
            continue
        p = os.path.join(a.out, f"{sc}_paths.png")
        fig.save(p)
        figs.append(fig)
        print(f"🖼  {p}")
    if len(figs) > 1:
        W = max(f.width for f in figs)
        allf = Image.new("RGB", (W, sum(f.height for f in figs)), (255, 255, 255))
        y = 0
        for f in figs:
            allf.paste(f, (0, y))
            y += f.height
        p = os.path.join(a.out, "overview_paths.png")
        allf.save(p)
        print(f"🖼  {p}")


if __name__ == "__main__":
    main()
