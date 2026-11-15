"""
indoor/make_synthetic_export.py ─ Unity 없이 파이프라인을 시험하기 위한 가짜 실내 씬 내보내기.

unity/Editor/Go2BevExporter.cs와 **같은 형식**(bev.png, occupancy.png, meta.json)을 만든다.
논문 실험용이 아니라 변환기·지표·실행 스크립트 점검용.

  python indoor/make_synthetic_export.py --out Go2Export --n 3 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

IMG_W, IMG_H, PPM = 1263, 1080, 90.0
ROBOT_PX = (421.0, 540.0)
CELL = 0.05
BAND = (0.05, 0.45)


def w2p(x, y):
    return ROBOT_PX[0] + x * PPM, ROBOT_PX[1] - y * PPM


def rect_px(x0, y0, x1, y1):
    u0, v0 = w2p(x0, y1)
    u1, v1 = w2p(x1, y0)
    return [u0, v0, u1, v1]


def make_room(rng: np.random.Generator):
    """(그릴 것들, 로봇 높이에서 막는 직사각형들, goal). 좌표는 로봇 기준 m."""
    x_lo, x_hi = -1.2, rng.uniform(7.0, 8.5)
    y_hi = rng.uniform(2.4, 3.2)
    y_lo = -y_hi
    t = 0.15
    walls = [(x_lo - t, y_lo - t, x_hi + t, y_lo), (x_lo - t, y_hi, x_hi + t, y_hi + t),
             (x_lo - t, y_lo, x_lo, y_hi), (x_hi, y_lo, x_hi + t, y_hi)]
    draw, block = [], []
    for w in walls:
        draw.append(("wall", w)); block.append(w)

    # 가운데 소파(전체가 막힘) 또는 수납장
    cx = rng.uniform(2.2, 3.6); cy = rng.uniform(-0.4, 0.4)
    sofa = (cx - 0.45, cy - 1.0, cx + 0.45, cy + 1.0)
    draw.append(("sofa", sofa)); block.append(sofa)

    # 식탁: 위에서는 상판이 보이지만 로봇 높이에서는 다리 4개만 막음
    tx = rng.uniform(4.8, 6.0); ty = rng.choice([-1, 1]) * rng.uniform(1.0, 1.5)
    top = (tx - 0.6, ty - 0.4, tx + 0.6, ty + 0.4)
    draw.append(("table", top))
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = tx + sx * 0.52, ty + sy * 0.32
            block.append((lx - 0.03, ly - 0.03, lx + 0.03, ly + 0.03))

    # 의자 몇 개
    for _ in range(rng.integers(1, 3)):
        px, py = rng.uniform(1.0, 6.5), rng.uniform(y_lo + 0.4, y_hi - 0.4)
        if abs(px - cx) < 1.0 or (abs(px - tx) < 1.0 and abs(py - ty) < 0.8) or np.hypot(px, py) < 1.0:
            continue
        ch = (px - 0.22, py - 0.22, px + 0.22, py + 0.22)
        draw.append(("chair", ch)); block.append(ch)

    goal = (x_hi - rng.uniform(0.6, 1.0), rng.uniform(-0.5, 0.5))
    return draw, block, goal


COLORS = {"wall": (215, 215, 210), "sofa": (95, 110, 140), "table": (150, 105, 70), "chair": (120, 85, 60)}


def render_bev(draw_items, rng) -> np.ndarray:
    img = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    walls = [r for k, r in draw_items if k == "wall"]
    xs = [r[0] for r in walls] + [r[2] for r in walls]
    ys = [r[1] for r in walls] + [r[3] for r in walls]
    d.rectangle(rect_px(min(xs), min(ys), max(xs), max(ys)), fill=(181, 140, 98))      # 나무 바닥
    for x in np.arange(min(xs), max(xs), 0.18):                                          # 마루 줄무늬
        u, _ = w2p(x, 0)
        d.line([(u, w2p(0, max(ys))[1]), (u, w2p(0, min(ys))[1])], fill=(165, 125, 86), width=1)
    for kind, r in draw_items:
        if kind == "wall":
            continue
        sh = 0.05
        d.rectangle(rect_px(r[0] + sh, r[1] - sh, r[2] + sh, r[3] - sh), fill=(120, 90, 60))   # 그림자
        d.rectangle(rect_px(*r), fill=COLORS[kind], outline=tuple(int(c * 0.7) for c in COLORS[kind]), width=2)
    for r in walls:
        d.rectangle(rect_px(*r), fill=COLORS["wall"])
    arr = np.array(img).astype(float)
    arr += rng.normal(0, 4, arr.shape)                                                   # 센서 노이즈 느낌
    return np.clip(arr, 0, 255).astype(np.uint8)


def rasterize(block, x_min, y_min, nx, ny) -> np.ndarray:
    xs = x_min + (np.arange(nx) + 0.5) * CELL
    ys = y_min + (np.arange(ny) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), bool)
    for x0, y0, x1, y1 in block:
        # 셀 중심이 아니라 셀 영역이 겹치면 점유 (Unity CheckBox와 같은 판정)
        occ |= (X + CELL / 2 > x0) & (X - CELL / 2 < x1) & (Y + CELL / 2 > y0) & (Y - CELL / 2 < y1)
    return occ


def export(out_root: str, name: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    draw_items, block, goal = make_room(rng)
    x_min, y_min = -ROBOT_PX[0] / PPM, (ROBOT_PX[1] - IMG_H) / PPM
    nx = int(np.ceil(IMG_W / PPM / CELL)); ny = int(np.ceil(IMG_H / PPM / CELL))
    occ = rasterize(block, x_min, y_min, nx, ny)
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    Image.fromarray(render_bev(draw_items, rng)).save(os.path.join(d, "bev.png"))
    Image.fromarray(np.flipud(occ).astype(np.uint8) * 255).save(os.path.join(d, "occupancy.png"))
    meta = {"scene_name": name, "unity_scene": "(synthetic)", "exported_at": "synthetic",
            "img_w": IMG_W, "img_h": IMG_H, "ppm": PPM, "robot_px": list(ROBOT_PX), "cell_m": CELL,
            "x_min": x_min, "x_max": x_min + nx * CELL, "y_min": y_min, "y_max": y_min + ny * CELL,
            "occ_w": nx, "occ_h": ny, "band": list(BAND), "goal": [float(goal[0]), float(goal[1])],
            "camera_height": 2.0, "n_auto_colliders": 0}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Go2Export")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for i in range(a.n):
        print(export(a.out, f"synthetic_{i:03d}", a.seed + i))
