"""
make_fig1.py ─ 논문 그림 1: (왼쪽) Replica BEV 위에 법정이 만든 경로와 실제 몸통 궤적,
                           (오른쪽) 같은 배치를 점유 상자로 세운 MuJoCo 씬에서 Go2가 그 경로를 걷는 모습.

  MUJOCO_GL=egl python analysis/make_fig1.py --trial data/replica_apartment_0_p0/C4/trial_00
  → results/figs/fig1_<scene>_<method>_<trial>.png   (+ 왼쪽/오른쪽 단독 그림)

오른쪽은 실행 궤적(states.npy)의 qpos를 전신 모델(go2.xml)로 재생해서 렌더링한다:
  여러 시점의 로봇을 반투명하게 겹치고(잔상), 계획 웨이포인트(빨강)와 실제 몸통 궤적(하늘색)을 함께 그린다.
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
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from analysis.collision_replay import build_replay_model  # noqa: E402
from analysis.episode_metrics import split_states  # noqa: E402

MODELS = os.path.join(REPO, "dial_mpc", "dial_mpc", "models", "unitree_go2")


def _font(size):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load(trial):
    scene = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(trial))))
    sm = json.load(open(os.path.join(REPO, "data", scene, "scene_meta.json")))
    path = np.array([[p["x"], p["y"]] for p in json.load(open(os.path.join(trial, "last_judged_path.json")))])
    st = sorted(glob.glob(os.path.join(trial, "*_states.npy")))
    if not st:
        raise FileNotFoundError(f"{trial}: *_states.npy 없음 (DIAL-MPC를 먼저 실행)")
    qpos = split_states(np.load(st[-1]))["qpos"]
    return scene, sm, path, qpos


def left_panel(scene, sm, path, qpos, size=900):
    im = Image.open(os.path.join(REPO, "data", scene, "bev_raw.png")).convert("RGB")
    ppm, (rx, ry) = float(sm["ppm"]), sm["robot_px"]
    to_px = lambda p: np.stack([rx + p[:, 0] * ppm, ry - p[:, 1] * ppm], 1)
    torso = qpos[:, :2] - qpos[0, :2]
    allp = np.concatenate([[[0, 0], sm["goal"]], path, torso])
    pp = to_px(allp)
    pad = 1.0 * ppm
    u0, v0 = np.maximum(pp.min(0) - pad, 0)
    u1, v1 = np.minimum(pp.max(0) + pad, im.size)
    side = max(u1 - u0, v1 - v0)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    box = (int(cu - side / 2), int(cv - side / 2), int(cu + side / 2), int(cv + side / 2))
    d = ImageDraw.Draw(im, "RGBA")
    w = max(3, int(side / 180))
    d.line([tuple(q) for q in to_px(torso)], fill=(0, 200, 255, 255), width=w + 2)
    q = [tuple(x) for x in to_px(np.vstack([[0, 0], path]))]
    d.line(q, fill=(40, 210, 70, 255), width=w)
    for x, y in q[1:]:
        d.ellipse([x - 2 * w, y - 2 * w, x + 2 * w, y + 2 * w], fill=(40, 210, 70, 255), outline=(0, 0, 0, 255))
    for (x, y), c in zip(to_px(np.array([[0, 0], sm["goal"]])), [(40, 90, 255), (30, 220, 60)]):
        r = 4 * w
        d.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(255, 255, 255), width=w)
    out = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (0, 0, 0))
    out.paste(im.crop(box), (0, 0))
    return out.resize((size, size))


def _add_sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _add_segment(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def right_panel(scene, sm, path, qpos, size=900, n_ghost=6, elevation=-58.0):
    m = build_replay_model(os.path.join(MODELS, f"{scene}.xml"))
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, size)
    m.vis.global_.offheight = max(m.vis.global_.offheight, size)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size, size)
    origin = qpos[0, :2]
    torso_w = qpos[:, :3]
    wp_w = path + origin
    allp = np.concatenate([torso_w[:, :2], wp_w, [origin + np.asarray(sm["goal"])]])
    lo, hi = allp.min(0), allp.max(0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    # 원근 때문에 카메라 쪽(-y)이 잘리지 않도록 바라보는 점을 조금 아래로
    cam.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2 - 0.08 * span, 0.0]
    cam.azimuth, cam.elevation = 90.0, elevation       # +x 오른쪽, +y 화면 위쪽 (BEV와 같은 방향)
    cam.distance = 1.45 * span + 1.5

    def render(q, overlays=False):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        if overlays:
            for k in range(0, len(torso_w) - 5, 5):
                _add_segment(r.scene, torso_w[k] * [1, 1, 0] + [0, 0, 0.03], torso_w[k + 5] * [1, 1, 0] + [0, 0, 0.03],
                             0.025, (0.0, 0.8, 1.0, 1.0))
            for p in wp_w:
                _add_sphere(r.scene, [p[0], p[1], 0.06], 0.06, (0.95, 0.15, 0.15, 1.0))
            g = origin + np.asarray(sm["goal"])
            _add_sphere(r.scene, [g[0], g[1], 0.08], 0.12, (0.1, 0.85, 0.25, 1.0))
        return r.render().astype(np.float32)

    away = qpos[0].copy()
    away[0] += 1000.0                                    # 로봇을 화면 밖으로 → 배경
    bg = render(away, overlays=True)
    out = bg.copy()
    idx = np.linspace(0, len(qpos) - 1, n_ghost).round().astype(int)
    for j, t in enumerate(idx):
        fr = render(qpos[t], overlays=True)
        mask = (np.abs(fr - bg).sum(-1) > 30)[..., None]
        a = 0.35 + 0.65 * (j / max(len(idx) - 1, 1))
        out = np.where(mask, (1 - a) * out + a * fr, out)
    r.close()
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", required=True, help="data/<scene>/<method>/trial_xx (states.npy가 있는 시행)")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--ghosts", type=int, default=6)
    ap.add_argument("--elevation", type=float, default=-58.0)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "figs"))
    a = ap.parse_args()
    scene, sm, path, qpos = load(a.trial)
    L = left_panel(scene, sm, path, qpos, a.size)
    R = right_panel(scene, sm, path, qpos, a.size, a.ghosts, a.elevation)
    gap, head = 20, 50
    fig = Image.new("RGB", (2 * a.size + gap, a.size + head), (255, 255, 255))
    fig.paste(L, (0, head))
    fig.paste(R, (a.size + gap, head))
    dr = ImageDraw.Draw(fig)
    f = _font(30)
    dr.text((10, 8), "(a) BEV input: planned path (green), executed torso (cyan)", fill=(0, 0, 0), font=_font(26))
    dr.text((a.size + gap + 10, 8), "(b) MuJoCo: same layout, occupancy as boxes", fill=(0, 0, 0), font=_font(26))
    os.makedirs(a.out, exist_ok=True)
    tag = "_".join(os.path.normpath(a.trial).split(os.sep)[-3:])
    fig.save(os.path.join(a.out, f"fig1_{tag}.png"))
    L.save(os.path.join(a.out, f"fig1a_{tag}.png"))
    R.save(os.path.join(a.out, f"fig1b_{tag}.png"))
    print(f"🖼  {os.path.join(a.out, f'fig1_{tag}.png')}")


if __name__ == "__main__":
    main()
