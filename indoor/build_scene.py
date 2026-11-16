"""
indoor/build_scene.py ─ Unity 내보내기(Go2Export/<scene>/) → 실험에 바로 쓰는 씬 일체 생성

입력  (unity/Editor/Go2BevExporter.cs 출력)
  <export>/bev.png, occupancy.png, meta.json

출력  (기존 oracle 씬과 같은 경로 규약 → courtroom / DIAL-MPC / 분석 코드가 그대로 동작)
  data/<scene>/oracle.png          VLM 입력: BEV + 0.5 m 격자 + 시작(파랑)/목표(초록) 표시
  data/<scene>/bev_raw.png         원본 BEV (이미지 분할 베이스라인 입력)
  data/<scene>/occupancy.npz       GT 점유 지도 (analysis.plan_metrics.OccGrid) = MuJoCo 장애물과 동일
  data/<scene>/align_check.png     GT 점유(빨강)를 BEV 위에 겹친 정합 확인용 이미지
  data/<scene>/scene_meta.json     목표점, 스케일, 최소 프롬프트, 기본 n_steps
  dial_mpc/.../models/unitree_go2/<scene>.xml   장애물 = 점유 셀을 직사각형으로 병합한 box geom
  dial_mpc/.../examples/<scene>.yaml            DIAL-MPC 설정

실행
  python indoor/build_scene.py Go2Export/indoor_000              # 하나
  python indoor/build_scene.py Go2Export/*                       # 전부
  python indoor/build_scene.py Go2Export/indoor_000 --mj-cell 0.1  # 장애물 box 수 줄이기 (GT도 같이 거칠어짐)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.plan_metrics import OccGrid  # noqa: E402

MODELS_DIR = os.path.join(REPO, "dial_mpc", "dial_mpc", "models", "unitree_go2")
EXAMPLES_DIR = os.path.join(REPO, "dial_mpc", "dial_mpc", "examples")
DATA_DIR = os.path.join(REPO, "data")

BOX_HEIGHT_M = 0.6
CRUISE_SPEED = 0.8          # unitree_go2_env.py의 cruise_speed와 동일
DT = 0.02
SETTLE_STEPS = 150          # 경로 끝 도달 후 여유 (ramp-up + 정지)

# 최소 프롬프트: 시작·목표와 과제만 준다. 장애물 위치·우회 방향 같은 힌트는 넣지 않는다.
SCENARIO_TEMPLATE = (
    "이미지는 실내 공간을 위에서 내려다본 모습이야. 로봇(go2)은 파란 원 위치 (0, 0)에서 "
    "+x 방향(화면 오른쪽)을 보고 서 있어. 목표 지점은 초록 원 위치 ({gx:.2f}, {gy:.2f}) m야. "
    "벽과 가구를 피해서 목표 지점까지 가는 {{num_waypoints}}개의 좌표를 제시해줘. "
    "마지막 좌표는 목표 지점이어야 해."
)


# ─── 점유 지도 ───────────────────────────────────────────────────────────
def load_export(export_dir: str) -> tuple[dict, np.ndarray, OccGrid]:
    meta = json.load(open(os.path.join(export_dir, "meta.json")))
    bev = np.array(Image.open(os.path.join(export_dir, "bev.png")).convert("RGB"))
    occ_png = np.array(Image.open(os.path.join(export_dir, "occupancy.png")).convert("L")) > 127
    occ = np.flipud(occ_png)                              # PNG 맨 위 = y_max → grid 행 r은 y 증가 방향
    assert occ.shape == (meta["occ_h"], meta["occ_w"]), f"occupancy 크기 {occ.shape} != meta"
    cell = float(meta["cell_m"])
    origin = (meta["x_min"] + cell / 2, meta["y_min"] + cell / 2)
    occ, n_filled = fill_enclosed(occ, cell, origin, [(0.0, 0.0), tuple(meta["goal"])])
    meta["n_filled_cells"] = n_filled
    grid = OccGrid(occ, cell, origin)
    assert bev.shape[:2] == (meta["img_h"], meta["img_w"]), f"bev 크기 {bev.shape} != meta"
    return meta, bev, grid


def fill_enclosed(occ: np.ndarray, cell: float, origin, keep_xy) -> tuple[np.ndarray, int]:
    """가구 내부의 빈 주머니를 장애물로 채운다.

    Unity의 Physics.CheckBox는 오목한 MeshCollider의 '표면'만 잡아서, 의자·수납장이 테두리만 점유로 나온다.
    지도 경계에 닿지 않고 시작/목표를 포함하지 않는 빈 영역(= 장애물로 완전히 둘러싸인 곳)만 채우므로,
    벽으로 둘러싸인 방 내부(시작점 포함)나 방 바깥(경계에 닿음)은 그대로 둔다.
    """
    from scipy import ndimage
    lab, n = ndimage.label(~occ)                          # 4-연결 빈 영역
    keep = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    H, W = occ.shape
    for x, y in keep_xy:
        c, r = int(round((x - origin[0]) / cell)), int(round((y - origin[1]) / cell))
        if 0 <= r < H and 0 <= c < W:
            keep.add(int(lab[r, c]))
    fill = ~occ & ~np.isin(lab, list(keep))
    return occ | fill, int(fill.sum())


def coarsen(grid: OccGrid, cell: float) -> OccGrid:
    """cell 크기로 블록 any() → 보수적으로 장애물 확장. GT와 MuJoCo가 같은 격자를 쓰도록 둘 다 이걸 사용."""
    k = int(round(cell / grid.res))
    if k <= 1:
        return grid
    H, W = grid.occ.shape
    Hc, Wc = -(-H // k), -(-W // k)
    pad = np.zeros((Hc * k, Wc * k), bool)
    pad[:H, :W] = grid.occ
    occ = pad.reshape(Hc, k, Wc, k).any(axis=(1, 3))
    x0 = grid.origin[0] - grid.res / 2 + k * grid.res / 2
    y0 = grid.origin[1] - grid.res / 2 + k * grid.res / 2
    return OccGrid(occ, k * grid.res, (x0, y0))


def merge_rectangles(occ: np.ndarray) -> list[tuple[int, int, int, int]]:
    """점유 셀 → 겹치지 않는 직사각형 (r, c, h, w). 행 방향으로 늘린 뒤 아래로 늘리는 greedy."""
    H, W = occ.shape
    used = np.zeros_like(occ)
    rects = []
    for r in range(H):
        c = 0
        while c < W:
            if not occ[r, c] or used[r, c]:
                c += 1
                continue
            w = 1
            while c + w < W and occ[r, c + w] and not used[r, c + w]:
                w += 1
            h = 1
            while r + h < H and occ[r + h, c:c + w].all() and not used[r + h, c:c + w].any():
                h += 1
            used[r:r + h, c:c + w] = True
            rects.append((r, c, h, w))
            c += w
    return rects


# ─── MuJoCo / DIAL-MPC 설정 ───────────────────────────────────────────────
def mujoco_xml(scene: str, grid: OccGrid) -> tuple[str, int]:
    rects = merge_rectangles(grid.occ)
    geoms = []
    for i, (r, c, h, w) in enumerate(rects):
        cx = grid.origin[0] + (c + (w - 1) / 2) * grid.res
        cy = grid.origin[1] + (r + (h - 1) / 2) * grid.res
        geoms.append(f'    <geom name="obs_{i}" type="box" pos="{cx:.4f} {cy:.4f} {BOX_HEIGHT_M / 2:.3f}" '
                     f'size="{w * grid.res / 2:.4f} {h * grid.res / 2:.4f} {BOX_HEIGHT_M / 2:.3f}" material="obstacle" '
                     f'contype="0" conaffinity="0"/>')
    xml = f"""<mujoco model="go2 indoor {scene}">
  <!-- indoor/build_scene.py 자동 생성. 장애물 = GT 점유 지도(data/{scene}/occupancy.npz)의 직사각형 병합.
       장애물은 제어기 시뮬레이션에서 충돌하지 않는다(contype=conaffinity=0). DIAL-MPC의 보상에는 장애물 항이 없어
       회피는 전적으로 웨이포인트가 결정하고, 충돌은 analysis/collision_replay.py가 전신 충돌 모델로 재생해 판정한다.
       MJX는 broadphase 없이 모든 geom 쌍의 접촉을 미리 할당하므로, 실제 스캔 씬(장애물 200~400개)에서
       발–장애물 접촉을 켜 두면 2048 샘플에서 GPU 메모리(16 GB)를 넘는다. -->
  <include file="mjx_go2_robot_only.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>
  <visual>
    <map zfar="100"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="-130" elevation="-20" offwidth="1263" offheight="1080"/>
  </visual>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="obstacle" rgba="0.55 0.55 0.6 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{chr(10).join(geoms)}
  </worldbody>
</mujoco>
"""
    return xml, len(rects)


def dial_yaml(scene: str, n_steps: int) -> str:
    return f"""# DIAL-MPC — indoor/build_scene.py 자동 생성
seed: 0
output_dir: data/{scene}
n_steps: {n_steps}

env_name: unitree_go2_walk
Nsample: 2048
Hsample: 16
Hnode: 4
Ndiffuse: 2
Ndiffuse_init: 10
temp_sample: 0.05
horizon_diffuse_factor: 0.9
traj_diffuse_factor: 0.5
update_method: mppi

dt: 0.02
timestep: 0.02
leg_control: torque
action_scale: 1.0

default_vx: 0.8
default_vy: 0.0
default_vyaw: 0.0
ramp_up_time: 1.0
gait: trot

scene_xml: {scene}.xml
vlm_path_json: data/{scene}/last_judged_path.json
"""


def n_steps_for(path_len_m: float) -> int:
    return int(np.ceil(path_len_m / CRUISE_SPEED / DT)) + SETTLE_STEPS


# ─── 이미지 ───────────────────────────────────────────────────────────────
def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_overlay(bev: np.ndarray, ppm: float, robot_px, goal_xy) -> Image.Image:
    """oracle_gen.py와 같은 격자(0.5 m 보조선, 1 m 라벨) + 시작/목표 마커."""
    H, W = bev.shape[:2]
    u0, v0 = robot_px
    w2p = lambda x, y: (u0 + x * ppm, v0 - y * ppm)
    base = Image.fromarray(bev).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f = _font(16)
    for x in np.arange(np.floor(-u0 / ppm / 0.5) * 0.5, (W - u0) / ppm + 1e-9, 0.5):
        u, _ = w2p(x, 0)
        axis = abs(x) < 1e-6
        d.line([(u, 0), (u, H)], fill=(255, 255, 0, 160) if axis else (255, 255, 255, 90), width=2 if axis else 1)
        if abs(x - round(x)) < 1e-6:
            d.text((u + 3, 4), f"{x:+.1f}m", font=f, fill=(255, 255, 0, 220))
    for y in np.arange(np.floor((v0 - H) / ppm / 0.5) * 0.5, v0 / ppm + 1e-9, 0.5):
        _, v = w2p(0, y)
        axis = abs(y) < 1e-6
        d.line([(0, v), (W, v)], fill=(255, 255, 0, 160) if axis else (255, 255, 255, 90), width=2 if axis else 1)
        if abs(y - round(y)) < 1e-6:
            d.text((4, v + 3), f"{y:+.1f}m", font=f, fill=(255, 255, 0, 220))
    r = 0.18 * ppm
    d.ellipse([u0 - r, v0 - r, u0 + r, v0 + r], outline=(40, 120, 255, 255), fill=(40, 120, 255, 120), width=3)
    d.line([(u0, v0), (u0 + 2.2 * r, v0)], fill=(40, 120, 255, 255), width=4)          # 전방(+x) 화살표
    gu, gv = w2p(*goal_xy)
    d.ellipse([gu - r, gv - r, gu + r, gv + r], outline=(40, 220, 80, 255), fill=(40, 220, 80, 120), width=3)
    d.text((gu + r + 4, gv - r), "G", font=_font(22), fill=(40, 220, 80, 255))
    return Image.alpha_composite(base, ov).convert("RGB")


def align_check(bev: np.ndarray, grid: OccGrid, ppm: float, robot_px) -> Image.Image:
    """GT 점유 셀을 BEV 픽셀로 투영해 빨갛게 겹친다 → Unity 렌더와 점유 지도가 같은 공간인지 눈으로 확인."""
    H, W = bev.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    x = (uu - robot_px[0]) / ppm
    y = (robot_px[1] - vv) / ppm
    occ = grid.occupied(np.stack([x, y], -1))
    out = bev.astype(float).copy()
    out[occ] = 0.45 * out[occ] + 0.55 * np.array([255, 0, 0])
    return Image.fromarray(out.astype(np.uint8))


# ─── 메인 ────────────────────────────────────────────────────────────────
def free_at(grid: OccGrid, xy, radius: float) -> bool:
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = [np.asarray(xy, float)] + [np.asarray(xy) + radius * np.array([np.cos(t), np.sin(t)]) for t in th]
    return not grid.occupied(np.array(pts)).any()


def build(export_dir: str, mj_cell: float | None = None, overwrite: bool = True) -> dict:
    meta, bev, grid = load_export(export_dir)
    scene = meta["scene_name"]
    if mj_cell:
        grid = coarsen(grid, mj_cell)
    goal = tuple(meta["goal"])
    ppm, robot_px = float(meta["ppm"]), tuple(meta["robot_px"])

    warnings = []
    if not free_at(grid, (0.0, 0.0), 0.2):
        warnings.append("시작점 반경 0.2 m 안에 장애물이 있음")
    if not free_at(grid, goal, 0.2):
        warnings.append("목표점 반경 0.2 m 안에 장애물이 있음")
    if not grid.occ.any():
        warnings.append("점유 셀이 0개 (Collider/레이어/Band 확인)")
    if not (meta["x_min"] + 0.5 <= goal[0] <= meta["x_max"] - 0.5 and meta["y_min"] + 0.5 <= goal[1] <= meta["y_max"] - 0.5):
        warnings.append(f"목표점 {goal[0]:.2f}, {goal[1]:.2f} 이 이미지 범위 밖 (Go2Start의 파란 축을 목표 쪽으로 돌릴 것)")
    elif goal[0] < 0.5 * abs(goal[1]):
        warnings.append("목표점이 로봇 전방이 아님 (Go2Start의 파란 축을 목표 쪽으로 돌릴 것)")

    out = os.path.join(DATA_DIR, scene)
    os.makedirs(out, exist_ok=True)
    Image.fromarray(bev).save(os.path.join(out, "bev_raw.png"))
    draw_overlay(bev, ppm, robot_px, goal).save(os.path.join(out, "oracle.png"))
    align_check(bev, grid, ppm, robot_px).save(os.path.join(out, "align_check.png"))
    grid.save(os.path.join(out, "occupancy.npz"))

    xml, n_boxes = mujoco_xml(scene, grid)
    open(os.path.join(MODELS_DIR, f"{scene}.xml"), "w").write(xml)
    straight = float(np.hypot(*goal))
    n_steps = n_steps_for(1.5 * straight)                 # 기본값. 실제 실행은 경로 길이로 다시 계산
    open(os.path.join(EXAMPLES_DIR, f"{scene}.yaml"), "w").write(dial_yaml(scene, n_steps))

    scene_meta = {
        "scene": scene, "source": "unity", "unity_scene": meta.get("unity_scene"),
        "goal": list(goal), "ppm": ppm, "robot_px": list(robot_px),
        "img_w": meta["img_w"], "img_h": meta["img_h"],
        "occ_cell_m": grid.res, "band_m": meta["band"], "n_obstacle_boxes": n_boxes,
        "straight_dist_m": straight, "default_n_steps": n_steps,
        # 웨이포인트 수: 우회 여유 1.4배 경로를 권장 간격 0.7 m로 나눈 값 (8~20개로 제한)
        "num_waypoints": int(np.clip(np.ceil(1.4 * straight / 0.7), 8, 20)),
        "scenario": SCENARIO_TEMPLATE.format(gx=goal[0], gy=goal[1]),
        "warnings": warnings,
    }
    json.dump(scene_meta, open(os.path.join(out, "scene_meta.json"), "w"), indent=2, ensure_ascii=False)
    return scene_meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", help="Go2Export/<scene> 폴더(들)")
    ap.add_argument("--mj-cell", type=float, default=None, help="장애물/GT 격자 크기 (m). 기본 = Unity cell 그대로")
    args = ap.parse_args()
    for e in args.exports:
        if not os.path.exists(os.path.join(e, "meta.json")):
            print(f"⚠️  {e}: meta.json 없음 → 건너뜀")
            continue
        m = build(e, args.mj_cell)
        flag = "  ⚠️ " + "; ".join(m["warnings"]) if m["warnings"] else ""
        print(f"✅ {m['scene']}: goal={tuple(round(v, 2) for v in m['goal'])}, boxes={m['n_obstacle_boxes']}, "
              f"cell={m['occ_cell_m']} m{flag}")


if __name__ == "__main__":
    main()
