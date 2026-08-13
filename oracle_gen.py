"""
oracle_gen.py  ─ HELM-Brains 오라클 렌더러 (순수 렌더러 버전)
================================================================
철학: 씬 내용(장애물)은 XML 이 진실. 이 스크립트는 카메라만 꽂고 top-down 으로 찍는다.
      → env 가 로드하는 씬 XML == 여기서 렌더한 씬 XML 이면 이미지=물리 정합 보장.

대상 씬: models/unitree_go2/oracle_scene_*.xml  (인자로 특정 파일 지정도 가능)
  씬 작성법: <include file="mjx_go2_robot_only.xml"/> + 바닥 + <worldbody> 안에
            원하는 장애물 body/geom 을 자유롭게 배치. (oracle_scene_A.xml 참고)

캘리브레이션 (main_court.py / courtroom.py 규약과 일치):
  - 이미지 1263 x 1080, 로봇 원점 = 화면 중앙 (631.5, 540)
  - 150 px/m. orthographic 속성은 MuJoCo 버전따라 없어서, 카메라를 높이(50m) 올리고
    fovy 를 좁혀(망원) orthographic 을 근사. 중심 150px/m 정확, 가장자리 왜곡 <0.5%.
  - 좌표 규약: 전방(+x)=화면 오른쪽(+u), +y=화면 위(-v)
    → 카메라 xyaxes = "1 0 0 0 1 0"

Grid overlay (신규):
  - VLM의 시각적 거리 추정 오차를 줄이기 위해, 렌더링 후 0.5m 간격 보조선 +
    1m 간격 라벨(반투명, 로봇 원점 기준 world 좌표)을 이미지에 합성한다.
  - 텍스트 프롬프트에 숫자 힌트를 넣는 게 아니라 "이미지 자체의 자를 제공"하는
    방식이라 Contribution #2(최소 정보)와 상충하지 않는다는 설계 의도.
  - `--no-grid`로 끌 수 있음 (기존 raw 이미지와 비교하고 싶을 때).

실행:
  Mac:    python oracle_gen.py                    # 전체 oracle_scene_*.xml
          python oracle_gen.py oracle_scene_A.xml # 특정 씬만
          python oracle_gen.py oracle_scene_A.xml --no-grid   # grid 없이
  리눅스: MUJOCO_GL=egl python oracle_gen.py
================================================================
"""
import os, sys, glob, argparse
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

# ─── 경로 ───────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__),
                          "dial_mpc", "dial_mpc", "models", "unitree_go2")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
SCENE_GLOB = "oracle_scene_*.xml"        # 인자 없을 때 렌더할 씬 패턴

# ─── 캘리브레이션 ───────────────────────────────────────────────────────
IMG_W, IMG_H = 1263, 1080
PPM_DEFAULT  = 150.0                     # px per meter (기본값)
# 씬별로 카메라 시야를 넓혀야 할 때만 등록. main_court.py 의 SCENE_SCALE 과 반드시 동일값 유지.
SCENE_PPM = {
    "oracle_scene_C": 90.0,
    "oracle_scene_E": 90.0,
    "oracle_scene_D": 90.0,
}
ROBOT_PX     = (IMG_W / 3, IMG_H / 2)    # 로봇을 화면 좌측 1/3에 배치 -> 전방 시야 확보 (기존 중앙 631.5 -> 421)
CAM_HEIGHT   = 50.0                      # 높을수록 orthographic 근접

# ─── Grid overlay 설정 ────────────────────────────────────────────────
GRID_MINOR_SPACING_M = 0.5   # 보조선 간격 (m)
GRID_LABEL_SPACING_M = 1.0   # 숫자 라벨 간격 (m)
GRID_LINE_RGBA  = (255, 255, 255, 90)   # 반투명 흰색 보조선
GRID_AXIS_RGBA  = (255, 255, 0, 160)    # x=0 / y=0 축선은 더 진하게(노란색)
GRID_LABEL_RGBA = (255, 255, 0, 220)    # 라벨 텍스트 색

def _fovy_for_ppm(ppm):
    v = IMG_H / ppm
    return 2 * np.degrees(np.arctan((v / 2) / CAM_HEIGHT))

# 하위호환: 모듈 로드 시 기본값으로 초기화 (world_to_pixel 등에서 참조)
PPM = PPM_DEFAULT
FOVY_DEG = _fovy_for_ppm(PPM)

# ═══════════════════════════════════════════════════════════════════════
def world_to_pixel(wx, wy):
    """월드(x=전방, y=좌우) → 픽셀. +x=오른쪽, +y=위."""
    return ROBOT_PX[0] + wx * PPM, ROBOT_PX[1] - wy * PPM

def _inject_camera(scene_path, fovy_deg):
    """씬 XML 에 top-down 오라클 카메라를 주입한 임시 파일을 같은 폴더에 생성."""
    txt = open(scene_path).read()
    # 카메라를 world (offset_x, offset_y) 만큼 이동 -> 로봇(world 0,0)이 화면의 ROBOT_PX 위치에 찍힘
    # 화면 중앙 대비 ROBOT_PX 가 왼쪽/위로 치우친 만큼, 카메라는 world 상에서 그 반대(오른쪽/아래)로 이동해야 함
    offset_x = (IMG_W / 2 - ROBOT_PX[0]) / PPM   # 픽셀 오프셋 -> 미터 환산
    offset_y = -(IMG_H / 2 - ROBOT_PX[1]) / PPM
    cam = (f'    <camera name="oracle" mode="fixed" '
           f'pos="{offset_x:.6f} {offset_y:.6f} {CAM_HEIGHT}" xyaxes="1 0 0 0 1 0" '
           f'fovy="{fovy_deg:.6f}"/>\n')
    if 'name="oracle"' not in txt:
        txt = txt.replace("</worldbody>", cam + "  </worldbody>")
    tmp = os.path.join(MODELS_DIR, "_oracle_tmp.xml")
    open(tmp, "w").write(txt)
    return tmp

def _report_obstacles(model):
    """정적 장애물(관절 없는, worldbody 직속 body)의 월드좌표→픽셀 출력."""
    for bid in range(1, model.nbody):
        if model.body_jntnum[bid] == 0 and model.body_parentid[bid] == 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
            wx, wy, _ = model.body_pos[bid]
            u, v = world_to_pixel(wx, wy)
            inside = "OK" if (0 <= u <= IMG_W and 0 <= v <= IMG_H) else "⚠️화면밖"
            print(f"     · {name:12s} world({wx:+.2f},{wy:+.2f}) → pixel({u:.0f},{v:.0f}) [{inside}]")

def _draw_grid_overlay(img_array):
    """
    렌더링된 numpy 이미지 위에 world-aligned 0.5m 격자 + 1m 라벨을 반투명 합성한다.
    world_to_pixel()의 현재 PPM/ROBOT_PX(둘 다 render_scene에서 씬별로 설정됨)를 그대로 사용하므로,
    격자가 실제 물리 좌표와 항상 정합된다.
    """
    base = Image.fromarray(img_array).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # 화면에 보이는 world 범위 역산
    wx_min = (0 - ROBOT_PX[0]) / PPM
    wx_max = (IMG_W - ROBOT_PX[0]) / PPM
    wy_min = (ROBOT_PX[1] - IMG_H) / PPM
    wy_max = (ROBOT_PX[1] - 0) / PPM

    def _snap(v, spacing):
        return np.floor(v / spacing) * spacing

    # 세로선 (world x = const)
    x = _snap(wx_min, GRID_MINOR_SPACING_M)
    while x <= wx_max:
        u, _ = world_to_pixel(x, 0)
        is_axis = abs(x) < 1e-6
        draw.line([(u, 0), (u, IMG_H)], fill=GRID_AXIS_RGBA if is_axis else GRID_LINE_RGBA,
                   width=2 if is_axis else 1)
        if abs(x / GRID_LABEL_SPACING_M - round(x / GRID_LABEL_SPACING_M)) < 1e-6:
            draw.text((u + 3, 4), f"{x:+.1f}m", font=font, fill=GRID_LABEL_RGBA)
        x += GRID_MINOR_SPACING_M

    # 가로선 (world y = const)
    y = _snap(wy_min, GRID_MINOR_SPACING_M)
    while y <= wy_max:
        _, v = world_to_pixel(0, y)
        is_axis = abs(y) < 1e-6
        draw.line([(0, v), (IMG_W, v)], fill=GRID_AXIS_RGBA if is_axis else GRID_LINE_RGBA,
                   width=2 if is_axis else 1)
        if abs(y / GRID_LABEL_SPACING_M - round(y / GRID_LABEL_SPACING_M)) < 1e-6:
            draw.text((4, v + 3), f"{y:+.1f}m", font=font, fill=GRID_LABEL_RGBA)
        y += GRID_MINOR_SPACING_M

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(composited)

def render_scene(scene_path, draw_grid=True):
    global PPM, FOVY_DEG
    stem = os.path.splitext(os.path.basename(scene_path))[0]
    # _viz 접미사가 붙은 씬도 원본 씬과 동일한 PPM을 쓰도록 조회
    lookup_stem = stem[:-4] if stem.endswith("_viz") else stem
    PPM = PPM_DEFAULT
    for scene_key, ppm_val in SCENE_PPM.items():
        if lookup_stem == scene_key or lookup_stem.startswith(scene_key + "_"):
            PPM = ppm_val
            break
    FOVY_DEG = _fovy_for_ppm(PPM)
    scene_out_dir = os.path.join(DATA_DIR, stem)
    os.makedirs(scene_out_dir, exist_ok=True)
    tmp = _inject_camera(scene_path, FOVY_DEG)
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
        data  = mujoco.MjData(model); mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=IMG_H, width=IMG_W) as r:
            r.update_scene(data, camera="oracle")
            img = r.render()
        if draw_grid:
            img = _draw_grid_overlay(img)
        out = os.path.join(scene_out_dir, "oracle.png")
        Image.fromarray(img).save(out)
        print(f"  ✅ {out}" + ("  [grid]" if draw_grid else "  [no-grid]"))
        _report_obstacles(model)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="*", help="특정 oracle_scene_*.xml 파일명(들). 없으면 전체.")
    parser.add_argument("--no-grid", action="store_true", help="0.5m 격자 오버레이 끄기 (raw 이미지)")
    args = parser.parse_args()

    if args.scenes:
        scenes = [os.path.join(MODELS_DIR, a) if not os.path.isabs(a) else a
                  for a in args.scenes]
    else:
        scenes = sorted(glob.glob(os.path.join(MODELS_DIR, SCENE_GLOB)))

    if not scenes:
        print(f"❌ 렌더할 씬 없음. {MODELS_DIR}/{SCENE_GLOB} 를 만들거나 인자로 지정해.")
        return

    print(f"[Calib] {IMG_W}x{IMG_H}, {PPM}px/m, cam H={CAM_HEIGHT}m fovy={FOVY_DEG:.3f}deg, 로봇중심={ROBOT_PX}")
    print(f"[Grid] {'ON (0.5m minor / 1m label)' if not args.no_grid else 'OFF'}")
    print(f"[Render] {len(scenes)}개 씬")
    for sp in scenes:
        if not os.path.exists(sp):
            print(f"  ⚠️ 없음: {sp}"); continue
        render_scene(sp, draw_grid=not args.no_grid)
    print(f"\n출력: {DATA_DIR}/<scene_name>/oracle.png")

if __name__ == "__main__":
    main()