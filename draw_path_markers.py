"""
draw_path_markers.py ─ VLM 경로를 씬 XML + YAML에 시각 마커로 삽입

data/<scene>/last_judged_path.json 의 waypoint 들을
씬 XML 복제본(oracle_scene_X_viz.xml)에 노란 구슬 + 빨간 연결선으로 그려넣고,
바로 실행 가능한 YAML(oracle_scene_X_viz.yaml)까지 함께 생성한다.
마커는 contype=0 conaffinity=0 이라 물리에 전혀 영향 없음 (순수 시각용).

사용법:
    python draw_path_markers.py oracle_scene_A
    -> models/unitree_go2/oracle_scene_A_viz.xml 생성
    -> examples/oracle_scene_A_viz.yaml 생성
    -> python dial_mpc/dial_mpc/core/dial_core.py --example oracle_scene_A_viz 로 바로 실행
"""
import os, sys, json

REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(REPO_ROOT, "dial_mpc", "dial_mpc", "models", "unitree_go2")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "dial_mpc", "dial_mpc", "examples")
DATA_ROOT    = os.path.join(REPO_ROOT, "data")

MARKER_RADIUS = 0.04
LINE_RADIUS   = 0.012
MARKER_HEIGHT = 0.02
MARKER_RGBA   = "1.0 0.9 0.0 0.95"   # 노란 waypoint 점 (matplotlib 검증 이미지와 통일)
LINE_RGBA     = "0.9 0.1 0.1 0.85"   # 빨간 연결선


def build_marker_xml(path_points):
    lines = ["", "    <!-- ===== VLM 경로 시각 마커 (자동 생성, 물리 영향 없음) ===== -->"]
    for i, (x, y) in enumerate(path_points):
        lines.append(
            f'    <geom name="wp_{i}" type="sphere" size="{MARKER_RADIUS}" '
            f'pos="{x:.4f} {y:.4f} {MARKER_HEIGHT}" rgba="{MARKER_RGBA}" '
            f'contype="0" conaffinity="0"/>'
        )
    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]
        lines.append(
            f'    <geom name="wpline_{i}" type="capsule" size="{LINE_RADIUS}" '
            f'fromto="{x1:.4f} {y1:.4f} {MARKER_HEIGHT} {x2:.4f} {y2:.4f} {MARKER_HEIGHT}" '
            f'rgba="{LINE_RGBA}" contype="0" conaffinity="0"/>'
        )
    lines.append("    <!-- ===== 경로 마커 끝 ===== -->")
    return "\n".join(lines)


def make_viz_yaml(scene):
    """원본 YAML을 복사해 scene_xml, output_dir 만 viz 버전으로 교체."""
    src = os.path.join(EXAMPLES_DIR, f"{scene}.yaml")
    dst = os.path.join(EXAMPLES_DIR, f"{scene}_viz.yaml")
    assert os.path.exists(src), f"원본 YAML 없음: {src}"

    txt = open(src).read()
    txt = txt.replace(f"scene_xml: {scene}.xml", f"scene_xml: {scene}_viz.xml")
    txt = txt.replace(f"output_dir: data/{scene}", f"output_dir: data/{scene}_viz")
    open(dst, "w").write(txt)
    return dst


def main():
    if len(sys.argv) < 2:
        print("사용법: python draw_path_markers.py <scene_name>  (예: oracle_scene_A)")
        return
    scene = sys.argv[1]

    json_path = os.path.join(DATA_ROOT, scene, "last_judged_path.json")
    scene_xml = os.path.join(MODELS_DIR, f"{scene}.xml")
    out_xml   = os.path.join(MODELS_DIR, f"{scene}_viz.xml")

    assert os.path.exists(json_path), f"경로 JSON 없음: {json_path} (courtroom 먼저 실행할 것)"
    assert os.path.exists(scene_xml), f"씬 XML 없음: {scene_xml}"

    with open(json_path) as f:
        path = json.load(f)
    pts = [(p["x"], p["y"]) for p in path]
    print(f"[{scene}] waypoint {len(pts)}개 로드")

    txt = open(scene_xml).read()
    marker_block = build_marker_xml(pts)
    assert "</worldbody>" in txt, "worldbody 닫는 태그를 못 찾음"
    txt = txt.replace("</worldbody>", marker_block + "\n  </worldbody>")

    open(out_xml, "w").write(txt)
    print(f"✅ XML 생성: {out_xml}")

    yaml_path = make_viz_yaml(scene)
    print(f"✅ YAML 생성: {yaml_path}")
    print(f"   실행: python dial_mpc/dial_mpc/core/dial_core.py --example {scene}_viz")


if __name__ == "__main__":
    main()
