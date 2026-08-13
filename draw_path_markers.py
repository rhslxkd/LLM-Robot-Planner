"""
draw_path_markers.py ─ VLM 경로를 씬 XML + YAML에 시각 마커로 삽입

data/<scene>/<variant>/last_judged_path.json 의 waypoint 들을
씬 XML 복제본(<scene>_<variant>_viz.xml)에 노란 구슬 + 빨간 연결선으로 그려넣고,
바로 실행 가능한 YAML(<scene>_<variant>_viz.yaml)까지 함께 생성한다.
마커는 contype=0 conaffinity=0 이라 물리에 전혀 영향 없음 (순수 시각용).

variant는 courtroom.py/main_court.py가 쓰는 백엔드/모델 식별자와 동일하게 맞춘다
(예: "gemini", "ollama_qwen2_5vl_7b"). 같은 씬을 여러 백엔드/모델로 돌려도
파일명이 겹치지 않도록 XML/YAML/output_dir 전부 <scene>_<variant> 조합으로 구분한다.

사용법:
    python draw_path_markers.py oracle_scene_A gemini
    -> models/unitree_go2/oracle_scene_A_gemini_viz.xml 생성
    -> examples/oracle_scene_A_gemini_viz.yaml 생성 (output_dir: data/oracle_scene_A/gemini/viz)
    -> python dial_mpc/dial_mpc/core/dial_core.py --example oracle_scene_A_gemini_viz 로 바로 실행

    python draw_path_markers.py oracle_scene_A ollama_qwen2_5vl_7b
    -> 위와 동일한 패턴으로 별도 파일 생성, 서로 덮어쓰지 않음
"""
import os, sys, json

REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(REPO_ROOT, "dial_mpc", "dial_mpc", "models", "unitree_go2")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "dial_mpc", "dial_mpc", "examples")
DATA_ROOT    = os.path.join(REPO_ROOT, "data")

MARKER_RADIUS = 0.09
LINE_RADIUS   = 0.030
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


def make_viz_yaml(scene, variant, viz_tag):
    """원본 YAML(scene 기준, 백엔드와 무관한 공용 템플릿)을 복사해
    scene_xml, output_dir 만 <scene>_<variant>_viz 버전으로 교체."""
    src = os.path.join(EXAMPLES_DIR, f"{scene}.yaml")
    dst = os.path.join(EXAMPLES_DIR, f"{viz_tag}.yaml")
    assert os.path.exists(src), f"원본 YAML 없음: {src}"

    txt = open(src).read()
    txt = txt.replace(f"scene_xml: {scene}.xml", f"scene_xml: {viz_tag}.xml")
    # output_dir은 courtroom.py의 중첩 규칙(data/<scene>/<variant>/)을 그대로 따라가되,
    # 시각화 결과는 그 아래 viz/ 서브폴더에 따로 모아서 last_judged_path.json 등과 안 섞이게 한다.
    txt = txt.replace(f"output_dir: data/{scene}", f"output_dir: data/{scene}/{variant}/viz")
    # vlm_path_json도 같은 variant 폴더를 가리키도록 맞춘다. 안 그러면 XML에 그려진 마커(variant 경로)와
    # 실제 planner가 로드하는 경로(원본 scene 경로)가 서로 달라지는 불일치가 생긴다.
    txt = txt.replace(
        f"vlm_path_json: data/{scene}/last_judged_path.json",
        f"vlm_path_json: data/{scene}/{variant}/last_judged_path.json",
    )
    open(dst, "w").write(txt)
    return dst


def main():
    if len(sys.argv) < 3:
        print("사용법: python draw_path_markers.py <scene_name> <variant>")
        print("  예: python draw_path_markers.py oracle_scene_A gemini")
        print("      python draw_path_markers.py oracle_scene_A ollama_qwen2_5vl_7b")
        return
    scene = sys.argv[1]
    variant = sys.argv[2]
    viz_tag = f"{scene}_{variant}_viz"  # 파일명 충돌 방지용 조합 태그

    json_path = os.path.join(DATA_ROOT, scene, variant, "last_judged_path.json")
    scene_xml = os.path.join(MODELS_DIR, f"{scene}.xml")
    out_xml   = os.path.join(MODELS_DIR, f"{viz_tag}.xml")

    assert os.path.exists(json_path), (
        f"경로 JSON 없음: {json_path} "
        f"(main_court.py --scene {scene} --backend ... 먼저 실행할 것)"
    )
    assert os.path.exists(scene_xml), f"씬 XML 없음: {scene_xml}"

    with open(json_path) as f:
        path = json.load(f)
    pts = [(p["x"], p["y"]) for p in path]
    print(f"[{scene}/{variant}] waypoint {len(pts)}개 로드")

    txt = open(scene_xml).read()
    marker_block = build_marker_xml(pts)
    assert "</worldbody>" in txt, "worldbody 닫는 태그를 못 찾음"
    txt = txt.replace("</worldbody>", marker_block + "\n  </worldbody>")

    open(out_xml, "w").write(txt)
    print(f"✅ XML 생성: {out_xml}")

    yaml_path = make_viz_yaml(scene, variant, viz_tag)
    print(f"✅ YAML 생성: {yaml_path}")
    print(f"   실행: python dial_mpc/dial_mpc/core/dial_core.py --example {viz_tag}")


if __name__ == "__main__":
    main()