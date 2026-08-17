"""
run_experiment_c.py ─ 실험 C 요약표 생성

여러 오라클 씬(oracle_scene_A, D, E ...)의 rollout 결과를 모아
data/experiment_C_summary.csv 로 저장한다. 이 CSV가 논문 실험 C 표의 원본.

씬/백엔드 폴더 구조는 main_court.py의 하위 폴더 규칙을 따른다:
  data/<scene>/<variant>/          (예: data/oracle_scene_A/gemini/,
                                         data/oracle_scene_A/ollama_qwen2_5vl_7b/)
load_latest_rollout()은 넘겨받은 문자열을 그대로 경로 조각으로 이어붙이므로,
"oracle_scene_A/gemini"처럼 슬래시 포함 문자열을 넘기면 별도 코드 수정 없이
중첩 폴더를 그대로 찾아간다.

⚠️ 확인 필요: DIAL-MPC(dial_core.py) 쪽 output_dir 설정도 이 중첩 구조
(data/<scene>/<variant>/)에 맞춰 states.npy를 저장하도록 별도로 맞춰야 한다.
courtroom.py 쪽 출력 경로만 바꿔서는 DIAL-MPC가 만드는 states.npy 위치까지
자동으로 안 맞을 수 있음 — DIAL-MPC 실행용 YAML/설정 파일 확인 필요.

obstacle_clearance 지표(충돌 탐지)를 계산하려면 실제 장애물 기하가 필요한데,
scene 문자열의 앞부분("oracle_scene_D/gemini_..." -> "oracle_scene_D")으로
dial_mpc/dial_mpc/models/unitree_go2/<base_scene>.xml 을 열어서
obstacles_from_mjmodel()로 실측한다 (mujoco 필요 -- 반드시 lab 서버에서 실행).
같은 base scene은 캐시해서 한 번만 로드한다.

실행 (레포 어디서든 가능, 경로는 항상 레포 루트 data/ 기준으로 앵커됨):
    python analysis/run_experiment_c.py
    python analysis/run_experiment_c.py --scenes oracle_scene_A/gemini oracle_scene_D/gemini
    # 모델 비교(Table 5/6)용 예시:
    python analysis/run_experiment_c.py --scenes \
        oracle_scene_A/ollama_qwen2_5vl_7b \
        oracle_scene_A/ollama_llava_13b \
        oracle_scene_A/ollama_llama3_2-vision_11b
"""
import argparse
import os
import sys
import pandas as pd

from load_rollout import load_latest_rollout, DATA_ROOT, REPO_ROOT
from metrics import summarize

# baselines/occupancy_grid.py의 obstacles_from_mjmodel()를 재사용 (실제 XML 기반
# 장애물 기하 -- 프롬프트 텍스트에서 좌표를 추측하지 않고 real ground truth를 쓴다).
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines"))
from occupancy_grid import obstacles_from_mjmodel

# Gemini 기준 Table 4용 기본 씬 목록 (중첩 경로: <scene>/<variant>).
DEFAULT_SCENES = ["oracle_scene_A/gemini", "oracle_scene_D/gemini", "oracle_scene_E/gemini"]

_obstacle_cache = {}


def _get_obstacles(scene_variant: str):
    """
    scene_variant: 'oracle_scene_D/gemini_..._ablation_rich' 형태 -- 슬래시 앞
    base scene 이름만 떼서 실제 XML을 로드해 장애물 목록을 가져온다.
    XML이 없거나 mujoco import가 실패하면 None을 반환(그 경우 summarize()가
    obstacle_clearance 없이 기존 지표만 계산하도록 상위에서 처리).
    """
    base_scene = scene_variant.split("/")[0]
    if base_scene in _obstacle_cache:
        return _obstacle_cache[base_scene]

    xml_path = os.path.join(REPO_ROOT, "dial_mpc", "dial_mpc", "models",
                             "unitree_go2", f"{base_scene}.xml")
    if not os.path.exists(xml_path):
        print(f"⚠️  {xml_path} 없음 -- {base_scene} obstacle_clearance 계산 스킵")
        _obstacle_cache[base_scene] = None
        return None

    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(xml_path)
        obstacles = obstacles_from_mjmodel(model)
    except Exception as e:
        print(f"⚠️  {base_scene} 장애물 로드 실패({e}) -- obstacle_clearance 계산 스킵")
        obstacles = None

    _obstacle_cache[base_scene] = obstacles
    return obstacles


def run(scenes: list) -> pd.DataFrame:
    results = []
    for scene in scenes:
        try:
            data = load_latest_rollout(scene)
        except FileNotFoundError as e:
            print(f"⚠️  건너뜀: {e}")
            continue
        obstacles = _get_obstacles(scene)
        row = {"scene": scene}
        row.update(summarize(data, obstacles=obstacles))
        results.append(row)
        print(f"✅ {scene} 처리 완료")

    if not results:
        raise RuntimeError("처리된 씬이 하나도 없음. data/<scene>/<variant>/*_states.npy 존재 여부 확인.")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes", nargs="+", default=DEFAULT_SCENES,
        help="집계할 씬 이름 목록, '<scene>/<variant>' 형식 (기본: oracle_scene_A/D/E의 gemini variant)",
    )
    args = parser.parse_args()

    df = run(args.scenes)

    if len(df) == 1:
        # 씬이 하나뿐이면 그 씬/variant 폴더 안에 저장
        scene_name = df.iloc[0]["scene"]
        out_path = os.path.join(DATA_ROOT, scene_name, "experiment_C_summary.csv")
    else:
        # 여러 씬이 섞이면 data/ 루트에 저장
        out_path = os.path.join(DATA_ROOT, "experiment_C_summary.csv")

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()