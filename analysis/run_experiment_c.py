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
import pandas as pd

from load_rollout import load_latest_rollout, DATA_ROOT, REPO_ROOT
from metrics import summarize

# Gemini 기준 Table 4용 기본 씬 목록 (중첩 경로: <scene>/<variant>).
DEFAULT_SCENES = ["oracle_scene_A/gemini", "oracle_scene_D/gemini", "oracle_scene_E/gemini"]


def run(scenes: list) -> pd.DataFrame:
    results = []
    for scene in scenes:
        try:
            data = load_latest_rollout(scene)
        except FileNotFoundError as e:
            print(f"⚠️  건너뜀: {e}")
            continue
        row = {"scene": scene}
        row.update(summarize(data))
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