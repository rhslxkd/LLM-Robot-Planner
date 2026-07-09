"""
run_experiment_c.py ─ 실험 C 요약표 생성

여러 오라클 씬(oracle_scene_A, B, C ...)의 rollout 결과를 모아
data/experiment_C_summary.csv 로 저장한다. 이 CSV가 논문 실험 C 표의 원본.

실행 (레포 어디서든 가능, 경로는 항상 레포 루트 data/ 기준으로 앵커됨):
    python analysis/run_experiment_c.py
    python analysis/run_experiment_c.py --scenes oracle_scene_A oracle_scene_B
"""
import argparse
import os
import pandas as pd

from load_rollout import load_latest_rollout, DATA_ROOT, REPO_ROOT
from metrics import summarize

DEFAULT_SCENES = ["oracle_scene_A", "oracle_scene_B", "oracle_scene_C"]


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
        raise RuntimeError("처리된 씬이 하나도 없음. data/<scene>/*_states.npy 존재 여부 확인.")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes", nargs="+", default=DEFAULT_SCENES,
        help="집계할 씬 이름 목록 (기본: oracle_scene_A B C)",
    )
    args = parser.parse_args()

    df = run(args.scenes)

    if len(df) == 1:
        # 씬이 하나뿐이면 그 씬 폴더 안에 저장
        scene_name = df.iloc[0]["scene"]
        out_path = os.path.join(DATA_ROOT, scene_name, "experiment_C_summary.csv")
    else:
        # 여러 씬이 섞이면 data/ 루트에 저장
        out_path = os.path.join(DATA_ROOT, "experiment_C_summary.csv")

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()