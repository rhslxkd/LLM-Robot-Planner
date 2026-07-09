"""
load_rollout.py ─ dial_core.py 가 저장한 {timestamp}_states.npy 로더

states.npy 컬럼 구조 (총 68열, dial_core.py 의 저장 순서와 정확히 일치):
    [0]      step index
    [1:20]   qpos            (19) - freejoint(7) + 관절각(12)
    [20:38]  qvel            (18) - freejoint(6) + 관절각속도(12)
    [38:50]  ctrl            (12) - 명령 토크 (액추에이터 12개)
    [50:68]  qfrc_actuator   (18) - 실제 인가 토크 (freejoint(6, 항상 0) + 관절(12))

qfrc_actuator 를 ctrl 과 짝지어 비교할 때는 qfrc_actuator[:, 6:18] 을 써야 한다
(앞 6개는 freejoint 성분이라 항상 0에 가깝다).
"""
import os
import glob
import numpy as np

# 레포 루트/data 를 실행 위치와 무관하게 항상 올바르게 찾는다
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
DATA_ROOT = os.path.join(REPO_ROOT, "data")

COLS = {
    "step": (0, 1),
    "qpos": (1, 20),
    "qvel": (20, 38),
    "ctrl": (38, 50),
    "qfrc_actuator": (50, 68),
}


def list_rollout_files(scene_name: str, data_root: str = DATA_ROOT):
    """씬 폴더 안의 모든 {timestamp}_states.npy 경로를 시간순으로 반환."""
    pattern = os.path.join(data_root, scene_name, "*_states.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"'{scene_name}' 에서 *_states.npy 를 찾지 못함. "
            f"경로 확인: {pattern}"
        )
    return files


def load_rollout(states_path: str) -> dict:
    """states.npy 경로 하나를 읽어 컬럼별로 분리한 dict 로 반환."""
    d = np.load(states_path)
    if d.shape[1] != 68:
        raise ValueError(
            f"예상 컬럼 수(68)와 다름: {states_path} 의 shape={d.shape}. "
            "dial_core.py 저장 포맷이 바뀌었는지 확인."
        )
    return {name: d[:, s:e] for name, (s, e) in COLS.items()}


def load_latest_rollout(scene_name: str, data_root: str = DATA_ROOT) -> dict:
    """씬의 가장 최근 rollout 을 읽어온다. 여러 번 실행했다면 최신 것만 사용."""
    files = list_rollout_files(scene_name, data_root)
    return load_rollout(files[-1])


def load_all_rollouts(scene_name: str, data_root: str = DATA_ROOT) -> list:
    """씬에 실행 이력이 여러 개면 전부(시간순) 반환. 반복 실험 비교용."""
    files = list_rollout_files(scene_name, data_root)
    return [load_rollout(f) for f in files]


if __name__ == "__main__":
    import sys

    scene = sys.argv[1] if len(sys.argv) > 1 else "oracle_scene_A"
    data = load_latest_rollout(scene)
    print(f"[{scene}] 로드 완료")
    for k, v in data.items():
        print(f"  {k:15s} shape={v.shape}")