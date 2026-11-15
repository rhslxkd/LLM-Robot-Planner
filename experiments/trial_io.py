"""trial_io.py ─ 시행 폴더 공통 처리."""
import glob
import os

EXEC_ARTIFACTS = ("*_states.npy", "dial_meta.json", "dial.log", "*_brax_visualization.html", "*.png")


def clear_execution(trial_dir: str) -> int:
    """경로를 새로 쓸 때, 예전 경로로 만든 DIAL-MPC 결과를 지운다 (옛 rollout이 새 경로와 짝지어 채점되는 것 방지)."""
    n = 0
    for pat in EXEC_ARTIFACTS:
        for f in glob.glob(os.path.join(trial_dir, pat)):
            os.remove(f)
            n += 1
    return n
