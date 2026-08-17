"""
test_neural_astar.py -- Neural A* 1차 feasibility 테스트.

Scene D 코리도(우리 baselines/occupancy_grid.py 재사용, MuJoCo 불필요 -- A*/skeleton
baseline __main__ 블록과 동일한 synthetic 벽 정의)를 mazes_032_moore_c8 프리트레인
체크포인트가 기대하는 32x32 네이티브 해상도로 직접 만들어서 넣어보고, Neural A* vs
Vanilla A* 출력을 나란히 비교 이미지로 저장한다.

전제:
  - neural-astar repo가 LLM-Robot-Planner 안에 clone/install 되어 있음 (vlm_courtroom,
    dial_mpc와 같은 레벨 -- 나중에 vlm_courtroom 입력으로 이어붙이기 편하도록).
    반드시 .gitignore에 "neural-astar/" 추가할 것 (자체 .git+submodule을 가진 별도
    repo라 안 그러면 우리 repo git 히스토리에 잘못 딸려 들어감).
    설치: git clone --recursive https://github.com/omron-sinicx/neural-astar && pip install .
  - 이 스크립트는 LLM-Robot-Planner repo 루트에서 실행 (baselines/occupancy_grid.py를
    import하기 위해)

실행:
    cd LLM-Robot-Planner
    source neural-astar/.venv/bin/activate   # neural-astar venv
    pip install matplotlib pillow --break-system-packages  # 없으면
    python test_neural_astar.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines"))
from occupancy_grid import GridSpec, build_occupancy_grid, inflate_grid  # noqa: E402

from neural_astar.planner import NeuralAstar, VanillaAstar  # noqa: E402
from neural_astar.utils.training import load_from_ptl_checkpoint  # noqa: E402
from neural_astar.utils.data import visualize_results  # noqa: E402

MAP_SIZE = 32  # mazes_032_moore_c8 체크포인트가 학습된 해상도
CKPT_DIR = "neural-astar/model/mazes_032_moore_c8/lightning_logs/"  # 경로 다르면 수정

# ---- Scene D 스타일 코리도 ----
# 주의: 처음엔 150px/m 네이티브 해상도로 만든 뒤 32x32로 다운샘플했는데, clearance(0.8m)
# 적용 후 코리도 통행 가능 폭이 몇 px밖에 안 남아서 다운샘플 중에 완전히 사라지는 버그가
# 있었음(start/goal이 둘 다 막힌 것으로 나옴). 그래서 처음부터 8m x 8m 정사각 영역을
# 4px/m로 잡아 정확히 32x32 네이티브 해상도로 만듦(다운샘플 없음). 이 1차 feasibility
# 테스트에서는 clearance 팽창도 생략(저해상도에서 또 같은 문제 재발 방지) -- 실제 파이프라인
# 통합 시에는 clearance 처리를 다시 넣어야 함.
spec = GridSpec(x_min=-2.0, x_max=6.0, y_min=-4.0, y_max=4.0, ppm=MAP_SIZE / 8.0)
assert spec.img_w == MAP_SIZE and spec.img_h == MAP_SIZE, (spec.img_w, spec.img_h)
walls = [((-2.0, 0.85), (6.0, 0.85)), ((-2.0, -0.85), (6.0, -0.85))]
grid = build_occupancy_grid(spec, [], walls=walls, wall_thickness_px=2)  # True = 막힘, clearance 미적용
map_design = (1.0 - grid.astype(np.float32))  # Neural A* 규약: 1 = 통행 가능 (공식 docstring 확인)

start_px = spec.world_to_pixel(0.0, 0.0)
goal_px = spec.world_to_pixel(5.0, 0.0)
sx, sy = int(round(start_px[0])), int(round(start_px[1]))
gx, gy = int(round(goal_px[0])), int(round(goal_px[1]))
assert map_design[sy, sx] == 1.0, "start가 막혀있음 -- clearance/wall 파라미터 확인"
assert map_design[gy, gx] == 1.0, "goal이 막혀있음 -- clearance/wall 파라미터 확인"
print(f"[test_neural_astar] start px=({sx},{sy}) goal px=({gx},{gy}) in {MAP_SIZE}x{MAP_SIZE} grid")

start_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
start_map[sy, sx] = 1.0
goal_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
goal_map[gy, gx] = 1.0

map_designs = torch.tensor(map_design).float().unsqueeze(0).unsqueeze(0)
start_maps = torch.tensor(start_map).float().unsqueeze(0).unsqueeze(0)
goal_maps = torch.tensor(goal_map).float().unsqueeze(0).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[test_neural_astar] device={device}")

neural_astar = NeuralAstar(encoder_arch="CNN").to(device)
neural_astar.load_state_dict(load_from_ptl_checkpoint(CKPT_DIR))
neural_astar.eval()

vanilla_astar = VanillaAstar().to(device)
vanilla_astar.eval()

with torch.no_grad():
    na_out = neural_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device))
    va_out = vanilla_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device))

fig, axes = plt.subplots(2, 1, figsize=[8, 4])
axes[0].imshow(visualize_results(map_designs.cpu(), na_out if device == "cpu" else
                                  type(na_out)(*[t.cpu() for t in na_out])))
axes[0].set_title("Neural A* (pretrained on mazes, NOT our corridor data)")
axes[0].axis("off")
axes[1].imshow(visualize_results(map_designs.cpu(), va_out if device == "cpu" else
                                  type(va_out)(*[t.cpu() for t in va_out])))
axes[1].set_title("Vanilla A* (ground-truth-optimal reference)")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("neural_astar_scene_d_test.png", dpi=150)
print("saved -> neural_astar_scene_d_test.png")
