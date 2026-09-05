# LLM-Robot-Planner: Neural A* + Deterministic Safety Correction + DIAL-MPC

씬 이미지에서 로봇(Unitree Go2)이 안전하게 걸을 수 있는 경로를 생성하고, 실제 물리 시뮬레이션(DIAL-MPC)으로 검증하는 파이프라인입니다. VLM 기반 경로 판단 대신, **Neural A\*로 초기 경로를 제안하고 distance-field 기반 결정론적 알고리즘으로 안전 마진을 보정**하는 방식으로 전환했습니다 (VLM 환각 문제 회피, 재현 가능성 확보).

---

## Pipeline Overview

```
씬 생성 (generate_random_baffle_maze.py)
    │  랜덤 seed → slalom(복도형) 또는 boxes(자유배치형) 레이아웃 XML 생성
    ▼
씬 렌더링 (oracle_gen.py)
    │  MuJoCo로 top-down 이미지 렌더 → data/<scene>/oracle.png
    ▼
Neural A* 초기 경로 제안 (run_neural_astar_step.py, conda env: neural-astar)
    │  oracle.png → 32x32 occupancy grid → 사전학습 Neural A* 순전파
    │  → line-of-sight 단순화 + 보폭 제약 → coordinate_proposal.json
    ▼
안전 마진 보정 (waypoint_generator.py)
    │  EDT(distance-field) 기반 결정론적 보정: 벽과의 clearance가
    │  HARD_RADIUS_M(0.4m) 미만이면 강제 이동, SOFT_RADIUS_M(0.6m)까지
    │  코너 우회 탐색. VLM 판단 없이 100% 재현 가능.
    │  → data/<scene>/waypoint_gen_v1/last_judged_path.json + verdict.png
    ▼
DIAL-MPC 물리 시뮬레이션 검증 (dial_mpc/, conda env: vlm_court)
    │  CMA-ES 기반 MPC로 실제 로봇이 해당 경로를 걸을 수 있는지 검증
    │  → data/<scene>/*_states.npy
    ▼
낙상 판정 (check_fall.py)
       states.npy의 z-height 궤적으로 낙상 여부 최종 판정
```

전체 파이프라인은 `core/run_random_batch_v2.py` 하나로 실행됩니다.

---

## Setup

### Conda 환경 (2개로 분리되어 있음)

- **`vlm_court`**: 씬 생성/렌더링/waypoint 보정/DIAL-MPC 실행용 (MuJoCo, JAX, matplotlib)
- **`neural-astar`**: Neural A* 추론/fine-tuning 전용 (PyTorch, PyTorch Lightning)

두 개로 나뉜 이유는 PyTorch/CUDA 버전 호환성 문제 때문입니다. `neural-astar` env의 PyTorch 빌드는 최신 GPU(sm_120 이상)를 지원하지 않아 **CPU로 폴백**합니다 — Neural A* 모델 자체가 작아서(391K 파라미터) CPU로도 실용적인 속도가 나옵니다.

### 환경별 설치

```bash
# --- vlm_court env: 씬 생성/렌더링/waypoint 보정/DIAL-MPC ---
conda create -n vlm_court python=3.10
conda activate vlm_court
pip install -r requirements-vlm_court.txt
pip install -e dial_mpc/   # repo에 이미 포함되어 있음 (아래 참고)

# --- neural-astar env: Neural A* 추론/fine-tuning ---
conda create -n neural-astar python=3.10
conda activate neural-astar
pip install -r requirements-neural-astar.txt

git clone https://github.com/omron-sinicx/neural-astar
pip install -e neural-astar/
```

`dial_mpc/`는 [LeCAR-Lab/dial-mpc](https://github.com/LeCAR-Lab/dial-mpc)를 이 프로젝트 요구에 맞게 직접 수정(패치)한 버전이라 **repo에 그대로 포함**되어 있습니다 (별도 clone 불필요). `neural-astar/`는 수정 없이 그대로 쓰는 순수 외부 의존성이라 `.gitignore` 처리되어 있으며, 별도로 clone해야 합니다. `run_neural_astar_step.py`의 `CKPT_PATH`가 `neural-astar/model/mazes_032_moore_c8/lightning_logs/version_0`을 기본 사전학습 체크포인트로 참조하므로, 클론된 `neural-astar/` 안에 해당 체크포인트가 포함되어 있는지 확인하세요.

`requirements-*.txt`는 핵심 패키지만 정리한 목록입니다. `neural-astar`를 `pip install -e`로 설치하면 `setup.py`/`pyproject.toml`에 정의된 나머지 의존성이 자동으로 딸려 설치됩니다.

⚠️ **GPU 사용 시 추가 설치 필요**: `requirements-vlm_court.txt`의 `jax==0.6.2`를 순정 `pip install`로 깔면 CPU 전용 빌드가 설치될 수 있습니다. DIAL-MPC는 GPU 가속을 전제로 하므로, 설치 후 GPU 인식이 안 되면 CUDA 버전에 맞는 JAX를 별도로 다시 설치하세요 (예: `pip install -U "jax[cuda12]"` — 정확한 extras 이름은 사용 중인 CUDA 버전에 맞춰 [JAX 공식 설치 가이드](https://docs.jax.dev/en/latest/installation.html) 확인). 또한 MuJoCo의 headless(EGL) 렌더링에는 `libegl1`/`libgl1` 같은 시스템 패키지가 필요할 수 있습니다 (Ubuntu: `sudo apt install libegl1 libgl1`).

⚠️ **Neural A* 사전학습 체크포인트**: `git clone https://github.com/omron-sinicx/neural-astar`만으로 `model/mazes_032_moore_c8/lightning_logs/version_0/checkpoints/*.ckpt`가 따라오지 않을 수 있습니다. 클론 후 아래 명령으로 실제 존재하는지 먼저 확인하세요:
```bash
find neural-astar/model/mazes_032_moore_c8 -name "*.ckpt"
```
아무것도 안 나오면 [neural-astar 저장소](https://github.com/omron-sinicx/neural-astar)의 README/Release를 참고해 체크포인트를 별도로 받아야 합니다.

### 카메라/좌표계 상수

`oracle_gen.py`, `run_neural_astar_step.py`, `waypoint_generator.py`가 공유하는 고정 상수입니다 (임의로 바꾸면 좌표계가 깨집니다):

```
ROBOT_PX = (421.0, 540.0)   # 로봇(카메라 원점) 픽셀 좌표
PPM = 150.0                  # pixel per meter
GRID = 32                    # Neural A* 입력 grid 해상도
CAMERA_VISIBLE_X_MAX_M ≈ 5.3 # 카메라가 실제로 담는 최대 x 범위 (비대칭 FOV)
```

---

## Quick Start

씬 1개를 처음부터 끝까지 (DIAL-MPC 포함) 돌리기:

```bash
conda activate vlm_court
python3 core/run_random_batch_v2.py --n-scenes 1 --start-seed 0 --run-dial
```

DIAL-MPC 없이 경로 생성까지만 (빠른 확인용):

```bash
python3 core/run_random_batch_v2.py --n-scenes 10 --start-seed 0
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--n-scenes` | 8 | 생성할 씬 개수 |
| `--start-seed` | 0 | 시작 seed (다른 배치와 안 겹치게 조정) |
| `--n-steps` | 800 | DIAL-MPC 시뮬레이션 스텝 수 |
| `--box-prob` | 0.5 | boxes 레이아웃 선택 확률 (나머지는 slalom) |
| `--run-dial` | off | DIAL-MPC까지 실행할지 여부 |
| `--prefix` | `B` | 씬 이름 접두사 (`oracle_scene_{prefix}{seed:04d}`) — 배치별로 분리할 때 사용 |
| `--manifest` | `data/random_batch_manifest_v2.csv` | 결과 기록 CSV 경로 |
| `--purpose` | `unlabeled` | 이 배치의 용도 태그. `data/runs/{날짜}_{purpose}_{prefix}/`에 `run_meta.json`(전체 설정값)+`scenes.txt`(포함된 씬 목록)가 자동 기록됨 |

씬 생성 자체도 매번 랜덤화됩니다 — 레이아웃(slalom/boxes)뿐 아니라 방 크기(`room_half_y`), 목표까지 거리(`min_dist_from_start`), 목표 x 범위(`goal_x_range`)까지 씬마다 다른 값으로 샘플링되어 Neural A* fine-tuning용 데이터 다양성을 확보합니다.

### 출력 구조 (씬 1개당)

```
data/<scene>/
├── oracle.png                          # 렌더링된 씬 이미지
├── neural_astar/
│   ├── coordinate_proposal.json        # Neural A* 초기 제안 경로
│   └── overlay_solo.png
├── waypoint_gen_v1/
│   ├── last_judged_path.json           # 최종 안전 보정 경로
│   ├── log.txt                         # 보정 과정 로그 (PASSED 여부 포함)
│   └── verdict.png                     # 경로 시각화 (반드시 확인할 것)
├── last_judged_path.json               # DIAL-MPC 입력용 복사본
└── *_states.npy                        # DIAL-MPC 시뮬레이션 결과 (--run-dial 시)
```

매니페스트 CSV(`data/random_batch_manifest_v*.csv`)에 씬별 `generator_passed`(안전 보정 성공 여부), `dial_mpc_ok`(실제 낙상 없이 완주 여부)가 기록됩니다. **`dial_mpc_ok=True`인 경로만 실제로 검증된 안전 경로입니다.**

### 개별 스테이지 실행 (테스트 씬 1개로 특정 단계만 확인)

전체 파이프라인을 다 안 돌리고 특정 단계 결과만 보고 싶을 때 사용합니다. 예: Neural A*가 뽑은 초기 경로만 빠르게 확인.

```bash
conda activate vlm_court

# 1. 테스트 씬 1개 생성
python3 core/generate_random_baffle_maze.py --n-scenes 1 --start-seed 500
# → dial_mpc/dial_mpc/models/unitree_go2/oracle_scene_R000.xml 생성됨
# ⚠️ 파일명은 --start-seed가 아니라 루프 인덱스(i) 기준으로 R000, R001...로 붙습니다.
#    즉 --n-scenes 1로 여러 번 실행하면 매번 oracle_scene_R000.xml을 덮어씁니다.
#    여러 개를 따로 보존하려면 --out-dir을 매번 다르게 지정하거나 생성 직후 파일명을 바꿔두세요.

# 2. 렌더링 → data/oracle_scene_R000/oracle.png 생성
# (디스플레이 없는 서버/SSH 환경이면 MUJOCO_GL=egl 지정 필요 -- run_random_batch_v2.py는
#  내부적으로 이미 이 환경변수를 자동 설정하지만, 이렇게 단독 실행할 땐 직접 줘야 함)
MUJOCO_GL=egl python3 core/oracle_gen.py oracle_scene_R000.xml

# 3. Neural A* 초기 경로만 확인 (--goal-x/--goal-y는 로봇 기준 world 좌표, 단위 m)
conda activate neural-astar
python3 core/run_neural_astar_step.py --scene oracle_scene_R000 --goal-x 3.0 --goal-y 1.0
# → data/oracle_scene_R000/neural_astar/{overlay_solo.png, coordinate_proposal.json, path_info.json}
```

여기서 멈추면 Neural A*의 raw 제안 경로(안전 마진 보정 전)만 본 것입니다. `core/waypoint_generator.py`는 별도 CLI가 없고 `core/run_random_batch_v2.py`에서 함수로 import해서 쓰는 모듈이라, 안전 마진 보정까지 단독으로 보려면 아래처럼 직접 호출합니다:

```python
# check_waypoint.py 같은 이름으로 레포 루트에 저장 후 conda activate vlm_court 상태에서 실행
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from waypoint_generator import generate_waypoints
from run_neural_astar_step import ROBOT_PX, PPM

scene = "oracle_scene_R000"
image_path = f"data/{scene}/oracle.png"
with open(f"data/{scene}/neural_astar/coordinate_proposal.json") as f:
    coordinate_proposal = json.load(f)

final_coords, passed, gen_log = generate_waypoints(image_path, coordinate_proposal, ROBOT_PX, PPM)
print("PASSED:", passed)
for line in gen_log:
    print(line)
```

DIAL-MPC 물리 검증까지 단독으로 돌리는 건 `dial_core.py` 호출이 훨씬 복잡해서 별도 단순 CLI가 없습니다 — 이 단계까지 필요하면 그냥 `core/run_random_batch_v2.py --n-scenes 1 --start-seed <seed> --run-dial`로 전체를 도는 게 제일 간단합니다.

### 운영 주의사항

- GPU가 다른 프로세스와 공유되는 환경에서 DIAL-MPC(JAX)가 `CUDA_ERROR_OUT_OF_MEMORY`로 죽는 문제는 `core/run_random_batch_v2.py`의 `run_dial_mpc()`에 `XLA_PYTHON_CLIENT_PREALLOCATE=false`/`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` 환경변수가 기본 반영되어 있어 자동으로 완화됩니다. 그래도 실패한 씬이 있다면 `elapsed_s`가 비정상적으로 짧게(수십 초 미만) 찍히는지로 리소스 경합 여부를 판별할 수 있습니다.
- 야간/장시간 배치는 반드시 `tmux` 세션 안에서 실행하세요 (`tmux new -d -s <이름> "<명령어>"`로 바로 백그라운드 시작 가능). 일반 SSH 터미널 종료 시 프로세스가 죽습니다. `conda activate` 대신 `conda run -n <env> --no-capture-output`을 쓰면 tmux의 비대화형 셸에서도 안전하게 동작합니다.
- 배치 실행 시 `--purpose` 태그를 지정하면 `data/runs/{날짜}_{purpose}_{prefix}/`에 그 배치의 설정값(`run_meta.json`)과 포함된 씬 목록(`scenes.txt`)이 자동 기록되어, 나중에 "이 배치가 뭐였는지" 추적하기 쉽습니다.

---

## Fine-tuning (`finetune/`)

사전학습된 Neural A*(`omron-sinicx/neural-astar`의 `mazes_032_moore_c8` 체크포인트)는 표준 미로 벤치마크로 학습되어 있어, 우리 씬(카메라 FOV 제약, 특정 장애물 분포)과는 도메인이 다릅니다. `finetune/`은 위 파이프라인으로 만든 **DIAL-MPC 검증까지 통과한 경로**를 GT로 삼아 fine-tuning하는 코드입니다.

⚠️ **선행 조건**: 이 섹션은 `data/random_batch_manifest_v2.csv`에 `generator_passed=True`이면서 `dial_mpc_ok=True`인 씬이 이미 여러 개 쌓여 있다는 걸 전제로 합니다. 새로 클론한 저장소라면 이 매니페스트가 비어있으므로, 먼저 위 **Quick Start**로 `--run-dial`을 켜서 최소 수십 개 이상의 씬을 돌려 매니페스트를 채운 뒤에 이 섹션을 진행하세요. 그렇지 않으면 `build_dataset.py`가 "포함된 샘플이 0개"로 끝납니다.

```
finetune/
├── build_dataset.py    # manifest에서 generator_passed & dial_mpc_ok 씬만 골라
│                        # (map, start, goal, opt_traj) 32x32 텐서로 변환, npz 캐시 생성
├── train.py             # 사전학습 체크포인트 로드 → RMSprop으로 낮은 lr fine-tune
├── eval_compare.py       # fine-tune 전/후 val set loss·p_opt·p_exp 비교
├── verify_compare.py     # held-out val 씬에서 GT/사전학습/fine-tuned 경로 시각 비교
└── pilot10_compare.py    # 학습에 안 쓰인 새 씬 생성 + 3-way 경로 비교 (일반화 검증)
```

### 실행 순서

```bash
conda activate vlm_court
python3 finetune/build_dataset.py          # cache/dataset.npz 생성

conda activate neural-astar
python3 finetune/train.py --epochs 30 --lr 1e-4
python3 finetune/eval_compare.py           # 정량 비교
python3 finetune/pilot10_compare.py        # 새 씬으로 정성 비교 (필수)
```

⚠️ `build_dataset.py`/`eval_compare.py`/`pilot10_compare.py`는 CLI 인자가 없습니다 (`train.py`만 `--epochs`/`--lr`/`--batch-size`/`--val-ratio` 지원). `build_dataset.py`는 15번 줄에 `MANIFEST = "data/random_batch_manifest_v2.csv"`로 **경로가 하드코딩**되어 있어, 다른 배치(예: v3)를 학습에 포함하려면 이 줄을 직접 고치거나 manifest CSV들을 먼저 병합해야 합니다.

### 방법론

- **입력**: `oracle.png`의 벽 마스크를 32x32로 다운샘플한 occupancy map + start/goal one-hot map (추론 파이프라인과 완전히 동일한 좌표 변환 함수 재사용, 좌표계 불일치 없음)
- **정답(GT)**: `waypoint_generator.py`가 보정하고 DIAL-MPC로 낙상 없이 검증된 최종 경로를 32x32 grid에 선분으로 래스터화
- **Loss**: `L1Loss(model.histories, GT_opt_traj)` — 미분가능 A* 탐색의 방문 히트맵을 GT 경로에 맞춤
- **⚠️ 데이터 규모 주의**: 80개 샘플로 실험한 결과, 30 epoch은 개선되지만 150 epoch은 **과적합**(train loss는 계속 감소하지만 val loss는 오히려 증가, 새 씬 일반화 실패)이 확인되었습니다. Epoch을 늘리기보다 씬 개수를 늘리는 쪽이 효과적입니다. Fine-tuning 전에는 반드시 `pilot10_compare.py`로 **학습에 안 쓰인 새 씬**에서 개선을 확인하세요 — val set만으로는 과적합을 못 잡습니다.

---

## Project Structure

```
LLM-Robot-Planner/
├── core/                             # 핵심 파이프라인 스크립트 (자주 쓰는 것들만 모음)
│   ├── generate_random_baffle_maze.py  # 씬 생성 (slalom / boxes 레이아웃, 다양성 랜덤화)
│   ├── oracle_gen.py                   # MuJoCo 렌더링
│   ├── run_neural_astar_step.py        # Neural A* 초기 경로 제안
│   ├── waypoint_generator.py           # EDT 기반 결정론적 안전 마진 보정
│   ├── run_random_batch_v2.py          # 전체 파이프라인 오케스트레이터 (메인 엔트리)
│   └── check_fall.py                   # DIAL-MPC 결과 낙상 판정 유틸 (수동 실행용)
├── finetune/                         # Neural A* fine-tuning
│   ├── build_dataset.py
│   ├── train.py
│   ├── eval_compare.py
│   ├── verify_compare.py
│   ├── pilot10_compare.py
│   └── visualize_gradient.py           # cost map/histories fine-tune 전후 비교 시각화
├── requirements-vlm_court.txt        # vlm_court env 핵심 패키지
├── requirements-neural-astar.txt     # neural-astar env 핵심 패키지
├── dial_mpc/                         # DIAL-MPC (LeCAR-Lab/dial-mpc를 이 프로젝트용으로 패치, repo에 포함)
└── neural-astar/                     # (순수 외부 의존성, .gitignore 처리, 별도 clone 필요)
```

`core/`의 스크립트들은 서로 bare import(`from waypoint_generator import ...`)로 연결되어 있어 같은 폴더 안에서만 동작합니다. `finetune/*.py`에서 이 모듈들을 가져다 쓸 때는 파일 상단에 `sys.path.insert(0, ..., "core")`가 이미 추가되어 있어 별도 조치 없이 `import run_neural_astar_step as nas`처럼 그대로 씁니다.

---

## References

- Neural A*: [omron-sinicx/neural-astar](https://github.com/omron-sinicx/neural-astar)
- DIAL-MPC: [LeCAR-Lab/dial-mpc](https://github.com/LeCAR-Lab/dial-mpc) ([논문](https://arxiv.org/abs/2409.15610))
- Unitree Go2 (MuJoCo MJX 물리 시뮬레이션)