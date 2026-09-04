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

전체 파이프라인은 `run_random_batch_v2.py` 하나로 실행됩니다.

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

git clone https://github.com/LeCAR-Lab/dial-mpc dial_mpc
pip install -e dial_mpc/

# --- neural-astar env: Neural A* 추론/fine-tuning ---
conda create -n neural-astar python=3.10
conda activate neural-astar
pip install -r requirements-neural-astar.txt

git clone https://github.com/omron-sinicx/neural-astar
pip install -e neural-astar/
```

두 외부 저장소(`dial_mpc/`, `neural-astar/`) 모두 repo 루트에 위치해야 하며, `.gitignore`에 의해 버전관리 대상에서 제외됩니다. `run_neural_astar_step.py`의 `CKPT_PATH`가 `neural-astar/model/mazes_032_moore_c8/lightning_logs/version_0`을 기본 사전학습 체크포인트로 참조하므로, 클론된 `neural-astar/` 안에 해당 체크포인트가 포함되어 있는지 확인하세요.

`requirements-*.txt`는 핵심 패키지만 정리한 목록입니다. `dial_mpc`/`neural-astar`를 `pip install -e`로 설치하면 각 프로젝트의 `setup.py`/`pyproject.toml`에 정의된 나머지 의존성이 자동으로 딸려 설치됩니다.

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
python3 run_random_batch_v2.py --n-scenes 1 --start-seed 0 --run-dial
```

DIAL-MPC 없이 경로 생성까지만 (빠른 확인용):

```bash
python3 run_random_batch_v2.py --n-scenes 10 --start-seed 0
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

### 운영 주의사항

- GPU가 다른 프로세스와 공유되는 환경이면 DIAL-MPC(JAX)가 `CUDA_ERROR_OUT_OF_MEMORY`로 죽을 수 있습니다. 이 경우 `elapsed_s`가 비정상적으로 짧게(수십 초) 찍히는 게 특징입니다 — 실제 물리 실패가 아니라 리소스 경합이니, `XLA_PYTHON_CLIENT_PREALLOCATE=false` 환경변수로 재시도하세요.
- 야간/장시간 배치는 반드시 `tmux` 세션 안에서 실행하세요 (`tmux new -d -s <이름> "<명령어>"`로 바로 백그라운드 시작 가능). 일반 SSH 터미널 종료 시 프로세스가 죽습니다.

---

## Fine-tuning (`finetune/`)

사전학습된 Neural A*(`omron-sinicx/neural-astar`의 `mazes_032_moore_c8` 체크포인트)는 표준 미로 벤치마크로 학습되어 있어, 우리 씬(카메라 FOV 제약, 특정 장애물 분포)과는 도메인이 다릅니다. `finetune/`은 위 파이프라인으로 만든 **DIAL-MPC 검증까지 통과한 경로**를 GT로 삼아 fine-tuning하는 코드입니다.

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

### 방법론

- **입력**: `oracle.png`의 벽 마스크를 32x32로 다운샘플한 occupancy map + start/goal one-hot map (추론 파이프라인과 완전히 동일한 좌표 변환 함수 재사용, 좌표계 불일치 없음)
- **정답(GT)**: `waypoint_generator.py`가 보정하고 DIAL-MPC로 낙상 없이 검증된 최종 경로를 32x32 grid에 선분으로 래스터화
- **Loss**: `L1Loss(model.histories, GT_opt_traj)` — 미분가능 A* 탐색의 방문 히트맵을 GT 경로에 맞춤
- **⚠️ 데이터 규모 주의**: 80개 샘플로 실험한 결과, 30 epoch은 개선되지만 150 epoch은 **과적합**(train loss는 계속 감소하지만 val loss는 오히려 증가, 새 씬 일반화 실패)이 확인되었습니다. Epoch을 늘리기보다 씬 개수를 늘리는 쪽이 효과적입니다. Fine-tuning 전에는 반드시 `pilot10_compare.py`로 **학습에 안 쓰인 새 씬**에서 개선을 확인하세요 — val set만으로는 과적합을 못 잡습니다.

---

## Project Structure

```
LLM-Robot-Planner/
├── generate_random_baffle_maze.py   # 씬 생성 (slalom / boxes 레이아웃)
├── oracle_gen.py                    # MuJoCo 렌더링
├── run_neural_astar_step.py         # Neural A* 초기 경로 제안
├── waypoint_generator.py            # EDT 기반 결정론적 안전 마진 보정
├── run_random_batch_v2.py           # 전체 파이프라인 오케스트레이터 (메인 엔트리)
├── check_fall.py                    # DIAL-MPC 결과 낙상 판정 유틸
├── requirements-vlm_court.txt       # vlm_court env 핵심 패키지
├── requirements-neural-astar.txt    # neural-astar env 핵심 패키지
├── finetune/                        # Neural A* fine-tuning
├── dial_mpc/                        # (외부 의존성, 별도 clone 필요)
└── neural-astar/                    # (외부 의존성, 별도 clone 필요)
```

---

## References

- Neural A*: [omron-sinicx/neural-astar](https://github.com/omron-sinicx/neural-astar)
- DIAL-MPC: [LeCAR-Lab/dial-mpc](https://github.com/LeCAR-Lab/dial-mpc) ([논문](https://arxiv.org/abs/2409.15610))
- Unitree Go2 (MuJoCo MJX 물리 시뮬레이션)