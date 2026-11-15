# TIIS 표 ①·② 실행 가이드 (`tiis/tables` 브랜치)

기준 브랜치: `feature-oracle_gen` (Courtroom TIIS 버전). 이 브랜치는 그 위에 실험 도구만 더했다.

## 0. 서버 준비 (한 번)
```bash
conda activate vlm_court
pip install pandas scipy pillow shapely     # 집계·씬 변환·LLM-A* 충돌 검사용
pip install --no-deps llm-astar==0.1.4      # LLM-A* 공식 패키지 (MIT). --no-deps 필수:
                                            # 그냥 설치하면 openai==0.27.2, torch, transformers가 깔려 환경이 깨질 수 있음
```
Unity 쪽은 `unity/Editor/Go2BevExporter.cs`를 Unity 프로젝트의 `Assets/Editor/`에 넣으면 메뉴 `Tools > Go2 BEV Exporter`가 생긴다.

## 1. Unity에서 씬 내보내기
1. 실내 씬에 빈 GameObject `Go2Start`(바닥 위, 파란 z축 = 로봇 전방), `Go2Goal` 배치.
   여러 쌍은 `Go2Start_01`/`Go2Goal_01` … 처럼 접미사를 맞춘다. 쌍 하나 = 실험 씬 하나.
2. 시작–목표 6~8 m, 사이에 가구. 화면 범위: 전방 약 9.3 m, 후방 4.7 m, 좌우 ±6 m (PPM 90 기준).
3. 천장이 있으면 Camera height를 천장 아래로.
4. Export → `<Unity 프로젝트>/Go2Export/<씬>/` (bev.png, occupancy.png, meta.json) 를 서버 레포 루트의 `Go2Export/`로 복사.

## 2. 한 번에 실행 (끊겨도 같은 명령으로 이어서 진행)
```bash
# 먼저 파일럿: 씬 3개 × 2회
SCENES="indoor_000 indoor_001 indoor_002" TRIALS=2 ./experiments/run_tables.sh

# 본 실행 (밤새)
SCENES="indoor_000 ... indoor_009" TRIALS=5 nohup ./experiments/run_tables.sh &
```
| 단계 | 하는 일 | 스크립트 | 자원 |
|---|---|---|---|
| 0 | Unity 출력 → VLM 입력 이미지·GT 점유 지도·MuJoCo 씬·DIAL 설정 | `indoor/build_scene.py` | CPU, 수 초 |
| 1 | 표 ① 구성 C1, C1xk, C2, C3, C4 | `vlm_courtroom/run_ablation.py` | Gemini API |
| 2 | 표 ② 고전 베이스라인 + LLM-A* | `baselines/run_table2_baselines.py`, `run_llmastar.py` | CPU 수 초 + Gemini |
| 3 | 모든 경로를 DIAL-MPC로 실행 | `experiments/run_dial.py` | GPU, 가장 김 |
| 4 | 지표 계산 → LaTeX 표 | `analysis/evaluate_all.py`, `make_tables.py` | CPU |

환경변수: `GEMINI_MODEL`(기본 gemini-2.5-pro), `IMAGE_MODE`(기본 all: 모든 agent가 이미지), `COORD_ABLATION`(기본 1: C2~C4를 Coordinate만 이미지를 보는 설정으로 한 번 더, 결과 *_coord), `WORKERS`(동시 시행 수), `STAGES`("0 1 2 3 4" 중 일부).

**씬 생성 직후 확인할 것**: `data/<씬>/align_check.png` (GT 점유가 빨갛게 가구·벽 위에 정확히 겹치는지), `data/<씬>/scene_meta.json`의 `warnings`.

## 2-1. 표 ② 구성
| 행 | 지도 | 비고 |
|---|---|---|
| A* / Dijkstra / RRT* | GT 점유 지도 | 지도가 완벽할 때의 상한 |
| A* / RRT* | 이미지 분할 지도 | BEV 바닥색 flood-fill (`baselines/image_occupancy.py`) |
| LLM-A* (Meng et al., EMNLP Findings 2024) | GT / 분할 | 공식 코드 그대로, LLM만 Gemini 2.5 Pro(temperature 0)로 교체 |
| Ours (Courtroom, C4) | 없음 | BEV 이미지 + 시작·목표만 |

LLM-A* 어댑터(`baselines/llmastar_adapter.py`)가 하는 일: 점유 지도를 A*와 같은 반경으로 팽창 → 0.2 m 격자로 축소 →
직사각형 테두리를 원본 형식(`[y, x_start, x_end]`, `[x, y_start, y_end]`)의 선분으로 변환 → 원본 5-shot 프롬프트와 탐색 실행
→ 다른 격자 방법과 같은 후처리(단축 + N개 waypoint). 원본의 `max_tokens=1000`은 Gemini thinking 토큰 때문에 적용하지 않는다.
변환 단계에서 목표가 끊기면 LLM을 부르지 않고 `lattice_disconnected`로 기록한다 (`--lattice-unit`을 줄이면 해결).

## 3. 결과 위치
```
data/<씬>/<방법>/trial_XX/   last_judged_path.json, meta.json, transcript.json, *_states.npy, dial_meta.json
results/trials.csv, groups.csv
results/table1_ablation.tex / .md, results/table2_comparison.tex / .md
```

## 4. 지표 정의 (논문에 그대로 옮길 것)
| 구분 | 지표 | 정의 |
|---|---|---|
| 계획 | Output OK | 2개 이상의 유한한 (x, y) 좌표를 파싱함 |
| 계획 | BEV collision-free | 직선 보간 경로를 따라 Go2 footprint 0.70×0.35 m(진행 방향 정렬)를 쓸었을 때 GT 점유와 겹치지 않음. 회전반경 < 0.5 m 꺾임점은 외접원(0.391 m)으로 검사 |
| 계획 | Valid | Output OK ∧ 마지막 점이 목표 0.3 m 이내 ∧ BEV collision-free |
| 계획 | Homotopy | 유효 경로들의 장애물 ray 교차 word(H-signature)가 다른 class 수, 씬 평균 |
| 실행 | SR | 목표 0.3 m 이내 도달 ∧ 낙상 없음 ∧ 충돌 없음 |
| 실행 | Fall | 몸통 z < 0.18 m 또는 기울기 > 90° |
| 실행 | Collision | rollout을 go2.xml 전신 충돌 모델(23개)로 재생했을 때 장애물에 1 cm 이상 침투한 step이 있음 |
| 실행 | Time | 처음 목표 반경에 들어온 시각 |
| 실행 | Track err. | 각 waypoint에 대한 실제 몸통 궤적 최근접 거리 평균 |
| 실행 | Torque sat. | |ctrl| ≥ 0.99 × 관절 한계(23.7 / 23.7 / 45.43 N·m) 비율 |

집계: 비율은 모든 씬·시행을 합쳐 계산, 연속값은 시행 평균 ± 표준편차, homotopy는 씬 평균 ± 표준편차.
`api_error`(할당량·네트워크) 시행은 모델 실패가 아니므로 제외되고, `run_ablation.py`를 다시 돌리면 그것만 재실행된다.

## 5. 이 브랜치에서 바뀐 것 / 알아둘 것
- **토크 한계 복구**: `mjx_go2_robot_only.xml`에 ctrlrange가 없어서 이전 실험은 토크가 무제한이었다. 이제 23.7 / 23.7 / 45.43 N·m로 clip된다.
- **충돌**: 시뮬레이션용 로봇 모델은 충돌 geom이 발끝 구 4개뿐이라 몸통이 장애물을 통과한다. 충돌은 전신 모델 재생으로만 판정한다 (`analysis/collision_replay.py`).
- **최소 프롬프트**: 실내 씬의 시나리오 문장은 시작·목표·과제만 담는다 (`indoor/build_scene.py`의 `SCENARIO_TEMPLATE`). 장애물·우회 방향 힌트 없음.
- **물리 제약 프롬프트**: `ROBOT_PHYSICAL_CONSTRAINTS`의 "1.6 m 미만 통로는 통과 불가"는 oracle 씬 기준이다. 실내 문(0.8~1.0 m)을 지나야 하는 씬이면 VLM이 경로를 거부할 수 있으니 파일럿에서 확인.
- **장애물 box 수**: Unity 점유 지도를 0.05 m 격자로 그대로 box로 만든다. box가 수백 개면 DIAL-MPC가 느려지니 `build_scene.py --mj-cell 0.1`로 줄인다 (GT도 같은 격자로 바뀌어 일관성 유지).
- **LaTeX**: `dial_core.py`는 scienceplots의 LaTeX 스타일로 그림을 그린 뒤 states.npy를 저장한다. 서버에 LaTeX(cm-super)가 없으면 저장 전에 죽는다.

## 6. 베이스라인 신뢰도 검증 (논문 부록·리뷰 대응)
자세한 명세와 결과는 `experiments/BASELINE_FIDELITY.md`.
```bash
python tests/verify_grid_planners.py                       # A*/Dijkstra 최적성(vs NetworkX), 경로 유효성, 정확한 선분 순회
git clone --depth 1 https://github.com/SilinMeng0510/llm-astar /tmp/llm-astar
python tests/verify_llmastar_equivalence.py --dataset /tmp/llm-astar/dataset/environment_50_30.json   # 원본 = 래퍼
python tests/replicate_llmastar_benchmark.py --dataset /tmp/llm-astar/dataset/environment_50_30.json --n 100  # Gemini로 원 논문 경향 재현
```

## 7. 테스트 (Unity 없이)
```bash
python indoor/make_synthetic_export.py --out Go2Export --n 3
SCENES="synthetic_000 synthetic_001 synthetic_002" TRIALS=2 STAGES="0 2" ./experiments/run_tables.sh
```
