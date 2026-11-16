# 표 ② 베이스라인 구현 명세와 검증

리뷰어가 물을 질문 세 가지에 답하기 위한 문서다: **(1) LLM-A\*를 원본 그대로 썼는가, (2) A\*·Dijkstra는 정확한가, (3) 비교 조건은 공정한가.**
검증 스크립트는 모두 `tests/`에 있고 누구나 다시 돌릴 수 있다.

## 1. LLM-A* (Meng et al., EMNLP Findings 2024)

### 원본과의 대응
| 구성 요소 | 원본 | 우리 | 변경 이유 |
|---|---|---|---|
| 코드 | `llm-astar==0.1.4` (PyPI, MIT) | **같은 패키지를 import해서 사용** (복사·수정 없음) | — |
| 탐색 알고리즘 (`LLMAStar.searching`) | 원본 | **원본 그대로** | — |
| 휴리스틱, target 갱신, 유효 노드 필터 | 원본 | **원본 그대로** | — |
| 프롬프트 | `gpt_prompt['standard']` (5-shot) | **원본 그대로** | — |
| 응답 파싱 (`list_parse`) | 원본 | **원본 그대로** | — |
| LLM | GPT-3.5-turbo-instruct, temperature 0 | Gemini 2.5 Pro, temperature 0 | 표의 모든 LLM/VLM 방법을 같은 모델로 맞추기 위해 |
| 출력 길이 제한 | `max_tokens=1000`, `stop="\n"` | 적용 안 함 | Gemini 2.5의 thinking 토큰이 한도를 소모함. 원본 파서가 응답의 마지막 `[[...]]`를 쓰므로 결과 형식은 같음 |
| 입력 지도 | 장애물 선분 목록을 직접 받음 | 점유 지도 → 선분 목록 변환 (아래) | 우리 씬은 점유 지도로 주어짐 |
| 시작/목표 | 격자점 | 가장 가까운 자유 격자점으로 이동 (같은 연결 성분 안) | 팽창·격자화 후 시작/목표가 장애물 경계에 걸릴 수 있음 |
| 출력 후처리 | 격자 경로 | line-of-sight 단축 → N개 waypoint | 다른 격자 방법과 같은 후처리로 맞춤 |
| 그림 저장 | 매 질의마다 저장 | 끔 | 탐색 후에 실행되는 출력 전용 단계 |

### 지도 변환 (`baselines/llmastar_adapter.py`)
1. GT(또는 분할) 점유 지도를 A*와 **같은 반경 0.441 m**로 팽창 (Go2 footprint 외접원 0.391 m + 0.05 m)
2. 0.2 m 격자로 축소 (셀 안에 장애물이 하나라도 있으면 장애물, 보수적). 원 논문 벤치마크(51×31)와 비슷한 규모(약 41~44 × 25~29)
3. 시작점과 연결된 빈 공간만 남기고 나머지(방 바깥)는 장애물로 채움
4. 장애물 셀을 직사각형으로 병합하고, 각 직사각형 테두리를 원본 형식 `[y, x_start, x_end]`, `[x, y_start, y_end]` 선분으로 변환
5. 변환 단계에서 목표가 시작점과 끊기면 LLM을 부르지 않고 `lattice_disconnected`로 기록 (방법의 실패와 구분)

### 검증: 원본과 래퍼의 동등성 (`tests/verify_llmastar_equivalence.py`)
원 논문 벤치마크 **1,000개 질의**(지도 100개 × 시작/목표 10쌍)에서, 원본 `LLMAStar`(그림 저장까지 무수정)와 래퍼에
같은 LLM 응답(정상 좌표 / 장애물·범위 밖 좌표 / 파싱 불가 문장 / 시작·목표만, 4종을 질의마다 결정적으로 생성)을 넣고 비교.

| 비교 항목 | 다른 질의 수 |
|---|---|
| LLM에 보낸 프롬프트 문자열 | **0 / 1,000** |
| 탐색 결과 (확장 노드 수, 저장 노드 수, 경로 길이, 필터 후 target) | **0 / 1,000** |
| 최종 격자 경로 | **0 / 1,000** |
| 예외 발생 여부·종류 | **0 / 1,000** |

→ **모든 질의에서 완전히 동일.** 래퍼가 바꾼 것은 LLM 호출 객체뿐이며, 같은 응답이 주어지면 원본과 같은 프롬프트를 보내고 같은 탐색을 수행해 같은 경로를 낸다.

### 검증: LLM을 Gemini로 바꿔도 원 논문 경향이 재현되는가 (`tests/replicate_llmastar_benchmark.py`, 서버에서 실행)
원 논문 벤치마크에서 원본 A*와 Gemini 기반 LLM-A*를 같은 질의로 돌려 논문 표 1과 같은 지표를 계산한다.

| | Operation ↓ | Storage ↓ | Path length ↓ | Valid |
|---|---|---|---|---|
| 원 논문 (GPT-3.5, few-shot) | 57.39 | 74.96 | 102.44 | 100 |
| 우리 (Gemini 2.5 Pro, 질의 100개) | (실행 후 기입) | | | |

연산·저장이 A*보다 줄고 경로 길이가 약 102%면, LLM 교체 후에도 방법이 의도대로 작동한다는 근거가 된다.

```bash
git clone --depth 1 https://github.com/SilinMeng0510/llm-astar /tmp/llm-astar
python tests/replicate_llmastar_benchmark.py --dataset /tmp/llm-astar/dataset/environment_50_30.json --n 100
```

## 2. A* / Dijkstra (`baselines/grid_planners.py`)

### 명세
| 항목 | 값 |
|---|---|
| 격자 | GT 점유 지도와 같은 0.05 m 격자 |
| 장애물 팽창 | 0.441 m (Go2 footprint 0.70×0.35 m의 외접원 0.391 m + 여유 0.05 m) → 어떤 방향으로 서 있어도 몸체가 장애물과 겹치지 않음 |
| 이웃 | 8-연결, 비용 1 (상하좌우) / √2 (대각선) |
| 모서리 끼기 | 대각선 이동 시 양옆 두 셀 중 하나라도 장애물이면 금지 |
| A* 휴리스틱 | octile 거리 (8-연결 격자에서 admissible·consistent → 최적 보장) |
| Dijkstra | 같은 코드에서 휴리스틱 = 0 |
| 시작/목표 | 팽창 영역에 걸리면 가장 가까운 빈 셀 |
| 후처리 (기본) | line-of-sight 단축: 앞에서부터 직선으로 갈 수 있는 가장 먼 점으로 건너뜀. 충돌 검사는 선분이 지나는 **모든 셀을 정확히 순회**(Amanatides & Woo, 1987) |
| 후처리 (부록용) | `astarraw_gt`: 단축 없이 격자 경로 그대로 (일직선 중간점만 제거, 기하는 동일) |
| waypoint 수 | 꺾인점은 모두 유지하고 가장 긴 구간을 반으로 나눠 VLM 방법과 같은 N개로 맞춤 (경로 기하 불변) |

### 검증 (`tests/verify_grid_planners.py`, 무작위 지도 500개)
| 항목 | 결과 |
|---|---|
| A* 최단 비용 = networkx Dijkstra (독립 구현, 같은 그래프) | **500/500 일치** (오차 0) |
| Dijkstra 최단 비용 = networkx | **500/500 일치** |
| 도달 가능 여부 판정 일치 | **500/500** |
| 경로 유효성 (인접 이동, 빈 셀, 모서리 끼기 없음) | **1,000/1,000 유효** |
| A* 확장 노드 ≤ Dijkstra 확장 노드 | **500/500** (평균 16.5%만 확장) |
| 선분 셀 순회 vs 셀 1/200 간격 샘플링 | 2,000개 선분에서 **놓친 셀 0개** |
| 단축 후 경로 충돌 | 250/250 충돌 없음 |

실내 씬(합성 3개)에서도 A*와 Dijkstra의 격자 경로 비용이 소수점까지 같고(175.995 / 162.882 / 170.610 셀), 단축 전후 길이는 8.78→8.61, 8.14→7.67, 8.53→8.09 m.

## 3. RRT*
| 항목 | 값 |
|---|---|
| 공간 | 연속 2D, 충돌 검사는 A*와 같은 팽창 지도 + 정확한 선분 셀 순회 |
| 반복 / 확장 길이 / 목표 편향 | 4,000회 / 0.5 m / 0.1 |
| rewire 반경 | 1.0 m 고정 (원 논문의 γ(log n / n)^{1/d} 축소 반경 대신 고정값. 명시할 것) |
| 목표 허용 오차 | 0.3 m (실행 지표의 목표 반경과 같음) |
| seed | 시행 번호 → 시행마다 다른 경로 |
| 후처리 | A*와 같은 단축 |

정상성 확인: 실내 씬 3개에서 RRT* 경로 길이는 A*(단축 후) 대비 −0.6% ~ +2.4%.

## 4. 이미지 분할 지도 (`baselines/image_occupancy.py`)
시작 픽셀 주변 바닥색 기준 flood-fill (RGB 거리 28 이내, 시작점과 연결된 영역 = 바닥, 나머지 = 장애물).
학습이 필요 없는 가장 단순한 분할이라 재현이 쉽다. 러그·무늬 바닥·그림자에서 틀리는 것이 의도된 실패 모드이며,
"고전 방법은 별도의 분할 단계가 필요하고 그 품질에 의존한다"를 보여주는 역할이다.
분할 품질은 로봇이 갈 수 있는 영역 안에서의 GT 일치율로 함께 보고한다 (`seg_cell_agreement`).

## 5. 모든 방법에 공통인 조건
- 같은 씬, 같은 시작/목표, 같은 waypoint 수 N
- 같은 실행기 (DIAL-MPC, 같은 설정), 같은 지표 코드, 같은 목표 반경 0.3 m
- 지도를 쓰는 방법은 같은 팽창 반경 0.441 m
- LLM/VLM을 쓰는 방법은 같은 모델 (Gemini 2.5 Pro)

## 6. 논문용 문장 (Implementation details)
> **Baselines.** A\* and Dijkstra run on an 8-connected occupancy grid (0.05 m) with unit and √2 step costs, corner cutting disallowed, and obstacles inflated by 0.441 m (the circumscribed radius of the Go2 footprint plus a 0.05 m margin); A\* uses the octile heuristic. Their optimal costs matched an independent Dijkstra implementation (NetworkX) on all 500 randomly generated test maps. Grid paths are post-processed by line-of-sight shortcutting with exact cell traversal [Amanatides & Woo]. RRT\* uses 4,000 iterations, a 0.5 m extension step, a fixed 1.0 m rewiring radius, and the same collision checker. For LLM-A\* [Meng et al.], we use the authors' released implementation (llm-astar 0.1.4) unchanged, including its search procedure and five-shot prompt, and only replace the language model with Gemini 2.5 Pro at temperature 0 for parity with our method. Our occupancy maps are converted to the barrier-segment format of LLM-A\* on a 0.2 m lattice after the same inflation. On all 1,000 queries of the original LLM-A\* benchmark, our wrapper produced prompts, search statistics, and paths identical to the unmodified implementation given the same model responses.

## 평가 지표 검증 (`tests/verify_metrics.py`)
| 지표 | 대조 대상 | 결과 |
|---|---|---|
| BEV 충돌 (Go2 0.70×0.35 m 직사각형 sweep + 급회전 원) | shapely 정확 기하 (구간별 convex hull ∪ 원) vs 점유 셀 | 1,500 사례 (정확한 충돌 731) 중 1,499 일치. 오경보 0. 놓친 충돌 1건, 침투 1.9 cm (샘플 간격 2.5 cm 미만) |
| 목표 도달 (0.3 m) | 직접 계산 | 0.29 m 도달, 0.31 m 미도달 |
| 기울기 | scipy Rotation (몸통 z축–세계 z축 각) | 2,000 자세, 최대 오차 1e-12° |
| 경로 이탈 (점–폴리라인 거리) | shapely distance | 300 사례, 최대 오차 3e-15 m |
| homotopy (H-signature) | 손 사례: 장애물 위/아래, 같은 쪽 두 경로, 한 바퀴 감기 | 같은 쪽 = 같음, 반대쪽 = 다름, 감기 = 별도 |
| 토크 포화 (≥ 0.99 × 한계) | 손으로 만든 ctrl | 기대값과 일치, 한계 순서 abduction/thigh/calf |
| 전신 충돌 재생 | 출발 자세 / 장애물 안에 놓은 자세 | 충돌 없음 / 충돌 |

검증 중 고친 것: 꺾이는 지점에서 직전 구간 자세의 몸체가 샘플링에서 빠져 최대 2.7 cm 침투를 놓치던 문제 → 구간 끝점을 그 구간의 heading으로도 샘플링.
남은 근사: 점 샘플링 간격(2.5 cm)보다 얕은 겹침은 놓칠 수 있다. 실행 충돌 판정의 허용 침투(1 cm)와 같은 크기다.
