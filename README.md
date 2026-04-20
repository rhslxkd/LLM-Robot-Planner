# LLM-to-Robot Framework

> **뇌(VLM Courtroom) + 몸(DIAL-MPC)** — 학습 데이터 없이 언어 모델이 계획하고, 로봇이 실행하는 자율 항법 프레임워크

---

## Overview

HELM-Brain은 두 가지 핵심 모듈을 결합한 자율 로봇 항법 시스템입니다.

- **Brain (VLM Courtroom)**: Vision-Language Model(VLM)이 카메라 이미지를 분석하고, 4개의 AI 에이전트가 법정 토론(Courtroom Debate) 방식으로 최적 경로를 합의 도출
- **Body (DIAL-MPC)**: CMA-ES(Covariance Matrix Adaptation Evolution Strategy) 기반 Model Predictive Control로, **별도의 학습 데이터나 사전 훈련 없이** 로봇이 결정된 경로를 실행

기존 로봇 항법 연구는 대규모 학습 데이터와 환경별 재훈련이 필요했습니다. 본 프로젝트는 이 두 가지 제약을 동시에 해결하며, 향후 **성격(Personality) 파라미터**를 통해 감정 기반 로봇 행동까지 확장 가능한 아키텍처를 설계했습니다.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        INPUT                            │
│              Camera Image + Task Description            │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  🧠  BRAIN                               │
│               VLM Courtroom                             │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  [Step 1] CoordinateAgent                       │    │
│  │   - VLM이 이미지 분석 → 장애물 인식             │    │
│  │   - 로봇 물리 제약 준수 10개 경유점 초안 생성   │    │
│  │   - ChromaDB에 경로 데이터 저장                 │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │  [Step 2] ProsecutorAgent                       │    │
│  │   - 제안 경로의 위험 요소 / 충돌 위험 반박      │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │  [Step 3] DefenseAttorneyAgent                  │    │
│  │   - 효율성 / 실현 가능성 관점에서 경로 변호     │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │  [Step 4] JudgeAgent  (Gemini 1.5 Pro)          │    │
│  │   - 양측 논거 종합 → 최종 경로 판결             │    │
│  │   - JSON 형식으로 최종 waypoint 출력            │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                               │
└─────────────────────────┼───────────────────────────────┘
                          │  Waypoints (x, y) JSON
┌─────────────────────────▼───────────────────────────────┐
│                  🤖  BODY                                │
│                  DIAL-MPC                               │
│                                                         │
│   CMA-ES 기반 Model Predictive Control                  │
│   - 학습 데이터 불필요 (Training-Free)                  │
│   - MuJoCo/MJX 물리 시뮬레이션 환경                    │
│   - Unitree Go2 / H1 로봇 지원                         │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                       OUTPUT                            │
│            Robot Physical Execution + Visualization     │
└─────────────────────────────────────────────────────────┘
```

---

## Why Courtroom Debate?

단일 LLM에게 경로를 요청하면 hallucination(환각)이나 물리적으로 불가능한 경로가 출력될 수 있습니다. 본 프로젝트는 이를 **적대적 다중 에이전트 토론(Adversarial Multi-Agent Debate)** 구조로 해결합니다.

| 역할 | 모델 | 기능 |
|------|------|------|
| CoordinateAgent | Gemini 1.5 Flash | 초기 경로 제안 |
| ProsecutorAgent | Gemini 1.5 Flash | 경로 위험성 검증 |
| DefenseAttorneyAgent | Gemini 1.5 Flash | 경로 효율성 변호 |
| **JudgeAgent** | **Gemini 1.5 Pro** | **최종 경로 확정** |

검사와 변호인의 논거를 모두 청취한 판사(Judge)가 최종 경로를 결정하므로, 단일 에이전트 대비 안전성과 신뢰성이 향상됩니다.

---

## Why DIAL-MPC?

기존 로봇 제어 방식의 한계:

| 방식 | 한계 |
|------|------|
| 강화학습(RL) | 환경별 수백만 스텝 재학습 필요 |
| 모방 학습 | 대규모 시연 데이터셋 필요 |
| 전통 MPC | 정밀한 수식 모델 설계 필요 |

**DIAL-MPC** ([논문 링크](https://arxiv.org/abs/2409.15610))는 CMA-ES(진화 전략) 기반으로 동작하여:
- 사전 학습 데이터 없이 실시간 최적화
- MuJoCo 물리 엔진으로 다중 궤적 병렬 시뮬레이션
- Unitree Go2 (4족 보행), H1 (휴머노이드) 즉시 지원

VLM이 계획한 waypoint를 받아 물리 법칙 내에서 실현 가능한 모션으로 변환합니다.

---

## Demo

### VLM Courtroom — Path Planning Results

<img width="667" height="531" alt="image" src="https://github.com/user-attachments/assets/dbc66380-aaf4-4fc3-bd83-f72ece4cc2e3" />


---

### DIAL-MPC — Unitree Go2 Walking

<!-- 로컬 GIF/MP4 사용 시 (GitHub은 GIF 자동 재생 지원) -->



https://github.com/user-attachments/assets/c6f617f0-534c-4fe9-84ec-8694296bfb28



---

## Key Features

- **Training-Free**: 새로운 환경, 새로운 로봇에 재학습 없이 즉시 적용
- **Multi-Agent Verification**: 토론 기반 경로 검증으로 단일 LLM 대비 신뢰성 향상
- **Visual Grounding**: 이미지 직접 분석 → 별도 센서 맵핑 파이프라인 불필요
- **Memory**: ChromaDB 벡터 DB로 경로 히스토리 저장 및 유사 상황 검색
- **Modular**: 뇌(Brain)와 몸(Body)이 독립 모듈 → 각각 교체/업그레이드 가능

---

## Robot Physical Constraints (Go2)

VLM 에이전트는 경로 생성 시 아래 물리 제약을 준수합니다:

```
- 동적 안전 반경: 0.5m
- 장애물 안전 마진: 0.5m 이상
- 통과 불가 갭: 0.8m 미만
- 스텝 길이: 0.4m ~ 1.0m (권장 0.6~0.7m)
- 최소 회전 반경: 0.5m (급격한 방향 전환 금지)
```

---

## Future Vision: Personality-Weighted Robot

본 아키텍처의 핵심 확장 가능성은 **Brain 모듈에 성격(Personality) 파라미터를 주입**하는 것입니다.

```python
# 예시: 성격 파라미터 주입
personality = {
    "aggression": 0.2,      # 낮을수록 보수적 경로
    "curiosity": 0.8,       # 높을수록 탐색적 행동
    "risk_tolerance": 0.3,  # 낮을수록 안전 마진 증가
    "empathy": 0.9          # 높을수록 주변 개체 회피 우선
}
```

이 파라미터들이 각 에이전트의 프롬프트와 판단 기준에 반영되면:

- **같은 환경**에서도 로봇마다 다른 행동 양식을 보임
- 상황에 따라 성격이 동적으로 변화하는 **감정 기반 로봇** 구현 가능

### 응용 분야

| 분야 | 적용 예시 |
|------|-----------|
| 로봇 경찰견 | 고위험 상황 → aggression↑, risk_tolerance↑ / 민간 구조 → empathy↑ |
| 의료 서비스 로봇 | empathy↑, aggression=0 으로 환자 친화적 행동 |
| 물류 로봇 | efficiency↑, risk_tolerance 중간 |
| 재난 구조 로봇 | curiosity↑ (미지 환경 탐색), risk_tolerance↑ |

단순 도구가 아닌, **맥락을 이해하고 성격에 따라 판단하는 로봇**으로의 전환점이 될 수 있습니다.

---

## Project Structure

```
HELM-Brain/
├── vlm_courtroom/              # 🧠 Brain Module
│   ├── config.py               # Vertex AI (Gemini) 설정
│   ├── main_court.py           # 실행 진입점
│   ├── agents/
│   │   ├── base_agent.py       # 에이전트 공통 인터페이스
│   │   └── specific_agents.py  # 4개 전문 에이전트 구현
│   ├── court/
│   │   └── courtroom.py        # 법정 진행 및 경로 시각화
│   ├── inputs/                 # 입력 이미지 저장
│   └── outputs/                # 경로 시각화 결과 저장
│
├── dial_mpc/                   # 🤖 Body Module (DIAL-MPC)
│   ├── dial_mpc/
│   │   ├── core/               # CMA-ES MPC 핵심 알고리즘
│   │   ├── envs/               # 로봇 환경 정의
│   │   ├── models/             # Unitree Go2, H1 MuJoCo 모델
│   │   └── deploy/             # 실제 로봇 배포 인터페이스
│   └── examples/               # 실행 예시 (trot, jump, loco)
│
└── unitree_go2_trot/           # 실험 결과 데이터
    └── *.html / *.pdf          # Brax 시뮬레이션 시각화
```

---

## Getting Started

### Requirements

```bash
# Python 3.10+
pip install google-cloud-aiplatform chromadb matplotlib
pip install -e dial_mpc/
```

### Vertex AI 설정

Google Cloud 서비스 계정 키를 발급 후:

```python
# vlm_courtroom/config.py
KEY_PATH = "/path/to/your/google_vertex_key.json"
PROJECT_ID = "your-gcp-project-id"
```

### 실행

```bash
python vlm_courtroom/main_court.py
```

### DIAL-MPC 라이브러리 다운로드 
dial-mpc 파일에 있는 README 확인.
---

## References

- **DIAL-MPC**: [Diffusion-Inspired Annealing for Loco-manipulation](https://arxiv.org/abs/2409.15610)
- **Unitree Go2**: Quadruped robot platform
- **MuJoCo MJX**: JAX-accelerated physics simulation
- **Gemini 1.5**: Google DeepMind multimodal language model
