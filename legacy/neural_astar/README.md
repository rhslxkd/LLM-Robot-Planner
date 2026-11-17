# 현수가 만든 Neural A* 파이프라인 원본 (수정 없음)

`feature/neural-astar-finetune` 브랜치의 `core/` 파일을 그대로 복사했다 (waypoint_generator 마지막 수정 091a16d).

- `run_neural_astar_step.py`: oracle 이미지의 빨간 장애물 → 32×32 축소(칸 안에 하나라도 빨강이면 장애물 = max-pool)
  → Neural A* → 원래 해상도로 확대 → line-of-sight 단순화(팽창 마스크) → 0.65 m 보폭 기준 개수로 보간 → 보폭 0.4~1.0 m 강제
- `waypoint_generator.py`: 거리장(EDT) 기반 안전마진 보정. 점 여유(push 0.4 m, 통과 0.3 m) → 코너 컷 → 보폭 → 수렴 판정

`experiments/walkthrough_neural_astar.py`가 이 두 파일의 함수를 그대로 불러서 Replica 씬 하나에 단계별로 적용한다.
