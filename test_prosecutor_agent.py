"""
test_prosecutor_agent.py
수정된 Prosecutor 프롬프트 단독 테스트 -- CoordinateAgent의 실제 출력(Scene D, 0.85m 통로)을 입력으로 사용.
"""
import sys, os
sys.path.insert(0, "/home/user/hyeonsoo/LLM-Robot-Planner")

from vlm_courtroom.config import init_vertex_ai
from vlm_courtroom.agents.specific_agents import ProsecutorAgent

init_vertex_ai()

agent = ProsecutorAgent(backend="gemini")

# 아까 CoordinateAgent가 실제로 출력한 응답을 그대로 붙여넣음
coordinate_output = """
## Path Reading
경로는 시작점 (0.0m, 0.0m)인 녹색 점에서 출발하여 목표점 (6.0m, 0.0m)인 붉은 점까지 이어집니다.
초기에 로봇은 X축 양의 방향으로 이동하며 Y축 양의 방향으로 약간 올라가 대략 (0.4m, 0.5m) 지점까지 이동합니다. 그 후, 경로는 두 개의 붉은색 장애물로 구성된 좁은 통로로 진입합니다.
이 통로는 X축 약 0.8m에서 5.8m까지 이어져 있으며, 상단 장애물은 Y축 약 +0.7m에서 +1.0m, 하단 장애물은 Y축 약 -0.2m에서 -0.5m 사이에 위치합니다.
경로는 이 통로 내부에서 Y축 약 +0.4m에서 +0.6m 사이를 오가며 지그재그 형태로 진행합니다. 통로를 벗어난 후, 경로는 Y축 음의 방향으로 내려와 X축을 따라 목표점까지 직선으로 향합니다.

## Coordinates
```json
[
  {"x": 0.00, "y": 0.00},
  {"x": 0.38, "y": 0.47},
  {"x": 0.96, "y": 0.60},
  {"x": 1.57, "y": 0.59},
  {"x": 2.17, "y": 0.47},
  {"x": 2.76, "y": 0.45},
  {"x": 3.36, "y": 0.57},
  {"x": 3.95, "y": 0.51},
  {"x": 4.55, "y": 0.41},
  {"x": 5.15, "y": 0.53},
  {"x": 5.73, "y": 0.52},
  {"x": 6.00, "y": 0.00}
]
```
"""

context = {
    "last_message_content": coordinate_output,
    "num_waypoints": 12,
}

msg = agent.process(context)
print("\n=== ProsecutorAgent 응답 ===")
print(msg.content)
