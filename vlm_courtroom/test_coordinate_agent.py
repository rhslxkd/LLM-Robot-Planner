"""
test_coordinate_agent.py
수정된 CoordinateAgent 프롬프트만 단독 테스트 -- Neural A* 오버레이 이미지 입력.
"""
import sys, os
sys.path.insert(0, "/home/user/hyeonsoo/LLM-Robot-Planner")

from vlm_courtroom.config import init_vertex_ai
from vlm_courtroom.agents.specific_agents import CoordinateAgent

init_vertex_ai()  # gemini 백엔드 기준

agent = CoordinateAgent(backend="gemini")

context = {
    "image_path": "data/oracle_scene_D/neural_A*/nearest_path_overlay_solo.png",
    "image_description": "좁은 통로를 지나는 로봇 씬. 이미지에 이미 경로가 그려져 있음.",
    "num_waypoints": 12,
}

msg = agent.process(context)
print("\n=== CoordinateAgent 응답 ===")
print(msg.content)