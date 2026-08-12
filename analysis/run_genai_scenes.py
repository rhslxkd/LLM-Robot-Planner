"""
run_genai_scenes.py -- Task #8: 생성형 AI가 만든 bird's-eye 장애물 이미지에서도
courtroom이 작동하는지 검증. DIAL-MPC는 못 돌림(대응하는 실제 MuJoCo 씬이 없음) --
courtroom의 판결(JSON 좌표) + 시각적 검증(verdict plot)까지만 확인.

로봇 world-origin(0,0) 픽셀 위치는 각 이미지 정중앙으로 가정, scale은
세로축이 -5m~+5m(10m)로 일관 프레이밍된 걸 이용해 (이미지 높이/10)으로 계산.
로봇 실제 시작점과 목표 좌표는 이미지 눈금 라벨을 직접 읽어 추정한 값(육안 추정,
±0.3~0.5m 오차 가능)을 프롬프트에 명시.
"""
import os
import sys
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from vlm_courtroom.config import init_vertex_ai
from vlm_courtroom.court.courtroom import VLMCourt

# (start_x, start_y, goal_x, goal_y) -- 이미지 눈금 라벨 육안 추정치
SCENES = {
    "1": (-1.5, 0.0, 2.2, 3.8),
    "2": (2.0, -3.5, -1.3, 4.3),
    "3": (-1.8, 0.0, 2.2, 3.5),
    "4": (-1.3, -3.3, 3.0, 4.0),
    "5": (-1.5, 4.0, 2.0, 3.5),
    "6": (-1.8, -3.3, 0.0, 0.0),   # 나선형, 실패 예상 -- 스트레스 테스트
    "8": (2.3, 0.0, -1.5, -0.5),
    "10": (-1.8, 0.0, 3.0, 3.0),
}

NUM_WAYPOINTS = 15

SCENARIO_TEMPLATE = """
로봇(go2)은 ({sx},{sy})에서 시작해서 ({gx},{gy})까지 이동해야 해.
이미지 속 빨간 벽/장애물을 반드시 피하면서, 초록색 원으로 표시된 목표 지점까지
안전하게 도달할 수 있는 {n}개의 좌표를 제시해줘.
"""

def main():
    init_vertex_ai()
    court = VLMCourt(backend="gemini", gemini_model="gemini-2.5-flash")

    for name, (sx, sy, gx, gy) in SCENES.items():
        image_path = os.path.join(project_root, "data", "AI", name, f"gptAI{name}.png")
        if not os.path.exists(image_path):
            print(f"⚠️  {image_path} 없음, 스킵")
            continue

        img = Image.open(image_path)
        w, h = img.size
        robot_pos = (w / 2.0, h / 2.0)
        scale = h / 10.0

        scenario = SCENARIO_TEMPLATE.format(sx=sx, sy=sy, gx=gx, gy=gy, n=NUM_WAYPOINTS)
        print(f"\n{'='*60}\n🧪 AI scene {name} ({w}x{h}px, robot_pos={robot_pos}, scale={scale:.2f})")
        print(f"   start=({sx},{sy}) goal=({gx},{gy})")

        try:
            court.run_case(
                scenario, image_path=image_path, robot_pos=robot_pos, scale=scale,
                scene_name=f"AI/{name}", num_waypoints=NUM_WAYPOINTS, variant="gemini",
            )
            print(f"✅ AI scene {name} 판결 완료")
        except Exception as e:
            print(f"❌ AI scene {name} 실패: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
