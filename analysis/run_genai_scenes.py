"""
run_genai_scenes.py -- Task #8.
2026-08-13 추가 수정 (10차): scene 2 재시도. Coordinate Agent가 장애물 좌표(H 등)를
정확히 알고 있으면서도 자기가 그린 경로 시작 구간(건물 H 경계선 x=0을 그대로
따라가는 구간)에는 0.8m 클리어런스 규칙을 스스로 재검증하지 않은 것 확인 (Prosecutor/
Judge도 못 잡음 -- 제기되지 않은 구간은 검증 안 되는 구조적 사각지대). 모든 웨이포인트를
자신이 나열한 장애물 좌표와 명시적으로 대조 재검증하라는 지시(SELF_CHECK_HINT) 추가.
"""
import os
import sys
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from vlm_courtroom.config import init_vertex_ai
from vlm_courtroom.court.courtroom import VLMCourt

NUM_WAYPOINTS = 15

TARGET_SCENES = ["2"]

PROMPT_LEVEL = {
    "2": "medium",
}

SCENE_ORIGIN_OVERRIDE = {
    "2": (423, 625),
    "3": (345, 508),
    "4": (423, 625),
}

SCENARIO_TEMPLATES_MEDIUM = {
    "2": """
로봇(go2)은 (0,0)에서 시작해서 ({gx},{gy})까지 이동해야 해.
이미지는 어두운 아스팔트 옥상/주차장 위에 여러 채의 회색 건물(옥상 구조물)이
불규칙하게 배치된 도심 지형이야. 화면 전체에 대략 10~14채 정도의 건물이 흩어져
있으니, 이미지 구석구석을 꼼꼼히 확인해서 모든 건물의 위치와 크기를 놓치지 않고
파악해줘. 이 건물들과 충돌하지 않고 건물 사이 빈 공간을 따라 이동하면서, 초록색
원으로 표시된 목표 지점까지 안전하게 도달할 수 있는 {n}개의 좌표를 제시해줘.
""",
}

DEFAULT_TEMPLATE = """
로봇(go2)은 (0,0)에서 시작해서 ({gx},{gy})까지 이동해야 해.
이미지 속 장애물을 반드시 피하면서, 초록색 원으로 표시된 목표 지점까지
안전하게 도달할 수 있는 {n}개의 좌표를 제시해줘.
"""

EFFICIENCY_HINT = """
불필요하게 크게 우회하지 마. 장애물 사이에 로봇이 통과할 수 있는 틈이 있다면
그 틈을 활용해서 최대한 직선에 가까운 효율적인 경로로 목표까지 이동해줘.
안전 마진은 지키되, 장애물과 멀리 떨어져서 큰 원을 그리며 돌아가는 지나치게
보수적인 경로는 피해줘.
"""

GOAL_MATCH_HINT = """
추가 CRITICAL 제약사항: 15번째(마지막) 좌표는 반드시 목표 좌표 ({gx},{gy})와
정확히 동일해야 한다.
"""

SELF_CHECK_HINT = """
추가 CRITICAL 제약사항: 장애물 좌표를 모두 나열한 뒤, 최종 제시할 15개 좌표
각각을 방금 나열한 모든 장애물 좌표와 하나하나 대조해서 유효 클리어런스
반경(0.8m) 이상 떨어져 있는지 직접 재검증해라. 경로의 시작 구간이나 장애물
경계선을 따라가는 구간도 예외 없이 전부 이 재검증 대상에 포함시켜야 한다.
"멀어 보이니 안전할 것"이라는 인상만으로 판단하지 말고, 반드시 좌표 간 거리를
계산해서 확인해라.
"""

GOAL_MARKER_COLOR = {
    "3": "magenta",
}


def detect_goal_world_xy(arr: np.ndarray, robot_px, scale, color="green"):
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    if color == "magenta":
        mask = (r > 150) & (b > 150) & (g < 100)
    else:
        mask = (g > 150) & (r < 100) & (b < 100)
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    cx, cy = xs.mean(), ys.mean()
    gx = (cx - robot_px[0]) / scale
    gy = (robot_px[1] - cy) / scale
    return round(float(gx), 2), round(float(gy), 2)


def compute_image_bounds(robot_px, scale, w, h):
    rx, ry = robot_px
    x_min, x_max = -rx / scale, (w - rx) / scale
    y_min, y_max = -(h - ry) / scale, ry / scale
    return x_min, x_max, y_min, y_max


def get_template(name: str):
    level = PROMPT_LEVEL.get(name, "minimal")
    if level == "medium" and name in SCENARIO_TEMPLATES_MEDIUM:
        return SCENARIO_TEMPLATES_MEDIUM[name], level
    return DEFAULT_TEMPLATE, "default"


def main():
    ai_dir = os.path.join(project_root, "data", "AI")
    if not os.path.isdir(ai_dir):
        print(f"❌ {ai_dir} 없음")
        return

    all_names = sorted(
        [d for d in os.listdir(ai_dir) if os.path.isdir(os.path.join(ai_dir, d))],
        key=lambda x: (len(x), x),
    )
    scene_names = [n for n in all_names if n in TARGET_SCENES]
    print(f"이번 배치 대상 씬: {scene_names}")

    init_vertex_ai()
    court = VLMCourt(backend="gemini", gemini_model="gemini-2.5-flash")

    for name in scene_names:
        scene_dir = os.path.join(ai_dir, name)
        candidates = [f for f in os.listdir(scene_dir) if f.lower().endswith(".png")]
        if not candidates:
            print(f"⚠️  {scene_dir}: png 없음, 스킵")
            continue
        image_path = os.path.join(scene_dir, candidates[0])

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape

        robot_px = SCENE_ORIGIN_OVERRIDE.get(name)
        if robot_px is None:
            print(f"⚠️ {name}: 원점 override 없음, 스킵")
            continue
        print(f"\n{'='*60}\n🧪 AI scene {name}: {image_path} ({w}x{h}px)  (수동 원점 지정: {robot_px})")

        scale = h / 10.0
        goal_color = GOAL_MARKER_COLOR.get(name, "green")
        goal_xy = detect_goal_world_xy(arr, robot_px, scale, color=goal_color)

        x_min, x_max, y_min, y_max = compute_image_bounds(robot_px, scale, w, h)
        print(f"   scale={scale:.2f}px/m  goal_color={goal_color}")
        print(f"   이미지 유효 world 범위: x:[{x_min:.2f},{x_max:.2f}] y:[{y_min:.2f},{y_max:.2f}]")

        if goal_xy is None:
            print(f"   ⚠️ 목표점 자동 검출 실패, 이 씬 스킵 (수동 확인 필요)")
            continue
        gx, gy = goal_xy

        buf = 0.5
        if not (x_min - buf <= gx <= x_max + buf and y_min - buf <= gy <= y_max + buf):
            print(f"   ⚠️ 목표 좌표({gx},{gy})가 유효 범위 밖 -- 스킵 (원점 재확인 필요)")
            continue
        print(f"   목표(goal) world 좌표 자동 검출: ({gx}, {gy})")

        template, level_used = get_template(name)
        print(f"   prompt_level={level_used}")

        boundary_hint = f"""
추가 CRITICAL 제약사항: 모든 좌표의 x 값은 {x_min:.2f}m ~ {x_max:.2f}m 범위 안에,
y 값은 {y_min:.2f}m ~ {y_max:.2f}m 범위 안에 있어야 한다. 이는 이 이미지가 실제로
담고 있는 화면 범위이며, 다른 모든 CRITICAL 물리적 제약사항과 동일한 우선순위의
하드 제약이다. 이 범위를 벗어나는 좌표는 어떤 이유로도 절대 허용되지 않는다.
"""
        goal_match_hint = GOAL_MATCH_HINT.format(gx=gx, gy=gy)
        scenario = (template + EFFICIENCY_HINT + boundary_hint + goal_match_hint + SELF_CHECK_HINT).format(gx=gx, gy=gy, n=NUM_WAYPOINTS)

        try:
            court.run_case(
                scenario, image_path=image_path, robot_pos=robot_px, scale=scale,
                scene_name=f"AI/{name}", num_waypoints=NUM_WAYPOINTS, variant="gemini",
            )
            print(f"✅ AI scene {name} 판결 완료")
        except Exception as e:
            print(f"❌ AI scene {name} 실패: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
