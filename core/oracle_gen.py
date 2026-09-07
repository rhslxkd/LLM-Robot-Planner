"""
================================================================================
 [STUDY VERSION] oracle_gen.py -- 함수/줄 단위 상세 해설판
================================================================================
 원본: core/oracle_gen.py  (실제 파이프라인에서 그대로 쓰이는 코드, 로직 100% 동일)
 이 파일은 교수님 미팅/코드리뷰 준비용으로 "왜 이렇게 짰는지"를 줄 단위로
 설명하기 위해 만든 학습용 사본입니다. 실행에는 원본(core/oracle_gen.py)을
 쓰고, 이 파일은 읽고 이해하는 용도로만 쓰세요.

 이 스크립트가 파이프라인에서 하는 일 (한 줄 요약):
   "이미 완성된 MuJoCo 씬 XML 위에 카메라 하나를 몰래 꽂아서,
    로봇 시점의 top-down(사실상 orthographic 근사) 이미지를 찍어
    data/<scene>/oracle.png 로 저장한다."

 왜 중요한가:
   이 oracle.png 가 Neural A* (run_neural_astar_step.py)의 유일한 입력
   이미지이기 때문에, 여기서 좌표계/스케일이 하나라도 틀리면 그 아래
   모든 단계(Neural A*, waypoint_generator, DIAL-MPC)의 좌표가 전부
   어긋나게 됩니다. 즉 이 파일은 파이프라인 전체의 "좌표계 기준점".
================================================================================
"""

# ─── 표준 라이브러리 ───────────────────────────────────────────────────
import os      # 파일 경로 조작 (dirname, join, exists 등)
import sys     # (이 파일에서는 직접 안 쓰이지만 import 관례상 남아있음 -- 다른 core/ 파일들과
               #  스타일 통일을 위해 남긴 것으로 보임. 실제 sys.* 호출은 없음)
import glob    # 와일드카드 패턴(oracle_scene_*.xml)으로 파일 목록 검색
import argparse  # CLI 인자 파싱 (--no-grid 같은 옵션 처리)

# ─── 서드파티 라이브러리 ───────────────────────────────────────────────
import numpy as np   # fovy 각도 계산(arctan), grid 좌표 스냅 계산 등에 사용
import mujoco         # MuJoCo 물리 엔진의 파이썬 바인딩. 모델 로드/시뮬레이션/렌더링 전부 이걸로 함
from PIL import Image, ImageDraw, ImageFont
    # Image: numpy 배열 <-> PNG 파일 변환
    # ImageDraw: 격자선/텍스트를 이미지 위에 그리는 데 사용
    # ImageFont: 라벨 텍스트에 쓸 폰트 로드


# ══════════════════════════════════════════════════════════════════════
# 섹션 1. 경로 상수
# ══════════════════════════════════════════════════════════════════════
#
# __file__ 은 "이 스크립트 파일 자신의 경로" (core/oracle_gen.py).
# os.path.dirname(__file__)              -> core/                (한 번 위로)
# os.path.dirname(os.path.dirname(__file__)) -> repo 루트          (두 번 위로)
#
# *** 이 부분은 core/ 폴더 이동(리팩터링) 때문에 실제로 버그가 났었던 지점 ***
# 원래 이 파일이 repo 루트에 있었을 때는 dirname(__file__) 한 번만 해도 repo
# 루트가 나왔는데, core/로 옮기면서 한 단계가 더 생겨서 dirname을 한 번 더
# 감싸야 했음. 안 그러면 MODELS_DIR이 "core/dial_mpc/..."로 계산되어 실제
# 파일이 없는 잘못된 경로를 가리키게 됨 (이 세션에서 직접 고친 버그).
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dial_mpc", "dial_mpc", "models", "unitree_go2"
)  # -> <repo_root>/dial_mpc/dial_mpc/models/unitree_go2
   #    씬 XML 파일들이 실제로 저장되어 있는 폴더. 여기서 읽고, 임시 카메라
   #    주입 파일(_oracle_tmp.xml)도 바로 이 폴더 안에 만듦.

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data"
)  # -> <repo_root>/data
   #    렌더링 결과(oracle.png)를 씬별 하위폴더(data/<scene>/)에 저장하는 최상위 폴더.
   #    파이프라인의 모든 산출물(oracle.png, coordinate_proposal.json, verdict.png 등)이
   #    전부 이 DATA_DIR 밑에 씬 이름별로 정리됨.

SCENE_GLOB = "oracle_scene_*.xml"
   # main()에서 인자 없이 실행됐을 때, MODELS_DIR 안에서 이 패턴에 맞는
   # 파일을 전부 찾아서 "한 번에 전체 렌더링"하는 배치 모드로 쓰임.


# ══════════════════════════════════════════════════════════════════════
# 섹션 2. 카메라/이미지 캘리브레이션 상수
# ══════════════════════════════════════════════════════════════════════

IMG_W, IMG_H = 1263, 1080
    # 렌더링될 이미지의 고정 해상도 (가로 x 세로, 픽셀).
    # 이 값이 바뀌면 ROBOT_PX, fovy 계산, 격자 오버레이 전부 다시 계산해야 해서
    # 사실상 파이프라인 전체의 "불변 상수"로 취급됨.

PPM_DEFAULT = 150.0
    # PPM = Pixels Per Meter. "1미터가 화면에서 몇 픽셀인가"를 나타내는 스케일 값.
    # 150이면 1m 이동이 화면에서 150px 이동으로 보임 -- 즉 값이 클수록 "확대"된 좁은 시야,
    # 작을수록 "축소"된 넓은 시야가 됨.

# 씬별로 카메라 시야를 넓혀야 하는 특수 케이스를 위한 오버라이드 딕셔너리.
# main_court.py(예전 VLM Courtroom 코드)의 SCENE_SCALE과 반드시 값을 맞춰야 한다는
# 주석이 달려 있는데, 지금 파이프라인 배치 씬 이름(예: oracle_scene_C1000)은
# 이 딕셔너리의 키("oracle_scene_C")와 "정확히 일치"하지 않기 때문에
# 실제로는 아래 render_scene()의 dict.get()에서 항상 매치 실패 -> PPM_DEFAULT로 폴백됨.
# ⚠️ 사실상 죽은 코드에 가까움 (옛날 손으로 만든 단일 데모 씬 oracle_scene_C.xml 등에만
#    적용됐던 설정으로 추정 -- 지금 배치 파이프라인엔 영향 없음). 코드리뷰 때 언급할 가치 있음.
SCENE_PPM = {
    "oracle_scene_C": 90.0,
    "oracle_scene_E": 90.0,
    "oracle_scene_D": 90.0,
    "oracle_scene_R": 90.0
}

ROBOT_PX = (IMG_W / 3, IMG_H / 2)
    # = (421.0, 540.0)
    # 로봇이 화면에서 찍힐 고정 픽셀 위치. 원래는 화면 정중앙(631.5, 540)이었는데,
    # 로봇을 왼쪽 1/3 지점으로 옮겨서 "전방 시야를 더 넓게" 확보하도록 바꿨다는 주석.
    # (로봇이 화면 중앙에 있으면 전방으로 볼 수 있는 화면 면적이 좌우 절반씩밖에 안 되는데,
    #  왼쪽으로 치우치면 전방 쪽에 더 넓은 여유 공간이 생김 -- 자율주행 카메라 배치와 비슷한 논리)
    #
    # *** 다른 파일과의 일관성 체크 포인트 ***
    # run_neural_astar_step.py, waypoint_generator.py 등에서는 이 값을 그냥
    # ROBOT_PX = (421.0, 540.0) 이라고 리터럴로 하드코딩해뒀는데, 여기서 공식으로
    # 계산해봐도 정확히 같은 값이 나옴 (421.0, 540.0). 즉 여러 파일에 흩어진 상수가
    # 실제로 서로 어긋나지 않는다는 걸 이 공식이 "증명"해주는 셈.

CAM_HEIGHT = 50.0
    # 카메라를 몇 미터 높이에 둘 것인가. 뒤에서 설명하�니와, 이 값이 클수록
    # perspective(원근) 투영이 orthographic(평행) 투영에 더 가까워짐.


# ══════════════════════════════════════════════════════════════════════
# 섹션 3. Grid overlay(격자 오버레이) 설정값
# ══════════════════════════════════════════════════════════════════════
# 렌더링된 이미지 위에 "자"처럼 격자선+숫자 라벨을 반투명하게 합성하는 기능.
# 원래 설계 의도(docstring)는 "VLM에게 텍스트로 숫자 힌트를 주지 않고, 이미지
# 자체에 눈금자를 그려줘서 최소 정보 원칙을 지키면서도 거리 추정을 돕는다"였음.
# 지금은 VLM이 파이프라인에서 빠졌기 때문에 이 원래 의도는 유효하지 않지만,
# 사람이 oracle.png를 눈으로 보고 스케일을 가늠할 때 여전히 유용해서 남아있음.

GRID_MINOR_SPACING_M = 0.5   # 보조선(가는 격자선)을 몇 미터 간격으로 그릴지
GRID_LABEL_SPACING_M = 1.0   # 숫자 라벨은 보조선보다 성긴 1m 간격으로만 표시 (너무 빽빽하면 안 보임)
GRID_LINE_RGBA  = (255, 255, 255, 90)   # 일반 보조선 색: 흰색, alpha=90/255 (꽤 투명해서 씬을 안 가림)
GRID_AXIS_RGBA  = (255, 255, 0, 160)    # x=0 또는 y=0인 "축선"만 노란색+더 불투명하게 강조
                                          # (로봇 원점을 한눈에 찾기 위한 기준선)
GRID_LABEL_RGBA = (255, 255, 0, 220)    # 숫자 텍스트 색: 노란색, 거의 불투명 (읽기 쉽게)


def _fovy_for_ppm(ppm):
    """
    [함수 역할]
    원하는 PPM(픽셀/미터 스케일)을 얻기 위해 카메라의 수직 시야각(fovy, degree)을
    역산한다. MuJoCo 이 버전에는 진짜 orthographic 카메라 모드가 안정적으로 없어서,
    "카메라를 아주 높이(CAM_HEIGHT=50m) 두고 화각을 좁게(망원렌즈처럼)" 잡는 방식으로
    orthographic을 근사하는 트릭의 핵심 계산.

    [수학적 원리]
    핀홀 카메라의 표준 FOV 공식: tan(fovy/2) = (담고 싶은 절반 높이) / (카메라~피사체 거리)
    여기서는:
      - "담고 싶은 절반 높이" = v/2  (v = 화면 세로가 커버해야 할 실제 월드 미터 범위)
      - "카메라~피사체 거리"  = CAM_HEIGHT (카메라가 바닥을 수직으로 내려다보니까,
                                            바닥까지의 거리를 그냥 카메라 높이로 근사)

    [왜 이게 orthographic 근사가 되는가]
    카메라가 아주 멀리(H=50m) 있으면, 화면에 담기는 지점들까지의 실제 거리
    d(r) = sqrt(H² + r²) (r=카메라 바로 아래 지점 기준 수평거리)가 r의 변화에
    거의 영향을 안 받고 전부 H에 가까운 값으로 수렴함. 원근 투영의 스케일은
    이 d(r)에 반비례하니까, d(r)이 거의 일정해지면 결과적으로 "어디를 찍어도
    같은 배율"인 평행투영(orthographic)과 사실상 같아짐.
    (실측: PPM=150 기준 화면 가장자리에서의 왜곡률 <0.5%로 검증됨 - docstring 참고)
    """
    v = IMG_H / ppm
        # 화면 세로(IMG_H 픽셀)가 실제로 몇 미터를 담아야 하는지 역산.
        # 예: IMG_H=1080, ppm=150 이면 v = 1080/150 = 7.2m
        #     (화면 세로 전체가 7.2m 범위의 월드를 보여줘야 한다는 뜻)
    return 2 * np.degrees(np.arctan((v / 2) / CAM_HEIGHT))
        # np.arctan((v/2)/CAM_HEIGHT): 반높이/거리 비율의 아크탄젠트 -> "반각"(라디안)
        # 2 * ... : 반각을 두 배 해서 전체 수직 화각으로 만듦
        # np.degrees(...): 라디안 -> 도(degree) 단위로 변환 (MuJoCo XML의 fovy 속성은 degree 단위)
        # 예: ppm=150 -> v=7.2 -> arctan(3.6/50)=4.12° -> fovy ≈ 8.24° (상당히 좁은 망원 화각)


# ─── 모듈 로드 시 기본값으로 전역 변수 초기화 ───────────────────────────
# 이 두 변수는 "하위호환" 주석이 달려 있는데, world_to_pixel() 등 다른 함수들이
# 인자로 안 받고 그냥 이 전역(global) 변수를 직접 읽는 구조이기 때문에 필요함.
PPM = PPM_DEFAULT
    # 모듈이 처음 import될 때의 초기값. 실제로는 render_scene() 안에서
    # "global PPM" 선언 후 씬마다 SCENE_PPM 조회 결과로 이 값이 덮어써짐.
    # *** 설계상 주의점: 이건 모듈 레벨의 가변 전역 상태(mutable global)임.
    # 지금은 main()이 씬을 for문으로 하나씩 순차 처리하니까 안전하지만,
    # 만약 나중에 여러 씬을 멀티프로세스/멀티스레드로 병렬 렌더링하게 되면
    # 이 전역 변수 때문에 레이스 컨디션(경쟁 상태) 버그가 생길 수 있음. ***
FOVY_DEG = _fovy_for_ppm(PPM)
    # 위 PPM 초기값에 대응하는 fovy를 미리 계산해서 전역에 저장.


# ═══════════════════════════════════════════════════════════════════════
# 섹션 4. 좌표 변환 함수
# ═══════════════════════════════════════════════════════════════════════

def world_to_pixel(wx, wy):
    """
    [함수 역할] 월드 좌표(미터 단위, 로봇 원점 기준) -> 화면 픽셀 좌표 변환.

    [좌표계 컨벤션]
    월드: +x = 로봇 전방, +y = 로봇 좌측(왼쪽)
    픽셀: +u(가로) = 화면 오른쪽, +v(세로) = 화면 아래쪽 (이미지 좌표계의 일반적인 관례)

    이 두 좌표계를 매핑하는 규칙: "전방(+x)=화면 오른쪽, +y=화면 위"
    라고 docstring에 정의되어 있고, 그 결과가 아래 반환식.
    """
    return ROBOT_PX[0] + wx * PPM, ROBOT_PX[1] - wy * PPM
        # --- x(가로) 성분 ---
        # ROBOT_PX[0]: 로봇의 화면상 기준 위치(원점 오프셋)
        # + wx * PPM : 전방으로 wx미터 만큼 이동한 거리를 픽셀 단위로 환산해서 더함
        #              (부호가 +인 이유: "전방 = 화면 오른쪽"이라는 카메라 배치 규칙)
        #
        # --- y(세로) 성분 ---
        # ROBOT_PX[1]: 로봇의 화면상 기준 위치
        # - wy * PPM : 부호가 "-"인 게 핵심. 월드에서 +y(왼쪽/위 방향)로 이동하면
        #              이미지 좌표계에서는 v값이 "작아져야"(위로 올라가야) 하기 때문.
        #              이미지 좌표는 아래로 갈수록 값이 커지는 관례라서, 월드의 "위로 이동"을
        #              픽셀의 "숫자 감소"로 뒤집어줘야 함. 이 부호를 안 뒤집으면
        #              로봇이 왼쪽으로 갔는데 화면에선 아래로 내려가는 상하좌우 반전 버그가 생김.


def _inject_camera(scene_path, fovy_deg):
    """
    [함수 역할]
    씬 XML 파일 내용을 읽어서, top-down으로 내려다보는 "oracle"이라는 이름의
    카메라 하나를 <worldbody> 태그 안에 주입한 임시 파일을 만들어 반환한다.

    [설계 철학과의 연결]
    이 함수는 "원본 씬 파일(scene_path)을 절대 직접 수정하지 않는다"는 원칙을
    지키기 위해, 항상 새로운 임시 파일(_oracle_tmp.xml)에 결과를 쓴다.
    원본 XML은 물리 시뮬레이션(DIAL-MPC 등)이 그대로 로드해서 쓰는 "진실의 원천"이므로
    렌더링 스크립트가 여기에 손대면 이미지와 실제 물리가 어긋날 위험이 있기 때문.
    """
    txt = open(scene_path).read()
        # 씬 XML 파일 전체를 문자열로 읽어들임. (XML 파서를 안 쓰고 그냥 텍스트로
        # 다루는 이유는, 다른 내용은 전혀 안 건드리고 딱 한 줄만 정확한 위치에
        # 삽입하면 되는 단순 작업이라 문자열 치환이 더 가볍고 안전하기 때문으로 보임)

    # ── 카메라 위치 오프셋 계산 ──
    # 로봇(월드 원점 0,0)이 화면 정중앙이 아니라 ROBOT_PX(왼쪽 1/3 지점)에 찍히게
    # 하려면, 카메라 자체를 "월드 좌표 원점"이 아니라 살짝 오프셋된 위치에 둬야 함.
    # 직관: 로봇이 화면에서 왼쪽으로 치우쳐 보이길 원한다면, 카메라를 오히려
    # "오른쪽"으로 옮겨서 찍어야 로봇이 상대적으로 왼쪽에 찍힘 (반대 방향 이동).
    offset_x = (IMG_W / 2 - ROBOT_PX[0]) / PPM
        # IMG_W/2 = 화면 가로 중앙 픽셀, ROBOT_PX[0] = 로봇이 찍히길 원하는 픽셀 위치.
        # 이 둘의 차이(픽셀 단위)를 PPM으로 나눠서 "미터" 단위 오프셋으로 환산.
        # 예: IMG_W/2=631.5, ROBOT_PX[0]=421 -> 차이=210.5px -> /150 = 1.4m
        #     즉 카메라를 월드 +x 방향(전방)으로 1.4m 옮겨야 함.
    offset_y = -(IMG_H / 2 - ROBOT_PX[1]) / PPM
        # 세로도 동일한 논리인데, y축은 world_to_pixel에서처럼 부호가 뒤집혀 있어서
        # 맨 앞에 "-"가 하나 더 붙음 (픽셀 오프셋 방향과 월드 y축 방향이 반대라서 보정).
        # ROBOT_PX[1]=540=IMG_H/2 이므로 실제로 이 예제에서는 offset_y = 0 이 됨
        # (로봇을 세로 방향으로는 안 옮겼으니까 당연한 결과 -- 가로만 옮겼다는 설계와 일치).

    cam = (f'    <camera name="oracle" mode="fixed" '
           f'pos="{offset_x:.6f} {offset_y:.6f} {CAM_HEIGHT}" xyaxes="1 0 0 0 1 0" '
           f'fovy="{fovy_deg:.6f}"/>\n')
        # MuJoCo XML의 <camera> 엘리먼트를 문자열로 직접 작성.
        # name="oracle"      : 나중에 render_scene()에서 renderer.update_scene(camera="oracle")로
        #                       이 이름으로 카메라를 지정해서 씀.
        # mode="fixed"        : 씬 안의 다른 body를 따라다니지 않고 월드 좌표에 고정된 카메라.
        # pos="ox oy CAM_HEIGHT" : 계산한 오프셋(ox,oy)과 고정 높이(50m)에 카메라를 배치.
        # xyaxes="1 0 0 0 1 0" : 카메라의 로컬 x축=월드 (1,0,0), y축=월드 (0,1,0)로 지정.
        #                        MuJoCo는 카메라가 기본적으로 -z 방향(자신의 아래쪽)을 바라보므로,
        #                        이 설정은 결과적으로 "카메라가 정확히 수직 아래(-z, 즉 바닥)를
        #                        내려다보게" 방향을 고정하는 것.
        # fovy="..."          : 위에서 _fovy_for_ppm()으로 계산한 화각.

    if 'name="oracle"' not in txt:
        # 이미 씬 XML에 "oracle"이라는 이름의 카메라가 있으면 또 추가하지 않음 (멱등성 보장).
        # 이 검사가 없으면 같은 씬을 두 번 렌더링할 때(혹은 이미 카메라가 있는 씬을 다시
        # 처리할 때) 카메라가 중복 삽입되어 MuJoCo가 "이름 중복" 에러를 낼 수 있음.
        txt = txt.replace("</worldbody>", cam + "  </worldbody>")
            # 문자열 치환으로 </worldbody> 닫는 태그 바로 앞에 카메라 태그를 끼워넣음.
            # (XML 파서 없이 문자열 replace 하나로 "안전하게" 삽입 가능한 이유는,
            #  </worldbody>가 씬 파일 안에 정확히 한 번만 나타난다는 것이 보장되기 때문)

    tmp = os.path.join(MODELS_DIR, "_oracle_tmp.xml")
        # 임시 파일을 씬 XML들과 "같은 폴더"(MODELS_DIR)에 만듦. 왜냐하면 씬 XML이
        # 다른 부속 파일(예: <include file="mjx_go2_robot_only.xml"/>)을 상대경로로
        # 참조하는 경우가 있는데, 그 상대경로 기준이 되는 폴더가 같아야 include가
        # 정상적으로 해석되기 때문. 다른 폴더(예: /tmp)에 임시파일을 만들면 include
        # 참조가 깨질 수 있음.
    open(tmp, "w").write(txt)
        # 수정된(카메라가 추가된) XML 내용을 임시 파일에 씀.
    return tmp
        # 호출자(render_scene)가 이 임시 파일 경로로 MuJoCo 모델을 로드하게 됨.


def _report_obstacles(model):
    """
    [함수 역할] 순수 디버깅/진단용 콘솔 출력 함수.
    씬 안의 "정적 장애물"(로봇 관절이 아니라 고정된 벽/기둥 같은 것들)의
    월드 좌표와 그게 화면 픽셀 어디에 찍히는지, 화면 안에 들어오는지 여부를 출력한다.
    이걸 통해 씬을 만들 때 장애물이 실수로 카메라 시야 밖에 배치되는 문제를
    렌더링 직후 바로 알아챌 수 있음.
    """
    for bid in range(1, model.nbody):
        # model.nbody: MuJoCo 모델에 정의된 전체 body(강체) 개수.
        # bid=0은 항상 "world" 자체(월드 좌표계 원점)를 가리키는 특수 body라서 1부터 순회.
        if model.body_jntnum[bid] == 0 and model.body_parentid[bid] == 0:
            # 두 조건을 동시에 만족해야 "독립적인 정적 장애물"로 판단:
            #  (1) model.body_jntnum[bid] == 0
            #      : 이 body에 연결된 관절(joint)이 하나도 없음 -> 움직이지 않는 고정 물체.
            #        (로봇 다리처럼 관절이 있는 body는 여기서 제외됨)
            #  (2) model.body_parentid[bid] == 0
            #      : 이 body의 부모가 곧바로 "world"(id=0)임 -> 로봇 몸체의 하위 부속(예: 로봇의
            #        발이나 팔 같은 sub-body)이 아니라, worldbody 바로 밑에 독립적으로 놓인
            #        장애물이라는 뜻. (로봇 자체도 관절 없는 링크가 있을 수 있지만, 그것들은
            #        parentid가 로봇 몸통이지 world가 아니므로 이 필터에 안 걸림)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
                # body의 id를 이름 문자열로 변환. XML에서 name 속성을 안 줬으면 None이
                # 반환되므로, or 뒤의 f"body{bid}"로 대체 이름을 만들어줌 (항상 출력 가능하게).
            wx, wy, _ = model.body_pos[bid]
                # 이 body의 월드 좌표 (x, y, z) 중 x, y만 사용 (z는 높이라서 top-down 뷰와 무관).
                # 참고: model.body_pos는 "부모 프레임 기준 상대 위치"인 경우도 있을 수 있는데,
                # worldbody 바로 밑의 body이므로(parentid==0) 여기서는 곧 절대 월드 좌표와 같음.
            u, v = world_to_pixel(wx, wy)
                # 위에서 정의한 좌표 변환 함수로 픽셀 좌표를 구함.
            inside = "OK" if (0 <= u <= IMG_W and 0 <= v <= IMG_H) else "⚠️화면밖"
                # 계산된 픽셀 좌표가 실제 이미지 범위(0~IMG_W, 0~IMG_H) 안에 드는지 검사.
                # 범위를 벗어나면 "이 장애물은 렌더링된 이미지에 안 보인다"는 경고.
            print(f"     · {name:12s} world({wx:+.2f},{wy:+.2f}) → pixel({u:.0f},{v:.0f}) [{inside}]")
                # 사람이 읽기 좋은 형식으로 한 줄 출력. {name:12s}는 이름을 12칸 폭으로
                # 왼쪽 정렬해서 여러 줄을 출력했을 때 표처럼 정렬되어 보이게 하는 포맷 트릭.


def _draw_grid_overlay(img_array):
    """
    [함수 역할]
    렌더링된 numpy 이미지 배열 위에, 0.5m 간격의 반투명 격자선과 1m 간격의
    좌표 라벨을 합성해서 반환한다. world_to_pixel()이 참조하는 현재 전역
    PPM/ROBOT_PX 값을 그대로 쓰기 때문에, 어떤 씬(PPM이 다르더라도)에 대해서도
    격자가 실제 물리 좌표와 항상 정확히 정합되도록 설계됨.
    """
    base = Image.fromarray(img_array).convert("RGBA")
        # numpy 배열(렌더링 결과)을 PIL Image 객체로 변환하고, RGBA(투명도 채널 포함)
        # 모드로 바꿈. 이후 반투명 오버레이를 "합성(composite)"하려면 알파 채널이 필요.
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        # base와 같은 크기의 완전히 투명한(alpha=0) 새 캔버스를 만듦.
        # 격자선/라벨을 원본 이미지에 직접 그리지 않고 이 별도 레이어에 그린 다음
        # 나중에 합성하는 이유: 반투명도(alpha blending)를 깔끔하게 제어하기 위함.
    draw = ImageDraw.Draw(overlay)
        # 이 overlay 레이어에 선/텍스트를 그릴 수 있는 Draw 객체 획득.

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
            # 시스템에 설치된 DejaVuSans 폰트를 16pt 크기로 로드 시도 (선명한 라벨을 위해).
    except Exception:
        font = ImageFont.load_default()
            # 폰트 파일을 못 찾는 환경(예: 폰트 미설치 서버)에서도 죽지 않도록
            # PIL 내장 기본 폰트로 안전하게 폴백.

    # ── 현재 화면에 실제로 보이는 월드 좌표 범위를 역산 ──
    # (world_to_pixel의 역함수를 손으로 풀어쓴 것: 픽셀 0/IMG_W/IMG_H를 다시 미터로 변환)
    wx_min = (0 - ROBOT_PX[0]) / PPM
        # 화면 왼쪽 끝(픽셀 u=0)에 대응하는 월드 x좌표.
    wx_max = (IMG_W - ROBOT_PX[0]) / PPM
        # 화면 오른쪽 끝(픽셀 u=IMG_W)에 대응하는 월드 x좌표.
    wy_min = (ROBOT_PX[1] - IMG_H) / PPM
        # 화면 맨 아래(픽셀 v=IMG_H)에 대응하는 월드 y좌표.
        # (world_to_pixel의 v = ROBOT_PX[1] - wy*PPM 을 wy에 대해 거꾸로 풂:
        #  IMG_H = ROBOT_PX[1] - wy*PPM  ->  wy = (ROBOT_PX[1]-IMG_H)/PPM)
    wy_max = (ROBOT_PX[1] - 0) / PPM
        # 화면 맨 위(픽셀 v=0)에 대응하는 월드 y좌표.

    def _snap(v, spacing):
        return np.floor(v / spacing) * spacing
            # 로컬 헬퍼: 주어진 값 v를 spacing의 배수 중 v보다 작거나 같은 가장 큰 값으로
            # "내림 스냅"한다. 예: v=1.3, spacing=0.5 -> floor(2.6)*0.5 = 2*0.5 = 1.0
            # 이렇게 하는 이유: 격자선을 세계 좌표의 "0, 0.5, 1.0, 1.5..." 같은 딱 떨어지는
            # 지점부터 그리기 시작해야 자연스러운 격자 모양이 나오기 때문
            # (화면 범위의 애매한 시작점(wx_min=-2.807 같은 값)부터 그리면 격자가 삐뚤어 보임).

    # ── 세로선 그리기 (world x = 상수인 선들) ──
    x = _snap(wx_min, GRID_MINOR_SPACING_M)
        # 화면에 보이는 범위 안에서 가장 왼쪽에 그릴 첫 세로선의 world x좌표.
    while x <= wx_max:
        # wx_min부터 wx_max까지 GRID_MINOR_SPACING_M(0.5m) 간격으로 순회하면서 선을 그림.
        u, _ = world_to_pixel(x, 0)
            # 이 x좌표(y는 아무 값이나 상관없으므로 0)에 대응하는 화면 가로 픽셀 위치 u를 구함.
            # (세로선이니까 어차피 화면 세로 전체(0~IMG_H)에 걸쳐 곧게 그릴 것이라 y는 의미 없음)
        is_axis = abs(x) < 1e-6
            # x가 부동소수점 오차 범위 내에서 0인지 검사 -> "월드 원점을 지나는 축선"인지 판별.
            # (완전히 x==0으로 비교하지 않고 1e-6 오차 허용하는 이유: 부동소수점 연산 특성상
            #  정확히 0.0이 안 나올 수도 있기 때문에 안전하게 근사 비교)
        draw.line([(u, 0), (u, IMG_H)],
                  fill=GRID_AXIS_RGBA if is_axis else GRID_LINE_RGBA,
                  width=2 if is_axis else 1)
            # (u, 0)에서 (u, IMG_H)까지 화면을 세로로 관통하는 직선을 그림.
            # 축선(x=0)이면 더 두껍고(width=2) 진한 노란색(GRID_AXIS_RGBA)으로,
            # 일반 보조선이면 얇고(width=1) 투명한 흰색(GRID_LINE_RGBA)으로 그림.
        if abs(x / GRID_LABEL_SPACING_M - round(x / GRID_LABEL_SPACING_M)) < 1e-6:
            # "x가 GRID_LABEL_SPACING_M(1.0m)의 정수배에 충분히 가까운가"를 검사하는 트릭.
            # x/1.0의 결과를 반올림한 값과 원래 값의 차이가 거의 0이면 정수배라는 뜻.
            # (0.5m 간격으로 도는데 라벨은 1m 간격에서만 찍고 싶어서 0.5m 지점은 건너뛰기 위함)
            draw.text((u + 3, 4), f"{x:+.1f}m", font=font, fill=GRID_LABEL_RGBA)
                # 세로선 바로 오른쪽(u+3픽셀), 화면 맨 위(y=4)에 "+1.5m" 같은 형식으로 라벨 표시.
                # {x:+.1f} 포맷: 부호(+/-)를 항상 표시하고 소수점 첫째자리까지.
        x += GRID_MINOR_SPACING_M
            # 다음 세로선 위치로 이동 (0.5m씩 증가).

    # ── 가로선 그리기 (world y = 상수인 선들) ──
    # 위 세로선 로직과 완전히 대칭적인 구조. u와 v, x와 y의 역할만 바뀜.
    y = _snap(wy_min, GRID_MINOR_SPACING_M)
    while y <= wy_max:
        _, v = world_to_pixel(0, y)
            # 이번엔 y좌표에 대응하는 화면 세로 픽셀 위치 v를 구함 (x는 0으로 고정, 의미 없음).
        is_axis = abs(y) < 1e-6
        draw.line([(0, v), (IMG_W, v)], fill=GRID_AXIS_RGBA if is_axis else GRID_LINE_RGBA,
                  width=2 if is_axis else 1)
        if abs(y / GRID_LABEL_SPACING_M - round(y / GRID_LABEL_SPACING_M)) < 1e-6:
            draw.text((4, v + 3), f"{y:+.1f}m", font=font, fill=GRID_LABEL_RGBA)
        y += GRID_MINOR_SPACING_M

    composited = Image.alpha_composite(base, overlay).convert("RGB")
        # base(원본 렌더링 결과)와 overlay(격자+라벨만 그려진 반투명 레이어)를
        # 알파 채널 값에 따라 자연스럽게 섞어 합성. 마지막에 .convert("RGB")로
        # 알파 채널을 제거하는 이유는, PNG로 저장할 때 불필요한 투명도 채널 없이
        # 일반 RGB 이미지로 저장하기 위함 (다운스트림 코드들도 RGB 3채널을 기대함).
    return np.array(composited)
        # 다시 numpy 배열로 변환해서 반환 (호출자가 PIL Image.fromarray로 저장하기 편하게).


# ═══════════════════════════════════════════════════════════════════════
# 섹션 5. 메인 렌더링 함수
# ═══════════════════════════════════════════════════════════════════════

def render_scene(scene_path, draw_grid=True):
    """
    [함수 역할] 씬 XML 파일 하나를 받아서, 전체 렌더링 파이프라인(카메라 주입 ->
    물리 로드 -> 렌더링 -> 격자 오버레이 -> 저장 -> 임시파일 정리)을 수행한다.
    이 스크립트에서 "실제 일이 벌어지는" 핵심 함수.
    """
    global PPM, FOVY_DEG
        # 함수 안에서 모듈 전역 변수 PPM, FOVY_DEG를 "읽기"만 하는 게 아니라
        # "새로 할당"할 것이기 때문에 반드시 global 선언이 필요함 (안 하면 파이썬이
        # 이 이름을 지역 변수로 새로 만들려다가 "할당 전에 참조" 에러를 냄).

    stem = os.path.splitext(os.path.basename(scene_path))[0]
        # 예: "/repo/.../oracle_scene_R000.xml" -> basename "oracle_scene_R000.xml"
        #     -> splitext로 확장자 분리 -> ("oracle_scene_R000", ".xml") -> [0]으로 이름만 추출
        # 이 stem이 이후 출력 폴더명(data/<stem>/)으로 쓰임.

    lookup_stem = stem[:-4] if stem.endswith("_viz") else stem
        # 씬 이름이 "..._viz"로 끝나면(시각화 전용 별도 변형 씬으로 추정) 뒤의 4글자
        # ("_viz")를 잘라내고 원본 씬 이름으로 SCENE_PPM을 조회하겠다는 의도.
        # 즉 "oracle_scene_C1000_viz"도 "oracle_scene_C1000"과 동일한 PPM 설정을
        # 쓰도록 맞춰주는 로직. (다만 위에서 지적했듯 SCENE_PPM 딕셔너리 키 자체가
        # seed 없는 짧은 이름이라 실제 배치 씬에서는 이 조회가 대부분 미스됨)

    PPM = SCENE_PPM.get(lookup_stem, PPM_DEFAULT)
        # lookup_stem이 SCENE_PPM 딕셔너리에 정확히 있으면 그 값, 없으면(대부분의 경우)
        # PPM_DEFAULT(150.0)로 전역 PPM을 갱신. 이 값이 이후 world_to_pixel(),
        # _draw_grid_overlay() 등 이 씬을 처리하는 동안 호출되는 모든 함수에 반영됨.
    FOVY_DEG = _fovy_for_ppm(PPM)
        # 갱신된 PPM에 맞춰 카메라 화각도 다시 계산.

    scene_out_dir = os.path.join(DATA_DIR, stem)
        # 이 씬의 결과물을 저장할 폴더 경로: data/<stem>/
    os.makedirs(scene_out_dir, exist_ok=True)
        # 폴더가 없으면 생성, 이미 있으면 에러 없이 통과(exist_ok=True).

    tmp = _inject_camera(scene_path, FOVY_DEG)
        # 카메라가 주입된 임시 XML 파일 경로를 받음.
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
            # 임시 XML로부터 MuJoCo 모델(물리 정의 전체)을 로드.
        data = mujoco.MjData(model)
            # 이 모델에 대한 "상태 데이터"(위치, 속도 등 시뮬레이션 중 변하는 값들) 객체 생성.
        mujoco.mj_forward(model, data)
            # 순동역학(forward kinematics/dynamics) 계산을 한 번 실행해서 모든 body의
            # 월드 좌표를 확정시킴. 렌더링 전에 반드시 필요한 단계 -- 안 하면 body_pos 등이
            # 초기화 직후의 미확정 상태일 수 있음 (특히 관절/제약이 있는 복잡한 모델에서).
        with mujoco.Renderer(model, height=IMG_H, width=IMG_W) as r:
            # MuJoCo의 오프스크린 렌더러를 IMG_H x IMG_W 해상도로 생성.
            # with문(context manager)을 쓰는 이유: 렌더러가 내부적으로 GPU/GL 컨텍스트 같은
            # 리소스를 잡고 있는데, with 블록을 벗어나면 자동으로 안전하게 해제되도록 하기 위함.
            r.update_scene(data, camera="oracle")
                # 현재 시뮬레이션 상태(data)를 카메라 "oracle" 시점 기준으로 렌더링 준비.
            img = r.render()
                # 실제 렌더링을 수행해서 numpy 배열(H x W x 3, RGB) 이미지를 얻음.
        if draw_grid:
            img = _draw_grid_overlay(img)
                # 옵션이 켜져 있으면(기본값 True) 격자 오버레이를 합성.
        out = os.path.join(scene_out_dir, "oracle.png")
        Image.fromarray(img).save(out)
            # numpy 배열을 PIL Image로 바꿔서 PNG로 디스크에 저장.
            # 이 파일이 바로 Neural A*가 읽게 될 그 oracle.png.
        print(f"  ✅ {out}" + ("  [grid]" if draw_grid else "  [no-grid]"))
            # 저장 완료 메시지 + 격자 On/Off 여부 표시.
        _report_obstacles(model)
            # 위에서 정의한 디버깅 함수로 장애물 위치 리포트 출력.
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
            # try 블록이 성공하든 예외로 중간에 실패하든 무조건 실행되는 finally에서
            # 임시 카메라 XML 파일을 삭제. 이걸 안 지우면 (a) 디스크에 쓰레기 파일이
            # 쌓이고, (b) 다음 씬을 처리할 때 이 파일이 남아있어서 혼란을 줄 수 있음
            # (다만 애초에 매번 새로 덮어쓰긴 하지만, 정리하는 게 안전한 습관).


# ═══════════════════════════════════════════════════════════════════════
# 섹션 6. CLI 진입점
# ═══════════════════════════════════════════════════════════════════════

def main():
    """
    [함수 역할] 커맨드라인 인자를 파싱해서, 렌더링할 씬 목록을 결정하고
    render_scene()을 각각에 대해 호출하는 배치 실행기.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="*",
                         help="특정 oracle_scene_*.xml 파일명(들). 없으면 전체.")
        # 위치 인자(positional argument) "scenes"를 정의. nargs="*"는 "0개 이상의
        # 값을 리스트로 받는다"는 뜻 -- 즉 인자를 안 줘도 되고(빈 리스트), 여러 개
        # 공백으로 나열해도 됨 (예: python oracle_gen.py A.xml B.xml).
    parser.add_argument("--no-grid", action="store_true",
                         help="0.5m 격자 오버레이 끄기 (raw 이미지)")
        # 플래그 옵션. action="store_true"는 "이 옵션이 있으면 True, 없으면 기본값 False"라는 뜻.
        # 즉 --no-grid를 붙이면 args.no_grid == True가 됨.
    args = parser.parse_args()
        # 실제로 sys.argv를 파싱해서 위 정의에 맞게 값을 채운 객체를 얻음.

    if args.scenes:
        # 사용자가 씬 이름을 하나 이상 직접 지정한 경우.
        scenes = [os.path.join(MODELS_DIR, a) if not os.path.isabs(a) else a
                  for a in args.scenes]
            # 각 인자 a에 대해:
            #  - 만약 이미 절대경로(os.path.isabs)면 그대로 사용
            #  - 절대경로가 아니면(보통 "oracle_scene_R000.xml"처럼 파일명만 준 경우)
            #    MODELS_DIR 기준 상대경로로 자동 완성.
            # 이 덕분에 사용자는 매번 긴 전체 경로를 안 쓰고 파일명만 넘기면 됨.
    else:
        # 인자가 하나도 없으면 "전체 배치 모드" -- MODELS_DIR 안의 모든
        # oracle_scene_*.xml 패턴 파일을 찾음.
        scenes = sorted(glob.glob(os.path.join(MODELS_DIR, SCENE_GLOB)))
            # sorted()로 정렬하는 이유: glob 결과의 순서는 파일시스템에 따라 달라질 수
            # 있는데, 정렬해두면 실행할 때마다 항상 같은 순서로 처리되어(결정론적) 로그를
            # 비교하거나 재현하기 편해짐.

    if not scenes:
        print(f"❌ 렌더할 씬 없음. {MODELS_DIR}/{SCENE_GLOB} 를 만들거나 인자로 지정해.")
        return
            # 렌더링할 씬이 하나도 없으면 에러 메시지 출력 후 조용히 함수 종료
            # (예외를 던지지 않고 그냥 return하는 이유: 이건 프로그램 버그가 아니라
            #  "아직 씬을 안 만들었다"는 정상적인 사용자 실수 상황이라 부드럽게 안내).

    print(f"[Calib] {IMG_W}x{IMG_H}, {PPM}px/m, cam H={CAM_HEIGHT}m fovy={FOVY_DEG:.3f}deg, 로봇중심={ROBOT_PX}")
        # 현재 사용될 캘리브레이션 값 요약 출력. 여기서 찍히는 PPM/FOVY_DEG는 아직
        # 모듈 로드 시점의 기본값(PPM_DEFAULT 기준)이고, 실제로는 각 씬을 처리할 때마다
        # render_scene() 내부에서 SCENE_PPM 조회로 다시 갱신됨 (이 출력은 "시작 시점"의
        # 참고용 정보이지, 모든 씬에 그대로 적용된다는 뜻은 아님 -- 약간 헷갈릴 수 있는 부분).
    print(f"[Grid] {'ON (0.5m minor / 1m label)' if not args.no_grid else 'OFF'}")
    print(f"[Render] {len(scenes)}개 씬")
    for sp in scenes:
        if not os.path.exists(sp):
            print(f"  ⚠️ 없음: {sp}"); continue
                # 지정된 경로에 파일이 실제로 없으면(오타 등) 경고만 찍고 다음 씬으로 넘어감
                # -- 하나가 없다고 전체 배치가 죽지 않도록 하는 방어적 코드.
        render_scene(sp, draw_grid=not args.no_grid)
            # --no-grid 플래그의 반대값을 draw_grid로 전달 (플래그가 있으면 grid 끔).
    print(f"\n출력: {DATA_DIR}/<scene_name>/oracle.png")
        # 마지막으로 결과물이 어디에 저장됐는지 안내.


if __name__ == "__main__":
    main()
        # 이 파일이 "python oracle_gen.py"처럼 직접 실행됐을 때만 main()을 호출.
        # 다른 파일에서 "import oracle_gen"으로 이 모듈을 불러올 경우에는
        # main()이 자동 실행되지 않음 (모듈 재사용성을 위한 파이썬 표준 관례).