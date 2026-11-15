"""
image_occupancy.py ─ BEV 이미지 → 점유 지도 (고전 알고리즘에 필요한 '별도 분할 단계')

표 ②에서 A*를 두 줄로 보고한다:
  A* + GT 지도      : 분할이 완벽하다면 (상한)
  A* + 이미지 분할   : 이 모듈의 결과를 지도로 사용 (현실)
두 줄의 차이가 "고전 알고리즘은 정확한 지도를 전제한다"는 점을 수치로 보여준다.

방법 (color_floodfill)
  로봇 시작 픽셀 주변 바닥색을 기준으로, 색 거리 tol 이내이며 시작점과 연결된 영역을 '바닥'으로 보고
  나머지를 장애물로 본다. 학습이 필요 없는 가장 단순한 분할이라 재현이 쉽다.
  (러그·무늬 바닥·그림자에서 틀리는 것이 이 베이스라인의 전형적 실패 모드)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from analysis.plan_metrics import OccGrid


def floor_mask_floodfill(bev: np.ndarray, seed_px, tol: float = 28.0, seed_win: int = 6,
                         close_px: int = 3) -> np.ndarray:
    img = bev.astype(float)
    u, v = int(round(seed_px[0])), int(round(seed_px[1]))
    patch = img[max(v - seed_win, 0):v + seed_win + 1, max(u - seed_win, 0):u + seed_win + 1].reshape(-1, 3)
    ref = np.median(patch, axis=0)
    close = np.linalg.norm(img - ref, axis=-1) <= tol
    if close_px > 0:                                           # 결 무늬(마루 줄 등) 메우기
        st = np.ones((2 * close_px + 1, 2 * close_px + 1), bool)
        close = ndimage.binary_closing(close, structure=st)
    lab, _ = ndimage.label(close)
    if lab[v, u] == 0:
        return np.zeros(close.shape, bool)
    return lab == lab[v, u]


def occupancy_from_image(bev: np.ndarray, ppm: float, robot_px, like: OccGrid, **kw) -> OccGrid:
    """like와 같은 격자(해상도·원점·크기)로 '바닥이 아니면 점유' 지도를 만든다."""
    floor = floor_mask_floodfill(bev, robot_px, **kw)
    H, W = like.occ.shape
    rr, cc = np.mgrid[0:H, 0:W]
    xy = like.to_world(rr, cc)
    u = np.round(robot_px[0] + xy[..., 0] * ppm).astype(int)
    v = np.round(robot_px[1] - xy[..., 1] * ppm).astype(int)
    inside = (u >= 0) & (u < bev.shape[1]) & (v >= 0) & (v < bev.shape[0])
    occ = np.ones((H, W), bool)
    occ[inside] = ~floor[v[inside], u[inside]]
    return OccGrid(occ, like.res, like.origin)
