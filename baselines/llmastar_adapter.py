"""
llmastar_adapter.py ─ LLM-A* (Meng et al., EMNLP Findings 2024; MIT) 공식 패키지를 우리 실내 씬에 연결

원본 그대로 쓰는 것
  - 탐색 알고리즘: llmastar.pather.LLMAStar (LLM 제안 waypoint를 순차 target으로 쓰는 A*)
  - 프롬프트: gpt_prompt['standard'] (5-shot, "Generated Path: [[x, y], ...]")
  - 응답 파싱: llmastar.utils.list_parse, 유효 노드 필터
바꾼 것 (논문에 명시)
  - LLM: GPT-3.5-turbo-instruct → Gemini 2.5 Pro (temperature 0, 원본과 같은 값). 다른 방법과 같은 모델로 맞추기 위함.
    원본의 max_tokens=1000은 적용하지 않음 (Gemini 2.5의 thinking 토큰이 출력 한도를 소모하기 때문).
  - 지도 입력: 원본은 장애물 선분 목록을 직접 받는다. 우리는 점유 지도(GT 또는 이미지 분할)를
      ① A* 베이스라인과 같은 반경으로 팽창 → ② 0.25 m 격자로 축소 (원 논문 맵 50×30과 비슷한 규모)
      → ③ 직사각형 병합 → ④ 각 직사각형의 테두리를 수평/수직 선분으로 변환해서 넘긴다.
  - 출력 후처리: 다른 격자 기반 방법과 같게 line-of-sight 단축 → N개 waypoint.

설치 (vlm_court 환경을 깨지 않도록 의존성 없이)
  pip install --no-deps llm-astar==0.1.4 shapely
  (패키지가 import하는 openai/torch/transformers는 LLaMA·GPT 경로에서만 쓰여서, 없으면 빈 모듈로 대체한다)
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
import types

import numpy as np
from scipy import ndimage

from analysis.plan_metrics import OccGrid
from baselines.grid_planners import DEFAULT_INFLATE_M, inflate, shortcut, to_n_waypoints
from indoor.build_scene import coarsen, merge_rectangles

LATTICE_UNIT_M = 0.2       # 0.25면 팽창 후 좁은 통로가 격자점 한 칸보다 좁아져 끊기는 경우가 있었음


def _import_llmastar():
    for mod in ("openai", "torch", "transformers"):
        if mod not in sys.modules and importlib.util.find_spec(mod) is None:
            sys.modules[mod] = types.ModuleType(mod)          # 쓰지 않는 GPT/LLaMA 경로의 import만 통과
    from llmastar.pather.llm_a_star.llm_a_star import LLMAStar
    from llmastar.pather.llm_a_star.prompt import gpt_prompt
    return LLMAStar, gpt_prompt


# ─── 점유 지도 → 격자점(lattice) + 장애물 선분 ─────────────────────────
class LatticeMap:
    """격자점 (i, j) ↔ world (x0 + i·u, y0 + j·u). 장애물 = 거친 셀 직사각형의 테두리 선분."""

    def __init__(self, grid: OccGrid, inflate_m: float = DEFAULT_INFLATE_M, unit_m: float = LATTICE_UNIT_M,
                 margin_cells: int = 1):
        self.fine = inflate(grid, inflate_m)                  # 단축 후처리용 (A*와 같은 팽창 지도)
        coarse = coarsen(self.fine, unit_m)
        u = coarse.res
        # 시작점에서 갈 수 있는 빈 공간의 bounding box만 잘라서 프롬프트를 짧게 유지
        lab, _ = ndimage.label(~coarse.occ)
        r0, c0 = (int(v) for v in coarse.to_cell(np.array([0.0, 0.0])))
        start_lab = lab[r0, c0] if 0 <= r0 < lab.shape[0] and 0 <= c0 < lab.shape[1] else 0
        if start_lab == 0:                                     # 시작 셀이 팽창에 걸리면 가장 가까운 빈 성분 사용
            dist, (rr, cc) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
            start_lab = lab[rr[r0, c0], cc[r0, c0]]
        rows, cols = np.where(lab == start_lab)
        rlo, rhi = max(rows.min() - margin_cells, 0), min(rows.max() + margin_cells, coarse.occ.shape[0] - 1)
        clo, chi = max(cols.min() - margin_cells, 0), min(cols.max() + margin_cells, coarse.occ.shape[1] - 1)
        occ = coarse.occ[rlo:rhi + 1, clo:chi + 1].copy()
        # 갈 수 없는 다른 빈 영역(방 바깥 등)도 장애물로 채움 → LLM에게 넘길 선분 수를 줄임
        occ |= (lab[rlo:rhi + 1, clo:chi + 1] != start_lab)
        self.occ, self.unit = occ, u
        # 거친 셀 (r, c)의 좌하단 모서리 world 좌표 = coarse.origin + ((c - 0.5)u, (r - 0.5)u)
        self.x0 = coarse.origin[0] + (clo - 0.5) * u
        self.y0 = coarse.origin[1] + (rlo - 0.5) * u
        H, W = occ.shape
        self.range_x, self.range_y = [0, W + 1], [0, H + 1]   # 격자점 0..W, 0..H (경계 = 장애물)
        hz, vt = set(), set()
        for r, c, h, w in merge_rectangles(occ):
            hz.add((r, c, c + w)); hz.add((r + h, c, c + w))
            vt.add((c, r, r + h)); vt.add((c + w, r, r + h))
        self.horizontal = [list(b) for b in sorted(hz)]
        self.vertical = [list(b) for b in sorted(vt)]
        # 격자점 연결성: 셀이 이어져 있어도 통로 폭이 셀 1칸이면 지나갈 격자점이 없다 → 격자점 그래프로 판정.
        # 두 자유 격자점 사이의 8방향 이동은 장애물 선분을 가로지르지 않는다 (인접 셀이 모두 비어 있으므로).
        free = np.zeros((W + 1, H + 1), bool)
        for i in range(1, W):
            for j in range(1, H):
                free[i, j] = not occ[j - 1:j + 1, i - 1:i + 1].any()
        self.node_label, _ = ndimage.label(free, structure=np.ones((3, 3)))

    def to_lattice(self, xy) -> tuple[int, int]:
        return (int(round((xy[0] - self.x0) / self.unit)), int(round((xy[1] - self.y0) / self.unit)))

    def to_world(self, ij) -> np.ndarray:
        ij = np.asarray(ij, float)
        return np.stack([self.x0 + ij[..., 0] * self.unit, self.y0 + ij[..., 1] * self.unit], -1)

    def node_free(self, i: int, j: int) -> bool:
        """격자점 주변 4개 셀이 모두 비어 있어야 자유 (선분 위·직사각형 안이 아님)."""
        H, W = self.occ.shape
        if not (0 < i < W and 0 < j < H):
            return False
        return not self.occ[j - 1:j + 1, i - 1:i + 1].any()

    def snap(self, xy, max_r: int = 8, component: int | None = None) -> tuple[int, int] | None:
        """가장 가까운 자유 격자점. component를 주면 그 연결 성분 안에서만 찾는다."""
        i0, j0 = self.to_lattice(xy)
        best = None
        for di in range(-max_r, max_r + 1):
            for dj in range(-max_r, max_r + 1):
                i, j = i0 + di, j0 + dj
                if not self.node_free(i, j):
                    continue
                if component is not None and self.node_label[i, j] != component:
                    continue
                if best is None or di * di + dj * dj < best[0]:
                    best = (di * di + dj * dj, (i, j))
        return best[1] if best else None

    def query(self, start_ij, goal_ij) -> dict:
        return {"start": list(start_ij), "goal": list(goal_ij), "size": [self.range_x[1], self.range_y[1]],
                "horizontal_barriers": [list(b) for b in self.horizontal],
                "vertical_barriers": [list(b) for b in self.vertical],
                "range_x": list(self.range_x), "range_y": list(self.range_y)}


# ─── Gemini 클라이언트 (원본 ChatGPT.ask와 같은 인터페이스) ──────────────
class GeminiAsk:
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.0, max_retries: int = 5):
        from vertexai.generative_models import GenerationConfig, GenerativeModel
        self.model = GenerativeModel(model_name)
        self.config = GenerationConfig(temperature=temperature)
        self.model_name, self.temperature, self.max_retries = model_name, temperature, max_retries
        self.calls: list[dict] = []

    def ask(self, prompt, stop=None, max_tokens=None):
        t0, last = time.perf_counter(), None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.model.generate_content(prompt, generation_config=self.config)
                um = getattr(r, "usage_metadata", None)
                self.calls.append({"latency_s": round(time.perf_counter() - t0, 3), "retries": attempt,
                                   "prompt_tokens": getattr(um, "prompt_token_count", None),
                                   "output_tokens": getattr(um, "candidates_token_count", None),
                                   "thinking_tokens": getattr(um, "thoughts_token_count", None),
                                   "api_error": False, "text": r.text})
                return r.text
            except Exception as e:
                last = e
                if not any(k in str(e) for k in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "Deadline")) \
                        or attempt == self.max_retries:
                    break
                time.sleep(5 * 2 ** attempt)
        self.calls.append({"latency_s": round(time.perf_counter() - t0, 3), "api_error": True, "error": str(last)})
        raise RuntimeError(f"Gemini API error: {last}")


class _NoPlot:
    def animation(self, *a, **k):
        pass


def make_llmastar(llm_client, prompt: str = "standard"):
    LLMAStar, _ = _import_llmastar()

    class LLMAStarGemini(LLMAStar):
        """원본 LLMAStar에서 LLM 객체만 바꾸고(GPT 프롬프트 템플릿 그대로), 그림 저장을 끄고, 경로를 돌려받는다."""

        def __init__(self, client, prompt="standard"):
            self.llm = "gpt"                                   # gpt_prompt 템플릿 사용
            self.model = client
            self.parser = None
            self.prompt = prompt
            self.path_ = None

        def _initialize_parameters(self, input_data):
            super()._initialize_parameters(input_data)
            self.plot = _NoPlot()

        def extract_path(self, PARENT):
            self.path_ = super().extract_path(PARENT)
            return self.path_

    return LLMAStarGemini(llm_client, prompt)


def plan_llmastar(grid: OccGrid, goal_xy, n_waypoints: int, llm_client, inflate_m: float = DEFAULT_INFLATE_M,
                  unit_m: float = LATTICE_UNIT_M, prompt: str = "standard") -> dict:
    t0 = time.perf_counter()
    lm = LatticeMap(grid, inflate_m, unit_m)
    s = lm.snap((0.0, 0.0))
    g = lm.snap(goal_xy, component=lm.node_label[s]) if s is not None else None
    info = {"lattice_size": [lm.range_x[1], lm.range_y[1]], "n_barriers": len(lm.horizontal) + len(lm.vertical),
            "lattice_unit_m": lm.unit, "inflate_m": inflate_m}
    if s is None or g is None:
        # 지도 변환(팽창 + 격자화) 단계에서 목표가 시작점과 끊긴 경우. LLM 호출 전이므로 방법의 실패가 아니라
        # 변환의 한계로 기록한다 (lattice_disconnected=True). --lattice-unit을 줄이면 대부분 해결된다.
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "lattice_disconnected": True,
                "reason": "start/goal not connected on lattice", **info}
    planner = make_llmastar(llm_client, prompt)
    try:
        res = planner.searching(query=lm.query(s, g))
    except RuntimeError as e:                                   # API 오류 → 제외 대상
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "reason": str(e), "api_error": True, **info}
    except Exception as e:                                      # 응답 파싱 실패·목표 도달 실패 → 방법의 실패
        return {"path": None, "plan_time_s": time.perf_counter() - t0, "reason": f"{type(e).__name__}: {e}", **info}
    poly = lm.to_world(planner.path_)
    poly[0], poly[-1] = [0.0, 0.0], np.asarray(goal_xy, float)
    poly = to_n_waypoints(shortcut(lm.fine, poly), n_waypoints)
    return {"path": poly, "plan_time_s": time.perf_counter() - t0, "llm_targets": res.get("llm_output"),
            "operation": res.get("operation"), "storage": res.get("storage"), **info}
