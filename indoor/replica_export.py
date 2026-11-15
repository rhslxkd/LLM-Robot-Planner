"""
indoor/replica_export.py ─ Replica 실내 스캔(mesh.ply) → Go2Export 형식 (bev.png, occupancy.png, meta.json).

Unity 내보내기(unity/Editor/Go2BevExporter.cs)와 같은 형식을 만들므로 이후 단계
(indoor/build_scene.py → run_tables.sh)는 그대로 쓴다.

데이터: Replica Dataset (Straub et al., 2019), https://github.com/facebookresearch/Replica-Dataset
        비상업 연구용 라이선스. 씬마다 <scene>/mesh.ply (정점 색이 있는 quad mesh, z-up, 단위 m).

사용법
  # 1) 씬 전체를 위에서 본 지도 + 시작/목표 후보 (한 번, 씬마다 수십 초~수 분)
  python indoor/replica_export.py scan  --mesh /data/replica/room_0/mesh.ply --name room_0
      → replica_cache/room_0/{master.npz, overview.png, pairs.json}
  # 2) 고른 후보를 내보내기 (overview.png의 번호)
  python indoor/replica_export.py export --name room_0 --pairs 0 3
      → Go2Export/replica_room_0_p0/, Go2Export/replica_room_0_p3/
  # 직접 지정: 씬 좌표 (m), yaw는 도
  python indoor/replica_export.py export --name room_0 --start 1.0 0.5 90 --goal 4.2 3.0 --tag a

정의 (논문에 그대로 옮길 것)
  - 바닥 높이 z_f : 위를 향한 수평 면의 면적 가중 높이 히스토그램에서 가장 낮은 주요 봉우리
  - 바닥 높이장 h : z_f ± 0.25 m 안에서 위를 향한 면의 위치별 최저 높이 (0.05 m 셀 중앙값, 0.35 m 중앙값 필터).
                    스캔 바닥의 기울기·휨을 흡수한다. 아래 높이는 모두 h 기준
  - BEV          : 높이 ≤ h + 1.8 m 인 표면을 위에서 정사영, 픽셀마다 가장 높은 점의 색 (천장 제거)
  - 점유         : h + 0.05 ~ h + 0.45 m 높이 띠(Go2 몸통 높이)에 표면이 있거나, 바닥 표면이 없는 셀
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plan_metrics import OccGrid  # noqa: E402
from baselines.grid_planners import grid_search, inflate  # noqa: E402

# Go2Export 규약 (Go2BevExporter.cs와 동일)
IMG_W, IMG_H, PPM = 1263, 1080, 90.0
ROBOT_PX = (421.0, 540.0)
CELL = 0.05
BAND = (0.05, 0.45)

MASTER_PPM = 180.0          # 씬 좌표계 원본 지도 해상도 (5.6 mm/px)
CUT_M = 1.8                 # 바닥 위 이 높이까지만 그림 (천장·상부장 제거)
FLOOR_TOL = 0.04            # 바닥 높이장에서 이 오차 안이면 바닥 표면
FLOOR_WIN = 0.25            # 바닥 높이장을 찾는 범위 (z_f ± 이 값): 스캔의 기울기·휨 허용
SAMPLE_XY = 0.004           # 표면 샘플 간격 (수평 투영, m) < 원본 지도 픽셀 0.0056
SAMPLE_Z = 0.02             # 표면 샘플 간격 (수직, m)
CACHE_DIR = "replica_cache"


# ─── PLY ────────────────────────────────────────────────────────────────
_PLY_T = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
          "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
          "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def load_ply(path: str):
    """binary_little_endian PLY → (V[N,3] float32, C[N,3] uint8, F[M,k] int32). 면은 모두 같은 꼭짓점 수여야 한다."""
    with open(path, "rb") as f:
        elems, cur = [], None
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line == "end_header":
                break
            t = line.split()
            if not t:
                continue
            if t[0] == "format":
                assert t[1] == "binary_little_endian", f"지원하지 않는 PLY 형식: {t[1]}"
            elif t[0] == "element":
                cur = {"name": t[1], "n": int(t[2]), "props": []}
                elems.append(cur)
            elif t[0] == "property":
                cur["props"].append(t[1:])
        V = C = F = None
        for e in elems:
            if e["name"] == "vertex":
                dt = np.dtype([(p[-1], "<" + _PLY_T[p[0]]) for p in e["props"]])
                a = np.fromfile(f, dt, e["n"])
                V = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
                C = (np.stack([a["red"], a["green"], a["blue"]], 1).astype(np.uint8)
                     if "red" in dt.names else np.full((e["n"], 3), 200, np.uint8))
            elif e["name"] == "face":
                lists = [p for p in e["props"] if p[0] == "list"]
                assert len(lists) == 1 and e["props"][0][0] == "list", "face 속성 형식이 예상과 다름"
                ct, it = _PLY_T[lists[0][1]], _PLY_T[lists[0][2]]
                pos = f.tell()
                k = int(np.fromfile(f, "<" + ct, 1)[0])
                f.seek(pos)
                extra = [(p[-1], "<" + _PLY_T[p[0]]) for p in e["props"][1:]]
                dt = np.dtype([("n", "<" + ct), ("v", "<" + it, (k,))] + extra)
                a = np.fromfile(f, dt, e["n"])
                assert (a["n"] == k).all(), "면의 꼭짓점 수가 섞여 있음 (삼각형/사각형 혼합은 미지원)"
                F = a["v"].astype(np.int32)
            else:
                raise ValueError(f"예상하지 못한 PLY element: {e['name']}")
    return V, C, F


def triangles(F: np.ndarray) -> np.ndarray:
    if F.shape[1] == 3:
        return F
    assert F.shape[1] == 4
    return np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])


# ─── 바닥 높이 ──────────────────────────────────────────────────────────
def floor_levels(V, T, bin_m=0.02):
    """위를 향한 수평 면의 면적 가중 z 히스토그램 → [(z, 면적 m²)] (면적 큰 순). 여러 층이면 여러 개."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    up = n[:, 2] > 0.95 * 2 * np.maximum(area, 1e-12)
    zc = (a[:, 2] + b[:, 2] + c[:, 2])[up] / 3
    z0 = float(V[:, 2].min())
    idx = ((zc - z0) / bin_m).astype(int)
    hist = np.bincount(idx, weights=area[up])
    sm = np.convolve(hist, np.ones(3), "same")
    peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > 1.0]
    out, used = [], []
    for i in sorted(peaks, key=lambda i: -sm[i]):
        if all(abs(i - j) * bin_m > 0.3 for j in used):
            lo, hi = max(i - 1, 0), min(i + 2, len(hist))
            z = z0 + (np.average(np.arange(lo, hi), weights=hist[lo:hi] + 1e-9) + 0.5) * bin_m
            out.append((float(z), float(sm[i])))
            used.append(i)
    return out


def pick_floor(levels):
    """가장 낮은 '주요' 봉우리 (최대 면적의 25% 이상). 탁자·침대 윗면보다 바닥이 아래에 있다."""
    amax = max(a for _, a in levels)
    return min(z for z, a in levels if a >= 0.25 * amax)


# ─── 표면 샘플링 → 원본 지도 ────────────────────────────────────────────
def _samples(V, C, T, with_color=True):
    """삼각형 표면을 SAMPLE_XY(수평)·SAMPLE_Z(수직) 간격 이하로 덮는 점들을 덩어리로 내놓는다."""
    A, B, Cc = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e = np.stack([B - A, Cc - B, A - Cc], 1)
    need = np.maximum(np.linalg.norm(e[:, :, :2], axis=2).max(1) / SAMPLE_XY, np.abs(e[:, :, 2]).max(1) / SAMPLE_Z)
    n_sub = np.clip(np.ceil(need).astype(int), 1, 1500)
    del A, B, Cc, e, need
    for n in np.unique(n_sub):
        tri = T[n_sub == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        m = i + j <= n
        wa, wb = (i[m] / n).astype(np.float32), (j[m] / n).astype(np.float32)
        wc = 1 - wa - wb
        chunk = max(1, 3_000_000 // len(wa))
        for s in range(0, len(tri), chunk):
            t = tri[s:s + chunk]
            P = (V[t[:, 0]][:, None] * wa[None, :, None] + V[t[:, 1]][:, None] * wb[None, :, None]
                 + V[t[:, 2]][:, None] * wc[None, :, None]).reshape(-1, 3)
            Col = None
            if with_color:
                Col = (C[t[:, 0]][:, None].astype(np.float32) * wa[None, :, None]
                       + C[t[:, 1]][:, None].astype(np.float32) * wb[None, :, None]
                       + C[t[:, 2]][:, None].astype(np.float32) * wc[None, :, None]).reshape(-1, 3).astype(np.uint8)
            yield P, Col


def _top_per_pixel(idx, val):
    """같은 픽셀에 떨어진 점들 중 val이 가장 큰 점의 위치(선택 인덱스)."""
    order = np.lexsort((val, idx))
    idx_s = idx[order]
    return order[np.r_[idx_s[1:] != idx_s[:-1], True]]


def build_master(V, C, T, z_f, ppm=MASTER_PPM, log=print):
    """씬 좌표계 원본 지도. 스캔은 바닥이 수 cm씩 기울거나 휘어 있으므로 바닥 높이를 상수가 아니라
    위치별 높이장 h(x, y)로 추정하고, 높이 띠·바닥 판정·천장 자르기를 모두 h 기준으로 한다."""
    keep = (V[T].min(1)[:, 2] <= z_f + FLOOR_WIN + CUT_M) & (V[T].max(1)[:, 2] >= z_f - FLOOR_WIN)
    T = T[keep]
    used = np.unique(T)
    x0, y0 = V[used, 0].min() - 0.1, V[used, 1].min() - 0.1
    W = int(np.ceil((V[used, 0].max() + 0.1 - x0) * ppm))
    H = int(np.ceil((V[used, 1].max() + 0.1 - y0) * ppm))

    def pix(P):
        u = ((P[:, 0] - x0) * ppm).astype(np.int64)
        v = ((P[:, 1] - y0) * ppm).astype(np.int64)
        return v * W + u, (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # 1) 바닥 높이장: z_f ± FLOOR_WIN 안의 '위를 향한' 면에서 픽셀별 최저 높이
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    up = n[:, 2] > 0.8 * np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    zc = (a[:, 2] + b[:, 2] + c[:, 2]) / 3
    Tf = T[up & (np.abs(zc - z_f) <= FLOOR_WIN)]
    del a, b, c, n, zc
    hmin = np.full(H * W, np.inf, np.float32)
    for P, _ in _samples(V, C, Tf, with_color=False):
        idx, ok = pix(P)
        idx, z = idx[ok], P[ok, 2]
        sel = _top_per_pixel(idx, -z)
        hmin[idx[sel]] = np.minimum(hmin[idx[sel]], z[sel])
    hmin = hmin.reshape(H, W)
    k = int(round(CELL * ppm))
    Hc, Wc = -(-H // k), -(-W // k)
    pad = np.full((Hc * k, Wc * k), np.nan, np.float32)
    pad[:H, :W] = np.where(np.isfinite(hmin), hmin, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hc = np.nanmedian(pad.reshape(Hc, k, Wc, k).transpose(0, 2, 1, 3).reshape(Hc, Wc, -1), axis=2)
    known = np.isfinite(hc)
    assert known.any(), "바닥 표면을 찾지 못함 (--floor-z 확인)"
    ind = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    hc = ndimage.median_filter(hc[ind[0], ind[1]], size=7)             # 0.35 m 창
    h = ndimage.zoom(hc, k, order=1)[:H, :W].astype(np.float32)        # 원본 해상도 높이장
    floor = np.isfinite(hmin) & (np.abs(hmin - h) <= FLOOR_TOL)
    log(f"   바닥 높이장: {np.nanmin(hc[known]):.2f} ~ {np.nanmax(hc[known]):.2f} m (기준 z_f = {z_f:.2f})")

    # 2) 색·높이 띠
    hflat = h.ravel()
    zbuf = np.full(H * W, -np.inf, np.float32)
    rgb = np.zeros((H * W, 3), np.uint8)
    band = np.zeros(H * W, bool)
    total = 0
    for P, Col in _samples(V, C, T):
        idx, ok = pix(P)
        P, Col, idx = P[ok], Col[ok], idx[ok]
        rel = P[:, 2] - hflat[idx]
        m = (rel <= CUT_M) & (rel >= -0.3)
        P, Col, idx, rel = P[m], Col[m], idx[m], rel[m]
        band[idx[(rel >= BAND[0]) & (rel <= BAND[1])]] = True
        sel = _top_per_pixel(idx, P[:, 2])
        p, z = idx[sel], P[sel, 2]
        better = z > zbuf[p]
        zbuf[p[better]] = z[better]
        rgb[p[better]] = Col[sel][better]
        total += len(P)
    log(f"   표면 샘플 {total / 1e6:.1f}M → 원본 지도 {W}×{H}px")
    has = np.isfinite(zbuf).reshape(H, W)
    rgb = rgb.reshape(H, W, 3)
    hole = ~has & ndimage.binary_closing(has, iterations=2)             # 샘플 사이 1~2px 구멍
    if hole.any():
        ind = ndimage.distance_transform_edt(~has, return_distances=False, return_indices=True)
        rgb[hole] = rgb[ind[0][hole], ind[1][hole]]
        has = has | hole
    floor = ndimage.binary_closing(floor, iterations=3)
    return dict(rgb=rgb, has=has, band=band.reshape(H, W), floor=floor,
                x0=np.float64(x0), y0=np.float64(y0), ppm=np.float64(ppm), z_f=np.float64(z_f))


def scene_grid(M, cell=CELL) -> OccGrid:
    """씬 좌표계 점유 격자. 점유 = 높이 띠에 표면이 있음 or 바닥 표면이 셀의 절반 미만."""
    k = int(round(cell * float(M["ppm"])))
    H, W = M["band"].shape
    Hc, Wc = H // k, W // k
    blk = lambda a: a[:Hc * k, :Wc * k].reshape(Hc, k, Wc, k)
    occ = (blk(M["band"]).sum((1, 3)) >= 2) | (blk(M["floor"]).mean((1, 3)) < 0.5)
    return OccGrid(occ, cell, (float(M["x0"]) + cell / 2, float(M["y0"]) + cell / 2))


# ─── 로봇 좌표계로 내보내기 ─────────────────────────────────────────────
def _to_scene(xr, yr, start):
    sx, sy, yaw = start
    c, s = np.cos(yaw), np.sin(yaw)
    return sx + c * xr - s * yr, sy + s * xr + c * yr


def _lookup(M, xs, ys):
    u = np.floor((xs - float(M["x0"])) * float(M["ppm"])).astype(np.int64)
    v = np.floor((ys - float(M["y0"])) * float(M["ppm"])).astype(np.int64)
    H, W = M["has"].shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.clip(v, 0, H - 1), np.clip(u, 0, W - 1), ok


def render_robot_view(M, start):
    """→ bev (H,W,3), occ (occ_h, occ_w; 행 0 = y_min)."""
    uu, vv = np.meshgrid(np.arange(IMG_W) + 0.5, np.arange(IMG_H) + 0.5)
    xs, ys = _to_scene((uu - ROBOT_PX[0]) / PPM, (ROBOT_PX[1] - vv) / PPM, start)
    r, c, ok = _lookup(M, xs, ys)
    bev = np.where((ok & M["has"][r, c])[..., None], M["rgb"][r, c], 0).astype(np.uint8)

    x_min, y_min = -ROBOT_PX[0] / PPM, (ROBOT_PX[1] - IMG_H) / PPM
    nx, ny = int(np.ceil(IMG_W / PPM / CELL)), int(np.ceil(IMG_H / PPM / CELL))
    sub = (np.arange(5) + 0.5) / 5 * CELL
    gx = (x_min + np.arange(nx)[:, None] * CELL + sub[None]).ravel()          # nx*5
    gy = (y_min + np.arange(ny)[:, None] * CELL + sub[None]).ravel()
    X, Y = np.meshgrid(gx, gy)
    xs, ys = _to_scene(X, Y, start)
    r, c, ok = _lookup(M, xs, ys)
    band = (ok & M["band"][r, c]).reshape(ny, 5, nx, 5)
    floor = (ok & M["floor"][r, c]).reshape(ny, 5, nx, 5)
    occ = (band.sum((1, 3)) >= 2) | (floor.mean((1, 3)) < 0.5)
    return bev, occ, (x_min, y_min, nx, ny)


def export_pair(M, name, start, goal_scene, out_root, mesh_path=""):
    sx, sy, yaw = start
    dx, dy = goal_scene[0] - sx, goal_scene[1] - sy
    c, s = np.cos(yaw), np.sin(yaw)
    goal = (float(c * dx + s * dy), float(-s * dx + c * dy))
    bev, occ, (x_min, y_min, nx, ny) = render_robot_view(M, start)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    Image.fromarray(bev).save(os.path.join(out, "bev.png"))
    Image.fromarray((np.flipud(occ) * 255).astype(np.uint8)).save(os.path.join(out, "occupancy.png"))
    meta = {
        "scene_name": name, "unity_scene": "", "source": "replica", "source_mesh": mesh_path,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "img_w": IMG_W, "img_h": IMG_H, "ppm": PPM, "robot_px": list(ROBOT_PX), "cell_m": CELL,
        "x_min": x_min, "x_max": x_min + nx * CELL, "y_min": y_min, "y_max": y_min + ny * CELL,
        "occ_w": nx, "occ_h": ny, "band": list(BAND), "goal": list(goal), "camera_height": CUT_M,
        "floor_z": float(M["z_f"]), "start_scene": [float(sx), float(sy), float(np.degrees(yaw))],
        "goal_scene": [float(goal_scene[0]), float(goal_scene[1])],
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    return out, goal


# ─── 시작/목표 후보 ─────────────────────────────────────────────────────
def suggest_pairs(grid: OccGrid, n=8, d_min=5.0, d_max=8.5, clear=0.6, detour=1.12, seed=0):
    """여유 공간이 clear 이상인 점들 중, 직선 거리 d_min~d_max이고 A* 경로가 직선보다 detour배 이상 긴 쌍."""
    rng = np.random.default_rng(seed)
    dist = ndimage.distance_transform_edt(~grid.occ) * grid.res
    inf = inflate(grid, 0.441)
    lab, _ = ndimage.label(~inf.occ, structure=np.ones((3, 3)))
    step = int(round(0.4 / grid.res))
    rr, cc = np.mgrid[step // 2:grid.occ.shape[0]:step, step // 2:grid.occ.shape[1]:step]
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(sizes.argmax())                                # 가장 큰 연결 영역만 (가구 속 빈 주머니 제외)
    cand = [(r, c) for r, c in zip(rr.ravel(), cc.ravel()) if dist[r, c] >= clear and lab[r, c] == main]
    if len(cand) < 2:
        return []
    cand = np.array(cand)
    xy = np.stack([grid.origin[0] + cand[:, 1] * grid.res, grid.origin[1] + cand[:, 0] * grid.res], 1)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    ii, jj = np.where((D >= d_min) & (D <= d_max))
    same = lab[cand[ii, 0], cand[ii, 1]] == lab[cand[jj, 0], cand[jj, 1]]
    ii, jj = ii[same], jj[same]
    order = rng.permutation(len(ii))[:150]
    found = []
    for o in order:
        a, b = ii[o], jj[o]
        res = grid_search(inf.occ, tuple(cand[a]), tuple(cand[b]), heuristic=True)
        path = res[0] if isinstance(res, tuple) else res
        if path is None:
            continue
        p = np.asarray(path, float)
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * grid.res)
        if L / D[a, b] < detour:
            continue
        # yaw: 목표 방향을 90° 단위로 맞추고, 목표가 화면 안(전방·좌우)에 들어오는지 확인
        d = xy[b] - xy[a]
        yaw = np.round(np.arctan2(d[1], d[0]) / (np.pi / 2)) * (np.pi / 2)
        gx = np.cos(yaw) * d[0] + np.sin(yaw) * d[1]
        gy = -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]
        if not (0.5 <= gx <= (IMG_W - ROBOT_PX[0]) / PPM - 0.7 and abs(gy) <= ROBOT_PX[1] / PPM - 0.7):
            continue
        found.append(dict(start=[float(xy[a, 0]), float(xy[a, 1]), float(np.degrees(yaw))],
                          goal=[float(xy[b, 0]), float(xy[b, 1])], straight_m=float(D[a, b]),
                          astar_m=L, detour=L / float(D[a, b])))
    # 서로 다른 곳을 고르도록: detour 큰 순으로 보되, 이미 고른 쌍과 시작·목표가 모두 1.5 m 이내면 건너뜀
    found.sort(key=lambda p: -p["detour"])
    picked = []
    for p in found:
        s, g = np.array(p["start"][:2]), np.array(p["goal"])
        if all(np.linalg.norm(s - np.array(q["start"][:2])) > 1.5 or np.linalg.norm(g - np.array(q["goal"])) > 1.5
               for q in picked):
            picked.append(p)
        if len(picked) >= n:
            break
    return picked


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_overview(M, grid: OccGrid, pairs, path):
    """씬 전체 지도(위 = +y) + 1 m 격자 + 점유(빨강) + 후보 쌍."""
    s = 0.5                                                   # 원본 지도 → overview 축소
    view = M["rgb"][::2, ::2].astype(np.float32)
    has = M["has"][::2, ::2]
    oh, ow = view.shape[:2]
    k = CELL * float(M["ppm"]) * s                            # overview 픽셀 / 셀
    ri = np.minimum((np.arange(oh) / k).astype(int), grid.occ.shape[0] - 1)
    ci = np.minimum((np.arange(ow) / k).astype(int), grid.occ.shape[1] - 1)
    red = grid.occ[ri][:, ci] & has
    view[red] = 0.55 * view[red] + 0.45 * np.array([255, 40, 40])
    im = Image.fromarray(np.flipud(view.astype(np.uint8)))
    d = ImageDraw.Draw(im)
    ppm = float(M["ppm"]) * s
    x0, y0 = float(M["x0"]), float(M["y0"])
    px = lambda x, y: ((x - x0) * ppm, oh - (y - y0) * ppm)
    f = _font(14)
    for gx in range(int(np.ceil(x0)), int(x0 + ow / ppm) + 1):
        d.line([px(gx, y0), px(gx, y0 + oh / ppm)], fill=(255, 255, 0), width=1)
        d.text((px(gx, 0)[0] + 2, 2), f"x={gx}", fill=(255, 255, 0), font=f)
    for gy in range(int(np.ceil(y0)), int(y0 + oh / ppm) + 1):
        d.line([px(x0, gy), px(x0 + ow / ppm, gy)], fill=(255, 255, 0), width=1)
        d.text((2, px(0, gy)[1] + 2), f"y={gy}", fill=(255, 255, 0), font=f)
    f2 = _font(22)
    for k, p in enumerate(pairs):
        a, b = px(*p["start"][:2]), px(*p["goal"])
        d.line([a, b], fill=(0, 200, 255), width=2)
        d.ellipse([a[0] - 9, a[1] - 9, a[0] + 9, a[1] + 9], fill=(40, 90, 255))
        d.ellipse([b[0] - 9, b[1] - 9, b[0] + 9, b[1] + 9], fill=(30, 220, 60))
        d.text((a[0] + 10, a[1] - 26), str(k), fill=(255, 255, 255), font=f2, stroke_width=3, stroke_fill=(0, 0, 0))
    im.save(path)


# ─── CLI ────────────────────────────────────────────────────────────────
def cmd_scan(a):
    out = os.path.join(a.cache, a.name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    V, C, F = load_ply(a.mesh)
    T = triangles(F)
    print(f"📦 {a.name}: 정점 {len(V) / 1e6:.2f}M, 삼각형 {len(T) / 1e6:.2f}M, "
          f"범위 x {V[:, 0].min():.1f}~{V[:, 0].max():.1f}, y {V[:, 1].min():.1f}~{V[:, 1].max():.1f}, "
          f"z {V[:, 2].min():.1f}~{V[:, 2].max():.1f} m")
    levels = floor_levels(V, T)
    z_f = a.floor_z if a.floor_z is not None else pick_floor(levels)
    print("   수평면 높이 후보 (z, 면적): " + ", ".join(f"{z:.2f} m ({ar:.0f} m²)" for z, ar in levels[:6])
          + f"  →  바닥 z_f = {z_f:.2f} m" + ("" if a.floor_z is None else " (직접 지정)"))
    M = build_master(V, C, T, z_f)
    M["mesh"] = np.str_(os.path.abspath(a.mesh))
    np.savez_compressed(os.path.join(out, "master.npz"), **M)
    grid = scene_grid(M)
    pairs = suggest_pairs(grid, n=a.n_pairs, seed=a.seed)
    json.dump(pairs, open(os.path.join(out, "pairs.json"), "w"), indent=2)
    draw_overview(M, grid, pairs, os.path.join(out, "overview.png"))
    free = float((~grid.occ).sum() * CELL ** 2)
    print(f"   빈 바닥 {free:.1f} m², 후보 쌍 {len(pairs)}개 → {out}/overview.png  ({time.time() - t0:.0f} s)")
    for k, p in enumerate(pairs):
        print(f"   [{k}] 직선 {p['straight_m']:.1f} m, A* {p['astar_m']:.1f} m (×{p['detour']:.2f})")


def cmd_export(a):
    cdir = os.path.join(a.cache, a.name)
    M = dict(np.load(os.path.join(cdir, "master.npz")))
    jobs = []
    if a.start:
        assert a.goal, "--start를 쓰면 --goal도 필요"
        jobs.append((a.tag or "manual", (a.start[0], a.start[1], np.radians(a.start[2])), tuple(a.goal)))
    else:
        pairs = json.load(open(os.path.join(cdir, "pairs.json")))
        for k in (a.pairs if a.pairs is not None else range(len(pairs))):
            p = pairs[k]
            jobs.append((f"p{k}", (p["start"][0], p["start"][1], np.radians(p["start"][2])), tuple(p["goal"])))
    for tag, start, goal in jobs:
        name = f"replica_{a.name}_{tag}"
        out, g = export_pair(M, name, start, goal, a.out, str(M.get("mesh", "")))
        print(f"✅ {name}: goal(로봇 좌표)=({g[0]:.2f}, {g[1]:.2f}) → {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--mesh", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--floor-z", type=float, default=None, help="바닥 높이 직접 지정 (여러 층인 씬)")
    s.add_argument("--n-pairs", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cache", default=CACHE_DIR)
    s.set_defaults(fn=cmd_scan)
    e = sub.add_parser("export")
    e.add_argument("--name", required=True)
    e.add_argument("--pairs", type=int, nargs="*", default=None, help="pairs.json의 번호 (생략 = 전부)")
    e.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"))
    e.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    e.add_argument("--tag", default=None)
    e.add_argument("--out", default="Go2Export")
    e.add_argument("--cache", default=CACHE_DIR)
    e.set_defaults(fn=cmd_export)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
