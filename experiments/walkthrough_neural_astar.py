"""
walkthrough_neural_astar.py ─ Replica 씬 하나를 현수의 원본 Neural A* 파이프라인(legacy/neural_astar/)으로 단계별 처리.
각 단계의 중간 결과를 그림·파일로 남겨서 눈으로 따라갈 수 있게 한다. (neural-astar 환경에서 실행)

  conda activate neural-astar
  python experiments/walkthrough_neural_astar.py --scene replica_apartment_0_p0
  → data/<scene>/neuralastar_legacy/trial_00/last_judged_path.json   (DIAL-MPC 입력, 다른 방법과 같은 형식)
    data/<scene>/neuralastar_legacy/walkthrough/0*_*.png              (단계별 그림)

단계
  1  입력       : data/<scene>/bev_raw.png (위에서 본 Replica 이미지) + occupancy.npz (정답 지도, 0.05 m 칸)
  2  정답 이미지 : 정답 지도의 장애물 칸을 이미지 해상도로 옮겨 빨강(255,0,0)으로 칠함 = 원본 코드가 읽는 oracle 형식
  3  32×32 축소 : 칸 안에 빨강이 하나라도 있으면 장애물 (max-pool)              ← run_neural_astar_step.py 그대로
  4  Neural A*  : 32×32 위에서 경로 탐색                                           ← 그대로 (체크포인트 선택 가능)
  5  확대·단순화 : 칸 중심 → 원래 픽셀, line-of-sight 단순화, 0.65 m 보폭 개수로 보간, 보폭 0.4~1.0 m 강제  ← 그대로
  6  안전마진   : 거리장(EDT) 기반 보정 (push 0.4 m / 통과 0.3 m / soft 0.6 m)      ← waypoint_generator.py 그대로
  7  저장·검사  : last_judged_path.json + 정답 지도 BEV 충돌 검사(evaluate_path)
DIAL-MPC 실행은 vlm_court 환경에서: python experiments/run_dial.py --scenes <scene> --methods neuralastar_legacy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "legacy", "neural_astar"))
from analysis.plan_metrics import OccGrid, evaluate_path  # noqa: E402

import run_neural_astar_step as na_step  # noqa: E402  (현수 원본: GRID, 단순화·보간 함수)
import waypoint_generator as wg  # noqa: E402            (현수 원본: 안전마진 보정)

DEFAULT_CKPT = os.path.join(REPO, "neural-astar", "model", "mazes_032_moore_c8")


def say(step, msg):
    print(f"\n[{step}] {msg}", flush=True)


def draw_path(img: Image.Image, pts_px, color, start_px, goal_px, r=5, width=4):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    if len(pts_px) > 1:
        d.line([tuple(p) for p in pts_px], fill=color, width=width)
    for x, y in pts_px:
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    for (x, y), c in ((start_px, (40, 90, 255)), (goal_px, (30, 200, 60))):
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=c, outline="white", width=3)
    return im


def load_ckpt(ckpt_dir):
    """neural_astar 패키지의 load_from_ptl_checkpoint와 같은 규칙. pytorch_lightning 없이도 동작."""
    import glob
    import re
    import torch
    f = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True))[-1]
    try:
        raw = torch.load(f, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(f, map_location="cpu")
    return {re.split("planner.", k)[-1]: v for k, v in raw["state_dict"].items() if "planner" in k}, f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="Neural A* 체크포인트 폴더 (파인튜닝 모델: finetune/checkpoints/combined 등)")
    ap.add_argument("--method", default="neuralastar_legacy", help="출력 폴더 이름 (data/<scene>/<method>/trial_00)")
    a = ap.parse_args()

    sd_dir = os.path.join(REPO, "data", a.scene)
    out = os.path.join(sd_dir, a.method, "trial_00")
    wdir = os.path.join(sd_dir, a.method, "walkthrough")
    os.makedirs(out, exist_ok=True)
    os.makedirs(wdir, exist_ok=True)
    t0 = time.perf_counter()

    # ── 1. 입력 ──────────────────────────────────────────────────────────────
    sm = json.load(open(os.path.join(sd_dir, "scene_meta.json")))
    grid = OccGrid.load(os.path.join(sd_dir, "occupancy.npz"))
    bev = Image.open(os.path.join(sd_dir, "bev_raw.png")).convert("RGB")
    W, H = bev.size
    ppm, (rx, ry) = float(sm["ppm"]), sm["robot_px"]
    gx, gy = sm["goal"]
    start_px, goal_px = (rx, ry), (rx + gx * ppm, ry - gy * ppm)
    say(1, f"입력: {a.scene}  이미지 {W}×{H} px, {ppm:.0f} px/m (= {W / ppm:.1f} m × {H / ppm:.1f} m)\n"
           f"     정답 지도 {grid.occ.shape[1]}×{grid.occ.shape[0]} 칸 × {grid.res} m, 장애물 칸 {int(grid.occ.sum())}개\n"
           f"     출발 (0, 0) = 픽셀 {start_px}, 목표 ({gx:.2f}, {gy:.2f}) m = 픽셀 ({goal_px[0]:.0f}, {goal_px[1]:.0f})")
    bev.save(os.path.join(wdir, "01_input_bev.png"))

    # ── 2. 정답 이미지 (빨강 = 장애물) ───────────────────────────────────────
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    xy = np.stack([(uu - rx) / ppm, (ry - vv) / ppm], -1)              # 픽셀 → 로봇 좌표 (m)
    red_mask = grid.occupied(xy)                                       # 지도 밖 = 장애물
    oracle = np.asarray(bev).copy()
    oracle[red_mask] = (255, 0, 0)
    oracle_path = os.path.join(wdir, "02_oracle_red.png")
    Image.fromarray(oracle).save(oracle_path)
    say(2, f"정답 이미지: 정답 지도의 장애물을 빨강으로 칠함 → {os.path.relpath(oracle_path, REPO)}\n"
           f"     빨간 픽셀 {red_mask.mean() * 100:.1f}%  (원본 코드의 판정식 r>180, g<120, b<120으로 다시 읽힘)")

    # ── 3. 32×32 축소 (max-pool) ─────────────────────────────────────────────
    G = na_step.GRID
    ch, cw = H / G, W / G
    obs = np.zeros((G, G), np.float32)
    for j in range(G):
        for i in range(G):
            obs[j, i] = 1.0 if red_mask[int(j * ch):int((j + 1) * ch), int(i * cw):int((i + 1) * cw)].any() else 0.0
    to_g = lambda p: (min(max(int(round(p[0] * G / W)), 0), G - 1), min(max(int(round(p[1] * G / H)), 0), G - 1))
    sg, gg = to_g(start_px), to_g(goal_px)
    say(3, f"{G}×{G} 축소: 칸 하나 = {cw / ppm:.2f} m × {ch / ppm:.2f} m, 장애물 칸 {int(obs.sum())}/{G * G}\n"
           f"     출발 칸 {sg} {'빈칸' if obs[sg[1], sg[0]] == 0 else '⚠️ 장애물'}, 목표 칸 {gg} "
           f"{'빈칸' if obs[gg[1], gg[0]] == 0 else '⚠️ 장애물'}")
    for j in range(G):
        print("     " + "".join("S" if (i, j) == sg else "G" if (i, j) == gg else "#" if obs[j, i] else "."
                                for i in range(G)))
    Image.fromarray(((1 - obs) * 255).astype(np.uint8)).resize((G * 12, G * 12), Image.NEAREST).save(
        os.path.join(wdir, "03_grid32.png"))

    # ── 4. Neural A* ─────────────────────────────────────────────────────────
    import torch
    sys.path.insert(0, os.environ.get("NEURAL_ASTAR_SRC", os.path.join(REPO, "neural-astar", "src")))
    from neural_astar.planner import NeuralAstar
    sd, ckpt_file = load_ckpt(a.ckpt)
    model = NeuralAstar(g_ratio=0.5, encoder_arch="CNN")
    model.load_state_dict(sd)
    model.eval()
    smap = np.zeros((G, G), np.float32); smap[sg[1], sg[0]] = 1
    gmap = np.zeros((G, G), np.float32); gmap[gg[1], gg[0]] = 1
    with torch.no_grad():
        o = model(torch.tensor(1 - obs)[None, None], torch.tensor(smap)[None, None], torch.tensor(gmap)[None, None])
    mask = o.paths[0, 0].numpy() > 0.5
    wp_grid = na_step.extract_ordered_path(mask, sg, gg)
    ok = bool(wp_grid) and tuple(wp_grid[-1]) == tuple(gg)
    say(4, f"Neural A*: {os.path.relpath(ckpt_file, REPO) if ckpt_file.startswith(REPO) else ckpt_file}\n"
           f"     탐색한 칸 {int((o.histories[0, 0].numpy() > 0.5).sum())}개, 경로 칸 {len(wp_grid)}개, 목표 도달 {ok}")
    vis = np.stack([(1 - obs) * 255] * 3, -1).astype(np.uint8)
    vis[o.histories[0, 0].numpy() > 0.5] = (180, 200, 255)
    vis[mask] = (255, 140, 0)
    Image.fromarray(vis).resize((G * 12, G * 12), Image.NEAREST).save(os.path.join(wdir, "04_neural_astar_grid.png"))
    if not ok:
        print("❌ 목표에 도달하지 못함 (32×32에서 길이 막혔을 수 있음: 03_grid32.png 확인)")
        sys.exit(1)

    # ── 5. 확대 · 단순화 · 개수 맞추기 ──────────────────────────────────────
    from scipy.ndimage import binary_dilation
    red_dil = binary_dilation(red_mask, iterations=na_step.WALL_DILATE_PX)
    wp_full = [na_step.grid_to_full(x, y, W, H) for x, y in wp_grid]
    wp_simp = na_step.simplify_line_of_sight(wp_full, red_dil)
    L = na_step.path_length_m(wp_simp, ppm)
    target_n = max(na_step.MIN_WAYPOINTS, min(na_step.MAX_WAYPOINTS, round(L / na_step.TARGET_STEP_M)))
    wp_up = na_step.enforce_step_constraints(na_step.upsample_to_target(wp_simp, target_n), red_dil, ppm)
    say(5, f"확대·단순화: 원시 {len(wp_full)}점 → 단순화 {len(wp_simp)}점 → 보간 {len(wp_up)}점 "
           f"(길이 {na_step.path_length_m(wp_up, ppm):.2f} m, 0.65 m 보폭 기준 {target_n}개)")
    draw_path(Image.fromarray(oracle), wp_full, (255, 140, 0), start_px, goal_px, r=3, width=2).save(
        os.path.join(wdir, "05a_upscaled_raw.png"))
    draw_path(bev, wp_up, (255, 140, 0), start_px, goal_px).save(os.path.join(wdir, "05b_simplified.png"))
    coords = [{"x": round((px - rx) / ppm, 2), "y": round((ry - py) / ppm, 2)} for px, py in wp_up]

    # ── 6. 안전마진 보정 (waypoint_generator) ────────────────────────────────
    final, passed, log = wg.generate_waypoints(oracle_path, coords, (rx, ry), ppm)
    say(6, f"안전마진 보정: 통과 {passed}, 반복 로그 {len(log)}줄 (log.txt)")
    for line in log[-5:]:
        print("     " + line)
    open(os.path.join(wdir, "06_safety_log.txt"), "w").write("\n".join(log) + "\n")
    fin_px = [(rx + c["x"] * ppm, ry - c["y"] * ppm) for c in final]
    draw_path(bev, fin_px, (40, 210, 70), start_px, goal_px).save(os.path.join(wdir, "06_final.png"))

    # ── 7. 저장 · 정답 지도 검사 ─────────────────────────────────────────────
    path = [{"x": float(c["x"]), "y": float(c["y"])} for c in final]
    if np.hypot(path[0]["x"], path[0]["y"]) < 1e-6:
        path = path[1:]                                                # 다른 방법과 같이 출발점 (0,0)은 빼고 저장
    json.dump(path, open(os.path.join(out, "last_judged_path.json"), "w"), indent=2)
    ev = evaluate_path(path, grid, sm["goal"])
    meta = {"scene": a.scene, "config": a.method, "trial": 0, "algorithm": "neuralastar_legacy", "map_source": "gt",
            "parse_ok": True, "n_points": len(path), "plan_time_s": round(time.perf_counter() - t0, 3), "n_calls": 0,
            "api_error": False, "ckpt": ckpt_file, "grid": G, "safety_passed": bool(passed)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, ensure_ascii=False)
    say(7, f"저장: {os.path.relpath(out, REPO)}/last_judged_path.json ({len(path)}점)\n"
           f"     정답 지도 검사: 유효 {ev['valid']}  (목표 도달 {ev['reaches_goal']}, BEV 충돌 {ev['bev_collision']}, "
           f"최소 여유 {ev.get('min_clearance_m', float('nan')):.2f} m, 길이 {ev['path_len_m']:.2f} m)")
    print(f"\n그림: {os.path.relpath(wdir, REPO)}/  (01 입력 → 02 정답 이미지 → 03 32×32 → 04 Neural A* → 05 확대·단순화 → 06 최종)")
    print(f"다음 (vlm_court 환경): python experiments/run_dial.py --scenes {a.scene} --methods {a.method}")


if __name__ == "__main__":
    main()
