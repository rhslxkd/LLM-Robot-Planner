"""
make_tables.py ─ results/trials.csv, groups.csv → 표 ①·② (LaTeX booktabs + 미리보기 Markdown)

집계 규칙 (논문 본문/캡션에 그대로 적을 것)
  - 비율 지표(%)   : 제외되지 않은 모든 시행을 합쳐서 계산 (pooled over scenes × trials)
  - 연속 지표       : 시행 평균 ± 표준편차. 도달 시간·추종 오차는 '실행된 시행' 중 해당 값이 있는 것만
  - homotopy class : (씬, 방법)마다 유효 경로의 class 수 → 씬 평균 ± 표준편차
  - 계획 실패(경로 없음)는 SR 계산에서 실패로 센다

  python analysis/make_tables.py --results results/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TABLE1_ROWS = [("C1", "Single agent"), ("C1xk", "Single agent $\\times k$ (medoid)"),
               ("C2", "+ Prosecutor"), ("C3", "+ Prosecutor + Defense"), ("C4", "+ Judge (full)"),
               # 이미지 접근 ablation: 주 실험(모든 agent가 이미지)과 같은 구성을 Coordinate만 이미지를 보게 해서 다시 실행
               ("C2_coord", "+ Prosecutor (image: Coordinate only)"),
               ("C3_coord", "+ Prosecutor + Defense (image: Coordinate only)"),
               ("C4_coord", "+ Judge (image: Coordinate only)")]
TABLE2_ROWS = [("astar_gt", "A* (GT map)", "GT map", "--"),
               ("astar_seg", "A* (segmented)", "Segmentation", "--"),
               ("dijkstra_gt", "Dijkstra (GT map)", "GT map", "--"),
               ("astarraw_gt", "A* (GT map, no smoothing)", "GT map", "--"),
               ("rrtstar_gt", "RRT* (GT map)", "GT map", "--"),
               ("rrtstar_seg", "RRT* (segmented)", "Segmentation", "--"),
               ("llmastar_gt", "LLM-A* (GT map)", "GT map", "Text map + 5-shot"),
               ("llmastar_seg", "LLM-A* (segmented)", "Segmentation", "Text map + 5-shot"),
               ("C4", "\\textbf{Ours (Courtroom)}", "None", "Start/goal only")]


def _pct(s: pd.Series) -> str:
    s = s.dropna().astype(float)
    return f"{100 * s.mean():.1f}" if len(s) else "--"


def _ms(s: pd.Series, fmt: str = "{:.2f}") -> str:
    s = s.dropna().astype(float)
    if not len(s):
        return "--"
    return (fmt + " $\\pm$ " + fmt).format(s.mean(), s.std(ddof=0)) if len(s) > 1 else fmt.format(s.mean())


def summarize(trials: pd.DataFrame, groups: pd.DataFrame, method: str) -> dict:
    t = trials[(trials.method == method) & (~trials.excluded.astype(bool))]
    g = groups[groups.method == method]
    reached = t.get("exec_reached")
    out = {
        "n": len(t),
        "output_ok": _pct(t["plan_output_ok"]),
        "valid": _pct(t["plan_valid"]),
        "bev_free": _pct(~t["plan_bev_collision"].astype(bool)),
        "homotopy": _ms(g["n_homotopy"], "{:.1f}") if len(g) else "--",
        "calls": _ms(t["n_calls"], "{:.0f}") if t["n_calls"].notna().any() else "--",
        "plan_time": _ms(t["plan_time_s"], "{:.1f}"),
        "sr": _pct(t["exec_success"].fillna(False).astype(bool)),
        "fall": _pct(t.loc[t.executed.astype(bool), "exec_fell"]) if "exec_fell" in t else "--",
        "collision": _pct(t.loc[t.executed.astype(bool), "exec_collided"]) if "exec_collided" in t else "--",
        "time": _ms(t.loc[reached.fillna(False).astype(bool), "exec_time_to_goal_s"], "{:.1f}") if reached is not None else "--",
        "track": _ms(t.loc[t.executed.astype(bool), "exec_track_mean_m"]) if "exec_track_mean_m" in t else "--",
        "tilt": _ms(t.loc[t.executed.astype(bool), "exec_tilt_max_deg"], "{:.1f}") if "exec_tilt_max_deg" in t else "--",
    }
    return out


def table1(trials, groups) -> tuple[str, str]:
    cols = [("output_ok", "Output OK (\\%) $\\uparrow$"), ("bev_free", "BEV coll.-free (\\%) $\\uparrow$"),
            ("homotopy", "Homotopy $\\uparrow$"), ("calls", "\\#Calls"),
            ("sr", "SR (\\%) $\\uparrow$"), ("fall", "Fall (\\%) $\\downarrow$"),
            ("collision", "Coll. (\\%) $\\downarrow$"), ("track", "Track err. (m) $\\downarrow$")]
    body, md = [], ["| Config | n | " + " | ".join(c[1].replace("\\%", "%").replace("$\\uparrow$", "↑")
                                                    .replace("$\\downarrow$", "↓").replace("\\#", "#") for c in cols) + " |",
                    "|" + "---|" * (len(cols) + 2)]
    for key, label in TABLE1_ROWS:
        if key not in set(trials.method):
            continue
        s = summarize(trials, groups, key)
        body.append(f"{label} & " + " & ".join(s[c] for c, _ in cols) + " \\\\")
        md.append(f"| {key} | {s['n']} | " + " | ".join(s[c].replace(' $\\pm$ ', ' ± ') for c, _ in cols) + " |")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Agent ablation. Rates are pooled over all scenes and trials; "
           "continuous values are mean $\\pm$ std over trials; homotopy is the number of distinct valid path classes "
           "per scene (mean $\\pm$ std).}\n\\label{tab:ablation}\n\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n"
           "Configuration & " + " & ".join(c[1] for c in cols) + " \\\\\n\\midrule\n" + "\n".join(body) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, "\n".join(md)


def table2(trials, groups) -> tuple[str, str]:
    cols = [("valid", "Valid path (\\%) $\\uparrow$"), ("homotopy", "Homotopy $\\uparrow$"),
            ("sr", "SR (\\%) $\\uparrow$"), ("fall", "Fall (\\%) $\\downarrow$"),
            ("collision", "Coll. (\\%) $\\downarrow$"), ("time", "Time (s) $\\downarrow$"),
            ("plan_time", "Plan time (s)")]
    body, md = [], ["| Method | Map needed | Prompt | n | " + " | ".join(
        c[1].replace("\\%", "%").replace("$\\uparrow$", "↑").replace("$\\downarrow$", "↓") for c in cols) + " |",
        "|" + "---|" * (len(cols) + 4)]
    for key, label, mapsrc, prompt in TABLE2_ROWS:
        if key not in set(trials.method):
            continue
        s = summarize(trials, groups, key)
        body.append(f"{label} & {mapsrc} & {prompt} & " + " & ".join(s[c] for c, _ in cols) + " \\\\")
        md.append(f"| {key} | {mapsrc} | {prompt} | {s['n']} | " +
                  " | ".join(s[c].replace(' $\\pm$ ', ' ± ') for c, _ in cols) + " |")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Comparison with classical and LLM-based planners on indoor BEV scenes. "
           "Classical planners and LLM-A* require an occupancy map (ground truth or segmented from the BEV image); "
           "LLM-A* additionally requires the map as text barriers with five in-context examples. "
           "Ours takes the BEV image with start and goal only.}\n\\label{tab:comparison}\n"
           "\\begin{tabular}{lll" + "c" * len(cols) + "}\n\\toprule\n"
           "Method & Map & Prompt & " + " & ".join(c[1] for c in cols) + " \\\\\n\\midrule\n" + "\n".join(body) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, "\n".join(md)


TABLE3_ROWS = [("C4", "Gemini 2.5 Pro"), ("C4_flash", "Gemini 2.5 Flash"),
               ("C4_qwen25vl", "Qwen2.5-VL-7B (local)"), ("C4_llavallama3", "LLaVA-Llama3-8B (local)")]


def table3(trials, groups) -> tuple[str, str]:
    cols = [("output_ok", "Output OK (\\%) $\\uparrow$"), ("valid", "Valid path (\\%) $\\uparrow$"),
            ("homotopy", "Homotopy $\\uparrow$"), ("sr", "SR (\\%) $\\uparrow$"), ("plan_time", "Plan time (s)")]
    body, md = [], ["| Backbone | n | " + " | ".join(c[1].replace("\\%", "%").replace("$\\uparrow$", "↑") for c in cols) + " |",
                    "|" + "---|" * (len(cols) + 2)]
    for key, label in TABLE3_ROWS:
        if key not in set(trials.method):
            continue
        s = summarize(trials, groups, key)
        body.append(f"{label} & " + " & ".join(s[c] for c, _ in cols) + " \\\\")
        md.append(f"| {label} | {s['n']} | " + " | ".join(s[c].replace(' $\\pm$ ', ' ± ') for c, _ in cols) + " |")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Full courtroom (C4) with different VLM backbones on the same "
           "evaluation pairs.}\n\\label{tab:backbones}\n\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n"
           "Backbone & " + " & ".join(c[1] for c in cols) + " \\\\\n\\midrule\n" + "\n".join(body) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return tex, "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    trials = pd.read_csv(os.path.join(a.results, "trials.csv"))
    groups = pd.read_csv(os.path.join(a.results, "groups.csv"))
    for name, fn in (("table1_ablation", table1), ("table2_comparison", table2), ("table3_backbones", table3)):
        tex, md = fn(trials, groups)
        open(os.path.join(a.results, f"{name}.tex"), "w").write(tex)
        open(os.path.join(a.results, f"{name}.md"), "w").write(md + "\n")
        print(f"\n### {name}\n{md}")


if __name__ == "__main__":
    main()
