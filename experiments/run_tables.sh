#!/usr/bin/env bash
#
# run_tables.sh ─ 표 ①(Agent ablation)·표 ②(타 방법 비교) 전체 실행 (연구실 서버)
# ─────────────────────────────────────────────────────────────────────────
#   단계 0) Unity 내보내기 → 실험 씬 생성       indoor/build_scene.py
#   단계 1) VLM 구성 C1, C1xk, C2, C3, C4       vlm_courtroom/run_ablation.py   (Gemini API)
#   단계 2) 고전 베이스라인                      baselines/run_table2_baselines.py (CPU, 수 초)
#   단계 3) DIAL-MPC 실행                       experiments/run_dial.py          (GPU, 가장 오래 걸림)
#   단계 4) 지표 계산 → 표                      analysis/evaluate_all.py, make_tables.py
#
# 각 단계는 이미 끝난 시행을 건너뛰므로, 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 진행된다.
#
# 사용법
#   SCENES="indoor_000 indoor_001 indoor_002" TRIALS=5 nohup ./experiments/run_tables.sh &
#   STAGES="1 2" ./experiments/run_tables.sh        # 일부 단계만
#
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCENES=${SCENES:?"SCENES 환경변수에 씬 이름을 공백으로 나열 (예: SCENES=\"indoor_000 indoor_001\")"}
TRIALS=${TRIALS:-5}
GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-pro}
IMAGE_MODE=${IMAGE_MODE:-coord}          # coord(기존 시스템) | all(모든 agent가 이미지)
WORKERS=${WORKERS:-2}
STAGES=${STAGES:-"0 1 2 3 4"}
EXPORT_ROOT=${EXPORT_ROOT:-Go2Export}

LOG_DIR="$REPO_ROOT/logs/tables_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/summary.log"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

log "씬: $SCENES | 시행: $TRIALS | 모델: $GEMINI_MODEL | image-mode: $IMAGE_MODE | 단계: $STAGES"

if has 0; then
  log "▶ 0) 실험 씬 생성"
  exports=(); for s in $SCENES; do exports+=("$EXPORT_ROOT/$s"); done
  python indoor/build_scene.py "${exports[@]}" 2>&1 | tee -a "$LOG_DIR/0_build.log"
fi

if has 1; then
  log "▶ 1) VLM 구성 (C1 C1xk C2 C3 C4)"
  python vlm_courtroom/run_ablation.py --scenes $SCENES --trials "$TRIALS" \
      --gemini-model "$GEMINI_MODEL" --image-mode "$IMAGE_MODE" --workers "$WORKERS" \
      2>&1 | tee -a "$LOG_DIR/1_vlm.log" | grep -E "^\[|🧪"
fi

if has 2; then
  log "▶ 2) 고전 베이스라인"
  python baselines/run_table2_baselines.py --scenes $SCENES --trials "$TRIALS" 2>&1 | tee -a "$LOG_DIR/2_baselines.log"
fi

if has 3; then
  log "▶ 3) DIAL-MPC"
  python experiments/run_dial.py --scenes $SCENES 2>&1 | tee -a "$LOG_DIR/3_dial.log"
fi

if has 4; then
  log "▶ 4) 지표 → 표"
  python analysis/evaluate_all.py --scenes $SCENES --out results 2>&1 | tee -a "$LOG_DIR/4_eval.log"
  python analysis/make_tables.py --results results 2>&1 | tee -a "$LOG_DIR/4_eval.log"
fi

log "🏁 끝. 표: results/table1_ablation.tex, results/table2_comparison.tex (미리보기 .md)"
