#!/usr/bin/env bash
# 4.5절: VLM 백본 비교 (Replica 평가 쌍, 전체 법정 C4).
#   Gemini 2.5 Pro 결과는 본 실험(data/<scene>/C4)을 그대로 쓰고, 여기서는 나머지 백본만 돌린다.
#   결과 폴더: data/<scene>/C4_<tag>/trial_xx   → results/table3_backbones.md
#
#   ./experiments/run_backbones.sh                       # 계획만 (기본)
#   EXECUTE=1 ./experiments/run_backbones.sh             # DIAL-MPC 실행까지
#   SCENES="..." TRIALS=3 ./experiments/run_backbones.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/dial_mpc:$REPO_ROOT:${PYTHONPATH:-}"
EXPORT_ROOT=${EXPORT_ROOT:-Go2Export}
if [ -z "${SCENES:-}" ]; then
  SCENES=$(for d in "$EXPORT_ROOT"/*/; do [ -f "$d/meta.json" ] && basename "$d"; done | tr '\n' ' ')
fi
TRIALS=${TRIALS:-3}
EXECUTE=${EXECUTE:-0}
IMAGE_MODE=${IMAGE_MODE:-all}
# 태그=백엔드:모델
BACKBONES=${BACKBONES:-"flash=gemini:gemini-2.5-flash qwen25vl=ollama:qwen2.5vl:7b llavallama3=ollama:llava-llama3"}
LOG_DIR="logs/backbones_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
echo "씬: $SCENES | 시행: $TRIALS | 백본: $BACKBONES | 실행: $EXECUTE"

methods=""
for spec in $BACKBONES; do
  tag=${spec%%=*}; rest=${spec#*=}; backend=${rest%%:*}; model=${rest#*:}
  echo "▶ C4_$tag  ($backend / $model)"
  if [ "$backend" = gemini ]; then
    python vlm_courtroom/run_ablation.py --scenes $SCENES --configs C4 --trials "$TRIALS" --backend gemini \
      --gemini-model "$model" --image-mode "$IMAGE_MODE" --suffix "_$tag" 2>&1 | tee -a "$LOG_DIR/vlm.log" | grep -E "^\[|🧪"
  else
    python vlm_courtroom/run_ablation.py --scenes $SCENES --configs C4 --trials "$TRIALS" --backend ollama \
      --ollama-model "$model" --image-mode "$IMAGE_MODE" --suffix "_$tag" 2>&1 | tee -a "$LOG_DIR/vlm.log" | grep -E "^\[|🧪"
  fi
  methods="$methods C4_$tag"
done
if [ "$EXECUTE" = 1 ]; then
  python experiments/run_dial.py --scenes $SCENES --methods $methods 2>&1 | tee -a "$LOG_DIR/dial.log"
fi
python analysis/evaluate_all.py --scenes $SCENES --out results 2>&1 | tail -2
python analysis/make_tables.py --results results 2>&1 | sed -n '/table3_backbones/,$p'
