#!/usr/bin/env bash
#
# run_backend_comparison_llava3.sh
# llava:13b가 JSON 포맷 준수율이 너무 낮아서(6개 중 2개만 성공) llava-llama3로 교체 재실행.
# Scene A/D/E x prompt-level {minimal, medium} = 6개 조합. qwen은 이미 결과 있어서 스킵.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SCENES=(oracle_scene_A oracle_scene_D oracle_scene_E)
LEVELS=(minimal medium)
declare -A MODELS=( ["llava3"]="llava-llama3" )

DIAL_TIMEOUT=180
DIAL_KILL_AFTER=15

LOG_DIR="$REPO_ROOT/logs/backend_comparison_llava3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

run_step() {
    local tag="$1"; shift
    if "$@" >> "$LOG_DIR/${tag}.log" 2>&1; then
        log "  ✅ ${tag}"
        return 0
    else
        log "  ❌ ${tag} 실패 (로그: ${tag}.log)"
        return 1
    fi
}

run_dial_and_verify() {
    local tag="$1" example="$2" out_dir="$3"; shift 3
    log "  ▶ DIAL-MPC 실행: --example ${example} (최대 ${DIAL_TIMEOUT}s + kill-after ${DIAL_KILL_AFTER}s)"
    timeout --signal=INT --kill-after="${DIAL_KILL_AFTER}s" "${DIAL_TIMEOUT}s" \
        python dial_mpc/dial_mpc/core/dial_core.py --example "$example" "$@" \
        >> "$LOG_DIR/${tag}.log" 2>&1
    local rc=$?

    if compgen -G "${out_dir}/*_states.npy" > /dev/null; then
        log "  ✅ ${tag} — states.npy 확인됨 (rc=${rc})"
        return 0
    else
        log "  ⚠️ ${tag} — states.npy 없음 (rc=${rc}), 타임아웃 전에 저장 못했을 가능성. 로그: ${tag}.log"
        return 1
    fi
}

log "🔍 ollama 모델 사전 점검"
for model_key in "${!MODELS[@]}"; do
    model_tag="${MODELS[$model_key]}"
    if ollama list 2>/dev/null | grep -q "^${model_tag}"; then
        log "  ✅ ${model_tag} 확인됨"
    else
        log "  ⚠️ ${model_tag} 이(가) 'ollama list'에 없음 — 스킵됩니다"
    fi
done

log "🧪 llava-llama3 재실행 시작 — 씬 ${SCENES[*]} x 레벨 ${LEVELS[*]}"
log "로그 디렉토리: $LOG_DIR"

for model_key in "${!MODELS[@]}"; do
    model_tag="${MODELS[$model_key]}"
    safe_model_tag="$(echo "$model_tag" | tr ':.' '__')"

    if ! ollama list 2>/dev/null | grep -q "^${model_tag}"; then
        log "⚠️  ${model_tag} 없음 — ${model_key} 전체 스킵"
        continue
    fi

    for scene in "${SCENES[@]}"; do
        oracle_png="$REPO_ROOT/data/$scene/oracle.png"
        if [[ ! -f "$oracle_png" ]]; then
            log "⚠️  $scene: oracle.png 없음 → 씬 전체 스킵"
            continue
        fi

        for level in "${LEVELS[@]}"; do
            variant="ollama_${safe_model_tag}_ablation_${level}"
            combo="${model_key}_${scene}_${level}"
            log "=============================================="
            log "🧪 ${model_key}(${model_tag}) / ${scene} / ${level} (variant=${variant})"

            if ! run_step "1_court_${combo}" python vlm_courtroom/main_court.py \
                    --scene "$scene" --backend ollama --ollama-model "$model_tag" --prompt-level "$level"; then
                log "  → 판결 스텝 자체가 비정상 종료, 이 조합 스킵"
                continue
            fi

            json_path="$REPO_ROOT/data/$scene/$variant/last_judged_path.json"
            if [[ ! -f "$json_path" ]]; then
                log "  ⚠️ $json_path 생성 안 됨 (JSON 파싱 실패), 이 조합 스킵"
                continue
            fi

            out_dir="$REPO_ROOT/data/$scene/$variant"
            run_dial_and_verify "2_dial_${combo}" "$scene" "$out_dir" \
                    --vlm-path-json "$json_path" \
                    --output-dir "data/$scene/$variant" \
                || log "  (DIAL-MPC 실패해도 courtroom 판결 자체는 이미 확보됨)"
        done
    done
done

log "=============================================="
log "🏁 llava-llama3 재실행 종료. 로그 전체: $LOG_DIR"
log "결과 위치: data/<scene>/ollama_llava-llama3_ablation_<minimal|medium>/"
