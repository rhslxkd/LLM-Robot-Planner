#!/usr/bin/env bash
#
# run_backend_comparison.sh
# ─────────────────────────────────────────────────────────────────────────
# Task #7: 오픈소스 VLM 백엔드 비교 (Qwen2.5-VL-7B, LLaVA-13B).
# Scene A/D/E x prompt-level {minimal, medium} = 12개 조합.
# rich는 스킵 (Task #5/#6에서 gemini 기준 가장 나쁜 결과였어서 우선순위 낮음).
# Gemini는 Task #5/#6 ablation에서 이미 minimal/medium 데이터 확보했으므로 재사용,
# 여기서는 안 돌림.
#
# 목적: Task #5/#6의 "minimal이 최선"이라는 결론이 gemini-2.5-flash(강한 frontier
# 모델) 하나에서만 나온 거라, 소형 오픈소스 모델에도 그대로 적용되는지 검증.
# 소형 모델은 공간추론이 약해서 오히려 medium(장애물 설명 포함)이 더 나을 수도 있음
# -- 이 자체가 "prompt richness 효과가 backbone 능력에 의존하는가"라는 새로운 결과.
#
#   1) main_court.py (VLM 판결)          -> data/<scene>/<variant>/last_judged_path.json
#   2) dial_core.py  (DIAL-MPC 정식 실행) -> data/<scene>/<variant>/*_states.npy
# 한 조합이 실패해도 나머지는 계속 진행 (run_prompt_ablation.sh와 동일 패턴).
#
# 사용법:
#   ./run_backend_comparison.sh                 # 포그라운드 실행
#   nohup ./run_backend_comparison.sh &          # 백그라운드로 돌려놓고 나가기
#
# 사전조건: `ollama list`에 qwen2.5vl:7b, llava:13b 둘 다 떠 있어야 함.
# 스크립트 시작할 때 자동으로 확인하고, 없는 모델은 관련 조합 전부 스킵하고 계속 진행함
# (스크립트가 중간에 죽지 않음 -- 로그에 스킵 사유 남음).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SCENES=(oracle_scene_A oracle_scene_D oracle_scene_E)
LEVELS=(minimal medium)
declare -A MODELS=( ["qwen"]="qwen2.5vl:7b" ["llava"]="llava:13b" )

DIAL_TIMEOUT=180     # dial-mpc가 뜨고 나서 죽이기까지 대기 시간(초)
DIAL_KILL_AFTER=15   # SIGINT로 안 죽으면 그만큼 더 기다린 후 SIGKILL

LOG_DIR="$REPO_ROOT/logs/backend_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

# 커맨드를 실행하고 stdout/stderr를 전용 로그파일로 보낸다. 실패해도 스크립트는 계속 진행.
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

# dial_core.py를 timeout으로 감싸서 실행하고, states.npy가 실제로 생겼는지까지 확인한다.
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
        log "  ⚠️ ${model_tag} 이(가) 'ollama list'에 없음 — 이 모델 관련 조합은 전부 스킵됩니다 (필요시 'ollama pull ${model_tag}' 먼저 실행)"
    fi
done

log "🧪 Task #7 백엔드 비교 시작 — 씬 ${SCENES[*]} x 레벨 ${LEVELS[*]} x 모델 ${MODELS[*]}"
log "로그 디렉토리: $LOG_DIR"

for model_key in "${!MODELS[@]}"; do
    model_tag="${MODELS[$model_key]}"
    # main_court.py와 동일한 방식으로 변환 (":" 와 "." 을 "_" 로): qwen2.5vl:7b -> qwen2_5vl_7b
    safe_model_tag="$(echo "$model_tag" | tr ':.' '__')"

    if ! ollama list 2>/dev/null | grep -q "^${model_tag}"; then
        log "⚠️  ${model_tag} 없음 — ${model_key} 전체 스킵"
        continue
    fi

    for scene in "${SCENES[@]}"; do
        oracle_png="$REPO_ROOT/data/$scene/oracle.png"
        if [[ ! -f "$oracle_png" ]]; then
            log "⚠️  $scene: oracle.png 없음 (oracle_gen.py 먼저 실행 필요) → 씬 전체 스킵"
            continue
        fi

        for level in "${LEVELS[@]}"; do
            variant="ollama_${safe_model_tag}_ablation_${level}"
            combo="${model_key}_${scene}_${level}"
            log "=============================================="
            log "🧪 ${model_key}(${model_tag}) / ${scene} / ${level} (variant=${variant})"

            # 1) VLM 판결
            if ! run_step "1_court_${combo}" python vlm_courtroom/main_court.py \
                    --scene "$scene" --backend ollama --ollama-model "$model_tag" --prompt-level "$level"; then
                log "  → 판결 스텝 자체가 비정상 종료, 이 조합 스킵"
                continue
            fi

            json_path="$REPO_ROOT/data/$scene/$variant/last_judged_path.json"
            if [[ ! -f "$json_path" ]]; then
                log "  ⚠️ $json_path 생성 안 됨 (내부적으로 파싱 실패했을 가능성 -- 소형 모델은 JSON 포맷 실패가 그 자체로 유의미한 데이터), 이 조합 스킵"
                continue
            fi

            # 2) DIAL-MPC 정식 실행 (씬별 기본 YAML: examples/<scene>.yaml)
            out_dir="$REPO_ROOT/data/$scene/$variant"
            run_dial_and_verify "2_dial_${combo}" "$scene" "$out_dir" \
                    --vlm-path-json "$json_path" \
                    --output-dir "data/$scene/$variant" \
                || log "  (DIAL-MPC 실패해도 courtroom 판결 자체는 이미 확보됨 — last_judged_path.json 참고)"
        done
    done
done

log "=============================================="
log "🏁 전체 백엔드 비교 종료. 로그 전체: $LOG_DIR"
log "결과 위치: data/<scene>/ollama_<model>_ablation_<minimal|medium>/"
log "  last_judged_path.json (판결 성공 여부, 특히 JSON 파싱 실패율 자체가 지표) / *_states.npy (analysis/run_experiment_c.py로 집계)"
