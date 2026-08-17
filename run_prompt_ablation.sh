#!/usr/bin/env bash
#
# run_prompt_ablation.sh
# ─────────────────────────────────────────────────────────────────────────
# Task #5/#6: Scene A/D/E x 프롬프트 정보량 3단계(rich/medium/minimal) ablation.
# gemini-2.5-flash 단일 모델로 통일해서 9개 조합을 순차 실행한다.
#   1) main_court.py (VLM 판결)          -> data/<scene>/<variant>/last_judged_path.json
#   2) dial_core.py  (DIAL-MPC 정식 실행) -> data/<scene>/<variant>/*_states.npy
# 한 조합이 실패해도 나머지는 계속 진행한다 (run_overnight_pipeline.sh와 동일한 패턴).
#
# 사용법:
#   ./run_prompt_ablation.sh                 # 포그라운드 실행
#   nohup ./run_prompt_ablation.sh &          # 백그라운드로 돌려놓고 나가기
#
# 주의: 각 씬의 n_steps(examples/<scene>.yaml)는 그 씬에서 "보통" 나오는 경로 길이
# 기준으로 맞춰둔 값이라, 프롬프트 레벨에 따라 예상외로 긴 우회 경로가 나오면
# 시뮬레이션이 완주 전에 끊길 수 있다 (Scene D 우회 경로에서 실제로 겪었던 문제).
# 완주 여부는 실행 후 analysis/load_rollout.py로 qpos 길이를 total_time_steps 계산과
# 대조해서 확인할 것.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SCENES=(oracle_scene_A oracle_scene_D oracle_scene_E)
LEVELS=(rich medium minimal)
GEMINI_MODEL="gemini-2.5-flash"

DIAL_TIMEOUT=180     # dial-mpc가 뜨고 나서 죽이기까지 대기 시간(초)
DIAL_KILL_AFTER=15   # SIGINT로 안 죽으면 그만큼 더 기다린 후 SIGKILL

LOG_DIR="$REPO_ROOT/logs/ablation_$(date +%Y%m%d_%H%M%S)"
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

safe_model_tag="$(echo "$GEMINI_MODEL" | tr ':.-' '___')"

log "🧪 프롬프트 ablation 시작 — 씬 ${SCENES[*]} x 레벨 ${LEVELS[*]} (모델: ${GEMINI_MODEL})"
log "로그 디렉토리: $LOG_DIR"

for scene in "${SCENES[@]}"; do
    oracle_png="$REPO_ROOT/data/$scene/oracle.png"
    if [[ ! -f "$oracle_png" ]]; then
        log "⚠️  $scene: oracle.png 없음 (oracle_gen.py 먼저 실행 필요) → 씬 전체 스킵"
        continue
    fi

    for level in "${LEVELS[@]}"; do
        variant="gemini_${safe_model_tag}_ablation_${level}"
        combo="${scene}_${level}"
        log "=============================================="
        log "🧪 ${scene} / ${level} (variant=${variant})"

        if ! run_step "1_court_${combo}" python vlm_courtroom/main_court.py \
                --scene "$scene" --backend gemini --gemini-model "$GEMINI_MODEL" --prompt-level "$level"; then
            log "  → 판결 스텝 자체가 비정상 종료, 이 조합 스킵"
            continue
        fi

        json_path="$REPO_ROOT/data/$scene/$variant/last_judged_path.json"
        if [[ ! -f "$json_path" ]]; then
            log "  ⚠️ $json_path 생성 안 됨 (내부적으로 파싱 실패했을 가능성), 이 조합 스킵"
            continue
        fi

        out_dir="$REPO_ROOT/data/$scene/$variant"
        run_dial_and_verify "2_dial_${combo}" "$scene" "$out_dir" \
                --vlm-path-json "$json_path" \
                --output-dir "data/$scene/$variant" \
            || log "  (DIAL-MPC 실패해도 courtroom 판결 자체는 이미 확보됨 — last_judged_path.json 참고)"
    done
done

log "=============================================="
log "🏁 전체 ablation 종료. 로그 전체: $LOG_DIR"
log "결과 위치: data/<scene>/gemini_${safe_model_tag}_ablation_<rich|medium|minimal>/"
log "  last_judged_path.json (판결 성공 여부) / *_states.npy (물리 실행 결과, analysis/run_experiment_c.py로 집계 가능)"
