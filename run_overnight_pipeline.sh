#!/usr/bin/env bash
#
# run_overnight_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────
# 씬 A/D/E × 모델 gemini/qwen/llava/llama 전체 조합에 대해
#   1) main_court.py       (VLM 판결)        -> data/<scene>/<variant>/last_judged_path.json
#   2) dial_core.py (순수)  (DIAL-MPC 정식 실행) -> data/<scene>/<variant>/*_states.npy  (Table 4/5/6 원본)
#   3) draw_path_markers.py (경로 시각화)     -> <scene>_<variant>_viz.xml / .yaml
#   4) dial_core.py (viz)  (DIAL-MPC 시각화 실행) -> data/<scene>/<variant>/viz/ (눈으로 확인용, 집계 대상 아님)
# 순서로 자동 실행한다. 한 조합이 실패해도 나머지 조합은 계속 진행한다.
#
# 사용법:
#   ./run_overnight_pipeline.sh                 # 포그라운드 실행
#   nohup ./run_overnight_pipeline.sh &          # 퇴근 전 백그라운드로 돌려놓고 나가기
#
set -uo pipefail  # 개별 스텝 실패로 전체가 멈추면 안 되므로 -e는 쓰지 않는다

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SCENES=(oracle_scene_A oracle_scene_D oracle_scene_E)   # oracle_gen.py로 oracle.png 이미 찍혀있는 씬들
MODELS=(gemini qwen llava llama)

# model 이름 -> "backend|ollama_model|variant"  (variant는 main_court.py/courtroom.py 규칙과 동일하게 맞춤)
declare -A MODEL_SPEC=(
  [gemini]="gemini||gemini"
  [qwen]="ollama|qwen2.5vl:7b|ollama_qwen2_5vl_7b"
  [llava]="ollama|llava:13b|ollama_llava_13b"
  [llama]="ollama|llama3.2-vision:11b|ollama_llama3_2-vision_11b"
)

# DIAL-MPC 순수 실행은 씬별 정식 YAML(examples/<scene>.yaml, draw_path_markers.py가
# viz 버전을 만들 때 쓰는 바로 그 원본)을 그대로 쓴다. 즉 --example "$scene" 자체.
# (예전엔 범용 "unitree_go2_trot"로 고정해뒀었는데, 이러면 씬별 장애물 XML이 아니라
#  기본 씬으로 돌아갈 위험이 있어서 제거함.)

# 실측(Gemini+SceneA) 기준 rollout만 38초 걸렸음 (JIT 컴파일 시간 별도).
# 30초는 states.npy 저장 전에 죽일 위험이 커서, 넉넉하게 늘림.
DIAL_TIMEOUT=180     # dial-mpc가 뜨고 나서 죽이기까지 대기 시간(초)
DIAL_KILL_AFTER=15   # SIGINT로 안 죽으면 그만큼 더 기다린 후 SIGKILL

LOG_DIR="$REPO_ROOT/logs/overnight_$(date +%Y%m%d_%H%M%S)"
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
# (main_court.py처럼 내부 예외를 삼키고 exit 0을 낼 수 있는 스크립트들이 있어서,
#  종료 코드만으론 "진짜 성공"을 확신할 수 없음 -> 산출물 존재 여부로 다시 한번 검증)
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

log "🌙 야간 파이프라인 시작 — 씬 ${SCENES[*]} × 모델 ${MODELS[*]}"
log "로그 디렉토리: $LOG_DIR"

for scene in "${SCENES[@]}"; do
    oracle_png="$REPO_ROOT/data/$scene/oracle.png"
    if [[ ! -f "$oracle_png" ]]; then
        log "⚠️  $scene: oracle.png 없음 (oracle_gen.py 먼저 실행 필요) → 씬 전체 스킵"
        continue
    fi

    for model in "${MODELS[@]}"; do
        IFS='|' read -r backend ollama_model variant <<< "${MODEL_SPEC[$model]}"
        combo="${scene}_${variant}"
        log "=============================================="
        log "🧪 ${scene} / ${model} (variant=${variant})"

        # 1) VLM 판결
        if [[ "$backend" == "gemini" ]]; then
            court_cmd=(python vlm_courtroom/main_court.py --scene "$scene" --backend gemini)
        else
            court_cmd=(python vlm_courtroom/main_court.py --scene "$scene" --backend ollama --ollama-model "$ollama_model")
        fi
        if ! run_step "1_court_${combo}" "${court_cmd[@]}"; then
            log "  → 판결 스텝 자체가 비정상 종료, 이 조합 스킵"
            continue
        fi

        json_path="$REPO_ROOT/data/$scene/$variant/last_judged_path.json"
        if [[ ! -f "$json_path" ]]; then
            log "  ⚠️ $json_path 생성 안 됨 (내부적으로 파싱 실패했을 가능성), 이 조합 스킵"
            continue
        fi

        # 2) DIAL-MPC 순수 실행 (씬별 정식 YAML: examples/<scene>.yaml) — 이게 Table 4/5/6에 들어가는 "진짜" 결과
        official_out_dir="$REPO_ROOT/data/$scene/$variant"
        if ! run_dial_and_verify "2_dial_official_${combo}" "$scene" "$official_out_dir" \
                --vlm-path-json "$json_path" \
                --output-dir "data/$scene/$variant"; then
            log "  → 정식 DIAL-MPC 실행 실패, 이 조합의 시각화 단계는 건너뜀"
            continue
        fi

        # 3) 경로 시각화 (XML 마커 + viz용 YAML 생성) — 이후 결과는 집계 대상 아님, 육안 확인용
        if ! run_step "3_draw_${combo}" python draw_path_markers.py "$scene" "$variant"; then
            log "  → 시각화 XML/YAML 생성 실패, viz 실행 스킵 (정식 결과는 이미 확보됨)"
            continue
        fi

        # 4) DIAL-MPC 시각화 실행 (마커 박힌 씬). vlm_path_json을 여기서도 명시적으로 다시 넘겨준다 —
        #    draw_path_markers.py가 만든 YAML은 scene_xml/output_dir만 바꾸고
        #    vlm_path_json은 건드리지 않으므로, 명시하지 않으면 엉뚱한 기본 경로를 탈 수 있다.
        viz_example="${scene}_${variant}_viz"
        viz_out_dir="$REPO_ROOT/data/$scene/$variant/viz"
        run_dial_and_verify "4_dial_viz_${combo}" "$viz_example" "$viz_out_dir" \
            --vlm-path-json "$json_path" \
            --output-dir "data/$scene/$variant/viz" \
            || log "  (viz 실행은 육안 확인용이라 실패해도 정식 결과엔 영향 없음)"
    done
done

log "=============================================="
log "🏁 전체 파이프라인 종료. 로그 전체: $LOG_DIR"
log "Table 4/5/6 집계용 결과 위치: data/<scene>/<variant>/*_states.npy"
log "육안 확인용(마커 포함) 결과 위치: data/<scene>/<variant>/viz/"