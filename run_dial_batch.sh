#!/bin/bash
mkdir -p dial_batch_logs

SCENES="R002 R003 R004 R006 R008 R009 R010 R011 R015"

run_and_kill() {
  local s=$1
  local scene="oracle_scene_${s}"
  local outdir="data/${scene}/batch"
  local before
  before=$(ls "$outdir"/*_states.npy 2>/dev/null | wc -l)

  python3 -u dial_mpc/dial_mpc/core/dial_core.py \
    --example "$scene" \
    --vlm-path-json "$outdir/last_judged_path.json" \
    --output-dir "$outdir" \
    --n-steps 1000 \
    > "dial_batch_logs/${scene}.log" 2>&1 &
  local pid=$!
  echo "[$scene] PID=$pid 시작 $(date +%H:%M:%S)"

  local waited=0
  while [ "$waited" -lt 900 ]; do
    local after
    after=$(ls "$outdir"/*_states.npy 2>/dev/null | wc -l)
    if [ "$after" -gt "$before" ]; then
      sleep 3
      echo "[$scene] states.npy 생성 확인 -> 서버 종료 $(date +%H:%M:%S)"
      kill -INT "$pid" 2>/dev/null
      sleep 2
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$scene] ⚠️ states.npy 없이 프로세스 먼저 종료됨 (로그 확인: dial_batch_logs/${scene}.log)"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done

  echo "[$scene] ⚠️ 15분 타임아웃 -> 강제 종료"
  kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  return 1
}

for s in $SCENES; do
  run_and_kill "$s"
done

echo "=== 전체 배치 완료 $(date +%H:%M:%S) ==="
