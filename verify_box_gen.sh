#!/bin/bash
set -e
echo "=== 0) 캐시 정리 ==="
rm -rf __pycache__ generate_random_baffle_maze.cpython*.pyc 2>/dev/null || true
find . -maxdepth 1 -name "*.pyc" -delete 2>/dev/null || true

echo "=== 1) 시그니처 확인 ==="
python3 -c "
from generate_random_baffle_maze import generate_boxes
import inspect
sig = inspect.signature(generate_boxes)
print(sig)
assert sig.parameters['goal_x_range'].default == (2.5, 4.4), '시그니처가 패치 전 상태!'
assert sig.parameters['room_margin'].default == 0.7, '시그니처가 패치 전 상태!'
print('시그니처 OK')
"

echo "=== 2) XML 생성 ==="
mkdir -p dial_mpc/dial_mpc/models/unitree_go2
python3 -c "
from generate_random_baffle_maze import generate_boxes, CAMERA_VISIBLE_X_MAX_M
import os
os.makedirs('dial_mpc/dial_mpc/models/unitree_go2', exist_ok=True)
bad = []
for seed in range(4):
    name = f'oracle_scene_Btest{seed}'
    xml, meta = generate_boxes(seed, name)
    room_x_max = meta['goal_x'] + 0.7
    ok = room_x_max <= CAMERA_VISIBLE_X_MAX_M
    print(name, meta['n_boxes'], 'goal=', round(meta['goal_x'],2), round(meta['goal_y'],2),
          'room_x_max=', round(room_x_max,2), 'OK' if ok else '!!!NG!!!')
    if not ok:
        bad.append(name)
    with open(f'dial_mpc/dial_mpc/models/unitree_go2/{name}.xml', 'w') as f:
        f.write(xml)
if bad:
    raise SystemExit(f'화면 범위 초과 씬: {bad}')
print('전체 4개 씬 room_x_max 카메라 범위 내 확인됨')
"

echo "=== 3) 렌더링 ==="
for i in 0 1 2 3; do
  MUJOCO_GL=egl python3 oracle_gen.py oracle_scene_Btest${i}.xml > /tmp/render_${i}.log 2>&1
  cat /tmp/render_${i}.log
done

echo "=== 4) 최종 검증 ==="
if grep -l "화면밖" /tmp/render_0.log /tmp/render_1.log /tmp/render_2.log /tmp/render_3.log 2>/dev/null; then
  echo "FAIL: 위 파일에 화면밖 잘림 있음"
  exit 1
else
  echo "PASS: 4개 씬 전부 화면밖 잘림 없음"
fi
