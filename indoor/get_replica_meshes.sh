#!/usr/bin/env bash
# Replica Dataset에서 씬별 mesh.ply만 받아 온다 (전체 약 34 GB를 스트리밍으로 읽고 mesh만 디스크에 남김).
#   ./indoor/get_replica_meshes.sh /data/replica
# 출처/라이선스: https://github.com/facebookresearch/Replica-Dataset (비상업 연구용, LICENSE 참조)
set -euo pipefail
DEST=${1:?저장할 폴더를 지정하세요}
mkdir -p "$DEST"
BASE=https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta
for p in a b c d e f g h i j k l m n o p q; do
  curl -sL --retry 5 "${BASE}${p}"
done | tar -xz -C "$DEST" --wildcards '*/mesh.ply'
ls -la "$DEST"/*/mesh.ply
