#!/usr/bin/env bash
# ============================================================
# backup.sh — 서버 데이터/모델/설정 백업 (일 1회 크론 권장)
# 사용: sudo bash scripts/backup.sh
# 크론: 30 4 * * * /bin/bash /root/workspace/edge-project/scripts/backup.sh
# ============================================================
set -uo pipefail
SRC="/root/workspace"
DST="/root/backups"
KEEP=7                                   # 보관일수
mkdir -p "$DST"
TS=$(date +%Y%m%d_%H%M)
NAME="edge-backup_$TS.tar.gz"
tar -czf "$DST/$NAME" \
  --exclude='*.pt' --exclude='*.onnx' --exclude='*.tflite' \
  --exclude='.hf-cache' \
  -C "$SRC" edge-project 2>/dev/null
ls -1t "$DST"/edge-backup_*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "백업 완료: $DST/$NAME ($(du -h "$DST/$NAME" | cut -f1)), 보관 ${KEEP}일"
