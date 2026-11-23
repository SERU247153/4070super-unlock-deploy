#!/usr/bin/env bash
# ============================================================
# export_tflite.sh — YOLO → TFLite INT8 변환 (일회성 격리 컨테이너)
# 메인 jupyter 환경을 건드리지 않는다 (torch/numpy 충돌 방지)
# 사용: sudo bash scripts/export_tflite.sh <weights.pt> [이미지크기]
# ============================================================
set -euo pipefail
WT="${1:-yolov8n.pt}"
SZ="${2:-640}"
OUTDIR="/root/workspace/edge-project/models"
mkdir -p "$OUTDIR"

docker run --rm \
  -v /root/workspace/edge-project:/work \
  -w /work \
  python:3.10-slim bash -c '
    pip install -q "ultralytics==8.3.180" "tensorflow-cpu==2.16.2" "torch" "torchvision" "opencv-python-headless" 2>/dev/null
    python -c "
from ultralytics import YOLO
import os, shutil
m = YOLO(\"/work/$WT\" if os.path.exists(\"/work/$WT\") else \"$WT\")
r = m.export(format=\"tflite\", imgsz='"$SZ"', int8=True, nms=False)
shutil.copy(r, \"/work/models/\")
print(\"TFLite INT8 변환 완료:\", os.path.basename(r), round(os.path.getsize(r)/1e6, 1), \"MB\")
"'
