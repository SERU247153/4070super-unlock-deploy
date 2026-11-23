#!/usr/bin/env bash
# ============================================================
# export_tflite.sh — YOLO → TFLite INT8 변환 (일회성 격리 컨테이너)
# ⚠️ 상태: 보류 — torch 2.4 × onnx-ir × ml_dtypes 버전 매트릭스 충돌
#    (2026-11-24 확인: 최신 onnx-ir은 tf 2.16의 ml_dtypes와 충돌,
#     onnx-ir 0.1.0은 torch 2.4 익스포터와 충돌)
#    해결 방향: tensorflow/tensorflow 공식 이미지 기반으로 재구성 예정
#    ※ Jetson Nano 주경로는 ONNX→TensorRT 이므로 무관
# ============================================================
set -euo pipefail
WT="${1:-yolov8n.pt}"
SZ="${2:-640}"
OUTDIR="/root/workspace/edge-project/models"
mkdir -p "$OUTDIR"

docker run --rm \
  -e WT="$WT" \
  -v /root/workspace/edge-project:/work \
  -w /work \
  python:3.10-slim bash -c '
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null 2>&1
    pip install -q "ultralytics==8.3.180" "tensorflow-cpu==2.16.2" "torch" "torchvision" "opencv-python-headless" "onnxscript==0.1.0" "onnx-ir==0.1.0" 2>/dev/null
    python -c "
from ultralytics import YOLO
import os, shutil
wt = os.environ[\"WT\"]
path = \"/work/\" + wt if os.path.exists(\"/work/\" + wt) else wt
m = YOLO(path)
r = m.export(format=\"tflite\", imgsz='"$SZ"', int8=True, nms=False)
shutil.copy(r, \"/work/models/\")
print(\"TFLite INT8 변환 완료:\", os.path.basename(r), round(os.path.getsize(r)/1e6, 1), \"MB\")
"'
