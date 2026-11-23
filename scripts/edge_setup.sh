#!/usr/bin/env bash
# ============================================================
# edge_setup.sh — 엣지 AI 프로젝트 환경 셋업 (4070S 서버, jupyter_ai 컨테이너)
# 재현용: 컨테이너 재생성 시 이 스크립트만 실행하면 동일 환경 복원
# 실행: sudo bash scripts/edge_setup.sh   (호스트에서!)
# ============================================================
# ⚠️ 버전 고정 원칙 (2026-11-24 실전 사고 교훈):
#   · torch/torchvision/flash-attn/transformer-engine = NGC 번들 그대로
#     (pip가 torch를 업그레이드하면 flash-attn ABI 파손 — 확인됨)
#   · 전 패키지를 2024년대(torch 2.4 호환) 버전으로 고정
#   · ultralytics==8.3.180 : TFLite 클래식(TF) 경로 (8.4.x는 LiteRT/jax 전환)
#   · numpy>=1.26,<2 : RAPIDS(cudf/cupy) + soxr 호환 유지
# ============================================================
set -euo pipefail
CONT=jupyter_ai

# 0) 사전 확인: NGC 번들 torch 보호 대상
docker exec $CONT python -c 'import torch; assert torch.__version__.startswith("2.4"), torch.__version__; print("torch", torch.__version__, "확인")'

# 1) 프로젝트 디렉토리 (호스트 /root/workspace는 컨테이너 /workspace로 마운트됨)
mkdir -p /root/workspace/edge-project/{data,models,notebooks,scripts}
mkdir -p /root/workspace/.hf-cache

# 2) 패키지 설치 — 전부 버전 고정
#    ※ tensorflow/TFLite 변환은 메인 컨테이너에 넣지 않는다 (tf-keras 충돌 확인)
#      → scripts/export_tflite.sh 의 일회성 변환 컨테이너로 분리
docker exec $CONT pip install -q \
  "ultralytics==8.3.180" \
  "transformers==4.44.2" \
  "accelerate==0.34.2" \
  "datasets==2.21.0" \
  "bitsandbytes==0.43.3" \
  "peft==0.13.2" \
  "numpy>=1.26,<2" \
  "opencv-python-headless==4.10.0.84"

# 2-1) torch 보호 검증 (업그레이드 시 즉시 중단)
docker exec $CONT python -c 'import torch; assert torch.__version__.startswith("2.4"), f"torch가 {torch.__version__}로 변경됨! 핀 점검 필요"'

# 3) 성능 자동 주입 (모든 노트북에 적용)
docker exec $CONT mkdir -p /root/.ipython/profile_default/startup
docker exec $CONT sh -c 'cat > /root/.ipython/profile_default/startup/10-gpu-perf.py <<EOF
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
EOF'

# 4) 모델 사전 다운로드 (스모크 테스트 겸 캐시)
docker exec $CONT python -c '
from ultralytics import YOLO
YOLO("yolov8n.pt")
from transformers import Dinov2Model
Dinov2Model.from_pretrained("facebook/dinov2-small", cache_dir="/workspace/.hf-cache")
print("모델 준비 완료")'

# 4-1) TFLite 변환은 별도 변환 컨테이너에서 (scripts/export_tflite.sh)
#     — 메인 환경의 numpy/torch를 건드리지 않는 격리 방식

# 5) 최종 검증
docker exec $CONT python -c '
import torch, numpy, flash_attn, cv2, ultralytics, transformers
print("torch", torch.__version__)
print("flash-attn", flash_attn.__version__)
print("numpy", numpy.__version__, "| opencv", cv2.__version__)
print("ultralytics", ultralytics.__version__, "| transformers", transformers.__version__)
print("OK")'

echo "엣지 프로젝트 환경 셋업 완료"
