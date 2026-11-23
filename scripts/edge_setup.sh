#!/usr/bin/env bash
# ============================================================
# edge_setup.sh — 엣지 AI 프로젝트 환경 셋업 (4070S 서버, jupyter_ai 컨테이너)
# 재현용: 컨테이너 재생성 시 이 스크립트만 실행하면 동일 환경 복원
# 실행: sudo bash scripts/edge_setup.sh
# ============================================================
set -euo pipefail
CONT=jupyter_ai

# 1) 프로젝트 디렉토리 (호스트 /root/workspace는 컨테이너 /workspace로 마운트됨)
mkdir -p /root/workspace/edge-project/{data,models,notebooks,scripts}
mkdir -p /root/workspace/.hf-cache

# 2) 패키지 설치 (transformers는 NGC 24.07/torch 2.4 호환 버전으로 고정,
#    numpy<2 는 컨테이너 내 soxr 등 C확장 ABI 호환용)
docker exec $CONT pip install -q \
  ultralytics "transformers==4.44.2" accelerate datasets bitsandbytes "numpy<2"

# 3) 성능 자동 주입 (모든 노트북에 적용)
docker exec $CONT mkdir -p /root/.ipython/profile_default/startup
docker exec $CONT sh -c 'cat > /root/.ipython/profile_default/startup/10-gpu-perf.py <<EOF
import os, torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")
EOF'

# 4) 모델 사전 다운로드 (선택 — 스모크 테스트 겸 캐시)
docker exec $CONT python -c '
from ultralytics import YOLO
YOLO("yolov8n.pt")
from transformers import Dinov2Model
Dinov2Model.from_pretrained("facebook/dinov2-small", cache_dir="/workspace/.hf-cache")
print("모델 준비 완료")'

# 5) 검증
docker exec $CONT python -c '
import ultralytics, transformers, torch
print("ultralytics", ultralytics.__version__)
print("transformers", transformers.__version__)
print("OK")'

echo "엣지 프로젝트 환경 셋업 완료"
