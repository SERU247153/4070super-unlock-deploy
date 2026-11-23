#!/usr/bin/env bash
# ============================================================
# 15_mps.sh — CUDA MPS (여러 명이 동시에 GPU 를 쓸 때 커널을 겹쳐 실행)
# ⚠️ 선택 항목. 기본 흐름(10/20)에 포함되지 않습니다.
#   - 컨테이너가 MPS 에 붙으려면 컨테이너를 다음 옵션으로 '재생성'해야 합니다:
#       -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host
#   - Ollama 등 기존 GPU 프로세스와의 동작은 환경마다 다를 수 있으니 실습 시간에만 켜는 것을 권장
# 사용법: sudo bash optimize/15_mps.sh enable | disable | status
# ============================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need_root
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
case "${1:-status}" in
  enable)
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if pgrep -x nvidia-cuda-mps-control >/dev/null; then ok "이미 실행 중"; exit 0; fi
    nvidia-cuda-mps-control -d
    sleep 1; pgrep -x nvidia-cuda-mps-control >/dev/null && ok "MPS 데몬 시작" || bad "MPS 시작 실패 ($CUDA_MPS_LOG_DIRECTORY 확인)"
    echo "  컨테이너 연결: docker run ... -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host ..."
    ;;
  disable)
    if pgrep -x nvidia-cuda-mps-control >/dev/null; then echo quit | nvidia-cuda-mps-control; ok "MPS 종료"; else ok "실행 중 아님"; fi
    ;;
  status)
    pgrep -x nvidia-cuda-mps-control >/dev/null && ok "MPS 실행 중" || echo "  MPS 꺼짐"
    ;;
  *) echo "사용법: $0 enable|disable|status"; exit 1;;
esac
