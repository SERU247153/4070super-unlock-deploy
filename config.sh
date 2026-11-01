#!/usr/bin/env bash
# ============================================================
# gpu-unlock 배포 공통 설정 (Rocky Linux 9 기준)
# 서버 사정에 맞게 이 파일만 수정하면 됩니다.
# 이 파일은 각 스크립트가 source 로 읽어들이므로 SC2034 경고는 무시합니다.
# shellcheck disable=SC2034
# ============================================================

# ---- 설치 ----
GIT_URL="https://github.com/Zagitta/gpu-unlock"
WORKDIR="/opt/gpu-unlock"          # 소스 클론 위치 (root 권한으로 빌드)

# ---- 커널 부트 옵션 ----
# all              : 지원되는 모든 미바운드 카드 언록
# "0000:01:00.0"   : 특정 카드만 (lspci -D 로 확인, 콤마 구분 가능)
GPU_UNLOCK_TARGETS="all"
# S3 절전 복귀 후 자동 재언록 (서버면 1 유지 권장, 0 이면 비활성)
GPU_UNLOCK_RESUME=1

# ---- 튜닝 ----
# max   : VBIOS가 허용하는 최대 전력까지 상향
# 242   : 특정 W 값 직접 지정
POWER_LIMIT_MODE="max"
# off              : 클럭 고정 안 함
# "2100,2730"      : "min MHz,max MHz" (nvidia-smi -lgc 형식)
CLOCK_LOCK="off"
# NVreg_EnableStreamMemOPs=1 등록 여부 (일부 CUDA/LLM 워크로드 +몇 %)
ENABLE_STREAM_MEM_OPS=1

# ---- 벤치마크 ----
# nvidia-container-toolkit 필요. 이미지가 없으면 최초 1회 pull 합니다.
BENCH_IMAGE="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
# 벤치마크 실행 위치:
#   ""            : BENCH_IMAGE 로 새 컨테이너 실행 (pull 필요)
#   "jupyter_ai"  : 기존 컨테이너 재사용 (NGC PyTorch + FlashAttention 내장, pull 불필요)
BENCH_CONTAINER="jupyter_ai"
BENCH_MATRIX_SIZE=8192   # 8192^3 GEMM, 12GB VRAM 기준 안전
BENCH_ITERS=7            # 측정 반복 횟수 (중앙값 사용)

# ---- 벤치마크 캘리브레이션 ----
# 클럭 고정: 전/후 실행 조건을 동일하게 만드는 핵심 (IEEE TPDS 방법론 [7])
#   "off"         : 미고정 (부스트 변동 허용)
#   "2505,2505"   : min,max MHz 고정 (본 서버 카드 실측 부스트 2760 MHz 이내)
BENCH_LOCK_CLOCKS="2505,2505"
# 실행 중 전력 샘플링 (1초 간격, 평균/최대를 결과 JSON에 기록)
BENCH_POWER_SAMPLE=1
