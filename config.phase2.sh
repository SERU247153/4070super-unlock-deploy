#!/usr/bin/env bash
# ============================================================
# 2단계(서버 최적화·실습·서빙) 공통 설정
#   config.sh(1단계: 언록) 다음에 _common.sh 가 자동으로 읽어들입니다.
#   서버 사정에 맞게 이 파일만 수정하면 됩니다.
#   ⚠️ 공개 저장소: 서버 IP·토큰·비밀번호는 절대 여기에 적지 마세요.
#      (HF 토큰은 실행 시 환경변수 HF_TOKEN 으로만 전달)
# shellcheck disable=SC2034
# ============================================================

# ---- 대상 컨테이너 ----
JUPYTER_CONTAINER="jupyter_ai"          # 실습용 주피터 컨테이너 (NGC PyTorch)
LAB_NETWORK="gpu-lab"                   # 주피터 ↔ 서빙 컨테이너 내부 통신용 도커 네트워크
OPENWEBUI_CONTAINER="open-webui"        # 있으면 LAB_NETWORK 에 함께 연결 (vLLM 을 OpenAI 연결로 사용 가능)

# ---- [1] HF 캐시 볼륨 / [2] 이미지 커밋 ----
HF_VOLUME="hf_cache"                    # 주피터·vLLM·SGLang·TRT-LLM 이 공유 → 모델 1회만 다운로드
HF_MOUNT="/root/.cache/huggingface"     # 컨테이너 내부 마운트 위치 (HF_HOME 으로도 지정)
COMMIT_REPO="jupyter_ai-opt"            # docker commit 결과 이미지 이름 (태그 = 날짜)
RECREATE_ADD_SYS_ADMIN=1                # 1: --cap-add SYS_ADMIN (Nsight Compute 카운터를 재부팅 없이 사용)
RECREATE_SHM_SIZE="16g"                 # DataLoader num_workers 용 공유메모리 (기존 값이 더 크면 유지)

# ---- [6~9] 컨테이너 라이브러리 ----
# 기존 NGC 패키지(torch·flash-attn·TE·numpy·jupyterlab 등)는 constraints 로 고정하고
# 아래 패키지만 "새로 추가/업그레이드 허용" — NGC 최적화 빌드를 깨지 않기 위함
LAB_PIP_PACKAGES="transformers accelerate peft trl bitsandbytes datasets openai sentencepiece \
protobuf evaluate onnx onnxruntime-gpu==1.19.2 jupyterlab-nvdashboard"
INSTALL_DALI=1                          # NVIDIA DALI (NGC 에 이미 있으면 건너뜀)
INSTALL_DEEPSPEED=1                     # ZeRO-Offload 실습용 (ops 는 JIT 빌드)
JUPYTER_VLLM_KERNEL=1                   # 1: 주피터 안에 'vLLM (venv)' 전용 커널 생성 (NGC torch 와 격리)
JUPYTER_VLLM_VERSION="0.31.0"           # venv 커널용 vLLM 버전
JUPYTER_SGLANG_KERNEL=0                 # 1: 'SGLang (venv)' 커널도 생성 (디스크 ~10GB 추가)
JUPYTER_SGLANG_VERSION="0.5.22"
LAB_NOTEBOOK_DIR=""                     # 템플릿 노트북 복사 위치 (빈 값 = 주피터 작업 디렉터리 자동 탐지)/gpu-lab-templates

# ---- [3] 전력 캡 (부팅 시 자동) ----
POWER_CAP_W=180                         # 캘리브레이션 결과 180W 에서 이미 최대 성능 (results/calibration)

# ---- 호스트: Ollama ----
OLLAMA_FLASH_ATTENTION=1                # FlashAttention
OLLAMA_KV_CACHE_TYPE="q8_0"             # f16 | q8_0 | q4_0 — q8_0: KV 메모리 절반, 품질 손실 미미
OLLAMA_KEEP_ALIVE="10m"                 # 유휴 후 모델 언로드 시간
OLLAMA_NUM_PARALLEL=2                   # 동시 요청 수

# ---- 호스트: taskset (CPU 고정) ----
# 빈 값 = 고정 안 함. 11_snapshot.sh 의 lscpu 결과를 보고 지정 ("0-5" 또는 "0,2,4")
# 예: 12코어라면 OLLAMA_CPUSET="8-11", SERVE_CPUSET="8-11", JUPYTER_CPUSET="0-7"
#     → DataLoader worker 와 LLM 서버 토크나이저 스레드가 서로 코어를 빼앗지 않음
OLLAMA_CPUSET=""
SERVE_CPUSET=""
JUPYTER_CPUSET=""

# ---- [4] MPS ----
MPS_PIPE_DIR="/tmp/nvidia-mps"
MPS_LOG_DIR="/var/log/nvidia-mps"
MPS_ACTIVE_THREAD_PCT=""                # 예: 50 → 클라이언트당 SM 50% 상한 (빈 값 = 제한 없음)
MPS_MEM_LIMIT=""                        # 예: 5G → 클라이언트당 VRAM 상한 (빈 값 = 제한 없음)

# ---- [5] 모니터링 ----
MON_PROJECT="gpu-monitoring"            # docker compose 프로젝트명 (Portainer 스택 목록에 표시)
GRAFANA_PORT=3001                       # open-webui 가 3000 을 쓰는 경우가 많아 3001
GRAFANA_BIND="0.0.0.0"                  # 내부망 전용이면 127.0.0.1 + SSH 터널 권장
PROMETHEUS_PORT=9091                    # Cockpit(9090) 충돌 회피, 127.0.0.1 바인드
MON_ENABLE_DCGM=1                       # dcgm-exporter 시도 (GeForce 는 일부 지표만 지원 → 실패해도 nvidia_gpu_exporter 로 계속)
IMG_PROMETHEUS="prom/prometheus:v2.53.0"
IMG_GRAFANA="grafana/grafana:11.1.0"
IMG_GPU_EXPORTER="utkuozdemir/nvidia_gpu_exporter:1.2.1"
IMG_DCGM_EXPORTER="nvcr.io/nvidia/k8s/dcgm-exporter:3.3.7-3.5.0-ubuntu22.04"

# ---- [20~24] 서빙 ----
SERVE_NAME="llm-serve"                  # 서빙 컨테이너 이름 = 주피터에서 접근하는 호스트명
SERVE_PORT=8000                         # 모든 엔진 공통 포트 → 노트북 코드 수정 없이 엔진만 교체
SERVE_BIND="127.0.0.1"                  # 호스트 공개 범위 (외부 공개 금지 권장, 주피터는 내부망으로 접근)
SERVE_API_KEY=""                        # 비우면 인증 없음 (내부망 전용일 때만)
VLLM_IMAGE="vllm/vllm-openai:v0.31.0"
SGLANG_IMAGE="lmsysorg/sglang:v0.5.22"
TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0"
IMAGE_FALLBACK_TAG="latest"             # 위 태그 pull 실패 시 대체 태그

# 모델 (모두 HF 토큰 불필요)
MODEL_DEFAULT="Qwen/Qwen2.5-7B-Instruct-AWQ"        # ~5.5GB
MODEL_GPTQ="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"     # AWQ 와 비교용
MODEL_BIG="Qwen/Qwen2.5-14B-Instruct-AWQ"           # [23] 12GB 에서 13B급 서빙 (~9.4GB, 짧은 컨텍스트)
MODEL_SERVED_NAME="lab-llm"                          # API 에서 쓰는 모델명 (엔진 바뀌어도 동일)
MAX_MODEL_LEN=8192
MAX_MODEL_LEN_BIG=4096
GPU_MEM_UTIL=0.88
GPU_MEM_UTIL_BIG=0.95
# [21] Speculative Decoding — 기본은 n-gram(프롬프트 룩업, 추가 모델 불필요)
SPEC_CONFIG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'

# 서빙 시작 시 Ollama 를 내렸다가 종료 시 다시 올림 (12GB 를 한 엔진이 독점해야 KV 캐시 확보)
SERVE_STOP_OLLAMA=1
SERVE_STOP_MPS=1                        # MPS 가 켜져 있으면 서빙 중 잠시 끔

# ---- TensorRT-LLM (선택) ----
TRTLLM_HF_MODEL="Qwen/Qwen2.5-7B-Instruct"   # BF16 원본 → INT4 weight-only 엔진 빌드
TRTLLM_ENGINE_VOLUME="trtllm_engines"
TRTLLM_MAX_BATCH=8
TRTLLM_MAX_INPUT=4096
TRTLLM_MAX_SEQ=8192
