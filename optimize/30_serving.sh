#!/usr/bin/env bash
# vLLM / SGLang 을 필요할 때만 기동. Ollama 에 올라간 모델을 먼저 내려 VRAM 을 비운다.
set -euo pipefail
source "$(dirname "$0")/lib.sh"; need_root
CMD="${1:-status}"
export VLLM_IMAGE VLLM_MODEL VLLM_PORT VLLM_GPU_UTIL VLLM_MAX_LEN SGLANG_IMAGE SGLANG_PORT
[[ "${USE_FLASHINFER:-0}" == 1 ]] && export VLLM_ATTENTION_BACKEND=FLASHINFER
DC="docker compose -f $OPT_DIR/serving/compose.yml"
mkdir -p /opt/hf-cache

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

unload_ollama() {
  have ollama || return 0
  for m in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do
    step "Ollama 모델 내림: $m"; ollama stop "$m" 2>/dev/null || \
      curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null || true
  done
  sleep 2
}

up() {
  local prof="$1" port="$2"
  unload_ollama
  local need; need=$(awk -v t="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -v u="$VLLM_GPU_UTIL" 'BEGIN{printf "%d", t*u}')
  local free; free=$(free_mib)
  if (( free < need )); then
    bad "VRAM 여유 ${free}MiB < 필요 ${need}MiB. Jupyter 커널(모델 로드된 노트북)을 종료하거나 opt.conf 의 VLLM_GPU_UTIL 을 낮추세요."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1
  fi
  step "$prof 기동 (모델: $VLLM_MODEL, 첫 실행은 다운로드로 수 분 소요)"
  $DC --profile "$prof" up -d "$prof"
  for i in $(seq 1 120); do
    curl -sf "localhost:$port/v1/models" >/dev/null && { ok "$prof 준비 완료 → http://<서버IP>:$port/v1 (Jupyter 안에서는 http://172.17.0.1:$port/v1)"; return; }
    sleep 5
  done
  warn "아직 응답 없음. 로그: docker logs -f $prof"
}

case "$CMD" in
  vllm-up)   up vllm "$VLLM_PORT" ;;
  sglang-up) up sglang "$SGLANG_PORT" ;;
  down)      $DC --profile vllm --profile sglang down; ok "서빙 종료 (Ollama 는 다음 요청 때 자동으로 모델을 다시 올림)" ;;
  status)    docker ps --filter name='^(vllm|sglang)$' --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'; echo "VRAM free: $(free_mib) MiB" ;;
  *) echo "사용법: $0 vllm-up|sglang-up|down|status  (FlashInfer: USE_FLASHINFER=1 $0 vllm-up)"; exit 2 ;;
esac
