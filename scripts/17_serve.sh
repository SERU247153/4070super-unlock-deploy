#!/usr/bin/env bash
# ============================================================
# 17_serve.sh — LLM 서빙 전환기  (항목 20·21·22·23·24)
#   모든 모드가 같은 이름(llm-serve)·포트(8000)·모델명(lab-llm)의 OpenAI 호환 API 를 제공
#   → 노트북/open-webui 는 그대로 두고 엔진·기법만 바꿔 가며 비교 실습
#
#   모드          내용
#   vllm          vLLM + AWQ 4bit (Marlin 커널)                 [20][23]
#   vllm-fi       vLLM + FlashInfer 어텐션 백엔드                 [22]
#   vllm-spec     vLLM + Speculative Decoding (n-gram)           [21]
#   vllm-gptq     vLLM + GPTQ 4bit (AWQ 와 비교)                  [23]
#   vllm-big      vLLM + 14B AWQ (12GB 에서 13B급, 컨텍스트 4K)   [23]
#   sglang        SGLang (RadixAttention 접두어 캐싱 + FlashInfer) [9][22]
#   trtllm        TensorRT-LLM (19_trtllm.sh build 선행 필요)
#   stop | status | logs
#   구조화 출력 [24] 은 모든 모드에서 response_format=json_schema 로 사용 (18 벤치에 포함)
#
#   서빙 중에는 Ollama·MPS 를 잠시 내리고, stop 시 원래 상태로 복구합니다.
# 사용법: sudo bash scripts/17_serve.sh <모드>
#         HF 게이트 모델을 쓸 때만: sudo HF_TOKEN=hf_xxx bash scripts/17_serve.sh vllm
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root
require_docker
mkdir -p "$STATE_DIR"
MODE="${1:-status}"

# shellcheck source=_serve_lib.sh
source "$SCRIPT_DIR/_serve_lib.sh"

run_vllm() {   # $1=model $2=max_len $3=mem_util, 이후 추가 인자
  local model="$1" maxlen="$2" util="$3"; shift 3
  local img; img=$(pull_image "$VLLM_IMAGE") || die "vLLM 이미지 pull 실패"
  common_run_args
  [ "$MODE" = "vllm-fi" ] && RUN+=(-e VLLM_ATTENTION_BACKEND=FLASHINFER)
  RUN+=(--entrypoint vllm "$img" serve "$model"
        --host 0.0.0.0 --port 8000
        --served-model-name "$MODEL_SERVED_NAME"
        --max-model-len "$maxlen" --gpu-memory-utilization "$util"
        --enable-auto-tool-choice --tool-call-parser hermes)   # [24] 도구 호출 (Qwen2.5 = hermes 형식)
  [ -n "$SERVE_API_KEY" ] && RUN+=(--api-key "$SERVE_API_KEY")
  RUN+=("$@")
  "${RUN[@]}" >/dev/null
}

run_sglang() {
  local img; img=$(pull_image "$SGLANG_IMAGE") || die "SGLang 이미지 pull 실패"
  common_run_args
  RUN+=(--entrypoint python3 "$img" -m sglang.launch_server
        --model-path "$MODEL_DEFAULT" --served-model-name "$MODEL_SERVED_NAME"
        --host 0.0.0.0 --port 8000
        --context-length "$MAX_MODEL_LEN" --mem-fraction-static 0.85
        --attention-backend flashinfer --enable-metrics
        --tool-call-parser qwen25)
  [ -n "$SERVE_API_KEY" ] && RUN+=(--api-key "$SERVE_API_KEY")
  "${RUN[@]}" >/dev/null
}

case "$MODE" in
  vllm|vllm-fi|vllm-spec|vllm-gptq|vllm-big|sglang)
    step "서빙 시작: $MODE"
    stop_serving >/dev/null 2>&1 || true
    free_gpu
    ensure_network
    case "$MODE" in
      vllm|vllm-fi) run_vllm "$MODEL_DEFAULT" "$MAX_MODEL_LEN" "$GPU_MEM_UTIL" ;;
      vllm-spec)    run_vllm "$MODEL_DEFAULT" "$MAX_MODEL_LEN" "$GPU_MEM_UTIL" --speculative-config "$SPEC_CONFIG" ;;
      vllm-gptq)    run_vllm "$MODEL_GPTQ"    "$MAX_MODEL_LEN" "$GPU_MEM_UTIL" ;;
      vllm-big)     run_vllm "$MODEL_BIG"     "$MAX_MODEL_LEN_BIG" "$GPU_MEM_UTIL_BIG" --max-num-seqs 16 ;;
      sglang)       run_sglang ;;
    esac
    echo "$MODE" > "$STATE_DIR/serve_mode"
    wait_ready
    show_summary ;;
  trtllm)
    exec bash "$SCRIPT_DIR/19_trtllm.sh" serve ;;
  stop)
    step "서빙 종료"; stop_serving ;;
  logs)
    docker logs --tail 80 -f "$SERVE_NAME" ;;
  status)
    if container_running "$SERVE_NAME"; then
      ok "실행 중: $(cat "$STATE_DIR/serve_mode" 2>/dev/null || docker inspect -f '{{index .Config.Labels "gpu-lab.mode"}}' "$SERVE_NAME")"
      curl -s -m 3 -H "$(api_hdr)" "http://127.0.0.1:${SERVE_PORT}/v1/models" | python3 -c "import json,sys;[print('         모델:',m['id'],'| max_len:',m.get('max_model_len','?')) for m in json.load(sys.stdin)['data']]" 2>/dev/null || warn "API 응답 없음 (로딩 중?)"
      show_summary
    else
      info "서빙 없음 (Ollama: $(systemctl is-active ollama 2>/dev/null || echo n/a))"
    fi ;;
  *) die "사용법: $0 vllm|vllm-fi|vllm-spec|vllm-gptq|vllm-big|sglang|trtllm|stop|status|logs" ;;
esac
