#!/usr/bin/env bash
# ============================================================
# 19_trtllm.sh — TensorRT-LLM: 접근 확인 → 엔진 빌드 → 서빙  (선택, 실험적)
#   check : HF·NGC 접근, 모델 접근 권한, GPU 아키텍처 확인 (아무것도 바꾸지 않음)
#   build : HF 체크포인트 → INT4 weight-only 변환 → trtllm-build 엔진 (볼륨 trtllm_engines)
#   serve : trtllm-serve 로 OpenAI 호환 API (llm-serve:8000, 17/18 과 동일 인터페이스)
# 사용법: sudo bash scripts/19_trtllm.sh check|build|serve
#         게이트 모델이면: sudo HF_TOKEN=hf_xxx bash scripts/19_trtllm.sh build
# 참고: TRT-LLM 은 버전마다 예제 경로·CLI 옵션이 바뀌므로 스크립트가 컨테이너 안에서 자동 탐지합니다.
#       첫 실행 결과(특히 build 로그)를 공유해 주시면 버전에 맞춰 보정합니다.
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root
require_docker
mkdir -p "$STATE_DIR"
# shellcheck disable=SC2034  # _serve_lib.sh 에서 사용
MODE="trtllm"
# shellcheck source=_serve_lib.sh
source "$SCRIPT_DIR/_serve_lib.sh"

NAME=$(basename "$TRTLLM_HF_MODEL" | tr 'A-Z' 'a-z')-int4wo
ENGINE_DIR="/engines/$NAME"

check() {
  step "TensorRT-LLM 사전 점검"
  local code
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
         "https://huggingface.co/api/models/${TRTLLM_HF_MODEL}")
  case "$code" in
    200) ok "HF 모델 접근 가능: $TRTLLM_HF_MODEL" ;;
    401|403) die "HF 접근 거부($code) — 게이트 모델이면 라이선스 승인 후 HF_TOKEN 전달" ;;
    000) die "huggingface.co 접속 불가 — 서버 네트워크 확인" ;;
    *) warn "HF 응답 코드 $code" ;;
  esac
  if curl -s -m 10 "https://huggingface.co/api/models/${TRTLLM_HF_MODEL}" | grep -q '"gated":\s*\("auto"\|"manual"\|true\)'; then
    [ -n "${HF_TOKEN:-}" ] || warn "게이트 모델 — 다운로드에 HF_TOKEN 필요"
  fi
  local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  [ "${cc%%.*}" -ge 8 ] && ok "GPU compute capability $cc (Ada sm_89 지원)" || die "compute capability $cc 미지원"
  local img; img=$(pull_image "$TRTLLM_IMAGE") || die "NGC 이미지 pull 실패 ($TRTLLM_IMAGE) — nvcr.io 접근 확인"
  echo "$img" > "$STATE_DIR/trtllm_image"
  ok "이미지: $img"
  docker run --rm --gpus all "$img" python3 -c "import tensorrt_llm as t; print('  tensorrt_llm', t.__version__)" 2>/dev/null | tail -1
  df -h /var/lib/docker | tail -1 | awk '{print "         docker 디스크 여유: "$4" (원본 ~15GB + 엔진 ~5GB 필요)"}'
}

trt_img() { cat "$STATE_DIR/trtllm_image" 2>/dev/null || pull_image "$TRTLLM_IMAGE"; }

build() {
  check
  step "엔진 빌드: $TRTLLM_HF_MODEL → INT4 weight-only ($ENGINE_DIR)"
  stop_serving >/dev/null 2>&1 || true
  free_gpu
  docker volume create "$TRTLLM_ENGINE_VOLUME" >/dev/null
  local img; img=$(trt_img)
  local t0=$SECONDS
  docker run --rm --gpus all --ipc=host \
    -v "${HF_VOLUME}:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface \
    -v "${TRTLLM_ENGINE_VOLUME}:/engines" ${HF_TOKEN:+-e HF_TOKEN} \
    -e MODEL="$TRTLLM_HF_MODEL" -e NAME="$NAME" \
    -e MAXB="$TRTLLM_MAX_BATCH" -e MAXIN="$TRTLLM_MAX_INPUT" -e MAXSEQ="$TRTLLM_MAX_SEQ" \
    "$img" bash -c '
      set -euo pipefail
      VER=$(python3 -c "import tensorrt_llm as t;print(t.__version__)" 2>/dev/null | tail -1)
      echo "[1/4] TensorRT-LLM $VER — 변환 스크립트 탐색"
      CONVERT=$(find / -path "*qwen*" -name convert_checkpoint.py 2>/dev/null | head -1 || true)
      if [ -z "$CONVERT" ]; then
        git clone -q --depth 1 -b "v$VER" https://github.com/NVIDIA/TensorRT-LLM /tmp/trtllm-src 2>/dev/null \
          || git clone -q --depth 1 https://github.com/NVIDIA/TensorRT-LLM /tmp/trtllm-src
        CONVERT=$(find /tmp/trtllm-src -path "*qwen*" -name convert_checkpoint.py | head -1)
      fi
      echo "      $CONVERT"
      echo "[2/4] HF 체크포인트 다운로드 (공유 캐시)"
      HFDIR=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s(\"$MODEL\"))" | tail -1)
      echo "[3/4] INT4 weight-only 변환 (CPU 로드, RAM ~16GB)"
      python3 "$CONVERT" --model_dir "$HFDIR" --output_dir "/engines/ckpt/$NAME" \
        --dtype float16 --use_weight_only --weight_only_precision int4 --load_model_on_cpu
      echo "[4/4] trtllm-build"
      trtllm-build --checkpoint_dir "/engines/ckpt/$NAME" --output_dir "/engines/$NAME" \
        --gemm_plugin auto --max_batch_size "$MAXB" --max_input_len "$MAXIN" --max_seq_len "$MAXSEQ"
      echo "$HFDIR" > "/engines/$NAME/hf_path"
      rm -rf "/engines/ckpt/$NAME"
      du -sh "/engines/$NAME"
    ' || { bad "빌드 실패 — 위 로그를 공유해 주세요"; stop_serving >/dev/null 2>&1; exit 1; }
  ok "빌드 완료 ($((SECONDS - t0))s)"
  stop_serving >/dev/null 2>&1 || true   # Ollama/MPS 복구
}

serve() {
  step "TensorRT-LLM 서빙"
  local img; img=$(trt_img)
  docker run --rm -v "${TRTLLM_ENGINE_VOLUME}:/engines" "$img" test -f "$ENGINE_DIR/config.json" 2>/dev/null \
    || die "엔진 없음 — 먼저: sudo bash scripts/19_trtllm.sh build"
  stop_serving >/dev/null 2>&1 || true
  free_gpu
  ensure_network
  common_run_args
  RUN+=(-v "${TRTLLM_ENGINE_VOLUME}:/engines" "$img" bash -c "
    HF=\$(cat $ENGINE_DIR/hf_path)
    EXTRA=''
    trtllm-serve --help 2>&1 | grep -q -- '--backend' && EXTRA='--backend tensorrt'
    exec trtllm-serve $ENGINE_DIR --tokenizer \"\$HF\" --host 0.0.0.0 --port 8000 \
      --max_batch_size $TRTLLM_MAX_BATCH \$EXTRA")
  "${RUN[@]}" >/dev/null
  echo "trtllm" > "$STATE_DIR/serve_mode"
  wait_ready
  show_summary
  info "모델명은 엔진 경로로 노출됩니다 (18 벤치는 /v1/models 에서 자동 인식)"
}

case "${1:-check}" in
  check) check ;;
  build) build ;;
  serve) serve ;;
  *) die "사용법: $0 check|build|serve" ;;
esac
