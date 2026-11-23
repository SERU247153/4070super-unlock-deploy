#!/usr/bin/env bash
# ============================================================
# _serve_lib.sh — 서빙 공통 함수 (17_serve.sh · 19_trtllm.sh 가 source)
#   MODE 변수를 먼저 지정한 뒤 source 하세요.
# ============================================================
api_hdr() { [ -n "$SERVE_API_KEY" ] && echo "Authorization: Bearer $SERVE_API_KEY" || echo "X-No-Auth: 1"; }

stop_serving() {
  if container_exists "$SERVE_NAME"; then
    docker rm -f "$SERVE_NAME" >/dev/null && ok "서빙 컨테이너 종료"
  fi
  if [ -f "$STATE_DIR/serve_stopped_ollama" ]; then
    systemctl start ollama && ok "Ollama 재가동"; rm -f "$STATE_DIR/serve_stopped_ollama"
  fi
  if [ -f "$STATE_DIR/serve_stopped_mps" ]; then
    bash "$SCRIPT_DIR/15_mps.sh" on >/dev/null && ok "MPS 재가동"; rm -f "$STATE_DIR/serve_stopped_mps"
  fi
  rm -f "$STATE_DIR/serve_mode"
}

free_gpu() {
  if [ "$SERVE_STOP_OLLAMA" = "1" ] && systemctl is-active --quiet ollama 2>/dev/null; then
    systemctl stop ollama && touch "$STATE_DIR/serve_stopped_ollama" && ok "Ollama 일시 중지 (stop 시 복구)"
  fi
  if [ "$SERVE_STOP_MPS" = "1" ] && mps_active; then
    bash "$SCRIPT_DIR/15_mps.sh" off >/dev/null && touch "$STATE_DIR/serve_stopped_mps" && ok "MPS 일시 중지 (stop 시 복구)"
  fi
  sleep 2
  local used; used=$(gpu_mem_used_mib)
  if [ "${used:-0}" -gt 1500 ]; then
    warn "GPU 메모리 ${used} MiB 사용 중 — 아래 프로세스가 점유 (노트북 커널이면 커널 종료 권장)"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/    /'
  fi
}

common_run_args() {
  RUN=(docker run -d --name "$SERVE_NAME" --gpus all --ipc=host
       --network "$LAB_NETWORK" -p "${SERVE_BIND}:${SERVE_PORT}:8000"
       -v "${HF_VOLUME}:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface
       --label gpu-lab.mode="$MODE")
  [ -n "${HF_TOKEN:-}" ] && RUN+=(-e HF_TOKEN)       # 값은 명령행에 남기지 않고 환경에서 전달
  [ -n "$SERVE_CPUSET" ] && RUN+=(--cpuset-cpus "$SERVE_CPUSET")
  return 0
}

wait_ready() {
  info "모델 로드 대기 (첫 실행은 다운로드 포함 수 분) — 로그: docker logs -f $SERVE_NAME"
  local t=0
  until curl -sf -m 3 -H "$(api_hdr)" "http://127.0.0.1:${SERVE_PORT}/v1/models" >/dev/null 2>&1; do
    if ! container_running "$SERVE_NAME"; then
      bad "서빙 컨테이너가 종료됨 — 마지막 로그:"
      docker logs --tail 40 "$SERVE_NAME" 2>&1 | sed 's/^/    /'
      stop_serving; exit 1
    fi
    t=$((t+5)); sleep 5
    if [ $((t % 30)) = 0 ]; then
      printf "    %4ss | %s\n" "$t" "$(docker logs --tail 1 "$SERVE_NAME" 2>&1 | cut -c1-110)"
    fi
    [ "$t" -ge 1800 ] && { bad "30분 내 준비되지 않음"; exit 1; }
  done
  ok "준비 완료 (${t}s)"
}

show_summary() {
  echo
  info "엔드포인트(호스트) : http://127.0.0.1:${SERVE_PORT}/v1   모델명: ${MODEL_SERVED_NAME}"
  info "엔드포인트(노트북) : http://${SERVE_NAME}:8000/v1        (도커 네트워크 ${LAB_NETWORK})"
  info "open-webui 연결    : 설정 → 연결 → OpenAI API → http://${SERVE_NAME}:8000/v1"
  info "백엔드 확인        : $(docker logs "$SERVE_NAME" 2>&1 | grep -oiE '(using|selected) [a-z_ ]*(flashinfer|flash_attn|flash attention|triton|xformers)[a-z_ ]*backend' | tail -1 || true)"
  info "벤치마크           : sudo bash scripts/18_serve_bench.sh"
  nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader | sed 's/^/         GPU: /'
}

