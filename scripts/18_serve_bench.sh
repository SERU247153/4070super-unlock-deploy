#!/usr/bin/env bash
# ============================================================
# 18_serve_bench.sh — 현재 서빙 모드 벤치마크 + 모드 간 비교표
#   TTFT · TPOT · 총 처리량(동시성 1/8) · 구조화 출력 유효율
#   결과: results/serving/<모드>.json  (커밋 대상, 보고서 그래프용)
# 사용법:
#   sudo bash scripts/18_serve_bench.sh            # 현재 모드 측정 후 비교표
#   sudo bash scripts/18_serve_bench.sh --all      # vllm → vllm-fi → vllm-spec → vllm-gptq → sglang 순회 측정
#   sudo bash scripts/18_serve_bench.sh --compare  # 비교표만
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root

OUT_DIR="$DEPLOY_DIR/results/serving"
mkdir -p "$OUT_DIR"
BENCH=(python3 "$DEPLOY_DIR/serving/bench_serve.py" --base "http://127.0.0.1:${SERVE_PORT}/v1"
       --model auto --key "$SERVE_API_KEY")

lock() {   # 측정 중 클럭 고정 (1단계 벤치와 동일 조건)
  if [ "${BENCH_LOCK_CLOCKS:-off}" != "off" ]; then
    nvidia-smi -lgc "$BENCH_LOCK_CLOCKS" >/dev/null 2>&1 && info "클럭 고정 $BENCH_LOCK_CLOCKS MHz"
    trap 'nvidia-smi -rgc >/dev/null 2>&1' EXIT
  fi
}

bench_current() {
  local mode; mode=$(cat "$STATE_DIR/serve_mode" 2>/dev/null || true)
  [ -n "$mode" ] && container_running "$SERVE_NAME" || die "서빙 중이 아님 — 먼저 sudo bash scripts/17_serve.sh vllm"
  step "벤치마크: $mode"
  "${BENCH[@]}" --label "$mode" --out "$OUT_DIR/$mode.json"
}

case "${1:-}" in
  --compare) ;;
  --all)
    lock
    for m in vllm vllm-fi vllm-spec vllm-gptq sglang; do
      bash "$SCRIPT_DIR/17_serve.sh" "$m" || { warn "$m 기동 실패 — 건너뜀"; continue; }
      bench_current
    done
    bash "$SCRIPT_DIR/17_serve.sh" stop ;;
  *)
    lock; bench_current ;;
esac
step "모드 비교"
"${BENCH[@]}" --compare "$OUT_DIR"
echo
info "읽는 법: rewrite 워크로드의 요청 tok/s → vllm-spec 이득 / prefix 의 TTFT → 접두어 캐싱 이득"
info "          c=8 총 tok/s → 동시 실습 처리량 / json 유효율 → 구조화 출력 [24]"
