#!/usr/bin/env bash
# ============================================================
# 07_attention_bench.sh — FlashAttention 실워크로드 성능 측정
# 사용법: sudo bash 07_attention_bench.sh
#   · 기존 컨테이너(BENCH_CONTAINER)에서 실행 — FlashAttention 필요
#   · 결과: results/attention.json
#   · 잠긴 상태 상한(≈71 TFLOPS) 초과 여부로 언록 효과 표시
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

RESULTS_DIR="$DEPLOY_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "== FlashAttention 벤치마크 =="
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  실행   : 컨테이너 '$BENCH_CONTAINER' (docker exec)"

set +e
docker cp "$SCRIPT_DIR/bench_attention.py" "$BENCH_CONTAINER:/tmp/bench_attention.py"
docker exec "$BENCH_CONTAINER" python /tmp/bench_attention.py --out /tmp/attention_result.json
RC=$?
if [ "$RC" = "0" ]; then
  docker cp "$BENCH_CONTAINER:/tmp/attention_result.json" "$RESULTS_DIR/attention.json"
fi
set -e

[ "$RC" = "0" ] || { echo "벤치마크 실행 실패 (exit $RC)" >&2; exit "$RC"; }
printf "\n결과 파일: %s\n" "$RESULTS_DIR/attention.json"
