#!/usr/bin/env bash
# ============================================================
# 04_benchmark.sh — 언록 전/후 FP16·BF16 GEMM 처리량 측정
# 사용법: sudo bash 04_benchmark.sh before|after
#   before/after 결과는 results/ 에 JSON으로 저장되고,
#   둘 다 있으면 자동으로 비교표를 출력합니다.
# 필요: docker + nvidia-container-toolkit (주피터 서버라면 이미 있음)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

PHASE="${1:-}"
if [ "$PHASE" != "before" ] && [ "$PHASE" != "after" ]; then
  echo "사용법: $0 before|after" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "오류: nvidia-smi 없음 — NVIDIA 드라이버 설치 확인 필요" >&2
  exit 1
}

RESULTS_DIR="$DEPLOY_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "== 벤치마크 ($PHASE) =="
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  이미지 : $BENCH_IMAGE (없으면 최초 1회 pull — 시간 소요)"
echo

docker run --rm --gpus all \
  -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
  -v "$SCRIPT_DIR/bench.py:/tmp/bench.py:ro,Z" \
  -v "$RESULTS_DIR:/results:Z" \
  "$BENCH_IMAGE" \
  python /tmp/bench.py \
    --label "$PHASE" \
    --size "$BENCH_MATRIX_SIZE" \
    --iters "$BENCH_ITERS" \
    --out "/results/${PHASE}.json"

echo
if [ -f "$RESULTS_DIR/before.json" ] && [ -f "$RESULTS_DIR/after.json" ]; then
  echo "== 전/후 비교 =="
  python3 - "$RESULTS_DIR" <<'PYEOF'
import json, sys

d = sys.argv[1]
b = json.load(open(f"{d}/before.json", encoding="utf-8"))
a = json.load(open(f"{d}/after.json", encoding="utf-8"))
print(f"{'dtype':<6} {'before':>10} {'after':>10} {'gain':>9}")
for k in a["results"]:
    if k in b["results"]:
        tb = b["results"][k]["tflops"]
        ta = a["results"][k]["tflops"]
        print(f"{k:<6} {tb:>9.1f}T {ta:>9.1f}T {ta/tb-1:>+8.1%}")
print("\n(FP16/BF16 이 ~2배에 가까우면 언록 성공, 1배면 미적용 상태)")
PYEOF
else
  echo "비교 표는 before/after 둘 다 실행된 뒤 표시됩니다."
fi
