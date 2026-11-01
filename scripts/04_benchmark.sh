#!/usr/bin/env bash
# ============================================================
# 04_benchmark.sh — 언록 전/후 FP16·BF16 GEMM 처리량 측정 + 캘리브레이션
# 사용법: sudo bash 04_benchmark.sh before|after
#   · 결과는 results/<phase>.json 으로 저장
#   · before/after 둘 다 있으면 자동 비교 + 캘리브레이션 판정
# 캘리브레이션 구성 (config.sh):
#   · BENCH_LOCK_CLOCKS: 클럭 고정으로 전/후 조건 통일 (권장)
#   · BENCH_POWER_SAMPLE: 실행 중 전력 샘플링 (평균/최대 기록)
# 필요: docker + nvidia-container-toolkit
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; N='\033[0m'

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
OUT_JSON="$RESULTS_DIR/${PHASE}.json"

# ---- 클럭 고정 (전/후 조건 통일) ----
LOCKED=0
PM_WAS_ENABLED=0
lock_clocks() {
  if [ "$BENCH_LOCK_CLOCKS" = "off" ]; then return 0; fi
  # -lgc 는 persistence mode 필요 → 임시 활성화 후 원상복구
  if nvidia-smi -q | grep -qi 'Persistence Mode.*: Enabled'; then PM_WAS_ENABLED=1; fi
  sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
  if sudo nvidia-smi -lgc "$BENCH_LOCK_CLOCKS" >/dev/null 2>&1; then
    LOCKED=1
    printf "${G}클럭 고정 적용: %s MHz (측정 중 유지)${N}\n" "$BENCH_LOCK_CLOCKS"
  else
    printf "${Y}클럭 고정 실패(%s) — 고정 없이 진행${N}\n" "$BENCH_LOCK_CLOCKS"
  fi
}
unlock_clocks() {
  if [ "$LOCKED" = "1" ]; then
    sudo nvidia-smi -rgc >/dev/null 2>&1 || true
    [ "$PM_WAS_ENABLED" = "1" ] || sudo nvidia-smi -pm 0 >/dev/null 2>&1 || true
    printf "${G}클럭 고정 해제 완료${N}\n"
  fi
}
trap unlock_clocks EXIT

echo "== 벤치마크 ($PHASE) =="
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  이미지 : $BENCH_IMAGE (없으면 최초 1회 pull — 시간 소요)"
lock_clocks
echo

# ---- 전력 샘플러 (1초 간격, 도커 실행 동안) ----
POWER_LOG=$(mktemp)
if [ "$BENCH_POWER_SAMPLE" = "1" ]; then
  ( while true; do
      nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits >>"$POWER_LOG" 2>/dev/null || break
      sleep 1
    done ) &
  SAMPLER_PID=$!
fi

# ---- 실행 ----
set +e
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
DOCKER_RC=$?
set -e

if [ "$BENCH_POWER_SAMPLE" = "1" ]; then
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
  # 전력 통계를 결과 JSON에 병합
  python3 - "$OUT_JSON" "$POWER_LOG" <<'PYEOF'
import json, sys
out, log = sys.argv[1], sys.argv[2]
try:
    vals = [float(x.strip()) for x in open(log) if x.strip()]
except OSError:
    vals = []
if vals:
    data = json.load(open(out, encoding="utf-8"))
    data.setdefault("calibration", {})["power"] = {
        "samples": len(vals),
        "avg_w": round(sum(vals) / len(vals), 2),
        "max_w": round(max(vals), 2),
    }
    json.dump(data, open(out, "w", encoding="utf-8"), indent=2)
    print(f"전력 샘플링: 평균 {sum(vals)/len(vals):.1f}W / 최대 {max(vals):.1f}W "
          f"({len(vals)} 샘플)")
PYEOF
fi
rm -f "$POWER_LOG"

[ "$DOCKER_RC" = "0" ] || { printf "${R}벤치마크 실행 실패 (docker exit %s)${N}\n" "$DOCKER_RC"; exit "$DOCKER_RC"; }

# ---- 전/후 비교 + 캘리브레이션 판정 ----
echo
if [ -f "$RESULTS_DIR/before.json" ] && [ -f "$RESULTS_DIR/after.json" ]; then
  python3 - "$RESULTS_DIR" <<'PYEOF'
import json, sys

d = sys.argv[1]
b = json.load(open(f"{d}/before.json", encoding="utf-8"))
a = json.load(open(f"{d}/after.json", encoding="utf-8"))

print("== 전/후 비교 (제한 경로: FP32 누적) ==")
print(f"{'dtype':<6} {'before':>10} {'after':>10} {'gain':>9}  {'eff(전/후)':>14}")
for k in a["results"]:
    if k in b["results"]:
        tb = b["results"][k]["tflops"]
        ta = a["results"][k]["tflops"]
        eb = b["results"][k]["efficiency"] * 100
        ea = a["results"][k]["efficiency"] * 100
        print(f"{k:<6} {tb:>9.1f}T {ta:>9.1f}T {ta/tb-1:>+8.1%}  {eb:>5.1f}%/{ea:>5.1f}%")

print("\n== 캘리브레이션 판정 ==")
ref_b = b.get("calibration", {}).get("reference", {}).get("tflops", 0)
ref_a = a.get("calibration", {}).get("reference", {}).get("tflops", 0)
if ref_b and ref_a:
    ref_ratio = ref_a / ref_b
    stable = 0.9 <= ref_ratio <= 1.1
    print(f"  참조 경로(fp16_acc, 언록 무관): {ref_b:.1f}T -> {ref_a:.1f}T "
          f"(비율 {ref_ratio:.2f}) — {'안정 ✓' if stable else '변동 ⚠'}")
else:
    stable = True

capped_gain = max(
    a["results"][k]["tflops"] / b["results"][k]["tflops"]
    for k in a["results"] if k in b["results"]
)
print()
if capped_gain >= 1.7 and stable:
    print("✅ 언록 성공 판정 — 제한 경로 ~2배 + 참조 경로 안정 (측정 조건 유효)")
elif capped_gain >= 1.7:
    print("⚠️ 제한 경로는 2배이나 참조 경로가 변동 — 클럭/전력 조건이 상이했을 수 있음. 재측정 권장")
elif capped_gain <= 1.15:
    print("❌ 언록 미적용 판정 — 05_verify.sh 로 부팅 로그를 확인하세요")
else:
    print("🔶 부분 적용 — 전력 스로틀링 또는 부분 언록 가능성. 로그 확인 후 재측정 권장")
PYEOF
else
  echo "비교 표는 before/after 둘 다 실행된 뒤 표시됩니다."
fi
