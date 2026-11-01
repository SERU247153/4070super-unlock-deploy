#!/usr/bin/env bash
# ============================================================
# 10_calibrate.sh — 언록 후 전력·클럭·성능 캘리브레이션
# 사용법: sudo bash 10_calibrate.sh
#
# A. 전력 스윕 (클럭 고정): 전력 제한별 BF16/FP16 처리량 + TFLOPS/W
# B. 클럭 스윕 (전력 최대): 클럭별 처리량 + 이론 피크 대비 효율
#
# 결과: results/calibration/*.json + 종합 리포트
# 목적: 이 서버의 성능/전력 스위트 스팟 실측 (24/7 운영점 결정)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

G='\033[0;32m'; Y='\033[1;33m'; N='\033[0m'

POWER_SWEEP="${POWER_SWEEP:-245 220 200 180 160}"
CLOCK_SWEEP="${CLOCK_SWEEP:-2100,2100 2300,2300 2505,2505 2700,2700}"

CAL_DIR="$DEPLOY_DIR/results/calibration"
mkdir -p "$CAL_DIR"

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi 없음" >&2; exit 1; }
docker ps --filter "name=^/${BENCH_CONTAINER}$" --format '{{.Names}}' | grep -q . \
  || { echo "컨테이너 '$BENCH_CONTAINER' 없음" >&2; exit 1; }

max_power() {
  nvidia-smi -q -d POWER | awk -F: '/Max Power Limit/{gsub(/[^0-9.]/,"",$2); if ($2 ~ /[0-9]/) {print $2; exit}}'
}
restore_state() {
  TARGET="$POWER_LIMIT_MODE"
  [ "$TARGET" = "max" ] && TARGET=$(max_power)
  sudo nvidia-smi -pl "$TARGET" >/dev/null 2>&1 || true
  sudo nvidia-smi -rgc >/dev/null 2>&1 || true
}
trap restore_state EXIT

sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
# 깨끗한 시작 상태: 최대 전력 + 클럭 고정 해제
sudo nvidia-smi -pl "$(max_power)" >/dev/null 2>&1 || true
sudo nvidia-smi -rgc >/dev/null 2>&1 || true

run_case() { # $1=label $2=출력 json 경로
  local label=$1 out=$2
  local plog
  plog=$(mktemp)
  ( while true; do
      nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits >>"$plog" 2>/dev/null || break
      sleep 1
    done ) &
  local spid=$!
  docker cp "$SCRIPT_DIR/bench.py" "$BENCH_CONTAINER:/tmp/bench.py" >/dev/null
  docker exec "$BENCH_CONTAINER" python /tmp/bench.py \
    --label "$label" --size "$BENCH_MATRIX_SIZE" --iters "$BENCH_ITERS" \
    --out /tmp/calres.json >/dev/null
  docker cp "$BENCH_CONTAINER:/tmp/calres.json" "$out" >/dev/null
  kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
  python3 - "$out" "$plog" <<'PYEOF'
import json, sys
out, plog = sys.argv[1], sys.argv[2]
try:
    vals = [float(x.strip()) for x in open(plog) if x.strip()]
except OSError:
    vals = []
data = json.load(open(out, encoding="utf-8"))
if vals:
    data.setdefault("calibration", {})["power"] = {
        "samples": len(vals),
        "avg_w": round(sum(vals) / len(vals), 2),
        "max_w": round(max(vals), 2),
    }
json.dump(data, open(out, "w", encoding="utf-8"), indent=2)
PYEOF
  rm -f "$plog"
}

echo "== A. 전력 스윕 (클럭 ${BENCH_LOCK_CLOCKS} MHz 고정) =="
sudo nvidia-smi -lgc "$BENCH_LOCK_CLOCKS" >/dev/null 2>&1 \
  && printf "${G}클럭 고정 적용: %s${N}\n" "$BENCH_LOCK_CLOCKS" \
  || printf "${Y}클럭 고정 실패 — 부스트 상태로 진행${N}\n"
for P in $POWER_SWEEP; do
  echo "  전력 제한 ${P}W 측정 중..."
  sudo nvidia-smi -pl "$P" >/dev/null 2>&1 || { echo "  전력 설정 실패(${P}W) — 스킵"; continue; }
  run_case "p${P}" "$CAL_DIR/p${P}.json"
done

echo "== B. 클럭 스윕 (전력 $(max_power)W 고정) =="
MAX_PL=$(max_power)
sudo nvidia-smi -pl "$MAX_PL" >/dev/null 2>&1
for LC in $CLOCK_SWEEP; do
  NAME="c${LC/,/_}"
  echo "  클럭 ${LC} MHz 측정 중..."
  if sudo nvidia-smi -lgc "$LC" >/dev/null 2>&1; then
    run_case "$NAME" "$CAL_DIR/$NAME.json"
  else
    printf "${Y}  클럭 고정 실패(%s) — 스킵${N}\n" "$LC"
  fi
done
sudo nvidia-smi -rgc >/dev/null 2>&1 || true

echo
echo "== 캘리브레이션 종합 리포트 =="
python3 - "$CAL_DIR" <<'PYEOF'
import glob, json, os, sys

d = sys.argv[1]

def load(p):
    return json.load(open(p, encoding="utf-8"))

p_files = sorted(glob.glob(f"{d}/p*.json"),
                 key=lambda p: int(os.path.basename(p)[1:-5]))
print("\n[전력 스윕] 클럭 고정 상태의 전력 제한별 처리량")
print(f"{'전력제한':>8} {'BF16 TFLOPS':>12} {'FP16 TFLOPS':>12} {'평균W':>7} {'최대W':>7} {'TFLOPS/W(BF16)':>15}")
best_w = (None, 0)
for p in p_files:
    j = load(p)
    pl = int(os.path.basename(p)[1:-5])
    bf = j["results"]["bf16"]["tflops"]
    fp = j["results"]["fp16"]["tflops"]
    pw = j.get("calibration", {}).get("power", {})
    avg = pw.get("avg_w", 0); mx = pw.get("max_w", 0)
    eff = bf / avg if avg else 0
    print(f"{pl:>7}W {bf:>11.1f} {fp:>11.1f} {avg:>6.1f} {mx:>6.1f} {eff:>14.2f}")
    if avg and (best_w[0] is None or eff > best_w[1]):
        best_w = (pl, eff)

c_files = sorted(glob.glob(f"{d}/c*.json"))
print("\n[클럭 스윕] 최대 전력 상태의 클럭별 처리량")
print(f"{'클럭(MHz)':>10} {'BF16 TFLOPS':>12} {'FP16 TFLOPS':>12} {'효율(%)':>8}")
for p in c_files:
    j = load(p)
    clk = j["results"]["bf16"].get("clock_mhz", 0)
    bf = j["results"]["bf16"]["tflops"]
    fp = j["results"]["fp16"]["tflops"]
    eff = j["results"]["bf16"]["efficiency"] * 100
    print(f"{clk:>10.0f} {bf:>11.1f} {fp:>11.1f} {eff:>7.1f}")

if best_w[0] is not None:
    print(f"\n⭐ 성능/전력 스위트 스팟: 전력 제한 {best_w[0]}W "
          f"(BF16 {best_w[1]:.2f} TFLOPS/W)")
    print("   참고: 24/7 연속 학습은 스위트 스팟, 단발성 대형 작업은 최대 전력 권장")
PYEOF
