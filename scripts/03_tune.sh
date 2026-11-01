#!/usr/bin/env bash
# ============================================================
# 03_tune.sh — 성능 튜닝 (언록 후 실행 권장)
#   1) persistence mode 켬
#   2) 전력 제한 상향 (config.sh POWER_LIMIT_MODE)
#   3) 클럭 고정 (선택, config.sh CLOCK_LOCK)
#   4) NVreg_EnableStreamMemOPs=1 등록 + initramfs 재빌드 (재부팅 시 반영)
#   5) ReBAR 상태 확인
# 사용법: sudo bash 03_tune.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; N='\033[0m'
ok()   { printf "${G}  [OK]${N}   %s\n" "$*"; }
warn() { printf "${Y}  [WARN]${N} %s\n" "$*"; }
bad()  { printf "${R}  [FAIL]${N} %s\n" "$*"; }

command -v nvidia-smi >/dev/null 2>&1 || bad "nvidia-smi 없음 — NVIDIA 드라이버 확인 필요"

echo "== 1) Persistence Mode =="
sudo nvidia-smi -pm 1 | sed 's/^/  /'

echo "== 2) 전력 제한 =="
POWER_INFO=$(nvidia-smi -q -d POWER)
# 드라이버 버전마다 라벨/중복이 다르므로: 첫 번째 '숫자 유효값'만 채택 (N/A 제외)
MIN_PL=$(echo "$POWER_INFO" | awk -F: '/Min Power Limit/{gsub(/[^0-9.]/,"",$2); if ($2 ~ /[0-9]/) {print $2; exit}}')
MAX_PL=$(echo "$POWER_INFO" | awk -F: '/Max Power Limit/{gsub(/[^0-9.]/,"",$2); if ($2 ~ /[0-9]/) {print $2; exit}}')
DEF_PL=$(echo "$POWER_INFO" | awk -F: '/Default Power Limit/{gsub(/[^0-9.]/,"",$2); if ($2 ~ /[0-9]/) {print $2; exit}}')
CUR_PL=$(echo "$POWER_INFO" | awk -F: '/Current Power Limit/{gsub(/[^0-9.]/,"",$2); if ($2 ~ /[0-9]/) {print $2; exit}}')
echo "  min=${MIN_PL}W / default=${DEF_PL}W / max=${MAX_PL}W / 현재=${CUR_PL}W"

if [ -z "$MIN_PL" ] || [ -z "$MAX_PL" ] || [ -z "$DEF_PL" ]; then
  bad "전력 제한 정보를 읽지 못함 — 이 항목 건너뜀"
else
  TARGET="$POWER_LIMIT_MODE"
  [ "$TARGET" = "max" ] && TARGET="$MAX_PL"
  if awk -v t="$TARGET" -v lo="$MIN_PL" -v hi="$MAX_PL" 'BEGIN{exit !(t>=lo && t<=hi)}'; then
    sudo nvidia-smi -pl "$TARGET" | sed 's/^/  /'
    ok "전력 제한 ${TARGET}W 적용"
  else
    warn "목표 ${TARGET}W 가 허용 범위(${MIN_PL}~${MAX_PL}W) 밖 — 설정 생략"
  fi
fi

echo "== 3) 클럭 고정 (선택) =="
if [ "$CLOCK_LOCK" != "off" ]; then
  sudo nvidia-smi -lgc "$CLOCK_LOCK" | sed 's/^/  /'
  ok "클럭 범위 고정: $CLOCK_LOCK MHz (해제: sudo nvidia-smi -rgc)"
else
  echo "  off — 건너뜀 (config.sh CLOCK_LOCK 에 'min,max' MHz 형식 지정 시 활성화)"
fi

echo "== 4) 드라이버 옵션 NVreg_EnableStreamMemOPs =="
if [ "$ENABLE_STREAM_MEM_OPS" = "1" ]; then
  CONF=/etc/modprobe.d/nvidia-stream-mem-ops.conf
  echo "options nvidia NVreg_EnableStreamMemOPs=1" | sudo tee "$CONF" >/dev/null
  sudo dracut --force --kver "$(uname -r)"
  ok "등록됨 ($CONF + initramfs 재빌드) — 재부팅 시 반영"
else
  echo "  off — 건너뜀"
fi

echo "== 5) ReBAR 상태 =="
if nvidia-smi -q 2>/dev/null | grep -qi 'resizable bar'; then
  nvidia-smi -q | grep -i 'resizable bar' | sed 's/^/  /'
  if ! nvidia-smi -q | grep -i 'resizable bar' | grep -qi 'enabled'; then
    warn "ReBAR 비활성 — BIOS에서 'Above 4G Decoding' + 'Re-Size BAR' 활성화 권장 (일부 워크로드 +5~10%)"
  fi
else
  warn "ReBAR 정보 확인 불가 (GPU에서 미지원일 수 있음)"
fi

echo
echo "== 적용 요약 =="
nvidia-smi --query-gpu=name,persistence_mode,power.limit,clocks.max.sm --format=csv | sed 's/^/  /'
printf "${G}튜닝 완료 (드라이버 옵션은 재부팅 시 반영)${N}\n"
