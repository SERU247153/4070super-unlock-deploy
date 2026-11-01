#!/usr/bin/env bash
# ============================================================
# 05_verify.sh — 언록 적용 확인 (재부팅 후 실행)
# 사용법: sudo bash 05_verify.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; N='\033[0m'
FAIL=0
ok()   { printf "${G}  [OK]${N}   %s\n" "$*"; }
warn() { printf "${Y}  [WARN]${N} %s\n" "$*"; }
bad()  { printf "${R}  [FAIL]${N} %s\n" "$*"; FAIL=1; }

echo "== 1) 부트 옵션 =="
if grep -q 'gpu_unlock' /proc/cmdline; then
  grep -o 'gpu_unlock[^ ]*' /proc/cmdline | sed 's/^/  /'
  ok "옵션 등록 확인"
else
  bad "현재 부팅에 gpu_unlock 옵션 없음 — 02 스크립트와 재부팅 여부 확인"
fi

echo "== 2) 부팅 로그 =="
LOG=$(journalctl -k -b 2>/dev/null | grep -i 'gpu_unlock' || true)
if [ -n "$LOG" ]; then
  echo "$LOG" | sed 's/^/  /'
  if echo "$LOG" | grep -qi 'unlocked'; then
    ok "'unlocked' 기록 확인 — 언록 성공"
  else
    warn "gpu_unlock 로그는 있으나 'unlocked' 문구 없음 — 위 로그 내용 검토"
  fi
else
  bad "부팅 로그에 gpu_unlock 없음 — 모듈 미로드 또는 저널 접근 권한 문제 (sudo 로 실행했는지 확인)"
fi

echo "== 3) PCI 바인딩 =="
PCI=$(lspci -D 2>/dev/null | grep -iE 'NVIDIA' | head -1 | awk '{print $1}')
if [ -n "$PCI" ]; then
  echo "  $PCI"
  lspci -nnk -s "$PCI" | grep -iE 'kernel driver' | sed 's/^/  /'
  if lspci -nnk -s "$PCI" | grep -qi 'nvidia'; then
    ok "nvidia 드라이버 정상 바인딩"
  else
    bad "nvidia 미바인딩 — GPU 사용 불가 상태"
  fi
else
  bad "NVIDIA 카드 미발견"
fi

echo "== 4) GPU 상태 =="
if nvidia-smi --query-gpu=name,driver_version,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'; then
  ok "nvidia-smi 통신 정상"
else
  bad "nvidia-smi 실패 — GPU가 드라이버와 통신 불가 (언록 실패로 GSP가 멈췄을 가능성. 06_rollback.sh 로 부트 옵션 제거 후 재부팅 필요)"
fi

echo "== 5) DKMS 상태 =="
sudo dkms status 2>/dev/null | grep gpu-unlock | sed 's/^/  /' || warn "dkms status 에 gpu-unlock 없음"

echo
if [ "$FAIL" = "1" ]; then
  printf "${R}검증 실패 항목 있음 — 위 내용 확인 후 필요하면 scripts/06_rollback.sh 로 원복${N}\n"
  exit 1
else
  printf "${G}언록 정상 동작 — 이제 scripts/04_benchmark.sh after 로 성능 확인${N}\n"
fi
