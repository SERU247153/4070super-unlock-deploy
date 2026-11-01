#!/usr/bin/env bash
# ============================================================
# 01_preflight.sh — 사전 점검 (Rocky Linux 9 + RTX 4070 SUPER)
# 사용법: sudo bash 01_preflight.sh [--fix]
#   --fix: 누락된 필수 패키지를 dnf로 자동 설치
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

FIX_MODE=0
[ "${1:-}" = "--fix" ] && FIX_MODE=1

echo "== 1) OS =="
# shellcheck disable=SC1091
. /etc/os-release
echo "  ${PRETTY_NAME:-$ID ${VERSION_ID:-}}"
case "${ID:-}:${ID_LIKE:-}" in
  rocky*|*rhel*|*centos*|rhel*|centos*) ok "RHEL 계열 — make install-rpm 경로 사용" ;;
  *) warn "RHEL 계열이 아님 (ID=$ID, ID_LIKE=$ID_LIKE). 스크립트는 Rocky 9 기준이라 수정이 필요할 수 있음" ;;
esac

echo "== 2) 커널 =="
echo "  $(uname -r)"

echo "== 3) NVIDIA GPU / 드라이버 =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,pci.device_id,driver_version --format=csv,noheader | sed 's/^/  /'
  DEVID=$(nvidia-smi --query-gpu=pci.device_id --format=csv,noheader | head -1 | tr -d '[:space:]')
  if [ "$DEVID" = "0x2783" ]; then
    ok "RTX 4070 SUPER (10de:2783) — gpu-unlock 지원 목록에 등록됨"
  else
    warn "장치 ID가 4070 SUPER(0x2783)와 다름: $DEVID — 지원 여부 재확인 필요"
  fi
else
  bad "nvidia-smi 없음 — NVIDIA 드라이버 설치가 선행되어야 함"
fi

echo "== 4) Secure Boot =="
if command -v mokutil >/dev/null 2>&1; then
  if mokutil --sb-state 2>/dev/null | grep -qi 'enabled'; then
    bad "Secure Boot 켜짐 — BIOS에서 비활성화 필요 (미서명 DKMS 모듈은 로드 불가)"
  else
    ok "Secure Boot 꺼짐"
  fi
else
  warn "mokutil 없음 — BIOS에서 Secure Boot 상태 직접 확인"
fi

echo "== 5) 필요 패키지 =="
MISSING=()
for p in gcc make python3 zstd git curl grubby pciutils dracut rpm-build dnf-plugins-core epel-release dkms "kernel-devel-$(uname -r)"; do
  if rpm -q "$p" >/dev/null 2>&1; then ok "$p"
  else bad "$p 없음"; MISSING+=("$p"); fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  if [ "$FIX_MODE" = "1" ]; then
    echo "  -> 누락 패키지 설치 중 (sudo 권한 필요)..."
    sudo dnf install -y epel-release || true
    sudo dnf install -y dkms rpm-build dnf-plugins-core "kernel-devel-$(uname -r)" \
         gcc make python3 zstd git curl grubby pciutils dracut
    FAIL=0
    echo "  -> 재확인:"
    for p in "${MISSING[@]}"; do
      if rpm -q "$p" >/dev/null 2>&1; then ok "$p"
      else bad "$p 여전히 없음"; fi
    done
  else
    warn "누락 패키지 있음 — '$0 --fix' 로 자동 설치하거나 직접:"
    echo "      sudo dnf install -y epel-release"
    echo "      sudo dnf install -y dkms rpm-build dnf-plugins-core kernel-devel-\$(uname -r) gcc make python3 zstd git curl grubby pciutils dracut"
  fi
fi

echo "== 6) 네트워크 =="
curl -sI --max-time 8 https://github.com >/dev/null 2>&1 \
  && ok "github.com 접근 가능" \
  || bad "github.com 접근 불가 — git clone 실패"
curl -sI --max-time 8 https://download.nvidia.com >/dev/null 2>&1 \
  && ok "download.nvidia.com 접근 가능" \
  || bad "download.nvidia.com 접근 불가 — 빌드 시 NVIDIA 펌웨어 소스 다운로드 실패"

echo "== 7) 도커 GPU =="
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  ok "docker 동작 중"
else
  warn "docker 미동작 (권한 문제라면 sudo docker, 벤치마크는 호스트 파이썬 폴백 불가 — 도커 확인 필요)"
fi

echo "== 8) 현재 전력/ReBAR 상태 =="
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q -d POWER 2>/dev/null | grep -iE 'power limit' | sed 's/^/  /'
  if nvidia-smi -q 2>/dev/null | grep -qi 'resizable bar'; then
    nvidia-smi -q | grep -i 'resizable bar' | sed 's/^/  /'
  fi
fi

echo
if [ "$FAIL" = "1" ]; then
  printf "${R}사전 점검 실패 항목 있음 — FAIL 해결 후 다시 실행하세요.${N}\n"
  exit 1
else
  printf "${G}사전 점검 통과 — scripts/02_install_unlock.sh 진행 가능${N}\n"
fi
