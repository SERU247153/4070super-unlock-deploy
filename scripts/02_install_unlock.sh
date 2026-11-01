#!/usr/bin/env bash
# ============================================================
# 02_install_unlock.sh — gpu-unlock 빌드/설치 + 부트 옵션 등록
# (Rocky Linux 9, RPM + grubby 경로)
# 사용법: sudo bash 02_install_unlock.sh [--reboot]
#   --reboot: 설치 후 재부팅 여부를 물어봅니다.
# 주의: 빌드 시 download.nvidia.com 에서 펌웨어 소스를 받아오므로
#       네트워크가 필요합니다. 적용은 재부팅 후입니다.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; N='\033[0m'
ok()   { printf "${G}  [OK]${N}   %s\n" "$*"; }
warn() { printf "${Y}  [WARN]${N} %s\n" "$*"; }
bad()  { printf "${R}  [FAIL]${N} %s\n" "$*"; exit 1; }

echo "== 1) 소스 준비 =="
if [ -d "$WORKDIR/.git" ]; then
  echo "  기존 소스 발견: $WORKDIR (git pull)"
  sudo git -C "$WORKDIR" pull --ff-only
else
  echo "  클론: $GIT_URL -> $WORKDIR"
  sudo mkdir -p "$WORKDIR"
  sudo git clone --depth 1 "$GIT_URL" "$WORKDIR"
fi
ok "소스 준비 완료"

echo "== 2) 빌드/설치 (make install → install-rpm) =="
echo "  (dnf install rpm-build → builddep → rpmbuild → dnf install 순서로 진행)"
sudo make -C "$WORKDIR" install
ok "패키지 설치 완료"

echo "== 3) 설치 결과 확인 =="
rpm -q gpu-unlock-dkms gpu-unlock-firmware | sed 's/^/  /'
sudo dkms status | grep gpu-unlock | sed 's/^/  /'

echo "== 4) 커널 부트 옵션 등록 =="
TARGETS_ARGS="gpu_unlock.targets=${GPU_UNLOCK_TARGETS}"
[ "$GPU_UNLOCK_RESUME" = "1" ] || TARGETS_ARGS="$TARGETS_ARGS gpu_unlock.resume=0"

EXISTING=$(sudo grubby --info=ALL 2>/dev/null | grep -o 'gpu_unlock\.[a-z]*=[^ ]*' | sort -u || true)
if [ -n "$EXISTING" ]; then
  warn "기존 등록 옵션 발견: $EXISTING — 제거 후 재등록"
  sudo grubby --update-kernel=ALL --remove-args="$(echo "$EXISTING" | tr '\n' ' ')"
fi
sudo grubby --update-kernel=ALL --args="$TARGETS_ARGS"
echo "  등록된 커널 항목:"
sudo grubby --info=ALL | grep -E '^(title|args)=' | sed 's/^/  /'
ok "부트 옵션 등록 완료"

echo
printf "${Y}설치 완료 — 적용에는 재부팅이 필요합니다.${N}\n"
echo "  * 서버 콘솔(원격 콘솔/IPMI)에 접근 가능한 상태에서 재부팅하세요."
echo "  * 재부팅 전에 성능 비교용 기준값을 남기려면: sudo bash scripts/04_benchmark.sh before"
if [ "${1:-}" = "--reboot" ]; then
  read -r -p "지금 재부팅할까요? (yes/no): " ANS
  if [ "$ANS" = "yes" ]; then
    echo "재부팅합니다..."
    sudo systemctl reboot
  fi
fi
