#!/usr/bin/env bash
# ============================================================
# 06_rollback.sh — 원상복구
#   - 부트 옵션 제거
#   - gpu-unlock 패키지 제거
#   - 전력 제한 기본값 복원 / 클럭 고정 해제 / persistence mode 끔
#   - NVreg_EnableStreamMemOPs 옵션 제거 + initramfs 재빌드
# 사용법: sudo bash 06_rollback.sh
# 주의: 완전한 순정 상태는 재부팅 후입니다.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"

G='\033[0;32m'; Y='\033[1;33m'; N='\033[0m'
ok()   { printf "${G}  [OK]${N}   %s\n" "$*"; }
warn() { printf "${Y}  [WARN]${N} %s\n" "$*"; }

cat <<EOF
${Y}이 스크립트는 다음을 수행합니다:${N}
  1) 커널 부트 옵션에서 gpu_unlock.* 제거
  2) gpu-unlock-dkms / gpu-unlock-firmware 패키지 제거
  3) 전력 제한 기본값 복원, 클럭 고정 해제, persistence mode 끔
  4) NVreg_EnableStreamMemOPs 옵션 제거 + initramfs 재빌드
EOF
read -r -p "계속하시겠습니까? (yes/no): " ANS
if [ "$ANS" != "yes" ]; then echo "취소됨"; exit 0; fi

echo "== 1) 부트 옵션 제거 =="
EXISTING=$(sudo grubby --info=ALL 2>/dev/null | grep -o 'gpu_unlock\.[a-z]*=[^ ]*' | sort -u || true)
if [ -n "$EXISTING" ]; then
  sudo grubby --update-kernel=ALL --remove-args="$(echo "$EXISTING" | tr '\n' ' ')"
  ok "제거됨: $EXISTING"
else
  echo "  등록된 옵션 없음"
fi

echo "== 2) 패키지 제거 =="
if rpm -q gpu-unlock-dkms >/dev/null 2>&1; then
  sudo dnf remove -y gpu-unlock-dkms gpu-unlock-firmware
  ok "패키지 제거 완료"
else
  echo "  설치된 패키지 없음"
fi

echo "== 3) GPU 튜닝 되돌리기 =="
DEF_PL=$(nvidia-smi -q -d POWER 2>/dev/null | awk -F: '/Default Power Limit/{gsub(/[^0-9.]/,"",$2); print $2}')
if [ -n "${DEF_PL:-}" ]; then
  sudo nvidia-smi -pl "$DEF_PL" | sed 's/^/  /'
  ok "전력 제한 기본값(${DEF_PL}W) 복원"
fi
sudo nvidia-smi -rgc 2>/dev/null || true
ok "클럭 고정 해제"
# 주의: 서버가 원래 persistence mode를 사용 중이었다면(nvidia-persistenced 등)
# 부팅 후 다시 켜주세요. 여기서는 본 키트의 03_tune.sh 가 켠 것을 되돌립니다.
sudo nvidia-smi -pm 0 2>/dev/null || true
ok "persistence mode 해제"
sudo nvidia-smi -rg 2>/dev/null || true

echo "== 4) 드라이버 옵션 제거 =="
CONF=/etc/modprobe.d/nvidia-stream-mem-ops.conf
if [ -f "$CONF" ]; then
  sudo rm -f "$CONF"
  sudo dracut --force --kver "$(uname -r)"
  ok "$CONF 제거 + initramfs 재빌드"
else
  echo "  $CONF 없음"
fi

echo
printf "${G}원복 완료. 재부팅하면 완전한 순정 상태로 돌아갑니다.${N}\n"
