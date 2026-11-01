#!/usr/bin/env bash
# ============================================================
# 09_safe_reboot.sh — 유지보수용 안전 웜 재부팅
# 사용법: sudo bash 09_safe_reboot.sh
#
# 언록 적용 중에는 웜 재부팅(전원을 끊지 않는 reboot)이 논콜드
# 카드 조건으로 언록 실패(-110)를 일으킬 수 있다. 이 스크립트는
#   부트 옵션 제거 → 웜 재부팅(안전, 언록 없이 부팅)
# 을 수행한다. 재부팅 후 GPU는 '잠긴 상태'로 정상 동작한다.
#
# 언록 재적용 방법:
#   1) 부트 옵션 재등록: sudo bash scripts/02_install_unlock.sh
#      (또는 grubby --update-kernel=ALL --args="gpu_unlock.targets=all")
#   2) 콜드 파워 사이클(전원 완전 차단 30초+) — 스마트 플러그 권장
# ============================================================
set -uo pipefail

echo "이 스크립트는 언록을 해제한 상태로 웜 재부팅합니다 (유지보수용)."
echo "재부팅 후에는 GPU가 잠긴 상태로 동작하며, 재언록에는 콜드 파워 사이클이 필요합니다."
read -r -p "계속하시겠습니까? (yes/no): " ANS
if [ "$ANS" != "yes" ]; then echo "취소됨"; exit 0; fi

EXISTING=$(sudo grubby --info=ALL 2>/dev/null | grep -o 'gpu_unlock\.[a-z]*=[^ ]*' | sort -u || true)
if [ -n "$EXISTING" ]; then
  sudo grubby --update-kernel=ALL --remove-args="$(echo "$EXISTING" | tr '\n' ' ')"
  echo "부트 옵션 제거됨: $EXISTING"
fi
sudo systemctl reboot
