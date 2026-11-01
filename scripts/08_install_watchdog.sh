#!/usr/bin/env bash
# ============================================================
# 08_install_watchdog.sh — GPU 언록 실패 자동 복구 워치독 설치
# 사용법: sudo bash 08_install_watchdog.sh
#
# 동작:
#   부팅 후 nvidia-smi 통신을 점검하고, 부팅 로그에
#   'gpu_unlock: unlock failed' 가 있으면(논콜드 웜 부팅 실패)
#   부트 옵션(gpu_unlock.targets)을 제거한 뒤 1회 재부팅하여
#   GPU를 자동 복구한다. 정상 부팅 시에는 복구 플래그를 초기화.
#   (실전 검증된 복구 절차의 자동화 — 원격/무인 환경용)
# ============================================================
set -euo pipefail

SVC_DIR=/usr/local/sbin
SVC=$SVC_DIR/gpu-unlock-watchdog.sh
UNIT=/etc/systemd/system/gpu-unlock-watchdog.service

sudo mkdir -p "$SVC_DIR"
sudo tee "$SVC" >/dev/null <<'EOF'
#!/usr/bin/env bash
# GPU 언록 실패 워치독 — 부팅 후 GPU 통신 불가 + 'unlock failed' 로그 시
# 부트 옵션 제거 후 1회 자동 재부팅으로 복구 (플래그로 루프 방지)
set -uo pipefail

FLAG=/var/lib/gpu-unlock/watchdog-recovered
LOG=/var/log/gpu-unlock-watchdog.log
log() { echo "$(date '+%F %T') $*" >>"$LOG"; }

# 부팅 직후 드라이버/도커 준비 대기 (최대 180초)
for _ in $(seq 1 36); do
  nvidia-smi >/dev/null 2>&1 && break
  sleep 5
done

if nvidia-smi >/dev/null 2>&1; then
  rm -f "$FLAG"          # 정상 부팅 → 플래그 초기화
  log "GPU 정상 — 워치독 통과"
  exit 0
fi

if journalctl -k -b 2>/dev/null | grep -q 'gpu_unlock: unlock failed'; then
  if [ -f "$FLAG" ]; then
    log "이미 1회 복구 시도함 — 추가 조치 없이 종료 (수동 점검 필요)"
    exit 0
  fi
  log "언록 실패 감지 — 부트 옵션 제거 후 복구 재부팅"
  mkdir -p /var/lib/gpu-unlock
  grubby --update-kernel=ALL --remove-args="gpu_unlock.targets=all" >>"$LOG" 2>&1
  touch "$FLAG"
  log "재부팅합니다 (복구 부팅은 언록 없이 진행)"
  systemctl reboot
else
  log "nvidia-smi 실패했으나 'unlock failed' 로그 없음 — 자동 조치 없이 종료 (수동 점검 권장)"
fi
EOF
sudo chmod +x "$SVC"

sudo tee "$UNIT" >/dev/null <<'EOF'
[Unit]
Description=GPU unlock failure watchdog (auto-recovery)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/gpu-unlock-watchdog.sh

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now gpu-unlock-watchdog.service
systemctl status gpu-unlock-watchdog.service --no-pager | head -6
