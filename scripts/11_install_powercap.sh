#!/usr/bin/env bash
# ============================================================
# 11_install_powercap.sh — 부팅 시 GPU 전력 캡 자동 적용 서비스
# 재부팅마다 전력 제한이 220W 기본값으로 리셋되는 문제 해결
# (캘리브레이션 실측 스위트 스팟 180W 자동 복귀)
# 사용: sudo bash 11_install_powercap.sh
# ============================================================
set -euo pipefail

UNIT=/etc/systemd/system/gpu-powercap.service
sudo tee "$UNIT" >/dev/null <<'EOF'
[Unit]
Description=GPU power cap 180W + persistence mode (calibration sweet spot)
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c "/usr/bin/nvidia-smi -pm 1 && /usr/bin/nvidia-smi -pl 180"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now gpu-powercap.service
systemctl status gpu-powercap.service --no-pager | head -4
