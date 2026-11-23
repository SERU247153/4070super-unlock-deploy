#!/usr/bin/env bash
# ============================================================
# 10_host.sh — 호스트(메인) 최적화. 재부팅하지 않습니다.
#   1) 전력 캡 자동 적용 (systemd, 부팅마다 POWER_CAP_W)
#   2) Ollama FlashAttention + KV 캐시 양자화 (서비스 재시작만)
#   3) Nsight Compute 컨테이너 사용 허용 (드라이버 옵션 — 다음 콜드 부팅부터)
#   4) GPU 모니터링 스택 (nvidia_gpu_exporter + Prometheus + Grafana)
# ⚠️ 웜 재부팅은 언록 -110 실패 위험이 있으므로 이 스크립트는 절대 재부팅하지 않습니다.
# 사용법: sudo bash optimize/10_host.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need_root
NVSMI="$(command -v nvidia-smi)"

step "1) 전력 캡 ${POWER_CAP_W}W 자동 적용"
MAX_PL=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
MIN_PL=$(nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ "$POWER_CAP_W" -lt "$MIN_PL" ] || [ "$POWER_CAP_W" -gt "$MAX_PL" ]; then
  warn "목표 ${POWER_CAP_W}W 가 허용 범위(${MIN_PL}~${MAX_PL}W) 밖 — 건너뜀"
else
  cat >/etc/systemd/system/gpu-power-cap.service <<UNIT
[Unit]
Description=GPU power cap ${POWER_CAP_W}W (optimize kit)
After=nvidia-persistenced.service systemd-modules-load.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=${NVSMI} -pm 1
ExecStart=${NVSMI} -pl ${POWER_CAP_W}

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable --now gpu-power-cap.service
  ok "현재 전력 제한: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
fi

step "2) Ollama FlashAttention + KV 캐시 ${OLLAMA_KV_CACHE_TYPE}"
if [ "$OLLAMA_TUNE" = "1" ] && systemctl list-unit-files | grep -q '^ollama.service'; then
  mkdir -p /etc/systemd/system/ollama.service.d
  cat >/etc/systemd/system/ollama.service.d/10-optimize.conf <<UNIT
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE}"
UNIT
  systemctl daemon-reload
  systemctl restart ollama
  sleep 2
  systemctl is-active --quiet ollama && ok "Ollama 재시작 완료 ($(systemctl show ollama -p Environment | tr '\n' ' '))" || bad "Ollama 재시작 실패 — journalctl -u ollama 확인"
else
  warn "Ollama 가 systemd 서비스가 아니거나 OLLAMA_TUNE=0 — 건너뜀"
fi

step "3) 컨테이너에서 Nsight Compute 사용 허용"
if [ "$PROFILING_FOR_ALL" = "1" ]; then
  CONF=/etc/modprobe.d/nvidia-profiling.conf
  echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" >"$CONF"
  dracut --force --kver "$(uname -r)" >/dev/null
  ok "$CONF 등록 + initramfs 재빌드 — ⚠️ 다음 '콜드' 파워 사이클 때 반영 (웜 재부팅 금지)"
  echo "  Nsight Systems(nsys)는 이 옵션 없이도 바로 사용 가능"
else
  echo "  PROFILING_FOR_ALL=0 — 건너뜀"
fi

step "4) GPU 모니터링 스택"
if [ "$MONITOR_ENABLE" = "1" ]; then
  if ss -ltn | awk '{print $4}' | grep -qE "[:.]${GRAFANA_PORT}\$"; then
    bad "포트 ${GRAFANA_PORT} 사용 중 — opt.conf 의 GRAFANA_PORT 변경 후 재실행"
  else
    MON=/opt/gpu-monitor
    mkdir -p "$MON"
    cp -r "$OPT_DIR/monitoring/." "$MON/"
    if [ ! -f "$MON/.env" ]; then
      PW=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 20)
      printf 'GRAFANA_PORT=%s\nGF_ADMIN_PASSWORD=%s\n' "$GRAFANA_PORT" "$PW" >"$MON/.env"
      chmod 600 "$MON/.env"
    fi
    # 대시보드 다운로드 (nvidia_gpu_exporter 공식 대시보드 14574) — 실패해도 진행
    DASH="$MON/grafana/dashboards/nvidia-gpu.json"
    if [ ! -s "$DASH" ]; then
      if curl -fsSL https://grafana.com/api/dashboards/14574/revisions/latest/download -o "$DASH.tmp"; then
        sed 's/\${DS_PROMETHEUS}/prometheus/g' "$DASH.tmp" >"$DASH"; rm -f "$DASH.tmp"
        ok "대시보드 다운로드 완료"
      else
        warn "대시보드 다운로드 실패 — Grafana 에서 Import → 14574 로 수동 추가"
        rm -f "$DASH.tmp"
      fi
    fi
    (cd "$MON" && docker compose up -d)
    ok "Grafana: http://<서버IP>:${GRAFANA_PORT}  (admin / $(grep GF_ADMIN_PASSWORD "$MON/.env" | cut -d= -f2))"
    echo "  비밀번호는 $MON/.env 에 저장됨. Portainer 에서도 gpu-monitor 스택으로 보입니다."
  fi
else
  echo "  MONITOR_ENABLE=0 — 건너뜀"
fi

step "호스트 작업 완료"
echo "  MPS(멀티 사용자 GPU 공유)는 컨테이너 재생성이 필요해 별도 스크립트: optimize/15_mps.sh"
