#!/usr/bin/env bash
# ============================================================
# 14_host_tune.sh — 호스트 튜닝  (항목 3 + Ollama FA·KV캐시 + taskset)
#   1) 부팅 시 전력 캡 자동 적용 서비스 (gpu-power-cap.service, 기본 180W)
#   2) Ollama: FlashAttention + KV 캐시 양자화(q8_0) + keep_alive/병렬 (systemd drop-in)
#   3) taskset: Ollama CPUAffinity (OLLAMA_CPUSET 지정 시)
#   ⚠️ 재부팅하지 않습니다 (웜 재부팅 금지 규칙). 전력 캡은 즉시 적용 + 다음 부팅부터 자동.
# 사용법: sudo bash scripts/14_host_tune.sh
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root
mkdir -p "$STATE_DIR"

step "1) 부팅 시 전력 캡 ${POWER_CAP_W}W 자동 적용 [3]"
MAX_PL=$(nvidia-smi -q -d POWER | awk -F: '/Max Power Limit/{gsub(/[^0-9.]/,"",$2); if ($2 ~ /[0-9]/) {print $2; exit}}')
MIN_PL=$(nvidia-smi -q -d POWER | awk -F: '/Min Power Limit/{gsub(/[^0-9.]/,"",$2); if ($2 ~ /[0-9]/) {print $2; exit}}')
awk -v t="$POWER_CAP_W" -v lo="$MIN_PL" -v hi="$MAX_PL" 'BEGIN{exit !(t>=lo && t<=hi)}' \
  || die "POWER_CAP_W=${POWER_CAP_W} 가 허용 범위(${MIN_PL}~${MAX_PL}W) 밖"

mkdir -p /etc/gpu-lab
echo "POWER_CAP_W=${POWER_CAP_W}" > /etc/gpu-lab/power.conf
cat > /usr/local/sbin/gpu-power-cap.sh <<'EOF'
#!/usr/bin/env bash
# 부팅 시 GPU 전력 캡 적용 (gpu-unlock-deploy 14) — 설정: /etc/gpu-lab/power.conf
set -u
. /etc/gpu-lab/power.conf
for _ in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 3; done
nvidia-smi -pm 1 >/dev/null
nvidia-smi -pl "$POWER_CAP_W"
EOF
chmod 755 /usr/local/sbin/gpu-power-cap.sh
cat > /etc/systemd/system/gpu-power-cap.service <<'EOF'
[Unit]
Description=GPU power cap (gpu-unlock-deploy phase2)
After=nvidia-persistenced.service gpu-unlock-watchdog.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/gpu-power-cap.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable gpu-power-cap.service >/dev/null 2>&1
systemctl restart gpu-power-cap.service
ok "현재 전력 제한: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader) (부팅마다 자동, 변경: /etc/gpu-lab/power.conf)"
info "참고: 03_tune.sh 를 다시 실행하면 POWER_LIMIT_MODE(max) 로 덮어쓰므로 config.sh 도 180 으로 맞추는 것을 권장"

step "2) Ollama FlashAttention · KV 캐시 · keep_alive"
if systemctl cat ollama >/dev/null 2>&1; then
  D=/etc/systemd/system/ollama.service.d
  mkdir -p "$D"
  [ -f "$D/gpu-lab.conf" ] && cp "$D/gpu-lab.conf" "$STATE_DIR/ollama-gpu-lab.conf.bak"
  {
    echo "# gpu-unlock-deploy phase2 (14_host_tune.sh) — 제거: rm 이 파일 && systemctl daemon-reload && systemctl restart ollama"
    echo "[Service]"
    echo "Environment=\"OLLAMA_FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION}\""
    echo "Environment=\"OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE}\""
    echo "Environment=\"OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}\""
    echo "Environment=\"OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}\""
    if [ -n "$OLLAMA_CPUSET" ]; then
      echo "CPUAffinity=$(echo "$OLLAMA_CPUSET" | tr ',' ' ')"
    fi
  } > "$D/gpu-lab.conf"
  systemctl daemon-reload
  if systemctl is-active --quiet ollama; then
    systemctl restart ollama && sleep 2
  fi
  systemctl show ollama -p Environment | tr ' ' '\n' | grep -E 'OLLAMA_(FLASH|KV|KEEP|NUM)' | sed 's/^/  /'
  [ -n "$OLLAMA_CPUSET" ] && info "CPUAffinity: $OLLAMA_CPUSET (taskset -cp $(systemctl show -p MainPID --value ollama) 로 확인)"
  ok "Ollama 적용 (FA 는 KV 캐시 양자화의 전제조건)"
elif docker ps --format '{{.Names}}' | grep -qi ollama; then
  warn "Ollama 가 컨테이너로 실행 중 — 컨테이너 재생성 시 아래 환경변수 추가:"
  info "-e OLLAMA_FLASH_ATTENTION=1 -e OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE} -e OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}"
else
  warn "Ollama 서비스를 찾지 못함 — 건너뜀"
fi

step "3) taskset (CPU 고정)"
if [ -z "$OLLAMA_CPUSET$SERVE_CPUSET$JUPYTER_CPUSET" ]; then
  info "config.phase2.sh 의 *_CPUSET 이 모두 비어 있어 건너뜀"
  info "CPU 구성: $(nproc)코어 논리 / NUMA $(lscpu | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2);print $2}')"
  info "11_snapshot.sh 결과를 공유하면 분할안을 맞춰 드립니다"
else
  info "OLLAMA_CPUSET=$OLLAMA_CPUSET (위에서 적용)"
  info "SERVE_CPUSET=$SERVE_CPUSET (17_serve.sh 가 --cpuset-cpus 로 적용)"
  if [ -n "$JUPYTER_CPUSET" ] && container_exists "$JUPYTER_CONTAINER"; then
    docker update --cpuset-cpus "$JUPYTER_CPUSET" "$JUPYTER_CONTAINER" >/dev/null && ok "$JUPYTER_CONTAINER → CPU $JUPYTER_CPUSET (재시작 없이 즉시)"
  fi
fi

printf "\n${G}14 완료${N} → 다음: sudo bash scripts/15_mps.sh on (다중 실습 시) / 16_monitoring.sh up\n"
