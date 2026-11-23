#!/usr/bin/env bash
# ============================================================
# 15_mps.sh — CUDA MPS (Multi-Process Service)  (항목 4)
#   여러 노트북 커널이 GPU 를 시분할(context switch) 대신 동시에 나눠 쓰도록 함
#   → 다인원 실습 시 작은 커널 여러 개의 총 처리량↑, 컨텍스트 메모리↓
# 사용법: sudo bash scripts/15_mps.sh on|off|status|enable|disable
#   on      : 지금 시작          off     : 지금 중지
#   enable  : 부팅 시 자동 시작   disable : 자동 시작 해제
# 주의:
#   · 컨테이너가 MPS 를 쓰려면 --ipc host + 파이프 디렉터리 공유 필요
#     → 'on' 후 sudo bash scripts/13_commit_recreate.sh --mps
#   · LLM 서빙(17_serve.sh)은 GPU 독점이 유리 → 서빙 중에는 자동으로 MPS 를 끔
#   · MPS 아래에서는 한 커널의 치명적 오류(예: 잘못된 메모리 접근)가 같은 서버의 다른 클라이언트도 멈출 수 있음
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root

UNIT=/etc/systemd/system/nvidia-mps.service
install_unit() {
  cat > "$UNIT" <<EOF
[Unit]
Description=NVIDIA CUDA MPS control daemon (gpu-unlock-deploy phase2)
After=nvidia-persistenced.service gpu-power-cap.service

[Service]
Type=forking
Environment=CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIR}
Environment=CUDA_MPS_LOG_DIRECTORY=${MPS_LOG_DIR}
ExecStartPre=/usr/bin/mkdir -p ${MPS_PIPE_DIR} ${MPS_LOG_DIR}
ExecStart=/usr/bin/nvidia-cuda-mps-control -d
ExecStartPost=/usr/local/sbin/gpu-mps-limits.sh
ExecStop=/bin/sh -c 'echo quit | CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIR} /usr/bin/nvidia-cuda-mps-control'

[Install]
WantedBy=multi-user.target
EOF
  cat > /usr/local/sbin/gpu-mps-limits.sh <<EOF
#!/bin/sh
# MPS 기본 한도 (15_mps.sh) — 빈 값이면 제한 없음
export CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIR}
sleep 1
[ -n "${MPS_ACTIVE_THREAD_PCT}" ] && echo "set_default_active_thread_percentage ${MPS_ACTIVE_THREAD_PCT}" | nvidia-cuda-mps-control
[ -n "${MPS_MEM_LIMIT}" ] && echo "set_default_device_pinned_mem_limit 0 ${MPS_MEM_LIMIT}" | nvidia-cuda-mps-control
exit 0
EOF
  chmod 755 /usr/local/sbin/gpu-mps-limits.sh
  systemctl daemon-reload
}

status() {
  if mps_active; then ok "MPS 제어 데몬 실행 중 (파이프: $MPS_PIPE_DIR)"
    echo get_server_list | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" nvidia-cuda-mps-control 2>/dev/null | sed 's/^/  서버 PID: /'
  else info "MPS 꺼짐"; fi
  info "부팅 자동 시작: $(systemctl is-enabled nvidia-mps 2>/dev/null || echo no)"
  info "컴퓨트 모드: $(nvidia-smi --query-gpu=compute_mode --format=csv,noheader)"
  if container_exists "$JUPYTER_CONTAINER"; then
    if docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' "$JUPYTER_CONTAINER" | grep -qw "$MPS_PIPE_DIR"; then
      ok "$JUPYTER_CONTAINER: MPS 파이프 공유됨"
    else
      warn "$JUPYTER_CONTAINER: MPS 미연결 → sudo bash scripts/13_commit_recreate.sh --mps"
    fi
  fi
}

case "${1:-status}" in
  on)
    install_unit
    if [ "$(gpu_mem_used_mib)" -gt 1500 ] 2>/dev/null; then
      warn "현재 GPU 사용 중($(gpu_mem_used_mib) MiB) — 기존 프로세스는 MPS 밖에서 계속 실행됩니다 (재시작해야 MPS 경유)"
    fi
    systemctl start nvidia-mps && sleep 1
    status ;;
  off)
    systemctl stop nvidia-mps 2>/dev/null || echo quit | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1; status ;;
  enable)  install_unit; systemctl enable nvidia-mps >/dev/null 2>&1; ok "부팅 시 MPS 자동 시작"; status ;;
  disable) systemctl disable nvidia-mps >/dev/null 2>&1 || true; ok "자동 시작 해제"; status ;;
  status)  status ;;
  *) die "사용법: $0 on|off|status|enable|disable" ;;
esac
