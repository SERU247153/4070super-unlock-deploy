#!/usr/bin/env bash
# ============================================================
# 16_monitoring.sh — GPU 대시보드: DCGM/nvidia_gpu_exporter + Prometheus + Grafana  (항목 5)
#   · compose 프로젝트 '${MON_PROJECT}' 로 실행 → Portainer 스택 목록에 자동 표시
#   · 대시보드 3종 자동 프로비저닝
#       GPU Lab Overview (자체 제작, 사용률·VRAM·온도·전력/제한·클럭·vLLM 토큰/s)
#       NVIDIA GPU Metrics (grafana.com #14574, nvidia_gpu_exporter)
#       NVIDIA DCGM Exporter (grafana.com #12239, dcgm 사용 시)
#   · GeForce 는 DCGM 공식 지원 대상이 아니라 일부 지표만 나옴 → nvidia_gpu_exporter 가 기본
# 사용법: sudo bash scripts/16_monitoring.sh up|down|status
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root
require_docker
docker compose version >/dev/null 2>&1 || die "docker compose 플러그인 없음 (dnf install docker-compose-plugin)"

MON_DIR="$DEPLOY_DIR/monitoring"
cd "$MON_DIR"

write_env() {
  cat > .env <<EOF
MON_PROJECT=${MON_PROJECT}
LAB_NETWORK=${LAB_NETWORK}
GRAFANA_PORT=${GRAFANA_PORT}
GRAFANA_BIND=${GRAFANA_BIND}
PROMETHEUS_PORT=${PROMETHEUS_PORT}
IMG_PROMETHEUS=${IMG_PROMETHEUS}
IMG_GRAFANA=${IMG_GRAFANA}
IMG_GPU_EXPORTER=${IMG_GPU_EXPORTER}
IMG_DCGM_EXPORTER=${IMG_DCGM_EXPORTER}
EOF
  # 관리자 비밀번호: 최초 1회 무작위 생성, root 전용 보관 (저장소에 올라가지 않음)
  mkdir -p /etc/gpu-lab
  if [ ! -f /etc/gpu-lab/grafana.env ]; then
    echo "GF_SECURITY_ADMIN_PASSWORD=$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)" > /etc/gpu-lab/grafana.env
    chmod 600 /etc/gpu-lab/grafana.env
  fi
  cp /etc/gpu-lab/grafana.env grafana.env && chmod 600 grafana.env
}

fetch_dashboard() {  # $1=grafana.com ID $2=파일명
  local f="grafana/dashboards/$2"
  [ -s "$f" ] && return 0
  if curl -fsSL -m 30 "https://grafana.com/api/dashboards/$1/revisions/latest/download" -o "$f.tmp"; then
    # 가져오기용 변수(${DS_PROMETHEUS} 등)를 프로비저닝 데이터소스 uid 로 치환
    python3 - "$f.tmp" "$f" <<'PY'
import json, re, sys
s = open(sys.argv[1]).read()
s = re.sub(r"\$\{DS_[A-Z0-9_]+\}", "prometheus", s)
d = json.loads(s)
d.pop("__inputs", None); d.pop("__requires", None); d["id"] = None
open(sys.argv[2], "w").write(json.dumps(d))
PY
    rm -f "$f.tmp"; ok "대시보드 #$1 → $2"
  else
    rm -f "$f.tmp"; warn "대시보드 #$1 다운로드 실패 (자체 대시보드만 사용)"
  fi
}

case "${1:-status}" in
  up)
    step "모니터링 스택 기동"
    write_env
    docker network inspect "$LAB_NETWORK" >/dev/null 2>&1 || docker network create "$LAB_NETWORK" >/dev/null
    fetch_dashboard 14574 nvidia-gpu-exporter.json
    PROFILES=()
    if [ "$MON_ENABLE_DCGM" = "1" ]; then
      fetch_dashboard 12239 dcgm-exporter.json
      PROFILES=(--profile dcgm)
    fi
    docker compose "${PROFILES[@]}" up -d
    sleep 12
    if [ "$MON_ENABLE_DCGM" = "1" ] && ! docker compose ps --status running --services | grep -qx dcgm-exporter; then
      warn "dcgm-exporter 가 GeForce 에서 기동 실패 (예상된 동작일 수 있음) — 로그:"
      docker compose logs --tail 5 dcgm-exporter 2>/dev/null | sed 's/^/    /'
      docker compose --profile dcgm stop dcgm-exporter >/dev/null 2>&1 || true
      info "nvidia_gpu_exporter 로 모든 핵심 지표 수집은 계속됩니다"
    fi
    "$0" status ;;
  down)
    docker compose --profile dcgm down
    ok "중지 (데이터 볼륨은 유지 — 완전 삭제: docker compose down -v)" ;;
  status)
    docker compose --profile dcgm ps --format 'table {{.Service}}\t{{.State}}\t{{.Ports}}' 2>/dev/null | sed 's/^/  /'
    if curl -s -m 3 "http://127.0.0.1:${PROMETHEUS_PORT}/api/v1/query?query=nvidia_smi_power_draw_watts" | grep -q '"value"'; then
      ok "Prometheus 가 GPU 지표 수집 중"
    else
      warn "아직 GPU 지표 없음 (기동 직후면 10초 후 재확인)"
    fi
    echo
    info "Grafana : http://<서버IP>:${GRAFANA_PORT}   (admin / 비밀번호: sudo cat /etc/gpu-lab/grafana.env)"
    info "          GRAFANA_BIND=127.0.0.1 이면: ssh -L ${GRAFANA_PORT}:localhost:${GRAFANA_PORT} <서버>"
    info "Prometheus: http://127.0.0.1:${PROMETHEUS_PORT} (서버 로컬 전용)"
    info "Portainer : Stacks 목록에 '${MON_PROJECT}' (외부 생성 스택으로 표시)" ;;
  *) die "사용법: $0 up|down|status" ;;
esac
