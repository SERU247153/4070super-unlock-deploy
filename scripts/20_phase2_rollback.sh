#!/usr/bin/env bash
# ============================================================
# 20_phase2_rollback.sh — 2단계 변경 원복 (1단계 언록·튜닝은 건드리지 않음)
#   서빙 종료 → 모니터링 중지 → MPS 해제 → 전력 캡 서비스 제거
#   → Ollama drop-in 제거 → (선택) 주피터 컨테이너를 13 이전 백업으로 복귀
#   HF 캐시 볼륨·커밋 이미지·Grafana 데이터는 보존 (수동 삭제 명령 안내)
# 사용법: sudo bash scripts/20_phase2_rollback.sh
# ============================================================
set -uo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root

confirm "2단계 변경을 원복할까요?" || exit 0

step "서빙 / 모니터링 / MPS"
bash "$SCRIPT_DIR/17_serve.sh" stop 2>/dev/null || true
(cd "$DEPLOY_DIR/monitoring" && docker compose --profile dcgm down 2>/dev/null) && ok "모니터링 중지"
bash "$SCRIPT_DIR/15_mps.sh" off >/dev/null 2>&1; bash "$SCRIPT_DIR/15_mps.sh" disable >/dev/null 2>&1
rm -f /etc/systemd/system/nvidia-mps.service /usr/local/sbin/gpu-mps-limits.sh && ok "MPS 해제"

step "전력 캡 서비스"
systemctl disable --now gpu-power-cap.service >/dev/null 2>&1
rm -f /etc/systemd/system/gpu-power-cap.service /usr/local/sbin/gpu-power-cap.sh /etc/gpu-lab/power.conf
ok "제거 (현재 전력 제한은 유지: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)) — 기본값: sudo nvidia-smi -pl <W>"

step "Ollama"
if [ -f /etc/systemd/system/ollama.service.d/gpu-lab.conf ]; then
  rm -f /etc/systemd/system/ollama.service.d/gpu-lab.conf
  systemctl daemon-reload; systemctl is-active --quiet ollama && systemctl restart ollama
  ok "drop-in 제거 + 재시작"
fi
systemctl daemon-reload

step "주피터 컨테이너"
BK=$(cat "$STATE_DIR/last_backup_container" 2>/dev/null || true)
if [ -n "$BK" ] && container_exists "$BK"; then
  if confirm "'$JUPYTER_CONTAINER' 를 13 이전 백업($BK)으로 되돌릴까요? (12 의 라이브러리 설치분도 사라짐)"; then
    docker rm -f "$JUPYTER_CONTAINER" >/dev/null
    docker rename "$BK" "$JUPYTER_CONTAINER"
    RP=$(python3 -c "import json,glob,sys;f=sorted(glob.glob('$STATE_DIR/${JUPYTER_CONTAINER}_inspect_*.json'))[-1];print((json.load(open(f))[0]['HostConfig'].get('RestartPolicy') or {}).get('Name') or 'no')" 2>/dev/null || echo unless-stopped)
    docker update --restart="$RP" "$JUPYTER_CONTAINER" >/dev/null
    docker start "$JUPYTER_CONTAINER" >/dev/null && ok "백업 컨테이너로 복귀 (restart=$RP)"
    rm -f "$STATE_DIR/last_backup_container"
  fi
else
  info "백업 컨테이너 없음 — 유지"
fi

echo
info "보존된 자원 (필요 시 수동 삭제):"
info "  docker volume rm $HF_VOLUME $TRTLLM_ENGINE_VOLUME     # 모델 캐시·엔진"
info "  docker image rm \$(docker images -q $COMMIT_REPO)       # 커밋 이미지"
info "  docker volume rm ${MON_PROJECT}_prom_data ${MON_PROJECT}_grafana_data"
printf "${G}2단계 원복 완료 (재부팅 불필요)${N}\n"
