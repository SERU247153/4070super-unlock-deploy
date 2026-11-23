#!/usr/bin/env bash
# 최적화 되돌리기. 재부팅하지 않음. 컨테이너 내 pip 패키지는 되돌리지 않음(아래 안내 참고).
set -uo pipefail
source "$(dirname "$0")/lib.sh"; need_root
step "전력 제한 서비스 제거 → 기본값 복원"
systemctl disable --now gpu-power-cap.service 2>/dev/null; rm -f /etc/systemd/system/gpu-power-cap.service
def=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1)
nvidia-smi -pl "${def%.*}" && ok "전력 제한 ${def}W"
step "Ollama 튜닝 제거"
rm -f /etc/systemd/system/ollama.service.d/10-optimize.conf; systemctl daemon-reload; systemctl restart ollama 2>/dev/null; ok "Ollama 재시작"
step "프로파일링 옵션 제거"
if [[ -f /etc/modprobe.d/nvidia-profiling.conf ]]; then rm -f /etc/modprobe.d/nvidia-profiling.conf; dracut --force && ok "다음 콜드 부팅부터 원복"; fi
step "모니터링 중지"
[[ -d /opt/gpu-monitor ]] && (cd /opt/gpu-monitor && docker compose down) && ok "중지 (데이터 볼륨 유지, 완전 삭제: docker compose down -v)"
docker compose -f "$OPT_DIR/serving/compose.yml" --profile vllm --profile sglang down 2>/dev/null
step "컨테이너 시작 스크립트 제거"
if container_running "$CONTAINER"; then
  docker exec "$CONTAINER" bash -c 'rm -f /root/.ipython/profile_default/startup/00-perf.py /home/*/.ipython/profile_default/startup/00-perf.py /etc/profile.d/optimize.sh; sed -i "/profile.d\/optimize.sh/d" /root/.bashrc' && ok "제거"
fi
warn "pip 패키지까지 되돌리려면 원본 이미지로 컨테이너를 재생성하세요: docker inspect $CONTAINER --format '{{.Config.Image}}'"
