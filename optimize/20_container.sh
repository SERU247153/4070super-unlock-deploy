#!/usr/bin/env bash
# ============================================================
# 20_container.sh — 주피터 컨테이너 최적화 (호스트에서 실행 → 컨테이너 안으로 전달)
#   1) HF 모델·데이터셋 캐시를 영구 마운트 안으로 (컨테이너 재생성해도 유지)
#   2) 실습 라이브러리 설치 (NGC torch 고정 — torch 를 절대 갈아끼우지 않음)
#   3) TF32 + cudnn.benchmark + 메모리 얼로케이터 설정 자동 주입 (IPython startup + 터미널)
#   4) jupyterlab-nvdashboard, ONNX Runtime GPU, DALI
#   5) 템플릿 노트북 3종 복사
#   6) docker commit 으로 이미지 보존
# 사용법: sudo bash optimize/20_container.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/lib.sh"
need_root
container_running || { bad "컨테이너 $CONTAINER 가 실행 중이 아님"; exit 1; }

step "1) 영구 저장 위치 탐지"
PERSIST="$HF_PERSIST_DIR"
if [ -z "$PERSIST" ]; then
  # 쓰기 가능한 마운트 중 작업 공간으로 보이는 곳을 우선 선택
  PERSIST=$(docker inspect "$CONTAINER" --format '{{range .Mounts}}{{if .RW}}{{.Destination}}{{"\n"}}{{end}}{{end}}' \
    | grep -vE '^/(dev|proc|sys|etc|tmp|var/run|run)' \
    | awk '{s=0} /workspace|notebook|work|home|data|root/{s=1} {print s" "$0}' | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [ -z "$PERSIST" ]; then
  warn "영구 마운트를 찾지 못함 — 캐시는 컨테이너 내부(/root/.cache)에 저장되어 재생성 시 사라집니다"
  warn "→ 나중에 컨테이너 재생성 시 -v /data/hf-cache:/hf-cache 추가 후 HF_PERSIST_DIR=/hf-cache 로 재실행 권장"
  CACHE_ROOT="/root/.cache"
else
  CACHE_ROOT="$PERSIST/.cache"
  ok "영구 마운트: $PERSIST → 캐시 위치 $CACHE_ROOT"
fi
TEMPLATE_DST="${PERSIST:-/root}/optimize-templates"

step "2~5) 컨테이너 내부 설정"
docker exec "$CONTAINER" mkdir -p /opt/optimize
docker cp "$OPT_DIR/container_setup.sh" "$CONTAINER":/opt/optimize/container_setup.sh
docker cp "$OPT_DIR/templates/." "$CONTAINER":/opt/optimize/templates/
docker exec -i \
  -e CACHE_ROOT="$CACHE_ROOT" -e TEMPLATE_DST="$TEMPLATE_DST" \
  -e PIP_PROFILE="$PIP_PROFILE" -e INSTALL_ORT="$INSTALL_ORT" \
  -e INSTALL_DALI="$INSTALL_DALI" -e INSTALL_DEEPSPEED="$INSTALL_DEEPSPEED" \
  "$CONTAINER" bash /opt/optimize/container_setup.sh

if [ "$RESTART_CONTAINER" = "1" ]; then
  step "컨테이너 재시작 (nvdashboard 활성화)"
  docker restart "$CONTAINER" >/dev/null && ok "재시작 완료"
else
  warn "nvdashboard 탭은 JupyterLab 재시작 후 보입니다 (RESTART_CONTAINER=1 로 재실행하거나, 실습 없는 시간에 docker restart $CONTAINER)"
fi

if [ "$COMMIT_IMAGE" = "1" ]; then
  step "6) 이미지 보존 (docker commit)"
  TAG="${CONTAINER}:optimized-$(date +%Y%m%d)"
  FREE_GB=$(df -BG --output=avail /var/lib/docker | tail -1 | tr -dc 0-9)
  if [ "${FREE_GB:-0}" -lt 30 ]; then
    warn "도커 저장 공간 ${FREE_GB}GB — 30GB 미만이라 커밋 생략 (공간 확보 후 docker commit $CONTAINER $TAG)"
  else
    docker commit -m "optimize kit $(date +%F)" "$CONTAINER" "$TAG" >/dev/null && ok "커밋 완료: $TAG"
    echo "  컨테이너를 재생성할 때 이 이미지를 쓰면 pip 설치분이 유지됩니다 (마운트 데이터는 원래 유지)"
  fi
fi
step "컨테이너 작업 완료 — 검증: sudo bash optimize/40_verify.sh"
