#!/usr/bin/env bash
# ============================================================
# 13_commit_recreate.sh — 이미지 커밋 + HF 캐시 볼륨 추가 재생성  (항목 1·2, +4·30 준비)
#   1) 현재 구성 백업 (docker inspect → results/phase2, /var/lib/gpu-lab)
#   2) docker commit → ${COMMIT_REPO}:<날짜>  (12 에서 설치한 pip·커널·startup 보존)
#   3) 동일 구성 + 아래를 추가한 새 컨테이너로 교체
#        · -v ${HF_VOLUME}:${HF_MOUNT}  + HF_HOME   ← 모델·datasets 캐시 영구화 [1][19]
#        · --cap-add SYS_ADMIN                       ← Nsight Compute 카운터 [30]
#        · --shm-size (DataLoader num_workers)       ← [31]
#        · --ipc host + MPS 파이프 (--mps 지정 또는 MPS 실행 중일 때) ← [4]
#        · JUPYTER_CPUSET (taskset 설계)
#   4) 기존 컨테이너는 '<이름>_backup_<시각>' 으로 정지 보관 → 실패 시 자동 원복
# 사용법: sudo bash scripts/13_commit_recreate.sh [--mps] [--commit-only]
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root
require_docker

WITH_MPS=0; COMMIT_ONLY=0
for a in "$@"; do
  case "$a" in
    --mps) WITH_MPS=1 ;;
    --commit-only) COMMIT_ONLY=1 ;;
    *) die "알 수 없는 옵션: $a" ;;
  esac
done
mps_active && WITH_MPS=1

C="$JUPYTER_CONTAINER"
container_exists "$C" || die "컨테이너 '$C' 없음"
TS=$(date +%Y%m%d_%H%M%S)
TAG="${COMMIT_REPO}:$(date +%Y%m%d)"
mkdir -p "$RESULTS2_DIR" "$STATE_DIR"
chmod 700 "$STATE_DIR"

step "1) 현재 구성 백업"
INSPECT="$STATE_DIR/${C}_inspect_${TS}.json"
docker inspect "$C" > "$INSPECT"
chmod 600 "$INSPECT"
ok "$INSPECT  (환경변수에 토큰이 있을 수 있어 root 전용 보관)"
ORIG_IMAGE=$(docker inspect -f '{{.Config.Image}}' "$C")
info "원본 이미지: $ORIG_IMAGE"

COMPOSE_PROJ=$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$C" 2>/dev/null || true)
[ "$COMPOSE_PROJ" = "<no value>" ] && COMPOSE_PROJ=""

step "2) 이미지 커밋 [2]"
docker commit -m "gpu-unlock-deploy phase2: lab libs + startup ($TS)" "$C" "$TAG" >/dev/null
docker tag "$TAG" "${COMMIT_REPO}:latest"
ok "$TAG ($(docker image inspect -f '{{.Size}}' "$TAG" | awk '{printf "%.1f GB", $1/1e9}'))"
echo "$TAG" > "$STATE_DIR/last_commit_tag"

if [ "$COMMIT_ONLY" = "1" ]; then
  ok "--commit-only: 재생성 없이 종료"; exit 0
fi

if [ -n "$COMPOSE_PROJ" ]; then
  warn "이 컨테이너는 docker compose/Portainer 스택('$COMPOSE_PROJ')으로 관리됩니다."
  info "스택 정의를 직접 수정하는 것이 안전합니다. compose 파일에 아래를 추가하세요:"
  cat <<EOF
    image: ${COMMIT_REPO}:latest
    shm_size: "${RECREATE_SHM_SIZE}"
    cap_add: [SYS_ADMIN]
    environment:
      - HF_HOME=${HF_MOUNT}
    volumes:
      - ${HF_VOLUME}:${HF_MOUNT}
    networks: [default, ${LAB_NETWORK}]
  # (최하단)
  volumes:
    ${HF_VOLUME}:
      external: true
  networks:
    ${LAB_NETWORK}:
      external: true
EOF
  docker volume create "$HF_VOLUME" >/dev/null
  docker network inspect "$LAB_NETWORK" >/dev/null 2>&1 || docker network create "$LAB_NETWORK" >/dev/null
  info "볼륨 '$HF_VOLUME' / 네트워크 '$LAB_NETWORK' 는 미리 만들어 두었습니다."
  confirm "그래도 docker run 방식으로 강제 재생성할까요? (스택 관리에서 분리됨)" || exit 0
fi

step "3) 새 컨테이너 명령 생성"
docker volume create "$HF_VOLUME" >/dev/null
EXTRA=(--volume "${HF_VOLUME}:${HF_MOUNT}" --env "HF_HOME=${HF_MOUNT}" --shm "$RECREATE_SHM_SIZE")
[ "$RECREATE_ADD_SYS_ADMIN" = "1" ] && EXTRA+=(--cap SYS_ADMIN)
[ "$WITH_MPS" = "1" ] && EXTRA+=(--mps-pipe "$MPS_PIPE_DIR") && mkdir -p "$MPS_PIPE_DIR"
[ -n "$JUPYTER_CPUSET" ] && EXTRA+=(--cpuset "$JUPYTER_CPUSET")
RUN_FILE="$STATE_DIR/${C}_run_${TS}.sh"
python3 "$DEPLOY_DIR/serving/docker_recreate.py" "$INSPECT" --image "$TAG" "${EXTRA[@]}" > "$RUN_FILE"
chmod 600 "$RUN_FILE"
echo "  ----"
grep -v '^#' "$RUN_FILE" | sed -E 's/(TOKEN|PASSWORD|SECRET|KEY)=[^ ]*/\1=***/Ig; s/^/  /'
echo "  ----"
[ "$WITH_MPS" = "1" ] && info "MPS 모드: --ipc host + $MPS_PIPE_DIR 공유"

confirm "기존 '$C' 를 정지하고 위 명령으로 교체할까요? (실행 중 커널 종료, 1~2분 중단)" || { info "취소 — 이미지 커밋만 유지"; exit 0; }

step "4) 교체"
BACKUP="${C}_backup_${TS}"
docker stop -t 30 "$C" >/dev/null
docker rename "$C" "$BACKUP"
docker update --restart=no "$BACKUP" >/dev/null
echo "$BACKUP" > "$STATE_DIR/last_backup_container"
ok "기존 컨테이너 보관: $BACKUP (정지)"

rollback() {
  bad "새 컨테이너 기동 실패 → 원복"
  docker rm -f "$C" >/dev/null 2>&1 || true
  docker rename "$BACKUP" "$C"
  docker update --restart="$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "$C" 2>/dev/null || echo unless-stopped)" "$C" >/dev/null 2>&1 || true
  docker start "$C" >/dev/null && ok "원래 컨테이너 복귀"
  exit 1
}
bash "$RUN_FILE" >/dev/null || rollback
# 원래 restart 정책을 백업이 아니라 원본 inspect 에서 복원
RP=$(python3 -c "import json,sys;print((json.load(open(sys.argv[1]))[0]['HostConfig'].get('RestartPolicy') or {}).get('Name') or 'no')" "$INSPECT")
for n in $(grep '^#NETCONNECT ' "$RUN_FILE" | cut -d' ' -f2); do
  docker network connect "$n" "$C" 2>/dev/null && info "네트워크 재연결: $n"
done
ensure_network

sleep 8
container_running "$C" || rollback

step "5) 검증"
jexec nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | sed 's/^/  GPU: /' || rollback
jexec python -c "import torch;assert torch.cuda.is_available();print('  torch', torch.__version__, 'CUDA OK')" || rollback
jexec sh -c 'echo "  HF_HOME=$HF_HOME"; du -sh "$HF_HOME" 2>/dev/null | sed "s/^/  캐시 크기: /"'
jexec sh -c 'ls "$(ipython locate profile default)"/startup/00-gpu-opt.py >/dev/null 2>&1' && ok "TF32 startup 유지" || warn "startup 파일 없음 — 12 재실행"
jexec sh -c 'jupyter kernelspec list 2>/dev/null' | sed 's/^/  /'
ok "재시작 정책: $RP"
printf "\n${G}13 완료${N}\n"
info "확인 후 백업 삭제:  docker rm $BACKUP"
info "원복:             docker rm -f $C && docker rename $BACKUP $C && docker start $C"
info "다음: sudo bash scripts/14_host_tune.sh"
