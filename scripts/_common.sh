#!/usr/bin/env bash
# ============================================================
# _common.sh — 2단계 스크립트 공통 헬퍼 (source 전용)
# ============================================================
# shellcheck disable=SC2034

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=../config.sh
source "$DEPLOY_DIR/config.sh"
# shellcheck source=../config.phase2.sh
source "$DEPLOY_DIR/config.phase2.sh"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; B='\033[1;34m'; N='\033[0m'
ok()   { printf "${G}  [OK]${N}   %s\n" "$*"; }
warn() { printf "${Y}  [WARN]${N} %s\n" "$*"; }
bad()  { printf "${R}  [FAIL]${N} %s\n" "$*"; }
info() { printf "         %s\n" "$*"; }
step() { printf "\n${B}== %s ==${N}\n" "$*"; }
die()  { bad "$*"; exit 1; }

RESULTS2_DIR="$DEPLOY_DIR/results/phase2"
STATE_DIR="/var/lib/gpu-lab"            # 롤백용 상태 기록 (호스트)

require_root() {
  [ "$(id -u)" = "0" ] || die "root 권한 필요 — sudo bash $0 $*"
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker 없음"
  docker info >/dev/null 2>&1 || die "docker 데몬 응답 없음"
}

container_exists()  { docker inspect "$1" >/dev/null 2>&1; }
container_running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; }

# 확인 질문 (ASSUME_YES=1 이면 자동 yes)
confirm() {
  local msg="$1"
  if [ "${ASSUME_YES:-0}" = "1" ]; then return 0; fi
  printf "${Y}  ? %s [yes/no]: ${N}" "$msg"
  local ans; read -r ans
  [ "$ans" = "yes" ] || [ "$ans" = "y" ]
}

# 이미지 pull — 지정 태그 실패 시 IMAGE_FALLBACK_TAG 로 재시도, 사용한 이미지명을 stdout 에 출력
pull_image() {
  local img="$1"
  if docker image inspect "$img" >/dev/null 2>&1; then echo "$img"; return 0; fi
  if docker pull "$img" >&2; then echo "$img"; return 0; fi
  local fb="${img%:*}:${IMAGE_FALLBACK_TAG}"
  printf "${Y}  [WARN]${N} %s pull 실패 → %s 로 재시도\n" "$img" "$fb" >&2
  docker pull "$fb" >&2 || return 1
  echo "$fb"
}

ensure_network() {
  docker network inspect "$LAB_NETWORK" >/dev/null 2>&1 || docker network create "$LAB_NETWORK" >/dev/null
  local c
  for c in "$JUPYTER_CONTAINER" "$OPENWEBUI_CONTAINER"; do
    [ -n "$c" ] && container_exists "$c" || continue
    if ! docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$c" | grep -qw "$LAB_NETWORK"; then
      docker network connect "$LAB_NETWORK" "$c" && info "$c → 네트워크 $LAB_NETWORK 연결"
    fi
  done
}

gpu_mem_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# 컨테이너 안의 파이썬 실행 (기본 사용자)
jexec() { docker exec -i "$JUPYTER_CONTAINER" "$@"; }

mps_active() { pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; }

cpuset_arg() { [ -n "$1" ] && echo "--cpuset-cpus=$1"; }
