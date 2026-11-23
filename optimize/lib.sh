#!/usr/bin/env bash
# 공통 함수
OPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=opt.conf
source "$OPT_DIR/opt.conf"
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; B='\033[1;34m'; N='\033[0m'
ok()   { printf "${G}  [OK]${N}   %s\n" "$*"; }
warn() { printf "${Y}  [WARN]${N} %s\n" "$*"; }
bad()  { printf "${R}  [FAIL]${N} %s\n" "$*"; }
step() { printf "\n${B}== %s ==${N}\n" "$*"; }
need_root() { [ "$(id -u)" -eq 0 ] || { bad "root 로 실행하세요 (sudo bash $0)"; exit 1; }; }
have() { command -v "$1" >/dev/null 2>&1; }
container_running() { docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; }
