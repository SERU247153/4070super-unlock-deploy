#!/usr/bin/env bash
# ============================================================
# 11_snapshot.sh — 2단계 사전 점검 (읽기 전용, 아무것도 바꾸지 않음)
#   GPU·드라이버·컨테이너 구성·NGC 패키지·Ollama·CPU 토폴로지·포트·디스크를 수집
#   결과: results/phase2/snapshot_<시각>.txt  (공유 전 IP 등 민감정보 확인)
# 사용법: sudo bash scripts/11_snapshot.sh
# ============================================================
set -uo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

mkdir -p "$RESULTS2_DIR"
OUT="$RESULTS2_DIR/snapshot_$(date +%Y%m%d_%H%M%S).txt"

sec() { printf "\n### %s\n" "$*"; }
run() { printf "\$ %s\n" "$*"; bash -c "$*" 2>&1 | sed 's/^/  /'; }

{
sec "GPU / 드라이버"
run "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,power.limit,power.max_limit,persistence_mode,compute_mode,clocks.max.sm --format=csv"
run "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
run "journalctl -k -b 2>/dev/null | grep -m3 'gpu_unlock' || echo '(gpu_unlock 로그 없음)'"
run "systemctl is-active nvidia-persistenced 2>/dev/null; systemctl list-unit-files 2>/dev/null | grep -Ei 'gpu|nvidia|ollama|mps' "
run "pgrep -a nvidia-cuda-mps || echo 'MPS 꺼짐'"
run "cat /proc/driver/nvidia/params 2>/dev/null | grep -Ei 'RestrictProfiling|StreamMemOPs'"

sec "CPU / 메모리 / 디스크 (taskset 설계용)"
run "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket|NUMA'"
run "lscpu -e=CPU,CORE,SOCKET,NODE,MAXMHZ 2>/dev/null | head -40"
run "free -h"
run "df -h / /var/lib/docker 2>/dev/null"
run "docker system df"

sec "Docker"
run "docker version --format '{{.Server.Version}}'; docker compose version 2>/dev/null || echo 'compose 플러그인 없음'"
run "docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"
run "docker network ls; docker volume ls"

sec "jupyter 컨테이너 구성 ($JUPYTER_CONTAINER)"
if container_exists "$JUPYTER_CONTAINER"; then
  run "docker inspect $JUPYTER_CONTAINER --format 'image={{.Config.Image}} user={{.Config.User}} restart={{.HostConfig.RestartPolicy.Name}} ipc={{.HostConfig.IpcMode}} shm={{.HostConfig.ShmSize}} net={{.HostConfig.NetworkMode}} runtime={{.HostConfig.Runtime}} caps={{.HostConfig.CapAdd}} cpuset={{.HostConfig.CpusetCpus}}'"
  run "docker inspect $JUPYTER_CONTAINER --format '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}}{{println}}{{end}}'"
  run "docker inspect $JUPYTER_CONTAINER --format '{{json .HostConfig.PortBindings}}'"
  run "docker inspect $JUPYTER_CONTAINER --format '{{range \$k,\$v := .Config.Labels}}{{\$k}}={{\$v}}{{println}}{{end}}' | grep -Ei 'compose|portainer' || echo '(compose/portainer 라벨 없음 → docker run 으로 생성된 컨테이너)'"
  run "docker exec $JUPYTER_CONTAINER sh -c 'whoami; echo HOME=\$HOME; echo HF_HOME=\${HF_HOME:-unset}; cat /etc/os-release | head -2'"
  run "docker exec $JUPYTER_CONTAINER python -c \"import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version())\""
  run "docker exec $JUPYTER_CONTAINER pip list 2>/dev/null | grep -Ei '^(torch|flash|transformer|accelerate|peft|trl|bitsandbytes|datasets|numpy|jupyterlab|jupyter-server|notebook|nvidia-dali|onnx|tensorrt|torch-tensorrt|deepspeed|triton|apex|openai) '"
  run "docker exec $JUPYTER_CONTAINER sh -c 'which nsys ncu; nsys --version 2>/dev/null; ncu --version 2>/dev/null | tail -1'"
  run "docker exec $JUPYTER_CONTAINER sh -c 'jupyter kernelspec list; ipython locate profile 2>/dev/null; ls \$(ipython locate profile 2>/dev/null)/startup 2>/dev/null'"
  run "docker exec $JUPYTER_CONTAINER df -h / | tail -1"
else
  echo "  컨테이너 '$JUPYTER_CONTAINER' 없음 — config.phase2.sh 의 JUPYTER_CONTAINER 확인"
fi

sec "Ollama"
run "systemctl is-active ollama 2>/dev/null || echo 'systemd ollama 서비스 없음'"
run "systemctl cat ollama 2>/dev/null | grep -E 'Environment|ExecStart|CPUAffinity'"
run "command -v ollama && ollama --version && ollama ps"
run "docker ps --format '{{.Names}} {{.Image}}' | grep -i ollama || true"

sec "포트 사용 현황 (충돌 확인: 3000/3001/8000/9090/9091)"
run "ss -ltnp 2>/dev/null | awk 'NR==1 || /:(3000|3001|8000|8888|9090|9091|9400|9835|11434|30000) /'"

sec "네트워크 (HF / NGC / GitHub 접근)"
for u in https://huggingface.co https://nvcr.io/v2/ https://github.com https://pypi.org/simple/ https://registry-1.docker.io/v2/; do
  printf "  %-40s %s\n" "$u" "$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$u")"
done
} | tee "$OUT"

echo
printf "${G}스냅샷 저장: %s${N}\n" "$OUT"
echo "  → 이 파일 내용을 Claude 에게 붙여넣으면 CPU 고정(taskset)·볼륨 경로 등을 맞춰 드립니다."
echo "    (공개 저장소에는 올리지 마세요: results/phase2/*.txt 는 .gitignore 대상)"
