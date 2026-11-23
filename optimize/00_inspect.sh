#!/usr/bin/env bash
# ============================================================
# 00_inspect.sh — 호스트·컨테이너 현황 조사 (읽기 전용, 아무것도 바꾸지 않음)
# 결과는 화면 출력 + results/optimize/inspect-<시각>.txt 저장
# 사용법: sudo bash optimize/00_inspect.sh
# ============================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT_DIR="$OPT_DIR/../results/optimize"; mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/inspect-$(date +%Y%m%d-%H%M%S).txt"
exec > >(tee "$OUT") 2>&1

step "호스트"
echo "  OS: $(. /etc/os-release; echo "$PRETTY_NAME") / 커널 $(uname -r)"
echo "  CPU: $(nproc)코어 — $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')"
free -h | sed 's/^/  /'
df -h / /var/lib/docker 2>/dev/null | sed 's/^/  /'

step "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,power.max_limit,persistence_mode,clocks.max.sm,temperature.gpu --format=csv | sed 's/^/  /'
journalctl -k -b 2>/dev/null | grep -m3 'gpu_unlock' | sed 's/^/  /' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/  /'
[ -f /etc/systemd/system/gpu-power-cap.service ] && echo "  전력 캡 유닛: 설치됨" || echo "  전력 캡 유닛: 없음"
pgrep -x nvidia-cuda-mps-control >/dev/null && echo "  MPS: 실행 중" || echo "  MPS: 꺼짐"
grep -h . /etc/modprobe.d/nvidia*.conf 2>/dev/null | sed 's/^/  modprobe: /'

step "Ollama"
if systemctl list-unit-files 2>/dev/null | grep -q '^ollama.service'; then
  echo "  서비스: $(systemctl is-active ollama)"
  systemctl show ollama -p Environment | sed 's/^/  /'
else
  echo "  systemd 서비스 아님 (도커 등 다른 방식일 수 있음)"
fi

step "Docker"
docker version --format '  엔진 {{.Server.Version}}' 2>/dev/null
docker ps --format '  {{.Names}}\t{{.Image}}\t{{.Ports}}'
if container_running; then
  step "컨테이너 $CONTAINER 마운트"
  docker inspect "$CONTAINER" --format '{{range .Mounts}}  {{.Type}}  {{.Source}} -> {{.Destination}} (rw={{.RW}}){{"\n"}}{{end}}'
  echo "  IPC 모드: $(docker inspect "$CONTAINER" --format '{{.HostConfig.IpcMode}}') / shm: $(docker inspect "$CONTAINER" --format '{{.HostConfig.ShmSize}}')"
  echo "  CPU 제한: cpuset=$(docker inspect "$CONTAINER" --format '{{.HostConfig.CpusetCpus}}') / 메모리=$(docker inspect "$CONTAINER" --format '{{.HostConfig.Memory}}')"

  step "컨테이너 $CONTAINER 내부 소프트웨어"
  docker exec -i "$CONTAINER" bash -s <<'IN'
python - <<'PY'
import importlib, subprocess, shutil
def v(m):
    try:
        mod = importlib.import_module(m); return getattr(mod, "__version__", "있음")
    except Exception as e: return f"없음 ({type(e).__name__})"
import torch
print(f"  torch {torch.__version__} / CUDA {torch.version.cuda} / cuDNN {torch.backends.cudnn.version()}")
print(f"  GPU {torch.cuda.get_device_name(0)} / sm_{''.join(map(str,torch.cuda.get_device_capability(0)))}")
for m in ["flash_attn","transformer_engine","torch_tensorrt","tensorrt","onnxruntime","nvidia.dali",
          "transformers","accelerate","peft","trl","bitsandbytes","datasets","deepspeed","jupyterlab_nvdashboard","jupyterlab"]:
    print(f"  {m:24s} {v(m)}")
for b in ["nsys","ncu","jupyter"]:
    print(f"  {b:24s} {shutil.which(b) or '없음'}")
PY
ls /etc/pip/constraint.txt 2>/dev/null && echo "  NGC pip 제약 파일 있음"
ps -eo user,args | grep -m1 '[j]upyter' | sed 's/^/  주피터 프로세스: /'
IN
else
  warn "컨테이너 $CONTAINER 가 실행 중이 아님"
fi
echo; ok "조사 결과 저장: $OUT"
