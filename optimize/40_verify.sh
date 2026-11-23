#!/usr/bin/env bash
# 적용 결과 점검 (읽기 전용 + 짧은 GPU 스모크 테스트). 결과는 results/optimize/verify-<ts>.txt
set -uo pipefail
source "$(dirname "$0")/lib.sh"; need_root
OUT="$OPT_DIR/../results/optimize/verify-$(date +%Y%m%d-%H%M%S).txt"; mkdir -p "$(dirname "$OUT")"
exec > >(tee "$OUT") 2>&1

step "호스트"
pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[[ "${pl%.*}" == "$POWER_CAP_W" ]] && ok "전력 제한 ${pl}W" || warn "전력 제한 ${pl}W (목표 ${POWER_CAP_W}W)"
systemctl is-enabled gpu-power-cap.service &>/dev/null && ok "부팅 시 전력 제한 서비스 enabled" || warn "gpu-power-cap.service 미설정"
[[ "$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)" == Enabled ]] && ok "persistence mode" || warn "persistence mode off"
journalctl -b -k --no-pager 2>/dev/null | grep -qi 'gpu_unlock.*-110' && bad "이번 부팅에서 unlock -110 흔적" || ok "unlock -110 흔적 없음"
env_o=$(systemctl show ollama -p Environment 2>/dev/null)
[[ "$env_o" == *OLLAMA_FLASH_ATTENTION=1* ]] && ok "Ollama FlashAttention·KV캐시: $(grep -o 'OLLAMA_KV_CACHE_TYPE=[^ ]*' <<<"$env_o")" || warn "Ollama 튜닝 미적용"
if [[ -f /etc/modprobe.d/nvidia-profiling.conf ]]; then
  v=$(cat /proc/driver/nvidia/params 2>/dev/null | awk -F': ' '/RmProfilingAdminOnly/{print $2}')
  [[ "$v" == 0 ]] && ok "일반 사용자 프로파일링 허용(적용됨)" || warn "프로파일링 허용: 설정됨, 다음 콜드 부팅(전원 OFF→ON) 후 적용"
fi
curl -sf "localhost:${GRAFANA_PORT}/api/health" >/dev/null && ok "Grafana http://<서버IP>:${GRAFANA_PORT}" || warn "Grafana 응답 없음"
curl -sf localhost:9090/-/ready >/dev/null && ok "Prometheus" || warn "Prometheus 응답 없음"

step "컨테이너 $CONTAINER"
container_running "$CONTAINER" || { bad "컨테이너 미실행"; exit 1; }
docker exec "$CONTAINER" ipython --no-banner -c "import torch;print('IPYTHON_TF32', torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark)" 2>/dev/null \
  | grep -q 'IPYTHON_TF32 True True' && ok "노트북 커널 시작 시 TF32·cudnn.benchmark 자동 적용" || warn "IPython 시작 스크립트 미적용"
docker exec -i "$CONTAINER" python - <<'PY'
import os, time, importlib, torch
def r(tag, ok, msg=""): print(("  [OK]  " if ok else "  [!!]  ")+tag+(" — "+msg if msg else ""))
r("CUDA", torch.cuda.is_available(), f"{torch.__version__} / {torch.cuda.get_device_name(0)}")
r("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF",""), os.environ.get("PYTORCH_CUDA_ALLOC_CONF","(셸 env 미설정)"))
torch.backends.cuda.matmul.allow_tf32=True
a=torch.randn(8192,8192,device="cuda",dtype=torch.bfloat16); torch.cuda.synchronize()
t=time.time()
for _ in range(20): a@a
torch.cuda.synchronize(); r("BF16 matmul", True, f"{20*2*8192**3/(time.time()-t)/1e12:.0f} TFLOPS (unlock 기대치 ~100+)")
try:
    from flash_attn import flash_attn_func
    q=torch.randn(4,2048,16,64,device="cuda",dtype=torch.bfloat16); flash_attn_func(q,q,q,causal=True); r("FlashAttention-2", True)
except Exception as e: r("FlashAttention-2", False, str(e)[:80])
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling
    l=te.Linear(1024,1024).cuda(); x=torch.randn(32,1024,device="cuda")
    with te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling()): l(x).sum().backward()
    r("FP8 (Transformer Engine)", True)
except Exception as e: r("FP8 (Transformer Engine)", False, str(e)[:80])
try:
    f=torch.compile(lambda x: torch.nn.functional.gelu(x)*2); f(torch.randn(1024,device="cuda")); r("torch.compile", True)
except Exception as e: r("torch.compile", False, str(e)[:80])
for m in ["bitsandbytes","peft","transformers","accelerate","trl","datasets","onnxruntime","nvidia.dali","torch_tensorrt","jupyterlab_nvdashboard","deepspeed"]:
    try: mod=importlib.import_module(m); r(m, True, getattr(mod,"__version__",""))
    except Exception as e: r(m, False, "미설치" if isinstance(e, ModuleNotFoundError) else str(e)[:60])
try:
    import onnxruntime as ort; p=ort.get_available_providers(); r("ORT TensorRT/CUDA EP", "CUDAExecutionProvider" in p, ",".join(p))
except Exception: pass
PY
for b in nsys ncu; do docker exec "$CONTAINER" bash -lc "command -v $b" &>/dev/null && ok "$b 있음" || warn "$b 없음"; done
docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${CONTAINER}:optimized" && ok "백업 이미지: $(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^${CONTAINER}:optimized" | head -1)" || warn "커밋 이미지 없음"
echo; ok "결과 저장: $OUT"
