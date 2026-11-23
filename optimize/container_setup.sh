#!/usr/bin/env bash
# ============================================================
# container_setup.sh — 컨테이너 안에서 실행됨 (20_container.sh 가 호출)
# 핵심 원칙: NGC 가 빌드한 torch / FlashAttention / TransformerEngine / TensorRT 는 절대 교체하지 않는다
# ============================================================
set -uo pipefail
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; N='\033[0m'
ok()   { printf "${G}  [OK]${N}   %s\n" "$*"; }
warn() { printf "${Y}  [WARN]${N} %s\n" "$*"; }
bad()  { printf "${R}  [FAIL]${N} %s\n" "$*"; }
: "${CACHE_ROOT:=/root/.cache}" "${TEMPLATE_DST:=/root/optimize-templates}"
: "${PIP_PROFILE:=pinned}" "${INSTALL_ORT:=1}" "${INSTALL_DALI:=1}" "${INSTALL_DEEPSPEED:=0}"
mkdir -p /opt/optimize

# ---- 주피터 실행 사용자 ----
JUSER=$(ps -eo user=,args= | grep -m1 '[j]upyter' | awk '{print $1}')
JUSER=${JUSER:-root}
JHOME=$(getent passwd "$JUSER" | cut -d: -f6); JHOME=${JHOME:-/root}
echo "  주피터 사용자: $JUSER ($JHOME)"

# ---- 1) pip 제약: NGC 핵심 패키지 고정 ----
CONS=/opt/optimize/constraints.txt
pip list --format=freeze 2>/dev/null \
  | grep -iE '^(torch|torchvision|torchaudio|triton|flash[-_]attn|transformer[-_]engine|torch[-_]tensorrt|tensorrt|apex|numpy|pytorch[-_]triton)==' >"$CONS"
[ -f /etc/pip/constraint.txt ] && cat /etc/pip/constraint.txt >>"$CONS"
ok "핵심 패키지 $(wc -l <"$CONS")개 고정 ($CONS)"
PIP="pip install --no-cache-dir -c $CONS"

# ---- 2) 실습 표준 스택 ----
if [ "$PIP_PROFILE" = "latest" ]; then
  PKGS="transformers accelerate peft trl bitsandbytes datasets"
else
  # torch 2.4 (NGC 24.07) 와 호환이 검증된 조합
  PKGS="transformers==4.46.3 accelerate==1.1.1 peft==0.13.2 trl==0.12.2 bitsandbytes==0.44.1 datasets==3.1.0"
fi
FAILED=""
for p in $PKGS jupyterlab-nvdashboard; do
  if $PIP "$p" >/tmp/pip_$$.log 2>&1; then ok "설치: $p"; else bad "설치 실패: $p (마지막 로그 ↓)"; tail -3 /tmp/pip_$$.log; FAILED="$FAILED $p"; fi
done

if [ "$INSTALL_ORT" = "1" ]; then
  pip show onnxruntime >/dev/null 2>&1 && pip uninstall -y onnxruntime >/dev/null 2>&1
  $PIP onnxruntime-gpu==1.19.2 >/tmp/pip_$$.log 2>&1 && ok "설치: onnxruntime-gpu 1.19.2" || { bad "onnxruntime-gpu 설치 실패"; FAILED="$FAILED onnxruntime-gpu"; }
fi
if [ "$INSTALL_DALI" = "1" ]; then
  if python -c "import nvidia.dali" 2>/dev/null; then ok "DALI 이미 있음"
  else $PIP --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120 >/tmp/pip_$$.log 2>&1 && ok "설치: DALI" || { bad "DALI 설치 실패"; FAILED="$FAILED dali"; }; fi
fi
if [ "$INSTALL_DEEPSPEED" = "1" ]; then
  DS_BUILD_OPS=0 $PIP deepspeed==0.15.4 >/tmp/pip_$$.log 2>&1 && ok "설치: DeepSpeed (ZeRO-Offload, 연산자는 첫 사용 시 JIT 빌드)" || { bad "DeepSpeed 설치 실패"; FAILED="$FAILED deepspeed"; }
fi
rm -f /tmp/pip_$$.log

# torch 가 바뀌지 않았는지 확인
python - <<'PY'
import torch, sys
print(f"  torch {torch.__version__} 유지 확인 / CUDA 사용 가능={torch.cuda.is_available()}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY
[ $? -eq 0 ] && ok "NGC torch 정상" || bad "torch CUDA 확인 실패 — 즉시 점검 필요"

# ---- 3) 캐시 위치 ----
mkdir -p "$CACHE_ROOT/huggingface/datasets" "$CACHE_ROOT/torch"
chown -R "$JUSER" "$CACHE_ROOT" 2>/dev/null || true

# ---- 4) 성능 설정 자동 주입 ----
STARTUP_SRC=/opt/optimize/00-perf.py
cat >"$STARTUP_SRC" <<PY
# optimize kit — 모든 노트북 커널 시작 시 자동 실행 (끄려면 이 파일 삭제)
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 메모리 조각화·OOM 완화
os.environ.setdefault("HF_HOME", "$CACHE_ROOT/huggingface")                   # 모델 재다운로드 방지
os.environ.setdefault("HF_DATASETS_CACHE", "$CACHE_ROOT/huggingface/datasets")
os.environ.setdefault("TORCH_HOME", "$CACHE_ROOT/torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True   # FP32 matmul 을 TF32 텐서코어로 (언록 상태에서 약 2배)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True          # 입력 크기가 고정된 학습에서 최적 커널 자동 선택
    torch.set_float32_matmul_precision("high")
except Exception as e:
    print("[optimize] torch 설정 생략:", e)
PY
for H in /root "$JHOME"; do
  D="$H/.ipython/profile_default/startup"; mkdir -p "$D"
  cp "$STARTUP_SRC" "$D/00-perf.py"
  chown -R "$(stat -c %U "$H")" "$H/.ipython" 2>/dev/null || true
done
ok "IPython startup 주입 (/root, $JHOME)"

cat >/etc/profile.d/optimize.sh <<SH
# optimize kit — 터미널/스크립트용 환경 변수
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$CACHE_ROOT/huggingface
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TORCH_HOME=$CACHE_ROOT/torch
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
SH
grep -q 'profile.d/optimize.sh' /root/.bashrc 2>/dev/null || echo '[ -f /etc/profile.d/optimize.sh ] && . /etc/profile.d/optimize.sh' >>/root/.bashrc
ok "터미널 환경 변수 등록 (/etc/profile.d/optimize.sh)"

# ---- 5) 템플릿 노트북 ----
mkdir -p "$TEMPLATE_DST"
cp -n /opt/optimize/templates/*.ipynb "$TEMPLATE_DST"/ 2>/dev/null || true
chown -R "$JUSER" "$TEMPLATE_DST" 2>/dev/null || true
ok "템플릿 3종 → $TEMPLATE_DST"

[ -n "$FAILED" ] && warn "실패 항목:$FAILED (나머지는 정상 적용)" || ok "모든 패키지 설치 성공"
