#!/usr/bin/env bash
# ============================================================
# 12_container_libs.sh — 실습 컨테이너 스택 구성  (항목 6·7·8·9·15·18·30·31·32·33·35)
#   1) NGC 기존 패키지 전체를 constraints 로 고정 (torch·flash-attn·TE·numpy·jupyterlab 보호)
#   2) 실습 라이브러리 설치: transformers·accelerate·peft·trl·bitsandbytes·datasets·openai·onnxruntime-gpu…
#   3) DALI · DeepSpeed(ZeRO-Offload) · jupyterlab-nvdashboard
#   4) TF32 + cudnn.benchmark + expandable_segments 자동 주입 (ipython startup)
#   5) Nsight Systems/Compute CLI 확인 (없으면 설치 시도)
#   6) 'vLLM (venv)' 전용 커널 — NGC torch 와 격리된 가상환경 (선택: SGLang 커널)
#   7) 주피터 재시작 (nvdashboard 탭 활성화)
# 사용법: sudo bash scripts/12_container_libs.sh
#   ※ 컨테이너 재생성 시 사라지지 않게 하려면 이어서 13_commit_recreate.sh 실행
# ============================================================
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"
require_root
require_docker
container_running "$JUPYTER_CONTAINER" || die "컨테이너 '$JUPYTER_CONTAINER' 가 실행 중이 아님"
mkdir -p "$RESULTS2_DIR"

step "1) NGC 패키지 constraints 생성"
# pip list(json) 기준으로 '이름==버전' 고정 — 로컬 휠(@ file://)로 설치된 NGC 빌드도 포함
ALLOW=$(echo "$LAB_PIP_PACKAGES deepspeed nvidia-dali-cuda120" | tr ' ' '\n' | sed -E 's/[<>=!~].*//' | grep -v '^$' | tr '\n' ' ')
jexec python - "$ALLOW" >/dev/null <<'PY'
import json, re, subprocess, sys
allow = {re.sub(r"[-_.]+", "-", a).lower() for a in sys.argv[1].split()}
pkgs = json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"]))
lines = []
for p in pkgs:
    n = re.sub(r"[-_.]+", "-", p["name"]).lower()
    if n in allow:
        continue
    lines.append(f'{p["name"]}=={p["version"]}')
open("/tmp/ngc-constraints.txt", "w").write("\n".join(sorted(lines)) + "\n")
PY
docker cp "$JUPYTER_CONTAINER:/tmp/ngc-constraints.txt" "$RESULTS2_DIR/ngc-constraints.txt"
jexec pip freeze > "$RESULTS2_DIR/pip_freeze_before.txt" 2>/dev/null || true
ok "고정 패키지 $(wc -l < "$RESULTS2_DIR/ngc-constraints.txt")개 (torch=$(grep -i '^torch==' "$RESULTS2_DIR/ngc-constraints.txt" | head -1))"

PIP="pip install --no-cache-dir -c /tmp/ngc-constraints.txt"

step "2) 실습 라이브러리 설치 [6][15][35]"
# shellcheck disable=SC2086
if jexec $PIP $LAB_PIP_PACKAGES; then
  ok "설치 완료"
else
  warn "일괄 설치 실패 → 패키지별로 재시도 (호환 버전이 없는 것만 건너뜀)"
  for p in $LAB_PIP_PACKAGES; do
    # shellcheck disable=SC2086
    jexec $PIP "$p" >/dev/null 2>&1 && ok "$p" || warn "$p 설치 실패 — NGC 고정 버전과 호환되는 릴리스 없음"
  done
fi

step "3) DALI [31] · DeepSpeed [18]"
if [ "$INSTALL_DALI" = "1" ]; then
  if jexec python -c "import nvidia.dali" 2>/dev/null; then ok "DALI 이미 내장 (NGC)"
  else
    # shellcheck disable=SC2086
    jexec $PIP --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120 && ok "DALI 설치" || warn "DALI 설치 실패"
  fi
fi
if [ "$INSTALL_DEEPSPEED" = "1" ]; then
  # shellcheck disable=SC2086
  if jexec env DS_BUILD_OPS=0 $PIP deepspeed; then ok "DeepSpeed 설치 (CPUAdam 등은 첫 사용 시 JIT 빌드)"
  else warn "DeepSpeed 설치 실패 — ZeRO-Offload 실습만 건너뜀"; fi
fi

step "4) TF32 + cudnn.benchmark + 할당자 자동 주입 [7][32]"
PROFILE_DIR=$(jexec sh -c 'ipython locate profile default 2>/dev/null || (ipython profile create default >/dev/null 2>&1 && ipython locate profile default)' | tail -1)
[ -n "$PROFILE_DIR" ] || PROFILE_DIR="/root/.ipython/profile_default"
jexec mkdir -p "$PROFILE_DIR/startup"
docker cp "$DEPLOY_DIR/serving/ipython_startup_gpu_opt.py" "$JUPYTER_CONTAINER:$PROFILE_DIR/startup/00-gpu-opt.py"
ok "$PROFILE_DIR/startup/00-gpu-opt.py (새 커널부터 적용)"
jexec ipython -c "import torch; print('  확인: allow_tf32 =', torch.backends.cuda.matmul.allow_tf32, '/ cudnn.benchmark =', torch.backends.cudnn.benchmark)" 2>/dev/null | tail -1 || true

step "5) Nsight Systems / Compute [30]"
if jexec sh -c 'command -v nsys >/dev/null && command -v ncu >/dev/null'; then
  ok "nsys $(jexec nsys --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9.]+' | head -1) / ncu 내장"
else
  warn "nsys/ncu 일부 없음 → NVIDIA apt 저장소로 설치 시도"
  jexec bash -c '
    set -e
    . /etc/os-release
    REPO="ubuntu${VERSION_ID//./}"
    apt-get update -qq && apt-get install -y -qq wget gnupg >/dev/null
    wget -qO /tmp/keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/keyring.deb >/dev/null && apt-get update -qq
    command -v nsys >/dev/null || apt-get install -y -qq nsight-systems-cli >/dev/null || apt-get install -y -qq "$(apt-cache search --names-only "^nsight-systems-[0-9]" | sort | tail -1 | cut -d" " -f1)" >/dev/null
    command -v ncu  >/dev/null || apt-get install -y -qq "$(apt-cache search --names-only "^nsight-compute-[0-9]" | sort | tail -1 | cut -d" " -f1)" >/dev/null
  ' && ok "Nsight 설치 완료" || warn "Nsight 설치 실패 — 수동 설치 필요 (docs/phase2-guide.md)"
fi
if grep -q 'RmProfilingAdminOnly: 0' /proc/driver/nvidia/params 2>/dev/null; then
  ok "GPU 성능 카운터: 일반 사용자 허용"
else
  info "ncu 카운터는 기본적으로 관리자 전용 → 13 스크립트가 컨테이너에 SYS_ADMIN 권한을 부여해 해결"
  info "(드라이버 옵션 방식은 모듈 재로드=재부팅이 필요해 웜 재부팅 금지 규칙상 사용하지 않음)"
fi

venv_kernel() {   # $1=name $2=display $3=pip spec
  local name="$1" disp="$2" spec="$3" venv="/opt/venvs/$1"
  info "$disp: $venv 생성 (수 분 소요)"
  jexec bash -c "
    set -e
    python -m pip install -q --no-cache-dir uv
    python -m uv venv --python 3.12 --seed $venv >/dev/null 2>&1 || python -m uv venv --seed $venv
    python -m uv pip install --python $venv/bin/python -q '$spec' ipykernel openai
    $venv/bin/python -m ipykernel install --user --name $name --display-name '$disp' >/dev/null
  " && ok "커널 '$disp' 등록" || warn "$disp 커널 생성 실패"
}

step "6) 노트북 전용 LLM 커널 [8][9]"
if [ "$JUPYTER_VLLM_KERNEL" = "1" ]; then venv_kernel vllm "vLLM (venv)" "vllm==$JUPYTER_VLLM_VERSION"; else info "vLLM 커널: 건너뜀"; fi
if [ "$JUPYTER_SGLANG_KERNEL" = "1" ]; then venv_kernel sglang "SGLang (venv)" "sglang[all]==$JUPYTER_SGLANG_VERSION"; else info "SGLang 커널: 건너뜀 (서빙은 17_serve.sh sglang 컨테이너로)"; fi

step "7) 템플릿 노트북 3종 배포"
NB_ROOT="$LAB_NOTEBOOK_DIR"
if [ -z "$NB_ROOT" ]; then
  # 주피터 실행 인자(--notebook-dir / root_dir) → 바인드 마운트 → WorkingDir 순으로 탐지
  NB_ROOT=$(jexec sh -c "ps -eo args | grep -oE -- '--(notebook-dir|ServerApp.root_dir|NotebookApp.notebook_dir)[= ][^ ]+' | head -1 | sed -E 's/^[^= ]+[= ]//'" 2>/dev/null || true)
  [ -n "$NB_ROOT" ] || NB_ROOT=$(docker inspect -f '{{range .Mounts}}{{if eq .Type "bind"}}{{.Destination}}{{println}}{{end}}{{end}}' "$JUPYTER_CONTAINER" | grep -vE '^/(tmp|dev|etc|var)' | head -1)
  [ -n "$NB_ROOT" ] || NB_ROOT=$(docker inspect -f '{{.Config.WorkingDir}}' "$JUPYTER_CONTAINER")
  [ -n "$NB_ROOT" ] || NB_ROOT="/workspace"
fi
jexec mkdir -p "$NB_ROOT/gpu-lab-templates"
for nb in "$DEPLOY_DIR"/notebooks/*.ipynb; do
  if jexec test -f "$NB_ROOT/gpu-lab-templates/$(basename "$nb")"; then
    info "$(basename "$nb"): 이미 있음 — 학생 수정본 보호를 위해 덮어쓰지 않음 (새 버전은 .new 로 복사)"
    docker cp "$nb" "$JUPYTER_CONTAINER:$NB_ROOT/gpu-lab-templates/$(basename "$nb").new"
  else
    docker cp "$nb" "$JUPYTER_CONTAINER:$NB_ROOT/gpu-lab-templates/"
  fi
done
ok "$NB_ROOT/gpu-lab-templates/ (JupyterLab 파일 탐색기에서 열기)"

step "8) 결과 기록 + 주피터 재시작 [33]"
jexec pip freeze > "$RESULTS2_DIR/pip_freeze_after.txt" 2>/dev/null || true
diff <(sed 's/==.*//' "$RESULTS2_DIR/pip_freeze_before.txt" | sort) <(sed 's/==.*//' "$RESULTS2_DIR/pip_freeze_after.txt" | sort) | grep '^>' | sed 's/^> /  + /' | head -60 || true
TORCH_NOW=$(jexec python -c "import torch;print(torch.__version__)")
grep -q "torch==$TORCH_NOW" "$RESULTS2_DIR/ngc-constraints.txt" && ok "NGC torch 유지 확인: $TORCH_NOW" || bad "torch 버전 변경 감지: $TORCH_NOW — 06 롤백 대신 13 의 백업 컨테이너로 복구"
if confirm "주피터 컨테이너를 재시작할까요? (실행 중 커널 종료됨 — nvdashboard 탭 활성화에 필요)"; then
  docker restart "$JUPYTER_CONTAINER" >/dev/null && ok "재시작 완료"
else
  info "나중에: docker restart $JUPYTER_CONTAINER"
fi
printf "\n${G}12 완료${N} → 다음: sudo bash scripts/13_commit_recreate.sh  (HF 캐시 볼륨 + 이미지 커밋)\n"
