# 00-gpu-opt.py — gpu-unlock-deploy 2단계: 모든 노트북 커널에 자동 주입되는 GPU 기본 최적화
#   [7]  TF32 matmul + cuDNN TF32 + cudnn.benchmark  (언록된 텐서코어 경로 사용)
#   [32] PYTORCH_CUDA_ALLOC_CONF=expandable_segments  (메모리 조각화·OOM 완화)
# 끄기: 노트북 실행 전 환경변수 GPU_OPT_DISABLE=1  /  셀에서 torch.backends.cuda.matmul.allow_tf32=False
import os
import sys

_is_venv_kernel = "/opt/venvs/" in sys.prefix   # vLLM/SGLang 전용 커널은 자체 할당자를 쓰므로 건드리지 않음

if os.environ.get("GPU_OPT_DISABLE") != "1" and not _is_venv_kernel:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("[gpu-opt] TF32 matmul/cuDNN ON · cudnn.benchmark ON · "
              "alloc=expandable_segments  (끄기: GPU_OPT_DISABLE=1)")
    except Exception as e:  # torch 가 없는 커널 등
        print(f"[gpu-opt] 건너뜀: {e}")
