# optimize/ — GPU 서버 최적화 키트 (호스트 + jupyter_ai 컨테이너)

Rocky 9 · RTX 4070 SUPER 12GB · gpu-unlock 적용 서버용. **어떤 스크립트도 재부팅하지 않습니다**
(웜 리부트 시 unlock -110 실패 위험). 설정은 `opt.conf` 한 곳에서 바꿉니다.

## 실행 순서 (서버에서 root)
```bash
cd ~/gpu-unlock-deploy
sudo bash optimize/00_inspect.sh     # 현황 조사만 (변경 없음) → results/optimize/inspect-*.txt
sudo bash optimize/10_host.sh        # 호스트: 전력제한 180W 영구화, Ollama 튜닝, 프로파일링 허용, Grafana
sudo bash optimize/20_container.sh   # 컨테이너: 라이브러리, TF32 자동 주입, 캐시, 템플릿, 이미지 백업
sudo bash optimize/40_verify.sh      # 점검 → results/optimize/verify-*.txt
```
필요할 때만:
```bash
sudo bash optimize/30_serving.sh vllm-up     # vLLM (AWQ 7B, OpenAI 호환 :8000, JSON 구조화 출력)
sudo bash optimize/30_serving.sh down        # 끄면 Ollama 가 다시 VRAM 사용
sudo bash optimize/15_mps.sh enable          # MPS (아래 '보류' 참고)
sudo bash optimize/99_rollback.sh            # 되돌리기
```
Grafana: `http://<서버IP>:3001` (admin / 비밀번호는 `/opt/gpu-monitor/.env`). 대시보드 14574 자동 등록.

## 항목별 처리 (요청 목록 1–38 기준)
| 구분 | 항목 | 처리 |
|---|---|---|
| **적용** | 전력 제한 180W 부팅 시 자동 + persistence | `10_host.sh` → `gpu-power-cap.service` |
| | Ollama FlashAttention + KV 캐시 q8_0 | systemd drop-in |
| | 모니터링 (DCGM 대신) | nvidia_gpu_exporter + Prometheus + Grafana. DCGM 은 GeForce 공식 미지원 |
| | TF32 · cudnn.benchmark · matmul precision 자동 | IPython 시작 스크립트 (모든 노트북 커널) |
| | PYTORCH_CUDA_ALLOC_CONF=expandable_segments | 시작 스크립트 + /etc/profile.d |
| | HF 모델/데이터셋 캐시 영구화 | 기존 영구 마운트 아래 `.cache` (컨테이너 재생성 불필요) |
| | transformers·accelerate·peft·trl·bitsandbytes·datasets | NGC torch 를 고정(constraints)한 채 호환 버전 설치 |
| | ONNX Runtime GPU, DALI, nvdashboard | 설치 |
| | docker commit 백업 | `jupyter_ai:optimized-YYYYMMDD` (디스크 30GB 이상일 때) |
| | vLLM / SGLang | 별도 컨테이너, 필요할 때만 기동 (`30_serving.sh`) |
| **템플릿 제공** | DataLoader·CUDA 스트림·BF16·GradScaler·fused AdamW·accumulation·checkpointing·torch.compile·Nsight | `templates/01_training_optimization.ipynb` |
| | QLoRA / DoRA / 8-bit paged Adam | `templates/02_llm_qlora_finetune.ipynb` |
| | Torch-TensorRT · ONNX→TensorRT EP · vLLM 구조화 출력 · AWQ | `templates/03_inference_deploy.ipynb` |
| | FlashAttention-2, FP8(Transformer Engine) | NGC 이미지 내장 → `40_verify.sh` 에서 동작 확인 |
| **다음 콜드 부팅 후** | 일반 사용자 ncu 프로파일링 | modprobe 옵션 기록됨. 전원 OFF→ON 시 반영 (재부팅 금지 원칙) |
| **보류 (물어보면 안내)** | MPS | 컨테이너를 `-v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host` 로 재생성해야 효과. 단일 사용자에선 이득 작음 |
| | CPU 코어 고정(taskset/cpuset) | 컨테이너 재생성 필요, 현 부하에선 이득 불명확 |
| | HF 캐시 전용 볼륨 | 영구 마운트가 없을 때만 재생성 필요 (00_inspect 결과로 판단) |
| | DeepSpeed ZeRO-Offload | 기본 off (`INSTALL_DEEPSPEED=1`). 12GB 에선 QLoRA 가 대체로 더 실용적 |
| | Speculative decoding | vLLM 0.6.3 지원하나 12GB 에 드래프트 모델 동시 적재가 빠듯 → 7B AWQ 단독 먼저 |
| | SGLang 이미지 태그 | `latest` 사용 중. 실행해 보고 맞는 버전으로 `opt.conf` 고정 권장 |

## 주의
- 12GB VRAM 은 Ollama · Jupyter · vLLM 이 공유합니다. `30_serving.sh` 는 Ollama 모델을 먼저 내리고 여유 VRAM 을 확인합니다. 노트북에 모델이 올라가 있으면 커널을 종료하세요.
- 이 저장소는 공개입니다. 비밀번호·토큰을 파일에 넣지 마세요 (Grafana 비밀번호는 서버의 `/opt/gpu-monitor/.env` 에만 생성).
