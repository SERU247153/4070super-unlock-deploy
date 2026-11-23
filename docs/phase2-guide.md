# 2단계 가이드 — 서버 최적화 · 실습 환경 · LLM 서빙

1단계(텐서코어 언록, BF16 132.9 TFLOPS)를 마친 서버에 **실습 스택 · 호스트 튜닝 · 모니터링 · LLM 서빙**을 올리는 키트입니다.
모든 스크립트는 `config.phase2.sh` 하나로 설정하며, **재부팅을 하지 않습니다**(웜 재부팅 금지 규칙 유지).

## 실행 순서

```bash
cd /opt/gpu-unlock-deploy && git pull            # 서버에서 최신 키트 받기

sudo bash scripts/11_snapshot.sh                 # ① 사전 점검 (읽기 전용) → 결과를 공유하면 CPU 고정값 등 맞춤 조정
sudo bash scripts/12_container_libs.sh           # ② 실습 라이브러리 · TF32 주입 · vLLM 커널 · Nsight · 노트북 배포
sudo bash scripts/13_commit_recreate.sh          # ③ 이미지 커밋 + HF 캐시 볼륨 + SYS_ADMIN (1~2분 중단)
sudo bash scripts/14_host_tune.sh                # ④ 전력 캡 180W 자동 · Ollama FA/KV캐시 · taskset
sudo bash scripts/16_monitoring.sh up            # ⑤ Grafana 대시보드 (포트 3001)
sudo bash scripts/17_serve.sh vllm               # ⑥ LLM 서빙 (OpenAI API)
sudo bash scripts/18_serve_bench.sh --all        # ⑦ 모드별 벤치마크 비교표

# 다인원 실습 시간에만
sudo bash scripts/15_mps.sh on && sudo bash scripts/13_commit_recreate.sh --mps

# 선택: TensorRT-LLM
sudo bash scripts/19_trtllm.sh check && sudo bash scripts/19_trtllm.sh build && sudo bash scripts/17_serve.sh trtllm

# 원복
sudo bash scripts/20_phase2_rollback.sh
```

## 항목별 구현 위치

### 컨테이너 환경
| # | 항목 | 구현 |
|---|---|---|
| 1 | HF 모델 캐시 볼륨 | `13` — `hf_cache` 볼륨을 `HF_HOME` 으로 마운트. vLLM·SGLang·TRT-LLM 도 같은 볼륨 공유 → 모델 1회 다운로드 |
| 2 | 이미지 커밋 | `13` — `jupyter_ai-opt:<날짜>` 커밋 후 같은 구성으로 재생성. 기존 컨테이너는 `_backup_<시각>` 으로 정지 보관 |
| 6 | 실습 라이브러리 | `12` — NGC 패키지 전체를 constraints 로 고정한 채 transformers·accelerate·peft·trl·bitsandbytes·datasets·openai 설치 |
| 7 | TF32 + cudnn.benchmark 자동 주입 | `12` — ipython startup `00-gpu-opt.py` (끄기: `GPU_OPT_DISABLE=1`) |
| 8 | vLLM | 서빙은 `17` 의 전용 컨테이너, 노트북 직접 실행은 `vLLM (venv)` 커널 (NGC torch 와 격리) |
| 9 | SGLang | `17_serve.sh sglang` (선택: `JUPYTER_SGLANG_KERNEL=1` 로 커널도 생성) |

### 호스트
| # | 항목 | 구현 |
|---|---|---|
| 3 | 부팅 시 전력 캡 180W | `14` — `gpu-power-cap.service` (설정 `/etc/gpu-lab/power.conf`) |
| 4 | MPS | `15` — `nvidia-mps.service`, on/off/enable/disable. 컨테이너는 `13 --mps` 로 파이프 공유 |
| 5 | DCGM + Grafana | `16` — nvidia_gpu_exporter(기본) + dcgm-exporter(시도) + Prometheus + Grafana, 대시보드 3종 자동 등록 |
| — | Ollama FA · KV 캐시 | `14` — systemd drop-in: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` |
| — | taskset | `14` / `17` — `OLLAMA_CPUSET` (CPUAffinity), `SERVE_CPUSET` · `JUPYTER_CPUSET` (`--cpuset-cpus`) |

### 학습 기법 (노트북 `01`, `02`)
| # | 기법 | 위치 |
|---|---|---|
| 10 | TF32 matmul | 01 §1 |
| 11 | BF16 autocast | 01 §2, 02 §4 |
| 12 | FlashAttention 2 | 01 §3, 02 §1 (`attn_implementation="flash_attention_2"`) |
| 13 | FP8 (Transformer Engine) | 01 §4 |
| 14 | QLoRA / DoRA | 02 §1–4 (`USE_DORA=True`) |
| 15 | 8-bit Adam + fused | 01 §5, 02 §4 (Paged 8-bit AdamW) |
| 16 | 그래디언트 체크포인팅 | 01 §5, 02 §2 |
| 17 | torch.compile | 01 §5 |
| 18 | ZeRO-Offload | 02 부록 B (DeepSpeed ZeRO-2 + CPU offload) |
| 19 | datasets 디스크 캐싱 | 01 §8 |

### 서빙 (`17` · `18` · 노트북 `03`)
| # | 기법 | 위치 |
|---|---|---|
| 20 | vLLM OpenAI API | `17_serve.sh vllm`, 03 §1–2 |
| 21 | Speculative Decoding | `17_serve.sh vllm-spec` (n-gram), 03 §3 |
| 22 | FlashInfer | `17_serve.sh vllm-fi`, SGLang 기본 백엔드 |
| 23 | AWQ/GPTQ 4비트 · 13B급 | `vllm` (AWQ), `vllm-gptq`, `vllm-big` (14B AWQ) |
| 24 | Structured Generation | 03 §4 (JSON 스키마 · guided_choice · 도구 호출), `18` 의 JSON 유효율 |

### 실습 도구
| # | 기법 | 위치 |
|---|---|---|
| 30 | Nsight Systems / Compute | `12` 설치 확인, `13` SYS_ADMIN, 01 §9 |
| 31 | DALI + pin_memory + num_workers | 01 §6 |
| 32 | PYTORCH_CUDA_ALLOC_CONF | `12` 자동 주입(expandable_segments), 01 §7 |
| 33 | jupyterlab-nvdashboard | `12` 설치 → 왼쪽 사이드바 GPU Dashboards |
| 34 | Torch-TensorRT | 03 §6 |
| 35 | ONNX Runtime (TensorRT EP) | `12` 설치(onnxruntime-gpu 1.19.2), 03 §7 |
| 36 | Gradient accumulation | 01 §5, 02 §4 |
| 37 | GradScaler + autocast | 01 §5 |
| 38 | CUDA 스트림 프리패치 | 01 §6 (`CUDAPrefetcher`) |

## 설계 결정과 이유

- **vLLM·SGLang 을 주피터 컨테이너에 pip 설치하지 않음** — 둘 다 자체 torch 버전을 요구해 NGC 전용 torch(FlashAttention·TE 포함)를 교체해 버립니다. 서빙은 별도 컨테이너, 노트북 실습은 venv 커널로 분리했습니다.
- **constraints 고정 설치** — `12` 는 설치 전 `pip list` 전체를 `이름==버전` 으로 고정하고, 지정한 실습 패키지만 추가를 허용합니다. NGC torch 가 바뀌면 경고를 출력합니다.
- **서빙 컨테이너 공개 범위** — API 는 `127.0.0.1:8000` 과 내부 도커 네트워크(`gpu-lab`)에만 열립니다. 노트북은 `http://llm-serve:8000/v1` 로 접근합니다.
- **Ollama 전환식 운영** — 12GB 에서 vLLM 이 KV 캐시를 충분히 확보하려면 GPU 를 독점해야 합니다. 서빙 중에는 Ollama 를 내리고 `stop` 할 때 다시 올립니다.
- **Nsight Compute 권한** — 드라이버 옵션(`NVreg_RestrictProfilingToAdminUsers=0`)은 모듈 재로드, 즉 재부팅이 필요합니다. 대신 컨테이너에 `SYS_ADMIN` 권한을 줍니다.
- **DCGM** — GeForce 는 DCGM 공식 지원 대상이 아니어서 일부 지표만 나오거나 기동이 실패할 수 있습니다. 그래서 nvidia-smi 기반 exporter 가 기본이고, DCGM 은 되면 쓰는 방식입니다.

## 운영 주의

- 🔴 **웜 재부팅 금지 규칙은 그대로** — 2단계 스크립트는 재부팅하지 않습니다. 커널 업데이트 등은 1단계 문서의 절차를 따르세요.
- 🔴 **공개 저장소** — 서버 IP·HF 토큰·Grafana 비밀번호를 커밋하지 마세요. 토큰은 `sudo HF_TOKEN=... bash ...` 로만 전달하고, 비밀번호는 `/etc/gpu-lab/grafana.env`(root 전용)에 있습니다.
- MPS 아래에서는 한 클라이언트의 치명적 오류가 같은 MPS 서버의 다른 클라이언트에 영향을 줄 수 있습니다. 실습 시간에만 켜는 것을 권장합니다.
- `03_tune.sh` 를 다시 실행하면 전력이 `POWER_LIMIT_MODE`(max)로 바뀝니다. 180W 운영이면 `config.sh` 도 180 으로 맞추세요.
- `results/phase2/` 의 스냅샷·inspect 파일은 `.gitignore` 대상입니다. 서빙 벤치 결과(`results/serving/*.json`)는 커밋 대상입니다.

## 첫 실행 후 공유해 주면 좋은 것

1. `11_snapshot.sh` 출력 — CPU 토폴로지(taskset 분할), 주피터 볼륨 구조, NGC 패키지 목록
2. `12` 에서 `[WARN]` 이 뜬 패키지 — 호환 버전으로 고정
3. `17_serve.sh` 첫 기동 로그 — 이미지 태그·플래그가 버전과 맞는지 확인
4. `19_trtllm.sh build` 로그 — TRT-LLM 버전별 예제 경로·옵션 보정
