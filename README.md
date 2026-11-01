# RTX 4070 SUPER — gpu-unlock 배포 키트 (Rocky Linux 9)

[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) 을 주피터 노트북 서버(Rocky Linux 9, Docker)에
안전하게 적용하기 위한 스크립트 묶음입니다.

- 언록은 **GPU 칩 레지스터 단위**로 걸리므로, 도커 컨테이너 안의 PyTorch 등 모든 워크로드에 자동 적용됩니다.
- GPU 자체에는 아무 흔적이 남지 않으며, 롤백 스크립트로 언제든 원복 가능합니다.

---

## 준비물 (서버 사전 조건)

| 항목 | 상태 |
|---|---|
| NVIDIA 드라이버 + `nvidia-smi` | 주피터 서버라면 이미 설치돼 있음 |
| Docker + nvidia-container-toolkit | 벤치마크용 (`--gpus all`) |
| Secure Boot | **꺼져 있어야 함** (BIOS) |
| 네트워크 | github.com, download.nvidia.com 접근 가능 |
| 콘솔 접근 | 첫 재부팅 시 원격 콘솔(IPMI 등) 접근 가능한 상태 권장 |

## 서버에 배포 (git clone)

```bash
# 서버에서 (private 저장소 — 최초 클론 시 GitHub 로그인/PAT 필요)
git clone https://github.com/SERU247153/4070super-unlock-deploy.git /opt/gpu-unlock-deploy
cd /opt/gpu-unlock-deploy
```

> SSH 대신 수동 전송이 필요하면: `scp -r gpu-unlock-deploy 사용자@서버:/opt/`

---

## 실행 순서

| 단계 | 명령 | 시점 |
|---|---|---|
| 1. 사전 점검 | `sudo bash scripts/01_preflight.sh --fix` | 재부팅 전 |
| 2. 기준 벤치 | `sudo bash scripts/04_benchmark.sh before` | 재부팅 전 |
| 3. 언록 설치 | `sudo bash scripts/02_install_unlock.sh` | 재부팅 전 |
| 4. 튜닝 | `sudo bash scripts/03_tune.sh` | 재부팅 전 (전력/클럭은 즉시, 드라이버 옵션은 재부팅 시 반영) |
| 5. **재부팅** | `sudo reboot` | ⭐ 언록 적용 시점 |
| 6. 검증 | `sudo bash scripts/05_verify.sh` | 재부팅 후 |
| 7. 성능 확인 | `sudo bash scripts/04_benchmark.sh after` | 재부팅 후 |

> 3→4→5를 묶어서 하면 재부팅 1회로 언록 + 드라이버 옵션이 함께 적용됩니다.

## 기대값 (4070 SUPER, 8192³ GEMM 기준)

| 항목 | 언록 전 | 언록 후 |
|---|---|---|
| FP16 / BF16 (FP32 축적) | ~70 TFLOPS | **~130~145 TFLOPS** (전력/클럭에 따라) |

- `04_benchmark.sh after` 에서 before 대비 **~2배**가 나오면 성공입니다.
- LLM 프리필 +25~55%, 이미지 생성 +30~55% 정도가 실사용 기대치입니다.
- 양자화(INT8/INT4/GGUF) 모델이나 토큰 디코드 속도는 거의 변하지 않습니다.

## 설정 변경

- `config.sh` 만 고치면 됩니다:
  - `GPU_UNLOCK_TARGETS`: `all` 또는 특정 카드(`0000:01:00.0`)
  - `POWER_LIMIT_MODE`: `max`(권장) 또는 특정 W 값
  - `CLOCK_LOCK`: `off` 또는 `"2100,2730"` 형식
  - `BENCH_MATRIX_SIZE`: VRAM 부족 시 6144 로 낮추면 됨

## 주의사항

1. **첫 재부팅**은 반드시 콘솔 접근 가능한 상태에서. (모듈이 실패해도 부팅은 정상 진행되도록 설계돼 있으나, 원격 서버는 보수적으로)
2. **Secure Boot**가 켜져 있으면 미서명 DKMS 모듈이 로드되지 않아 언록이 조용히 무시됩니다. `01_preflight.sh` 가 걸러냅니다.
3. **전원 완전 차단/재부팅** 시 언록은 리셋되지만, 부팅 때 자동으로 다시 적용됩니다.
4. **최대절전(hibernation)** 은 미지원 — 서버라면 거의 쓸 일 없음.
5. NVIDIA 드라이버 업데이트해도 언록은 유지됩니다. 커널 업데이트 시 DKMS가 자동 재빌드합니다.
6. Rocky 9는 dkms 가 **EPEL** 저장소에 있으므로, 01 스크립트가 `epel-release` 를 먼저 설치합니다.

## 롤백

```bash
sudo bash scripts/06_rollback.sh   # 확인 질문에 yes
sudo reboot
```

부트 옵션 제거 → 패키지 제거 → 전력/클럭 기본값 복원 → 드라이버 옵션 제거까지 한 번에 수행합니다.
