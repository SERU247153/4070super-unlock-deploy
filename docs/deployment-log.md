# 실전 배포 일지

**대상**: 200.168.18.98 (Rocky Linux 9.8, RTX 4070 SUPER AD104 `10de:2783`, 드라이버 610.57.04)
**일자**: 2026-11-01

---

## 배포 전 서버 상태

| 항목 | 값 |
|---|---|
| 커널 | 5.14.0-687.61.1.el9_8.x86_64 |
| GPU | RTX 4070 SUPER 12GB, SM 56, 최대 클럭 3120 MHz, 전력 220W(max 245W) |
| Secure Boot | Disabled |
| ReBAR | BAR1 16GB 활성 |
| Docker | 29.9.1 — jupyter_ai(NGC pytorch 24.07, torch 2.4 + FlashAttention 2.4.2), open-webui, portainer |
| Ollama | active (11434) |
| gpu-unlock | 미설치 |

## 1차 측정 (언록 전, 클럭 고정 2505 MHz)

| 경로 | 측정값 | 비고 |
|---|---|---|
| fp16 · FP32 누적 | **70.3 TFLOPS** | 잠김 상태 |
| bf16 · FP32 누적 | **69.1 TFLOPS** | 잠긴 이론치(71.8@2505MHz)의 96% |
| fp32(CUDA 코어) 참조 | 22.8 TFLOPS | 이론 29.7의 76.7% |

## 설치 과정에서 발견·해결된 이슈 2건

1. **rpm 4.16 매크로 제한** — `%global nv` 거부 → `%nvv` 자동 치환 패치 (02 스크립트 1.5절)
2. **RHEL 9.8 aperture 백포트** — 6.0 API가 5.14에 백포트됨 → `RHEL_RELEASE_CODE` 분기 보강 패치 (02 스크립트 1.6절)
3. (부수) /boot 공간 부족 — installonly_limit=2 로 커널 정리

## 1차 언록 시도 → 실패 (논콜드 카드)

- 셀렉터 기록(SS0=0x88888888, SS1=0x8)까지는 성공
- FLR 후 GSP 재기동 대기에서 `-110 ETIMEDOUT` (gpu-unlock 이슈 #23과 동일 패턴)
- 결과: nvidia 드라이버 통신 불가 → jupyter/ollama GPU 워크로드 중단

## 복구 (검증된 절차, 원격 5분)

```
grubby --update-kernel=ALL --remove-args="gpu_unlock.targets=all" → systemctl reboot
```

- 웜 재부팅만으로 GPU 완전 복구 확인 (nvidia-smi 정상, 도커 컨테이너 자동 복귀, Ollama 정상)

## 현재 상태

- 서버: 정상 운영 (잠긴 상태, ~70 TFLOPS)
- 부트 옵션: 재등록 완료 (`gpu_unlock.targets=all`)
- **대기**: 물리적 전원 차단(30초+) 후 재부팅 → 콜드 카드 언록 재시도 예정

## 교훈

- Ada(40세대) 언록은 **반드시 완전 콜드 파워 사이클** 이후 시도할 것
- 웜 재부팅만 반복하면 동일한 `-110` 실패가 재현된다
- 실패 시에도 복구는 단순(옵션 제거 → 재부팅)하므로 원격에서 안전하게 재시도 가능
