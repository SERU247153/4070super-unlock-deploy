# RTX 4070 SUPER 텐서코어 소프트웨어 언록 시스템의 설계 및 배포 방법론

**A Design and Deployment Methodology for Software-Based Tensor Core Unlocking on the NVIDIA RTX 4070 SUPER**

---

## 초록 (Abstract)

NVIDIA는 소비자용 GeForce 그래픽 카드의 텐서코어 연산률(FP16·BF16·TF32, FP32 누적)을 동일 실리콘을 사용하는 전문가용(A-시리즈) 카드 대비 50% 수준으로 제한하고 있다. 본 문서는 Linux 기반 주피터 노트북 서버(Rocky Linux 9)에 탑재된 RTX 4070 SUPER(AD104, `10de:2783`)를 대상으로, 부팅 초기 단계에서 속도 셀렉터 레지스터(SS0/SS1)를 재기록하는 오픈소스 소프트웨어 언록 기법(gpu-unlock)을 **안전하게 배포·검증·원복**하기 위한 자동화 키트의 설계와 구현을 기술한다. 제안하는 키트는 ① 사전 점검, ② 전·후 대비 벤치마크, ③ 설치, ④ 성능 튜닝, ⑤ 검증, ⑥ 원상복구의 6개 스크립트로 구성되며, 언록이 GPU 칩 레지스터 단위로 적용되는 특성을 활용하여 호스트와 모든 도커 컨테이너 워크로드에 일괄 적용된다. 벤치마크는 FP32 누적 GEMM의 중앙값 처리량을 지표로 삼아 언록 성공 여부를 통계적으로 판별한다.

**키워드**: GPU 언록, 텐서코어, RTX 4070 SUPER, AD104, gpu-unlock, Rocky Linux, 도커, GEMM 벤치마크

---

## 1. 서론

### 1.1 연구 배경

NVIDIA는 시장 세분화(market segmentation) 전략에 따라, 동일한 실리콘 다이로 생산되는 GeForce(소비자용)와 RTX A-시리즈(전문가용) 카드 간에 인위적인 연산 성능 차등을 두고 있다 [1]. 구체적으로 GeForce 계열은 FP16/BF16/TF32 입력에 대한 FP32 누적 텐서 연산 속도가 전문가용 카드의 절반으로 고정되는데, 이는 인공지능 학습·추론 워크로드의 사실상 표준 연산 경로에 해당한다.

RTX 4070 SUPER(AD104)의 경우 이 제한은 이론 최대 처리량 약 142 TFLOPS를 약 71 TFLOPS로 절반 감소시킨다.

### 1.2 문제 정의

본 연구의 대상 환경인 주피터 노트북 서버는 다음 조건을 갖는다.

- OS: Rocky Linux 9 (RHEL 계열, dracut 기반 initramfs, GRUB2/BLS 부팅)
- GPU: NVIDIA RTX 4070 SUPER (PCI `10de:2783`, AD104, VRAM 12 GB)
- 실행 환경: Docker + nvidia-container-toolkit 기반 컨테이너 워크로드

이 환경에서 언록을 적용하기 위해서는 (a) 부팅 극초기(드라이버 바인딩 이전)에 펌웨어 권한 레지스터 기록을 수행하고, (b) 적용 결과를 계측으로 검증하며, (c) 운영 서버 특성상 언제든 원상복구 가능해야 한다. 본 문서는 이 세 요구를 충족하는 자동화 키트의 설계·구현·실험 계획을 다룬다.

### 1.3 기여

1. RHEL 계열 배포판(Rocky 9) 특성을 반영한 gpu-unlock 배포 파이프라인 자동화
2. SELinux, Secure Boot, EPEL 의존성 등 운영 환경 위험요인을 반영한 사전 점검 절차
3. FP32 누적 GEMM 중앙값 처리량 기반의 전·후 통계적 검증 방법론

---

## 2. 배경 및 관련 연구

### 2.1 텐서코어 반속 제한의 구조

언록 대상은 물리적 퓨즈가 아닌 **속도 셀렉터(Speed Selector, SS0/SS1) 레지스터**의 설정값이다 [1]. GPU 부팅 시 자체 펌웨어가 카드의 제품 계열을 판별하여 SS0/SS1에 반속 값을 기록하며, 이 레지스터는 펌웨어 서명 인증 없이는 쓰기가 거부된다. 따라서 제한은 하드웨어적 성질이 아니라 소프트웨어적 성질을 띠며, 전원 차단 시 초기화되어 부팅마다 재설정이 필요하다.

### 2.2 CMP V67/PLM 기법

레지스터 쓰기 금지는 서명된 펌웨어(Booter) 코드의 실행 경로를 재구성하여 우회된다. Petri Krohn의 cmpunlocker 계열 연구 [2]에서 확립된 V67/PLM 기법은 서명 이미지 내 기존 코드 조각(gadget)의 재배열을 통해 펌웨어 본인 권한으로 원하는 레지스터를 기록하는 방식으로, 서명 검증을 통과하면서도 임의 쓰기가 가능하다. PLM(Privileged Lookup/Modify) 게이트를 연 뒤 SS0에는 `0x88888888`(전속 BF16 선택), SS1에는 `0x8`(IMLA4)을 기록한다.

### 2.3 Zagitta/gpu-unlock

Zagitta/gpu-unlock [1]은 위 기법을 Linux 커널 모듈로 구현하여, initramfs에서 NVIDIA 드라이버 로드 **이전**에 언록을 수행한다. 패치된 드라이버나 VBIOS 플래시를 요구하지 않으며, 커널 커맨드라인 옵트인(`gpu_unlock.targets=`)을 통해 활성화된다. RTX 30세대(Ampere)는 Stable, RTX 40세대(Ada)와 20세대(Turing)는 Experimental로 분류된다 [1][3].

### 2.4 Ada(AD10x) 지원 현황

AD104 기반 RTX 4070 Ti(`10de:2782`)에서 BF16/FP16 +83~84%의 실측 보고가 존재하며 [4], 본 대상인 RTX 4070 SUPER(`10de:2783`)는 지원 장치 목록에 등록되어 있고 동일 AD102 부트로더 자산으로 언록된 사례가 보고되었다 [3][5].

---

## 3. 시스템 설계

### 3.1 요구사항

| ID | 요구사항 | 반영 스크립트 |
|---|---|---|
| R1 | 운영 전 위험요인(보안 부팅, 패키지 누락 등) 사전 탐지 | `01_preflight.sh` |
| R2 | 언록 전·후 성능 계측을 통한 적용 여부 판별 | `04_benchmark.sh` |
| R3 | Rocky 9(RHEL 계열) 고유 설치 경로 자동화 | `02_install_unlock.sh` |
| R4 | 전력·클럭·드라이버 옵션 최적화 | `03_tune.sh` |
| R5 | 부팅 로그 기반 적용 검증 | `05_verify.sh` |
| R6 | 완전 원상복구 | `06_rollback.sh` |

### 3.2 아키텍처 개요

```
┌──────────────┐   ┌───────────────────────────────┐
│  config.sh   │──▶│  단계별 실행 스크립트 (01~06)  │
│  (설정 단일화) │   │  · source 로 설정 주입          │
└──────────────┘   │  · 공통: set -u/pipefail, 색상  │
                   │  · 배포: dnf + rpmbuild + DKMS │
                   │  · 검증: journalctl/lspci/GEMM │
                   └───────────────┬───────────────┘
                                   ▼
                   NVIDIA 드라이버 로드 이전(initramfs)
                   gpu_unlock 모듈 → SS0/SS1 재기록 → FLR
```

언록은 GPU 칩 레지스터 단위로 유지되므로, 이후 바인딩되는 모든 드라이버 세션(호스트 및 컨테이너)에 동일하게 적용된다.

### 3.3 배포 파이프라인

배포는 재부팅 1회로 완결되도록 설계하였다: `02`(설치) → `03`(튜닝) → 재부팅 → `05`(검증) → `04 after`(계측). 전력·클럭 항목은 즉시 반영되고, 드라이버 옵션(`NVreg_EnableStreamMemOPs`)과 언록은 부팅 시 반영되므로 두 시점이 단일 재부팅으로 수렴한다.

### 3.4 원상복구 전략

`06_rollback.sh`는 ① grubby 부트 인자 제거, ② RPM 패키지 제거(DKMS 항목 포함), ③ 전력 제한 기본값 복원·클럭 고정 해제, ④ 추가 드라이버 옵션 파일 제거 및 initramfs 재빌드를 순차 수행한다. 모든 변경은 GPU 하드웨어에 비영구적이므로 재부팅 후 순정 상태로 수렴한다.

---

## 4. 구현

### 4.1 스크립트 구성

| 스크립트 | 핵심 구현 사항 |
|---|---|
| `01_preflight.sh` | os-release 계열 판별, `rpm -q` 패키지 검사, Secure Boot(`mokutil`) 탐지, 네트워크·도커 툴킷 점검, `--fix` 자동 설치 |
| `02_install_unlock.sh` | 소스 클론 → `make install`(RHEL 계열 자동 감지) → RPM/DKMS 설치 → `grubby --update-kernel=ALL` 부트 인자 등록(중복 제거 포함) |
| `03_tune.sh` | persistence mode, 전력 제한(min/default/max 파싱 후 허용 범위 검증), 선택적 클럭 고정(`-lgc`), modprobe 옵션 등록 + dracut 재빌드, ReBAR 점검 |
| `04_benchmark.sh` | 도커 `--gpus all` 실행, SELinux 마운트 라벨(`:Z`), 전·후 JSON 비교표 자동 산출 |
| `05_verify.sh` | `/proc/cmdline` 확인, `journalctl -k -b`의 `unlocked` 기록 탐지, `lspci -nnk` 바인딩 확인, DKMS 상태 점검 |
| `06_rollback.sh` | 3.4절의 4단계 원복 절차 |

### 4.2 벤치마크 방법론

측정 대상은 NVIDIA가 반속으로 잠근 연산 경로인 **FP16/BF16 입력 · FP32 누적 GEMM**이다. PyTorch에서 다음 플래그를 강제하여 해당 경로를 정확히 격리한다.

- `torch.backends.cuda.matmul.allow_tf32 = False` — TF32 경유 차단
- `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False` — FP16 누적을 FP32로 강제

처리량은 식 (1)로 산출한다.

> **TFLOPS = 2·N³ / t · 10⁻¹²**  … (1)   (N=8192, t: 중앙값 실행 시간[초])

반복 7회의 중앙값을 사용하여 이상치(outlier)와 초기 워밍업 영향을 배제한다.

### 4.3 캘리브레이션(교정) 방법론

전·후 비교의 내적 타당성을 확보하기 위해 IEEE TPDS 방법론 [7]과 NVIDIA 아키텍처 백서 [8][9]의 실측 상수에 근거한 3중 교정을 적용한다.

**(a) 이론 피크 모델.** Ada 4세대 텐서코어는 SM당 4개, TC당 FP16 256 FMA/clk를 수행한다 [8]. Sun et al. [7]의 실측에 따르면 FP32 누적 경로는 FP16 누적의 절반인 SM당 512 FMA/clk, FP16 누적 경로는 1024 FMA/clk이다. 따라서 실측 SM 클럭 f로부터 이론 피크를 식 (2)·(3)과 같이 도출하고, 측정값과의 비(효율)를 산출한다.

> **P_fp32acc = N_SM × 512 × 2 × f**  … (2)
> **P_fp16acc = N_SM × 1024 × 2 × f**  … (3)

RTX 4070 SUPER(56 SM, f=2.475 GHz)의 경우 P_fp32acc ≈ 142 TFLOPS이며, 언록 전에는 이 값의 절반(≈71 TFLOPS)으로 잠겨 있다.

**(b) 참조 경로(reference path).** TF32를 차단한 FP32 GEMM은 CUDA 코어에서 실행되며, NVIDIA의 속도 제한은 텐서코어 FP16/BF16/TF32·FP32 누적 경로에만 적용되므로 [1] 참조 경로는 언록과 무관하게 항상 동일 속도로 동작한다. Ada의 CUDA 코어 FP32는 SM당 128 FMA/clk이며, 이론 피크는 식 (4)와 같다. 따라서 FP32 처리량은 기기의 실질 상태(클럭·전력·발열 조건의 총합)를 나타내며, 언록 전·후 두 실행에서 참조 경로 처리량이 **±10% 이내로 유지**되어야 두 실행의 측정 조건이 동등하다고 판정한다.

> **P_fp32cuda = N_SM × 128 × 2 × f**  … (4)

주: 참조 경로 후보로 검토한 INT8 GEMM은 PyTorch 미지원(`addmm_cuda not implemented for 'Char'`), FP16·FP16 누적 강제 경로는 NGC alpha 빌드에서 플래그가 무시됨이 실측으로 확인되어, FP32 CUDA 코어 경로를 채택하였다.

**(c) 조건 통제.** `BENCH_LOCK_CLOCKS`로 SM 클럭을 고정하여 부스트 클럭 변동을 제거하고, 실행 중 전력 소비를 1초 간격으로 샘플링하여 평균·최대를 결과에 기록한다. 이는 Sun et al. [7]이 제시한 클럭 고정 기반 마이크로벤치마크 관행을 따른다.

**판정 규칙**은 다음과 같다.

| 조건 | 판정 |
|---|---|
| 제한 경로 gain ≥ 1.7 ∧ 참조 경로 비율 ∈ [0.9, 1.1] | ✅ 언록 성공 (측정 유효) |
| 제한 경로 gain ≥ 1.7 ∧ 참조 경로 변동 | ⚠️ 조건 상이 — 재측정 권장 |
| gain ≤ 1.15 | ❌ 미적용 |
| 1.15 < gain < 1.7 | 🔶 부분 적용 (전력 스로틀링 등) |

### 4.4 운영 환경 안전장치

| 위험 | 대응 |
|---|---|
| Secure Boot에 의한 미서명 모듈 차단 | `mokutil --sb-state` 사전 탐지, FAIL 처리 |
| SELinux enforcing의 바인드 마운트 거부 | 볼륨 옵션에 `:Z` 레이블 부여 |
| EPEL 미구성(dkms 부재) | `epel-release` 선설치 후 dkms 설치 |
| `nvidia-smi` 출력 형식 차이 | Enforced Power Limit 기준 파싱 + 공란 가드 |
| 커널 업데이트 후 모듈 소실 | DKMS %posttrans의 전체 커널 재빌드 |

---

## 5. 실험 계획

### 5.1 실험 환경

- GPU: RTX 4070 SUPER 12GB (AD104, `10de:2783`), 스톡 전력 제한 220 W
- 호스트: Rocky Linux 9, NVIDIA 독점 드라이버, Docker + nvidia-container-toolkit
- 벤치마크 컨테이너: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`
- 측정: 8192³ GEMM, FP16/BF16, FP32 누적, 7회 반복 중앙값

### 5.2 평가 지표

1. **GEMM 처리량**: 식 (1)의 TFLOPS, 전·후 비율(gain)
2. **효율**: 식 (2) 이론 피크 대비 측정 비율 (캘리브레이션 a)
3. **참조 경로 안정성**: FP32(CUDA 코어) 경로의 전·후 비율 ∈ [0.9, 1.1] (캘리브레이션 b)
4. **언록 로그**: 부팅 저널 내 `gpu_unlock: unlocked` 기록 존재 여부
5. **드라이버 바인딩 정상성**: 언록 후 `nvidia` 정상 바인딩(`lspci -nnk`)

### 5.3 기대 성능

| 지표 | 언록 전 | 언록 후(기대) |
|---|---|---|
| BF16 GEMM (FP32 누적) | ≈ 71 TFLOPS | ≈ 130–142 TFLOPS |
| FP16 GEMM (FP32 누적) | ≈ 71 TFLOPS | ≈ 130–142 TFLOPS |
| LLM 프리필 (실워크로드) | 기준 | +25~55% |
| 이미지 생성 (FP16) | 기준 | +30~55% |

### 5.4 1차 배포 실측 (2026-11-01, 대상 서버)

클럭 고정 2505 MHz 조건에서 언록 전 기준값을 측정하였다.

| 경로 | 실측 | 이론 대비 |
|---|---|---|
| BF16 · FP32 누적 | 69.1 TFLOPS | 잠긴 이론치(71.8)의 96.2% |
| FP16 · FP32 누적 | 70.3 TFLOPS | — |
| FP32(CUDA 코어) 참조 | 22.8 TFLOPS | 이론(29.7)의 76.7% |

1차 언록 시도에서 셀렉터 기록(SS0=0x88888888, SS1=0x8)까지는 성공했으나, FLR 후 GSP 재기동 대기에서 `-110`(ETIMEDOUT)으로 실패하였다. 이는 완전 콜드(전원 차단) 상태가 아닌 카드에서 재현되는 알려진 실패 패턴이다(gpu-unlock 이슈 #23). 웜 재부팅만으로 GPU 전원이 차단되지 않아 펌웨어 상태가 초기화되지 않기 때문이며, 부트 옵션 제거 후 재부팅하는 절차로 GPU는 완전 복구됨을 확인하였다.

### 5.5 최종 실측 결과 (콜드 파워 사이클 후 언록 성공)

전원 완전 차단(30초+) 후 재부팅에서 `gpu_unlock: unlocked` (`stage=handoff ret=0 cleanup=0`)을 확인하였다. 클럭 2505 MHz 고정 조건의 최종 결과는 다음과 같다.

| 경로 | 잠김(220W) | 언록(220W) | 언록(245W) | 향상 |
|---|---|---|---|---|
| FP16 · FP32 누적 | 70.3 TFLOPS | 126.9 TFLOPS | 124.4 TFLOPS | +77~81% |
| BF16 · FP32 누적 | 69.1 TFLOPS | 125.3 TFLOPS | 132.9 TFLOPS | +81~92% |
| FP32(CUDA) 참조 | 22.8 TFLOPS | 23.1 TFLOPS (비율 1.01) | 25.1 TFLOPS | — |

- 220W 조건에서 참조 경로 비율 1.01로 캘리브레이션 판정 통과 → 언록 성공 확정
- 언록 시 BF16 효율 95.1%(이론 피크 143.6 TFLOPS 대비)로 텐서코어가 사실상 완전 개방됨
- 245W 전력 상향은 BF16에서 추가 +6% (125.3 → 132.9 TFLOPS)
- 운영상 주의: 웜 재부팅 시 논콜드 조건으로 `-110` 재현 가능 — 재부팅 필요 시 부트 옵션 제거 후 콜드 파워 사이클 재적용 권장

### 5.6 캘리브레이션 실측: 전력·클럭·성능 특성

안정화 워밍업(폐기 측정) 후 본 측정하는 절차로 전력·클럭 스윕을 수행하였다 (10_calibrate.sh).

**전력 스윕**(자연 부스트): BF16 처리량은 180W에서 이미 131.5 TFLOPS로 245W(131.2)와 동급이며, 160W(83.7)에서만 실질 하락한다. 따라서 **180W가 성능/전력 스위트 스팟**이다.

**클럭 스윕**(245W 고정): 풀부하 지속 클럭은 ~2400 MHz에서 포화하며(2505/2700 요청 모두 2392~2408 실측), 전 구간에서 이론 피크 대비 효율 92.9~96.6%를 기록하여 식 (2)의 이론 모델을 실측으로 검증하였다.

---

## 6. 한계 및 위험 분석

1. **Experimental 단계**: Ada 세대는 안정성이 완전히 검증되지 않았으며, 보드/펌웨어 조합에 따라 언록 실패 사례가 보고됨 [1].
2. **콜드 상태 요구 (실측 확인)**: 완전 전원 차단을 거치지 않은 카드에서는 FLR 후 GSP 재기동이 `-110`으로 실패하여 GPU가 드라이버와 통신 불가 상태가 될 수 있다 (1차 배포 실측, gpu-unlock 이슈 #23과 동일 패턴). 언록 시도 전 반드시 물리적 파워 사이클(30초+)이 필요하며, 실패 시 부트 옵션 제거 후 웜 재부팅으로 복구 가능함을 확인하였다.
3. **비영구성**: 전원 차단·재부팅 시 언록이 초기화된다. 부팅 자동 재적용으로 실운영 영향은 없으나, 최대절전(hibernation) 경로는 미지원이다.
4. **전력 병목**: 풀속 텐서코어는 전력 소모를 증가시키며, 220 W 기본 제한 하에서 대형 GEMM은 스로틀링으로 실측 이득이 이론치에 미치지 못할 수 있다. 본 키트는 전력 제한 상향(R4, 본 서버는 245 W)으로 이를 완화한다.
5. **워크로드 의존성**: INT8/INT4/GGUF 등 양자화 경로와 디코드(메모리 대역폭 병목) 단계에는 효과가 미미하다.

---

## 7. 결론

본 문서는 NVIDIA의 인위적 반속 제한을 소프트웨어적으로 해제하는 gpu-unlock 기법을 Rocky Linux 9 기반 도커 서버에 안전하게 배포하기 위한 자동화 키트의 설계·구현·검증 방법론을 제시하였다. 언록이 칩 레지스터 단위로 적용되는 구조적 특성 덕분에 호스트·컨테이너를 불문하고 FP16/BF16 워크로드 전반에서 최대 2배의 처리량 개선이 기대되며, 전·후 대칭 벤치마크와 완전 원복 절차를 통해 운영 서버에서의 위험을 통제한다.

---

## 참고문헌

[1] Zagitta, *gpu-unlock: Your GeForce runs AI at half speed. This fixes it.*, GitHub 저장소, 2026. https://github.com/Zagitta/gpu-unlock

[2] P. Krohn, *cmpunlocker* 계열 연구(V67/PLM 레지스터 언록 기법), gpu-unlock 저장소 `recipes.c` 주석 및 관련 문서에서 인용.

[3] *AD103 / RTX 4080 SUPER support: offline Booter evidence and remaining validation gates*, gpu-unlock 이슈 #4, 2026.

[4] *Compatibility report: first AD104 — Inno3D RTX 4070 Ti (10de:2782)*, gpu-unlock 이슈 #87, 2026. (+83~84% GEMM 실측)

[5] JustAResearcher, *gpu-unlock-ada*, GitHub 포크, 2026. (AD104 `2783` RTX 4070 SUPER 매칭 리포트 포함)

[6] impacifist, *Windows-GPU-UNLOCK*, GitHub 저장소, 2026. (Windows 포트, 본 연구 범위 밖)

[7] W. Sun, A. Li, T. Geng, S. Stuijk, and H. Corporaal, "Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors," *IEEE Transactions on Parallel and Distributed Systems*, vol. 34, no. 1, pp. 246–261, 2023. doi:10.1109/TPDS.2022.3217824 (FP16·FP32 누적 512 FMA/clk/SM, FP16 누적 1024 FMA/clk/SM 실측)

[8] NVIDIA, *NVIDIA Ada GPU Architecture* (whitepaper V2.02), 2023. (4세대 텐서코어 4개/SM, TC당 FP16 256 FMA/clk)

[9] NVIDIA, *NVIDIA A100 Tensor Core GPU Architecture* (whitepaper), 2020. (3세대 텐서코어 TC당 FP16 256 FMA/clk, SM당 1024 FMA/clk — Ada와 동일 규격의 근거)
