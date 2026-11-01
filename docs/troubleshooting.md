# 트러블슈팅 가이드

실전 배포(Rocky 9.8 + RTX 4070 SUPER, 헤드리스 주피터 서버)에서 실제로 만난 문제와 검증된 해결책.

---

## 1. 빌드/설치 단계 실패

### 1-A. `Macro %nv has illegal name (%global)`

- **증상**: `make install` 중 `dnf builddep` 단계에서 spec 파싱 실패
- **원인**: Rocky 9의 rpm 4.16은 **2글자 매크로 이름을 거부** (`nv`, `v` 불가 / `foo`, `ver` 허용). gpu-unlock spec의 `%global nv` 와 충돌
- **대응**: `02_install_unlock.sh` 1.5절이 자동 감지 후 `%nv → %nvv` 치환 (idempotent)
- **수동 복구**: `sed -i 's/%global nv /%global nvv /; s/%{nv}/%{nvv}/g' packaging/rpm/gpu-unlock.spec`

### 1-B. DKMS 빌드 실패 — `implicit declaration of 'remove_conflicting_pci_framebuffers'`

- **증상**: 패키지는 설치되지만 `%posttrans` DKMS 빌드에서 `unlock_610.c:1501` 컴파일 에러
- **원인**: RHEL 9.x 커널은 **6.0의 aperture API를 5.14에 백포트**했다 (`aperture_remove_conflicting_pci_devices` 존재, 구형 `remove_conflicting_pci_framebuffers` 없음). 상류 소스의 `LINUX_VERSION_CODE >= 6.0` 분기만으로는 구형 API로 빠져 실패
- **대응**: `02_install_unlock.sh` 1.6절이 `RHEL_RELEASE_CODE` 매크로로 분기를 보강 (자동)
- **수동 복구** (패키지 설치가 이미 끝난 경우):
  ```bash
  sed -i 's/#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 0, 0)/#if defined(RHEL_RELEASE_CODE) || LINUX_VERSION_CODE >= KERNEL_VERSION(6, 0, 0)/g' \
    /usr/src/gpu-unlock-2.0.0/unlock_610.c
  dkms build -m gpu-unlock -v 2.0.0 --force
  dkms install -m gpu-unlock -v 2.0.0
  dracut --force --kver "$(uname -r)"
  ```
- **확인**: `dkms status | grep gpu-unlock` → `installed`, `lsinitrd /boot/initramfs-$(uname -r).img | grep gpu_unlock`

### 1-C. dracut `No space left on device` (/boot 가득 참)

- **원인**: /boot 파티션이 작고 오래된 커널 이미지가 누적
- **해결**: `dnf remove -y --oldinstallonly --setopt installonly_limit=2 kernel`
- **참고**: 제거 후 `/lib/modules/` 에 잔여 디렉토리가 남을 수 있으나 무해

---

## 2. 언록 실패 (재부팅 후)

### 2-A. `unlock failed (-110)` + nvidia-smi 통신 불가 — **논콜드 카드**

- **증상**:
  ```
  pci 0000:01:00.0: gpu_unlock: stage=host ss0=0x88888888 ss1=0x8   ← 셀렉터 기록은 성공
  pci 0000:01:00.0: gpu_unlock: failure wait_post_flr_gfw:1460 error=-110
  pci 0000:01:00.0: gpu_unlock: unlock failed (-110)
  NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
  ```
- **원인**: `-110`(ETIMEDOUT)은 **완전 콜드(전원 차단) 상태가 아닌 카드**에서 FLR 후 GSP 재기동이 실패하는 알려진 패턴 (gpu-unlock 이슈 #23). 웜 재부팅으로는 PCIe 대기전력이 유지되어 GPU 펌웨어 상태가 초기화되지 않음
- **복구 절차 (검증됨, 원격 가능, 약 5분)**:
  ```bash
  grubby --update-kernel=ALL --remove-args="gpu_unlock.targets=all"
  systemctl reboot
  # 부팅 후 nvidia-smi 정상 확인 → 도커 컨테이너 자동 복귀
  ```
- **재시도 요건**: 부트 옵션 재등록 후 **반드시 물리적 전원 차단(PSU/PDU/BMC Power Off) 30초 이상** → 전원 재투입. 웜 재부팅으로의 재시도는 동일 실패를 반복한다. (실전 확인: 콜드 파워 사이클 후 `unlocked` 성공, BF16 +92%)
- **운영상 주의**: 언록 성공 이후에도 **웜 재부팅(전원을 끊지 않는 reboot)은 피할 것**. 부팅 시 모듈이 재실행되며 논콜드 조건으로 `-110`이 재현될 수 있다. 재부팅이 필요하면 ① 부트 옵션 제거 → ② 웜 재부팅(업데이트 등) → ③ 콜드 파워 사이클로 재적용 순서를 권장한다.

### 2-B. `unlock failed (-16)` / `card already initialised since power on (WPR2 enabled)`

- **증상**: 부팅 로그에 EBUSY 계열 거부
- **원인**: 전원 인가 후 이미 드라이버 등이 카드를 초기화한 상태
- **대응**: 2-A 와 동일 — 완전 파워 사이클 후 재시도

---

## 3. 검증 단계 주의

- `05_verify.sh` 는 **nvidia-smi 통신까지 확인**해야 한다 (단순히 `lspci` 바인딩만 보면 오판).
- 로그에 `gpu_unlock: unlocked` 문구가 있어야 성공. `stage=host ss0=0x88888888` 이후 `failure ...` 가 나오면 셀렉터 기록 후 FLR 단계에서 실패한 것.

---

## 4. 원격 운영 (물리 접근 없이) — 웜 부팅 문제 보완

언록 적용 중에는 웜 재부팅이 위험하고, 재언록에는 콜드 파워 사이클이 필요하다.
물리 접근이 어려운 환경(원격 서버)을 위한 3겹 보완책:

### 4-A. 자동 복구 워치독 (설치됨, `08_install_watchdog.sh`)

- 부팅 후 `nvidia-smi` 통신 불가 + 부팅 로그에 `gpu_unlock: unlock failed` 감지 시
- 부트 옵션 자동 제거 → 1회 자동 재부팅으로 GPU 복구 (플래그 파일로 루프 방지)
- 효과: 실수로 웜 재부팅해도 **무인 자가치유** — GPU만 잠긴 상태로 복구됨
- 로그: `/var/log/gpu-unlock-watchdog.log`, 플래그: `/var/lib/gpu-unlock/watchdog-recovered`
- 상태 확인: `systemctl status gpu-unlock-watchdog`

### 4-B. 스마트 플러그 + BIOS AC 복구 (근본 해결, ⭐ 권장)

데스크톱 서버(B650M 등)라면 WiFi 스마트 플러그로 원격 콜드 사이클이 가능하다:

1. BIOS: **Settings → Platform Power → AC BACK → Always On**
   (전원 복구 시 자동 부팅. Gigabyte B650M 기준, 보드별 명칭은 AC Power Loss/Restore 등)
2. 서버 전원 코드를 **WiFi 스마트 플러그**에 연결
3. 이후 모든 재부팅을 스마트 플러그 OFF → 30초 → ON 으로 수행 = **항상 콜드 부팅**
4. 부수 효과: 정전 → 복전도 콜드 사이클이므로 언록이 자동으로 다시 적용됨 (정전은 위험이 아니라 기회)

### 4-C. 안전 웜 재부팅 헬퍼 (`09_safe_reboot.sh`)

- 스마트 플러그가 아직 없다면, 원격 유지보수(커널 업데이트 등) 재부팅 시 이 스크립트 사용
- 부트 옵션 제거 → 웜 재부팅 → GPU는 잠긴 상태로 정상 동작
- 재언록: 부트 옵션 재등록 후 콜드 파워 사이클 (대전 방문 시, 또는 스마트 플러그로)
