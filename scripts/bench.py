#!/usr/bin/env python3
"""
gpu-unlock 전/후 벤치마크 + 캘리브레이션 (IEEE TPDS 방법론 기반)
========================================================================

측정 경로 (NVIDIA가 반속으로 잠근 연산):
  FP16/BF16 입력 · FP32 누적 GEMM  → 언록 전 약 절반 / 언록 후 약 2배

캘리브레이션 (교정):
  1) 참조 경로 — FP32(CUDA 코어) GEMM. TF32를 차단한 FP32 행렬곱은
     CUDA 코어에서 실행되며 NVIDIA가 제한하지 않는 경로이므로, 언록과
     무관하게 항상 동일 속도로 동작한다. 전/후 실행에서 참조 경로가
     ±10% 이내로 유지되어야 두 실행의 측정 조건(클럭·전력·발열)이
     동등하다고 판정한다.
     ※ INT8 matmul은 PyTorch에서 미지원(addmm_cuda not implemented for
       'Char'), FP16·FP16 누적 강제 플래그는 NGC alpha 빌드에서 무시됨이
       실측으로 확인됨 → FP32 CUDA 코어 경로를 참조로 채택.
  2) 이론 피크 모델 — 각 경로의 실측 SM 클럭으로 이론 처리량을 계산해
     효율(measured/theoretical)을 산출한다.

이론 피크 근거:
  · NVIDIA Ada 아키텍처: 4세대 텐서코어 4개/SM, TC당 FP16 256 FMA/clk  [8]
  · Sun et al., IEEE TPDS 34(1), 2023: FP16·FP32 누적 = 512 FMA/clk/SM,
    FP16·FP16 누적 = 1024 FMA/clk/SM (Ampere 실측, Ada 동일 규격)      [7]
  · CUDA 코어 FP32: SM당 128 FMA/clk (Ada 스펙)
  → P(fp32 acc) = N_SM ×  512 × 2 × f   [FLOP/s]
  → P(fp16 acc) = N_SM × 1024 × 2 × f   [FLOP/s]
  → P(fp32 cuda)= N_SM ×  128 × 2 × f   [FLOP/s]
"""
import argparse
import json
import statistics
import time

import torch

FMA_PER_SM_FP32_ACC = 512    # FMA/clk/SM — FP16/BF16 입력, FP32 누적 [7]
FMA_PER_SM_FP16_ACC = 1024   # FMA/clk/SM — FP16 입력, FP16 누적 [7]
FMA_PER_SM_FP32_CUDA = 128   # FMA/clk/SM — FP32 CUDA 코어 (TF32 차단 시)


def get_sm_clock() -> float:
    """실측 SM 클럭(MHz). PyTorch 버전에 따라 없으면 0."""
    try:
        clock_mhz = float(torch.cuda.clock_rate())
    except (AttributeError, RuntimeError):
        return 0.0
    return clock_mhz


def bench_path(a: torch.Tensor, b: torch.Tensor, iters: int):
    """워밍업 3회 → 클럭 샘플 → iters 회 반복의 중앙값 시간(초) → 클럭 샘플.

    클럭은 경로별로 워밍업 이후에 측정하여 유휴(P8) 클럭 오염을 배제한다.
    """
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    c1 = get_sm_clock()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    c2 = get_sm_clock()
    return statistics.median(times), (c1 + c2) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8192, help="정방 행렬 크기 N (N^3 GEMM)")
    ap.add_argument("--iters", type=int, default=7, help="경로별 측정 반복 (중앙값 사용)")
    ap.add_argument("--label", default="unknown", help="결과 라벨 (before/after)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    # TF32 경유 차단 (전역)
    torch.backends.cuda.matmul.allow_tf32 = False

    assert torch.cuda.is_available(), "CUDA 사용 불가"
    props = torch.cuda.get_device_properties(0)
    sm = props.multi_processor_count
    size, iters = args.size, args.iters

    def tflops_of(med: float) -> float:
        return 2.0 * size ** 3 / med / 1e12

    def theoretical(fma_per_sm: int, clock_mhz: float) -> float:
        if clock_mhz <= 0:
            return 0.0
        return sm * fma_per_sm * 2.0 * (clock_mhz * 1e6) / 1e12

    print(f"GPU    : {torch.cuda.get_device_name(0)} (SM {sm})")
    print(f"torch  : {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"GEMM   : {size}^3, 중앙값 {iters}회\n")

    # ---------- 1) 참조 경로: FP32 CUDA 코어 (언록 무관, 항상 동일 속도) ----------
    # allow_tf32=False 이므로 cuBLAS SGEMM(CUDA 코어) 경로가 선택된다.
    a32 = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b32 = torch.randn(size, size, device="cuda", dtype=torch.float32)
    med_ref, clock_ref = bench_path(a32, b32, iters)
    ref_tflops = tflops_of(med_ref)
    theo_ref = theoretical(FMA_PER_SM_FP32_CUDA, clock_ref)
    del a32, b32
    torch.cuda.empty_cache()

    # ---------- 2) 제한 경로: FP16/BF16 · FP32 누적 (언록 대상) ----------
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    capped = {}
    for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        med, clock = bench_path(a, b, iters)
        capped[dtype_name] = (med, clock)
        del a, b
        torch.cuda.empty_cache()

    # ---------- 3) 캘리브레이션: 이론 피크 & 효율 (경로별 클럭 기준) ----------
    theo_fp32 = theoretical(FMA_PER_SM_FP32_ACC, max(clock for _, clock in capped.values()))
    results = {}
    for name, (med, clock) in capped.items():
        tflops = tflops_of(med)
        theo_path = theoretical(FMA_PER_SM_FP32_ACC, clock)
        results[name] = {
            "median_s": med,
            "tflops": tflops,
            "clock_mhz": clock,
            "efficiency": (tflops / theo_path) if theo_path > 0 else 0.0,
        }

    # ---------- 4) 출력 ----------
    print("== 캘리브레이션 ==")
    print(f"  참조 경로(fp32_cuda, {clock_ref:.0f} MHz): {ref_tflops:7.1f} TFLOPS "
          f"(이론 {theo_ref:.1f}, 효율 {ref_tflops / theo_ref * 100:.1f}%)" if theo_ref > 0
          else f"  참조 경로(fp32_cuda, {clock_ref:.0f} MHz): {ref_tflops:7.1f} TFLOPS")
    print(f"  제한 경로 이론 피크(FP32 누적, 최대 클럭): {theo_fp32:6.1f} TFLOPS")
    print("== 제한 경로 (언록 대상) ==")
    for name, r in results.items():
        eff = f"효율 {r['efficiency'] * 100:.1f}%" if r["efficiency"] > 0 else ""
        print(f"  [{name}] {r['median_s'] * 1000:8.1f} ms -> {r['tflops']:7.1f} TFLOPS "
              f"({r['clock_mhz']:.0f} MHz)  {eff}")

    # ---------- 5) 저장 ----------
    if args.out:
        payload = {
            "label": args.label,
            "gpu": torch.cuda.get_device_name(0),
            "sm": sm,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "size": size,
            "iters": iters,
            "calibration": {
                "theoretical_fp32acc_tflops": theo_fp32,
                "reference": {
                    "dtype": "fp32_cuda",
                    "median_s": med_ref,
                    "tflops": ref_tflops,
                    "clock_mhz": clock_ref,
                    "efficiency": (ref_tflops / theo_ref) if theo_ref > 0 else 0.0,
                },
            },
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
