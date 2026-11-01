#!/usr/bin/env python3
"""
gpu-unlock 전/후 벤치마크 + 캘리브레이션 (IEEE TPDS 방법론 기반)
========================================================================

측정 경로 (NVIDIA가 반속으로 잠근 연산):
  FP16/BF16 입력 · FP32 누적 GEMM  → 언록 전 약 절반 / 언록 후 약 2배

캘리브레이션 (교정):
  1) 참조 경로 — FP16·FP16 누적 GEMM. 이 경로는 언록과 무관하게 항상
     풀속으로 동작하므로, 기기의 "실제 천장"(클럭·전력·발열 조건 포함)을
     측정한다. 전/후 실행에서 참조 경로가 ±10% 이내로 유지되어야 두
     실행의 조건이 동등하다고 판정한다.
  2) 이론 피크 모델 — 실측 SM 클럭으로부터 이론 처리량을 계산해
     효율(measured/theoretical)을 산출한다.

이론 피크 근거:
  · NVIDIA Ada 아키텍처: 4세대 텐서코어 4개/SM, TC당 FP16 256 FMA/clk  [8]
  · Sun et al., IEEE TPDS 34(1), 2023: FP16·FP32 누적 = 512 FMA/clk/SM,
    FP16·FP16 누적 = 1024 FMA/clk/SM (Ampere 실측, Ada 동일 규격)      [7]
  → P(fp32 acc) = N_SM × 512 × 2 × f_clk   [FLOP/s]
  → P(fp16 acc) = N_SM × 1024 × 2 × f_clk  [FLOP/s]
  (RTX 4070 SUPER: 56 SM × 512 × 2 × 2.475 GHz ≈ 142 TFLOPS, 언록 시)
"""
import argparse
import json
import statistics
import time

import torch

FMA_PER_SM_FP32_ACC = 512   # FMA/clk/SM — FP16/BF16 입력, FP32 누적 [7]
FMA_PER_SM_FP16_ACC = 1024  # FMA/clk/SM — FP16 입력, FP16 누적 [7]


def get_sm_clock() -> float:
    """실측 SM 클럭(MHz). PyTorch 버전에 따라 없으면 0."""
    try:
        clock_mhz = float(torch.cuda.clock_rate())
    except (AttributeError, RuntimeError):
        return 0.0
    return clock_mhz


def bench_matmul(a: torch.Tensor, b: torch.Tensor, iters: int) -> float:
    """워밍업 3회 + iters 회 반복의 중앙값 실행 시간(초)."""
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


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

    clock_before = get_sm_clock()
    size, iters = args.size, args.iters

    def tflops_of(med: float) -> float:
        return 2.0 * size ** 3 / med / 1e12

    print(f"GPU    : {torch.cuda.get_device_name(0)} (SM {sm})")
    print(f"torch  : {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"GEMM   : {size}^3, 중앙값 {iters}회")
    print(f"SM 클럭 : {clock_before:.0f} MHz (측정 시작 시점)\n")

    # ---------- 1) 참조 경로: FP16·FP16 누적 (언록 무관, 항상 풀속) ----------
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    med_ref = bench_matmul(a, b, iters)
    del a, b
    torch.cuda.empty_cache()

    # ---------- 2) 제한 경로: FP16/BF16 · FP32 누적 (언록 대상) ----------
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    capped = {}
    for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        med = bench_matmul(a, b, iters)
        capped[dtype_name] = med
        del a, b
        torch.cuda.empty_cache()

    clock_after = get_sm_clock()
    clock_avg = (clock_before + clock_after) / 2.0

    # ---------- 3) 캘리브레이션: 이론 피크 & 효율 ----------
    def theoretical(fma_per_sm: int) -> float:
        if clock_avg <= 0:
            return 0.0
        return sm * fma_per_sm * 2.0 * (clock_avg * 1e6) / 1e12

    theo_fp32 = theoretical(FMA_PER_SM_FP32_ACC)   # 언록 시 도달 가능한 상한
    theo_fp16 = theoretical(FMA_PER_SM_FP16_ACC)   # 참조 경로 상한

    ref_tflops = tflops_of(med_ref)
    results = {}
    for name, med in capped.items():
        tflops = tflops_of(med)
        results[name] = {
            "median_s": med,
            "tflops": tflops,
            "efficiency": (tflops / theo_fp32) if theo_fp32 > 0 else 0.0,
        }

    # ---------- 4) 출력 ----------
    print("== 캘리브레이션 ==")
    print(f"  이론 피크(FP32 누적, {clock_avg:.0f} MHz): {theo_fp32:6.1f} TFLOPS")
    print(f"  이론 피크(FP16 누적, {clock_avg:.0f} MHz): {theo_fp16:6.1f} TFLOPS")
    print(f"  참조 경로(fp16_acc): {ref_tflops:7.1f} TFLOPS "
          f"(효율 {ref_tflops / theo_fp16 * 100:.1f}%)" if theo_fp16 > 0
          else f"  참조 경로(fp16_acc): {ref_tflops:7.1f} TFLOPS")
    print("== 제한 경로 (언록 대상) ==")
    for name, r in results.items():
        eff = f"효율 {r['efficiency'] * 100:.1f}%" if r["efficiency"] > 0 else ""
        print(f"  [{name}] {r['median_s'] * 1000:8.1f} ms -> {r['tflops']:7.1f} TFLOPS  {eff}")

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
                "clock_mhz_avg": clock_avg,
                "theoretical_fp32acc_tflops": theo_fp32,
                "theoretical_fp16acc_tflops": theo_fp16,
                "reference": {
                    "dtype": "fp16_acc",
                    "median_s": med_ref,
                    "tflops": ref_tflops,
                    "efficiency": (ref_tflops / theo_fp16) if theo_fp16 > 0 else 0.0,
                },
            },
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
