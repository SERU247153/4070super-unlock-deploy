#!/usr/bin/env python3
"""
gpu-unlock 전/후 벤치마크
FP16 / BF16 대형 GEMM (FP32 accumulate 경로) 처리량 측정.
- 이 경로가 바로 NVIDIA가 반속으로 잠가둔 연산입니다.
- 언록 전: 약 절반 속도 / 언록 후: 약 2배 (RTX 4070 SUPER 기준 ~70 → ~140 TFLOPS)
도커 컨테이너 안에서 실행됩니다 (04_benchmark.sh 참고).
"""
import argparse
import json
import statistics
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8192, help="정방 행렬 크기 N (N^3 GEMM)")
    ap.add_argument("--iters", type=int, default=7, help="측정 반복 횟수 (중앙값 사용)")
    ap.add_argument("--label", default="unknown", help="결과 라벨 (before/after)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    # TF32 경유 금지, FP16 축적도 FP32로 강제 → 측정 대상이 '잠긴 경로'가 되도록
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    assert torch.cuda.is_available(), "CUDA 사용 불가"
    props = torch.cuda.get_device_properties(0)

    result = {
        "label": args.label,
        "gpu": torch.cuda.get_device_name(0),
        "sm": props.multi_processor_count,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "size": args.size,
        "iters": args.iters,
        "results": {},
    }
    print(f"GPU    : {result['gpu']} (SM {result['sm']})")
    print(f"torch  : {torch.__version__} / CUDA {result['cuda']}")
    print(f"GEMM   : {args.size}^3, FP32 accumulate, median of {args.iters}\n")

    for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        a = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
        b = torch.randn(args.size, args.size, device="cuda", dtype=dtype)

        for _ in range(3):          # 워밍업
            torch.matmul(a, b)
        torch.cuda.synchronize()

        times = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.matmul(a, b)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        med = statistics.median(times)
        tflops = 2.0 * args.size ** 3 / med / 1e12
        result["results"][dtype_name] = {"median_s": med, "tflops": tflops}
        print(f"[{dtype_name}] {med * 1000:8.1f} ms  ->  {tflops:7.1f} TFLOPS")

        del a, b
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
