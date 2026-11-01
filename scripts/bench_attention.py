#!/usr/bin/env python3
"""
FlashAttention 실워크로드 벤치마크 (gpu-unlock 언록 후 성능 측정용)
========================================================================

측정: FlashAttention 2 커널의 fwd / fwd+bwd 처리량 (BF16, causal)
  · 이 커널은 FP32 누적 경로 → 언록 대상 (README 기준 +50~60% 기대)
  · 잠긴 상태의 4070 SUPER 텐서코어 상한 ≈ 71 TFLOPS (@2.475 GHz)
    → 측정값이 이를 크게 상회하면 언록 효과가 실측된 것

FLOP 관례 (FlashAttention 논문 기준):
  fwd       = 4·B·H·q·k·d
  fwd+bwd   = 12·B·H·q·k·d
"""
import argparse
import json
import statistics
import time

import torch
from flash_attn import flash_attn_func

LOCKED_CEILING = 71.0  # TFLOPS, 잠긴 4070 SUPER의 FP32 누적 상한 (참고값)


def bench_attention(batch, heads, seqlen, headdim, iters, causal, bwd):
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, heads, headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, heads, headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, heads, headdim, device="cuda", dtype=torch.bfloat16)
    if bwd:
        q = q.detach().requires_grad_()
        k = k.detach().requires_grad_()
        v = v.detach().requires_grad_()

    def step():
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
        if bwd:
            out.sum().backward()
        torch.cuda.synchronize()

    for _ in range(3):
        step()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    headdim = 128
    heads = 32
    cases = [
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (2, 2048),
    ]

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"flash_attn {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"잠긴 상태 텐서코어 상한(참고): {LOCKED_CEILING} TFLOPS\n")

    results = []
    print(f"{'case':>18} {'mode':>9} {'median(ms)':>11} {'TFLOPS':>9} {'vs잠금상한':>10}")
    for batch, seqlen in cases:
        for bwd_i, mode in ((False, "fwd"), (True, "fwd+bwd")):
            med = bench_attention(batch, heads, seqlen, headdim, args.iters, True, bwd_i)
            flops = (12 if bwd_i else 4) * batch * heads * seqlen * seqlen * headdim
            tflops = flops / med / 1e12
            flag = "✅" if tflops > LOCKED_CEILING * 1.1 else "⬜"
            print(f"{f'B{batch}xH{heads}xS{seqlen}':>18} {mode:>9} {med*1000:>10.1f} {tflops:>8.1f} {tflops/LOCKED_CEILING:>9.2f}x {flag}")
            results.append({
                "batch": batch, "heads": heads, "seqlen": seqlen, "headdim": headdim,
                "mode": mode, "causal": True, "median_s": med, "tflops": tflops,
            })

    print("\n✅ = 잠긴 상한(71 TFLOPS) 초과 — 언록이 어텐션 경로에 반영됨")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "results": results}, f, indent=2)
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
