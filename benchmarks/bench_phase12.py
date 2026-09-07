"""torchburn Phase 1 + Phase 2 optimization benchmark.

Measures before/after kernel performance on the local machine:
  1. CPU int4 GEMV micro-benchmark (v1 vs v2 interleaved layout)
  2. End-to-end Qwen2.5-0.5B int4 decode throughput (tok/s), CPU vs iGPU

Usage:
    python benchmarks/bench_phase12.py --yes          # full run (~350MB download on first use)
    python benchmarks/bench_phase12.py --gemv-only    # kernel-only, no model download
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench_gemv():
    """Kernel-level A/B: v1 (per-row + f32 scales) vs v2 (interleaved + f16
    scales), via the Python-facing quantized-linear ops on realistic shapes."""
    import torch

    import torchburn as tb
    from torchburn import _torchburn as _native

    print("=" * 70)
    print("  Quantized-linear benchmark: v1 (per-row) vs v2 (interleaved fused)")
    print("=" * 70)
    print(f"  CPU tier: {_native.cpu_features_report()}")

    # Shapes matching Qwen2.5-0.5B int4 decoder projections.
    shapes = [
        ("qkv     (n=768,   k=896)", 768, 896),
        ("o       (n=896,   k=896)", 896, 896),
        ("down    (n=896,   k=4864)", 896, 4864),
        ("lm_head (n=151936, k=896)", 151936, 896),
    ]
    group = 64
    gen = torch.Generator().manual_seed(42)
    for name, n, k in shapes:
        w = torch.randn(n, k, generator=gen) * 0.02
        x = torch.randn(1, k, generator=gen)

        w_v1, s_v1 = tb.quantize_weight_int4_grouped(w, group_size=group)
        w_v2 = tb.quantize_weight_int4_grouped_v2(w, group_size=group)

        out_v1 = tb.w4a32_grouped_linear(x, w_v1, s_v1, group_size=group)
        out_v2 = tb.w4a32_grouped_linear_v2(x, w_v2, group_size=group)
        max_diff = (out_v1 - out_v2).abs().max().item()

        reps = 50 if n < 100000 else 10
        t0 = time.perf_counter()
        for _ in range(reps):
            tb.w4a32_grouped_linear(x, w_v1, s_v1, group_size=group)
        t_v1 = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            tb.w4a32_grouped_linear_v2(x, w_v2, group_size=group)
        t_v2 = (time.perf_counter() - t0) / reps

        print(f"  {name:26s} v1: {t_v1*1e3:8.3f} ms | v2: {t_v2*1e3:8.3f} ms"
              f" | speedup: {t_v1/t_v2:5.2f}x | max|d|: {max_diff:.2e}")


def bench_llm(device: str):
    """End-to-end decode throughput on Qwen2.5-0.5B-Instruct int4."""
    import torchburn as tb

    print("=" * 70)
    print(f"  LLM decode benchmark: Qwen2.5-0.5B-Instruct int4, device={device}")
    print("=" * 70)
    llm = tb.LLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", quant="int4", device=device)

    # Warmup (compiles wgpu pipelines / heats caches).
    list(llm.stream("Warmup.", max_tokens=8, temperature=0.7))

    rates = []
    for i in range(3):
        start = time.perf_counter()
        ntok = 0
        for _tok in llm.stream(
            "Explain what a torch.compile backend does in two sentences.",
            max_tokens=64,
            temperature=0.7,
        ):
            ntok += 1
        elapsed = time.perf_counter() - start
        rates.append(ntok / elapsed)
        print(f"  run {i + 1}: {ntok} tokens in {elapsed:.2f}s = {ntok / elapsed:.1f} tok/s")

    print(f"  median: {statistics.median(rates):.1f} tok/s | best: {max(rates):.1f} tok/s")
    return statistics.median(rates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip download confirmation")
    ap.add_argument("--gemv-only", action="store_true", help="kernel benchmark only")
    args = ap.parse_args()

    if not args.gemv_only and not args.yes:
        print("LLM benchmark downloads ~350MB from Hugging Face. "
              "Re-run with --yes, or use --gemv-only.")
        sys.exit(0)

    bench_gemv()
    if args.gemv_only:
        return

    print()
    cpu = bench_llm("cpu")
    print()
    igpu = bench_llm("gpu")
    print()
    print("=" * 70)
    print(f"  SUMMARY  CPU: {cpu:.1f} tok/s | iGPU: {igpu:.1f} tok/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
