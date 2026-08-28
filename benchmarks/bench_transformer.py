"""Benchmark: transformer block (embedding + MHA + MLP + layer_norm) end-to-end.

Compiles a real ``nn.Module`` block with ``torch.compile(backend="torchburn")``
and compares steady-state per-call latency against eager PyTorch on the same
input.  Also reports:

  * first-call latency (dynamo trace + BLAKE3 cache miss + plan build),
  * fallback coverage: how many graph nodes the parser classified as
    ``unsupported`` (and which targets), plus any runtime eager fallbacks
    observed during a steady-state call,
  * correctness (compiled vs eager, allclose).

Run once per engine and compare:

    python benchmarks/bench_transformer.py                      # native CPU
    TORCHBURN_ENGINE=burn python benchmarks/bench_transformer.py
    TORCHBURN_ENGINE=burn-wgpu python benchmarks/bench_transformer.py

Caveats:
  * ``torch.compile`` caches the plan per (signature, shape); the first call
    pays tracing/compilation, steady-state calls reuse it.
  * The transformer block decomposes into the Phase 1-4 op set (embedding,
    linear, reshape/permute/select, SDPA, layer_norm, add, relu) — every op
    is natively supported, so the only parser-level fallbacks are the
    ``aten.sym_size.int`` metadata nodes produced by ``x.shape`` (they
    resolve to plain ints and never touch the engine).
"""

from __future__ import annotations

import statistics
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchburn
from torchburn._parser import parse_graph


class TransformerBlock(nn.Module):
    """One pre-norm transformer block: embedding -> LN -> MHA -> LN -> MLP."""

    def __init__(self, vocab: int = 2048, d: int = 128, heads: int = 4, ff: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.heads = heads
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.ln1(x)
        b, s, _ = x.shape
        h, hd = self.heads, self.d // self.heads
        qkv = self.qkv(x).reshape(b, s, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.reshape(b, s, self.d))
        x = self.ln2(x)
        x = x + self.ff2(F.relu(self.ff1(x)))
        return x


def _bench(fn, x, rounds: int = 3, warmup: int = 5, iters: int = 15) -> float:
    """Median per-call latency over several rounds of warmup + timed iters."""
    latencies = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn(x)
        start = time.perf_counter()
        for _ in range(iters):
            fn(x)
        latencies.append((time.perf_counter() - start) / iters)
    return statistics.median(latencies)


def _fallback_report(mod, x) -> tuple[int, int, list[str]]:
    """(total nodes, unsupported nodes, unsupported targets) from the plan.

    ``torch.compile`` wraps the backend callable in an OptimizedModule, so the
    live plan is reached by re-exporting the same graph (identical for the
    same module + input) and parsing it — this is exactly what the backend
    does at compile time.
    """
    result = torch._dynamo.export(mod)(x)
    gm = result.gm if hasattr(result, "gm") else result[0]
    plan, _ = parse_graph(gm, [x])
    nodes = plan["nodes"]
    unsupported = [n for n in nodes if n["op"] == "unsupported"]
    targets = sorted({u.get("fx_target", u.get("target", "?")) for u in unsupported})
    return len(nodes), len(unsupported), targets


def _runtime_fallbacks(fn, x) -> int:
    """Count torchburn eager-fallback warnings emitted during a call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(x)
    return sum(1 for w in caught if "torchburn" in str(w.message))


def main() -> None:
    torch.manual_seed(0)
    engine = torchburn._torchburn.active_engine()
    print(f"engine: {engine}")
    print()

    mod = TransformerBlock()
    x = torch.randint(0, 2048, (4, 64))  # int64 indices -> embedding lookup

    # Correctness first: compiled output must match eager.
    compiled = torch.compile(mod, backend="torchburn")
    with torch.no_grad():
        out = compiled(x)
    with torch.no_grad():
        ref = mod(x)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    print(f"correctness (compiled vs eager, allclose): {ok}")
    print()

    # First-call latency = dynamo trace + BLAKE3 signature + plan build.
    torchburn.cache_clear()
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(x)
    t_first = time.perf_counter() - t0

    # Steady-state: eager vs compiled on the cached plan.
    t_eager = _bench(lambda t: mod(t), x)
    t_tb = _bench(lambda t: compiled(t), x)
    runtime_fallbacks = _runtime_fallbacks(lambda t: compiled(t), x)

    total, unsupported, targets = _fallback_report(mod, x)

    print(f"first call (trace + compile + cache miss): {t_first * 1e3:8.1f} ms")
    print(f"eager torch, steady-state:                 {t_eager * 1e3:8.2f} ms/call")
    print(f"torchburn (torch.compile), steady-state:   {t_tb * 1e3:8.2f} ms/call")
    print(f"speedup vs eager:                          {t_eager / t_tb:6.2f}x")
    print()
    print(f"graph: {total} plan nodes, {unsupported} unsupported "
          f"({unsupported / total * 100:.1f}% fallback coverage)")
    print(f"unsupported targets: {targets or 'none'}")
    print(f"runtime eager fallbacks per call: {runtime_fallbacks}")
    print(f"cache stats: {torchburn.cache_stats()}")
    print()
    print("note: 'aten.sym_size.int' nodes are x.shape metadata - they resolve")
    print("      to plain ints for reshape/permute and never touch the engine.")


if __name__ == "__main__":
    main()
