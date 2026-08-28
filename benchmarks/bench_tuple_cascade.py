"""Benchmark: eager fallback vs pure native to quantify the unbind/chunk cascade gap.

Tests three transformer variants:
  1. Pure native — SDPA-based (no unbind/chunk, all ops natively dispatched)
  2. Chunk cascade — QKV via chunk(3) → getitem → matmul (falls back to eager)
  3. Unbind cascade — QKV via unbind → getitem → matmul (falls back to eager)

Also benchmarks standalone ops: sort (values-only vs values+indices),
max/min (values-only vs values+indices), and elementwise chains.

Run:
    python benchmarks/bench_tuple_cascade.py
"""
from __future__ import annotations

import statistics
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchburn


# ── Models ──────────────────────────────────────────────────────────────

class TransformerSDPA(nn.Module):
    """Pure native path: uses F.scaled_dot_product_attention (no unbind/chunk)."""

    def __init__(self, d: int = 128, heads: int = 4, ff: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.heads = heads
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class TransformerChunk(nn.Module):
    """Chunk cascade: QKV via chunk(3) → getitem(0/1/2) → matmul."""

    def __init__(self, d: int = 128, heads: int = 4, ff: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.heads = heads
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        b, s, _ = x.shape
        h, hd = self.heads, self.d // self.heads
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, h, hd).transpose(1, 2)
        k = k.view(b, s, h, hd).transpose(1, 2)
        v = v.view(b, s, h, hd).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d)
        x = x + self.proj(out)
        x = self.ln2(x)
        x = x + self.ff2(F.relu(self.ff1(x)))
        return x


class TransformerUnbind(nn.Module):
    """Unbind cascade: QKV via unbind → getitem → matmul."""

    def __init__(self, d: int = 128, heads: int = 4, ff: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.heads = heads
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        b, s, _ = x.shape
        h, hd = self.heads, self.d // self.heads
        qkv = self.qkv(x).reshape(b, s, 3, h, hd)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d)
        x = x + self.proj(out)
        x = self.ln2(x)
        x = x + self.ff2(F.relu(self.ff1(x)))
        return x


# ── Bench helpers ───────────────────────────────────────────────────────

def bench(fn, x, rounds: int = 5, warmup: int = 10, iters: int = 30) -> float:
    """Median per-call latency (seconds)."""
    latencies = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn(x)
        start = time.perf_counter()
        for _ in range(iters):
            fn(x)
        latencies.append((time.perf_counter() - start) / iters)
    return statistics.median(latencies)


def count_warnings(fn, x) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(x)
    return sum(1 for w in caught if "torchburn" in str(w.message))


def fallback_report(mod, x) -> tuple[int, int, list[str]]:
    from torchburn._parser import parse_graph
    result = torch._dynamo.export(mod)(x)
    gm = result.gm if hasattr(result, "gm") else result[0]
    plan, _ = parse_graph(gm, [x])
    nodes = plan["nodes"]
    unsupported = [n for n in nodes if n["op"] == "unsupported"]
    targets = sorted({u.get("fx_target", u.get("target", "?")) for u in unsupported})
    return len(nodes), len(unsupported), targets


def verify(mod, x, label: str):
    compiled = torch.compile(mod, backend="torchburn")
    with torch.no_grad():
        out = compiled(x)
        ref = mod(x)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    diff = (out - ref).abs().max().item()
    print(f"  {label:40s}  correct={ok}  max_diff={diff:.2e}")
    return ok


# ── Main ────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    engine = torchburn._torchburn.active_engine()
    print(f"Engine: {engine}")
    print()

    B, S, D, H, FF = 4, 64, 128, 4, 512

    models = {
        "SDPA (pure native)": TransformerSDPA(D, H, FF),
        "Chunk cascade": TransformerChunk(D, H, FF),
        "Unbind cascade": TransformerUnbind(D, H, FF),
    }

    # ── Correctness ──
    print("=== Correctness ===")
    for label, mod in models.items():
        mod.eval()
        x = torch.randn(B, S, D)
        verify(mod, x, label)
    print()

    # ── Fallback coverage ──
    print("=== Fallback coverage ===")
    for label, mod in models.items():
        mod.eval()
        x = torch.randn(B, S, D)
        total, unsupported, targets = fallback_report(mod, x)
        pct = unsupported / total * 100 if total else 0
        print(f"  {label:40s}  {unsupported}/{total} unsupported ({pct:.1f}%)")
        if targets:
            for t in targets[:5]:
                print(f"    -> {t}")
    print()

    # ── Performance ──
    print("=== Performance (ms/call, steady-state) ===")
    results = {}
    for label, mod in models.items():
        mod.eval()
        x = torch.randn(B, S, D)
        compiled = torch.compile(mod, backend="torchburn")
        # warmup
        with torch.no_grad():
            for _ in range(5):
                compiled(x)
                mod(x)
        t_eager = bench(lambda t: mod(t), x)
        t_tb = bench(lambda t: compiled(t), x)
        warns = count_warnings(lambda t: compiled(t), x)
        speedup = t_eager / t_tb
        results[label] = (t_eager, t_tb, speedup, warns)
        print(f"  {label:40s}  eager={t_eager*1e3:7.2f}ms  torchburn={t_tb*1e3:7.2f}ms  "
              f"speedup={speedup:.2f}x  warnings={warns}")
    print()

    # ── Standalone op benchmarks ──
    print("=== Standalone op benchmarks ===")

    # Sort: values-only (native) vs values+indices (eager)
    class SortValuesOnly(nn.Module):
        def forward(self, x):
            v, _ = torch.sort(x, dim=-1)
            return v

    class SortBoth(nn.Module):
        def forward(self, x):
            v, i = torch.sort(x, dim=-1)
            return v + i.float()

    # Max: values-only (native) vs values+indices (eager)
    class MaxValuesOnly(nn.Module):
        def forward(self, x):
            v, _ = torch.max(x, dim=-1)
            return v

    class MaxBoth(nn.Module):
        def forward(self, x):
            v, i = torch.max(x, dim=-1)
            return v + i.float()

    # Elementwise chain (pure native)
    class ElemChain(nn.Module):
        def forward(self, x):
            return torch.relu(torch.sigmoid(torch.add(x, 1.0)))

    op_models = {
        "sort values-only": (SortValuesOnly(), torch.randn(B, S, 256)),
        "sort values+indices": (SortBoth(), torch.randn(B, S, 256)),
        "max values-only": (MaxValuesOnly(), torch.randn(B, S, 256)),
        "max values+indices": (MaxBoth(), torch.randn(B, S, 256)),
        "elemwise chain (add->sigmoid->relu)": (ElemChain(), torch.randn(B, S, 256)),
    }

    for label, (mod, x) in op_models.items():
        mod.eval()
        compiled = torch.compile(mod, backend="torchburn")
        with torch.no_grad():
            for _ in range(5):
                compiled(x)
                mod(x)
        t_eager = bench(lambda t: mod(t), x)
        t_tb = bench(lambda t: compiled(t), x)
        warns = count_warnings(lambda t: compiled(t), x)
        speedup = t_eager / t_tb
        print(f"  {label:40s}  eager={t_eager*1e3:7.3f}ms  torchburn={t_tb*1e3:7.3f}ms  "
              f"speedup={speedup:.2f}x  warnings={warns}")
    print()

    # -- Summary --
    print("=== Summary ===")
    sdpa_eager, sdpa_tb, sdpa_sp, _ = results["SDPA (pure native)"]
    chunk_eager, chunk_tb, chunk_sp, _ = results["Chunk cascade"]
    unbind_eager, unbind_tb, unbind_sp, _ = results["Unbind cascade"]
    print(f"  Pure native (SDPA):     {sdpa_sp:.2f}x vs eager")
    print(f"  Chunk cascade:          {chunk_sp:.2f}x vs eager  "
          f"({(1 - chunk_sp/sdpa_sp)*100:.1f}% slower than native)")
    print(f"  Unbind cascade:         {unbind_sp:.2f}x vs eager  "
          f"({(1 - unbind_sp/sdpa_sp)*100:.1f}% slower than native)")
    print()
    print("  The gap between SDPA and chunk/unbind variants quantifies the")
    print("  performance cost of the unbind/chunk eager-fallback cascade.")


if __name__ == "__main__":
    main()
