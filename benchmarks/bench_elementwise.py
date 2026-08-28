"""Benchmark: TorchBurn vs eager PyTorch on elementwise chains.

Reports:
  * per-call latency of a compiled 24-op chain,
  * single-op graph latency (FFI + interpreter + one kernel),
  * the same workload in eager PyTorch for reference.

Caveat (v0.3): the native engine runs one kernel per graph node — operator
fusion (single-pass execution) is the Burn engine's roadmap deliverable
(REQ-004), so multi-op chains trail eager torch until fusion lands. Single
kernels are comparable.

Run with:  python benchmarks/bench_elementwise.py
"""

from __future__ import annotations

import time

import torch
import torchburn


def chain(x):
    y = x
    for _ in range(8):
        y = torch.relu(y * 1.001 + 0.5) - 0.25
    return y


def bench(fn, x, iters=50, warmup=10):
    for _ in range(warmup):
        fn(x)
    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    return (time.perf_counter() - start) / iters


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, 1024)

    t_eager = bench(chain, x)
    compiled = torch.compile(chain, backend="torchburn")
    t_tb = bench(compiled, x)

    single = torch.compile(lambda t: t + 1.0, backend="torchburn")
    t_single = bench(single, x)

    print(f"engine:                    {torchburn._torchburn.active_engine()}")
    print(f"eager 24-op chain:         {t_eager * 1e3:8.3f} ms")
    print(f"torchburn 24-op chain:     {t_tb * 1e3:8.3f} ms   ({t_eager / t_tb:.2f}x vs eager, unfused)")
    print(f"torchburn single add call: {t_single * 1e6:8.1f} µs  (interpreter + FFI + 1 kernel)")
    print(f"cache stats:               {torchburn.cache_stats()}")
    print("note: op fusion (REQ-004) is the Burn engine's next deliverable; per-op")
    print("      kernels run today, so multi-op chains are expected to trail eager.")


if __name__ == "__main__":
    main()
