"""Benchmark: Burn wgpu (GPU) vs native CPU engine throughput.

Runs *identical* payloads through the raw FFI on the active engine and
reports per-call latency.  Run once per engine and compare:

    python benchmarks/bench_wgpu_vs_native.py                    # native CPU
    TORCHBURN_ENGINE=burn-wgpu python benchmarks/bench_wgpu_vs_native.py

Fair-test caveats:
  * Only ops the burn engine executes natively are real GPU workloads:
    elementwise math, activations, softmax, and (since the matmul/linear
    work) ``matmul``, ``bmm``, ``linear`` and ``addmm``.  Matrix-vector
    forms and other mixed-rank shapes still delegate to the native engine.
  * The wgpu path copies tensors into Burn-managed buffers on every call
    (kernel-owned memory, REQ-003 zero-copy belongs to the native engine);
    native is zero-copy DLPack.  Small workloads favour native, large
    compute-dense ones favour the GPU.  Both numbers are reported.
  * Payloads go through the raw FFI (no Python interpreter), which adds
    identical overhead to both engines — a pure engine-to-engine compare.
"""

from __future__ import annotations

import time

import torch
import torchburn
from torchburn import _torchburn as _native
from torchburn._parser import payload_json

torch.manual_seed(0)


def spec(t: torch.Tensor) -> dict:
    return {"dtype": "f32", "shape": list(t.shape)}


def run_payload(payload: dict, tensors: list[torch.Tensor]) -> list:
    capsules = [t.__dlpack__() for t in tensors]
    return _native.execute(payload_json(payload), capsules)


def bench(payload: dict, tensors: list[torch.Tensor], iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        run_payload(payload, tensors)
    start = time.perf_counter()
    for _ in range(iters):
        run_payload(payload, tensors)
    return (time.perf_counter() - start) / iters


def elementwise_chain_payload(n_ops: int, shape: list[int]) -> tuple[dict, list[torch.Tensor]]:
    """Alternating mul / add / sigmoid chain, all tensor-tensor ops.

    Slots: input x = 0; every op writes the next slot in order.
    """
    x = torch.randn(*shape)
    nodes = []
    slot = 0
    for i in range(n_ops):
        op = ["mul", "add", "sigmoid"][i % 3]
        if op == "sigmoid":
            args = [{"kind": "slot", "index": slot}]
        else:
            # second operand: x itself (slot 0) keeps shapes equal
            args = [{"kind": "slot", "index": slot}, {"kind": "slot", "index": 0}]
        slot += 1
        nodes.append({"id": i, "target": op, "args": args, "kwargs": {}})
    payload = {
        "inputs": [spec(x)],
        "nodes": nodes,
        "outputs": [n_ops - 1],
    }
    return payload, [x]


def softmax_payload(shape: list[int]) -> tuple[dict, list[torch.Tensor]]:
    x = torch.randn(*shape)
    payload = {
        "inputs": [spec(x)],
        "nodes": [
            {"id": 0, "target": "softmax", "args": [{"kind": "slot", "index": 0}], "kwargs": {"dim": -1}},
        ],
        "outputs": [0],
    }
    return payload, [x]


def small_add_payload() -> tuple[dict, list[torch.Tensor]]:
    x = torch.randn(64, 64)
    y = torch.randn(64, 64)
    payload = {
        "inputs": [spec(x), spec(y)],
        "nodes": [
            {"id": 0, "target": "add", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}], "kwargs": {}},
        ],
        "outputs": [0],
    }
    return payload, [x, y]


def matmul_payload(m: int, k: int, n: int) -> tuple[dict, list[torch.Tensor]]:
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    payload = {
        "inputs": [spec(a), spec(b)],
        "nodes": [
            {"id": 0, "target": "matmul", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}], "kwargs": {}},
        ],
        "outputs": [0],
    }
    return payload, [a, b]


def linear_payload(m: int, k: int, n: int) -> tuple[dict, list[torch.Tensor]]:
    x = torch.randn(m, k)
    w = torch.randn(n, k)  # torch linear weight layout [out, in]
    bias = torch.randn(n)
    payload = {
        "inputs": [spec(x), spec(w), spec(bias)],
        "nodes": [
            {"id": 0, "target": "linear", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}, {"kind": "slot", "index": 2}], "kwargs": {}},
        ],
        "outputs": [0],
    }
    return payload, [x, w, bias]


def main() -> None:
    engine = _native.active_engine()
    print(f"engine: {engine}")
    print()

    rows = []

    # 1) elementwise chain, 12 ops on 4096x4096 (16.8M elements)
    payload, tensors = elementwise_chain_payload(12, [4096, 4096])
    t = bench(payload, tensors)
    rows.append(("12-op elementwise chain, 4096x4096", t))

    # 2) softmax on 4096x4096
    payload, tensors = softmax_payload([4096, 4096])
    t = bench(payload, tensors)
    rows.append(("softmax, 4096x4096", t))

    # 3) softmax on 2048x2048 (16.8M elements — memory bound; stays under the
    #    128MB dedicated VRAM of the Intel Iris Xe iGPU found on this machine)
    payload, tensors = softmax_payload([2048, 2048])
    t = bench(payload, tensors, iters=20, warmup=5)
    rows.append(("softmax, 2048x2048", t))

    # 4) matmul 1024x1024x1024 (1 GFLOP)
    payload, tensors = matmul_payload(1024, 1024, 1024)
    t = bench(payload, tensors, iters=20, warmup=5)
    rows.append(("matmul 1024^3 (1 GFLOP)", t))

    # 5) matmul 2048x2048x2048 (8.6 GFLOP — compute-dense, GPU favourite)
    payload, tensors = matmul_payload(2048, 2048, 2048)
    t = bench(payload, tensors, iters=10, warmup=3)
    rows.append(("matmul 2048^3 (8.6 GFLOP)", t))

    # 6) linear 1024x2048->1024 with bias (typical MLP layer)
    payload, tensors = linear_payload(1024, 2048, 1024)
    t = bench(payload, tensors, iters=20, warmup=5)
    rows.append(("linear 1024x2048->1024 + bias", t))

    # 7) tiny add — shows per-call FFI + buffer overhead
    payload, tensors = small_add_payload()
    t = bench(payload, tensors, iters=200, warmup=50)
    rows.append(("single add, 64x64 (overhead probe)", t))

    width = max(len(name) for name, _ in rows)
    for name, t in rows:
        unit = "us" if t < 1e-3 else "ms"
        value = t * 1e6 if t < 1e-3 else t * 1e3
        print(f"{name:<{width}}  {value:10.2f} {unit}/call")

    print(f"\ncache stats: {torchburn.cache_stats()}")


if __name__ == "__main__":
    main()
