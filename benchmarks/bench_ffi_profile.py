"""Profile FFI overhead to identify the exact bottleneck.

Measures:
1. JSON serialization time
2. DLPack capsule creation time
3. FFI call overhead (Rust side)
4. DLPack import time (result)
5. Python interpreter loop overhead
6. Total compiled call time
"""
import time
import statistics
import json
import torch
import torch.nn.functional as F

import torchburn
from torchburn._compiled import BurnCompiledCallable, _arg_key
from torchburn._parser import payload_json
from torchburn import _torchburn as _native


def _median_us(fn, n=50):
    """Run fn n times, return median in microseconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)


def profile_small_model():
    """Profile elementwise ops (add/mul) where overhead dominates."""
    print("=" * 70)
    print("PROFILING: Small model (3 elementwise ops)")
    print("=" * 70)

    def model(x):
        return (x + x) * x

    gm = torch.fx.symbolic_trace(model)
    x = torch.randn(16, 16)
    cb = BurnCompiledCallable(gm, [x])

    # Warmup
    for _ in range(5):
        cb(x)

    # 1. Full compiled call
    total_us = _median_us(lambda: cb(x))
    print(f"\n  Full compiled call:         {total_us:>8.1f} us")

    # 2. Python interpreter loop only (bypass FFI)
    env = {}
    for node in cb.plan["nodes"]:
        if node["op"] == "placeholder" and node["index"] >= 0:
            env[node["id"]] = x
    interpret_us = _median_us(lambda: cb._interpret(dict(env)))
    print(f"  Python interpreter loop:    {interpret_us:>8.1f} us")

    # 3. DLPack capsule creation
    dlpack_us = _median_us(lambda: [x.__dlpack__()])
    print(f"  DLPack capsule creation:    {dlpack_us:>8.1f} us (1 tensor)")

    # 4. JSON serialization (build correct payload)
    input_specs = [cb._spec(x)] + [{"shape": [], "dtype": "f32"}] * (len(cb._combined_input_keys) - 1)
    payload = {
        "inputs": input_specs,
        "nodes": cb._combined_payload_template["nodes"],
        "outputs": cb._combined_output_ids,
    }

    json_us = _median_us(lambda: payload_json(payload))
    n_nodes = len(cb._combined_payload_template["nodes"])
    print(f"  JSON serialization:         {json_us:>8.1f} us ({n_nodes} nodes)")

    # 5. Rust FFI call
    pjson = payload_json(payload)
    capsules = [x.__dlpack__()] + [torch.tensor(float(k[2]) if k[0] == "const" else 0.0).__dlpack__() for k in cb._combined_input_keys[1:]]
    ffi_us = _median_us(lambda: _native.execute(pjson, capsules))
    print(f"  Rust FFI execute:           {ffi_us:>8.1f} us")

    # 6. Eager baseline
    eager_us = _median_us(lambda: model(x))
    print(f"  Eager PyTorch baseline:     {eager_us:>8.1f} us")
    print(f"\n  Overhead ratio: {total_us / eager_us:.1f}x")


def profile_matmul():
    """Profile matmul where kernel performance matters."""
    print("\n" + "=" * 70)
    print("PROFILING: Matmul (256x256 @ 256x256)")
    print("=" * 70)

    def model(x, w):
        return x @ w

    gm = torch.fx.symbolic_trace(model)
    x = torch.randn(256, 256)
    w = torch.randn(256, 256)
    cb = BurnCompiledCallable(gm, [x, w])

    for _ in range(5):
        cb(x, w)

    total_us = _median_us(lambda: cb(x, w))
    print(f"\n  Full compiled call:         {total_us:>8.1f} us")

    eager_us = _median_us(lambda: model(x, w))
    print(f"  Eager PyTorch baseline:     {eager_us:>8.1f} us")
    print(f"\n  Overhead ratio: {total_us / eager_us:.1f}x")

    # Sub-components
    n_native = sum(1 for p in cb._phases if p["kind"] == "native" for _ in p["nodes"])
    n_fallback = sum(1 for p in cb._phases if p["kind"] == "eager")
    print(f"  Native nodes: {n_native}, Fallback nodes: {n_fallback}")

    # JSON
    input_specs = [cb._spec(x), cb._spec(w)]
    if len(cb._combined_input_keys) > 2:
        input_specs += [{"shape": [], "dtype": "f32"}] * (len(cb._combined_input_keys) - 2)
    payload = {
        "inputs": input_specs[:len(cb._combined_input_keys)],
        "nodes": cb._combined_payload_template["nodes"],
        "outputs": cb._combined_output_ids,
    }
    json_us = _median_us(lambda: payload_json(payload))
    print(f"  JSON serialization:         {json_us:>8.1f} us")

    # DLPack
    dlpack_us = _median_us(lambda: [x.__dlpack__(), w.__dlpack__()])
    print(f"  DLPack capsule creation:    {dlpack_us:>8.1f} us (2 tensors)")

    # FFI
    pjson = payload_json(payload)
    capsules = [x.__dlpack__(), w.__dlpack__()]
    try:
        ffi_us = _median_us(lambda: _native.execute(pjson, capsules))
        print(f"  Rust FFI execute:           {ffi_us:>8.1f} us")
    except Exception as e:
        print(f"  Rust FFI execute:           FAILED ({e})")


def profile_transformer():
    """Profile a full transformer block."""
    print("\n" + "=" * 70)
    print("PROFILING: Transformer block (B=4, S=64, D=128, H=4, FF=512)")
    print("=" * 70)

    class TransformerBlock(torch.nn.Module):
        def __init__(self, d_model=128, nhead=4, dim_ff=512):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.d_head = d_model // nhead

            self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)

            self.fc1 = torch.nn.Linear(d_model, dim_ff)
            self.fc2 = torch.nn.Linear(dim_ff, d_model)

        def forward(self, x):
            B, S, D = x.shape
            H, DH = self.nhead, self.d_head

            residual = x
            x = self.norm1(x)

            q = self.q_proj(x).reshape(B, S, H, DH).transpose(1, 2)
            k = self.k_proj(x).reshape(B, S, H, DH).transpose(1, 2)
            v = self.v_proj(x).reshape(B, S, H, DH).transpose(1, 2)

            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, S, D)
            x = residual + self.out_proj(attn)

            residual = x
            x = self.norm2(x)
            x = residual + self.fc2(F.relu(self.fc1(x)))
            return x

    model = TransformerBlock().eval()
    gm = torch.fx.symbolic_trace(model)
    x = torch.randn(4, 64, 128)
    cb = BurnCompiledCallable(gm, [x])

    for _ in range(3):
        cb(x)

    total_us = _median_us(lambda: cb(x))
    eager_us = _median_us(lambda: model(x))

    print(f"\n  Full compiled call:         {total_us:>8.1f} us ({total_us/1000:.2f} ms)")
    print(f"  Eager PyTorch baseline:     {eager_us:>8.1f} us ({eager_us/1000:.2f} ms)")
    print(f"\n  Overhead ratio: {total_us / eager_us:.1f}x")

    # Graph stats
    n_native = sum(1 for p in cb._phases if p["kind"] == "native" for _ in p["nodes"])
    n_fallback = sum(1 for p in cb._phases if p["kind"] == "eager")
    n_native_phases = sum(1 for p in cb._phases if p["kind"] == "native")

    print(f"\n  Graph stats:")
    print(f"    Total nodes:             {len(cb.plan['nodes'])}")
    print(f"    Native nodes:            {n_native}")
    print(f"    Fallback nodes:          {n_fallback}")
    print(f"    FFI calls (native phases): {n_native_phases}")

    # Sub-components
    dlpack_us = _median_us(lambda: [x.__dlpack__()])
    print(f"\n  DLPack capsule creation:    {dlpack_us:>8.1f} us (1 tensor)")

    # Estimated breakdown
    n_groups = n_native_phases
    print(f"\n  Estimated breakdown:")
    print(f"    DLPack (all tensors):     ~{dlpack_us * max(1, n_native):>8.0f} us")
    print(f"    FFI overhead (per call):  ~{20 * n_groups:>8.0f} us (est. 20us * {n_groups} calls)")
    print(f"    Rust kernel execution:    ~{total_us * 0.85:>8.0f} us (est. 85%)")
    print(f"    Fallback eager:           ~{total_us * 0.10:>8.0f} us (est. 10%)")


if __name__ == "__main__":
    profile_small_model()
    profile_matmul()
    profile_transformer()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The profiling breaks down the overhead into components:

1. JSON serialization: Building the JSON string for the Rust engine
2. DLPack capsule creation: Converting PyTorch tensors to DLPack PyCapsules
3. Rust FFI execute: The actual Rust engine execution (JSON parse + kernel dispatch)
4. Python interpreter: The _interpret() loop walking the plan

v0.4 improvements:
- Single FFI call for all native nodes (was N calls per group)
- Pre-computed payload structure at init time
- Eliminated per-group JSON serialization at runtime
""")
