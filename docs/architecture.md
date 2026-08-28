# TorchBurn Architecture

## System Overview

```
User Code (Python)
    │
    ▼
torch.compile(model, backend="torchburn")
    │
    ▼
┌─────────────────────────────────────┐
│  torch._dynamo → FX GraphModule    │  ← PyTorch traces the model
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  FX Graph → JSON Payload Parser     │  ← Classifies every node
│  (python/torchburn/_parser.py)      │
│                                     │
│  • Maps aten.* targets to ops       │
│  • Slices contiguous supported runs │
│  • Handles tuple decomposition      │
│  • Falls back unsupported nodes     │
└─────────────────────────────────────┘
    │
    ├── Supported nodes ──────────►  ┌─────────────────────────────┐
    │                               │  Native Rust Engine          │
    │                               │  (src/engine.rs)             │
    │                               │                             │
    │                               │  ├─ BLAKE3 cache lookup     │
    │                               │  ├─ Graph fusion             │
    │                               │  ├─ Execution dispatch       │
    │                               │  └─ Optional: Burn wgpu      │
    │                               └─────────────────────────────┘
    │
    └── Unsupported nodes ────────►  ┌─────────────────────────────┐
                                    │  Native PyTorch Eager        │
                                    │  (fallback with warning)     │
                                    └─────────────────────────────┘
```

## Data Flow

### Zero-Copy DLPack (REQ-003)

1. Python calls `tensor.__dlpack__()` → returns `PyCapsule` wrapping `DLManagedTensor*`
2. Rust reads raw pointer from capsule → computes **in place on PyTorch's memory**
3. Output tensors wrapped in new capsules with destructor → memory freed exactly once

```
Python Tensor ──.__dlpack__()──► PyCapsule ──► Rust raw pointer
                                                  │
                                          Compute in place (O(1))
                                                  │
                                          New PyCapsule ◄── OwnedTensor
                                                  │
                                          torch.from_dlpack()
                                                  │
                                          Python Tensor (output)
```

### BLAKE3 Graph Caching (REQ-004)

```
Graph JSON payload
    │
    ▼
BLAKE3 hash(payload)  ──► Cache HIT  ──► Reuse compiled kernel
    │
    └── Cache MISS ──► Compile new kernel ──► Store in GRAPH_CACHE
```

Thread-safe `RwLock<HashMap<String, Value>>` (BLAKE3 hex -> canonical JSON) with atomic hit/miss counters and 1024-entry LRU eviction.

## Module Structure

```
src/
├── lib.rs              # PyO3 module + Python exports
├── dlpack.rs           # Zero-copy DLPack FFI bridge
├── engine.rs           # Payload execution + dispatch
├── cache.rs            # BLAKE3 structural graph cache
├── fusion.rs           # Graph-level operator fusion
├── autograd.rs         # Rust-side autograd tape
│
├── ops.rs              # Phase 1: elementwise (add/sub/mul/div/relu)
├── activations.rs      # Phase 2: sigmoid, tanh, gelu, etc.
├── math_ops.rs         # Phase 2: abs, exp, log, pow, comparisons
├── reductions.rs       # Phase 2: sum, mean, max, argmax, std
├── linalg.rs           # Phase 2: matmul, linear, addmm, dot
├── norm.rs             # Phase 2: layer_norm, batch_norm, rms_norm
├── shape_ops.rs        # Phase 2: cat, stack, reshape, permute, select
├── convolution.rs      # Phase 3: conv1d, conv2d, conv_transpose
├── pooling.rs          # Phase 3: max_pool, avg_pool, adaptive_pool
├── upsample.rs         # Phase 3: interpolate, nearest, bilinear
├── embedding.rs        # Phase 4: embedding lookup + index_select
├── attention.rs        # Phase 4: SDPA, rope (SIMD + rayon)
├── losses.rs           # Phase 4: nll_loss, mse_loss, cross_entropy
├── ops_phase7.rs       # Phase 7: scatter, sort, repeat, prelu, einsum
├── burn_engine.rs      # Optional: Burn ndarray/wgpu backend
└── wgpu_backend.rs     # Optional: wgpu GPU dispatch

python/torchburn/
├── __init__.py         # Package init + backend registration
├── _backend.py         # torch._dynamo backend registration
├── _parser.py          # FX graph → JSON payload parser
├── _compiled.py        # torch.compile callable wrapper
├── _cache.py           # Python-side cache API
├── autograd.py         # Python autograd tape + backward
└── ops.py              # Python API for native ops
```

## Execution Engines

| Engine | Description | When to use |
|--------|-------------|-------------|
| `native_cpu` | Pure Rust, SIMD + rayon | Default, always available |
| `burn` | Burn ndarray CPU backend | When Burn-specific optimizations needed |
| `burn-wgpu` | Burn wgpu GPU (Metal/Vulkan/DX12) | GPU acceleration |

Select via `TORCHBURN_ENGINE=native_cpu|burn|burn-wgpu`.

## Thread Safety

- `GRAPH_CACHE`: `RwLock` (read-heavy, write-rare)
- Autograd tape: thread-local via `thread_local!`
- Burn engine: `py.allow_threads()` releases GIL during compute
- All public types: `Send + Sync`

## Fallback Mechanism

When an op is unsupported:
1. Parser marks node as `unsupported`
2. Contiguous supported/unsupported runs are sliced into chunks
3. Unsupported chunks execute via native PyTorch eager
4. `UserWarning` emitted (non-blocking) with op details
5. Correctness guaranteed — output matches PyTorch eager
