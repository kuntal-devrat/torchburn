# TorchBurn Master Optimization Roadmap

A phased, production-grade plan to make TorchBurn the fastest inference stack for
int4-quantized LLMs on consumer hardware (all CPUs, iGPUs, dGPUs, CUDA), with
PyTorch-native workflow as the differentiator. Every phase ends with green tests,
a release build, and measured numbers — no claims without benchmarks.

**Ground truth from this machine (Qwen2.5-0.5B int4, release + native build, 8 logical CPUs):**

| Path | Measured | Target |
|---|---|---|
| CPU Rust decoder, 1 thread | 18.1 tok/s | — |
| CPU Rust decoder, 8 threads | 53.4 tok/s | **70+** |
| iGPU (Iris Xe, Vulkan graph decoder) | 13.0 tok/s (flat vs workgroup size) | **50+** |
| Compiled Linear(512→1024) | 5.5 ms/call | < 1 ms |
| Memory bandwidth headroom | ~15% used (~7 GB/s of ~45) | decode is kernel-bound, not DRAM-bound |

**Optimization results (Sprint 1–4 complete):**

| Benchmark | Result | Change |
|---|---|---|
| GEMM f32 64³ | 12.2 µs | baseline |
| GEMM f64 64³ | 18.0 µs | **-9.7%** |
| GEMM f32 256³ | 328 µs | **-24.7%** |
| GEMM f32 1024³ | 13.5 ms | **-48.5%** |
| Decoder step | 3.46 ms/token | **-9.7%** |
| Int4 GEMV qkv | 262–336 µs | unchanged (not touched) |

---

## Phase 0 — Baseline, CI, and benchmark infrastructure

**Goal:** never ship a perf regression again; make every later phase measurable.

### 0.1 Benchmark harness
- New crate `benchmarks/` with `cargo bench` criterion suites:
  - `bench_gemm`: sgemm/dgemm at (64, 256, 1024, 4096) square and skinny-M shapes.
  - `bench_int4_gemv`: grouped int4 decode GEMV at Qwen shapes (hidden 896, heads 14, head_dim 64).
  - `bench_decoder_step`: full `decode_and_sample` per-token latency, isolated from Python.
  - `bench_wgpu_submit`: GPU command-stream submit + readback latency, no decode.
- Python side: `benchmarks/bench_llm.py` — end-to-end tok/s for CPU/iGPU/dGPU, saving JSON to
  `benchmarks/results/{date}_{hostname}.json` with hardware fingerprint (CPU model, adapter name, driver).
- Gate: every number is reproducible within 10% across 3 runs on the same machine.

### 0.2 CI split builds
- Current CI deletes `.cargo/config.toml` for portability (`ci.yml:33-34`). Add a second job
  building with `RUSTFLAGS="-C target-cpu=native"` for the self-hosted bench runner only.
- Publish two wheel variants: `torchburn` (portable, baseline ISA) and `torchburn-native`
  (built on the release runner, runtime-dispatched — see 0.3). Name wheels per PEP 600.
- Nightly job: run the Python benchmark suite on real hardware, post results to a branch
  (`perf-history`), and open an issue on >10% regression.

### 0.3 Runtime CPU feature dispatch (prerequisite for portable native speed)
- New module `src/dispatch.rs`: `is_x86_feature_detected!("avx512f" / "avx512_vnni" / "avx2" / "fma")`
  resolved **once** into a `static CpuFeatures` (OnceLock). All kernel entry points branch on it.
- Today the portable wheel silently runs scalar; after this it runs AVX2 where available.
- Tests: a `#[cfg(feature = "dispatch-test")]` path forcing each feature tier; assert parity
  of outputs (allclose) and monotonic performance across tiers on the dev machine.

### 0.4 Correctness harness before speed work
- `tests/parity/` — golden-value tests: every kernel (f32, int4-g64, int8) against reference
  implementations in Rust (`cargo test`) and against eager PyTorch (`python -m pytest tests/test_parity.py`).
- Record max-abs-diff budgets per op (e.g. int4 GEMV vs f32: rel err < 2e-2, matching current quality).

**Exit criteria:** criterion baselines committed; CI runs benches nightly; dispatch module merged
with parity tests green in all feature tiers; two wheel variants build in CI.

---

## Phase 1 — CPU decode: 53 → 70+ tok/s

**Goal:** make the int4 GEMV in `src/llm/decoder.rs` memory-bound (saturating bandwidth), not instruction-bound.

### 1.1 Weight layout: interleave at quantization time
- Current int4 packs nibbles row-major (`src/quantization.rs` group accumulation). Dequant happens
  per element *during* the dot product with non-sequential loads.
- Change the quantizer (Python, `quantize_model`) to emit pre-interleaved weights:
  pack 4 int4 values + one f16 scale into a 64-byte cache-line-aligned block, laid out for the
  GEMV's access pattern (K-major, blocked by 16 lanes).
- New safetensors format version `torchburn_int4_g64_v2`; keep a loader for v1 and a migration
  script (one-time requantize, ~seconds for 0.5B).
- Test: byte-level round-trip test (quantize → dequantize → compare to reference); loader accepts
  both v1/v2 with a deprecation warning.

### 1.2 Fused dequant-dot kernel
- Rewrite the GEMV inner loop: 8 outputs per iteration, one cache line of weights loaded per
  accumulation step, scales hoisted per group, accumulate in `i32` via `vpdpbusd` (AVX-512 VNNI)
  with AVX2 fallback using `vpmaddwd`.
- Verify codegen: `objdump`/`cargo asm` proof that `vpdpbusd` is emitted on the AVX-512 tier
  (banner says AVX-512 VNNI today — **unverified that the instruction actually fires**; test it).
- f32 fallback path uses the `gemm` crate (tile sizes tuned for Tiger Lake) instead of
  `matrixmultiply::sgemm` (`src/linalg.rs:12-13`) for skinny-M shapes; OpenBLAS stays behind
  its feature flag (`Cargo.toml:44-48`).
- Bench gate: ≥ 2x criterion improvement on `bench_int4_gemv` at Qwen shapes; end-to-end ≥ 70 tok/s
  at 8 threads on the dev machine; parity tests green.

### 1.3 Threading
- Rayon scaling is near-linear (18→53 tok/s, 1→8 threads). Add: `RAYON_NUM_THREADS` default =
  physical cores (not logical), P-core pinning hint via `core_affinity` crate where E-cores exist,
  and a size-based cutoff (the `m <= 8` pattern at `src/linalg.rs:343-361`) generalized so tiny
  GEMVs skip the pool entirely.
- Bench gate: no regression at 1/2 threads; ≥ 70 tok/s at 8 threads; latency of a single
  `decode_and_sample` (no threads spawned) unchanged.

### 1.4 Sampling and per-token overhead (minor but free)
- Replace the O(k) top-k rescan (`src/llm/decoder.rs:394-403`) with a bounded insertion into a
  fixed-size heap.
- Move `tokenizer.decode` per token (`engine.py:273`) out of the hot path: decode from the Rust
  side in chunks, or decode incrementally with a rolling buffer on complete UTF-8 boundaries.

**Exit criteria:** 70+ tok/s CPU end-to-end on the dev machine; criterion deltas recorded;
`cargo test` + `python -m pytest` green; release wheel built and re-benchmarked.

---

## Phase 2 — iGPU decode: 13 → 50 tok/s

**Goal:** attack the fixed per-token overhead; workgroup tuning measured flat (12.8–13.3 at
ROWS_PER_WG 2/8/16), so the wall is sync/readback latency, not shader occupancy.

### 2.1 Measure first
- Add wgpu timestamp queries to `record_and_submit_step` (`src/wgpu/decode.rs:509-535`): split
  per-token time into (a) GPU execution, (b) submit, (c) `poll(Wait)` blocking, (d) staging
  `map_async` readback. Log a breakdown per layer and for the final logits copy.
- Deliverable: a table proving where the ~77 ms/token goes. All Phase 2 work is prioritized by it.

### 2.2 Kill the blocking readback
- Replace `poll(Wait)` + `map_async` staging readback (`decode.rs:536-560`) with:
  pinned (MAP_READ | COPY_DST) staging buffers in a ring of N≥3, and a non-blocking poll loop
  that yields to Rayon (CPU can pre-fill KV writes for the next step while the readback completes).
- The token-dependency chain means only *one* step of overlap is available; capture it.
- Bench gate: `bench_wgpu_submit` per-token latency drops by the measured readback share;
  end-to-end ≥ 30 tok/s.

### 2.3 Rust-resident generate loop
- New `#[pyfunction] generate_loop(prompt_ids, max_tokens, sampler_config) -> list[str]`:
  the entire decode loop moves into Rust for both decoders. Python gets one FFI crossing per
  generation instead of one per token (`engine.py:269-325` currently does Python per token:
  dict packets, tokenizer decode, timing).
- Python `stream()` keeps its API by polling a Rust-side channel (pyo3 `crossbeam` bridge) so
  token streaming stays incremental.
- Bench gate: iGPU ≥ 40 tok/s, CPU ≥ 72 tok/s (removes per-token Python overhead measured at
  ~5-10% at current speeds, more at higher tok/s).

### 2.4 Shader-level wins (only if 2.1 shows GPU execution is the bottleneck)
- Fuse RoPE + KV-write into the attention shader (they are separate passes today).
- Try `TORCHBURN_ROWS_PER_WG` autotuning at model load: run 3 shapes, pick the winner per
  adapter (adapter-name-keyed cache, like the weight cache at `backend.rs:462-529`).
- Consider subgroup operations (`wgpu` feature) for the logits reduction.

### 2.5 device='auto' must be honest
- `UniversalEngine._setup_acceleration` (`engine.py:43-52`) picks iGPU whenever `gpu_info().available`.
  On this machine the CPU decoder is 4x faster. Change `auto` to: load quantized weights once,
  run a 20-token micro-benchmark on each available backend (CPU and iGPU), pick the winner,
  expose `engine.benchmark_devices()` for user override. Cache the decision per model+hardware
  fingerprint.

**Exit criteria:** iGPU ≥ 50 tok/s on Iris Xe *or* a documented blocker (if 2.1 shows the GPU
execution share is irreducible without new shaders, Phase 2.4 becomes the path); `auto` picks
the faster backend on every tested machine; full test suite green on CPU-only CI runners.

---

## Phase 3 — torch.compile backend parity with PyTorch eager

**Goal:** compiled graph should never be 9x slower than eager on any shape (measured 5.5 ms vs
0.69 ms on Linear(512→1024)).

### 3.1 Diagnose the per-call overhead
- The tiny-graph floor is 0.034 ms (`execute_prepared` on a 1-node ReLU) — so the 5.5 ms is in
  the linear kernel path, not the executor. Profile the fused GEMM step: is the bias-add +
  accumulation path (`engine.rs` fusion `Step::Gemm`) hitting the same slow sgemm?
- Suspect: the transposed-B sgemm call (`gemm_f32_trans_b_into_accum`, `src/linalg.rs:334-380`)
  on skinny-M matrices is memory-hostile. Test the `gemm` crate's packed-B layout instead.

### 3.2 Make small-matmul cases fall back gracefully
- Add a cost model: if the graph's total FLOPs are below a threshold where eager PyTorch's
  MKL-bound kernels win, route the whole graph to eager (documented, measurable, honest).
  The `capture()` API (`capture.py:40-76`) remains for users who want the Rust path.
- Bench gate: compiled is never > 1.5x eager on any tested shape; faster on fused chains
  (Linear+ReLU+Linear) where the fusion epilogue pays off.

### 3.3 Broader op coverage for real models
- 453 ops supported (`supported_ops()`); the gaps that matter for transformer inference are
  torch.where, index_select, and baddbmm — add them with parity tests before optimization.

**Exit criteria:** a real BERT-tiny inference is faster compiled than eager end-to-end on CPU;
release wheel re-benchmarked; no shape regresses > 1.5x.

---

## Phase 4 — dGPU and CUDA backends

**Goal:** stop ignoring 80% of the GPU market.

### 4.1 CUDA backend (new)
- New crate `src/cuda/` using `cudarc` (safe CUDA driver API bindings): same `Backend` trait the
  wgpu decoder implements — quantized weight buffers, one command stream per token, one readback.
- int4 dequant as a CUDA kernel (reuse the same math as the Vulkan shader); cuBLAS GEMV for f32
  fallback; leverage tensor cores (mma) for the int8 path.
- Ship as `torchburn[cuda]` extra with dynamic PyTorch-style CUDA detection.
- Tests: parity tests against the CPU reference on CI's CUDA runner (self-hosted), plus a
  no-CUDA smoke test that the import path degrades cleanly.

### 4.2 Vulkan on discrete GPUs (AMD/Intel Arc)
- The existing wgpu path should scale; validate workgroup autotuning (2.4) on a dGPU where
  occupancy actually matters (unlike Iris Xe).
- Measure: make `bench_wgpu_submit` a first-class target on AMD hardware in nightly CI.

### 4.3 Metal (macOS)
- wgpu already targets Metal; the work is in the unified-memory path (no staging readback needed —
  read results directly from shared buffers). This could make Apple Silicon the *best* torchburn
  target, not an afterthought.

**Exit criteria:** at least one non-wgpu GPU backend shipping with parity tests; a published
benchmark table across NVIDIA / AMD / Intel / Apple hardware with llama.cpp numbers alongside.

---

## Phase 5 — Quantization formats and model surface

**Goal:** close the accuracy/perf gap with ggml formats where they matter, without losing the
PyTorch-native quantize path.

### 5.1 Advanced formats
- Add Q4_K-style super-block quantization (6-bit scales over 32-element sub-blocks, min/max pairs)
  alongside the current int4-g64. Same kernel interface, better perplexity at similar size.
- Keep PyTorch-side quantization (`quantize_model`) — the differentiator — but make the on-disk
  format identical to what the Rust kernels read, no requantization step.

### 5.2 KV-cache quantization
- int8 KV cache (per-head scales), doubling effective context on 8 GB machines. Rust-side only;
  parity tests on long-context recall.

### 5.3 Speculative decoding
- Draft model (0.1B) + target (0.5B) both resident; Rust-side acceptance loop. This is the
  standard route past the memory-bandwidth wall for another 2-3x on both CPU and GPU.

### 5.4 GGUF import
- Read GGUF files directly (llama.cpp's format) so users can load any community quantized model.
  Not a new kernel — a loader. Removes the "torchburn only runs torchburn-quantized models"
  objection. Map GGUF quant types onto our kernel set (Q4_K, Q8_0, Q6_K).

**Exit criteria:** perplexity delta vs f32 documented per format; GGUF models load and decode
correctly; speculative decoding ships behind a flag with measured acceptance rates.

---

## Phase 6 — Production hardening and release

### 6.1 Testing matrix
- Rust: `cargo test` across feature tiers (`default`, `openblas`, `burn`, `burn-wgpu`, `cuda`),
  each with the parity suite.
- Python: `python -m pytest` with the parity + end-to-end suites on CPU; GPU suites on self-hosted
  runners; smoke tests on Python 3.9/3.12/3.13 (abi3 wheel covers 3.9+).
- Fuzzing: `cargo fuzz` on the payload parser (`dict_to_payload`, the 10 MB DoS limit at
  `engine.rs:62`) and the safetensors loader.
- Memory: `cargo miri test` on the FFI boundary (DLPack capsule lifetime is the risky part —
  the weakref finalizer at `_interpreter.py:112-113` and the consumed-capsule contract).

### 6.2 Release engineering
- `cargo build --release` with `lto = "thin"`, `codegen-units = 1` (already set, `Cargo.toml:50-53`)
  plus `panic = "abort"` evaluation for the cdylib.
- maturin release builds per platform; the two-variant wheel scheme from 0.2; PyPI trusted
  publishing already works (`publish-pypi` green in CI).
- Version gate: a release cannot ship if nightly benchmarks regressed > 10% on any tracked
  hardware config.

### 6.3 Public benchmark publication
- `benchmarks/public/` — reproducible scripts + a README table comparing torchburn vs llama.cpp
  (same model, same quantization, same machine) on: laptop CPU, Iris Xe iGPU, NVIDIA dGPU, Apple
  Silicon. Numbers or it did not happen; the scripts must be runnable by anyone.

---

## Execution order and dependencies

```
Phase 0 ──► Phase 1 (CPU) ──► Phase 2 (iGPU) ──► Phase 4 (dGPU/CUDA) ──► Phase 5 (formats)
    └───────────► Phase 3 (torch.compile parity) ────────────────────────────┘
                                   Phase 6 (hardening/release) runs continuously,
                                   gating each phase's exit.
```

Phase 0 is non-negotiable first. Phases 1 and 2 are independent of each other. Phase 3 is
independent of everything but benefits from 0.3's dispatch. Phase 5 needs Phase 1's layout work
(v2 weight format) settled first.

## Non-goals (explicit)

- Beating llama.cpp on NVIDIA data-center GPUs with fp16/fp8 batching — vLLM/TensorRT territory.
- Replacing PyTorch training — the backend only supports inference-mode graphs today; autograd
  training via AOTAutograd (`_backend.py:31-48`) stays at "works but not the point."
- WASM targets — after the desktop story is complete.

## Success metric

A user on a 2019+ laptop with no discrete GPU runs `pip install torchburn[avx2]`, loads a
0.5B int4 model, and gets **70+ tok/s on CPU or 50+ tok/s on iGPU** — faster than llama.cpp on
the same machine, with a one-line PyTorch integration llama.cpp cannot offer.

---

# Appendix A — Full Codebase Audit Findings

Audit date: 2026-09-07. Every `.rs` file in `src/`, every `.py` file in `python/`,
all 8 WGSL shaders, build config, tests, and benchmarks were read.

## A.1 Critical Performance Bugs (sorted by estimated impact)

| # | Location | Issue | Est. Impact |
|---|----------|-------|-------------|
| 1 | `src/engine/dispatch_op.rs:37-102` | Linear chain-of-responsibility scan through 22 `try_dispatch` modules per node. For a 1000-node graph = 22,000 string comparisons. Should be a `HashMap<&str, dispatch_fn>` built once at init. | Every node, every graph |
| 2 | `src/engine/payload.rs:92-568` | `supported_targets()` allocates ~450 `String` objects via `.into()` on every call. Should be `OnceLock<&'static [&'static str]>`. | Python profiler, parser init |
| 3 | `src/engine/helpers.rs:25-26` | `slot_view()` clones `shape` and `strides` `Vec<i64>` for every `Slot::View` access. Called for every argument of every node. Fix: store as `Arc<[i64]>` in `Slot::View`. | Every node execution |
| 4 | `src/autograd/backward_ops/matmul_linear.rs:86-126` | Backward-only matmul uses naive O(MKN) triple-nested loop with no tiling, blocking, or cache optimization. | Training backward |
| 5 | `src/autograd/batch/single.rs:168-438` | Matmul backward in batch FFI path: same naive triple loops. | Training backward |
| 6 | `src/nn/losses.rs` (all kernels) | Every loss (MSE, smooth_l1, BCE) allocates a full `Vec<T>` intermediate buffer and runs single-threaded. No rayon, no SIMD. | Loss computation |
| 7 | `src/nn/embedding.rs:23-111` | Embedding lookup is single-threaded with sequential `copy_from_slice` per row. | Large embeddings |
| 8 | `src/engine/graph_cache.rs:118-124` | FIFO eviction, not LRU. Frequently-accessed early graphs get evicted. | Cache hit rate |
| 9 | `src/autograd/backward_ops/norm_act.rs:107-207` | LayerNorm backward is **approximate** (just copies upstream as grad_input). The correct full implementation exists at `batch/single.rs:1068-1238`. | Training accuracy |
| 10 | `src/nn/activations.rs:56,104,205,252` | Per-element `Vec<usize>` coords allocation in `apply_elementwise` for non-contiguous tensors. Should use stack array. | Non-contiguous tensor ops |

## A.2 SIMD Gaps (kernels missing vectorization)

| Kernel File | Current SIMD | What's Needed |
|-------------|-------------|---------------|
| `src/nn/activations.rs` | Only exact GELU (`approximate="none"`) uses `wide::f32x8` | Add SIMD for: sigmoid, tanh, silu, elu, selu, softmax, log_softmax |
| `src/nn/convolution.rs` (910 lines) | Zero SIMD | Add SIMD inner loops for f32 conv2d/conv1d |
| `src/nn/pooling.rs` (1118 lines) | Zero SIMD | Add SIMD for avg_pool2d, adaptive_avg_pool2d |
| `src/nn/losses.rs` (386 lines) | Zero SIMD + zero parallelism | Add rayon parallel + SIMD for MSE, smooth_l1, BCE inner loops |
| `src/autograd/tape.rs:251-272` | `add_in_place` is scalar | SIMD + parallel chunk accumulation |
| `src/autograd/backward_ops/*.rs` | All backward ops scalar | Add SIMD to high-traffic backward ops (mul, add, relu, sigmoid) |
| `src/kernels/elementwise.rs` | Uses rayon but inner loops scalar | Add SIMD to hot elementwise ops (add, mul, relu) |

## A.3 WGSL Shader Gaps

| Shader | Lines | Issue | Fix |
|--------|-------|-------|-----|
| `attn_decode.wgsl` | 95 | Workgroup size 16 wastes 48/64 wave lanes; serial per-token `workgroupBarrier()` across all KV positions | Increase workgroup to 64; use subgroup reductions; tile over sequence |
| `rmsnorm.wgsl` | 55 | Scalar f32 accumulation in strided loop; 5-stage serial barrier reduction | Vectorize to `vec4<f32>` loads; use `subgroupAdd` |
| `fused_add_rmsnorm.wgsl` | 58 | Same scalar + serial barrier pattern as rmsnorm | Same vectorization fix |
| `swiglu.wgsl` | 24 | One f32 per thread, no vectorization | Process `vec4<f32>` per thread (4 elements) |
| `residual_add.wgsl` | 19 | Trivial element-wise, could be fused into other kernels | Fuse into gemv or rmsnorm |
| `rope_append.wgsl` | 81 | Scalar f32 loads, under-utilizes 64 threads for head_dim<=128 | Vectorize to `vec2<f32>` pairs; adjust thread mapping |
| All shaders | — | No use of `subgroup` operations (shuffle, add) | Use subgroup ops for reductions instead of expensive `workgroupBarrier()` |

## A.4 Architecture/Design Issues

| Area | File | Issue | Recommended Fix |
|------|------|-------|-----------------|
| Dispatch | `dispatch_op.rs` | 22-module linear chain per node | Build flat `HashMap<&str, fn>` at startup |
| Kwargs | `helpers.rs:140-155` | `kw_opt_dims()` heap-allocates `Vec` for single-dim | Use `SmallVec<[isize; 4]>` or `enum {One(isize), Many(Vec<isize>)}` |
| Non-contiguous | `activations.rs` | Per-element `Vec<usize>` coords | Use `SmallVec<[usize; 8]>` or strided index computation |
| Cache | `graph_cache.rs` | FIFO not LRU | Add `access_count` or promote-on-access in VecDeque |
| Autograd | `tape.rs` | Global singleton tape — not thread-safe | Per-thread tape or `Mutex<Tape>` |
| Code duplication | `backward_ops/*.rs` | Every op duplicated for f32 + f64 | Macro or generic `fn backward_f32_or_f64<T: Float>()` |
| Code duplication | `convolution.rs`, `pooling.rs` | f32/f64 near-identical kernels | Same macro/generic approach |
| Dead code | `tape.rs:153-169` | Initial backward iteration attempt is discarded | Remove dead code |
| Accuracy | `norm_act.rs:107-207` | LayerNorm backward is approximate | Replace with correct impl from `batch/single.rs:1068-1238` |
| Silent fallback | `batch/single.rs:1478-1485` | `_ =>` returns zero gradients for unsupported ops | Log warning or panic in debug mode |

## A.5 Python Layer Issues

| File | Issue | Severity |
|------|-------|----------|
| `_interpreter.py:280-282` | fp16/bf16 → f32 `.to()` copy per FFI call | Medium (necessary) |
| `_interpreter.py:290-295` | Grad-enabled fallback kills batch fast-path | Low (correctness) |
| `autograd.py:75` | Global `_tape` singleton — not thread-safe | Medium |
| `engine.py:167-171` | `logits[0,-1,:].clone()` per sample | Low (necessary for DLPack) |
| `llm/tokenizer.py:109-110` | `inspect.signature()` on every `encode()` | Low |
| `_cache.py` | `_LOCK` (RLock) serializes first-time compilations | Low |

## A.6 What's Already Well-Optimized (no changes needed)

- **GEMV kernels** (`quantization/gemv.rs`): AVX2/AVX-512/VNNI with interleaved V2 layout
- **WGSL GEMV shaders** (`gemv_w4a32.wgsl`, `gemv_swiglu_w4a32.wgsl`): 128-bit coalesced loads, hardware `dot()`, multi-row tiling
- **Fusion engine** (`fusion.rs`): Elementwise chain + GEMM epilogue fusion
- **Memory pool** (`memory_pool.rs`): Thread-local + global free list
- **BLAKE3 graph cache** (`cache.rs`): Structural hashing with LRU eviction
- **DLPack FFI** (`dlpack.rs`): True zero-copy
- **Rust generate_loop** (`decoder.rs`): Entire decode loop in Rust, one FFI crossing per generation
- **Streaming quantization** (`loader.py`): Layer-by-layer with `gc.collect()` for bounded RAM
- **StaticKVCache** (`model.py`): Pre-allocated zero-allocation decode
- **Fused attention/MLP** (`fused_ops.rs`): Rust-side fused kernels for int4/int8

## A.7 Prioritized Implementation Order

### Sprint 1 — Quick Wins (< 1 day each, high impact) ✅ ALL DONE

1. ✅ **`dispatch_op.rs`**: Replace linear chain with `OnceLock<HashMap<&str, fn>>` dispatch table
2. ✅ **`payload.rs`**: Make `supported_targets()` a `OnceLock<&'static [&'static str]>`
3. ✅ **`helpers.rs`**: Store `shape`/`strides` as `Arc<[i64]>` in `Slot::View` to eliminate clone overhead
4. ✅ **`norm_act.rs`**: Replace approximate LayerNorm backward with correct impl from `single.rs`
5. ✅ **`graph_cache.rs`**: Add LRU promotion on access (track last-access time or use `LinkedHashMap`)
6. ✅ **`activations.rs`**: Add SIMD paths for sigmoid, tanh, silu using `wide::f32x8`
7. ✅ **`tokenizer.py`**: Cache `inspect.signature()` result instead of calling per encode

### Sprint 2 — Kernel Optimization (2-5 days) ✅ ALL DONE

8. ✅ **`attn_decode.wgsl`**: Rewrite with workgroup size 64, subgroup reductions, sequence tiling
9. ✅ **`rmsnorm.wgsl` + `fused_add_rmsnorm.wgsl`**: Vectorize to `vec4<f32>` loads, subgroup reductions
10. ✅ **`swiglu.wgsl`**: Vectorize to `vec4<f32>` per thread
11. ✅ **`losses.rs`**: Add rayon parallelism to MSE, smooth_l1, BCE, nll_loss inner loops
12. ✅ **`embedding.rs`**: Add rayon parallelism for row-gather
13. ✅ **`activations.rs`**: Add SIMD for softmax and log_softmax
14. ⏭️ **`convolution.rs`**: Already has rayon parallelism; SIMD inner loops deferred (diminishing returns)

### Sprint 3 — Autograd & Training (3-5 days) ✅ ALL DONE

15. ✅ **`backward_ops/matmul_linear.rs`**: Rayon-parallel matmul backward (parallelize over output rows)
16. ⏭️ **`backward_ops/*.rs`**: Macro-based deduplication deferred (large refactor, low perf impact)
17. ✅ **`tape.rs`**: Thread-safe via thread-local tape (already correct design)
18. ✅ **`tape.rs`**: Rayon-parallel `add_in_place` for large tensors
19. ✅ **`batch/single.rs`**: Rayon-parallel 2D matmul backward

### Sprint 4 — Architecture (1-2 days) ✅ ALL DONE

20. ✅ **`helpers.rs`**: `SmallVec<[isize; 4]>` for kw_opt_dims single-dim fast path
21. ✅ **`activations.rs`**: Stack-allocated `[usize; 8]` coords for non-contiguous tensors
22. ⏭️ **`convolution.rs`, `pooling.rs`**: Macro/generic dedup deferred (low perf impact, high refactor cost)
23. ✅ **`batch/single.rs`**: `eprintln!` warning on unsupported backward op fallback

## A.8 Files Audited (complete list)

### Rust (src/)
- `lib.rs`, `engine.rs`, `dispatch.rs`, `fusion.rs`, `memory_pool.rs`, `cache.rs`, `dlpack.rs`
- `engine/graph_cache.rs`, `engine/payload.rs`, `engine/helpers.rs`, `engine/dispatch_op.rs`
- `engine/dispatch_ops/d_elementwise.rs`, `d_reducers.rs`, `d_norm.rs`, `d_shape.rs`, `d_nn.rs`, `d_scatter.rs`, `d_math_extra.rs`, `d_batch2.rs`, `d_float_special.rs`, `d_blas_extra.rs`, `d_act_loss.rs`, `d_decomp.rs`, `d_scatter2.rs`, `d_shape2.rs`, `d_nn3d.rs`, `d_random.rs`, `d_rnn.rs`, `d_solve.rs`, `d_fused.rs`, `d_quant.rs`, `d_fft.rs`, `d_batch4.rs`
- `kernels/elementwise.rs`, `kernels/reductions.rs`, `kernels/linalg.rs`
- `quantization/per_tensor.rs`, `quantization/packed_v2.rs`, `quantization/gemv.rs`, `quantization/fused_ops.rs`
- `llm/mod.rs`, `llm/decoder.rs`
- `nn/attention.rs`, `nn/convolution.rs`, `nn/pooling.rs`, `nn/upsample.rs`, `nn/embedding.rs`, `nn/losses.rs`, `nn/activations.rs`, `nn/norm.rs`
- `autograd/tape.rs`, `autograd/batch.rs`, `autograd/batch/single.rs`
- `autograd/backward_ops.rs`, `autograd/backward_ops/binary.rs`, `unary.rs`, `shape.rs`, `matmul_linear.rs`, `norm_act.rs`, `losses.rs`
- `wgpu/backend.rs`, `wgpu/pipelines.rs`, `wgpu/decode.rs`
- `shaders/gemv_w4a32.wgsl`, `gemv_swiglu_w4a32.wgsl`, `attn_decode.wgsl`, `rmsnorm.wgsl`, `fused_add_rmsnorm.wgsl`, `swiglu.wgsl`, `residual_add.wgsl`, `rope_append.wgsl`

### Python (python/)
- `_backend.py`, `_interpreter.py`, `_compiled.py`, `_cache.py`, `capture.py`, `ops.py`, `profiler.py`, `autograd.py`, `quantization.py`, `__init__.py`
- `_parser/__init__.py`, `_parser/op_registry.py`, `_parser/graph_walker.py`
- `llm/__init__.py`, `llm/__main__.py`, `llm/_registry.py`, `llm/config.py`, `llm/model.py`, `llm/tokenizer.py`, `llm/loader.py`, `llm/engine.py`, `llm/cli.py`, `llm/api.py`

### Config & Build
- `Cargo.toml`, `build.rs`, `.cargo/config.toml`

### Tests & Benchmarks
- `tests/parity.rs`, `tests/parity/f32.rs`, `tests/parity/int4.rs`, `tests/parity/int8.rs`, `tests/parity/decoder.rs`, `tests/parity/dispatch_tiers.rs`, `tests/test_avx512_intrinsics.rs`
- `benches/bench_gemm.rs`, `benches/bench_wgpu_submit.rs`, `benches/bench_decoder_step.rs`, `benches/bench_int4_gemv.rs`
- `benchmarks/` — 12 Python benchmark scripts
- `examples/mlp.py`, `examples/llm_inference.py`, `examples/llm_chat.py`
