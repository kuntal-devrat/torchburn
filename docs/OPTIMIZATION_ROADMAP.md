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
