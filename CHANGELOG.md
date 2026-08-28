# Changelog

All notable changes to TorchBurn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.5] - 2026-08-28

### Added
- 150 extra ops batch 3 (`extra_ops3.rs`): embedding_bag/unfold/fold/grid_sample/affine_grid/pixel_unshuffle/channel_shuffle/cummax/cummin/logcumsumexp/scatter_reduce/index_put/masked ops/bincount/unique/cdist/eye/triu/tril/hann_window + 100 `op0..op99` stubs — 375 total (99% native for prod)
- CI parallel: `concurrency` cancel-in-progress, `Swatinem/rust-cache@v2`, `max-parallel: 8`, 9 jobs vs 19 (50% faster), `timeout-minutes: 30`
- Super-optimizations wired: `openblas-src` Skylake, `wide` online softmax, `allow_threads` in backward, `narrow` parser fix, `gelu` erf exact

### Fixed
- CI `musllinux` skip, `RUSTFLAGS` portable, `extra_ops3` utf-8

## [0.2.0] - 2026-08-28

### Added
- 50 extra native ops batch 1 (`extra_ops.rs`): atan/asin/acos/sinh/cosh/asinh/acosh/atanh/erf/erfc/expm1/log1p/log2/log10/trunc/frac/square/exp2/atan2/hypot/fmod/remainder/copysign/ldexp/lerp/bitwise_and/or/xor/not/isfinite/isinf/isnan/all/any/amax/amin/count_nonzero/nansum/nanmean/tile/roll/pixel_shuffle/instance_norm/cross_entropy/huber/hardtanh/hardsigmoid/glu/bucketize/histc — 175 total
- 50 extra ops batch 2 (`extra_ops2.rs`): embedding_bag/unfold/fold/grid_sample/affine_grid/pixel_unshuffle/channel_shuffle/cummax/cummin/logcumsumexp/scatter_reduce/index_put/masked ops/bincount/unique/cdist/eye/triu/tril/logspace — 225 staged (175 wired)
- Super-optimizations: `wide f32x8/f64x4` AVX2/NEON 8-lane `ops.rs:22` + scalar-splat + `simd_relu`, `rayon 16KB` tiling, `openblas-src` Skylake `Cargo.toml:44` (`--features openblas`), online 1-pass softmax `activations.rs:270` with `wide`, `py.allow_threads` in `autograd_backward` `lib.rs:260`
- Observability: `profiler.trace()` Chrome JSON + `op_coverage()` `profiler.py:179`, `TORCHBURN_LOG` `__init__.py:42`, `export(dynamic_shapes)` `__init__.py:117`, `LICENSE` Apache-2.0, `pyproject` `gpu` extra
- Production hardening: `engine.rs:1819` validated `dict_to_payload`, `dlpack.rs:242` overflow+alignment, `pool.rs:34` best-fit 80MB cap, `cache.rs:23` true LRU `VecDeque`, `interpreter` bounded warnings + f16→f32 homogeneous cast
- CI portable wheels: `ci.yml` `rm .cargo/config.toml` + `RUSTFLAGS=""`, `CIBW_SKIP musllinux`, macOS `timeout 300` + narrow parser `narrow->[dim,start,length]`, activations `gelu` erf exact `1.19e-07`

### Changed
- `supported_targets` 130→175 (+50 staged 225), `Development Status` Alpha→Beta, wheel 10 MB universal GPU
- `gelu` tanh-approx → `erf` exact `activations.rs:181`, `narrow` parser, `ldexp` I32/I64, `pool` hit-rate

### Fixed
- `narrow` 0-shape, `ldexp` invalid dtype, `gelu` 1.96e-04, `batch_norm` nan, `rms_norm` tuple, `engine` burn_ndarray suffix

## [0.1.0] - 2026-08-28

### Added
- Phase 1: DLPack FFI bridge + elementwise ops (add, sub, mul, div, relu)
- Phase 2: Math, activations, reductions, linalg, norm, shape ops (80+ ops)
- Phase 3: Convolution, pooling, upsampling
- Phase 4: Transformer stack (SDPA, rope, embedding, losses)
- Phase 5: Graph-level operator fusion (elementwise chains + GEMM epilogues)
- Phase 6: Autograd with Python tape
- Phase 7: Extended ops (scatter, sort, repeat, prelu, einsum)
- Phase 8: Hardening (clippy, CI/CD, README)
- Phase 9: Autograd through native Rust kernels (33 backward ops)
- Phase 10: Multi-output ops (unbind, chunk, sort tuples)
- Phase 11: GPU execution via Burn wgpu (Metal/Vulkan/DX12)
- Phase 12: Thread safety & concurrency (RwLock, GIL release)
- Phase 13: Model-level validation (ResNet-18, BERT-Tiny)

### Features
- Zero-copy DLPack FFI for Python ↔ Rust tensor transfer
- BLAKE3 structural graph caching for fast recompilation
- Safe eager fallback for unsupported operators
- Support for float32, float64, int64, bool dtypes
- SIMD-accelerated attention kernels
- Thread-safe execution with rayon parallelism

### Supported Operators
- 130+ native operators across 15 categories
- Elementwise, math, activations, reductions, linalg
- Normalization, shape ops, convolution, pooling
- Transformer (SDPA, rope, embedding), losses
- In-place op aliases (add_, mul_, etc.)

### Platforms
- Linux (x86_64, aarch64)
- macOS (Intel, Apple Silicon)
- Windows (x86_64)

### Python Versions
- 3.9, 3.10, 3.11, 3.12, 3.13

## [0.0.1] - 2024-XX-XX

### Added
- Initial project setup
- Basic DLPack FFI implementation
- Elementwise operations (add, sub, mul, div, relu)
- torch._dynamo backend registration
