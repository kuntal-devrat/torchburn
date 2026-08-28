# Changelog

All notable changes to TorchBurn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 14: Distribution & Packaging
  - Multi-platform CI/CD pipeline (macOS, Linux, Windows)
  - Multi-Python version support (3.9-3.13)
  - PyPI publishing workflow (TestPyPI on PR, PyPI on tag)
  - Comprehensive documentation (architecture, ops coverage, contributing)

### Changed
- Upgraded CI/CD to use cibuildwheel for cross-platform wheel builds
- Improved error messages with context for debugging

### Fixed
- Fixed `_operator.iadd` fallback by adding proper parser mapping
- Fixed thread safety issues with `RwLock` for graph cache
- Fixed unwrap() calls in hot paths with proper error handling

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
