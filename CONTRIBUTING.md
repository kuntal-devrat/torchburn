# Contributing to TorchBurn

Thank you for your interest in contributing to TorchBurn! We welcome contributions from the community to help make portable, high-efficiency deep learning inference and training accessible everywhere without CUDA lock-in.

---

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any violations or concerns to [security@torchburn.org](mailto:security@torchburn.org).

---

## Architecture Overview

TorchBurn is organized into three synergistic layers:

```
┌─────────────────────────────────────────────────────────────┐
│                   Python Front-End                         │
│  - torch.compile backend (FX graph partitioning & codegen)  │
│  - High-level LLM API (UniversalTransformer, Engine)        │
│  - Safe fallback to PyTorch eager for exotic operations      │
└──────────────────────────────┬──────────────────────────────┘
                               │ PyO3 FFI
┌──────────────────────────────▼──────────────────────────────┐
│                    Native Rust Core                         │
│  - Micro-kernels (AVX-512, AVX2, NEON runtime dispatch)     │
│  - INT4 / INT8 blocked quantization (packed_v2, grouped)    │
│  - Zero-Python pure Rust Transformer decoders               │
│  - Autograd engine with backward graph recording            │
└──────────────────────────────┬──────────────────────────────┘
                               │ wgpu / Metal
┌──────────────────────────────▼──────────────────────────────┐
│                Cross-Platform GPU Kernels                   │
│  - Pure WGSL compute shaders (Vulkan / Metal / DirectX 12)  │
│  - Single-pass fused GEMV-SwiGLU & FlashAttention decoding  │
│  - Zero CUDA runtime requirement                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Development Setup

### Prerequisites

1. **Rust toolchain** (1.75+ recommended):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup component add clippy rustfmt
   ```

2. **Python 3.10+**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install maturin pytest pytest-timeout torch numpy
   ```

3. **Node.js** (for Pyright static type checking):
   ```bash
   node --version  # v18+ recommended
   ```

---

## Building from Source

### Fast Development Build (in-tree editable)
```bash
maturin develop
```

### Release Build (optimized SIMD runtime)
```bash
maturin develop --release
```

### Building with GPU Backends
- **WGPU (Vulkan / Metal / DirectX 12)**:
  ```bash
  maturin develop --release --features burn-wgpu
  ```

- **CUDA** (optional compile-time check):
  ```bash
  maturin build --features cuda
  ```

---

## Running Verification & Tests

Before submitting a Pull Request, please run the full verification suite:

### 1. Static Type Checking (Pyright)
The Python codebase maintains strict 100% type safety with zero errors and zero warnings:
```bash
npx --yes pyright python/torchburn
```

### 2. Rust Unit Tests & Clippy
```bash
# Unit tests
cargo test --no-default-features --features matrixmultiply

# Linter checks
cargo clippy --no-default-features --features matrixmultiply -- -D warnings
cargo fmt -- --check
```

### 3. Python Integration Tests
```bash
pytest tests/ -v
```

### 4. Hardware Benchmarks
```bash
python benchmarks/bench_all_450_ops.py
python benchmarks/bench_wgpu_vs_native.py
```

---

## Code Style Guidelines

### Python
- Type annotations are required on all public APIs, methods, and functions.
- Avoid loose `Any` casts where explicit typing or protocol types are possible.
- Preserve backward compatibility and fallback behavior. If an FX operator cannot be lowered to native SIMD or WGPU, provide a graceful fallback to PyTorch eager.
- Do not mutate shared parameter storages (e.g. avoid zeroing out `linear.weight.data` when weights may be tied to embeddings).

### Rust
- Format code using `cargo fmt`.
- All native kernels must maintain memory safety (`unsafe` blocks must be documented with safety invariants).
- Use runtime SIMD feature detection (`std::is_x86_feature_detected!` or target attributes) rather than hardcoded compile-time flags to preserve portability across different processor generations.

### WGSL Shaders
- Shaders live in `src/shaders/`.
- Keep workgroup sizes power-of-two (e.g., 64 threads) for wave efficiency across AMD, Intel, Apple, and NVIDIA architectures.
- Ensure all buffer indexing is bounds-checked or strictly governed by uniform dimension parameters.

---

## Pull Request Guidelines

1. **Fork and branch**: Create a descriptive branch name from `main` (e.g., `feat/avx512-fp16-gemv` or `fix/static-kv-cache-indexing`).
2. **Keep PRs focused**: Address a single feature, bug fix, or optimization per PR.
3. **Add tests**: Any new operator, engine feature, or bug fix must include corresponding tests in `tests/`.
4. **Commit messages**: Use Conventional Commits formatting:
   - `feat: add fused add-rmsnorm shader`
   - `fix: prevent tied weight corruption during quantize_model`
   - `perf: optimize INT4 GEMV multi-row reduction`
   - `docs: update LLM quickstart guide`

---

## Reporting Issues

If you encounter unexpected behavior, crashes, or performance regressions:
- Check existing [GitHub Issues](https://github.com/torchburn/torchburn/issues) to avoid duplicates.
- Open an issue with our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.yml), including your operating system, CPU architecture, GPU adapter, and Python/PyTorch versions.
