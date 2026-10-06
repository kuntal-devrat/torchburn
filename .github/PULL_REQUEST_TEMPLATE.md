## Description

Please provide a summary of your changes, including rationale and affected components (Rust core, WGPU shaders, Python compiler/LLM API).

Fixes #(issue)

---

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ⚡ Performance optimization (latency reduction, memory savings, shader throughput)
- [ ] ✨ New feature (new operator, hardware target, model architecture)
- [ ] 📚 Documentation update (README, docstrings, examples)
- [ ] 🧹 Refactoring / Code cleanup (no behavioral changes)

---

## Verification & Testing

Please check the verification steps that were performed:

- [ ] Static type check passes: `npx --yes pyright python/torchburn` (0 errors, 0 warnings)
- [ ] Rust tests pass: `cargo test --no-default-features --features matrixmultiply`
- [ ] Rust linter passes: `cargo clippy --no-default-features --features matrixmultiply -- -D warnings`
- [ ] Rust formatting matches: `cargo fmt -- --check`
- [ ] Python test suite passes: `pytest tests/`
- [ ] Benchmarks verified without regression (if applicable)

---

## Checklist

- [ ] My code adheres to the project's coding standards and style guidelines.
- [ ] I have commented complex or non-obvious code (especially `unsafe` SIMD logic or shader workgroup indexing).
- [ ] I have updated corresponding documentation and type stubs (`.pyi`) if public APIs changed.
- [ ] No extraneous files or debug prints are left behind.
