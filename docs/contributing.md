# Contributing to TorchBurn

Thank you for your interest in contributing to TorchBurn! This guide will help you get started.

## Development Setup

### Prerequisites

- **Rust**: 1.75+ (install via [rustup](https://rustup.rs/))
- **Python**: 3.9+ with pip
- **maturin**: Build tool for Rust extensions (`pip install maturin`)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/torchburn/torchburn.git
cd torchburn

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Build the extension
maturin develop -r

# Run tests
pytest tests/ -x -q
```

### GPU Backend (Optional)

```bash
# Build with Burn wgpu support
maturin develop --features burn-wgpu -r

# Test with GPU engine
TORCHBURN_ENGINE=burn-wgpu pytest tests/ -x -q
```

## Project Structure

```
torchburn/
├── src/                    # Rust source code
│   ├── lib.rs              # PyO3 module exports
│   ├── engine.rs           # Main execution engine
│   ├── dlpack.rs           # DLPack FFI bridge
│   ├── cache.rs            # BLAKE3 graph caching
│   ├── fusion.rs           # Graph-level operator fusion
│   ├── autograd.rs         # Rust autograd tape
│   └── ops*.rs             # Operator implementations
│
├── python/torchburn/       # Python package
│   ├── __init__.py         # Package init
│   ├── _backend.py         # torch._dynamo registration
│   ├── _parser.py          # FX graph parser
│   ├── _compiled.py        # Compiled callable wrapper
│   └── autograd.py         # Python autograd tape
│
├── tests/                  # Test suite
└── benchmarks/             # Performance benchmarks
```

## Adding a New Operator

### Step 1: Implement the Rust kernel

Create or modify the appropriate `src/*.rs` file:

```rust
// src/math_ops.rs

pub fn my_new_op(a: &BorrowedTensor, b: &BorrowedTensor) -> Result<OwnedTensor, String> {
    // Validate inputs
    if a.shape != b.shape {
        return Err("Shape mismatch".into());
    }
    
    // Allocate output
    let mut out = OwnedTensor::zeros(&a.shape, a.dtype);
    
    // Compute
    for i in 0..a.elem_count() {
        let a_val = a.read::<f32>(i);
        let b_val = b.read::<f32>(i);
        out.write(i, a_val + b_val);  // Example: elementwise add
    }
    
    Ok(out)
}
```

### Step 2: Add dispatch in engine.rs

```rust
// src/engine.rs

"my_new_op" => {
    let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
    let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
    let result = math_ops::my_new_op(&a, &b)
        .map_err(|e| unsupported(&format!("my_new_op: {e}")))?;
    slots.push(Slot::Owned(result));
}
```

### Step 3: Add to supported_targets

```rust
// src/engine.rs

pub fn supported_targets() -> &'static [&'static str] {
    &[
        // ... existing targets ...
        "my_new_op",
    ]
}
```

### Step 4: Add parser mapping

```python
# python/torchburn/_parser.py

_FUNCTION_TO_OP = {
    # ... existing mappings ...
    "torch.my_new_op": "my_new_op",
    "aten.my_new_op.default": "my_new_op",
}
```

### Step 5: Write tests

```python
# tests/test_my_new_op.py

import torch
import torchburn

def test_my_new_op_basic():
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    
    model = torch.nn.Linear(4, 4).eval()
    compiled = torch.compile(model, backend="torchburn")
    
    # Test that it compiles and runs
    output = compiled(a)
    assert output.shape == a.shape
```

### Step 6: Add backward pass (if needed)

```rust
// src/autograd.rs

"my_new_op" => {
    // grad_a = grad_output
    // grad_b = grad_output
    saved[0].clone(),  // grad_a
    saved[1].clone(),  // grad_b
}
```

## Code Style

### Rust

- Follow `rustfmt` defaults
- Use `cargo clippy` to catch common issues
- Document public functions with `///` doc comments
- Handle errors with `Result<T, String>` (no panics in production code)

### Python

- Follow PEP 8 style
- Use type hints for all public functions
- Document with docstrings (Google style)

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -x -q

# Run specific test file
pytest tests/test_ops.py -x -q

# Run with coverage
pytest tests/ --cov=torchburn --cov-report=html
```

### Test Categories

- `test_*.py` - Unit tests for specific features
- `test_phase*.py` - Integration tests for development phases
- `test_burn_engine.py` - Burn engine specific tests
- `test_training.py` - End-to-end training tests

### Writing Tests

1. Test correctness against PyTorch eager
2. Test edge cases (empty tensors, scalar, etc.)
3. Test dtypes (float32, float64, int64, bool)
4. Test shapes (broadcasting, memory layout)

## Performance Benchmarks

```bash
# Run transformer benchmark
python benchmarks/bench_transformer.py

# Run custom benchmark
python -c "
import torch
import torchburn
import time

model = torch.nn.Linear(1024, 1024).eval()
compiled = torch.compile(model, backend='torchburn')

x = torch.randn(32, 1024)

# Warmup
for _ in range(10):
    compiled(x)

# Benchmark
start = time.perf_counter()
for _ in range(100):
    compiled(x)
elapsed = time.perf_counter() - start

print(f'{elapsed/100*1000:.2f} ms/iter')
"
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the full test suite (`pytest tests/ -x -q`)
6. Run clippy (`cargo clippy -- -D warnings`)
7. Submit a pull request

### PR Checklist

- [ ] Tests pass locally
- [ ] No clippy warnings
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for notable changes)

## Reporting Issues

- Use GitHub Issues for bug reports
- Include minimal reproducible example
- Specify OS, Python version, PyTorch version
- Include error messages and stack traces

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
