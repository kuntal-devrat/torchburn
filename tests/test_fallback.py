"""Dynamic operator fallback & safety loop tests (REQ-002).

Unsupported operators must never crash the compilation path: the graph is
sliced, supported runs go to Rust, and unsupported nodes execute instantly
in native PyTorch with a non-blocking UserWarning.
"""

from __future__ import annotations

import warnings

import torch

import torchburn


def _assert_fallback_warns(call):
    from torchburn._interpreter import _WARNED
    _WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("torchburn: falling back" in m for m in messages), messages
    return result


def test_unsupported_op_inside_supported_graph():
    """mul/add runs in Rust, fft runs eagerly, results stay correct."""

    def model(x):
        y = x * 2
        z = torch.fft.fft(y).real  # unsupported (fft not in 375)
        return z + 1

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(6, 6)

    out = _assert_fallback_warns(lambda: compiled(x))
    assert torch.allclose(out, model(x))


def test_linear_layer_runs_native():
    """nn.Linear (matmul) is Phase 2; it must run natively (no fallback warning)."""

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 4)

        def forward(self, x):
            return torch.relu(self.lin(x))

    net = Net().eval()
    compiled = torch.compile(net, backend="torchburn")
    x = torch.randn(3, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compiled(x)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any("torchburn: falling back" in m for m in messages), messages
    with torch.no_grad():
        assert torch.allclose(out, net(x))


def test_conv2d_runs_native():
    """nn.Conv2d is Phase 3; it must run natively (no fallback warning)."""

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return torch.relu(self.conv(x))

    net = Net().eval()
    compiled = torch.compile(net, backend="torchburn")
    x = torch.randn(2, 3, 16, 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compiled(x)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any("torchburn: falling back" in m for m in messages), messages
    with torch.no_grad():
        # conv accumulation order differs from torch -> allow float32 noise
        assert torch.allclose(out, net(x), atol=1e-4)


def test_mixed_run_and_fallback_ordering():
    """Supported nodes before AND after an unsupported one all run correctly."""

    def model(x):
        a = x * 2          # rust
        b = torch.fft.fft(a).real  # unsupported -> eager
        c = b + 1          # rust
        return c

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(4, 4)
    out = _assert_fallback_warns(lambda: compiled(x))
    assert torch.allclose(out, model(x))


def test_fallback_warns_once_per_target():
    from torchburn._interpreter import _WARNED
    _WARNED.clear()

    def model(x):
        return torch.special.airy_ai(x)[0] + torch.special.airy_ai(x * 2)[0]

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(4, 4).abs() + 0.1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled(x)
    messages = [str(w.message) for w in caught if "airy_ai" in str(w.message)]
    assert len(messages) == 1  # once per target, not per node


def test_float16_input_auto_casts_to_f32():
    """Float16 inputs are auto-cast to f32 for native execution, then cast back."""

    def model(x):
        return x + x

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(4, 4, dtype=torch.float16)
    out = compiled(x)
    # Should produce float16 output (auto-cast back)
    assert out.dtype == torch.float16, f"expected float16, got {out.dtype}"
    # Should match eager result within float16 tolerance
    ref = model(x)
    assert torch.allclose(out.float(), ref.float(), atol=1e-3)


def test_unsupported_whole_graph():
    def model(x):
        return torch.fft.fft(x).real

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(8)
    out = _assert_fallback_warns(lambda: compiled(x))
    assert torch.allclose(out, model(x))
