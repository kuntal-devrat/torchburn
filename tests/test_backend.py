"""End-to-end ``torch.compile(..., backend="torchburn")`` tests (REQ-001)."""

from __future__ import annotations

import warnings

import pytest
import torch

import torchburn
from torchburn._compiled import BurnCompiledCallable


@pytest.fixture(autouse=True)
def _clean_cache():
    torchburn.cache_clear()
    yield
    torchburn.cache_clear()


def test_backend_registered():
    assert "torchburn" in torch._dynamo.list_backends(exclude_tags=set())


def test_elementwise_model_matches_eager():
    def model(x):
        return torch.relu(x * 2 + 1) - 0.5

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(4, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no fallback expected
        out = compiled(x)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, model(x))


def test_module_with_parameters_matches_eager():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(1, 8))

        def forward(self, x):
            return torch.relu(x * self.w + 0.1)

    net = Net()
    compiled = torch.compile(net, backend="torchburn")
    x = torch.randn(4, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = compiled(x)
    assert torch.allclose(out, net(x))


def test_double_precision_model():
    def model(x):
        return x * x + x

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(5, 5, dtype=torch.float64)
    out = compiled(x)
    assert out.dtype == torch.float64
    assert torch.allclose(out, model(x))


def test_different_shapes_same_callable():
    def model(x):
        return torch.relu(x * 3)

    compiled = torch.compile(model, backend="torchburn")
    for shape in [(2, 2), (3, 4, 5), (7,)]:
        x = torch.randn(*shape)
        assert torch.allclose(compiled(x), model(x))


def test_multiple_outputs_matches_eager():
    def model(x):
        return x * 2, x + 1

    compiled = torch.compile(model, backend="torchburn")
    x = torch.randn(3, 3)
    out = compiled(x)
    ref = model(x)
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.allclose(out[0], ref[0])
    assert torch.allclose(out[1], ref[1])


def test_backend_returns_compiled_callable():
    def spy(gm, inputs):
        assert isinstance(gm, torch.fx.GraphModule)
        return BurnCompiledCallable(gm, inputs)

    torch._dynamo.register_backend(name="torchburn_test_spy")(spy)

    def model(x):
        return x + 1

    compiled = torch.compile(model, backend="torchburn_test_spy")
    x = torch.randn(2, 2)
    out = compiled(x)
    assert torch.allclose(out, model(x))
