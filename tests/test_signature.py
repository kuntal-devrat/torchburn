"""FFI surface sanity: engine name, supported targets, signature stability."""

from __future__ import annotations

import torch

import torchburn
from torchburn import _torchburn as tb


def test_engine_surface():
    assert tb.active_engine() in ("native_cpu", "burn_ndarray", "burn_wgpu")
    targets = tb.supported_targets()
    # Phase 1 ops are always present
    assert {"add", "sub", "mul", "div", "relu"}.issubset(set(targets))
    # Phase 2 ops are also present
    assert {"matmul", "linear", "bmm", "sum", "mean", "sigmoid", "layer_norm"}.issubset(set(targets))
    # There are many more ops now
    assert len(targets) >= 40


def test_version_surface():
    assert tb.__version__ == torchburn.__version__ == "0.1.0"


def test_signature_deterministic():
    payload = (
        '{"inputs": [{"dtype": "f32", "shape": [2, 3]}],'
        ' "nodes": [{"args": [{"index": 0, "kind": "slot"}], "id": 0, "target": "add"}],'
        ' "outputs": [0]}'
    )
    assert tb.signature(payload) == tb.signature(payload)
    assert len(tb.signature(payload)) == 64  # BLAKE3 hex digest


def test_cache_clear_resets_stats():
    torchburn.cache_clear()
    assert torchburn.cache_stats()["size"] == 0
    assert torchburn.cache_stats()["hits"] == 0
    assert torchburn.cache_stats()["misses"] == 0


def test_compile_convenience():
    def model(x):
        return torch.relu(x * 2)

    compiled = torchburn.compile(model)
    x = torch.randn(3, 3)
    assert torch.allclose(compiled(x), model(x))
