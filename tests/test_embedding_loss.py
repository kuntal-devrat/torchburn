"""Phase 4 — embedding and losses.

Covers the ops API, the raw FFI kernels, and the export path (make_fx +
BurnCompiledCallable) including the int64 index/target tensors that Phase 4
added as first-class engine dtypes.  All engines must match torch with zero
fallbacks.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
import pytest

import torchburn
from torchburn import ops
from torchburn import _torchburn as _native
from torchburn._compiled import BurnCompiledCallable
from torchburn._parser import payload_json
from torch.fx.experimental.proxy_tensor import make_fx

torch.manual_seed(0)


def _spec(t: torch.Tensor) -> dict:
    dtype = {torch.float32: "f32", torch.float64: "f64", torch.int64: "i64", torch.int32: "i32"}[t.dtype]
    return {"shape": list(t.shape), "dtype": dtype}


def _run(payload: dict, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    caps = [t.__dlpack__() for t in tensors]
    return [torch.from_dlpack(c) for c in _native.execute(payload_json(payload), caps)]


def _compile(fn, *args):
    gm = make_fx(fn)(*args)
    return BurnCompiledCallable(gm, [a for a in args])


def _assert_native(fn, *args):
    """Compile fn and assert it runs with zero eager fallbacks, matching torch."""
    compiled = _compile(fn, *args)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = compiled(*args)
    assert torch.allclose(got, fn(*args), atol=1e-4)


class TestEmbedding:
    def test_embedding_ops_api(self):
        w = torch.randn(100, 16)
        idx = torch.randint(0, 100, (4, 6))
        assert torch.allclose(ops.embedding(idx, w), F.embedding(idx, w))

    def test_embedding_int32_indices(self):
        w = torch.randn(50, 8)
        idx = torch.randint(0, 50, (5,), dtype=torch.int32)
        payload = {
            "inputs": [_spec(w), _spec(idx)],
            "nodes": [{"id": 0, "target": "embedding", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}], "kwargs": {}}],
            "outputs": [0],
        }
        got = _run(payload, [w, idx])[0]
        assert torch.allclose(got, F.embedding(idx, w))

    def test_embedding_out_of_range_falls_back(self):
        w = torch.randn(10, 4)
        idx = torch.tensor([3, 15])  # 15 >= 10
        payload = {
            "inputs": [_spec(w), _spec(idx)],
            "nodes": [{"id": 0, "target": "embedding", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}], "kwargs": {}}],
            "outputs": [0],
        }
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            _run(payload, [w, idx])

    def test_embedding_export(self):
        w = torch.randn(50, 8)
        idx = torch.randint(0, 50, (4, 5))
        _assert_native(lambda w, i: F.embedding(i, w), w, idx)


class TestNLLCrossEntropy:
    def test_nll_mean(self):
        lp = torch.randn(8, 10).log_softmax(1)
        t = torch.randint(0, 10, (8,))
        assert torch.allclose(ops.nll_loss(lp, t), F.nll_loss(lp, t), atol=1e-5)

    def test_nll_ignore_index(self):
        lp = torch.randn(8, 10).log_softmax(1)
        t = torch.tensor([0, -100, 3, 1, -100, 5, 2, 4])
        assert torch.allclose(ops.nll_loss(lp, t, ignore_index=-100), F.nll_loss(lp, t, ignore_index=-100), atol=1e-5)

    def test_nll_sum_and_none(self):
        lp = torch.randn(6, 5).log_softmax(1)
        t = torch.randint(0, 5, (6,))
        assert torch.allclose(ops.nll_loss(lp, t, reduction="sum"), F.nll_loss(lp, t, reduction="sum"), atol=1e-5)
        assert torch.allclose(ops.nll_loss(lp, t, reduction="none"), F.nll_loss(lp, t, reduction="none"), atol=1e-5)

    def test_cross_entropy_ops(self):
        logits = torch.randn(32, 100)
        t = torch.randint(0, 100, (32,))
        assert torch.allclose(ops.cross_entropy(logits, t), F.cross_entropy(logits, t), atol=1e-4)

    def test_cross_entropy_export(self):
        """The make_fx decomposition (_log_softmax + nll_loss_forward) must be native."""
        x = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        _assert_native(lambda x, t: F.cross_entropy(x, t), x, t)

    def test_nll_export(self):
        lp = torch.randn(4, 10).log_softmax(1)
        t = torch.randint(0, 10, (4,))
        _assert_native(lambda lp, t: F.nll_loss(lp, t), lp, t)


class TestRegressionLosses:
    def test_mse_loss(self):
        a, b = torch.randn(8, 10), torch.randn(8, 10)
        for red in ("mean", "sum", "none"):
            assert torch.allclose(ops.mse_loss(a, b, reduction=red), F.mse_loss(a, b, reduction=red), atol=1e-5)

    def test_l1_loss_decomposes(self):
        """aten.l1_loss traces to sub->abs->mean: already-native primitives."""
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        _assert_native(lambda a, b: F.l1_loss(a, b), a, b)

    def test_smooth_l1_loss(self):
        a, b = torch.randn(8, 10), torch.randn(8, 10)
        for red in ("mean", "sum", "none"):
            assert torch.allclose(ops.smooth_l1_loss(a, b, reduction=red), F.smooth_l1_loss(a, b, reduction=red), atol=1e-5)

    def test_binary_cross_entropy(self):
        a = torch.sigmoid(torch.randn(8, 10))
        b = torch.rand(8, 10)
        for red in ("mean", "sum", "none"):
            assert torch.allclose(ops.binary_cross_entropy(a, b, reduction=red), F.binary_cross_entropy(a, b, reduction=red), atol=1e-5)

    def test_mse_export(self):
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        _assert_native(lambda a, b: F.mse_loss(a, b), a, b)

    def test_smooth_l1_export(self):
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        _assert_native(lambda a, b: F.smooth_l1_loss(a, b), a, b)

    def test_bce_export(self):
        a = torch.sigmoid(torch.randn(4, 8))
        b = torch.rand(4, 8)
        _assert_native(lambda a, b: F.binary_cross_entropy(a, b), a, b)

