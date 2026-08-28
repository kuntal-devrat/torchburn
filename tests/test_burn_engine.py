"""Burn-engine linalg regression tests.

These exercise matmul/bmm/linear/addmm through the raw FFI.  They pass on
every engine, but they specifically guard the *burn* path: burn's `matmul`
panics (PanicException) instead of returning on shape mismatch, so the burn
engine must pre-validate shapes and reject with a clean `TB_UNSUPPORTED`
(native fallback) — and it must compute correct results for valid shapes.

Run under any engine:
    python -m pytest tests/test_burn_engine.py -q
    TORCHBURN_ENGINE=burn        python -m pytest tests/test_burn_engine.py -q
    TORCHBURN_ENGINE=burn-wgpu   python -m pytest tests/test_burn_engine.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

import torchburn
from torchburn import _torchburn as _native
from torchburn._parser import payload_json


def _spec(t: torch.Tensor) -> dict:
    return {"dtype": "f32", "shape": list(t.shape)}


def _run(payload: dict, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    caps = [t.__dlpack__() for t in tensors]
    return [torch.from_dlpack(c) for c in _native.execute(payload_json(payload), caps)]


def _linalg_payload(target: str, tensors: list[torch.Tensor], n_args: int) -> dict:
    return {
        "inputs": [_spec(t) for t in tensors],
        "nodes": [
            {
                "id": 0,
                "target": target,
                "args": [{"kind": "slot", "index": i} for i in range(n_args)],
                "kwargs": {},
            }
        ],
        "outputs": [0],
    }


class TestBurnLinalg:
    def test_matmul_native(self):
        a, b = torch.randn(64, 128), torch.randn(128, 32)
        p = _linalg_payload("matmul", [a, b], 2)
        got = _run(p, [a, b])[0]
        assert torch.allclose(got, a @ b, atol=1e-4)

    def test_bmm_native(self):
        a, b = torch.randn(4, 16, 32), torch.randn(4, 32, 8)
        p = _linalg_payload("bmm", [a, b], 2)
        got = _run(p, [a, b])[0]
        assert torch.allclose(got, torch.bmm(a, b), atol=1e-4)

    def test_linear_native_with_bias(self):
        x, w, bias = torch.randn(16, 64), torch.randn(32, 64), torch.randn(32)
        p = _linalg_payload("linear", [x, w, bias], 3)
        got = _run(p, [x, w, bias])[0]
        assert torch.allclose(got, F.linear(x, w, bias), atol=1e-4)

    def test_linear_native_no_bias(self):
        x, w = torch.randn(16, 64), torch.randn(32, 64)
        p = _linalg_payload("linear", [x, w], 2)
        got = _run(p, [x, w])[0]
        assert torch.allclose(got, F.linear(x, w), atol=1e-4)

    def test_addmm_native(self):
        # aten.addmm(bias, mat1, mat2): mat2 is NOT transposed.
        bias, m1, m2 = torch.randn(16), torch.randn(32, 64), torch.randn(64, 16)
        p = _linalg_payload("addmm", [bias, m1, m2], 3)
        got = _run(p, [bias, m1, m2])[0]
        assert torch.allclose(got, m1 @ m2 + bias, atol=1e-4)

    def test_mlp_chain(self):
        """linear -> relu -> linear in one chunk: the real MLP pattern."""
        x = torch.randn(8, 16)
        w1, b1 = torch.randn(32, 16), torch.randn(32)
        w2, b2 = torch.randn(8, 32), torch.randn(8)
        tensors = [x, w1, b1, w2, b2]
        payload = {
            "inputs": [_spec(t) for t in tensors],
            "nodes": [
                {"id": 0, "target": "linear", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}, {"kind": "slot", "index": 2}], "kwargs": {}},
                {"id": 1, "target": "relu", "args": [{"kind": "slot", "index": 5}], "kwargs": {}},
                {"id": 2, "target": "linear", "args": [{"kind": "slot", "index": 6}, {"kind": "slot", "index": 3}, {"kind": "slot", "index": 4}], "kwargs": {}},
            ],
            "outputs": [2],
        }
        got = _run(payload, tensors)[0]
        ref = F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_matmul_mismatch_clean_rejection(self):
        """burn's matmul panics on mismatch; the engine must reject cleanly."""
        a, b = torch.randn(4, 3), torch.randn(5, 4)
        p = _linalg_payload("matmul", [a, b], 2)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            _run(p, [a, b])

    def test_bmm_batch_mismatch_clean_rejection(self):
        a, b = torch.randn(2, 4, 4), torch.randn(3, 4, 4)
        p = _linalg_payload("bmm", [a, b], 2)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            _run(p, [a, b])
