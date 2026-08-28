"""Native kernel correctness tests: add/sub/mul/div/relu vs. PyTorch.

Covers broadcasting, scalar-free tensor operands, f32/f64, and shape edge
cases through the full Rust payload engine.
"""

from __future__ import annotations

import json

import pytest
import torch

from torchburn import _torchburn as tb


def run_binary(target: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inputs = [
        {"shape": [int(s) for s in a.shape], "dtype": "f32" if a.dtype == torch.float32 else "f64"},
        {"shape": [int(s) for s in b.shape], "dtype": "f32" if b.dtype == torch.float32 else "f64"},
    ]
    payload = json.dumps(
        {
            "inputs": inputs,
            "nodes": [{"id": 0, "target": target, "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}]}],
            "outputs": [0],
        },
        sort_keys=True,
    )
    (capsule,) = tb.execute(payload, [a.__dlpack__(), b.__dlpack__()])
    return torch.from_dlpack(capsule)


def run_unary(target: str, a: torch.Tensor) -> torch.Tensor:
    payload = json.dumps(
        {
            "inputs": [{"shape": [int(s) for s in a.shape], "dtype": "f32" if a.dtype == torch.float32 else "f64"}],
            "nodes": [{"id": 0, "target": target, "args": [{"kind": "slot", "index": 0}]}],
            "outputs": [0],
        },
        sort_keys=True,
    )
    (capsule,) = tb.execute(payload, [a.__dlpack__()])
    return torch.from_dlpack(capsule)


OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
]

SHAPES = [
    ((2, 3), (2, 3)),
    ((4,), (4,)),
    ((2, 3), (3,)),
    ((4, 1), (1, 4)),
    ((2, 3, 4), (2, 3, 4)),
    ((2, 1, 4), (1, 3, 4)),
    ((5,), (1,)),
]


@pytest.mark.parametrize("op,ref", OPS)
@pytest.mark.parametrize("shape_a,shape_b", SHAPES)
def test_binary_matches_torch(op, ref, shape_a, shape_b):
    a = torch.randn(*shape_a, dtype=torch.float32) + 1.0
    b = torch.randn(*shape_b, dtype=torch.float32) + 1.0
    try:
        expected = ref(a, b)
    except RuntimeError:
        pytest.skip("shapes not broadcastable")
    got = run_binary(op, a, b)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("op,ref", OPS)
def test_binary_f64(op, ref):
    a = torch.randn(5, 7, dtype=torch.float64)
    b = torch.randn(5, 7, dtype=torch.float64) + 1.0
    got = run_binary(op, a, b)
    assert torch.allclose(got, ref(a, b))


def test_relu_matches_torch():
    a = torch.randn(6, 6, dtype=torch.float32) * 3 - 1
    got = run_unary("relu", a)
    assert torch.allclose(got, torch.relu(a))


def test_relu_f64():
    a = torch.randn(3, 4, dtype=torch.float64) * 2 - 1
    got = run_unary("relu", a)
    assert torch.allclose(got, torch.relu(a))


def test_scalar_0d_tensor():
    a = torch.tensor(3.5)
    b = torch.tensor(1.5)
    got = run_binary("add", a, b)
    assert got.shape == ()
    assert torch.allclose(got, a + b)


def test_broadcast_scalar_vs_matrix():
    a = torch.randn(2, 3)
    b = torch.tensor(2.0)
    got = run_binary("mul", a, b)
    assert torch.allclose(got, a * 2.0)


def test_dtype_mismatch_rejected():
    a = torch.randn(4, dtype=torch.float32)
    b = torch.randn(4, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
        run_binary("add", a, b)


def test_unknown_target_rejected():
    payload = json.dumps(
        {
            "inputs": [{"shape": [2], "dtype": "f32"}],
            "nodes": [{"id": 0, "target": "conv2d", "args": [{"kind": "slot", "index": 0}]}],
            "outputs": [0],
        }
    )
    a = torch.randn(2)
    with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
        tb.execute(payload, [a.__dlpack__()])


def test_non_contiguous_input_read_via_strides():
    """Strided (non-contiguous) inputs are read correctly through their strides."""
    base = torch.randn(6, 6)
    a = base[::2, ::2]  # non-contiguous view
    assert not a.is_contiguous()
    got = run_binary("add", a, a)
    assert torch.allclose(got, a + a)
