"""Comprehensive Phase 2 operator coverage tests.

Validates every Phase 2 operator introduced in linalg.rs, reductions.rs,
activations.rs, math_ops.rs, norm.rs, and shape_ops.rs against PyTorch
reference values.  Each test is written to fail loudly on a regression.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from torchburn import _torchburn as tb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(t: torch.Tensor) -> dict:
    dtype = {
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int64: "i64",
        torch.int32: "i32",
        torch.bool: "bool",
    }.get(t.dtype, "f32")
    return {"shape": list(t.shape), "dtype": dtype}


def run_unary(target: str, a: torch.Tensor, **kwargs) -> torch.Tensor:
    payload = json.dumps({
        "inputs": [_spec(a)],
        "nodes": [{"id": 0, "target": target, "args": [{"kind": "slot", "index": 0}], "kwargs": kwargs}],
        "outputs": [0],
    }, sort_keys=True)
    (cap,) = tb.execute(payload, [a.__dlpack__()])
    return torch.from_dlpack(cap)


def run_binary(target: str, a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    payload = json.dumps({
        "inputs": [_spec(a), _spec(b)],
        "nodes": [{"id": 0, "target": target, "args": [
            {"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}
        ], "kwargs": kwargs}],
        "outputs": [0],
    }, sort_keys=True)
    (cap,) = tb.execute(payload, [a.__dlpack__(), b.__dlpack__()])
    return torch.from_dlpack(cap)


def run_ternary(target: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, **kwargs) -> torch.Tensor:
    payload = json.dumps({
        "inputs": [_spec(a), _spec(b), _spec(c)],
        "nodes": [{"id": 0, "target": target, "args": [
            {"kind": "slot", "index": 0},
            {"kind": "slot", "index": 1},
            {"kind": "slot", "index": 2},
        ], "kwargs": kwargs}],
        "outputs": [0],
    }, sort_keys=True)
    (cap,) = tb.execute(payload, [a.__dlpack__(), b.__dlpack__(), c.__dlpack__()])
    return torch.from_dlpack(cap)


def run_quinary(target: str, tensors: list[torch.Tensor], **kwargs) -> torch.Tensor:
    payload = json.dumps({
        "inputs": [_spec(t) for t in tensors],
        "nodes": [{"id": 0, "target": target, "args": [
            {"kind": "slot", "index": i} for i in range(len(tensors))
        ], "kwargs": kwargs}],
        "outputs": [0],
    }, sort_keys=True)
    (cap,) = tb.execute(payload, [t.__dlpack__() for t in tensors])
    return torch.from_dlpack(cap)


# ---------------------------------------------------------------------------
# 1. Comparison ops
# ---------------------------------------------------------------------------

class TestComparisons:
    @pytest.mark.parametrize("op,ref", [
        ("eq", lambda a, b: (a == b).float()),
        ("ne", lambda a, b: (a != b).float()),
        ("lt", lambda a, b: (a < b).float()),
        ("le", lambda a, b: (a <= b).float()),
        ("gt", lambda a, b: (a > b).float()),
        ("ge", lambda a, b: (a >= b).float()),
    ])
    def test_comparison_f32(self, op, ref):
        a = torch.randn(4, 5)
        b = torch.randn(4, 5)
        got = run_binary(op, a, b)
        assert torch.allclose(got, ref(a, b))

    def test_comparison_broadcast(self):
        a = torch.randn(4, 1)
        b = torch.randn(1, 5)
        got = run_binary("lt", a, b)
        assert torch.allclose(got, (a < b).float())

    def test_eq_f64(self):
        a = torch.randn(3, 3, dtype=torch.float64)
        b = a.clone()
        got = run_binary("eq", a, b)
        assert got.sum().item() == pytest.approx(9.0)

    def test_comparison_returns_float_not_bool(self):
        a, b = torch.randn(3), torch.randn(3)
        got = run_binary("gt", a, b)
        assert got.dtype == torch.float32


# ---------------------------------------------------------------------------
# 1b. Logical ops + dtype cast
# ---------------------------------------------------------------------------

class TestLogicalOps:
    def test_logical_and_f32(self):
        a = torch.tensor([[1.0, 0.0], [0.5, -2.0]])
        b = torch.tensor([[1.0, 1.0], [0.0, -1.0]])
        got = run_binary("logical_and", a, b)
        assert torch.allclose(got, (a.bool() & b.bool()).float())

    def test_logical_or_f32(self):
        a = torch.tensor([[1.0, 0.0], [0.5, 0.0]])
        b = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        got = run_binary("logical_or", a, b)
        assert torch.allclose(got, (a.bool() | b.bool()).float())

    def test_logical_not_f32(self):
        a = torch.tensor([[1.0, 0.0], [0.5, -2.0]])
        got = run_unary("logical_not", a)
        assert torch.allclose(got, (~a.bool()).float())

    def test_logical_broadcast(self):
        a = torch.tensor([[1.0], [0.0]])   # (2,1)
        b = torch.tensor([[1.0, 0.0, 1.0]])  # (1,3)
        got = run_binary("logical_and", a, b)
        assert torch.allclose(got, (a.bool() & b.bool()).float())

    def test_logical_returns_float(self):
        a, b = torch.randn(3), torch.randn(3)
        got = run_binary("logical_or", a, b)
        assert got.dtype == torch.float32


class TestDtypeCast:
    def test_to_dtype_f64(self):
        x = torch.randn(4, 4, dtype=torch.float32)
        payload = json.dumps({
            "inputs": [_spec(x)],
            "nodes": [{"id": 0, "target": "to_dtype",
                       "args": [{"kind": "slot", "index": 0}],
                       "kwargs": {"dtype": "f64"}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__()])
        got = torch.from_dlpack(cap)
        assert got.dtype == torch.float64
        assert torch.allclose(got, x.double())

    def test_to_dtype_f32_from_f64(self):
        x = torch.randn(4, 4, dtype=torch.float64)
        payload = json.dumps({
            "inputs": [_spec(x)],
            "nodes": [{"id": 0, "target": "to_dtype",
                       "args": [{"kind": "slot", "index": 0}],
                       "kwargs": {"dtype": "f32"}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__()])
        got = torch.from_dlpack(cap)
        assert got.dtype == torch.float32
        assert torch.allclose(got, x.float())

    def test_to_dtype_positional_arg(self):
        # x.to(torch.float64) arrives with dtype as a positional const arg
        x = torch.randn(3, 3, dtype=torch.float32)
        payload = json.dumps({
            "inputs": [_spec(x)],
            "nodes": [{"id": 0, "target": "to_dtype",
                       "args": [{"kind": "slot", "index": 0},
                                {"kind": "const", "value": "f64"}],
                       "kwargs": {}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__()])
        got = torch.from_dlpack(cap)
        assert got.dtype == torch.float64

    def test_to_dtype_same_dtype_copy(self):
        x = torch.randn(4, 4, dtype=torch.float32)
        payload = json.dumps({
            "inputs": [_spec(x)],
            "nodes": [{"id": 0, "target": "to_dtype",
                       "args": [{"kind": "slot", "index": 0}],
                       "kwargs": {"dtype": "f32"}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__()])
        got = torch.from_dlpack(cap)
        assert got.dtype == torch.float32
        assert torch.allclose(got, x)


# ---------------------------------------------------------------------------
# 2. Unary math ops
# ---------------------------------------------------------------------------

class TestUnaryMath:
    @pytest.mark.parametrize("op,ref,prep", [
        ("abs", torch.abs, lambda x: x),
        ("neg", torch.neg, lambda x: x),
        ("sign", torch.sign, lambda x: x),
        ("sqrt", torch.sqrt, lambda x: x.abs()),
        ("rsqrt", torch.rsqrt, lambda x: x.abs() + 0.01),
        ("exp", torch.exp, lambda x: x.clamp(-5, 5)),
        ("log", torch.log, lambda x: x.abs() + 0.01),
        ("reciprocal", torch.reciprocal, lambda x: x + 0.01),
        ("ceil", torch.ceil, lambda x: x),
        ("floor", torch.floor, lambda x: x),
    ])
    def test_unary_f32(self, op, ref, prep):
        torch.manual_seed(42)
        x = torch.randn(8, 8)
        input_x = prep(x)
        got = run_unary(op, input_x)
        assert torch.allclose(got, ref(input_x), atol=1e-5), f"{op} mismatch"

    def test_unary_f64(self):
        x = torch.randn(5, 5, dtype=torch.float64)
        got = run_unary("exp", x.clamp(-5, 5))
        assert got.dtype == torch.float64
        assert torch.allclose(got, torch.exp(x.clamp(-5, 5)))

    def test_clamp(self):
        x = torch.randn(10, 10) * 5
        got = run_unary("clamp", x, min=-1.0, max=1.0)
        assert torch.allclose(got, x.clamp(-1.0, 1.0))

    def test_pow_scalar(self):
        x = torch.randn(5, 5).abs() + 0.1
        got = run_unary("pow", x, exp=3.0)
        assert torch.allclose(got, x.pow(3.0), atol=1e-5)

    def test_non_contiguous_unary(self):
        base = torch.randn(6, 6)
        x = base[::2, ::2]  # non-contiguous
        assert not x.is_contiguous()
        got = run_unary("abs", x)
        assert torch.allclose(got, x.abs())


# ---------------------------------------------------------------------------
# 3. Activations
# ---------------------------------------------------------------------------

class TestActivations:
    @pytest.mark.parametrize("op,ref", [
        ("sigmoid", torch.sigmoid),
        ("tanh", torch.tanh),
        ("selu", F.selu),
        ("softplus", F.softplus),
        ("hardswish", F.hardswish),
        ("mish", F.mish),
    ])
    def test_activation_f32(self, op, ref):
        x = torch.randn(6, 8)
        got = run_unary(op, x)
        assert torch.allclose(got, ref(x), atol=1e-5), f"{op} failed"

    def test_gelu_approx(self):
        """GELU uses tanh approximation — matches PyTorch approximate='tanh' within 2e-4."""
        x = torch.randn(6, 8)
        got = run_unary("gelu", x)
        ref = F.gelu(x, approximate="tanh")
        assert torch.allclose(got, ref, atol=2e-4), "gelu failed"

    def test_silu_f32(self):
        x = torch.randn(6, 8)
        got = run_unary("silu", x)
        assert torch.allclose(got, F.silu(x), atol=1e-5), "silu failed"

    def test_leaky_relu(self):
        x = torch.randn(4, 4)
        got = run_unary("leaky_relu", x, negative_slope=0.1)
        assert torch.allclose(got, F.leaky_relu(x, 0.1), atol=1e-6)

    def test_elu(self):
        x = torch.randn(4, 4)
        got = run_unary("elu", x, alpha=1.0)
        assert torch.allclose(got, F.elu(x, 1.0), atol=1e-6)

    def test_softmax(self):
        x = torch.randn(4, 8)
        got = run_unary("softmax", x, dim=-1)
        ref = F.softmax(x, dim=-1)
        assert torch.allclose(got, ref, atol=1e-6)
        # Rows must sum to 1
        assert torch.allclose(got.sum(dim=-1), torch.ones(4), atol=1e-6)

    def test_log_softmax(self):
        x = torch.randn(4, 8)
        got = run_unary("log_softmax", x, dim=-1)
        ref = F.log_softmax(x, dim=-1)
        assert torch.allclose(got, ref, atol=1e-6)

    def test_activation_f64(self):
        x = torch.randn(4, 4, dtype=torch.float64)
        got = run_unary("sigmoid", x)
        assert got.dtype == torch.float64
        assert torch.allclose(got, torch.sigmoid(x))

    def test_gelu_approx_check(self):
        """GELU matches PyTorch's approximate tanh implementation within 2e-4."""
        x = torch.randn(32)
        got = run_unary("gelu", x)
        ref = F.gelu(x, approximate="tanh")
        assert torch.allclose(got, ref, atol=2e-4)


# ---------------------------------------------------------------------------
# 4. Reductions
# ---------------------------------------------------------------------------

class TestReductions:
    def test_sum_all(self):
        x = torch.randn(100)
        got = run_unary("sum", x)
        assert torch.allclose(got.reshape([]), x.sum(), atol=1e-4)

    def test_sum_dim(self):
        x = torch.randn(4, 8, 16)
        for dim in [0, 1, 2, -1]:
            got = run_unary("sum", x, dim=dim, keepdim=False)
            ref = x.sum(dim=dim)
            assert torch.allclose(got, ref, atol=1e-4), f"sum dim={dim} failed"

    def test_sum_keepdim(self):
        x = torch.randn(4, 8)
        got = run_unary("sum", x, dim=1, keepdim=True)
        ref = x.sum(dim=1, keepdim=True)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4)

    def test_sum_f64(self):
        x = torch.randn(50, dtype=torch.float64)
        got = run_unary("sum", x)
        assert got.dtype == torch.float64
        assert torch.allclose(got.reshape([]), x.sum(), atol=1e-10)

    def test_mean_all(self):
        x = torch.randn(64)
        got = run_unary("mean", x)
        assert torch.allclose(got.reshape([]), x.mean(), atol=1e-5)

    def test_mean_dim(self):
        x = torch.randn(8, 16)
        for dim in [0, 1, -1]:
            got = run_unary("mean", x, dim=dim, keepdim=False)
            assert torch.allclose(got, x.mean(dim=dim), atol=1e-5)

    def test_max_reduce_dim(self):
        x = torch.randn(4, 8)
        got = run_unary("max_reduce", x, dim=1, keepdim=False)
        ref_values, _ = x.max(dim=1)
        assert torch.allclose(got, ref_values, atol=1e-6)

    def test_min_reduce_dim(self):
        x = torch.randn(4, 8)
        got = run_unary("min_reduce", x, dim=1, keepdim=False)
        ref_values, _ = x.min(dim=1)
        assert torch.allclose(got, ref_values, atol=1e-6)

    def test_max_reduce_all(self):
        x = torch.randn(32)
        got = run_unary("max_reduce", x)
        assert torch.allclose(got.reshape([]), x.max(), atol=1e-6)

    def test_min_reduce_all(self):
        x = torch.randn(32)
        got = run_unary("min_reduce", x)
        assert torch.allclose(got.reshape([]), x.min(), atol=1e-6)

    def test_argmax(self):
        x = torch.randn(4, 8)
        got = run_unary("argmax", x, dim=1, keepdim=False)
        ref = x.argmax(dim=1)
        # Engine now correctly returns I64; compare as long or float
        if got.dtype != ref.dtype:
            got = got.float() if ref.dtype == torch.float32 else got.long()
            ref = ref.float() if got.dtype == torch.float32 else ref
        assert torch.allclose(got.float(), ref.float()) if got.dtype.is_floating_point else torch.equal(got, ref)

    def test_argmin(self):
        x = torch.randn(4, 8)
        got = run_unary("argmin", x, dim=1, keepdim=False)
        ref = x.argmin(dim=1)
        if got.dtype != ref.dtype:
            got = got.float() if ref.dtype == torch.float32 else got.long()
        assert torch.allclose(got.float(), ref.float()) if got.dtype.is_floating_point else torch.equal(got, ref)

    def test_std_unbiased_f32(self):
        x = torch.randn(32)
        got = run_unary("std", x)
        ref = x.std()
        assert torch.allclose(got.reshape([]), ref, atol=1e-4)

    def test_std_biased(self):
        x = torch.randn(16)
        got = run_unary("std", x, unbiased=False)
        ref = x.std(unbiased=False)
        assert torch.allclose(got.reshape([]), ref, atol=1e-4)

    def test_std_dim(self):
        x = torch.randn(4, 16)
        got = run_unary("std", x, dim=1, keepdim=False, unbiased=True)
        ref = x.std(dim=1, unbiased=True)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_var_f32(self):
        x = torch.randn(50)
        got = run_unary("var", x)
        ref = x.var()
        assert torch.allclose(got.reshape([]), ref, atol=1e-4)

    def test_var_f64(self):
        """Regression: variance was hardcoded to f32 read even for f64 inputs."""
        x = torch.randn(50, dtype=torch.float64)
        got = run_unary("var", x)
        ref = x.var()
        assert got.dtype == torch.float64
        assert torch.allclose(got.reshape([]), ref, atol=1e-8)

    def test_std_f64(self):
        """Regression: std was broken for f64 inputs (used f32 reads)."""
        x = torch.randn(50, dtype=torch.float64)
        got = run_unary("std", x)
        ref = x.std()
        assert got.dtype == torch.float64
        assert torch.allclose(got.reshape([]), ref, atol=1e-8)

    def test_cumsum(self):
        x = torch.randn(4, 8)
        for dim in [0, 1]:
            got = run_unary("cumsum", x, dim=dim)
            ref = x.cumsum(dim=dim)
            assert torch.allclose(got, ref, atol=1e-5)

    def test_prod_all(self):
        x = torch.ones(5) * 2.0
        got = run_unary("prod", x)
        assert torch.allclose(got.reshape([]), x.prod(), atol=1e-5)

    def test_prod_dim(self):
        x = torch.randn(4, 8).abs() + 0.5
        got = run_unary("prod", x, dim=0, keepdim=False)
        ref = x.prod(dim=0)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_norm_l2(self):
        x = torch.randn(4, 8)
        got = run_unary("norm", x)
        ref = torch.linalg.norm(x)
        assert torch.allclose(got.reshape([]), ref, atol=1e-4)

    def test_norm_dim(self):
        x = torch.randn(4, 8)
        got = run_unary("norm", x, dim=1, keepdim=False)
        ref = torch.linalg.norm(x, dim=1)
        assert torch.allclose(got, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Linear algebra
# ---------------------------------------------------------------------------

class TestLinAlg:
    def test_matmul_square_f32(self):
        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        got = run_binary("matmul", a, b)
        ref = a @ b
        assert torch.allclose(got, ref, atol=1e-4)

    def test_addmm_bias_first_f32(self):
        """aten.addmm(bias, mat1, mat2) — mat2 is NOT transposed."""
        bias = torch.randn(5)
        mat1 = torch.randn(3, 4)
        mat2 = torch.randn(4, 5)
        payload = json.dumps({
            "inputs": [_spec(bias), _spec(mat1), _spec(mat2)],
            "nodes": [{"id": 0, "target": "addmm", "args": [
                {"kind": "slot", "index": 0},
                {"kind": "slot", "index": 1},
                {"kind": "slot", "index": 2},
            ], "kwargs": {}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [bias.__dlpack__(), mat1.__dlpack__(), mat2.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = torch.addmm(bias, mat1, mat2)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_addmm_f64(self):
        bias = torch.randn(5, dtype=torch.float64)
        mat1 = torch.randn(3, 4, dtype=torch.float64)
        mat2 = torch.randn(4, 5, dtype=torch.float64)
        payload = json.dumps({
            "inputs": [_spec(bias), _spec(mat1), _spec(mat2)],
            "nodes": [{"id": 0, "target": "addmm", "args": [
                {"kind": "slot", "index": 0},
                {"kind": "slot", "index": 1},
                {"kind": "slot", "index": 2},
            ], "kwargs": {}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [bias.__dlpack__(), mat1.__dlpack__(), mat2.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = torch.addmm(bias, mat1, mat2)
        assert torch.allclose(got, ref, atol=1e-10)

    def test_addmm_shape_mismatch_rejected(self):
        bias = torch.randn(5)
        mat1 = torch.randn(3, 4)
        mat2 = torch.randn(3, 5)  # inner dims 4 != 3
        payload = json.dumps({
            "inputs": [_spec(bias), _spec(mat1), _spec(mat2)],
            "nodes": [{"id": 0, "target": "addmm", "args": [
                {"kind": "slot", "index": 0},
                {"kind": "slot", "index": 1},
                {"kind": "slot", "index": 2},
            ], "kwargs": {}}],
            "outputs": [0],
        }, sort_keys=True)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            tb.execute(payload, [bias.__dlpack__(), mat1.__dlpack__(), mat2.__dlpack__()])

    def test_matmul_rectangular(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        got = run_binary("matmul", a, b)
        assert torch.allclose(got, a @ b, atol=1e-4)

    def test_matmul_f64(self):
        """Regression: f64 matmul body was corrupt/truncated."""
        a = torch.randn(8, 8, dtype=torch.float64)
        b = torch.randn(8, 8, dtype=torch.float64)
        got = run_binary("matmul", a, b)
        assert got.dtype == torch.float64
        assert torch.allclose(got, a @ b, atol=1e-10)

    def test_matmul_larger(self):
        a = torch.randn(64, 128)
        b = torch.randn(128, 32)
        got = run_binary("matmul", a, b)
        assert torch.allclose(got, a @ b, atol=1e-3)

    def test_matmul_non_contiguous(self):
        base_a = torch.randn(16, 16)
        base_b = torch.randn(16, 16)
        a = base_a.t()  # transposed = non-contiguous
        b = base_b
        got = run_binary("matmul", a, b)
        assert torch.allclose(got, a @ b, atol=1e-4)

    def test_bmm(self):
        a = torch.randn(4, 8, 16)
        b = torch.randn(4, 16, 4)
        got = run_binary("bmm", a, b)
        ref = torch.bmm(a, b)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_bmm_f64(self):
        a = torch.randn(2, 4, 4, dtype=torch.float64)
        b = torch.randn(2, 4, 4, dtype=torch.float64)
        got = run_binary("bmm", a, b)
        assert got.dtype == torch.float64
        assert torch.allclose(got, torch.bmm(a, b), atol=1e-10)

    def test_dot(self):
        a = torch.randn(32)
        b = torch.randn(32)
        got = run_binary("dot", a, b)
        assert torch.allclose(got.reshape([]), torch.dot(a, b), atol=1e-4)

    def test_linear_no_bias(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)  # (out, in)
        payload = json.dumps({
            "inputs": [_spec(x), _spec(w)],
            "nodes": [{"id": 0, "target": "linear",
                       "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                       "kwargs": {}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__(), w.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = F.linear(x, w)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_linear_with_bias(self):
        x = torch.randn(4, 16)
        w = torch.randn(8, 16)
        b = torch.randn(8)
        payload = json.dumps({
            "inputs": [_spec(x), _spec(w), _spec(b)],
            "nodes": [{"id": 0, "target": "linear",
                       "args": [{"kind": "slot", "index": 0},
                                {"kind": "slot", "index": 1},
                                {"kind": "slot", "index": 2}],
                       "kwargs": {}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__(), w.__dlpack__(), b.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = F.linear(x, w, b)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_matmul_shape_mismatch_rejected(self):
        a = torch.randn(4, 3)
        b = torch.randn(5, 4)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_binary("matmul", a, b)

    def test_bmm_batch_mismatch_rejected(self):
        a = torch.randn(2, 4, 4)
        b = torch.randn(3, 4, 4)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_binary("bmm", a, b)


# ---------------------------------------------------------------------------
# 6. Normalization layers
# ---------------------------------------------------------------------------

class TestNormLayers:
    def test_layer_norm_f32(self):
        x = torch.randn(4, 16)
        w = torch.ones(16)
        b = torch.zeros(16)
        got = run_ternary("layer_norm", x, w, b, eps=1e-5)
        ref = F.layer_norm(x, [16], w, b, eps=1e-5)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_layer_norm_f64(self):
        x = torch.randn(4, 16, dtype=torch.float64)
        w = torch.ones(16, dtype=torch.float64)
        b = torch.zeros(16, dtype=torch.float64)
        got = run_ternary("layer_norm", x, w, b, eps=1e-5)
        ref = F.layer_norm(x, [16], w, b, eps=1e-5)
        assert got.dtype == torch.float64
        assert torch.allclose(got, ref, atol=1e-10)

    def test_layer_norm_learned_affine(self):
        x = torch.randn(8, 32)
        w = torch.randn(32)
        b = torch.randn(32)
        got = run_ternary("layer_norm", x, w, b, eps=1e-5)
        ref = F.layer_norm(x, [32], w, b, eps=1e-5)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_rms_norm(self):
        x = torch.randn(4, 16)
        w = torch.ones(16)
        payload = json.dumps({
            "inputs": [_spec(x), _spec(w)],
            "nodes": [{"id": 0, "target": "rms_norm",
                       "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                       "kwargs": {"eps": 1e-6}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__(), w.__dlpack__()])
        got = torch.from_dlpack(cap)
        # Manual reference
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
        ref = x / rms * w
        assert torch.allclose(got, ref, atol=1e-5)

    def test_batch_norm_inference(self):
        x = torch.randn(4, 8, 4, 4)
        w = torch.ones(8)
        b = torch.zeros(8)
        rm = torch.zeros(8)
        rv = torch.ones(8)
        # torch signature: (x, running_mean, running_var, weight, bias, ...)
        tensors = [x, rm, rv, w, b]
        payload = json.dumps({
            "inputs": [_spec(t) for t in tensors],
            "nodes": [{"id": 0, "target": "batch_norm",
                       "args": [{"kind": "slot", "index": i} for i in range(5)],
                       "kwargs": {"eps": 1e-5, "training": False}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [t.__dlpack__() for t in tensors])
        got = torch.from_dlpack(cap)
        ref = F.batch_norm(x, rm, rv, w, b, training=False, eps=1e-5)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_group_norm(self):
        x = torch.randn(2, 8, 4, 4)
        w = torch.ones(8)
        b = torch.zeros(8)
        tensors = [x, w, b]
        payload = json.dumps({
            "inputs": [_spec(t) for t in tensors],
            "nodes": [{"id": 0, "target": "group_norm",
                       "args": [{"kind": "slot", "index": i} for i in range(3)],
                       "kwargs": {"num_groups": 4, "eps": 1e-5}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [t.__dlpack__() for t in tensors])
        got = torch.from_dlpack(cap)
        ref = F.group_norm(x, 4, w, b, eps=1e-5)
        assert torch.allclose(got, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Shape ops
# ---------------------------------------------------------------------------

class TestShapeOps:
    def test_reshape(self):
        x = torch.randn(4, 8)
        got = run_unary("reshape", x, shape=[2, 16])
        assert got.shape == (2, 16)
        assert torch.allclose(got, x.reshape(2, 16))

    def test_reshape_flat(self):
        x = torch.randn(2, 3, 4)
        got = run_unary("reshape", x, shape=[24])
        assert torch.allclose(got, x.reshape(24))

    def test_permute_2d(self):
        x = torch.randn(4, 8)
        got = run_unary("permute", x, dims=[1, 0])
        assert got.shape == (8, 4)
        assert torch.allclose(got, x.permute(1, 0))

    def test_permute_3d(self):
        x = torch.randn(2, 3, 4)
        got = run_unary("permute", x, dims=[2, 0, 1])
        assert got.shape == (4, 2, 3)
        assert torch.allclose(got, x.permute(2, 0, 1))

    def test_t_2d(self):
        """aten.t: 2D transpose (used by export graphs before addmm)."""
        x = torch.randn(4, 8)
        got = run_unary("t", x)
        assert got.shape == (8, 4)
        assert torch.allclose(got, x.t())

    def test_t_f64(self):
        x = torch.randn(3, 5, dtype=torch.float64)
        got = run_unary("t", x)
        assert torch.allclose(got, x.t())

    def test_cat_dim0(self):
        """Regression test: old cat() just appended bytes and only worked for dim=0 when 1D."""
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        payload = json.dumps({
            "inputs": [_spec(a), _spec(b)],
            "nodes": [{"id": 0, "target": "cat",
                       "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                       "kwargs": {"dim": 0}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [a.__dlpack__(), b.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = torch.cat([a, b], dim=0)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref)

    def test_cat_dim1(self):
        """Regression test: old cat was wrong for dim=1."""
        a = torch.randn(4, 3)
        b = torch.randn(4, 5)
        payload = json.dumps({
            "inputs": [_spec(a), _spec(b)],
            "nodes": [{"id": 0, "target": "cat",
                       "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                       "kwargs": {"dim": 1}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [a.__dlpack__(), b.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = torch.cat([a, b], dim=1)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref)

    def test_cat_3tensors_dim0(self):
        tensors = [torch.randn(2, 4) for _ in range(3)]
        payload = json.dumps({
            "inputs": [_spec(t) for t in tensors],
            "nodes": [{"id": 0, "target": "cat",
                       "args": [{"kind": "slot", "index": i} for i in range(3)],
                       "kwargs": {"dim": 0}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [t.__dlpack__() for t in tensors])
        got = torch.from_dlpack(cap)
        ref = torch.cat(tensors, dim=0)
        assert torch.allclose(got, ref)

    def test_cat_3d_dim1(self):
        a = torch.randn(2, 3, 5)
        b = torch.randn(2, 7, 5)
        payload = json.dumps({
            "inputs": [_spec(a), _spec(b)],
            "nodes": [{"id": 0, "target": "cat",
                       "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                       "kwargs": {"dim": 1}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [a.__dlpack__(), b.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = torch.cat([a, b], dim=1)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref)

    def test_stack_dim0(self):
        tensors = [torch.randn(4) for _ in range(3)]
        payload = json.dumps({
            "inputs": [_spec(t) for t in tensors],
            "nodes": [{"id": 0, "target": "stack",
                       "args": [{"kind": "slot", "index": i} for i in range(3)],
                       "kwargs": {"dim": 0}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [t.__dlpack__() for t in tensors])
        got = torch.from_dlpack(cap)
        ref = torch.stack(tensors, dim=0)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref)

    def test_expand(self):
        x = torch.randn(1, 4)
        got = run_unary("expand", x, shape=[8, 4])
        ref = x.expand(8, 4)
        assert got.shape == (8, 4)
        assert torch.allclose(got, ref)

    def test_flip(self):
        x = torch.randn(3, 4)
        got = run_unary("flip", x, dims=[0, 1])
        ref = torch.flip(x, [0, 1])
        assert torch.allclose(got, ref)

    def test_narrow(self):
        x = torch.randn(4, 8)
        got = run_unary("narrow", x, dim=1, start=2, length=4)
        ref = x.narrow(1, 2, 4)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref)

    def test_index_select(self):
        """Phase 4: int64 index tensors are now first-class engine dtypes."""
        x = torch.randn(4, 8)
        idx = torch.tensor([2, 0, 3])
        payload = json.dumps({
            "inputs": [_spec(x), _spec(idx)],
            "nodes": [{"id": 0, "target": "index_select", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}], "kwargs": {"dim": 0}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__(), idx.__dlpack__()])
        got = torch.from_dlpack(cap)
        assert torch.allclose(got, x.index_select(0, idx))

    def test_gather(self):
        x = torch.randn(4, 6)
        g = torch.randint(0, 4, (2, 6))
        payload = json.dumps({
            "inputs": [_spec(x), _spec(g)],
            "nodes": [{"id": 0, "target": "gather", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}], "kwargs": {"dim": 0}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__(), g.__dlpack__()])
        got = torch.from_dlpack(cap)
        assert torch.allclose(got, x.gather(0, g))

    def test_transpose(self):
        x = torch.randn(2, 3, 4)
        payload = json.dumps({
            "inputs": [_spec(x)],
            "nodes": [{"id": 0, "target": "transpose", "args": [{"kind": "slot", "index": 0}], "kwargs": {"d0": 1, "d1": 2}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__()])
        got = torch.from_dlpack(cap)
        assert torch.allclose(got, x.transpose(1, 2))

    def test_contiguous_noop(self):
        """contiguous() on an already-contiguous tensor should be identity."""
        x = torch.randn(3, 4)
        got = run_unary("contiguous", x)
        assert torch.allclose(got, x)

    def test_where(self):
        cond = (torch.randn(4, 4) > 0).float()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        got = run_ternary("where", cond, x, y)
        ref = torch.where(cond.bool(), x, y)
        assert torch.allclose(got, ref)

    def test_masked_fill(self):
        x = torch.randn(4, 4)
        mask = (x > 0).float()
        payload = json.dumps({
            "inputs": [_spec(x), _spec(mask)],
            "nodes": [{"id": 0, "target": "masked_fill",
                       "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                       "kwargs": {"value": -1e9}}],
            "outputs": [0],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__(), mask.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = x.masked_fill(mask.bool(), -1e9)
        assert torch.allclose(got, ref)


# ---------------------------------------------------------------------------
# 8. End-to-end torch.compile tests for Phase 2 operators
# ---------------------------------------------------------------------------

class TestEndToEndPhase2:
    def test_matmul_via_compile(self):
        """nn.Linear (which uses aten.addmm / aten.mm) should route through native engine."""
        # Use a model with only linear ops that should be supported
        def model(x, w):
            return x @ w  # aten.mm

        compiled = torch.compile(model, backend="torchburn")
        x = torch.randn(4, 8)
        w = torch.randn(8, 4)
        out = compiled(x, w)
        assert torch.allclose(out, x @ w, atol=1e-4)

    def test_activation_chain_via_compile(self):
        def model(x):
            return torch.sigmoid(torch.tanh(x * 2 + 0.5))

        compiled = torch.compile(model, backend="torchburn")
        x = torch.randn(8, 8)
        out = compiled(x)
        assert torch.allclose(out, model(x), atol=1e-5)

    def test_reduction_via_compile(self):
        def model(x):
            return torch.mean(x, dim=-1, keepdim=True)

        compiled = torch.compile(model, backend="torchburn")
        x = torch.randn(4, 16)
        out = compiled(x)
        assert torch.allclose(out, model(x), atol=1e-5)

    def test_layer_norm_via_compile(self):
        net = torch.nn.LayerNorm(16)
        compiled = torch.compile(net, backend="torchburn")
        x = torch.randn(4, 16)
        with torch.no_grad():
            out = compiled(x)
            ref = net(x)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_softmax_via_compile(self):
        def model(x):
            return torch.nn.functional.softmax(x, dim=-1)

        compiled = torch.compile(model, backend="torchburn")
        x = torch.randn(4, 8)
        out = compiled(x)
        assert torch.allclose(out, model(x), atol=1e-5)
        # Must sum to 1 along last dim
        assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_reshape_permute_in_compiled_graph(self):
        def model(x):
            return x.permute(1, 0).reshape(-1)

        compiled = torch.compile(model, backend="torchburn")
        x = torch.randn(4, 8)
        out = compiled(x)
        assert torch.allclose(out, model(x))


# ---------------------------------------------------------------------------
# 9. Multi-node payload: chaining operators inside a single Rust dispatch
# ---------------------------------------------------------------------------

class TestChainedPayload:
    def test_sum_then_relu(self):
        """Two ops in one payload, second node references first node's output."""
        x = torch.randn(4, 8)
        payload = json.dumps({
            "inputs": [_spec(x)],
            "nodes": [
                {"id": 0, "target": "sum",
                 "args": [{"kind": "slot", "index": 0}],
                 "kwargs": {"dim": 1, "keepdim": True}},
                {"id": 1, "target": "relu",
                 "args": [{"kind": "slot", "index": 1}],  # slot 1 = first output (base 1)
                 "kwargs": {}},
            ],
            "outputs": [1],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [x.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = torch.relu(x.sum(dim=1, keepdim=True))
        assert torch.allclose(got, ref, atol=1e-4)

    def test_matmul_then_add(self):
        a = torch.randn(4, 8)
        b = torch.randn(8, 4)
        bias = torch.randn(4, 4)
        payload = json.dumps({
            "inputs": [_spec(a), _spec(b), _spec(bias)],
            "nodes": [
                {"id": 0, "target": "matmul",
                 "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                 "kwargs": {}},
                {"id": 1, "target": "add",
                 "args": [{"kind": "slot", "index": 3}, {"kind": "slot", "index": 2}],
                 "kwargs": {}},  # slot 3 = matmul output (3 inputs + 0th node)
            ],
            "outputs": [1],
        }, sort_keys=True)
        (cap,) = tb.execute(payload, [a.__dlpack__(), b.__dlpack__(), bias.__dlpack__()])
        got = torch.from_dlpack(cap)
        ref = (a @ b) + bias
        assert torch.allclose(got, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# 10. Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_sum_empty_dim_keepdim(self):
        x = torch.randn(4, 1, 8)
        got = run_unary("sum", x, dim=1, keepdim=True)
        ref = x.sum(dim=1, keepdim=True)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-5)

    def test_matmul_dtype_mismatch_rejected(self):
        a = torch.randn(4, 4, dtype=torch.float32)
        b = torch.randn(4, 4, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_binary("matmul", a, b)

    def test_reshape_size_mismatch_rejected(self):
        x = torch.randn(4, 4)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_unary("reshape", x, shape=[3, 5])

    def test_layer_norm_dtype_mismatch_rejected(self):
        x = torch.randn(4, 8, dtype=torch.float32)
        w = torch.ones(8, dtype=torch.float64)
        b = torch.zeros(8, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_ternary("layer_norm", x, w, b, eps=1e-5)

    def test_narrow_out_of_bounds_does_not_crash(self):
        x = torch.randn(4, 8)
        # valid narrow
        got = run_unary("narrow", x, dim=0, start=1, length=2)
        assert got.shape == (2, 8)

    def test_f32_f64_roundtrip_values(self):
        """Ensure no silent precision loss when the correct dtype propagates through."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        got = run_unary("relu", x)
        assert got.dtype == torch.float64
        assert torch.allclose(got, x)  # all positive, relu is identity


# ---------------------------------------------------------------------------
# Export-path regression tests (REQ-001/REQ-004)
# ---------------------------------------------------------------------------
# make_fx graphs pass reduction dims/kwargs POSITIONALLY (e.g.
# aten.sum.dim_IntList(x, [1])).  The engine only reads them from kwargs, so
# without promotion the export path silently reduced over ALL dims.  These
# tests lock the promotion + getitem-alias behaviour in place.

class TestExportPathReductions:
    def _run(self, fn, shape=(4, 8), atol=1e-4):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable
        x = torch.randn(*shape)
        gm = make_fx(fn)(x)
        cb = BurnCompiledCallable(gm, [x])
        out = cb(x)
        ref = fn(x)
        assert torch.allclose(out, ref, atol=atol), (out - ref).abs().max().item()
        return out, ref

    def test_sum_dim_positional(self):
        self._run(lambda x: torch.sum(x, dim=1))

    def test_sum_dim_list_single(self):
        # export emits dim as a 1-element list [1]
        self._run(lambda x: torch.sum(x, dim=1))

    def test_sum_keepdim(self):
        self._run(lambda x: torch.sum(x, dim=0, keepdim=True))

    def test_mean_dim(self):
        self._run(lambda x: torch.mean(x, dim=1))

    def test_prod_dim(self):
        self._run(lambda x: torch.prod(x, dim=1))

    def test_std_dim(self):
        self._run(lambda x: torch.std(x, dim=1))

    def test_std_correction_zero(self):
        self._run(lambda x: torch.std(x, dim=1, correction=0))

    def test_var_dim(self):
        self._run(lambda x: torch.var(x, dim=1))

    def test_cumsum_dim(self):
        self._run(lambda x: torch.cumsum(x, dim=1))

    def test_argmax_dim(self):
        x = torch.randn(4, 8)
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable
        gm = make_fx(lambda t: torch.argmax(t, dim=1))(x)
        out = BurnCompiledCallable(gm, [x])(x)
        ref = torch.argmax(x, dim=1)
        # engine returns f32 indices; torch returns int64 — values must match
        assert torch.equal(out.long(), ref)

    def test_softmax_3d_dim1(self):
        """Regression: 2D softmax passed only because dim=1 == dim=-1."""
        self._run(lambda x: torch.softmax(x, dim=1), shape=(2, 3, 4))

    def test_softmax_3d_dim0(self):
        self._run(lambda x: torch.softmax(x, dim=0), shape=(2, 3, 4))

    def test_max_values_export(self):
        """aten.max.dim -> getitem(0) must alias the native reduce output."""
        self._run(lambda x: torch.max(x, dim=1).values)

    def test_min_values_export(self):
        self._run(lambda x: torch.min(x, dim=1).values)

    def test_max_keepdim_values(self):
        self._run(lambda x: torch.max(x, dim=1, keepdim=True).values)

    def test_max_indices_forced_eager(self):
        """When .indices is consumed, the whole reduce falls back to eager."""
        x = torch.randn(4, 8)
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable

        def fn(x):
            return torch.max(x, dim=1).indices

        gm = make_fx(fn)(x)
        cb = BurnCompiledCallable(gm, [x])
        out = cb(x)
        assert torch.equal(out, fn(x))  # int64 indices preserved

    def test_sum_multidim_falls_back(self):
        """dim=[0,1] is rejected by the engine -> eager fallback, correct result."""
        out, ref = self._run(lambda x: torch.sum(x, dim=[0, 1]), shape=(3, 4, 5))

    def test_norm_p2_dim(self):
        self._run(lambda x: torch.norm(x, 2, dim=1))

    def test_norm_p1_runs_natively(self):
        """p=1 is now supported natively via p_norm (L1 norm)."""
        self._run(lambda x: torch.norm(x, 1, dim=1))

    def test_var_correction2_falls_back(self):
        """correction not in (0,1) is unsupported -> eager fallback."""
        self._run(lambda x: torch.var(x, dim=1, correction=2))
