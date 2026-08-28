"""Tests for Phase 10: backward routed through Rust backward_single FFI.

Every test compares the Rust backward output against PyTorch's autograd to
verify numerical correctness within tight tolerances.
"""
import pytest
import torch
import torch.nn.functional as F
from torchburn.autograd import (
    Tensor, enable, disable, reset, tape_len,
    _add, _sub, _mul, _div, _matmul,
    linear, relu, sigmoid, tanh_act, gelu, softmax,
    layer_norm, mse_loss, cross_entropy, sum_op,
)


@pytest.fixture(autouse=True)
def _setup_teardown():
    enable()
    yield
    reset()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compare(ta, ref, rtol=1e-5, atol=1e-5):
    if ta.grad is None:
        assert ref.grad is None, "torchburn grad is None but reference has grad"
        return
    assert ref.grad is not None, "reference grad is None but torchburn has grad"
    torch.testing.assert_close(ta.grad, ref.grad, rtol=rtol, atol=atol)


def _backward_and_compare(tb_tensors, ref_tensors, out_tb, out_ref, rtol=1e-4, atol=1e-4):
    """Run backward on both paths and compare leaf gradients."""
    grad_out = torch.ones_like(out_tb.data)
    out_tb.backward(grad_output=grad_out)
    out_ref.backward(grad_out)
    for tb, ref in zip(tb_tensors, ref_tensors):
        _compare(tb, ref, rtol=rtol, atol=atol)


def _backward_scalar(tb_tensors, ref_tensors, out_tb, out_ref, rtol=1e-4, atol=1e-4):
    """For non-scalar outputs: pass ones_like grad to both."""
    grad_out = torch.ones_like(out_tb.data)
    out_tb.backward(grad_output=grad_out)
    out_ref.backward(grad_out)
    for tb, ref in zip(tb_tensors, ref_tensors):
        _compare(tb, ref, rtol=rtol, atol=atol)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Binary elementwise
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryBackward:
    def test_add(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        b = Tensor(torch.randn(4, 4), requires_grad=True)
        out = a + b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref + b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)

    def test_sub(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        b = Tensor(torch.randn(4, 4), requires_grad=True)
        out = a - b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref - b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)

    def test_mul(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        b = Tensor(torch.randn(4, 4), requires_grad=True)
        out = a * b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref * b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)

    def test_div(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        b = Tensor(torch.rand(4, 4) + 0.1, requires_grad=True)
        out = a / b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref / b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)

    def test_add_broadcast(self):
        reset(); enable()
        a = Tensor(torch.randn(3, 4), requires_grad=True)
        b = Tensor(torch.randn(4), requires_grad=True)
        out = a + b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref + b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)

    def test_mul_broadcast(self):
        reset(); enable()
        a = Tensor(torch.randn(3, 4), requires_grad=True)
        b = Tensor(torch.randn(4), requires_grad=True)
        out = a * b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref * b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Linear algebra
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearAlgebraBackward:
    def test_matmul_2d(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 8), requires_grad=True)
        b = Tensor(torch.randn(8, 3), requires_grad=True)
        out = a @ b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref @ b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)

    def test_matmul_square(self):
        reset(); enable()
        a = Tensor(torch.randn(5, 5), requires_grad=True)
        b = Tensor(torch.randn(5, 5), requires_grad=True)
        out = a @ b
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref @ b_ref
        _backward_and_compare([a, b], [a_ref, b_ref], out, out_ref)

    def test_linear_no_bias(self):
        reset(); enable()
        x = Tensor(torch.randn(4, 8), requires_grad=True)
        w = Tensor(torch.randn(3, 8), requires_grad=True)
        out = linear(x, w)
        x_ref = x.data.clone().requires_grad_(True)
        w_ref = w.data.clone().requires_grad_(True)
        out_ref = F.linear(x_ref, w_ref)
        _backward_and_compare([x, w], [x_ref, w_ref], out, out_ref)

    def test_linear_with_bias(self):
        reset(); enable()
        x = Tensor(torch.randn(4, 8), requires_grad=True)
        w = Tensor(torch.randn(3, 8), requires_grad=True)
        b = Tensor(torch.randn(3), requires_grad=True)
        out = linear(x, w, b)
        x_ref = x.data.clone().requires_grad_(True)
        w_ref = w.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = F.linear(x_ref, w_ref, b_ref)
        _backward_and_compare([x, w, b], [x_ref, w_ref, b_ref], out, out_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Activations
# ─────────────────────────────────────────────────────────────────────────────

class TestActivationBackward:
    def test_relu(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4) - 0.5, requires_grad=True)
        out = relu(a)
        a_ref = a.data.clone().requires_grad_(True)
        out_ref = torch.relu(a_ref)
        _backward_and_compare([a], [a_ref], out, out_ref)

    def test_sigmoid(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        out = sigmoid(a)
        a_ref = a.data.clone().requires_grad_(True)
        out_ref = torch.sigmoid(a_ref)
        _backward_and_compare([a], [a_ref], out, out_ref)

    def test_tanh(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        out = tanh_act(a)
        a_ref = a.data.clone().requires_grad_(True)
        out_ref = torch.tanh(a_ref)
        _backward_and_compare([a], [a_ref], out, out_ref)

    def test_gelu(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        out = gelu(a)
        a_ref = a.data.clone().requires_grad_(True)
        out_ref = F.gelu(a_ref)
        _backward_and_compare([a], [a_ref], out, out_ref, rtol=1e-3, atol=1e-3)

    def test_softmax(self):
        reset(); enable()
        a = Tensor(torch.randn(4, 8), requires_grad=True)
        out = softmax(a, dim=-1)
        a_ref = a.data.clone().requires_grad_(True)
        out_ref = torch.softmax(a_ref, dim=-1)
        _backward_and_compare([a], [a_ref], out, out_ref)

    def test_relu_negative_inputs(self):
        reset(); enable()
        x = Tensor(torch.tensor([-2.0, -1.0, 0.5, 1.0]), requires_grad=True)
        y = relu(x)
        y.backward(torch.ones_like(y.data))
        ref = torch.tensor([-2.0, -1.0, 0.5, 1.0]).requires_grad_(True)
        F.relu(ref).sum().backward()
        _compare(x, ref)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormBackward:
    def test_layer_norm(self):
        reset(); enable()
        a = Tensor(torch.randn(2, 8), requires_grad=True)
        w = Tensor(torch.randn(8), requires_grad=True)
        b = Tensor(torch.randn(8), requires_grad=True)
        out = layer_norm(a, w, b)
        a_ref = a.data.clone().requires_grad_(True)
        w_ref = w.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = F.layer_norm(a_ref, w_ref.shape, w_ref, b_ref)
        _backward_and_compare([a, w, b], [a_ref, w_ref, b_ref], out, out_ref, rtol=1e-3, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Reductions
# ─────────────────────────────────────────────────────────────────────────────

class TestReductionBackward:
    def test_sum(self):
        reset(); enable()
        x = Tensor(torch.randn(3, 4), requires_grad=True)
        y = x.sum()
        y.backward()
        ref = torch.randn(3, 4).requires_grad_(True)
        ref.sum().backward()
        _compare(x, ref)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Losses
# ─────────────────────────────────────────────────────────────────────────────

class TestLossBackward:
    def test_mse_loss_mean(self):
        reset(); enable()
        x = Tensor(torch.randn(4, 4), requires_grad=True)
        t = Tensor(torch.randn(4, 4), requires_grad=False)
        out = mse_loss(x, t, reduction='mean')
        x_ref = x.data.clone().requires_grad_(True)
        out_ref = F.mse_loss(x_ref, t.data, reduction='mean')
        out.backward()
        out_ref.backward()
        _compare(x, x_ref, rtol=1e-4, atol=1e-4)

    def test_mse_loss_sum(self):
        reset(); enable()
        x = Tensor(torch.randn(4, 4), requires_grad=True)
        t = Tensor(torch.randn(4, 4), requires_grad=False)
        out = mse_loss(x, t, reduction='sum')
        x_ref = x.data.clone().requires_grad_(True)
        out_ref = F.mse_loss(x_ref, t.data, reduction='sum')
        out.backward()
        out_ref.backward()
        _compare(x, x_ref, rtol=1e-4, atol=1e-4)

    def test_cross_entropy(self):
        reset(); enable()
        x = Tensor(torch.randn(4, 10), requires_grad=True)
        target = Tensor(torch.randint(0, 10, (4,)), requires_grad=False)
        out = cross_entropy(x, target)
        x_ref = x.data.clone().requires_grad_(True)
        out_ref = F.cross_entropy(x_ref, target.data)
        out.backward()
        out_ref.backward()
        _compare(x, x_ref, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Chained ops — verify gradients flow through multiple Rust backward calls
# ─────────────────────────────────────────────────────────────────────────────

class TestChainedBackward:
    def test_add_mul_chain(self):
        """a * b + a -> grad_a = b + 1, grad_b = a"""
        reset(); enable()
        a = Tensor(torch.randn(4), requires_grad=True)
        b = Tensor(torch.randn(4), requires_grad=True)
        out = a * b + a
        a_ref = a.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        out_ref = a_ref * b_ref + a_ref
        _backward_scalar([a, b], [a_ref, b_ref], out, out_ref)

    def test_linear_relu_chain(self):
        reset(); enable()
        x = Tensor(torch.randn(2, 8), requires_grad=True)
        w = Tensor(torch.randn(4, 8), requires_grad=True)
        b = Tensor(torch.randn(4), requires_grad=True)
        y = linear(x, w, b)
        z = relu(y)
        loss = z.sum()
        loss.backward()

        x_ref = x.data.clone().requires_grad_(True)
        w_ref = w.data.clone().requires_grad_(True)
        b_ref = b.data.clone().requires_grad_(True)
        y_ref = F.linear(x_ref, w_ref, b_ref)
        z_ref = F.relu(y_ref)
        z_ref.sum().backward()

        _compare(x, x_ref, rtol=1e-4, atol=1e-4)
        _compare(w, w_ref, rtol=1e-4, atol=1e-4)
        _compare(b, b_ref, rtol=1e-4, atol=1e-4)

    def test_matmul_matmul_chain(self):
        reset(); enable()
        A = Tensor(torch.randn(3, 4), requires_grad=True)
        B = Tensor(torch.randn(4, 5), requires_grad=True)
        C = Tensor(torch.randn(5, 2), requires_grad=True)
        out = (A @ B) @ C
        grad_out = torch.ones_like(out.data)
        out.backward(grad_output=grad_out)

        A_ref = A.data.clone().requires_grad_(True)
        B_ref = B.data.clone().requires_grad_(True)
        C_ref = C.data.clone().requires_grad_(True)
        out_ref = (A_ref @ B_ref) @ C_ref
        out_ref.backward(grad_out)

        _compare(A, A_ref)
        _compare(B, B_ref)
        _compare(C, C_ref)

    def test_sigmoid_matmul_chain(self):
        reset(); enable()
        x = Tensor(torch.randn(4, 4), requires_grad=True)
        w = Tensor(torch.randn(4, 4), requires_grad=True)
        out = sigmoid(x) @ w
        grad_out = torch.ones_like(out.data)
        out.backward(grad_output=grad_out)

        x_ref = x.data.clone().requires_grad_(True)
        w_ref = w.data.clone().requires_grad_(True)
        out_ref = torch.sigmoid(x_ref) @ w_ref
        out_ref.backward(grad_out)

        _compare(x, x_ref, rtol=1e-4, atol=1e-4)
        _compare(w, w_ref)

    def test_layer_norm_matmul(self):
        reset(); enable()
        x = Tensor(torch.randn(2, 8), requires_grad=True)
        w_ln = Tensor(torch.randn(8), requires_grad=True)
        b_ln = Tensor(torch.randn(8), requires_grad=True)
        w = Tensor(torch.randn(8, 4), requires_grad=True)
        y = layer_norm(x, w_ln, b_ln)
        out = y @ w
        grad_out = torch.ones_like(out.data)
        out.backward(grad_output=grad_out)

        x_ref = x.data.clone().requires_grad_(True)
        w_ln_ref = w_ln.data.clone().requires_grad_(True)
        b_ln_ref = b_ln.data.clone().requires_grad_(True)
        w_ref = w.data.clone().requires_grad_(True)
        y_ref = F.layer_norm(x_ref, w_ln_ref.shape, w_ln_ref, b_ln_ref)
        out_ref = y_ref @ w_ref
        out_ref.backward(grad_out)

        _compare(x, x_ref, rtol=1e-3, atol=1e-3)
        _compare(w_ln, w_ln_ref, rtol=1e-3, atol=1e-3)
        _compare(b_ln, b_ln_ref, rtol=1e-3, atol=1e-3)
        _compare(w, w_ref)

    def test_softmax_cross_entropy(self):
        reset(); enable()
        logits = Tensor(torch.randn(4, 10), requires_grad=True)
        target = Tensor(torch.randint(0, 10, (4,)), requires_grad=False)
        loss = cross_entropy(logits, target)
        loss.backward()

        logits_ref = logits.data.clone().requires_grad_(True)
        loss_ref = F.cross_entropy(logits_ref, target.data)
        loss_ref.backward()

        _compare(logits, logits_ref, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_scalar_loss(self):
        reset(); enable()
        x = Tensor(torch.randn(4), requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
        assert x.grad is not None
        torch.testing.assert_close(x.grad, 2 * x.data, rtol=1e-5, atol=1e-5)

    def test_no_grad_input(self):
        reset(); enable()
        a = Tensor(torch.randn(4), requires_grad=True)
        b = Tensor(torch.randn(4), requires_grad=False)
        out = a + b
        out.backward()
        assert a.grad is not None

    def test_tape_cleared_after_backward(self):
        reset(); enable()
        x = Tensor(torch.randn(4), requires_grad=True)
        y = x * 2
        y.backward()
        assert tape_len() == 0

    def test_reuse_after_reset(self):
        reset(); enable()
        x = Tensor(torch.randn(4), requires_grad=True)
        y = x * 2
        y.backward()
        reset()
        enable()
        x2 = Tensor(torch.randn(4), requires_grad=True)
        y2 = x2 * 3
        y2.backward()
        assert x2.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Training scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingScenario:
    def test_linear_regression(self):
        reset(); enable()
        torch.manual_seed(42)
        x_data = torch.randn(32, 1)
        y_data = 3.0 * x_data + 1.0

        w = Tensor(torch.randn(1, 1), requires_grad=True)
        b = Tensor(torch.zeros(1), requires_grad=True)

        for epoch in range(200):
            pred = Tensor(x_data) @ w + b
            target_t = Tensor(y_data, requires_grad=False)
            loss = mse_loss(pred, target_t, reduction='mean')
            loss.backward()
            w.data -= 0.01 * w.grad
            b.data -= 0.01 * b.grad
            w.grad = None
            b.grad = None

        assert abs(w.data.item() - 3.0) < 0.5, f"w={w.data.item()}"
        assert abs(b.data.item() - 1.0) < 0.5, f"b={b.data.item()}"

    def test_two_layer_mlp(self):
        reset(); enable()
        torch.manual_seed(0)
        N, D, H, C = 64, 16, 32, 3

        w1 = Tensor(torch.randn(H, D), requires_grad=True)
        b1 = Tensor(torch.zeros(H), requires_grad=True)
        w2 = Tensor(torch.randn(C, H), requires_grad=True)
        b2 = Tensor(torch.zeros(C), requires_grad=True)

        x = Tensor(torch.randn(N, D))
        y = Tensor(torch.randint(0, C, (N,)), requires_grad=False)

        initial_w2 = w2.data.clone()

        for epoch in range(200):
            h = linear(x, w1, b1)
            h = relu(h)
            logits = linear(h, w2, b2)
            loss = cross_entropy(logits, y)
            loss.backward()
            w1.data -= 0.01 * w1.grad
            b1.data -= 0.01 * b1.grad
            w2.data -= 0.01 * w2.grad
            b2.data -= 0.01 * b2.grad
            w1.grad = None
            b1.grad = None
            w2.grad = None
            b2.grad = None

        assert not torch.allclose(w2.data, initial_w2)
