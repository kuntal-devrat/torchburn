"""
Tests for Phase 6: autograd backward pass.

Gradient correctness verified against PyTorch's autograd as reference.
"""
import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torchburn.autograd as ta
from torchburn.autograd import (
    Tensor, enable, disable, reset, tape_len,
    linear, relu, sigmoid, tanh_act, gelu, softmax,
    layer_norm, dropout, mse_loss, nll_loss, cross_entropy, sum_op,
)


@pytest.fixture(autouse=True)
def setup_autograd():
    """Enable autograd and reset tape before each test."""
    reset()
    enable()
    yield
    disable()
    reset()


# ---------------------------------------------------------------------------
# 1. Elementwise backward (via __add__, __sub__, __mul__, __truediv__)
# ---------------------------------------------------------------------------

class TestElementwiseBackward:
    def test_add_grad(self):
        a = Tensor(torch.randn(4, 4), requires_grad=True)
        b = Tensor(torch.randn(4, 4), requires_grad=True)
        c = a + b
        c.backward(torch.ones_like(c.data))
        torch.testing.assert_close(a.grad, torch.ones(4, 4))
        torch.testing.assert_close(b.grad, torch.ones(4, 4))

    def test_sub_grad(self):
        a = Tensor(torch.randn(3, 3), requires_grad=True)
        b = Tensor(torch.randn(3, 3), requires_grad=True)
        c = a - b
        c.backward(torch.ones_like(c.data))
        torch.testing.assert_close(a.grad, torch.ones(3, 3))
        torch.testing.assert_close(b.grad, -torch.ones(3, 3))

    def test_mul_grad(self):
        a = Tensor(torch.tensor([2.0, 3.0]), requires_grad=True)
        b = Tensor(torch.tensor([4.0, 5.0]), requires_grad=True)
        c = a * b
        c.backward(torch.ones(2))
        torch.testing.assert_close(a.grad, torch.tensor([4.0, 5.0]))
        torch.testing.assert_close(b.grad, torch.tensor([2.0, 3.0]))

    def test_div_grad(self):
        a = Tensor(torch.tensor([6.0, 8.0]), requires_grad=True)
        b = Tensor(torch.tensor([2.0, 4.0]), requires_grad=True)
        c = a / b
        c.backward(torch.ones(2))
        torch.testing.assert_close(a.grad, torch.tensor([0.5, 0.25]))
        expected_b = -torch.tensor([6.0, 8.0]) / torch.tensor([4.0, 16.0])
        torch.testing.assert_close(b.grad, expected_b)

    def test_neg_grad(self):
        a = Tensor(torch.tensor([1.0, -2.0, 3.0]), requires_grad=True)
        b = -a
        b.backward(torch.ones(3))
        torch.testing.assert_close(a.grad, -torch.ones(3))


# ---------------------------------------------------------------------------
# 2. Activation backward
# ---------------------------------------------------------------------------

class TestActivationBackward:
    def test_relu_grad(self):
        a = Tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)
        c = relu(a)
        c.backward(torch.ones(4))
        torch.testing.assert_close(a.grad, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_sigmoid_grad(self):
        a = Tensor(torch.tensor([0.0, 1.0, -1.0]), requires_grad=True)
        c = sigmoid(a)
        c.backward(torch.ones(3))
        s = torch.sigmoid(a.data)
        expected = s * (1 - s)
        torch.testing.assert_close(a.grad, expected, rtol=1e-5, atol=1e-5)

    def test_tanh_grad(self):
        a = Tensor(torch.tensor([0.0, 1.0, -1.0]), requires_grad=True)
        c = tanh_act(a)
        c.backward(torch.ones(3))
        t = torch.tanh(a.data)
        expected = 1 - t * t
        torch.testing.assert_close(a.grad, expected, rtol=1e-5, atol=1e-5)

    def test_gelu_grad(self):
        a = Tensor(torch.randn(4), requires_grad=True)
        c = gelu(a)
        c.backward(torch.ones(4))
        ref = a.data.clone().requires_grad_(True)
        F.gelu(ref).backward(torch.ones(4))
        torch.testing.assert_close(a.grad, ref.grad, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Linear backward
# ---------------------------------------------------------------------------

class TestLinearBackward:
    def test_linear_no_bias(self):
        inp = Tensor(torch.randn(2, 4), requires_grad=True)
        w = Tensor(torch.randn(3, 4), requires_grad=True)
        out = linear(inp, w)
        grad_upstream = torch.randn(2, 3)
        out.backward(grad_upstream)

        ref_inp = inp.data.clone().requires_grad_(True)
        ref_w = w.data.clone().requires_grad_(True)
        ref_out = F.linear(ref_inp, ref_w)
        ref_out.backward(grad_upstream)

        torch.testing.assert_close(inp.grad, ref_inp.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w.grad, ref_w.grad, rtol=1e-4, atol=1e-4)

    def test_linear_with_bias(self):
        inp = Tensor(torch.randn(4, 8), requires_grad=True)
        w = Tensor(torch.randn(16, 8), requires_grad=True)
        b = Tensor(torch.randn(16), requires_grad=True)
        out = linear(inp, w, b)
        grad_upstream = torch.randn(4, 16)
        out.backward(grad_upstream)

        ref_inp = inp.data.clone().requires_grad_(True)
        ref_w = w.data.clone().requires_grad_(True)
        ref_b = b.data.clone().requires_grad_(True)
        ref_out = F.linear(ref_inp, ref_w, ref_b)
        ref_out.backward(grad_upstream)

        torch.testing.assert_close(inp.grad, ref_inp.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w.grad, ref_w.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b.grad, ref_b.grad, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Softmax backward
# ---------------------------------------------------------------------------

class TestSoftmaxBackward:
    def test_softmax_grad(self):
        a = Tensor(torch.randn(2, 5), requires_grad=True)
        c = softmax(a, dim=-1)
        grad_upstream = torch.randn(2, 5)
        c.backward(grad_upstream)

        ref = a.data.clone().requires_grad_(True)
        ref_out = torch.softmax(ref, dim=-1)
        ref_out.backward(grad_upstream)
        torch.testing.assert_close(a.grad, ref.grad, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# 5. Sum backward
# ---------------------------------------------------------------------------

class TestSumBackward:
    def test_sum_grad(self):
        a = Tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        c = sum_op(a)
        c.backward(torch.tensor(1.0))
        torch.testing.assert_close(a.grad, torch.ones(2, 2))


# ---------------------------------------------------------------------------
# 6. Loss backward
# ---------------------------------------------------------------------------

class TestLossBackward:
    def test_mse_loss_grad(self):
        inp = Tensor(torch.randn(4, 4), requires_grad=True)
        target = Tensor(torch.randn(4, 4))
        loss = mse_loss(inp, target, reduction='mean')
        loss.backward()

        ref = inp.data.clone().requires_grad_(True)
        F.mse_loss(ref, target.data, reduction='mean').backward()
        torch.testing.assert_close(inp.grad, ref.grad, rtol=1e-4, atol=1e-4)

    def test_nll_loss_grad(self):
        inp = Tensor(torch.randn(4, 10), requires_grad=True)
        target = Tensor(torch.tensor([2, 5, 7, 3]))
        loss = nll_loss(inp, target, reduction='mean')
        loss.backward()

        ref = inp.data.clone().requires_grad_(True)
        F.nll_loss(ref, target.data.long(), reduction='mean').backward()
        torch.testing.assert_close(inp.grad, ref.grad, rtol=1e-4, atol=1e-4)

    def test_cross_entropy_grad(self):
        inp = Tensor(torch.randn(4, 10), requires_grad=True)
        target = Tensor(torch.tensor([2, 5, 7, 3]))
        loss = cross_entropy(inp, target, reduction='mean')
        loss.backward()

        ref = inp.data.clone().requires_grad_(True)
        F.cross_entropy(ref, target.data.long(), reduction='mean').backward()
        torch.testing.assert_close(inp.grad, ref.grad, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. Chained operations
# ---------------------------------------------------------------------------

class TestChainedOps:
    def test_add_then_relu(self):
        a = Tensor(torch.randn(3, 3), requires_grad=True)
        b = Tensor(torch.randn(3, 3), requires_grad=True)
        c = relu(a + b)
        c.backward(torch.ones_like(c.data))

        ref_a = a.data.clone().requires_grad_(True)
        ref_b = b.data.clone().requires_grad_(True)
        torch.relu(ref_a + ref_b).backward(torch.ones(3, 3))

        torch.testing.assert_close(a.grad, ref_a.grad)
        torch.testing.assert_close(b.grad, ref_b.grad)

    def test_linear_relu_chain(self):
        inp = Tensor(torch.randn(2, 8), requires_grad=True)
        w = Tensor(torch.randn(4, 8), requires_grad=True)
        out = relu(linear(inp, w))
        out.backward(torch.ones_like(out.data))

        ref_inp = inp.data.clone().requires_grad_(True)
        ref_w = w.data.clone().requires_grad_(True)
        torch.relu(F.linear(ref_inp, ref_w)).backward(torch.ones(2, 4))

        torch.testing.assert_close(inp.grad, ref_inp.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w.grad, ref_w.grad, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 8. Tape management
# ---------------------------------------------------------------------------

class TestTapeManagement:
    def test_tape_len(self):
        a = Tensor(torch.randn(2, 2), requires_grad=True)
        b = Tensor(torch.randn(2, 2), requires_grad=True)
        _ = a + b
        _ = a * b
        assert tape_len() == 2

    def test_reset_clears(self):
        a = Tensor(torch.randn(2, 2), requires_grad=True)
        _ = a + a
        assert tape_len() > 0
        reset()
        assert tape_len() == 0
        enable()  # re-enable for fixture teardown

    def test_no_grad_tensors(self):
        a = Tensor(torch.ones(4), requires_grad=False)
        b = Tensor(torch.ones(4), requires_grad=True)
        c = a + b
        c.backward(torch.ones(4))
        assert a.grad is None
        assert b.grad is not None

    def test_registry_tracking(self):
        a = Tensor(torch.randn(2, 2), requires_grad=True)
        b = Tensor(torch.randn(2, 2), requires_grad=False)
        assert a._id in Tensor._registry
        assert b._id not in Tensor._registry
