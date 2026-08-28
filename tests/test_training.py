"""
End-to-end training tests: verify that torchburn autograd can train a
simple model and the loss actually decreases.
"""
import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torchburn.autograd as ta


@pytest.fixture(autouse=True)
def setup():
    ta.reset()
    ta.enable()
    yield
    ta.disable()
    ta.reset()


class TestTrainingLoop:
    def test_linear_regression(self):
        """Train a single linear layer to fit y = 3x + 1."""
        torch.manual_seed(42)

        x_data = torch.randn(32, 1)
        y_data = 3.0 * x_data + 1.0

        w = ta.Tensor(torch.randn(1, 1) * 0.01, requires_grad=True)
        b = ta.Tensor(torch.zeros(1), requires_grad=True)

        losses = []
        lr = 0.01

        for epoch in range(50):
            ta.reset()
            ta.enable()
            # Re-register params after reset clears the registry
            ta.Tensor._registry[w._id] = w
            ta.Tensor._registry[b._id] = b

            x = ta.Tensor(x_data)
            pred = ta.linear(x, w, b)
            loss = ta.mse_loss(pred, ta.Tensor(y_data))
            loss.backward()

            losses.append(loss.data.item())

            w.data -= lr * w.grad
            b.data -= lr * b.grad
            w.grad = None
            b.grad = None

        assert losses[-1] < losses[0] * 0.1, f"Loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_two_layer_mlp(self):
        """Train a 2-layer MLP on a toy classification task."""
        torch.manual_seed(42)

        n = 100
        x_pos = torch.randn(n, 2) + torch.tensor([1.0, 1.0])
        x_neg = torch.randn(n, 2) - torch.tensor([1.0, 1.0])
        x_data = torch.cat([x_pos, x_neg], dim=0)
        y_data = torch.cat([torch.ones(n), torch.zeros(n)])

        w1 = ta.Tensor(torch.randn(16, 2) * 0.1, requires_grad=True)
        b1 = ta.Tensor(torch.zeros(16), requires_grad=True)
        w2 = ta.Tensor(torch.randn(1, 16) * 0.1, requires_grad=True)
        b2 = ta.Tensor(torch.zeros(1), requires_grad=True)

        losses = []
        lr = 0.05

        for epoch in range(100):
            ta.reset()
            ta.enable()
            for param in [w1, b1, w2, b2]:
                ta.Tensor._registry[param._id] = param

            x = ta.Tensor(x_data)
            h = ta.relu(ta.linear(x, w1, b1))
            out = ta.linear(h, w2, b2)
            loss = ta.mse_loss(out, ta.Tensor(y_data.unsqueeze(1)))
            loss.backward()

            losses.append(loss.data.item())

            for param in [w1, b1, w2, b2]:
                param.data -= lr * param.grad
                param.grad = None

        assert losses[-1] < losses[0] * 0.5, f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_attention_block_backward(self):
        """Test gradients flow through matmul + softmax + matmul."""
        torch.manual_seed(42)

        # (batch=2, seq=4, dim=8)
        x = ta.Tensor(torch.randn(2, 4, 8))
        w1 = ta.Tensor(torch.randn(8, 8) * 0.1, requires_grad=True)
        w2 = ta.Tensor(torch.randn(8, 8) * 0.1, requires_grad=True)

        h = ta.relu(x @ w1)      # matmul + relu
        scores = h @ w2           # matmul
        attn = ta.softmax(scores, dim=-1)
        out = attn @ ta.Tensor(torch.randn(2, 8, 8))  # matmul with constant

        loss = ta.sum_op(out)
        loss.backward()

        assert w1.grad is not None
        assert w2.grad is not None
        assert torch.isfinite(w1.grad).all()
        assert torch.isfinite(w2.grad).all()


class TestGradientAccumulation:
    def test_multiple_backward(self):
        """Multiple uses of the same tensor should accumulate gradients."""
        a = ta.Tensor(torch.ones(4), requires_grad=True)
        b = a * a  # b = a^2
        c = b + a  # c = a^2 + a
        c.backward(torch.ones(4))
        expected = 2 * a.data + 1
        torch.testing.assert_close(a.grad, expected)


class TestNoGrad:
    def test_no_grad_tensors(self):
        """Tensors with requires_grad=False should not accumulate gradients."""
        a = ta.Tensor(torch.ones(4), requires_grad=False)
        b = ta.Tensor(torch.ones(4), requires_grad=True)
        c = a + b
        c.backward(torch.ones(4))
        assert a.grad is None
        assert b.grad is not None
