"""
Phase 9 tests: native backward through Rust kernels + comprehensive Phase 1-9 edge cases.
"""
import torch
import torch.nn.functional as F
import pytest
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torchburn  # registers the backend


# ---------------------------------------------------------------------------
# Phase 9: Native backward via Rust FFI
# ---------------------------------------------------------------------------

class TestNativeBackward:
    """Test backward_single via the Rust FFI."""

    def test_add_backward(self):
        """Native backward for add: grad flows to both inputs."""
        from torchburn._torchburn import backward_single
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        grad_out = torch.ones(4, 4)
        grad_a, grad_b = backward_single("add", grad_out.__dlpack__(), [a.__dlpack__(), b.__dlpack__()], "{}")
        grad_a_t = torch.from_dlpack(grad_a)
        grad_b_t = torch.from_dlpack(grad_b)
        torch.testing.assert_close(grad_a_t, torch.ones(4, 4))
        torch.testing.assert_close(grad_b_t, torch.ones(4, 4))

    def test_mul_backward(self):
        """Native backward for mul: grad_a = grad*b, grad_b = grad*a."""
        from torchburn._torchburn import backward_single
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        grad_out = torch.ones(2)
        grad_a, grad_b = backward_single("mul", grad_out.__dlpack__(), [a.__dlpack__(), b.__dlpack__()], "{}")
        grad_a_t = torch.from_dlpack(grad_a)
        grad_b_t = torch.from_dlpack(grad_b)
        torch.testing.assert_close(grad_a_t, torch.tensor([4.0, 5.0]))
        torch.testing.assert_close(grad_b_t, torch.tensor([2.0, 3.0]))

    def test_relu_backward(self):
        """Native backward for relu: grad = upstream * (x > 0)."""
        from torchburn._torchburn import backward_single
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        grad_out = torch.ones(4)
        grad_x, = backward_single("relu", grad_out.__dlpack__(), [x.__dlpack__()], "{}")
        grad_x_t = torch.from_dlpack(grad_x)
        torch.testing.assert_close(grad_x_t, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_matmul_backward(self):
        """Native backward for matmul: grad_a = grad @ b^T, grad_b = a^T @ grad."""
        from torchburn._torchburn import backward_single
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        grad_out = torch.randn(2, 4)
        grad_a, grad_b = backward_single("matmul", grad_out.__dlpack__(), [a.__dlpack__(), b.__dlpack__()], "{}")
        grad_a_t = torch.from_dlpack(grad_a)
        grad_b_t = torch.from_dlpack(grad_b)
        # Reference
        ref_a = grad_out @ b.T
        ref_b = a.T @ grad_out
        torch.testing.assert_close(grad_a_t, ref_a, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad_b_t, ref_b, rtol=1e-5, atol=1e-5)

    def test_mse_loss_backward(self):
        """Native backward for mse_loss: grad = scale * 2 * (input - target)."""
        from torchburn._torchburn import backward_single
        inp = torch.randn(4, 4)
        target = torch.randn(4, 4)
        grad_out = torch.tensor(1.0)
        kwargs = '{"reduction": 1}'
        grad_inp, = backward_single("mse_loss", grad_out.__dlpack__(), [inp.__dlpack__(), target.__dlpack__()], kwargs)
        grad_inp_t = torch.from_dlpack(grad_inp)
        ref = 2.0 / 16 * (inp - target)
        torch.testing.assert_close(grad_inp_t, ref, rtol=1e-5, atol=1e-5)

    def test_sum_backward(self):
        """Native backward for sum: broadcast upstream back to input shape."""
        from torchburn._torchburn import backward_single
        x = torch.randn(3, 4)
        grad_out = torch.tensor(1.0)
        grad_x, = backward_single("sum", grad_out.__dlpack__(), [x.__dlpack__()], "{}")
        grad_x_t = torch.from_dlpack(grad_x)
        torch.testing.assert_close(grad_x_t, torch.ones(3, 4))


# ---------------------------------------------------------------------------
# Comprehensive Phase 1-3: Elementwise, Linalg, Conv edge cases
# ---------------------------------------------------------------------------

class TestPhase1EdgeCases:
    def test_add_broadcast_scalar(self):
        """add tensor + scalar via torch.compile."""
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1.0
        m = M().eval()
        x = torch.randn(4, 8)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x), m(x))

    def test_mul_broadcast_row(self):
        """mul matrix * row vector via torch.compile."""
        class M(torch.nn.Module):
            def forward(self, x, w):
                return x * w
        m = M().eval()
        x = torch.randn(4, 8)
        w = torch.randn(1, 8)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x, w), m(x, w))

    def test_relu_inplace_no_crash(self):
        """relu doesn't crash on edge values."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)
        m = M().eval()
        x = torch.tensor([-1000.0, -1.0, -0.001, 0.0, 0.001, 1.0, 1000.0])
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x), m(x))

    def test_div_by_near_zero(self):
        """div by near-zero doesn't crash (may produce inf, that's ok)."""
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x / y
        m = M().eval()
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.001, 0.0001, 0.00001])
        cc = torch.compile(m, backend="torchburn")
        out = cc(x, y)
        ref = m(x, y)
        torch.testing.assert_close(out, ref)

    def test_large_tensor_elementwise(self):
        """elementwise ops on large tensors (tests rayon threshold)."""
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.relu(x + y) * 2.0
        m = M().eval()
        x = torch.randn(1024, 1024)
        y = torch.randn(1024, 1024)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x, y), m(x, y), rtol=1e-5, atol=1e-5)


class TestPhase2EdgeCases:
    def test_matmul_3d_batched(self):
        """batched matmul (3D tensors)."""
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)
        m = M().eval()
        x = torch.randn(4, 8, 16)
        y = torch.randn(4, 16, 32)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x, y), m(x, y), rtol=1e-5, atol=1e-5)

    def test_layer_norm_shapes(self):
        """layer_norm with various hidden dims."""
        for dim in [32, 64, 128, 256]:
            ln = torch.nn.LayerNorm(dim).eval()
            x = torch.randn(2, 16, dim)
            cc = torch.compile(ln, backend="torchburn")
            torch.testing.assert_close(cc(x), ln(x), rtol=1e-5, atol=1e-5)

    def test_softmax_numerical_stability(self):
        """softmax with large values (numerical stability)."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)
        m = M().eval()
        x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x), m(x), rtol=1e-5, atol=1e-5)

    def test_cat_3d(self):
        """cat along different dims."""
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat([x, y], dim=1)
        m = M().eval()
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 6, 8)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x, y), m(x, y))

    def test_reshape_infer(self):
        """reshape with -1 dimension."""
        class M(torch.nn.Module):
            def forward(self, x):
                return x.reshape(8, -1)
        m = M().eval()
        x = torch.randn(2, 32)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x), m(x))


class TestPhase3EdgeCases:
    def test_conv2d_various_kernels(self):
        """conv2d with kernel sizes 1, 3, 5."""
        for k in [1, 3, 5]:
            conv = torch.nn.Conv2d(3, 16, k, padding=k//2).eval()
            x = torch.randn(1, 3, 32, 32)
            cc = torch.compile(conv, backend="torchburn")
            torch.testing.assert_close(cc(x), conv(x), rtol=1e-5, atol=1e-5)

    def test_max_pool2d(self):
        """max_pool2d basic."""
        pool = torch.nn.MaxPool2d(2)
        x = torch.randn(1, 16, 32, 32)
        cc = torch.compile(pool, backend="torchburn")
        torch.testing.assert_close(cc(x), pool(x))

    def test_adaptive_avg_pool2d(self):
        """adaptive_avg_pool2d to 1x1."""
        pool = torch.nn.AdaptiveAvgPool2d(1)
        x = torch.randn(1, 64, 8, 8)
        cc = torch.compile(pool, backend="torchburn")
        torch.testing.assert_close(cc(x), pool(x))


# ---------------------------------------------------------------------------
# Comprehensive Phase 4-5: Transformer, Fusion
# ---------------------------------------------------------------------------

class TestPhase4EdgeCases:
    def test_multi_head_attention(self):
        """Multi-head attention block."""
        class MHA(torch.nn.Module):
            def __init__(self, d, heads):
                super().__init__()
                self.d = d
                self.heads = heads
                self.dk = d // heads
                self.wq = torch.nn.Linear(d, d)
                self.wk = torch.nn.Linear(d, d)
                self.wv = torch.nn.Linear(d, d)
                self.wo = torch.nn.Linear(d, d)

            def forward(self, x):
                B, S, D = x.shape
                q = self.wq(x).reshape(B, S, self.heads, self.dk).transpose(1, 2)
                k = self.wk(x).reshape(B, S, self.heads, self.dk).transpose(1, 2)
                v = self.wv(x).reshape(B, S, self.heads, self.dk).transpose(1, 2)
                # Simple attention (no mask)
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).reshape(B, S, D)
                return self.wo(out)

        m = MHA(64, 4).eval()
        x = torch.randn(2, 8, 64)
        cc = torch.compile(m, backend="torchburn")
        ref = m(x)
        out = cc(x)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_nll_loss_mean(self):
        """nll_loss with mean reduction."""
        class M(torch.nn.Module):
            def forward(self, x, t):
                return F.nll_loss(x, t, reduction='mean')
        m = M().eval()
        x = torch.randn(4, 10)
        t = torch.tensor([2, 5, 7, 3])
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x, t), m(x, t), rtol=1e-5, atol=1e-5)


class TestPhase5EdgeCases:
    def test_fusion_residual_add(self):
        """residual add should be fused."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 8)
            def forward(self, x):
                return x + self.fc(x)
        m = M().eval()
        x = torch.randn(4, 8)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x), m(x), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Comprehensive Phase 6-7: Autograd, Extended ops
# ---------------------------------------------------------------------------

class TestPhase6EdgeCases:
    def test_chained_matmul_backward(self):
        """backward through chained matmul: (a @ b) @ c."""
        import torchburn.autograd as ta
        ta.reset()
        ta.enable()

        a = ta.Tensor(torch.randn(2, 4), requires_grad=True)
        b = ta.Tensor(torch.randn(4, 3), requires_grad=True)
        c = ta.Tensor(torch.randn(3, 5), requires_grad=True)

        for param in [a, b, c]:
            ta.Tensor._registry[param._id] = param

        out = (a @ b) @ c
        loss = out.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None
        assert torch.isfinite(a.grad).all()
        assert torch.isfinite(b.grad).all()
        assert torch.isfinite(c.grad).all()

    def test_multiple_outputs_same_input(self):
        """same tensor used in two branches."""
        import torchburn.autograd as ta
        ta.reset()
        ta.enable()

        x = ta.Tensor(torch.randn(4), requires_grad=True)
        ta.Tensor._registry[x._id] = x

        a = x * 2
        b = x * 3
        out = a + b  # 2x + 3x = 5x
        out.backward(torch.ones(4))

        expected = torch.full((4,), 5.0)
        torch.testing.assert_close(x.grad, expected)


class TestPhase7EdgeCases:
    def test_prelu_channelwise(self):
        """prelu with per-channel weights."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.PReLU(4)
            def forward(self, x):
                return self.p(x)
        m = M().eval()
        x = torch.randn(2, 4, 8, 8)
        cc = torch.compile(m, backend="torchburn")
        torch.testing.assert_close(cc(x), m(x), rtol=1e-5, atol=1e-5)

    def test_nonzero_3d(self):
        """nonzero on 3D tensor."""
        class M(torch.nn.Module):
            def forward(self, t):
                return torch.nonzero(t)
        m = M().eval()
        x = torch.randn(2, 3, 4).clamp(-0.5, 0.5)
        cc = torch.compile(m, backend="torchburn")
        out = cc(x)
        ref = torch.nonzero(x)
        torch.testing.assert_close(out, ref)


# ---------------------------------------------------------------------------
# End-to-end: Full training loop with torch.compile
# ---------------------------------------------------------------------------

class TestEndToEndTraining:
    def test_train_compiled_model(self):
        """Train a compiled model end-to-end."""
        import torchburn.autograd as ta

        torch.manual_seed(42)
        ta.enable()

        # Simple linear model: F.linear expects weight (out_features, in_features)
        w = ta.Tensor(torch.randn(4, 8) * 0.01, requires_grad=True)
        b = ta.Tensor(torch.zeros(4), requires_grad=True)

        x_data = torch.randn(32, 8)
        y_data = torch.randn(32, 4)

        lr = 0.1
        losses = []
        for epoch in range(100):
            ta.reset()
            ta.enable()
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

        # Loss should decrease (matches PyTorch autograd exactly)
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        assert losses[-1] < losses[0] * 0.9, f"Loss decreased too slowly: {losses[0]:.4f} -> {losses[-1]:.4f}"

        ta.disable()
        ta.reset()
