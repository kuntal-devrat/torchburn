"""Comprehensive regression test suite for production readiness audit fixes.

Validates:
1. Loss reduction convention alignment (0=none, 1=mean, 2=sum).
2. Autograd chain rule compliance with non-unit upstream gradients.
3. 3D transformer attention mask indexing (B < H).
4. Non-contiguous sliced/transposed views in reductions and softmax.
5. Zero-dimension conv2d and pooling edge cases.
6. Parser scatter vs scatter_max resolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

import torchburn
from torchburn import ops as tb_ops
from torchburn import _torchburn as tb_native
from torchburn import autograd as tb_autograd
from torchburn._parser import _FUNCTION_TO_OP


# --------------------------------------------------------------------------- #
# 1. Loss Reduction Alignment Tests
# --------------------------------------------------------------------------- #

class TestLossReductions:
    def test_mse_loss_all_reductions(self):
        x = torch.randn(4, 5, dtype=torch.float32)
        y = torch.randn(4, 5, dtype=torch.float32)

        for red in ("none", "mean", "sum"):
            expected = F.mse_loss(x, y, reduction=red)
            actual = tb_ops.mse_loss(x, y, reduction=red)
            assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch for reduction={red}"

        # Test integer codes directly
        assert torch.allclose(tb_ops.mse_loss(x, y, reduction=0), F.mse_loss(x, y, reduction="none"), atol=1e-5)
        assert torch.allclose(tb_ops.mse_loss(x, y, reduction=1), F.mse_loss(x, y, reduction="mean"), atol=1e-5)
        assert torch.allclose(tb_ops.mse_loss(x, y, reduction=2), F.mse_loss(x, y, reduction="sum"), atol=1e-5)

    def test_nll_loss_all_reductions(self):
        logits = torch.randn(6, 4, dtype=torch.float32)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = torch.tensor([0, 2, 1, 3, 0, 1], dtype=torch.int64)

        for red in ("none", "mean", "sum"):
            expected = F.nll_loss(log_probs, targets, reduction=red)
            actual = tb_ops.nll_loss(log_probs, targets, reduction=red)
            assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch for nll_loss reduction={red}"

    def test_smooth_l1_loss_all_reductions(self):
        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)

        for red in ("none", "mean", "sum"):
            expected = F.smooth_l1_loss(x, y, reduction=red)
            actual = tb_ops.smooth_l1_loss(x, y, reduction=red)
            assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch for smooth_l1 reduction={red}"

    def test_binary_cross_entropy_all_reductions(self):
        x = torch.rand(4, 4, dtype=torch.float32).clamp(0.01, 0.99)
        y = torch.randint(0, 2, (4, 4), dtype=torch.float32)

        for red in ("none", "mean", "sum"):
            expected = F.binary_cross_entropy(x, y, reduction=red)
            actual = tb_ops.binary_cross_entropy(x, y, reduction=red)
            assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch for bce reduction={red}"


# --------------------------------------------------------------------------- #
# 2. Autograd Chain Rule Compliance
# --------------------------------------------------------------------------- #

class TestAutogradChainRule:
    def test_mse_loss_backward_scalar_upstream(self):
        x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
        y = torch.randn(3, 4, dtype=torch.float32)

        # PyTorch reference with non-unit upstream gradient (e.g. 3.5)
        loss_ref = F.mse_loss(x, y, reduction="mean")
        loss_ref.backward(gradient=torch.tensor(3.5))
        ref_grad = x.grad.clone()

        # TorchBurn backward
        grad_out = tb_autograd.backward_single(
            target="mse_loss",
            upstream=torch.tensor(3.5),
            saved_inputs=[x.detach(), y],
            kwargs={"reduction": 1},
        )
        assert torch.allclose(grad_out[0], ref_grad, atol=1e-5)

    def test_mse_loss_backward_none_reduction_upstream(self):
        x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
        y = torch.randn(3, 4, dtype=torch.float32)
        upstream = torch.randn(3, 4, dtype=torch.float32)

        # PyTorch reference with elementwise upstream tensor
        loss_ref = F.mse_loss(x, y, reduction="none")
        loss_ref.backward(gradient=upstream)
        ref_grad = x.grad.clone()

        # TorchBurn backward with reduction=0 (none)
        grad_out = tb_autograd.backward_single(
            target="mse_loss",
            upstream=upstream,
            saved_inputs=[x.detach(), y],
            kwargs={"reduction": 0},
        )
        assert torch.allclose(grad_out[0], ref_grad, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. 3D Attention Mask Tests
# --------------------------------------------------------------------------- #

class TestAttention3DMask:
    def test_3d_mask_head_specific(self):
        """Transformer pattern: B=2, H=8, T=16 with 3D mask [H, T, T]."""
        B, H, T, D = 2, 8, 16, 32
        q = torch.randn(B, H, T, D, dtype=torch.float32)
        k = torch.randn(B, H, T, D, dtype=torch.float32)
        v = torch.randn(B, H, T, D, dtype=torch.float32)

        # 3D attention mask [H, T, T]
        mask = torch.randn(H, T, T, dtype=torch.float32)

        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        actual = tb_ops._execute("scaled_dot_product_attention", [q, k, v, mask], {"scale": 1.0 / (D ** 0.5)})
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_4d_mask_batch_broadcast(self):
        """Transformer pattern: B=2, H=8, T=16 with 4D mask [B, 1, T, T]."""
        B, H, T, D = 2, 8, 16, 32
        q = torch.randn(B, H, T, D, dtype=torch.float32)
        k = torch.randn(B, H, T, D, dtype=torch.float32)
        v = torch.randn(B, H, T, D, dtype=torch.float32)

        mask4 = torch.randn(B, 1, T, T, dtype=torch.float32)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask4)
        actual = tb_ops._execute("scaled_dot_product_attention", [q, k, v, mask4], {"scale": 1.0 / (D ** 0.5)})
        assert torch.allclose(actual, expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# 4. Non-Contiguous Views in Reductions and Softmax
# --------------------------------------------------------------------------- #

class TestNonContiguousViews:
    def test_sum_transposed(self):
        x = torch.randn(4, 7, dtype=torch.float32).t()  # non-contiguous
        assert not x.is_contiguous()

        assert torch.allclose(tb_ops.sum(x), x.sum(), atol=1e-5)
        assert torch.allclose(tb_ops.sum(x, dim=0), x.sum(dim=0), atol=1e-5)
        assert torch.allclose(tb_ops.sum(x, dim=1), x.sum(dim=1), atol=1e-5)

    def test_max_reduce_sliced(self):
        full = torch.randn(6, 8, dtype=torch.float32)
        x = full[::2, ::2]  # non-contiguous strided slice
        assert not x.is_contiguous()

        tb_val = tb_ops.max(x, dim=0)
        pt_val = x.max(dim=0).values
        assert torch.allclose(tb_val, pt_val, atol=1e-5)

    def test_softmax_transposed(self):
        x = torch.randn(8, 12, dtype=torch.float32).t()
        assert not x.is_contiguous()

        expected = F.softmax(x, dim=-1)
        actual = tb_ops.softmax(x, dim=-1)
        assert torch.allclose(actual, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# 5. Zero-Dimension Spatial Edge Cases
# --------------------------------------------------------------------------- #

class TestZeroDimensionSafety:
    def test_empty_batch_conv2d(self):
        conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        x = torch.randn(0, 3, 16, 16, dtype=torch.float32)
        out = tb_ops._execute("conv2d", [x, conv.weight.detach(), conv.bias.detach()], {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1})
        assert out.shape == (0, 8, 16, 16)

    def test_empty_batch_pooling(self):
        x = torch.randn(0, 4, 8, 8, dtype=torch.float32)
        out_max = tb_ops._execute("max_pool2d", [x], {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "dilation": [1, 1], "ceil_mode": False})
        assert out_max.shape == (0, 4, 4, 4)

        out_avg = tb_ops._execute("avg_pool2d", [x], {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": False, "count_include_pad": True})
        assert out_avg.shape == (0, 4, 4, 4)


# --------------------------------------------------------------------------- #
# 6. Parser Target Mappings
# --------------------------------------------------------------------------- #

class TestParserMappings:
    def test_scatter_not_overwritten_by_scatter_max(self):
        assert _FUNCTION_TO_OP["torch.scatter"] == "scatter"
        assert _FUNCTION_TO_OP["torch.scatter_max"] == "scatter_max"
        assert _FUNCTION_TO_OP["torch.scatter_min"] == "scatter_min"
