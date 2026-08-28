"""Tests for conv2d → batch_norm → relu fusion (REQ-004).

Verifies that the fused path produces identical results to running conv2d,
batch_norm, and relu as separate operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import pytest

import torchburn  # registers the backend


class ConvBnReluModel(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block (inference mode)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvBnReluBiasModel(nn.Module):
    """Conv2d (with bias) → BatchNorm2d → ReLU block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class DoubleConvBnRelu(nn.Module):
    """Two consecutive Conv2d → BatchNorm2d → ReLU blocks."""

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ConvBnNoRelu(nn.Module):
    """Conv2d → BatchNorm2d (no relu) — should NOT be fused."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class TestConvBnReluFusion:
    """Test conv2d → batch_norm → relu fusion via torch.compile path."""

    def test_conv_bn_relu_basic(self):
        """Basic conv+bn+relu produces correct output."""
        torch.manual_seed(42)
        model = ConvBnReluModel(3, 16).eval()
        x = torch.randn(2, 3, 8, 8)

        with torch.no_grad():
            eager_out = model(x)
            compiled = torch.compile(model, backend="torchburn")
            compiled_out = compiled(x)

        assert torch.allclose(eager_out, compiled_out, atol=1e-4), \
            f"Max diff: {(eager_out - compiled_out).abs().max().item()}"

    def test_conv_bn_relu_shapes(self):
        """Various input/channel configurations produce correct shapes."""
        torch.manual_seed(42)
        configs = [
            (1, 8, 4, 4, 16, 3, 1, 1),   # in=1, out=8, H=W=4
            (3, 16, 8, 8, 32, 3, 1, 1),   # in=3, out=16, H=W=8
            (16, 32, 8, 8, 64, 3, 1, 1),  # in=16, out=32, H=W=8
            (3, 8, 16, 16, 16, 3, 2, 1),  # stride=2
            (3, 8, 16, 16, 8, 5, 1, 2),   # kernel=5, padding=2
        ]
        for in_ch, out_ch, h, w, _, k, s, p in configs:
            model = ConvBnReluModel(in_ch, out_ch, kernel_size=k, stride=s, padding=p).eval()
            x = torch.randn(2, in_ch, h, w)
            with torch.no_grad():
                eager_out = model(x)
                compiled = torch.compile(model, backend="torchburn")
                compiled_out = compiled(x)
            expected_h = (h + 2 * p - k) // s + 1
            expected_w = (w + 2 * p - k) // s + 1
            assert compiled_out.shape == (2, out_ch, expected_h, expected_w), \
                f"Shape mismatch: {compiled_out.shape} vs expected (2, {out_ch}, {expected_h}, {expected_w})"
            assert torch.allclose(eager_out, compiled_out, atol=1e-4), \
                f"Value mismatch for config ({in_ch},{out_ch},{h},{w},{k},{s},{p}): " \
                f"max diff = {(eager_out - compiled_out).abs().max().item()}"

    def test_conv_bn_relu_with_bias(self):
        """Conv2d with bias → BN → ReLU works correctly."""
        torch.manual_seed(42)
        model = ConvBnReluBiasModel(3, 16).eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            eager_out = model(x)
            compiled = torch.compile(model, backend="torchburn")
            compiled_out = compiled(x)
        assert torch.allclose(eager_out, compiled_out, atol=1e-4)

    def test_double_conv_bn_relu(self):
        """Two consecutive Conv+BN+ReLU blocks fuse correctly."""
        torch.manual_seed(42)
        model = DoubleConvBnRelu(3, 16, 32).eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            eager_out = model(x)
            compiled = torch.compile(model, backend="torchburn")
            compiled_out = compiled(x)
        assert torch.allclose(eager_out, compiled_out, atol=1e-4), \
            f"Max diff: {(eager_out - compiled_out).abs().max().item()}"

    def test_conv_bn_no_relu_not_fused(self):
        """Conv+BN without ReLU should NOT trigger conv_bn_relu fusion."""
        torch.manual_seed(42)
        model = ConvBnNoRelu(3, 16).eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            eager_out = model(x)
            compiled = torch.compile(model, backend="torchburn")
            compiled_out = compiled(x)
        assert torch.allclose(eager_out, compiled_out, atol=1e-4)

    def test_conv_bn_relu_batch_size_1(self):
        """Batch size=1 works (BN uses running stats)."""
        torch.manual_seed(42)
        model = ConvBnReluModel(3, 16).eval()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            eager_out = model(x)
            compiled = torch.compile(model, backend="torchburn")
            compiled_out = compiled(x)
        assert torch.allclose(eager_out, compiled_out, atol=1e-4)

    def test_conv_bn_relu_large_spatial(self):
        """Large spatial dimensions (32x32) work correctly."""
        torch.manual_seed(42)
        model = ConvBnReluModel(3, 16).eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            eager_out = model(x)
            compiled = torch.compile(model, backend="torchburn")
            compiled_out = compiled(x)
        assert torch.allclose(eager_out, compiled_out, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
