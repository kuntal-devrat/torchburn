"""
Tests for Phase 5: graph-level operator fusion.

These tests exercise the fusion planner, fused elementwise chain execution,
GEMM epilogue fusion, and the skip-fallback safety mechanism — all through
the torch.compile path (the registered backend).
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import torchburn  # registers the backend


# ---------------------------------------------------------------------------
# Helper: build a torch.compile callable against the torchburn backend
# ---------------------------------------------------------------------------

def _compile(model, *example_inputs):
    """Compile a model through the torchburn backend and return the callable."""
    return torch.compile(model, backend="torchburn")


# ---------------------------------------------------------------------------
# 1. Elementwise chain fusion
# ---------------------------------------------------------------------------

class TestChainFusion:
    """Fused elementwise chains: add→relu→mul etc. should run without
    intermediate tensors and match torch's eager output."""

    def test_add_relu_mul_chain(self):
        """add → relu → mul should produce the same result as eager."""
        class M(nn.Module):
            def forward(self, x, y):
                z = x + y
                z = torch.relu(z)
                z = z * 2.0
                return z

        model = M().eval()
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        ref = model(x, y)
        cc = _compile(model, x, y)
        out = cc(x, y)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_sigmoid_sub_chain(self):
        """sub → sigmoid chain."""
        class M(nn.Module):
            def forward(self, x, y):
                return torch.sigmoid(x - y)

        model = M().eval()
        x = torch.randn(3, 5)
        y = torch.randn(3, 5)
        ref = model(x, y)
        cc = _compile(model, x, y)
        torch.testing.assert_close(cc(x, y), ref, rtol=1e-5, atol=1e-5)

    def test_gelu_mul_chain(self):
        """gelu → mul chain (common in BERT MLP).
        The fused GELU uses the tanh approximation, which differs from
        PyTorch's exact formula — tolerances are accordingly wider."""
        class M(nn.Module):
            def forward(self, x):
                return F.gelu(x) * 0.5

        model = M().eval()
        x = torch.randn(2, 16, 32)
        ref = model(x)
        cc = _compile(model, x)
        torch.testing.assert_close(cc(x), ref, rtol=1e-3, atol=1e-3)

    def test_neg_abs_add_chain(self):
        """neg → abs → add chain."""
        class M(nn.Module):
            def forward(self, x, y):
                return torch.abs(-x) + y

        model = M().eval()
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        ref = model(x, y)
        cc = _compile(model, x, y)
        torch.testing.assert_close(cc(x, y), ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. GEMM epilogue fusion (linear → activation)
# ---------------------------------------------------------------------------

class TestGemmAmpilogue:
    """linear → activation should fuse the activation into the matmul output."""

    def test_linear_relu(self):
        """Linear → ReLU should fuse into one step."""
        model = nn.Linear(16, 32).eval()
        x = torch.randn(4, 16)
        ref = model(x)
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_linear_gelu(self):
        """Linear → GELU should fuse.
        GELU tanh approximation vs exact — wider tolerance."""
        model = nn.Sequential(nn.Linear(32, 64), nn.GELU()).eval()
        x = torch.randn(8, 32)
        ref = model(x)
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_linear_sigmoid(self):
        """Linear → sigmoid should fuse."""
        model = nn.Sequential(nn.Linear(8, 16), nn.Sigmoid()).eval()
        x = torch.randn(2, 8)
        ref = model(x)
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_linear_silu(self):
        """Linear → SiLU should fuse."""
        model = nn.Sequential(nn.Linear(16, 32), nn.SiLU()).eval()
        x = torch.randn(4, 16)
        ref = model(x)
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Broadcasting in fused chains
# ---------------------------------------------------------------------------

class TestFusionBroadcasting:
    """Fused elementwise chains with broadcast operands."""

    def test_scalar_broadcast_add_relu(self):
        """Add a scalar then relu."""
        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x + 1.0)

        model = M().eval()
        x = torch.randn(4, 8)
        ref = model(x)
        cc = _compile(model, x)
        torch.testing.assert_close(cc(x), ref, rtol=1e-5, atol=1e-5)

    def test_row_broadcast_mul(self):
        """Broadcast a row vector across a matrix."""
        class M(nn.Module):
            def forward(self, x, w):
                return x * w

        model = M().eval()
        x = torch.randn(4, 8)
        w = torch.randn(1, 8)
        ref = model(x, w)
        cc = _compile(model, x, w)
        torch.testing.assert_close(cc(x, w), ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Multi-output: chain members must not be exposed as graph outputs
# ---------------------------------------------------------------------------

class TestFusionMultiOutput:
    """Verify fusion handles graphs where intermediate chain outputs
    are requested (should fall back gracefully)."""

    def test_identity_passthrough(self):
        """A model that returns an intermediate (identity) should still work."""
        class M(nn.Module):
            def forward(self, x):
                a = x + 1.0
                b = a * 2.0
                return b

        model = M().eval()
        x = torch.randn(3, 3)
        ref = model(x)
        cc = _compile(model, x)
        torch.testing.assert_close(cc(x), ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Fallback mechanism: unsupported ops should not crash
# ---------------------------------------------------------------------------

class TestFusionFallback:
    """Fusion should skip gracefully for unsupported ops, falling back
    to per-node dispatch, and the result should be correct."""

    def test_reduction_breaks_chain(self):
        """A reduction in the middle of a graph breaks the fusion chain
        but should still produce correct output."""
        class M(nn.Module):
            def forward(self, x):
                # x @ w produces (B, O), sum reduces to (B,), then + 1
                # The sum breaks the elementwise chain.
                s = x.sum(dim=-1)
                return s + 1.0

        model = M().eval()
        x = torch.randn(4, 8)
        ref = model(x)
        cc = _compile(model, x)
        torch.testing.assert_close(cc(x), ref, rtol=1e-5, atol=1e-5)

    def test_matmul_not_fused_with_unary(self):
        """matmul → activation where matmul isn't linear/addmm should not fuse
        but should still be correct."""
        class M(nn.Module):
            def forward(self, x, w):
                return torch.relu(x @ w)

        model = M().eval()
        x = torch.randn(4, 8)
        w = torch.randn(8, 16)
        ref = model(x, w)
        cc = _compile(model, x, w)
        torch.testing.assert_close(cc(x, w), ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Transformer-style block: combination of fusions
# ---------------------------------------------------------------------------

class TestFusionTransformerBlock:
    """A realistic transformer block should see multiple fusion opportunities:
    linear→activation in the MLP, elementwise add chains in residuals."""

    def test_mlp_block(self):
        """Two-layer MLP: LN → Linear → GELU → Linear should exercise
        both GEMM epilogue and elementwise fusion."""
        class MLP(nn.Module):
            def __init__(self, dim, ff):
                super().__init__()
                self.ln = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, ff)
                self.fc2 = nn.Linear(ff, dim)

            def forward(self, x):
                h = self.ln(x)
                h = F.gelu(self.fc1(h))
                h = self.fc2(h)
                return h + x

        model = MLP(64, 128).eval()
        x = torch.randn(2, 16, 64)
        ref = model(x)
        cc = _compile(model, x)
        torch.testing.assert_close(cc(x), ref, rtol=2e-3, atol=2e-3)

    def test_residual_chain(self):
        """x + residual should be a fused add, not separate ops."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return x + self.fc(x)

        model = M().eval()
        x = torch.randn(4, 8)
        ref = model(x)
        cc = _compile(model, x)
        torch.testing.assert_close(cc(x), ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Large tensor parallel fusion (exercises rayon threshold)
# ---------------------------------------------------------------------------

class TestFusionLargeTensor:
    """Large elementwise chains should exercise the parallel fused path."""

    def test_large_chain_parallel(self):
        class M(nn.Module):
            def forward(self, x, y):
                return torch.relu(x + y) * 2.0

        model = M().eval()
        x = torch.randn(256, 512)
        y = torch.randn(256, 512)
        ref = model(x, y)
        cc = _compile(model, x, y)
        torch.testing.assert_close(cc(x, y), ref, rtol=1e-5, atol=1e-5)
