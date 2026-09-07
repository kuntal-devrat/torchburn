"""
Tests for Phase 7: extended operator coverage.
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
# scatter
# ---------------------------------------------------------------------------

class TestScatter:
    def test_scatter_basic(self):
        """scatter(src, dim, index) fills output at index positions."""
        src = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        index = torch.tensor([[0, 1, 2, 0, 1, 2]])
        ref = torch.zeros(1, 3)
        ref.scatter_(1, index, src)

        class M(torch.nn.Module):
            def forward(self, s, idx):
                return torch.zeros(1, 3).scatter(1, idx, s)

        model = M().eval()
        cc = _compile(model, src, index)
        out = cc(src, index)
        torch.testing.assert_close(out, ref)


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

class TestRepeat:
    def test_repeat_1d(self):
        """repeat tiles a 1D tensor."""
        x = torch.tensor([1.0, 2.0, 3.0])

        class M(torch.nn.Module):
            def forward(self, t):
                return t.repeat(3)

        model = M().eval()
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, x.repeat(3))

    def test_repeat_2d(self):
        """repeat tiles a 2D tensor."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        class M(torch.nn.Module):
            def forward(self, t):
                return t.repeat(2, 3)

        model = M().eval()
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, x.repeat(2, 3))


# ---------------------------------------------------------------------------
# prelu
# ---------------------------------------------------------------------------

class TestPrelu:
    def test_prelu_basic(self):
        """prelu applies parametric ReLU."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.PReLU()
                self.p.weight.data = torch.tensor([0.25])

            def forward(self, t):
                return self.p(t)

        model = M().eval()
        cc = _compile(model, x)
        out = cc(x)
        ref = F.prelu(x, torch.tensor([0.25]))
        torch.testing.assert_close(out, ref)


# ---------------------------------------------------------------------------
# nonzero
# ---------------------------------------------------------------------------

class TestNonzero:
    def test_nonzero_basic(self):
        """nonzero returns indices of non-zero elements."""
        x = torch.tensor([0.0, 1.0, 0.0, 3.0, 0.0])

        class M(torch.nn.Module):
            def forward(self, t):
                return torch.nonzero(t)

        model = M().eval()
        cc = _compile(model, x)
        out = cc(x)
        ref = torch.nonzero(x)
        torch.testing.assert_close(out, ref)

    def test_nonzero_2d(self):
        """nonzero on 2D tensor."""
        x = torch.tensor([[0.0, 1.0], [2.0, 0.0]])

        class M(torch.nn.Module):
            def forward(self, t):
                return torch.nonzero(t)

        model = M().eval()
        cc = _compile(model, x)
        out = cc(x)
        ref = torch.nonzero(x)
        torch.testing.assert_close(out, ref)


# ---------------------------------------------------------------------------
# einsum
# ---------------------------------------------------------------------------

class TestEinsum:
    def test_einsum_matmul(self):
        """einsum 'ij,jk->ik' is matrix multiply."""
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)

        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.einsum('ij,jk->ik', x, y)

        model = M().eval()
        cc = _compile(model, a, b)
        out = cc(a, b)
        ref = torch.einsum('ij,jk->ik', a, b)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end: Phase 7 ops in a training-style model
# ---------------------------------------------------------------------------

class TestPhase7EndToEnd:
    def test_model_with_prelu(self):
        """Model using PReLU in forward pass."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)
                self.act = torch.nn.PReLU()

            def forward(self, x):
                return self.act(self.fc(x))

        model = M().eval()
        x = torch.randn(2, 8)
        ref = model(x)
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_model_with_einsum(self):
        """Model using einsum in forward pass."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(8, 8))

            def forward(self, x):
                return torch.einsum('bi,ij->bj', x, self.w)

        model = M().eval()
        x = torch.randn(2, 8)
        ref = model(x)
        cc = _compile(model, x)
        out = cc(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
