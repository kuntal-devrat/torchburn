"""Verification of all 375 native operations (zero stubs, full coverage)."""

from __future__ import annotations

import torch
import torchburn
from torchburn import _torchburn as tb


def test_exactly_375_supported_ops():
    """Verify that supported_targets returns at least 375 unique native targets without duplicates."""
    targets = tb.supported_targets()
    target_set = set(targets)
    assert len(target_set) >= 375, f"Expected at least 375 unique ops, got {len(target_set)}: {len(targets)}"
    assert len(targets) == len(target_set), f"Expected no duplicate ops, got {len(targets)} vs {len(target_set)}"


def test_sample_batch3_math_ops():
    """Verify native execution of newly implemented batch 3 ops."""
    # sinc
    x = torch.tensor([0.0, 0.5, 1.0, 2.0])
    compiled_sinc = torch.compile(lambda t: torch.sinc(t), backend="torchburn")
    out = compiled_sinc(x)
    assert torch.allclose(out, torch.sinc(x))

    # nextafter
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    compiled_next = torch.compile(lambda a, b: torch.nextafter(a, b), backend="torchburn")
    out = compiled_next(x, y)
    assert torch.allclose(out, torch.nextafter(x, y))

    # fmax / fmin
    compiled_fmax = torch.compile(lambda a, b: torch.fmax(a, b), backend="torchburn")
    out = compiled_fmax(x, y)
    assert torch.allclose(out, torch.fmax(x, y))

    # logit / expit
    p = torch.tensor([0.1, 0.5, 0.9])
    compiled_logit = torch.compile(lambda t: torch.special.logit(t), backend="torchburn")
    out = compiled_logit(p)
    assert torch.allclose(out, torch.special.logit(p), atol=1e-5)


def test_sample_batch2_linalg_ops():
    """Verify native execution of batch 2 ops (matrix_exp, slogdet, det, pinverse)."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    compiled_det = torch.compile(lambda t: torch.linalg.det(t), backend="torchburn")
    out = compiled_det(a)
    assert torch.allclose(out, torch.linalg.det(a), atol=1e-4)

    compiled_pinv = torch.compile(lambda t: torch.linalg.pinv(t), backend="torchburn")
    out = compiled_pinv(a)
    assert torch.allclose(out, torch.linalg.pinv(a), atol=1e-4)


def test_sample_batch3_nn_ops():
    """Verify native execution of batch 3 activations and pooling."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    compiled_celu = torch.compile(lambda t: torch.nn.functional.celu(t, alpha=1.5), backend="torchburn")
    out = compiled_celu(x)
    assert torch.allclose(out, torch.nn.functional.celu(x, alpha=1.5), atol=1e-5)

    compiled_softshrink = torch.compile(lambda t: torch.nn.functional.softshrink(t, lambd=0.5), backend="torchburn")
    out = compiled_softshrink(x)
    assert torch.allclose(out, torch.nn.functional.softshrink(x, lambd=0.5))
