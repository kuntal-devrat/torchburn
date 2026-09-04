"""Public operator API (Phase 4).

Each function builds a one-node engine payload and executes it through the
native zero-copy FFI, returning a fresh ``torch.Tensor``.  These mirror the
engine's canonical op names, so they run identically under every backend
(native CPU, Burn ndarray, Burn wgpu).

    >>> import torch, torchburn
    >>> torchburn.ops.embedding(torch.tensor([1, 2]), torch.randn(10, 4)).shape
    torch.Size([2, 4])
"""

from __future__ import annotations

import torch

from . import _torchburn as _native
from ._parser import payload_json


def _spec(t: torch.Tensor) -> dict:
    dtype = {
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int64: "i64",
        torch.int32: "i32",
        torch.bool: "bool",
    }.get(t.dtype, "f32")
    return {"shape": [int(s) for s in t.shape], "dtype": dtype}


def _execute(target: str, tensors: list[torch.Tensor], kwargs: dict | None = None) -> torch.Tensor:
    payload = {
        "inputs": [_spec(t) for t in tensors],
        "nodes": [
            {
                "id": 0,
                "target": target,
                "args": [{"kind": "slot", "index": i} for i in range(len(tensors))],
                "kwargs": kwargs or {},
            }
        ],
        "outputs": [0],
    }
    capsules = [t.__dlpack__() for t in tensors]
    (out_capsule,) = _native.execute(payload_json(payload), capsules)
    return torch.from_dlpack(out_capsule)


# --------------------------------------------------------------------------- #
# Attention (REQ Phase 4.1)
# --------------------------------------------------------------------------- #

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention over [B, H, T, D] tensors.

    Supports optional additive/bool ``attn_mask`` and ``is_causal`` masking.
    Dropout is not applied (inference semantics).
    """
    tensors = [query, key, value]
    if attn_mask is not None:
        tensors.append(attn_mask)
    return _execute("scaled_dot_product_attention", tensors, {"is_causal": is_causal})


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary positional embedding (HF split-half convention)."""
    return _execute("rope", [x, cos, sin])


# --------------------------------------------------------------------------- #
# Embedding (REQ Phase 4.3)
# --------------------------------------------------------------------------- #

def embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Row-gather from an embedding table: ``weight[indices]``."""
    return _execute("embedding", [weight, indices])


# --------------------------------------------------------------------------- #
# Losses (REQ Phase 4.3)
# --------------------------------------------------------------------------- #

def nll_loss(input: torch.Tensor, target: torch.Tensor, reduction: str | int = "mean", ignore_index: int = -100) -> torch.Tensor:
    """Negative log-likelihood loss over log-probabilities."""
    red = {"none": 0, "mean": 1, "sum": 2}.get(reduction, reduction if isinstance(reduction, int) else 1)
    return _execute("nll_loss_forward", [input, target], {"reduction": red, "ignore_index": ignore_index})


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, reduction: str | int = "mean") -> torch.Tensor:
    """Cross-entropy loss = nll_loss(log_softmax(logits), target)."""
    return nll_loss(torch.log_softmax(logits, dim=-1), target, reduction=reduction)


def mse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str | int = "mean") -> torch.Tensor:
    red = {"none": 0, "mean": 1, "sum": 2}.get(reduction, reduction if isinstance(reduction, int) else 1)
    return _execute("mse_loss", [input, target], {"reduction": red})


def smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, reduction: str | int = "mean", beta: float = 1.0) -> torch.Tensor:
    red = {"none": 0, "mean": 1, "sum": 2}.get(reduction, reduction if isinstance(reduction, int) else 1)
    return _execute("smooth_l1_loss", [input, target], {"reduction": red, "beta": beta})


def binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction: str | int = "mean") -> torch.Tensor:
    red = {"none": 0, "mean": 1, "sum": 2}.get(reduction, reduction if isinstance(reduction, int) else 1)
    return _execute("binary_cross_entropy", [input, target], {"reduction": red})


def select(x: torch.Tensor, dim: int = 0, index: int = 0) -> torch.Tensor:
    """Index a tensor along a dim, dropping it (aten.select)."""
    return _execute("select", [x], {"dim": dim, "index": index})


def sum(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    kwargs = {"keepdim": keepdim}
    if dim is not None:
        kwargs["dim"] = dim
    return _execute("sum", [x], kwargs)


def mean(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    kwargs = {"keepdim": keepdim}
    if dim is not None:
        kwargs["dim"] = dim
    return _execute("mean", [x], kwargs)


def max(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    kwargs = {"keepdim": keepdim}
    if dim is not None:
        kwargs["dim"] = dim
    return _execute("max", [x], kwargs)


def min(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    kwargs = {"keepdim": keepdim}
    if dim is not None:
        kwargs["dim"] = dim
    return _execute("min", [x], kwargs)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _execute("softmax", [x], {"dim": dim})


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _execute("log_softmax", [x], {"dim": dim})

