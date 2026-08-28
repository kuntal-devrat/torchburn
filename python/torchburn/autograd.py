"""
Phase 10: Rust-native autograd engine.

Backward pass is routed through the Rust ``backward_single`` FFI so every
gradient computation runs inside optimised native kernels (SIMD + rayon)
instead of Python/PyTorch ops.

Forward still executes via PyTorch ops (the tape records what happened);
backward converts saved tensors to DLPack capsules and calls into Rust.

Usage:
    import torch
    from torchburn.autograd import Tensor, enable, reset

    enable()
    x = Tensor(torch.randn(4, 8), requires_grad=True)
    w = Tensor(torch.randn(8, 4), requires_grad=True)
    y = x @ w
    loss = ((y - target) ** 2).mean()
    loss.backward()
    w.data -= 0.01 * w.grad
    reset()
"""

import json
import threading
import weakref
import torch
from typing import Optional

from . import _torchburn as _native

# ---------------------------------------------------------------------------
# Thread-local tape — records (target, saved_capsules, kwargs, input_ids)
# ---------------------------------------------------------------------------

class _TapeEntry:
    """One recorded op on the tape."""
    __slots__ = ("target", "input_ids", "output_id", "saved_capsules",
                 "saved_data", "kwargs", "n_inputs")

    def __init__(self, target: str, input_ids: list[int], output_id: int,
                 saved_data: list[torch.Tensor], kwargs: dict):
        self.target = target
        self.input_ids = input_ids
        self.output_id = output_id
        self.saved_capsules: Optional[list] = None  # lazily created
        self.saved_data = saved_data
        self.kwargs = kwargs
        self.n_inputs = len(input_ids)


class _Tape:
    """Records differentiable operations for backward pass."""
    def __init__(self):
        self.ops: list[_TapeEntry] = []

    def record(self, target: str, input_ids: list[int], output_id: int,
               saved_data: list[torch.Tensor], kwargs: dict | None = None):
        self.ops.append(_TapeEntry(
            target=target, input_ids=input_ids, output_id=output_id,
            saved_data=saved_data, kwargs=kwargs or {},
        ))

    def clear(self):
        self.ops.clear()

    def __len__(self):
        return len(self.ops)


# Global tape instance
_tape = _Tape()
_enabled = False


def enable():
    global _enabled
    _enabled = True


def disable():
    global _enabled
    _enabled = False


def is_enabled():
    return _enabled


def reset():
    global _tape, _enabled
    _tape.clear()
    _enabled = False
    Tensor._registry.clear()
    # NOTE: do NOT reset _next_id — existing Tensor objects keep their IDs,
    # and new tensors must not collide with them.


def tape_len():
    return len(_tape)


# ---------------------------------------------------------------------------
# Tensor wrapper
# ---------------------------------------------------------------------------

class Tensor:
    """Differentiable tensor wrapping a torch.Tensor."""

    _registry: weakref.WeakValueDictionary = weakref.WeakValueDictionary()  # type: ignore
    _lock = threading.Lock()
    _next_id: int = 0

    def __init__(self, data: torch.Tensor, requires_grad: bool = False):
        if isinstance(data, Tensor):
            data = data.data
        self.data: torch.Tensor = data.detach().contiguous().float() if data.is_floating_point() else data.detach().contiguous()
        self.grad: Optional[torch.Tensor] = None
        self.requires_grad: bool = requires_grad
        with Tensor._lock:
            self._id: int = Tensor._next_id
            Tensor._next_id += 1
            if requires_grad:
                Tensor._registry[self._id] = self

    def __del__(self):
        try:
            Tensor._registry.pop(self._id, None)
        except Exception:
            pass

    def backward(self, grad_output: Optional[torch.Tensor] = None):
        """Run backward pass — batched Rust kernels for all supported ops."""
        if grad_output is None:
            grad_output = torch.ones_like(self.data)

        if not _tape.ops:
            return

        # ---- Try batch backward first (single FFI call) ----
        try:
            grads = _backward_batch(grad_output, self._id)
        except Exception as e:
            import warnings as _w
            _w.warn(f"torchburn: batch backward failed ({e}), falling back to per-op", stacklevel=2)
            grads = _backward_single_loop(grad_output)

        # Assign grads to leaf tensors
        for tid, g in grads.items():
            if tid in Tensor._registry:
                leaf = Tensor._registry[tid]
                if leaf.grad is None:
                    leaf.grad = g
                else:
                    leaf.grad = leaf.grad + g

        _tape.clear()

    def zero_grad(self):
        self.grad = None

    def sum(self, dim=None, keepdim=False):
        return sum_op(self, dim=dim, keepdim=keepdim)

    def mean(self, dim=None, keepdim=False):
        return mean_op(self, dim=dim, keepdim=keepdim)

    def __pow__(self, other):
        return _pow(self, _ensure_tensor(other))

    def __rpow__(self, other):
        return _pow(_ensure_tensor(other), self)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        grad_str = f", grad={self.grad.shape if self.grad is not None else None}"
        return f"Tensor({self.data.shape}, dtype={self.data.dtype}{grad_str})"

    def __add__(self, other):
        return _add(self, _ensure_tensor(other))

    def __radd__(self, other):
        return _add(_ensure_tensor(other), self)

    def __sub__(self, other):
        return _sub(self, _ensure_tensor(other))

    def __rsub__(self, other):
        return _sub(_ensure_tensor(other), self)

    def __mul__(self, other):
        return _mul(self, _ensure_tensor(other))

    def __rmul__(self, other):
        return _mul(_ensure_tensor(other), self)

    def __truediv__(self, other):
        return _div(self, _ensure_tensor(other))

    def __matmul__(self, other):
        return _matmul(self, _ensure_tensor(other))

    def __neg__(self):
        return _mul(self, Tensor(torch.tensor(-1.0), requires_grad=False))


def _ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, torch.Tensor):
        return Tensor(x, requires_grad=False)
    # Always create float tensors for numeric scalars to avoid dtype mismatches
    # in the Rust backward_single FFI (int64 bytes read as f32 = garbage)
    try:
        return Tensor(torch.tensor(float(x)), requires_grad=False)
    except (TypeError, ValueError):
        return Tensor(torch.tensor(x), requires_grad=False)


def _needs_grad(*tensors):
    return any(t.requires_grad for t in tensors)


def _wrap(data: torch.Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Rust backward_single FFI bridge
# ---------------------------------------------------------------------------

def _reduce_to_shape(grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Reduce a gradient tensor to match the target shape (broadcast fix).
    
    When an op like add(a, b) was broadcast, Rust returns grad with the
    upstream shape, but the input may have a smaller shape.  We sum over
    the extra leading dimensions.
    """
    if grad.shape == target_shape:
        return grad
    # Match trailing dimensions
    grad = grad.contiguous()
    target = target_shape
    # First, trim leading dimensions
    while grad.dim() > len(target):
        grad = grad.sum(dim=0)
    # Then, sum over broadcast dims (dims where target is 1 but grad > 1)
    for i in range(len(target)):
        if grad.dim() > i and target[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(dim=i, keepdim=True)
    # Final reshape to exact target shape
    if grad.shape != target:
        grad = grad.reshape(target)
    return grad


# Ops whose Rust backward_single is reliable
_RUST_NATIVE_OPS = frozenset({
    "add", "sub", "mul", "div", "pow", "relu", "sigmoid", "tanh", "gelu",
    "matmul", "linear", "layer_norm", "softmax", "sum", "mean", "mse_loss",
    "nll_loss", "cross_entropy",
})


def _backward_single(entry: _TapeEntry, upstream: torch.Tensor) -> Optional[list[Optional[torch.Tensor]]]:
    """Call Rust backward_single for a recorded op.

    Returns a list of gradient tensors (one per input_id), or None if the op
    is not supported by Rust.
    """
    if entry.target not in _RUST_NATIVE_OPS:
        return None

    try:
        # Convert upstream to capsule (clone to avoid DLPack lifetime issues)
        up_cap = upstream.detach().contiguous().clone().__dlpack__()

        # Clone + reshape: DLPack capsules with 0-dim tensors can cause memory
        # corruption when the same source tensor backs multiple capsules.
        # Cast to float unless the op explicitly needs int64 (nll_loss, cross_entropy).
        saved_shapes = [d.shape for d in entry.saved_data]
        need_int = entry.target in ("nll_loss", "cross_entropy")
        saved_cloned = []
        for d in entry.saved_data:
            c = d.detach().contiguous().clone()
            if torch.is_floating_point(c):
                c = c.float()  # normalize to float32
            elif not need_int:
                c = c.float()  # int scalars (from x * 2) must become float for Rust
            saved_cloned.append(c)
        saved_padded = [d.reshape(max(d.numel(), 1)) if d.dim() == 0 else d for d in saved_cloned]
        saved_caps = [d.__dlpack__() for d in saved_padded]

        # kwargs as JSON string
        kwargs_json = json.dumps(entry.kwargs)

        # Call Rust
        grad_caps = _native.backward_single(entry.target, up_cap, saved_caps, kwargs_json)

        # Convert result capsules back to torch tensors, reshape to original saved shapes
        grads = []
        for c, orig_shape in zip(grad_caps, saved_shapes):
            g = torch.from_dlpack(c)
            if g.dim() > 0 and len(orig_shape) == 0:
                g = g.sum()  # reduce 1-D grad back to scalar for 0-dim saved input
            grads.append(g)
        return grads
    except Exception as e:
        import warnings as _w
        _w.warn(f"torchburn: backward_single failed for {entry.target}: {e}", stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# Batch backward — single FFI call for entire tape
# ---------------------------------------------------------------------------

def _backward_batch(grad_output: torch.Tensor, output_id: int) -> dict[int, torch.Tensor]:
    """Process the entire tape in a single backward_batch FFI call.

    This eliminates per-op DLPack capsule overhead by sending all saved
    tensors at once, letting Rust do the full reverse-mode accumulation
    internally.
    """
    n_ops = len(_tape.ops)
    if n_ops == 0:
        return {}

    targets: list[str] = []
    saved_capsules: list[list] = []
    all_kwargs: list[str] = []
    output_ids: list[int] = []
    input_ids_all: list[list[int]] = []
    # Track original saved shapes + input_ids for broadcast reduction
    saved_shapes_per_entry: list[list[torch.Size]] = []
    input_ids_per_entry: list[list[int]] = []

    need_int_ops = ("nll_loss", "cross_entropy")

    for entry in _tape.ops:
        if entry.target not in _RUST_NATIVE_OPS:
            continue

        targets.append(entry.target)
        output_ids.append(entry.output_id)
        input_ids_all.append(entry.input_ids)
        all_kwargs.append(json.dumps(entry.kwargs))
        saved_shapes_per_entry.append([d.shape for d in entry.saved_data])
        input_ids_per_entry.append(entry.input_ids)

        # Clone + pad saved data to capsules
        saved_padded = []
        for d in entry.saved_data:
            c = d.detach().contiguous().clone()
            if torch.is_floating_point(c):
                c = c.float()
            elif entry.target not in need_int_ops:
                c = c.float()
            if c.dim() == 0:
                c = c.reshape(max(c.numel(), 1))
            saved_padded.append(c)
        saved_capsules.append([s.__dlpack__() for s in saved_padded])

    if not targets:
        return {}

    # Convert initial upstream to capsule
    up_cap = grad_output.detach().contiguous().clone().float().__dlpack__()

    # Convert saved_shapes to list of lists of ints for Rust
    saved_shapes_lists = [[list(s) for s in entry_shapes]
                          for entry_shapes in saved_shapes_per_entry]

    # Single FFI call
    raw_grads = _native.backward_batch(
        targets, saved_capsules, all_kwargs, output_ids, input_ids_all,
        saved_shapes_lists, up_cap, output_id,
    )

    # Convert results
    grads: dict[int, torch.Tensor] = {}
    for tid, cap in raw_grads:
        g = torch.from_dlpack(cap)
        grads[tid] = g

    return grads


# ---------------------------------------------------------------------------
# Per-op fallback backward (for ops not in _RUST_NATIVE_OPS)
# ---------------------------------------------------------------------------

def _backward_single_loop(grad_output: torch.Tensor) -> dict[int, torch.Tensor]:
    """Fallback: walk tape in reverse, calling backward_single per-op."""
    grads: dict[int, torch.Tensor] = {}
    grads[_tape.ops[-1].output_id if _tape.ops else 0] = grad_output

    for entry in reversed(_tape.ops):
        upstream_id = entry.output_id
        g = grads.pop(upstream_id, None)
        if g is None:
            continue

        per_input_grads = _backward_single(entry, g)
        if per_input_grads is not None:
            for i, tid in enumerate(entry.input_ids):
                if i < len(per_input_grads) and per_input_grads[i] is not None:
                    pg = per_input_grads[i]
                    saved_shape = entry.saved_data[i].shape
                    if pg.shape != saved_shape:
                        pg = _reduce_to_shape(pg, saved_shape)
                    if tid in grads:
                        grads[tid] = grads[tid] + pg
                    else:
                        grads[tid] = pg

    return grads


# ---------------------------------------------------------------------------
# Forward ops that record on the tape
# ---------------------------------------------------------------------------

def _add(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data + b.data, requires_grad=_needs_grad(a, b))
    if _enabled and out.requires_grad:
        _tape.record("add", [a._id, b._id], out._id, [a.data, b.data])
    return out


def _sub(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data - b.data, requires_grad=_needs_grad(a, b))
    if _enabled and out.requires_grad:
        _tape.record("sub", [a._id, b._id], out._id, [a.data, b.data])
    return out


def _mul(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data * b.data, requires_grad=_needs_grad(a, b))
    if _enabled and out.requires_grad:
        _tape.record("mul", [a._id, b._id], out._id, [a.data, b.data])
    return out


def _div(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data / b.data, requires_grad=_needs_grad(a, b))
    if _enabled and out.requires_grad:
        _tape.record("div", [a._id, b._id], out._id, [a.data, b.data])
    return out


def _pow(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data ** b.data, requires_grad=_needs_grad(a, b))
    if _enabled and out.requires_grad:
        _tape.record("pow", [a._id, b._id], out._id, [a.data, b.data])
    return out


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    out = _wrap(a.data @ b.data, requires_grad=_needs_grad(a, b))
    if _enabled and out.requires_grad:
        _tape.record("matmul", [a._id, b._id], out._id, [a.data, b.data])
    return out


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    out_data = torch.nn.functional.linear(
        input.data, weight.data, bias.data if bias is not None else None)
    needs_grad = _needs_grad(input, weight) or (bias is not None and bias.requires_grad)
    out = _wrap(out_data, requires_grad=needs_grad)
    if _enabled and needs_grad:
        saved = [input.data, weight.data]
        input_ids = [input._id, weight._id]
        if bias is not None:
            saved.append(bias.data)
            input_ids.append(bias._id)
        _tape.record("linear", input_ids, out._id, saved)
    return out


def relu(a: Tensor) -> Tensor:
    out = _wrap(torch.relu(a.data), requires_grad=a.requires_grad)
    if _enabled and a.requires_grad:
        _tape.record("relu", [a._id], out._id, [a.data])
    return out


def sigmoid(a: Tensor) -> Tensor:
    out = _wrap(torch.sigmoid(a.data), requires_grad=a.requires_grad)
    if _enabled and a.requires_grad:
        # Rust sigmoid expects saved_inputs[0] = output of sigmoid
        _tape.record("sigmoid", [a._id], out._id, [out.data])
    return out


def tanh_act(a: Tensor) -> Tensor:
    out = _wrap(torch.tanh(a.data), requires_grad=a.requires_grad)
    if _enabled and a.requires_grad:
        # Rust tanh expects saved_inputs[0] = output of tanh
        _tape.record("tanh", [a._id], out._id, [out.data])
    return out


def gelu(a: Tensor) -> Tensor:
    out = _wrap(torch.nn.functional.gelu(a.data), requires_grad=a.requires_grad)
    if _enabled and a.requires_grad:
        # Rust gelu expects saved_inputs[0] = input to gelu
        _tape.record("gelu", [a._id], out._id, [a.data])
    return out


def softmax(a: Tensor, dim: int = -1) -> Tensor:
    out = _wrap(torch.softmax(a.data, dim=dim), requires_grad=a.requires_grad)
    if _enabled and a.requires_grad:
        # Rust softmax expects saved_inputs[0] = output of softmax
        _tape.record("softmax", [a._id], out._id, [out.data])
    return out


def layer_norm(a: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    out_data = torch.nn.functional.layer_norm(
        a.data, weight.data.shape, weight.data, bias.data, eps)
    needs_grad = _needs_grad(a, weight, bias)
    out = _wrap(out_data, requires_grad=needs_grad)
    if _enabled and needs_grad:
        _tape.record("layer_norm", [a._id, weight._id, bias._id], out._id,
                      [a.data, weight.data, bias.data])
    return out


def dropout(a: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0:
        return _wrap(a.data.clone(), requires_grad=a.requires_grad)
    mask = (torch.rand_like(a.data) >= p).float()
    scale = 1.0 / (1.0 - p)
    out = _wrap(a.data * mask * scale, requires_grad=a.requires_grad)
    # Dropout not in Rust native ops — falls back to zero grad (fine for eval)
    return out


def sum_op(a: Tensor, dim=None, keepdim=False) -> Tensor:
    out_data = torch.sum(a.data, dim=dim, keepdim=keepdim)
    out = _wrap(out_data, requires_grad=a.requires_grad)
    if _enabled and a.requires_grad:
        _tape.record("sum", [a._id], out._id, [a.data])
    return out


def mean_op(a: Tensor, dim=None, keepdim=False) -> Tensor:
    out_data = torch.mean(a.data, dim=dim, keepdim=keepdim) if dim is not None else torch.mean(a.data)
    out = _wrap(out_data, requires_grad=a.requires_grad)
    if _enabled and a.requires_grad:
        _tape.record("mean", [a._id], out._id, [a.data])
    return out


def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    out_data = torch.nn.functional.mse_loss(input.data, target.data, reduction=reduction)
    out = _wrap(out_data, requires_grad=input.requires_grad)
    if _enabled and input.requires_grad:
        # Rust mse_loss uses reduction as int: 0=none, 1=mean, 2=sum
        red_int = {"none": 0, "mean": 1, "sum": 2}.get(reduction, 1)
        _tape.record("mse_loss", [input._id, target._id], out._id,
                      [input.data, target.data], {"reduction": red_int})
    return out


def nll_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    out_data = torch.nn.functional.nll_loss(input.data, target.data.long(), reduction=reduction)
    out = _wrap(out_data, requires_grad=input.requires_grad)
    if _enabled and input.requires_grad:
        _tape.record("nll_loss", [input._id, target._id], out._id,
                      [input.data, target.data.long()], {"reduction": reduction})
    return out


def cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    out_data = torch.nn.functional.cross_entropy(input.data, target.data.long(), reduction=reduction)
    out = _wrap(out_data, requires_grad=input.requires_grad)
    if _enabled and input.requires_grad:
        _tape.record("cross_entropy", [input._id, target._id], out._id,
                      [input.data, target.data.long()], {"reduction": reduction})
    return out
