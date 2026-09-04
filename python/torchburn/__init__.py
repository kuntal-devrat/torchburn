"""TorchBurn — a hardware-agnostic PyTorch compilation backend written in Rust.

Phase 1 (FFI Bridge Foundation) ships:

* ``torch.compile(model, backend="torchburn")`` backend registration (REQ-001),
* zero-copy DLPack tensor passing between PyTorch and Rust (REQ-003),
* elementary kernels (add / sub / mul / div / relu) executing in Rust,
* BLAKE3 structural graph caching with a thread-safe global map (REQ-004),
* safe eager-PyTorch fallback for unsupported operators (REQ-002).

Usage::

    import torch
    import torchburn  # registers the backend

    model = torch.nn.Sequential(...)
    compiled = torch.compile(model, backend="torchburn")

``import torchburn`` registers the backend automatically; ``torchburn.register()``
re-registers it explicitly if needed.
"""

from __future__ import annotations

import os
import sys

# On Windows, ensure OpenBLAS DLL directory is registered for dynamic loading
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        os.add_dll_directory(_pkg_dir)
    except Exception:
        pass
    _vendor_bin = os.path.abspath(os.path.join(_pkg_dir, "..", "..", "vendor", "OpenBLAS_prebuilt", "bin"))
    if os.path.isdir(_vendor_bin):
        try:
            os.add_dll_directory(_vendor_bin)
        except Exception:
            pass

import logging

# Structured logging via TORCHBURN_LOG env (debug/info/warning/error)
_log_level = os.getenv("TORCHBURN_LOG", "").strip().lower()
if _log_level in ("debug", "info", "warning", "error", "critical"):
    _lvl = getattr(logging, _log_level.upper(), logging.INFO)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[torchburn:%(levelname)s] %(message)s"))
    _log = logging.getLogger("torchburn")
    _log.setLevel(_lvl)
    _log.addHandler(_handler)
    _log.propagate = False
    # Also set root for Rust side via env
    os.environ.setdefault("RUST_LOG", _log_level)

import torch

from ._backend import register, torchburn_backend
from ._cache import cache_clear, cache_stats
from ._compiled import BurnCompiledCallable
# Default RAYON_NUM_THREADS to physical core count if not set to prevent hyperthread contention
if "RAYON_NUM_THREADS" not in os.environ:
    try:
        import psutil
        _phys_cores = psutil.cpu_count(logical=False) or 4
    except Exception:
        _phys_cores = 4
    os.environ["RAYON_NUM_THREADS"] = str(_phys_cores)

from .capture import TorchBurnModule, capture
from . import _torchburn as _native
from .profiler import (
    profile,
    coverage_report,
    memory_stats,
    reset_stats,
    supported_ops,
    active_engine,
    memory_pool_stats,
    clear_memory_pool,
    trace,
    op_coverage,
)
from . import quantization
from .quantization import (
    QuantizedLinear,
    quantize_model,
    w8a32_linear,
    w4a32_linear,
    w4a32_grouped_linear,
    fused_swiglu_mlp,
    fused_attention_step,
    quantize_weight_int8,
    quantize_weight_int4,
    quantize_weight_int4_grouped,
)

try:
    __version__ = _native.__version__  # from Cargo.toml
except AttributeError:
    try:
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("torchburn")
    except Exception:
        __version__ = "0.1.0"


def gpu_info():
    """Return GPU adapter information as a dict.

    Keys:
        available (bool): whether a GPU adapter was detected.
        adapter_name (str): human-readable adapter name.
        backend (str): graphics API (Metal, Vulkan, DirectX 12, none).
        vram_bytes (int): estimated VRAM in bytes (0 if unknown).
        device_override (str): current TORCHBURN_DEVICE env var value.
    """
    return _native.gpu_info()


def gpu_available() -> bool:
    """Check if a GPU adapter is available for wgpu execution."""
    return _native.gpu_available()


def gpu_backend() -> str:
    """Return the name of the active GPU backend (e.g. 'Metal', 'Vulkan', 'none')."""
    return _native.gpu_backend()


def rayon_threads() -> int:
    """Return the active number of Rayon worker threads."""
    return _native.rayon_threads()


__all__ = [
    "BurnCompiledCallable",
    "TorchBurnModule",
    "cache_clear",
    "cache_stats",
    "rayon_threads",
    "capture",
    "compile",
    "gpu_available",
    "gpu_backend",
    "gpu_info",
    "register",
    "torchburn_backend",
    "__version__",
    "profile",
    "coverage_report",
    "memory_stats",
    "reset_stats",
    "supported_ops",
    "active_engine",
    "memory_pool_stats",
    "clear_memory_pool",
    "trace",
    "op_coverage",
    "quantization",
    "QuantizedLinear",
    "quantize_model",
    "w8a32_linear",
    "w4a32_linear",
    "w4a32_grouped_linear",
    "fused_swiglu_mlp",
    "quantize_weight_int8",
    "quantize_weight_int4",
    "quantize_weight_int4_grouped",
]


def compile(model, **kwargs):
    """Convenience wrapper: ``torchburn.compile(model)`` == ``torch.compile(model, backend="torchburn")``."""
    return torch.compile(model, backend="torchburn", **kwargs)


def export(model, args=None, kwargs=None, dynamic_shapes=None, **compile_kwargs):
    """Export a model via ``torch.export`` and compile with TorchBurn.

    This is a thin wrapper that first tries ``torch.export.export`` (available
    in torch>=2.1) to capture the graph, then compiles the resulting
    ``ExportedProgram`` with the TorchBurn backend. On older torch or on
    failure it falls back to ``torch.compile``.

    Args:
        model: nn.Module to export.
        args: tuple of example inputs.
        kwargs: dict of keyword inputs.
        dynamic_shapes: dynamic shape spec for ``torch.export``.
        **compile_kwargs: extra kwargs forwarded to ``torch.compile``.
    """
    args = args or ()
    kwargs = kwargs or {}
    # Try torch.export path first
    try:
        from torch.export import export as torch_export
        # torch.export.export expects args as tuple
        ep = torch_export(model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
        # ExportedProgram has a `module` method that returns GraphModule
        if hasattr(ep, "module"):
            try:
                gm = ep.module()
                return torch.compile(gm, backend="torchburn", **compile_kwargs)
            except Exception:
                pass
        # Fallback: return the ExportedProgram itself compiled via torch.compile on its run
        return torch.compile(model, backend="torchburn", **compile_kwargs)
    except (ImportError, AttributeError, Exception) as e:
        import warnings
        warnings.warn(f"torchburn.export: torch.export not available or failed ({e}); falling back to torch.compile")
        return torch.compile(model, backend="torchburn", **compile_kwargs)


register()
