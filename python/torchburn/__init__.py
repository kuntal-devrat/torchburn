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

import torch

from ._backend import register, torchburn_backend
from ._cache import cache_clear, cache_stats
from ._compiled import BurnCompiledCallable
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
)

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


__all__ = [
    "BurnCompiledCallable",
    "TorchBurnModule",
    "cache_clear",
    "cache_stats",
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
]


def compile(model, **kwargs):
    """Convenience wrapper: ``torchburn.compile(model)`` == ``torch.compile(model, backend="torchburn")``."""
    return torch.compile(model, backend="torchburn", **kwargs)


register()
