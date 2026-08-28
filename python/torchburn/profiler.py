"""TorchBurn profiling and diagnostics API.

Provides:
- ``profile()`` context manager for timing compiled vs eager execution
- ``coverage_report()`` showing operator coverage breakdown
- ``memory_stats()`` showing allocation statistics
"""

from __future__ import annotations

import contextlib
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Generator

from . import _torchburn as _native


@dataclass
class ProfileResult:
    """Result from a profiling session."""
    wall_time_ms: float = 0.0
    num_nodes: int = 0
    num_supported: int = 0
    num_unsupported: int = 0
    num_fallbacks: int = 0
    engine: str = ""
    graph_signature: str = ""

    @property
    def native_ratio(self) -> float:
        """Fraction of nodes executed natively."""
        total = self.num_supported + self.num_unsupported
        return self.num_supported / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"TorchBurn Profile",
            f"  Engine:      {self.engine}",
            f"  Wall time:   {self.wall_time_ms:.2f} ms",
            f"  Nodes:       {self.num_nodes} total ({self.num_supported} native, "
            f"{self.num_unsupported} eager fallback)",
            f"  Native %:    {self.native_ratio*100:.1f}%",
            f"  Fallbacks:   {self.num_fallbacks}",
        ]
        if self.graph_signature:
            lines.append(f"  Signature:   {self.graph_signature[:32]}...")
        return "\n".join(lines)


@dataclass
class _Stats:
    """Global profiling statistics."""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _calls: int = 0
    _total_ms: float = 0.0
    _total_nodes: int = 0
    _total_native: int = 0
    _total_fallbacks: int = 0

    def record(self, result: ProfileResult) -> None:
        with self._lock:
            self._calls += 1
            self._total_ms += result.wall_time_ms
            self._total_nodes += result.num_nodes
            self._total_native += result.num_supported
            self._total_fallbacks += result.num_fallbacks

    def reset(self) -> None:
        with self._lock:
            self._calls = 0
            self._total_ms = 0.0
            self._total_nodes = 0
            self._total_native = 0
            self._total_fallbacks = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "total_ms": self._total_ms,
                "avg_ms": self._total_ms / self._calls if self._calls > 0 else 0.0,
                "total_nodes": self._total_nodes,
                "total_native": self._total_native,
                "total_fallbacks": self._total_fallbacks,
                "native_ratio": (
                    self._total_native / self._total_nodes
                    if self._total_nodes > 0 else 0.0
                ),
            }


_STATS = _Stats()

# Thread-local current profile for _interpreter to populate
_thread_local = threading.local()


def _set_current_profile(result: ProfileResult | None) -> None:
    _thread_local.current = result


def _get_current_profile() -> ProfileResult | None:
    return getattr(_thread_local, "current", None)


@contextlib.contextmanager
def profile() -> Generator[ProfileResult, None, None]:
    """Context manager for profiling TorchBurn execution.

    Usage::

        with torchburn.profiler.profile() as result:
            output = compiled_model(input_tensor)
        print(result.summary())

    The result is populated after the context manager exits.
    """
    result = ProfileResult()
    result.engine = _native.active_engine()
    _set_current_profile(result)
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result.wall_time_ms = elapsed * 1000.0
        _set_current_profile(None)
        _STATS.record(result)


def coverage_report() -> dict[str, Any]:
    """Get operator coverage statistics from cached graph compilations.

    Returns a dict with:
        - supported_targets: list of native targets
        - target_count: number of supported targets
        - engine: active engine name
    """
    targets = _native.supported_targets()
    return {
        "supported_targets": sorted(targets),
        "target_count": len(targets),
        "engine": _native.active_engine(),
    }


def memory_stats() -> dict[str, Any]:
    """Get accumulated profiling statistics since last reset.

    Returns:
        calls: number of profiled executions
        total_ms: total wall time across all profiled calls
        avg_ms: average wall time per call
        total_nodes: total graph nodes across all calls
        total_native: total native nodes across all calls
        total_fallbacks: total fallback nodes across all calls
        native_ratio: fraction of nodes executed natively
    """
    return _STATS.stats()


def reset_stats() -> None:
    """Reset accumulated profiling statistics."""
    _STATS.reset()


def supported_ops() -> list[str]:
    """Return a sorted list of all operators the native engine supports."""
    return sorted(_native.supported_targets())


def active_engine() -> str:
    """Return the name of the active execution engine."""
    return _native.active_engine()


def memory_pool_stats() -> dict[str, Any]:
    """Return memory pool diagnostics including allocation count, hit rate, and cached buffers."""
    try:
        return _native.memory_pool_stats()
    except AttributeError:
        return {
            "alloc_count": 0,
            "hit_count": 0,
            "recycle_count": 0,
            "cached_buffers": 0,
            "cached_words": 0,
            "hit_rate": 0.0,
        }


def clear_memory_pool() -> None:
    """Clear all cached buffers from the thread memory pool."""
    try:
        _native.clear_memory_pool()
    except AttributeError:
        pass


def trace(model, example_inputs, **kwargs) -> dict[str, Any]:
    """Generate a Chrome trace JSON for a model (stub for ROADMAP 15.5).

    Returns a dict with `traceEvents` that can be loaded in `chrome://tracing`
    or `perfetto`. Currently profiles the native execution via `profile()`.
    """
    import torch

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]
    # Compile and profile
    compiled = torch.compile(model, backend="torchburn", **kwargs)
    with profile() as result:
        # Warmup + timed run
        _ = compiled(*example_inputs)
    # Build minimal Chrome trace format
    return {
        "traceEvents": [
            {
                "name": "torchburn::execute",
                "cat": "torchburn",
                "ph": "X",
                "ts": 0,
                "dur": result.wall_time_ms * 1000,
                "pid": 1,
                "tid": 1,
                "args": {
                    "engine": result.engine,
                    "nodes": result.num_nodes,
                    "native": result.num_supported,
                    "fallback": result.num_unsupported,
                },
            }
        ],
        "displayTimeUnit": "ms",
        "metadata": {
            "engine": result.engine,
            "wall_time_ms": result.wall_time_ms,
        },
    }


def op_coverage(model, example_inputs, **kwargs) -> dict[str, Any]:
    """Analyze operator coverage for a specific model without executing.

    Returns:
        total_nodes: total FX nodes
        native_nodes: count of native ops
        fallback_nodes: count of fallback ops
        native_ratio: fraction native
        unsupported_ops: list of fallback op targets
        engine: active engine
    """
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from ._parser import parse_graph

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]
    # Try to get FX graph
    try:
        gm = make_fx(model)(*example_inputs)
    except Exception:
        # Fallback to torch.compile's dynamo capture via a dummy
        gm = model
        return {
            "total_nodes": 0,
            "native_nodes": 0,
            "fallback_nodes": 0,
            "native_ratio": 0.0,
            "unsupported_ops": [],
            "engine": active_engine(),
            "error": "could not capture graph via make_fx; use torch.compile for full trace",
        }
    try:
        plan, _ = parse_graph(gm, list(example_inputs))
        total = len([n for n in plan["nodes"] if n["op"] in ("supported", "unsupported")])
        native = len([n for n in plan["nodes"] if n["op"] == "supported"])
        fallback = len([n for n in plan["nodes"] if n["op"] == "unsupported"])
        unsupported = [n["fx_target"] for n in plan["nodes"] if n["op"] == "unsupported"]
        return {
            "total_nodes": total,
            "native_nodes": native,
            "fallback_nodes": fallback,
            "native_ratio": native / total if total else 0.0,
            "unsupported_ops": unsupported,
            "engine": active_engine(),
        }
    except Exception as exc:
        return {
            "total_nodes": 0,
            "native_nodes": 0,
            "fallback_nodes": 0,
            "native_ratio": 0.0,
            "unsupported_ops": [],
            "engine": active_engine(),
            "error": str(exc),
        }
