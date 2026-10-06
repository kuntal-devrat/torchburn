"""Thread-safe global graph cache (REQ-004).

A structural BLAKE3 signature (computed in Rust) keys the cache. On a hit,
tracing/parsing is skipped entirely and the existing compiled plan is reused.

Lookup is Python-first: the Python map is authoritative for the *parsed*
plan, so a Rust-side LRU eviction or `cache_clear` in another process cannot
desync the two layers. If Python holds a plan the Rust side no longer has,
the entry is re-persisted (self-healing) so native `cache_get` hits resume.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any

from . import _torchburn as _native
from ._parser import payload_json

# Bound Python-side cache to match the 1024-entry Rust LRU cache limit.
MAX_PYTHON_CACHE_SIZE = 1024

# signature -> {"plan": plan}
GRAPH_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_LOCK = threading.RLock()


def lookup(signature: str) -> dict[str, Any] | None:
    """Return the cached plan for `signature`, independent of Rust state.

    Self-heals Rust-side eviction: if Python still has the plan but the
    native cache dropped it (LRU eviction >1024 graphs, or `cache_clear`),
    re-persist it so `cache_get`-based consumers hit again.
    """
    with _LOCK:
        cached = GRAPH_CACHE.get(signature)
        if cached is not None:
            GRAPH_CACHE.move_to_end(signature)
            if not _native.cache_contains(signature):
                _native.cache_put(signature, payload_json(cached["plan"]))
            return cached
        return None


def store(signature: str, plan: dict[str, Any]) -> None:
    with _LOCK:
        _native.cache_put(signature, payload_json(plan))
        if signature in GRAPH_CACHE:
            GRAPH_CACHE.move_to_end(signature)
        GRAPH_CACHE[signature] = {"plan": plan}
        while len(GRAPH_CACHE) > MAX_PYTHON_CACHE_SIZE:
            GRAPH_CACHE.popitem(last=False)
            # Deliberately no Rust-side eviction: Python eviction is the LRU
            # tail, and a native entry left behind is inert — it only costs a
            # slot until the Rust LRU (same 1024 bound) evicts it itself.


def cache_stats() -> dict[str, int]:
    """Current cache state: size, hits, misses, evictions."""
    size, hits, misses = _native.cache_stats()
    evictions = _native.cache_evictions() if hasattr(_native, "cache_evictions") else 0
    with _LOCK:
        return {
            "size": size,
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
            "python_size": len(GRAPH_CACHE),
        }


def cache_clear() -> None:
    """Reset both the Rust and Python cache maps."""
    _native.cache_clear()
    with _LOCK:
        GRAPH_CACHE.clear()
