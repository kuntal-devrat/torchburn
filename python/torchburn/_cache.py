"""Thread-safe global graph cache (REQ-004).

A structural BLAKE3 signature (computed in Rust) keys the cache. On a hit,
tracing/parsing is skipped entirely and the existing compiled plan is reused.
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
    with _LOCK:
        cached = _native.cache_get(signature)
        if cached is not None and signature in GRAPH_CACHE:
            GRAPH_CACHE.move_to_end(signature)
            return GRAPH_CACHE[signature]
        return None


def store(signature: str, plan: dict[str, Any]) -> None:
    with _LOCK:
        _native.cache_put(signature, payload_json(plan))
        if signature in GRAPH_CACHE:
            GRAPH_CACHE.move_to_end(signature)
        GRAPH_CACHE[signature] = {"plan": plan}
        while len(GRAPH_CACHE) > MAX_PYTHON_CACHE_SIZE:
            GRAPH_CACHE.popitem(last=False)


def cache_stats() -> dict[str, int]:
    """Current cache state: size, hits, misses."""
    size, hits, misses = _native.cache_stats()
    with _LOCK:
        return {
            "size": int(size),
            "hits": int(hits),
            "misses": int(misses),
            "python_size": len(GRAPH_CACHE),
        }


def cache_clear() -> None:
    """Reset both the Rust and Python cache maps."""
    _native.cache_clear()
    with _LOCK:
        GRAPH_CACHE.clear()
