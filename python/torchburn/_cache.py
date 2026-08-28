"""Thread-safe global graph cache (REQ-004).

A structural BLAKE3 signature (computed in Rust) keys the cache. On a hit,
tracing/parsing is skipped entirely and the existing compiled plan is reused.
"""

from __future__ import annotations

import threading
from typing import Any

from . import _torchburn as _native
from ._parser import payload_json

# signature -> {"plan": plan}
GRAPH_CACHE: dict[str, dict[str, Any]] = {}
_LOCK = threading.RLock()


def lookup(signature: str) -> dict[str, Any] | None:
    with _LOCK:
        cached = _native.cache_get(signature)
        if cached is not None:
            return GRAPH_CACHE.get(signature)
        return None


def store(signature: str, plan: dict[str, Any]) -> None:
    with _LOCK:
        _native.cache_put(signature, payload_json(plan))
        GRAPH_CACHE[signature] = {"plan": plan}


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
