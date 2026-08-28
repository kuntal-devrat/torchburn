"""BLAKE3 structural graph cache tests (REQ-004)."""

from __future__ import annotations

import json

import torch

import torchburn
from torchburn import _torchburn as tb
from torchburn._compiled import BurnCompiledCallable
from torchburn._parser import payload_json, parse_graph


_REGISTERED = False
_CALLABLES: list[BurnCompiledCallable] = []


def _spy_backend(gm, example_inputs):
    callable_ = BurnCompiledCallable(gm, example_inputs)
    _CALLABLES.append(callable_)
    return callable_


if not _REGISTERED:
    torch._dynamo.register_backend(name="torchburn_test_cache")(_spy_backend)
    _REGISTERED = True


def _compile(model, x):
    _CALLABLES.clear()
    compiled = torch.compile(model, backend="torchburn_test_cache")
    out = compiled(x)
    assert _CALLABLES, "backend was not invoked"
    return out, _CALLABLES[-1]


def test_identical_structures_share_cache_entry():
    torchburn.cache_clear()

    def model_a(x):
        return torch.relu(x * 2 + 1)

    def model_b(x):  # same structure, different function object
        return torch.relu(x * 2 + 1)

    x = torch.randn(4, 4)
    out_a, ca = _compile(model_a, x)
    out_b, cb = _compile(model_b, x)
    assert ca.signature == cb.signature
    assert torch.allclose(out_a, model_a(x))
    assert torch.allclose(out_b, model_b(x))
    stats = torchburn.cache_stats()
    assert stats["size"] == 1
    assert stats["python_size"] == 1


def test_different_structure_misses_cache():
    torchburn.cache_clear()

    def model_a(x):
        return x * 2

    def model_b(x):
        return x * 2 + 1

    x = torch.randn(4, 4)
    _, ca = _compile(model_a, x)
    _, cb = _compile(model_b, x)
    assert ca.signature != cb.signature
    assert torchburn.cache_stats()["size"] == 2


def test_signature_is_structural_not_data_dependent():
    torchburn.cache_clear()
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    plan_a, _ = parse_graph(torch.fx.symbolic_trace(lambda x: x + 1), [a])
    plan_b, _ = parse_graph(torch.fx.symbolic_trace(lambda x: x + 1), [b])
    assert tb.signature(payload_json(plan_a)) == tb.signature(payload_json(plan_b))


def test_signature_changes_with_shape():
    plan_2x2, _ = parse_graph(torch.fx.symbolic_trace(lambda x: x + 1), [torch.randn(2, 2)])
    plan_3x3, _ = parse_graph(torch.fx.symbolic_trace(lambda x: x + 1), [torch.randn(3, 3)])
    assert tb.signature(payload_json(plan_2x2)) != tb.signature(payload_json(plan_3x3))


def test_rust_cache_lookup_roundtrip():
    torchburn.cache_clear()
    payload = json.dumps(
        {"inputs": [{"shape": [2], "dtype": "f32"}], "nodes": [{"id": 0, "target": "relu", "args": []}], "outputs": [0]},
        sort_keys=True,
    )
    sig = tb.signature(payload)
    assert tb.cache_get(sig) is None
    tb.cache_put(sig, payload)
    assert tb.cache_get(sig) is not None
    size, hits, misses = tb.cache_stats()
    assert size == 1 and hits == 1 and misses == 1
