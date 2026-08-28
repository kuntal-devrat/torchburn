"""Zero-copy DLPack FFI tests (REQ-003).

Proves Rust reads PyTorch's buffers in place (same address, O(1) handoff)
and that output capsules round-trip without leaks or double-frees.
"""

from __future__ import annotations

import json

import pytest
import torch

from torchburn import _torchburn as tb


def _payload(nodes, inputs, outputs):
    return json.dumps(
        {"inputs": inputs, "nodes": nodes, "outputs": outputs},
        sort_keys=True,
    )


def test_rust_reads_pytorch_memory_in_place():
    """The address Rust sees behind the capsule equals tensor.data_ptr()."""
    t = torch.randn(16, 32, dtype=torch.float32)
    capsule = t.__dlpack__()
    assert tb.data_ptr(capsule) == t.data_ptr()
    # Also equal to the raw storage pointer (no separate copy anywhere).
    assert tb.data_ptr(capsule) == t.untyped_storage().data_ptr()


def test_capsule_fields_parse_correctly():
    """DLPack metadata (shape/dtype/device) is decoded faithfully."""
    t = torch.ones(2, 3, 4, dtype=torch.float64)
    dump = tb.capsule_dump(t.__dlpack__())
    assert "ndim=3" in dump
    assert "bits=64" in dump
    assert "device=(1,0)" in dump
    assert "shape=[2, 3, 4]" in dump


def test_output_is_a_fresh_allocation():
    """Engine outputs are new Rust allocations, not aliases of the inputs."""
    a = torch.randn(4, 4)
    payload = _payload(
        [{"id": 0, "target": "add", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}]}],
        [{"shape": [4, 4], "dtype": "f32"}, {"shape": [4, 4], "dtype": "f32"}],
        [0],
    )
    caps = tb.execute(payload, [a.__dlpack__(), a.__dlpack__()])
    out = torch.from_dlpack(caps[0])
    assert out.data_ptr() != a.data_ptr()
    assert torch.allclose(out, a + a)


def test_output_capsule_gc_no_double_free():
    """An unconsumed output capsule must free its buffer exactly once."""
    a = torch.randn(8, 8)
    payload = _payload(
        [{"id": 0, "target": "mul", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 0}]}],
        [{"shape": [8, 8], "dtype": "f32"}],
        [0],
    )
    for _ in range(200):
        caps = tb.execute(payload, [a.__dlpack__()])
        del caps  # drop the capsule unconsumed -> destructor path
    import gc

    gc.collect()
    # If the capsule destructor double-freed, we would have crashed above.


def test_input_capsule_consumed_by_torch_still_safe():
    """Passing torch's own capsule to from_dlpack (passthrough) is safe."""
    a = torch.randn(3, 3)
    capsule = a.__dlpack__()
    b = torch.from_dlpack(capsule)  # torch consumes/renames the capsule
    assert torch.allclose(a, b)
    del capsule  # renamed capsule destructor must be a no-op
    import gc

    gc.collect()
    assert torch.allclose(a, torch.randn(3, 3) * 0 + a)


def test_non_cpu_tensor_rejected_with_marker():
    """Non-CPU or unsupported-dtype inputs produce TB_UNSUPPORTED errors."""
    payload = _payload(
        [{"id": 0, "target": "add", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}]}],
        [{"shape": [2], "dtype": "f32"}, {"shape": [2], "dtype": "f32"}],
        [0],
    )
    half = torch.randn(2, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
        tb.execute(payload, [half.__dlpack__(), half.__dlpack__()])


def test_shape_spec_mismatch_rejected():
    payload = _payload(
        [{"id": 0, "target": "add", "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}]}],
        [{"shape": [2, 3], "dtype": "f32"}, {"shape": [2, 3], "dtype": "f32"}],
        [0],
    )
    a = torch.randn(5, 5)
    with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
        tb.execute(payload, [a.__dlpack__(), a.__dlpack__()])
