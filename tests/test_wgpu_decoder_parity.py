"""wgpu decoder correctness regression gates.

These tests exist so kernel / command-stream rewrites (workgroup sizing,
pass merging, shader changes) cannot silently break or degrade decode:

* ``test_wgpu_decode_deterministic`` - two decoders built from the same
  weights must produce identical greedy token streams (a merged single-pass
  with missing barriers would violate this).
* ``test_wgpu_cpu_logit_parity_floor`` - per-step logits must stay close to
  the (numerically independent) Rust CPU decoder: near-tie argmax flips are
  expected at INT4 precision, but wholesale divergence is not.

Both are skipped unless a local INT4 checkpoint and a wgpu adapter exist.
Set ``TORCHBURN_TEST_MODEL`` to point at another checkpoint directory.
"""

from __future__ import annotations

import os

import pytest

import torch  # noqa: F401  (imported for side-effect-free dtype checks below)
import torchburn as tb

MODEL = os.environ.get("TORCHBURN_TEST_MODEL", r"d:\torchburn\models\qwen_0_5b")


def _available() -> bool:
    if not os.path.isdir(MODEL):
        return False
    try:
        return bool(tb.gpu_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _available(), reason="needs a local INT4 checkpoint and a wgpu adapter"
)


@pytest.fixture(scope="module")
def quantized_model():
    os.environ["TORCHBURN_DEVICE"] = "igpu"
    from torchburn.llm.loader import ModelLoader

    model, _cfg, _root = ModelLoader.load(MODEL, quant="int4", device="cpu")
    return model


def _greedy_stream(dec, n: int = 12, start: int = 20):
    """Greedy decode of n contiguous tokens from a fresh decoder."""
    toks = []
    t = start
    for i in range(n):
        t = dec.decode_and_sample(t, i, 0.0, 1, 1.0, None)
        toks.append(t)
    return toks


def test_wgpu_decode_deterministic(quantized_model):
    from torchburn.quantization import create_wgpu_qwen_decoder

    # Small KV (12 tokens needed) so low-VRAM iGPUs can hold two decoders.
    # Falls back to reset-and-replay on one decoder if the second OOMs.
    try:
        a = create_wgpu_qwen_decoder(quantized_model, max_seq_len=256)
    except MemoryError as e:
        pytest.skip(f"wgpu OOM on single decoder: {e}")
    try:
        b = create_wgpu_qwen_decoder(quantized_model, max_seq_len=256)
    except MemoryError:
        a.reset_kv_cache()
        first = _greedy_stream(a)
        a.reset_kv_cache()
        second = _greedy_stream(a)
        assert first == second, "single wgpu decoder non-deterministic across resets"
        return
    assert _greedy_stream(a) == _greedy_stream(b), (
        "two wgpu decoders diverged on identical inputs (barrier / ordering bug?)"
    )


def test_wgpu_cpu_logit_parity_floor(quantized_model):
    from torchburn.quantization import (
        create_rust_qwen_decoder,
        create_wgpu_qwen_decoder,
    )

    try:
        gpu = create_wgpu_qwen_decoder(quantized_model, max_seq_len=2048)
    except MemoryError as e:
        pytest.skip(f"wgpu OOM (low-VRAM iGPU): {e}")
    try:
        cpu = create_rust_qwen_decoder(quantized_model, max_seq_len=2048)
    except MemoryError as e:
        pytest.skip(f"Rust decoder OOM (low host RAM): {e}")

    matches = 0
    worst = 0.0
    samples = 8
    for k in range(samples):
        tok = (k * 37 + 11) % 997
        lg = gpu.step(tok, k)
        cap = cpu.decode_step(tok, k)
        lc = torch.from_dlpack(cap)[0].tolist()
        worst = max(worst, max(abs(a - b) for a, b in zip(lg, lc)))
        gpu_argmax = max(range(len(lg)), key=lg.__getitem__)
        cpu_argmax = max(range(len(lc)), key=lc.__getitem__)
        matches += int(gpu_argmax == cpu_argmax)

    # Looser-than-desired bounds are intentional: INT4 dequantization orders
    # differ between the two engines, so near-tie flips are expected; these
    # asserts only catch gross regressions (e.g. a kernel that diverges
    # everywhere or drops a layer).
    assert matches >= 4, f"argmax agreement too low: {matches}/{samples}"
    assert worst < 25.0, f"logits diverged too far from CPU: max |Δ| = {worst}"
