"""Phase 4 — attention: SDPA (plain/causal/masked), RoPE, MHA chain.

Runs through the raw FFI and the export path (make_fx + BurnCompiledCallable).
Every engine (native / burn ndarray / burn wgpu) must produce torch-matching
results with zero eager fallbacks.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
import pytest

import torchburn
from torchburn import ops
from torchburn import _torchburn as _native
from torchburn._compiled import BurnCompiledCallable
from torchburn._parser import payload_json
from torch.fx.experimental.proxy_tensor import make_fx

torch.manual_seed(0)


def _spec(t: torch.Tensor) -> dict:
    dtype = {torch.float32: "f32", torch.float64: "f64", torch.bool: "bool"}[t.dtype]
    return {"shape": list(t.shape), "dtype": dtype}


def _run(payload: dict, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    caps = [t.__dlpack__() for t in tensors]
    return [torch.from_dlpack(c) for c in _native.execute(payload_json(payload), caps)]


def _sdpa_payload(tensors: list[torch.Tensor], is_causal: bool = False) -> dict:
    return {
        "inputs": [_spec(t) for t in tensors],
        "nodes": [
            {
                "id": 0,
                "target": "scaled_dot_product_attention",
                "args": [{"kind": "slot", "index": i} for i in range(len(tensors))],
                "kwargs": {"is_causal": is_causal},
            }
        ],
        "outputs": [0],
    }


class TestScaledDotProductAttention:
    def test_plain(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        got = _run(_sdpa_payload([q, k, v]), [q, k, v])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v), atol=1e-4)

    def test_is_causal(self):
        q = torch.randn(1, 4, 32, 64)
        k = torch.randn(1, 4, 32, 64)
        v = torch.randn(1, 4, 32, 64)
        got = _run(_sdpa_payload([q, k, v], is_causal=True), [q, k, v])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-4)

    def test_bool_mask(self):
        T = 16
        q = torch.randn(2, 4, T, 32)
        k = torch.randn(2, 4, T, 32)
        v = torch.randn(2, 4, T, 32)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        got = _run(_sdpa_payload([q, k, v, mask]), [q, k, v, mask])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-4)

    def test_float_mask(self):
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        mask = torch.randn(B, H, T, T)
        got = _run(_sdpa_payload([q, k, v, mask]), [q, k, v, mask])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-4)

    def test_float_mask_tt_broadcast(self):
        T = 12
        q = torch.randn(2, 2, T, 24)
        k = torch.randn(2, 2, T, 24)
        v = torch.randn(2, 2, T, 24)
        mask = torch.randn(T, T)
        got = _run(_sdpa_payload([q, k, v, mask]), [q, k, v, mask])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-4)

    def test_ops_api(self):
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        assert torch.allclose(ops.scaled_dot_product_attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-4)

    def test_large_parallel_path(self):
        """Regression: the SIMD+rayon kernel (>= 1M MACs, B*H blocks) must
        match torch — the pre-optimization suite only covered the serial path."""
        q = torch.randn(4, 4, 64, 32)
        k = torch.randn(4, 4, 64, 32)
        v = torch.randn(4, 4, 64, 32)
        got = _run(_sdpa_payload([q, k, v]), [q, k, v])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v), atol=1e-4)

    def test_large_parallel_causal(self):
        q = torch.randn(4, 4, 64, 32)
        k = torch.randn(4, 4, 64, 32)
        v = torch.randn(4, 4, 64, 32)
        got = _run(_sdpa_payload([q, k, v], is_causal=True), [q, k, v])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-4)

    def test_large_parallel_bool_mask(self):
        """Parallel path + broadcast bool mask (fully/partially masked rows)."""
        T = 64
        q = torch.randn(4, 4, T, 32)
        k = torch.randn(4, 4, T, 32)
        v = torch.randn(4, 4, T, 32)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        got = _run(_sdpa_payload([q, k, v, mask]), [q, k, v, mask])[0]
        assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-4)

    def test_export_path(self):
        """Full parser/interpreter path — must be fully native (no fallback)."""
        B, H, T, D = 1, 2, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        gm = make_fx(fn)(q, k, v)
        compiled = BurnCompiledCallable(gm, [q, k, v])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = compiled(q, k, v)
        assert torch.allclose(got, fn(q, k, v), atol=1e-4)

    def test_export_causal(self):
        q = torch.randn(1, 2, 16, 32)
        k = torch.randn(1, 2, 16, 32)
        v = torch.randn(1, 2, 16, 32)

        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        gm = make_fx(fn)(q, k, v)
        compiled = BurnCompiledCallable(gm, [q, k, v])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = compiled(q, k, v)
        assert torch.allclose(got, fn(q, k, v), atol=1e-4)


class TestRotaryEmbedding:
    def test_rope_split_half(self):
        T, D = 8, 16
        x = torch.randn(1, 4, T, D)
        cos = torch.randn(T, D // 2)
        sin = torch.randn(T, D // 2)
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        ref = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        got = ops.rotary_embedding(x, cos, sin)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_rope_rank3_broadcast(self):
        T, D = 6, 8
        x = torch.randn(2, T, D)
        cos = torch.randn(1, T, D // 2)
        sin = torch.randn(1, T, D // 2)
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        ref = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        got = ops.rotary_embedding(x, cos, sin)
        assert torch.allclose(got, ref, atol=1e-5)


class TestMultiHeadAttentionChain:
    def test_mha_style_graph(self):
        """A BERT-style MHA layer decomposed into primitives (linear + SDPA)."""
        B, T, D, H = 2, 8, 32, 4
        head = D // H
        x = torch.randn(B, T, D)
        w_q = torch.randn(D, D)
        w_k = torch.randn(D, D)
        w_v = torch.randn(D, D)
        w_o = torch.randn(D, D)

        def mha(x):
            q = x @ w_q
            k = x @ w_k
            v = x @ w_v
            q = q.view(B, T, H, head).transpose(1, 2)
            k = k.view(B, T, H, head).transpose(1, 2)
            v = v.view(B, T, H, head).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            return out @ w_o

        gm = make_fx(mha)(x)
        compiled = BurnCompiledCallable(gm, [x])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = compiled(x)
        assert torch.allclose(got, mha(x), atol=1e-3)


class TestFusedQKVExportPath:
    """Regression: fused-QKV split (qkv[0] → select), SDPA with None mask,
    and slice/ellipsis round-tripping in rope graphs (all found in the
    Phase 4 re-audit)."""

    def test_fused_qkv_sdpa_with_mask(self):
        torch.manual_seed(0)

        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = torch.nn.Linear(64, 192)
                self.proj = torch.nn.Linear(64, 64)

            def forward(self, x, mask=None):
                b, s, _ = x.shape
                qkv = self.qkv(x).reshape(b, s, 3, 8, 8).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                return self.proj(a.reshape(b, s, 64))

        mod = Attn()
        x = torch.randn(2, 6, 64)
        mask = torch.ones(2, 8, 6, 6, dtype=torch.bool)
        compiled = torch.compile(mod, backend="torchburn")
        with torch.no_grad():
            out = compiled(x, mask)
        with torch.no_grad():
            ref = mod(x, mask)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_rope_slice_ellipsis_roundtrip(self):
        torch.manual_seed(0)

        def rope(x):
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1, x2], dim=-1)

        x = torch.randn(2, 6, 64)
        compiled = torch.compile(rope, backend="torchburn")
        with torch.no_grad():
            out = compiled(x)
        assert torch.allclose(out, rope(x), atol=1e-4)

    def test_select_op_payload(self):
        x = torch.randn(3, 2, 4)
        payload = {
            "inputs": [_spec(x)],
            "nodes": [
                {"id": 0, "target": "select", "args": [{"kind": "slot", "index": 0}],
                 "kwargs": {"dim": 0, "index": 1}},
            ],
            "outputs": [0],
        }
        got = _run(payload, [x])[0]
        assert torch.allclose(got, x[1], atol=1e-6)
        assert list(got.shape) == [2, 4]

    def test_cat_seq_payload(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 5)
        payload = {
            "inputs": [_spec(a), _spec(b)],
            "nodes": [
                {"id": 0, "target": "cat",
                 "args": [{"kind": "slot", "value": [0, 1]}],
                 "kwargs": {"dim": 1}},
            ],
            "outputs": [0],
        }
        got = _run(payload, [a, b])[0]
        assert torch.allclose(got, torch.cat([a, b], dim=1), atol=1e-6)


class TestTransformerBlockEndToEnd:
    """A real transformer block (embedding + MHA + MLP + layer_norm) compiled
    end-to-end via torch.compile — regression for the Phase 4 audit fixes:
    F.embedding's functional target mapping, its Python (input, weight) arg
    order vs the ATen (weight, indices) schema, and dynamo's injected default
    const args (padding_idx/max_norm/...) tripping the bool-const guard."""

    def _make_block(self, vocab=1000, d=64, heads=4, ff=128):
        class TransformerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab, d)
                self.ln1 = torch.nn.LayerNorm(d)
                self.qkv = torch.nn.Linear(d, 3 * d)
                self.proj = torch.nn.Linear(d, d)
                self.ln2 = torch.nn.LayerNorm(d)
                self.ff1 = torch.nn.Linear(d, ff)
                self.ff2 = torch.nn.Linear(ff, d)
                self.heads = heads
                self.d = d

            def forward(self, x):
                x = self.embed(x)
                x = self.ln1(x)
                b, s, _ = x.shape
                h, hd = self.heads, self.d // self.heads
                qkv = self.qkv(x).reshape(b, s, 3, h, hd).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                a = F.scaled_dot_product_attention(q, k, v)
                x = x + self.proj(a.reshape(b, s, self.d))
                x = self.ln2(x)
                x = x + self.ff2(F.relu(self.ff1(x)))
                return x

        return TransformerBlock()

    def test_compile_correct_and_native(self):
        torch.manual_seed(0)
        mod = self._make_block()
        x = torch.randint(0, 1000, (2, 16))
        compiled = torch.compile(mod, backend="torchburn")
        with torch.no_grad():
            out = compiled(x)
        with torch.no_grad():
            ref = mod(x)
        assert torch.allclose(out, ref, atol=1e-3)

        # second call: no eager fallback warnings (embedding now native)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                compiled(x)
        fallbacks = [w for w in caught if "torchburn" in str(w.message)]
        assert fallbacks == [], [str(w.message) for w in fallbacks]
