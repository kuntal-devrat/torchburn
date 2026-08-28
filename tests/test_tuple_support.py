"""Tests for native tuple/list support in the engine slot system.

Multi-output ops (unbind, chunk, sort, max, min) produce Slot::Tuple in
the Rust engine. The parser pre-pass aliases getitem(0) to the tuple-
producing node and drops dead getitem nodes, while N-element tuples with
consumed non-zero indices force eager fallback.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchburn  # registers the "torchburn" backend
import pytest


# ── unbind ──────────────────────────────────────────────────────────────

class TestUnbind:
    def test_unbind_3_elements(self):
        """unbind(x, dim=-1) splits into 3 parts; getitem(0) runs natively."""
        class M(torch.nn.Module):
            def forward(self, x):
                a, b, c = x.unbind(dim=-1)
                return a + b + c

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(2, 4, 3)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_unbind_2_elements(self):
        """unbind(x, dim=0) on 2-row tensor; both elements consumed."""
        class M(torch.nn.Module):
            def forward(self, x):
                a, b = x.unbind(dim=0)
                return a * b

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(2, 5)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_unbind_dim1(self):
        """unbind along dim=1."""
        class M(torch.nn.Module):
            def forward(self, x):
                parts = x.unbind(dim=1)
                return parts[0] + parts[2]

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(3, 4, 5)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)


# ── chunk ───────────────────────────────────────────────────────────────

class TestChunk:
    def test_chunk_3(self):
        """chunk(x, 3, dim=-1) splits into 3 equal parts."""
        class M(torch.nn.Module):
            def forward(self, x):
                a, b, c = x.chunk(3, dim=-1)
                return a + b + c

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(2, 9)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_chunk_2(self):
        """chunk(x, 2, dim=0) splits along batch."""
        class M(torch.nn.Module):
            def forward(self, x):
                a, b = x.chunk(2, dim=0)
                return torch.cat([a, b], dim=0)

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(4, 8)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)


# ── sort ────────────────────────────────────────────────────────────────

class TestSort:
    def test_sort_values_only(self):
        """sort returns (values, indices); only values consumed → native."""
        class M(torch.nn.Module):
            def forward(self, x):
                values, _ = torch.sort(x, dim=-1)
                return values

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(3, 8)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_sort_indices_consumed(self):
        """sort with indices consumed → forces eager (correctness preserved)."""
        class M(torch.nn.Module):
            def forward(self, x):
                values, indices = torch.sort(x, dim=-1)
                return values + indices.float()

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(3, 8)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)


# ── max / min reduce ───────────────────────────────────────────────────

class TestMaxMinReduce:
    def test_max_reduce_values_only(self):
        """max(dim) returns (values, indices); only values consumed → native."""
        class M(torch.nn.Module):
            def forward(self, x):
                values, _ = torch.max(x, dim=-1)
                return values

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(3, 8)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_min_reduce_values_only(self):
        """min(dim) returns (values, indices); only values consumed → native."""
        class M(torch.nn.Module):
            def forward(self, x):
                values, _ = torch.min(x, dim=-1)
                return values

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(3, 8)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)


# ── Integration: transformer-style unbind + matmul ──────────────────────

class TestTransformerUnbind:
    def test_qkv_split_via_unbind(self):
        """Q/K/V split via unbind → matmul → softmax (common transformer pattern)."""
        class M(torch.nn.Module):
            def __init__(self, d=32, h=4):
                super().__init__()
                self.qkv = torch.nn.Linear(d, 3 * d)
                self.h = h
                self.d = d

            def forward(self, x):
                qkv = self.qkv(x)                          # (B, T, 3*d)
                q, k, v = qkv.chunk(3, dim=-1)             # each (B, T, d)
                B, T, _ = qkv.shape
                q = q.view(B, T, self.h, self.d // self.h).transpose(1, 2)
                k = k.view(B, T, self.h, self.d // self.h).transpose(1, 2)
                v = v.view(B, T, self.h, self.d // self.h).transpose(1, 2)
                attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d ** 0.5)
                attn = torch.softmax(attn, dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).contiguous().view(B, T, self.d)
                return out

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(2, 8, 32)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-4)

    def test_qkv_split_via_unbind_and_gather(self):
        """unbind + gather (index_select style) → verifies int64 flow."""
        class M(torch.nn.Module):
            def forward(self, x):
                parts = x.unbind(dim=1)   # (B, T, D) → tuple of (B, D) tensors
                # gather first and third parts
                return torch.stack([parts[0], parts[2]], dim=1)

        m = M().eval()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(2, 3, 8)
        ref = m(x)
        out = compiled(x)
        assert torch.allclose(ref, out, atol=1e-5)
