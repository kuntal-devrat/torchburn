"""Tests for Phase fixes: expand(-1), multi-norm, mixed precision, profiling, chunk splitting."""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


class TestExpandDimMinusOne:
    """expand with -1 dimension (keep same size)."""

    def _run(self, fn, shape):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable
        x = torch.randn(*shape)
        gm = make_fx(fn)(x)
        cb = BurnCompiledCallable(gm, [x])
        out = cb(x)
        ref = fn(x)
        assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max().item()

    def test_expand_minus1_last_dim(self):
        # unsqueeze(0) on (4,8) -> (1,4,8), expand(3, -1, -1) -> (3,4,8)
        def fn(x):
            return x.unsqueeze(0).expand(3, -1, -1)
        self._run(fn, (4, 8))

    def test_expand_minus1_broadcast(self):
        # unsqueeze(0) on (1,4,8) -> (1,1,4,8), expand(-1, 5, -1, -1) -> (1,5,4,8)
        def fn(x):
            return x.unsqueeze(0).expand(-1, 5, -1, -1)
        self._run(fn, (1, 4, 8))

    def test_expand_minus1_multiple(self):
        # (2,4) -> unsqueeze(0) -> (1,2,4), expand(3, -1, -1) -> (3,2,4)
        def fn(x):
            return x.unsqueeze(0).expand(-1, -1, 4)
        self._run(fn, (2, 4))


class TestMultiNorm:
    """General p-norm via linalg_vector_norm."""

    def _run(self, fn, shape=(4, 8), atol=1e-4):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable
        x = torch.randn(*shape)
        gm = make_fx(fn)(x)
        cb = BurnCompiledCallable(gm, [x])
        out = cb(x)
        ref = fn(x)
        assert torch.allclose(out, ref, atol=atol), (out - ref).abs().max().item()

    def test_l1_norm(self):
        self._run(lambda x: torch.linalg.vector_norm(x, ord=1, dim=1))

    def test_l2_norm(self):
        self._run(lambda x: torch.linalg.vector_norm(x, ord=2, dim=1))

    def test_linf_norm(self):
        self._run(lambda x: torch.linalg.vector_norm(x, ord=float('inf'), dim=1))

    def test_l0_norm(self):
        self._run(lambda x: torch.linalg.vector_norm(x, ord=0, dim=1))

    def test_p3_norm(self):
        self._run(lambda x: torch.linalg.vector_norm(x, ord=3, dim=1))

    def test_norm_no_dim(self):
        self._run(lambda x: torch.linalg.vector_norm(x, ord=2))


class TestMixedPrecision:
    """fp16/bf16 auto-cast to f32 for native execution."""

    def test_fp16_add(self):
        def fn(x):
            return x + x
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable
        x = torch.randn(4, 8, dtype=torch.float16)
        gm = make_fx(fn)(x)
        cb = BurnCompiledCallable(gm, [x])
        out = cb(x)
        assert out.dtype == torch.float16, f"expected float16, got {out.dtype}"
        assert torch.allclose(out.float(), (x + x).float(), atol=1e-3)

    def test_bf16_add(self):
        def fn(x):
            return x + x
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        gm = make_fx(fn)(x)
        cb = BurnCompiledCallable(gm, [x])
        out = cb(x)
        assert out.dtype == torch.bfloat16, f"expected bfloat16, got {out.dtype}"
        assert torch.allclose(out.float(), (x + x).float(), atol=1e-1)

    def test_fp16_mul(self):
        def fn(x):
            return x * 2.0
        from torch.fx.experimental.proxy_tensor import make_fx
        from torchburn._compiled import BurnCompiledCallable
        x = torch.randn(4, 8, dtype=torch.float16)
        gm = make_fx(fn)(x)
        cb = BurnCompiledCallable(gm, [x])
        out = cb(x)
        assert out.dtype == torch.float16


class TestChunkSplitting:
    """Large chunks are split into sub-chunks."""

    def test_split_threshold(self):
        from torchburn._compiled import BurnCompiledCallable
        assert BurnCompiledCallable._MAX_CHUNK_NODES == 128

    def test_medium_model_runs(self):
        class MediumModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(10)])
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        model = MediumModel()
        compiled = torch.compile(model, backend='torchburn')
        x = torch.randn(4, 32)
        out = compiled(x)
        ref = model(x)
        assert torch.allclose(out, ref, atol=1e-4)


class TestProfilingAPI:
    """torchburn.profiler module."""

    def test_profile_context_manager(self):
        import torchburn
        with torchburn.profile() as p:
            pass
        assert p.wall_time_ms >= 0
        # burn_ndarray may appear with wgpu unavailable suffix on headless CI
        assert p.engine in ("native_cpu", "burn_ndarray", "burn_wgpu") or p.engine.startswith("burn_ndarray")

    def test_coverage_report(self):
        import torchburn
        report = torchburn.coverage_report()
        assert "supported_targets" in report
        assert "target_count" in report
        assert report["target_count"] > 100

    def test_memory_stats(self):
        import torchburn
        torchburn.reset_stats()
        stats = torchburn.memory_stats()
        assert stats["calls"] == 0
        assert stats["total_ms"] == 0.0

    def test_supported_ops(self):
        import torchburn
        ops = torchburn.supported_ops()
        assert "add" in ops
        assert "matmul" in ops
        assert "layer_norm" in ops
        assert len(ops) > 100
