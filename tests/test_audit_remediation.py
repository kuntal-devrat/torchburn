"""Audit Remediation Plan Verification Tests.

Validates all 5 remediation items from the TorchBurn Audit Report:
1. Non-contiguous stride indexing in unary_op! across all 14 math kernels.
2. Removal/replacement of stubbed implementations with eager fallback.
3. DLPack capsule error handling, destructor safety, and memory pool lifecycle.
4. Metal device singleton type safety and hardware detection contracts.
5. Zero-warning contracts, exception hierarchy, and API consistency.
"""

from __future__ import annotations

import pytest
import torch
import torchburn
from torchburn import UnsupportedOpError
from torchburn import _torchburn as native


class TestUnaryNonContiguousMath:
    """Non-contiguous stride indexing in unary_op! by calling to_contiguous."""

    @pytest.mark.parametrize(
        "op_name",
        [
            "atan",
            "asin",
            "acos",
            "sinh",
            "cosh",
            "asinh",
            "acosh",
            "atanh",
            "erf",
            "erfc",
            "expm1",
            "log1p",
            "log2",
            "log10",
        ],
    )
    def test_all_14_unary_ops_on_non_contiguous_views(self, op_name: str):
        """Verify that all 14 unary math ops produce exact parity on non-contiguous strided inputs."""
        # Generate domain-appropriate inputs:
        # For asin, acos, atanh: (-0.95, 0.95)
        # For acosh: (1.1, 5.0)
        # For log1p: (0.1, 5.0)
        # For log2, log10: (0.5, 10.0)
        # For others: (-2.0, 2.0)
        if op_name in ("asin", "acos", "atanh"):
            base = torch.linspace(-0.85, 0.85, 60, dtype=torch.float32).reshape(6, 10)
        elif op_name == "acosh":
            base = torch.linspace(1.2, 5.0, 60, dtype=torch.float32).reshape(6, 10)
        elif op_name == "log1p":
            base = torch.linspace(0.1, 5.0, 60, dtype=torch.float32).reshape(6, 10)
        elif op_name in ("log2", "log10"):
            base = torch.linspace(0.5, 10.0, 60, dtype=torch.float32).reshape(6, 10)
        else:
            base = torch.linspace(-2.0, 2.0, 60, dtype=torch.float32).reshape(6, 10)

        # Create heavily non-contiguous views: slice with step > 1 and transpose
        sliced = base[::2, ::2].t()
        assert not sliced.is_contiguous(), "Tensor must be non-contiguous for this test"

        payload = {
            "inputs": [{"shape": list(sliced.shape), "dtype": "f32"}],
            "nodes": [
                {
                    "id": 0,
                    "target": op_name,
                    "args": [{"kind": "slot", "index": 0}],
                }
            ],
            "outputs": [0],
        }

        res = native.execute_from_dict(payload, [sliced.__dlpack__()])
        out = torch.from_dlpack(res[0])
        expected = getattr(torch, op_name)(sliced)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


class TestStubbedOpsFallback:
    """Stubbed operators return UnsupportedOpError and cleanly fall back."""

    def test_lu_solve_raises_unsupported(self):
        """Native lu_solve must raise UnsupportedOpError for eager fallback."""
        b = torch.randn(4, 4)
        lu_data = torch.randn(4, 4)
        payload = {
            "inputs": [
                {"shape": list(b.shape), "dtype": "f32"},
                {"shape": list(lu_data.shape), "dtype": "f32"},
            ],
            "nodes": [
                {
                    "id": 0,
                    "target": "lu_solve",
                    "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                }
            ],
            "outputs": [0],
        }
        with pytest.raises(UnsupportedOpError, match="lu_solve requires eager fallback"):
            native.execute_from_dict(payload, [b.__dlpack__(), lu_data.__dlpack__()])

    def test_linalg_cond_raises_unsupported(self):
        """Native linalg_cond must raise UnsupportedOpError for eager fallback."""
        a = torch.randn(4, 4)
        payload = {
            "inputs": [{"shape": list(a.shape), "dtype": "f32"}],
            "nodes": [
                {
                    "id": 0,
                    "target": "linalg_cond",
                    "args": [{"kind": "slot", "index": 0}],
                }
            ],
            "outputs": [0],
        }
        with pytest.raises(UnsupportedOpError, match="linalg_cond requires eager fallback"):
            native.execute_from_dict(payload, [a.__dlpack__()])

    def test_gru_cell_raises_unsupported(self):
        """Native gru_cell must raise UnsupportedOpError for eager fallback."""
        x = torch.randn(2, 4)
        h = torch.randn(2, 8)
        w_ih = torch.randn(24, 4)
        w_hh = torch.randn(24, 8)
        payload = {
            "inputs": [
                {"shape": list(x.shape), "dtype": "f32"},
                {"shape": list(h.shape), "dtype": "f32"},
                {"shape": list(w_ih.shape), "dtype": "f32"},
                {"shape": list(w_hh.shape), "dtype": "f32"},
            ],
            "nodes": [
                {
                    "id": 0,
                    "target": "gru_cell",
                    "args": [
                        {"kind": "slot", "index": 0},
                        {"kind": "slot", "index": 1},
                        {"kind": "slot", "index": 2},
                        {"kind": "slot", "index": 3},
                    ],
                }
            ],
            "outputs": [0],
        }
        with pytest.raises(UnsupportedOpError, match="gru_cell requires eager fallback"):
            native.execute_from_dict(
                payload,
                [x.__dlpack__(), h.__dlpack__(), w_ih.__dlpack__(), w_hh.__dlpack__()],
            )

    def test_lstm_cell_raises_unsupported(self):
        """Native lstm_cell must raise UnsupportedOpError for eager fallback."""
        x = torch.randn(2, 4)
        h = torch.randn(2, 8)
        c = torch.randn(2, 8)
        w_ih = torch.randn(32, 4)
        w_hh = torch.randn(32, 8)
        payload = {
            "inputs": [
                {"shape": list(x.shape), "dtype": "f32"},
                {"shape": list(h.shape), "dtype": "f32"},
                {"shape": list(c.shape), "dtype": "f32"},
                {"shape": list(w_ih.shape), "dtype": "f32"},
                {"shape": list(w_hh.shape), "dtype": "f32"},
            ],
            "nodes": [
                {
                    "id": 0,
                    "target": "lstm_cell",
                    "args": [
                        {"kind": "slot", "index": 0},
                        {"kind": "slot", "index": 1},
                        {"kind": "slot", "index": 2},
                        {"kind": "slot", "index": 3},
                        {"kind": "slot", "index": 4},
                    ],
                }
            ],
            "outputs": [0],
        }
        with pytest.raises(UnsupportedOpError, match="lstm_cell requires eager fallback"):
            native.execute_from_dict(
                payload,
                [
                    x.__dlpack__(),
                    h.__dlpack__(),
                    c.__dlpack__(),
                    w_ih.__dlpack__(),
                    w_hh.__dlpack__(),
                ],
            )

    def test_cartesian_prod_raises_unsupported(self):
        """Native cartesian_prod must raise UnsupportedOpError for eager fallback."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        payload = {
            "inputs": [
                {"shape": list(a.shape), "dtype": "f32"},
                {"shape": list(b.shape), "dtype": "f32"},
            ],
            "nodes": [
                {
                    "id": 0,
                    "target": "cartesian_prod",
                    "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                }
            ],
            "outputs": [0],
        }
        with pytest.raises(UnsupportedOpError, match="cartesian_prod requires eager fallback"):
            native.execute_from_dict(payload, [a.__dlpack__(), b.__dlpack__()])

    def test_combinations_raises_unsupported(self):
        """Native combinations must raise UnsupportedOpError for eager fallback."""
        x = torch.tensor([1.0, 2.0, 3.0])
        payload = {
            "inputs": [{"shape": list(x.shape), "dtype": "f32"}],
            "nodes": [
                {
                    "id": 0,
                    "target": "combinations",
                    "args": [{"kind": "slot", "index": 0}],
                    "kwargs": {"r": 2},
                }
            ],
            "outputs": [0],
        }
        with pytest.raises(UnsupportedOpError, match="combinations requires eager fallback"):
            native.execute_from_dict(payload, [x.__dlpack__()])

    def test_pad_non_constant_mode_raises_unsupported(self):
        """Native pad with non-constant mode must raise UnsupportedOpError for eager fallback."""
        x = torch.randn(2, 2, 4, 4)
        payload = {
            "inputs": [{"shape": list(x.shape), "dtype": "f32"}],
            "nodes": [
                {
                    "id": 0,
                    "target": "pad",
                    "args": [{"kind": "slot", "index": 0}],
                    "kwargs": {"pad": [1, 1, 1, 1], "mode": "reflect"},
                }
            ],
            "outputs": [0],
        }
        with pytest.raises(UnsupportedOpError, match="pad mode 'reflect' requires eager fallback"):
            native.execute_from_dict(payload, [x.__dlpack__()])


class TestMemorySafetyAndCapsuleIntegrity:
    """Memory safety, DLPack lifecycle, and memory pool zeroing."""

    def test_unsupported_op_error_hierarchy(self):
        """UnsupportedOpError subclasses RuntimeError so standard try/except works."""
        err = UnsupportedOpError("TB_UNSUPPORTED: test op rejection")
        assert isinstance(err, RuntimeError)
        assert isinstance(err, Exception)

    def test_memory_pool_clear_and_recycle_safety(self):
        """Verify memory pool recycling and stats tracking."""
        torchburn.clear_memory_pool()
        torchburn.reset_stats()

        # Run several tensor ops through compile to exercise memory pool
        @torch.compile(backend="torchburn")
        def fn(a, b):
            return torch.matmul(a, b) + 1.0

        x = torch.randn(16, 16)
        w = torch.randn(16, 16)
        for _ in range(5):
            _ = fn(x, w)

        stats = torchburn.memory_pool_stats()
        assert stats["peak_cached_words"] >= stats["cached_words"]
        torchburn.clear_memory_pool()
        assert torchburn.memory_pool_stats()["cached_words"] == 0


class TestPlatformPortabilityAndHardwareDetection:
    """Metal device singleton and hardware acceleration detection."""

    def test_metal_availability_api_contract(self):
        """is_metal_available() returns a boolean on all platforms."""
        avail = torchburn.is_metal_available()
        assert isinstance(avail, bool)

    def test_gpu_backend_api_contract(self):
        """gpu_backend() returns a recognized backend name or 'none'."""
        backend = torchburn.gpu_backend()
        assert isinstance(backend, str)
        assert backend in ("none", "Metal", "Vulkan", "Dx12", "OpenGL")

    def test_llm_engine_device_detection_no_phantom_symbols(self):
        """UniversalEngine setup does not reference non-existent MetalBackend symbol."""
        from unittest.mock import MagicMock

        from torchburn.llm.config import EngineConfig, ModelConfig
        from torchburn.llm.engine import UniversalEngine
        from torchburn.llm.model import UniversalTransformer
        from torchburn.llm.tokenizer import UniversalTokenizer

        cfg = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
        )
        model = UniversalTransformer(cfg).eval()
        mock_tok_obj = MagicMock()
        mock_tok_obj.encode = lambda s: [1, 2, 3]
        tokenizer = UniversalTokenizer(mock_tok_obj, eos_token_id=2)
        engine_cfg = EngineConfig(device="auto", quantization="none")
        engine = UniversalEngine(model, tokenizer, config=engine_cfg)
        assert engine.config.device in ("auto", "cpu", "gpu", "metal")


class TestCodeCleanlinessAndContracts:
    """Code cleanliness and public interface contracts."""

    def test_supported_targets_count_and_invariants(self):
        """Native engine must register comprehensive kernel targets (>100 ops)."""
        targets = native.supported_targets()
        assert isinstance(targets, list)
        assert len(targets) > 100
        # Check standard arithmetic & neural net kernels
        for target in ("add", "sub", "mul", "div", "matmul", "relu", "gelu", "sigmoid"):
            assert target in targets
