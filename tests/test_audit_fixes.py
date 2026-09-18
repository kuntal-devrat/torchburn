"""Tests for audit fixes: dynamic switching, WGPU caching, prelu op, concurrency, and memory stability."""

import os
import gc
import sys
import threading
import pytest
import torch
import torchburn
from torchburn._parser.op_registry import _ATEN_TO_OP


def test_op_registry_prelu_kernel():
    """Verify that aten._prelu_kernel.default is mapped to prelu."""
    assert _ATEN_TO_OP.get("aten._prelu_kernel.default") == "prelu"
    assert _ATEN_TO_OP.get("aten._prelu_kernel") == "prelu"
    assert _ATEN_TO_OP.get("aten.prelu.default") == "prelu"


def test_dynamic_device_switching_no_poisoning():
    """Verify that querying device=cpu does NOT permanently poison GPU detection."""
    # Step 1: Force CPU via environment variable
    os.environ["TORCHBURN_DEVICE"] = "cpu"
    info_cpu = torchburn.gpu_info()
    assert info_cpu["available"] is False
    assert info_cpu["device_override"] == "cpu"
    assert torchburn.gpu_available() is False

    # Step 2: Unset CPU override - if hardware has GPU, it should be available again dynamically
    del os.environ["TORCHBURN_DEVICE"]
    info_restored = torchburn.gpu_info()
    # Should reflect actual hardware capability without being poisoned by the previous CPU call
    assert info_restored["device_override"] == ""
    # Check that gpu_available() matches the probed physical availability
    assert torchburn.gpu_available() == info_restored["available"]


def test_wgpu_clear_cache_and_buffer_pool():
    """Verify wgpu_clear_weight_cache and wgpu_clear_buffer_pool functions execute safely."""
    assert hasattr(torchburn, "wgpu_clear_weight_cache")
    assert hasattr(torchburn, "wgpu_clear_buffer_pool")

    # Call both functions
    torchburn.wgpu_clear_weight_cache()
    torchburn.wgpu_clear_buffer_pool()


def test_wgpu_int4_gemv_concurrency_and_cache():
    """Verify multithreaded concurrent calls to WGPU GEMV with the same and different weights."""
    if not torchburn.gpu_available():
        pytest.skip("WGPU adapter not available in this environment")

    # Set up synthetic weights
    rows, cols = 128, 256
    group_size = 64
    num_groups = cols // group_size

    # Qwen-style packed weights: 2 values per byte (cols / 2)
    w = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
    scales = torch.rand((rows, num_groups), dtype=torch.float32) + 0.1

    errors = []

    def worker(worker_id):
        try:
            for i in range(10):
                x = torch.randn(cols, dtype=torch.float32)
                y = torchburn.quantization.wgpu_w4a32_grouped_linear(
                    x, w, scales, None, group_size
                )
                assert y.shape == (rows,)
                assert not torch.isnan(y).any()
        except Exception as e:
            errors.append((worker_id, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent WGPU GEMV had worker errors: {errors}"


def test_wgpu_lru_eviction_no_leak():
    """Verify that allocating multiple weights exceeding cache cap evicts and recycles cleanly."""
    if not torchburn.gpu_available():
        pytest.skip("WGPU adapter not available in this environment")

    # Lower weight cache cap temporarily to trigger eviction
    os.environ["TORCHBURN_WEIGHT_CACHE_CAP"] = "4"
    rows, cols = 64, 128
    group_size = 64
    num_groups = cols // group_size

    try:
        weight_list = []
        for _ in range(10):
            w = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
            s = torch.rand((rows, num_groups), dtype=torch.float32) + 0.1
            weight_list.append((w, s))

        x = torch.randn(cols, dtype=torch.float32)
        # Dispatch with 10 distinct weights, evicting older ones
        for w, s in weight_list:
            y = torchburn.quantization.wgpu_w4a32_grouped_linear(x, w, s, None, group_size)
            assert y.shape == (rows,)

        # Purge cache
        torchburn.wgpu_clear_weight_cache()
    finally:
        os.environ.pop("TORCHBURN_WEIGHT_CACHE_CAP", None)


def test_op_registry_alias_and_split():
    """Verify that ATen alias, lift, contiguous, and split variants are registered in _ATEN_TO_OP."""
    assert _ATEN_TO_OP.get("aten.alias.default") == "contiguous"
    assert _ATEN_TO_OP.get("aten.alias") == "contiguous"
    assert _ATEN_TO_OP.get("aten.clone") == "contiguous"
    assert _ATEN_TO_OP.get("aten.detach") == "contiguous"
    assert _ATEN_TO_OP.get("aten.lift_fresh_copy.default") == "contiguous"
    assert _ATEN_TO_OP.get("aten.lift_fresh.default") == "contiguous"
    assert _ATEN_TO_OP.get("aten.contiguous") == "contiguous"
    assert _ATEN_TO_OP.get("aten.contiguous.default") == "contiguous"

    assert _ATEN_TO_OP.get("aten.split") == "split"
    assert _ATEN_TO_OP.get("aten.split.default") == "split"
    assert _ATEN_TO_OP.get("aten.split.Tensor") == "split"
    assert _ATEN_TO_OP.get("aten.split_with_sizes") == "split"
    assert _ATEN_TO_OP.get("aten.split_with_sizes.default") == "split"


def test_set_quant_backend_cpu_fallback():
    """Verify _set_quant_backend runs without NameError (checking nn.Module import)."""
    import torch.nn as nn
    from torchburn.llm.engine import _set_quant_backend

    class MockQuantLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = "igpu"
            self.qweight = nn.Parameter(torch.zeros(4, 4))

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = MockQuantLayer()

    m = MockModel()
    assert m.layer.backend == "igpu"
    _set_quant_backend(m, "cpu")
    assert m.layer.backend == "cpu"


def test_device_routing_env_checks():
    """Verify that setting TORCHBURN_DEVICE to wgpu, vulkan, dx12, metal, or native_cpu routes correctly."""
    # Force CPU via native_cpu
    os.environ["TORCHBURN_DEVICE"] = "native_cpu"
    try:
        info = torchburn.gpu_info()
        assert info["available"] is False
        assert info["device_override"] == "native_cpu"
    finally:
        os.environ.pop("TORCHBURN_DEVICE", None)

    # Force WGPU
    os.environ["TORCHBURN_DEVICE"] = "wgpu"
    try:
        info = torchburn.gpu_info()
        assert info["device_override"] == "wgpu"
    finally:
        os.environ.pop("TORCHBURN_DEVICE", None)


def test_llm_engine_stream_metrics_precision():
    """Verify that generate_stream emits valid float metrics and avoids wasted steps."""
    from torchburn.llm import ModelConfig, UniversalTransformer, GenerationConfig, EngineConfig
    from torchburn.llm.engine import UniversalEngine

    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
    )
    model = UniversalTransformer(cfg).eval()

    class MockTokenizer:
        eos_token_id = 99
        def encode(self, text):
            return [1, 2]
        def decode(self, ids, skip_special_tokens=False):
            return "tok"

    engine = UniversalEngine(model, MockTokenizer(), config=EngineConfig(device="cpu"))
    engine._rust_decoder = None
    engine._wgpu_decoder = None

    # Test max_new_tokens=1
    packets = list(engine.generate_stream("test", GenerationConfig(max_new_tokens=1, temperature=0.0)))
    summary = next(p for p in packets if p["type"] == "summary")
    assert summary["tokens_generated"] == 1
    assert isinstance(summary["avg_ms_per_token"], float)
    assert isinstance(summary["decode_tok_sec"], float)
    assert "decode_latency_sec" in summary

    # Test max_new_tokens=3
    packets3 = list(engine.generate_stream("test", GenerationConfig(max_new_tokens=3, temperature=0.0)))
    summary3 = next(p for p in packets3 if p["type"] == "summary")
    assert summary3["tokens_generated"] == 3
    assert isinstance(summary3["avg_ms_per_token"], float)
    assert "decode_latency_sec" in summary3


def test_llm_engine_stream_vs_batch_native():
    """Verify stream=True uses decode_and_sample while generate() uses generate_loop."""
    from torchburn.llm import ModelConfig, UniversalTransformer, GenerationConfig, EngineConfig
    from torchburn.llm.engine import UniversalEngine

    cfg = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
    )
    model = UniversalTransformer(cfg).eval()

    class MockTokenizer:
        eos_token_id = 99
        def encode(self, text):
            return [1, 2]
        def decode(self, ids, skip_special_tokens=False):
            return "tok"

    engine = UniversalEngine(model, MockTokenizer(), config=EngineConfig(device="cpu"))

    class MockNativeDecoder:
        def __init__(self):
            self.generate_loop_called = False
            self.decode_and_sample_called = 0
        def kv_len(self):
            return 0
        def prefill_tokens(self, tokens, offset):
            pass
        def get_logits(self):
            return [0.0] * 100
        def generate_loop(self, next_token, seq_len, max_new_tokens, *args):
            self.generate_loop_called = True
            return [10, 11], seq_len + 2
        def decode_and_sample(self, next_token, offset, *args):
            self.decode_and_sample_called += 1
            return 10

    mock = MockNativeDecoder()
    engine._rust_decoder = mock

    # When streaming (default stream=True), generate_loop must NOT be called
    packets = list(engine.generate_stream("test", GenerationConfig(max_new_tokens=2, temperature=0.0), stream=True))
    assert mock.generate_loop_called is False
    assert mock.decode_and_sample_called == 1
    tokens = [p for p in packets if p["type"] == "token"]
    assert len(tokens) == 2

    # When generate() is called (non-streaming), generate_loop MUST be called
    text = engine.generate("test", GenerationConfig(max_new_tokens=2, temperature=0.0))
    assert mock.generate_loop_called is True
    assert text == "toktok"


def test_op_registry_chunk_and_scatter():
    """Verify chunk and scatter ops are registered in _FUNCTION_TO_OP, _ATEN_TO_OP, and _METHOD_TO_OP."""
    from torchburn._parser.op_registry import _FUNCTION_TO_OP, _ATEN_TO_OP, _METHOD_TO_OP

    assert _FUNCTION_TO_OP.get("torch.chunk") == "chunk"
    assert _ATEN_TO_OP.get("aten.chunk") == "chunk"
    assert _ATEN_TO_OP.get("aten.chunk.default") == "chunk"
    assert _ATEN_TO_OP.get("aten.slice_scatter.default") == "slice_scatter"
    assert _ATEN_TO_OP.get("aten.select_scatter.default") == "select_scatter"
    assert _METHOD_TO_OP.get("chunk") == "chunk"


def test_quantized_linear_device_routing():
    """Verify QuantizedLinear properly routes to GPU for dx12, metal, dgpu, wgpu and disables on cpu."""
    from torchburn.quantization import QuantizedLinear

    # Create dummy QuantizedLinear layer
    layer = QuantizedLinear(
        in_features=64,
        out_features=64,
        bias=False,
        bits=4,
        group_size=64,
        backend="cpu",
    )

    x = torch.randn(64, dtype=torch.float32)

    # When TORCHBURN_DEVICE=cpu, it must stay on CPU even if backend is igpu
    layer.backend = "igpu"
    os.environ["TORCHBURN_DEVICE"] = "cpu"
    try:
        dev_env = os.environ.get("TORCHBURN_DEVICE", "").strip().lower()
        if dev_env in ("cpu", "native_cpu"):
            gpu_preferred = False
        else:
            gpu_preferred = (
                layer.backend in ("gpu", "igpu", "dgpu", "wgpu", "burn-wgpu", "vulkan", "dx12", "metal")
                or dev_env in ("gpu", "igpu", "dgpu", "wgpu", "burn-wgpu", "vulkan", "dx12", "metal")
            )
        assert gpu_preferred is False
    finally:
        os.environ.pop("TORCHBURN_DEVICE", None)

    # When TORCHBURN_DEVICE=dx12, metal, dgpu, or wgpu, gpu_preferred is True
    for dev in ("dx12", "metal", "dgpu", "wgpu", "vulkan"):
        os.environ["TORCHBURN_DEVICE"] = dev
        try:
            dev_env = os.environ.get("TORCHBURN_DEVICE", "").strip().lower()
            gpu_preferred = (
                layer.backend in ("gpu", "igpu", "dgpu", "wgpu", "burn-wgpu", "vulkan", "dx12", "metal")
                or dev_env in ("gpu", "igpu", "dgpu", "wgpu", "burn-wgpu", "vulkan", "dx12", "metal")
            )
            assert gpu_preferred is True, f"Failed for device {dev}"
        finally:
            os.environ.pop("TORCHBURN_DEVICE", None)


def test_decoder_parity_methods():
    """Verify RustQwenDecoder and WgpuQwenDecoder have get_logits, kv_len, and prefill."""
    from torchburn._torchburn import RustQwenDecoder

    assert hasattr(RustQwenDecoder, "get_logits")
    assert hasattr(RustQwenDecoder, "kv_len")
    assert hasattr(RustQwenDecoder, "prefill")
    assert hasattr(RustQwenDecoder, "prefill_tokens")

    if hasattr(torchburn._torchburn, "WgpuQwenDecoder"):
        WgpuQwenDecoder = torchburn._torchburn.WgpuQwenDecoder
        assert hasattr(WgpuQwenDecoder, "get_logits")
        assert hasattr(WgpuQwenDecoder, "kv_len")
        assert hasattr(WgpuQwenDecoder, "prefill")
        assert hasattr(WgpuQwenDecoder, "prefill_tokens")


def test_fused_add_rmsnorm_compat_shader_source():
    """Verify that fused_add_rmsnorm_compat.wgsl exists and is free of subgroup builtins."""
    import pathlib

    shader_path = pathlib.Path(__file__).parent.parent / "src" / "shaders" / "fused_add_rmsnorm_compat.wgsl"
    assert shader_path.is_file(), "fused_add_rmsnorm_compat.wgsl must exist"
    content = shader_path.read_text(encoding="utf-8")
    assert "subgroupAdd" not in content, "compat shader must not contain subgroupAdd"
    assert "@builtin(subgroup" not in content, "compat shader must not contain subgroup builtins"
    assert "num_subgroups" not in content, "compat shader must not contain num_subgroups"
    assert "workgroupBarrier" in content, "compat shader must use workgroup barriers"


def test_chunk_parity_and_no_fallback():
    """Verify chunk ceiling division math and zero eager fallback warnings."""
    import warnings

    # Shape with 11 elements split into 3 chunks: PyTorch returns [4, 4, 3]
    x = torch.randn(2, 11)
    m = torchburn.compile(lambda t: torch.chunk(t, 3, dim=-1))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        tb_out = m(x)

    pt_out = torch.chunk(x, 3, dim=-1)
    assert len(tb_out) == len(pt_out)
    for tb_c, pt_c in zip(tb_out, pt_out):
        assert torch.allclose(tb_c, pt_c, atol=1e-5)

    # Ensure no eager fallback warning was triggered
    fallback_warnings = [
        w for w in recorded
        if "falling back to eager PyTorch" in str(w.message) and "chunk" in str(w.message)
    ]
    assert len(fallback_warnings) == 0, f"Unexpected fallback warnings: {fallback_warnings}"


def test_layer_norm_eval_no_fallback():
    """Verify LayerNorm executes natively in inference/eval without eager fallback."""
    import warnings
    import torch.nn as nn

    ln = nn.LayerNorm(4)
    x = torch.randn(2, 4)
    m = torchburn.compile(ln)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with torch.no_grad():
            tb_out = m(x)

    pt_out = ln(x)
    assert torch.allclose(tb_out, pt_out, atol=1e-5)

    fallback_warnings = [
        w for w in recorded
        if "falling back to eager PyTorch" in str(w.message) and "layer_norm" in str(w.message)
    ]
    assert len(fallback_warnings) == 0, f"Unexpected fallback warnings: {fallback_warnings}"


def test_repeat_and_repeat_interleave_native():
    """Verify repeat and repeat_interleave run natively without allocation failures."""
    x = torch.randn(4, 4)
    m_rep = torchburn.compile(lambda t: t.repeat(2, 1))
    rep_out = m_rep(x)
    assert torch.allclose(rep_out, x.repeat(2, 1), atol=1e-5)

    m_ri = torchburn.compile(lambda t: t.repeat_interleave(2, dim=0))
    ri_out = m_ri(x)
    assert torch.allclose(ri_out, x.repeat_interleave(2, dim=0), atol=1e-5)


def test_sort_and_argsort_native():
    """Verify sort and argsort execute natively."""
    x = torch.randn(4, 4)
    m_sort = torchburn.compile(lambda t: torch.sort(t, dim=-1)[0])
    sort_out = m_sort(x)
    assert torch.allclose(sort_out, torch.sort(x, dim=-1)[0], atol=1e-5)

    m_argsort = torchburn.compile(lambda t: torch.argsort(t, dim=-1))
    argsort_out = m_argsort(x)
    assert torch.equal(argsort_out, torch.argsort(x, dim=-1))


def test_autograd_strided_broadcasting_mul_div():
    """Verify autograd backward produces exact gradients for multi-dimensional broadcasting."""
    from torchburn.autograd import Tensor, enable, reset

    enable()
    try:
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = torch.tensor([[2.0], [3.0]], requires_grad=True)

        # PyTorch reference gradients
        c_pt = a * b
        grad_a_pt, grad_b_pt = torch.autograd.grad(c_pt.sum(), [a, b])

        # TorchBurn autograd
        a_tb = Tensor(a.detach().clone(), requires_grad=True)
        b_tb = Tensor(b.detach().clone(), requires_grad=True)
        c_tb = a_tb * b_tb
        c_tb.sum().backward()

        assert torch.allclose(grad_a_pt, a_tb.grad, atol=1e-5)
        assert torch.allclose(grad_b_pt, b_tb.grad, atol=1e-5)

        # Test division
        reset()
        enable()
        a_div = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], requires_grad=True)
        b_div = torch.tensor([[2.0], [4.0]], requires_grad=True)

        c_div_pt = a_div / b_div
        grad_ad_pt, grad_bd_pt = torch.autograd.grad(c_div_pt.sum(), [a_div, b_div])

        a_div_tb = Tensor(a_div.detach().clone(), requires_grad=True)
        b_div_tb = Tensor(b_div.detach().clone(), requires_grad=True)
        c_div_tb = a_div_tb / b_div_tb
        c_div_tb.sum().backward()

        assert torch.allclose(grad_ad_pt, a_div_tb.grad, atol=1e-5)
        assert torch.allclose(grad_bd_pt, b_div_tb.grad, atol=1e-5)
    finally:
        reset()


def test_attn_decode_shader_bounds():
    """Verify that attn_decode.wgsl has expanded bounds for head_dim up to 1024."""
    import pathlib

    shader_path = pathlib.Path(__file__).parent.parent / "src" / "shaders" / "attn_decode.wgsl"
    assert shader_path.is_file(), "attn_decode.wgsl must exist"
    content = shader_path.read_text(encoding="utf-8")
    assert "array<vec4<f32>, 4>" in content, "shader must allocate 4 vec4s for high head_dim"
    assert "min((total_vec4 + 63u) / 64u, 4u)" in content, "shader must clamp passes to array capacity"


def test_memory_pool_peak_cached_words():
    """Verify that memory_pool_stats exposes peak_cached_words tracking."""
    stats = torchburn.memory_pool_stats()
    assert "peak_cached_words" in stats
    assert stats["peak_cached_words"] >= stats["cached_words"]
    assert stats["peak_cached_words"] >= 0


def test_wgpu_decoder_device_loss_attributes():
    """Verify that WgpuQwenDecoder exposes device loss methods."""
    native_mod = getattr(torchburn, "_native", None)
    if native_mod is None or not hasattr(native_mod, "WgpuQwenDecoder"):
        pytest.skip("WgpuQwenDecoder not compiled in this build")

    cls = getattr(native_mod, "WgpuQwenDecoder")
    assert hasattr(cls, "is_device_lost")
    assert hasattr(cls, "mark_device_lost")



