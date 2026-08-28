"""
Tests for Phase 11: GPU Execution via wgpu.

Tests device detection, TORCHBURN_DEVICE env var, GPU info API,
and GPU execution correctness for elementwise, matmul, and transformer ops.
"""
import os
import torch
import torch.nn.functional as F
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torchburn


# ---------------------------------------------------------------------------
# GPU Detection & Info
# ---------------------------------------------------------------------------

class TestGPUDetection:
    def test_gpu_info_returns_dict(self):
        """gpu_info() returns a dict with expected keys."""
        info = torchburn.gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "adapter_name" in info
        assert "backend" in info
        assert "vram_bytes" in info
        assert "device_override" in info

    def test_gpu_info_types(self):
        """gpu_info() values have correct types."""
        info = torchburn.gpu_info()
        assert isinstance(info["available"], bool)
        assert isinstance(info["adapter_name"], str)
        assert isinstance(info["backend"], str)
        assert isinstance(info["vram_bytes"], int)
        assert isinstance(info["device_override"], str)

    def test_gpu_available_returns_bool(self):
        """gpu_available() returns a boolean."""
        result = torchburn.gpu_available()
        assert isinstance(result, bool)

    def test_gpu_backend_returns_string(self):
        """gpu_backend() returns a string."""
        result = torchburn.gpu_backend()
        assert isinstance(result, str)
        assert result in ("Metal", "Vulkan", "DirectX 12", "WebGPU", "none")

    def test_gpu_info_consistency(self):
        """gpu_available() and gpu_info()['available'] agree."""
        assert torchburn.gpu_available() == torchburn.gpu_info()["available"]

    def test_gpu_backend_consistency(self):
        """gpu_backend() matches gpu_info()['backend']."""
        assert torchburn.gpu_backend() == torchburn.gpu_info()["backend"]


# ---------------------------------------------------------------------------
# Device Override (TORCHBURN_DEVICE)
# ---------------------------------------------------------------------------

class TestDeviceOverride:
    def test_device_override_cpu(self):
        """TORCHBURN_DEVICE=cpu forces CPU execution."""
        os.environ["TORCHBURN_DEVICE"] = "cpu"
        try:
            info = torchburn.gpu_info()
            assert info["device_override"] == "cpu"
        finally:
            del os.environ["TORCHBURN_DEVICE"]

    def test_device_override_auto(self):
        """TORCHBURN_DEVICE=auto selects best available device."""
        os.environ["TORCHBURN_DEVICE"] = "auto"
        try:
            info = torchburn.gpu_info()
            assert info["device_override"] == "auto"
        finally:
            del os.environ["TORCHBURN_DEVICE"]

    def test_device_override_empty(self):
        """No TORCHBURN_DEVICE means empty override."""
        if "TORCHBURN_DEVICE" in os.environ:
            del os.environ["TORCHBURN_DEVICE"]
        info = torchburn.gpu_info()
        assert info["device_override"] == ""


# ---------------------------------------------------------------------------
# GPU Execution Correctness
# ---------------------------------------------------------------------------

class TestGPUElementwise:
    """Elementwise ops should produce correct results on GPU."""

    def test_add_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        m = M()
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        ref = x + y
        out = torch.compile(m, backend="torchburn")(x, y)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_mul_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        m = M()
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        ref = x * y
        out = torch.compile(m, backend="torchburn")(x, y)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_relu_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        m = M()
        x = torch.randn(4, 8)
        ref = torch.relu(x)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_sigmoid_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        m = M()
        x = torch.randn(4, 8)
        ref = torch.sigmoid(x)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_tanh_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        m = M()
        x = torch.randn(4, 8)
        ref = torch.tanh(x)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_gelu_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return F.gelu(x)

        m = M()
        x = torch.randn(4, 8)
        ref = F.gelu(x)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-3)

    def test_abs_neg_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.neg(torch.abs(x)) + y

        m = M()
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        ref = torch.neg(torch.abs(x)) + y
        out = torch.compile(m, backend="torchburn")(x, y)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_clamp_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-1.0, max=1.0)

        m = M()
        x = torch.randn(4, 8) * 3
        ref = torch.clamp(x, min=-1.0, max=1.0)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_sin_cos_gpu(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.sin(x) + torch.cos(y)

        m = M()
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        ref = torch.sin(x) + torch.cos(y)
        out = torch.compile(m, backend="torchburn")(x, y)
        assert torch.allclose(ref, out, atol=1e-3)


class TestGPUMatmul:
    """Matrix multiplication ops on GPU."""

    def test_matmul_2d(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x @ y

        m = M()
        x = torch.randn(4, 8)
        y = torch.randn(8, 6)
        ref = x @ y
        out = torch.compile(m, backend="torchburn")(x, y)
        assert torch.allclose(ref, out, atol=1e-3)

    def test_linear_gpu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 4)

            def forward(self, x):
                return self.linear(x)

        m = M()
        x = torch.randn(2, 8)
        ref = m(x)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-3)


class TestGPUTransformer:
    """Full transformer block on GPU."""

    def test_attention_block(self):
        class AttentionBlock(torch.nn.Module):
            def __init__(self, d_model=64, n_heads=4):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.qkv = torch.nn.Linear(d_model, 3 * d_model)
                self.proj = torch.nn.Linear(d_model, d_model)
                self.ln = torch.nn.LayerNorm(d_model)

            def forward(self, x):
                B, T, C = x.shape
                h = self.ln(x)
                qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
                q, k, v = qkv.unbind(2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
                att = torch.softmax(att, dim=-1)
                out = att @ v
                out = out.transpose(1, 2).reshape(B, T, C)
                return x + self.proj(out)

        m = AttentionBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64)
        ref = m(x)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-3)


# ---------------------------------------------------------------------------
# Engine Selection
# ---------------------------------------------------------------------------

class TestEngineSelection:
    def test_active_engine_reflects_env(self):
        """active_engine() returns the expected engine name."""
        engine = torchburn._torchburn.active_engine()
        assert isinstance(engine, str)
        # Should be one of the known engines (or fallback if headless without GPU)
        assert any(engine.startswith(base) for base in ("native_cpu", "burn_ndarray", "burn_wgpu"))


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestGPUFallback:
    def test_unsupported_op_falls_back(self):
        """Unsupported ops fall back to eager PyTorch correctly."""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.erf(x)  # unsupported -> eager

        m = M()
        x = torch.randn(4, 4)
        ref = torch.erf(x)
        out = torch.compile(m, backend="torchburn")(x)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_mixed_native_and_fallback(self):
        """Mix of native and fallback ops produces correct results."""
        class M(torch.nn.Module):
            def forward(self, x, y):
                a = x + y         # native
                b = torch.erf(a)  # fallback
                c = b * x         # native
                return c

        m = M()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        ref = torch.erf(x + y) * x
        out = torch.compile(m, backend="torchburn")(x, y)
        assert torch.allclose(ref, out, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
