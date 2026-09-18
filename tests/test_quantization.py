import torch
import pytest
import torchburn

def test_quantize_dequantize_per_tensor():
    class QuantDequantModule(torch.nn.Module):
        def forward(self, x):
            # Scale and round quantization simulate
            scale = 0.05
            zero_point = 0
            q = torch.clamp(torch.round(x / scale) + zero_point, -128, 127)
            dq = (q - zero_point) * scale
            return dq

    mod = QuantDequantModule()
    x = torch.randn(4, 16)
    compiled = torchburn.compile(mod)
    out = compiled(x)
    expected = mod(x)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

def test_quantized_linear_sim():
    class QuantizedMatmul(torch.nn.Module):
        def forward(self, a, b):
            # Simulated INT8 GEMM with scales
            return torch.matmul(a, b) * 0.01

    mod = QuantizedMatmul()
    a = torch.randint(-128, 127, (8, 16), dtype=torch.float32)
    b = torch.randint(-128, 127, (16, 32), dtype=torch.float32)

    compiled = torchburn.compile(mod)
    out = compiled(a, b)
    expected = mod(a, b)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_quantize_weight_int8_and_w8a32_linear():
    torch.manual_seed(42)
    weight = torch.randn(16, 32, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)
    x = torch.randn(4, 32, dtype=torch.float32)

    qw, sc = torchburn.quantization.quantize_weight_int8(weight)
    assert qw.dtype == torch.int8
    assert qw.shape == (16, 32)
    assert sc.shape == (16,)

    out = torchburn.quantization.w8a32_linear(x, qw, sc, bias)
    assert out.shape == (4, 16)
    # Check that quantization preserves the rough linear projection
    float_out = torch.nn.functional.linear(x, weight, bias)
    # Cosine similarity should be very high (> 0.95)
    cos_sim = torch.nn.functional.cosine_similarity(out.flatten(), float_out.flatten(), dim=0)
    assert cos_sim > 0.95


def test_quantize_weight_int4_grouped_and_linear():
    torch.manual_seed(42)
    weight = torch.randn(16, 64, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)
    x = torch.randn(4, 64, dtype=torch.float32)

    packed, sc = torchburn.quantization.quantize_weight_int4_grouped(weight, group_size=64)
    assert packed.dtype == torch.uint8
    assert packed.shape == (16, 32)
    assert sc.shape == (16, 1)

    out = torchburn.quantization.w4a32_grouped_linear(x, packed, sc, bias, group_size=64)
    assert out.shape == (4, 16)
    float_out = torch.nn.functional.linear(x, weight, bias)
    cos_sim = torch.nn.functional.cosine_similarity(out.flatten(), float_out.flatten(), dim=0)
    assert cos_sim > 0.90


def test_quantized_linear_module_from_float():
    torch.manual_seed(42)
    linear = torch.nn.Linear(64, 32, bias=True)
    x = torch.randn(2, 64)

    # 8-bit QuantizedLinear
    linear8 = torch.nn.Linear(64, 32, bias=True)
    ql8 = torchburn.quantization.QuantizedLinear.from_float(linear8, bits=8)
    out8 = ql8(x)
    assert out8.shape == (2, 32)

    # 4-bit QuantizedLinear
    linear4 = torch.nn.Linear(64, 32, bias=True)
    ql4 = torchburn.quantization.QuantizedLinear.from_float(linear4, bits=4, group_size=64)
    out4 = ql4(x)
    assert out4.shape == (2, 32)


def test_quantization_validation_errors():
    # 1D weight to quantize_weight_int8 must raise ValueError
    with pytest.raises(ValueError, match="Expected 2D weight"):
        torchburn.quantization.quantize_weight_int8(torch.randn(16))

    # 3D weight to quantize_weight_int8 must raise ValueError
    with pytest.raises(ValueError, match="Expected 2D weight"):
        torchburn.quantization.quantize_weight_int8(torch.randn(2, 16, 32))

    # 1D weight to quantize_weight_int4 must raise ValueError
    with pytest.raises(ValueError, match="Expected 2D weight"):
        torchburn.quantization.quantize_weight_int4(torch.randn(16))

    # Non-divisible K to quantize_weight_int4_grouped must raise ValueError
    with pytest.raises(ValueError, match="must be divisible by group_size"):
        torchburn.quantization.quantize_weight_int4_grouped(torch.randn(16, 50), group_size=64)
