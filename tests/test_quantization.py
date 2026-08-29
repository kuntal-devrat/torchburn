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
