import torch
import pytest
import torchburn

def test_complex_and_real_imag():
    class ComplexOpsModule(torch.nn.Module):
        def forward(self, re, im):
            # Polar / Angle / Conj / Real / Imag
            c_real = torch.sin(re)
            c_imag = torch.cos(im)
            mag = torch.sqrt(c_real * c_real + c_imag * c_imag)
            return mag

    mod = ComplexOpsModule()
    re = torch.randn(4, 8)
    im = torch.randn(4, 8)

    compiled = torchburn.compile(mod)
    out = compiled(re, im)
    expected = mod(re, im)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

def test_fft_shifts():
    class FFTShiftModule(torch.nn.Module):
        def forward(self, x):
            # FFT shift circular shift emulation
            return torch.roll(x, shifts=(x.shape[-1] // 2,), dims=(-1,))

    mod = FFTShiftModule()
    x = torch.randn(4, 16)
    compiled = torchburn.compile(mod)
    out = compiled(x)
    expected = mod(x)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
