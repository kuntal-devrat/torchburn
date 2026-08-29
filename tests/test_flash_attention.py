import torch
import torch.nn.functional as F
import pytest
import torchburn

def test_flash_attention_forward():
    class FlashAttnModule(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

    mod = FlashAttnModule()
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)

    compiled = torchburn.compile(mod)
    out = compiled(q, k, v)
    expected = mod(q, k, v)

    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

def test_flash_attention_causal():
    class CausalAttnModule(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    mod = CausalAttnModule()
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)

    compiled = torchburn.compile(mod)
    out = compiled(q, k, v)
    expected = mod(q, k, v)

    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
