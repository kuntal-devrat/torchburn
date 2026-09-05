"""Tests for TorchBurn Universal LLM API and UniversalTransformer architecture."""

import os
import pytest
import torch
import torchburn as tb
from torchburn.llm import (
    LLM,
    ModelConfig,
    GenerationConfig,
    EngineConfig,
    UniversalTransformer,
    UniversalTokenizer,
)


def test_model_config_defaults():
    """Verify ModelConfig initialization and dictionary parsing."""
    cfg = ModelConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        qkv_bias=True,
        tie_word_embeddings=True,
    )
    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 896
    assert cfg.num_attention_heads == 14
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 64
    assert cfg.qkv_bias is True
    assert cfg.tie_word_embeddings is True


def test_universal_transformer_forward_and_decode():
    """Verify UniversalTransformer forward pass (prefill) and decode step (T=1)."""
    cfg = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        qkv_bias=True,
    )
    model = UniversalTransformer(cfg).eval()

    # Prefill step (batch=1, seq_len=4)
    x = torch.randint(0, 1000, (1, 4))
    with torch.no_grad():
        logits, kv_caches = model(x)
    assert logits.shape == (1, 4, 1000)
    assert len(kv_caches) == 2
    # Verify KV cache shapes: [1, 2, 4, 64]
    k, v = kv_caches[0]
    assert k.shape == (1, 2, 4, 64)

    # Decode step (batch=1, seq_len=1, offset=4)
    x_next = torch.tensor([[42]])
    with torch.no_grad():
        next_logits, next_caches = model(x_next, kv_caches=kv_caches, offset=4)
    assert next_logits.shape == (1, 1, 1000)
    assert next_caches[0][0].shape == (1, 2, 5, 64)


def test_universal_transformer_quantization():
    """Verify INT4 quantization on UniversalTransformer layers."""
    cfg = ModelConfig(
        vocab_size=500,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
    )
    model = UniversalTransformer(cfg).eval()
    model.fuse_qkv()
    tb.quantize_model(model, bits=4, exclude_modules=[], backend="cpu")

    x = torch.randint(0, 500, (1, 1))
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (1, 1, 500)
    assert not torch.isnan(logits).any()


def test_llm_api_local_pretrained():
    """Verify tb.LLM.from_pretrained works on local model checkpoint if available."""
    local_path = r"d:\torchburn\models\qwen_0_5b"
    if not os.path.exists(os.path.join(local_path, "qwen2.5-0.5b-model.safetensors")):
        pytest.skip("Local Qwen weights not present")

    llm = tb.LLM.from_pretrained(local_path, quant="int4", device="cpu")
    resp = llm.generate("Hello world", max_tokens=10, temperature=0.0)
    assert isinstance(resp, str)
    assert len(resp) > 0


def test_wgpu_decoder_igpu():
    """Verify WGPU End-to-End GPU Compute Graph Decoder on iGPU."""
    local_path = r"d:\torchburn\models\qwen_0_5b"
    if not os.path.exists(os.path.join(local_path, "qwen2.5-0.5b-model.safetensors")):
        pytest.skip("Local Qwen weights not present")

    try:
        gpu_info = tb._torchburn.gpu_info()
        if not gpu_info.get("available", False):
            pytest.skip("WGPU device not available")
    except Exception:
        pytest.skip("WGPU not enabled")

    llm = tb.LLM.from_pretrained(local_path, quant="int4", device="igpu")
    assert llm.engine._wgpu_decoder is not None
    resp = llm.generate("The capital of France is", max_tokens=8, temperature=0.0)
    assert isinstance(resp, str)
    assert len(resp) > 0

