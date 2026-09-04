"""Tests for Pretrained Qwen 0.5B Model support in NoCudaAI."""

import os
import pytest
import torch
from nocudaAI.config import MODEL_PROFILES, ModelConfig, GenerationConfig, EngineConfig
from nocudaAI.model import NoCudaModel
from nocudaAI.tokenizer import get_tokenizer, PretrainedTokenizerWrapper


def test_qwen_profile():
    """Verify Qwen 0.5B profile hyperparameters match official Qwen2/Qwen2.5-0.5B."""
    assert "qwen_0_5b" in MODEL_PROFILES
    cfg = MODEL_PROFILES["qwen_0_5b"]
    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 896
    assert cfg.intermediate_size == 4864
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 14
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 64
    assert cfg.qkv_bias is True
    assert cfg.tie_word_embeddings is True


def test_qwen_tokenizer():
    """Verify Qwen tokenizer loading and tokenization."""
    tok = get_tokenizer("qwen_0_5b")
    text = "Hello, TorchBurn on Qwen!"
    tokens = tok.encode(text)
    assert len(tokens) > 0
    decoded = tok.decode(tokens)
    assert "TorchBurn" in decoded or "Hello" in decoded


def test_qwen_architecture_dry_run():
    """Verify NoCudaModel with Qwen architecture performs valid forward and decode steps."""
    # Test a mini slice of Qwen (2 layers) to verify shapes and GQA
    mini_qwen_cfg = ModelConfig(
        vocab_size=1000,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        qkv_bias=True,
    )
    model = NoCudaModel(mini_qwen_cfg).eval()

    # Prefill step
    x = torch.randint(0, 1000, (1, 4))
    with torch.no_grad():
        logits, kv_caches = model(x)
    assert logits.shape == (1, 4, 1000)
    assert len(kv_caches) == 2
    # Check GQA KV cache shape: [1, 2, 4, 64]
    k, v = kv_caches[0]
    assert k.shape == (1, 2, 4, 64)

    # Decode step (T=1, offset=4)
    x_next = torch.tensor([[42]])
    with torch.no_grad():
        next_logits, next_caches = model(x_next, kv_caches=kv_caches, offset=4)
    assert next_logits.shape == (1, 1, 1000)
    assert next_caches[0][0].shape == (1, 2, 5, 64)
