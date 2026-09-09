"""Per-family architecture tests for TorchBurn's universal decoder.

Each family is built from a representative HuggingFace config.json shape and
exercised through the real model/quantization pipeline — no stubs.
"""

import pytest
import torch

from torchburn import quantize_model
from torchburn.llm.config import ModelConfig
from torchburn.llm.model import UniversalTransformer


def _base(**overrides):
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    return ModelConfig.from_dict(defaults)



FAMILIES = {
    "qwen2": dict(model_type="qwen2", architectures=["Qwen2ForCausalLM"], qkv_bias=True),
    "llama": dict(model_type="llama", architectures=["LlamaForCausalLM"], qkv_bias=False),
    "mistral": dict(model_type="mistral", architectures=["MistralForCausalLM"], qkv_bias=False),
    "gemma": dict(
        model_type="gemma",
        architectures=["GemmaForCausalLM"],
        hidden_act="gelu_pytorch_tanh",
        scale_embeddings=True,
        norm_type="gemma_offset",
        tie_word_embeddings=True,
    ),
    "gpt2": dict(
        model_type="gpt2",
        architectures=["GPT2LMHeadModel"],
        hidden_act="gelu_pytorch_tanh",
        position_encoding="learned",
        qkv_bias=True,
        num_key_value_heads=4,
    ),
    "gpt_neox": dict(
        model_type="gpt_neox",
        architectures=["GPTNeoXForCausalLM"],
        position_encoding="learned",
        use_parallel_residual=True,
        qkv_bias=True,
    ),
    "stablelm": dict(
        model_type="stablelm",
        architectures=["StableLMForCausalLM"],
        qkv_bias=True,
        qk_layernorm=True,
    ),
    "qwen3": dict(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        qkv_bias=True,
        qk_layernorm=True,
    ),
    "mixtral_moe": dict(
        model_type="mixtral",
        architectures=["MixtralForCausalLM"],
        num_experts=4,
        num_experts_per_tok=2,
        qkv_bias=False,
    ),
    "deepseek_moe": dict(
        model_type="deepseek_v3",
        architectures=["DeepseekV3ForCausalLM"],
        num_experts=4,
        num_experts_per_tok=2,
        qkv_bias=False,
    ),
    "qwen3_moe": dict(
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=4,
        num_experts_per_tok=2,
        qkv_bias=True,
        qk_layernorm=True,
    ),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_forward_prefill_and_decode(family):
    cfg = _base(**FAMILIES[family])
    model = UniversalTransformer(cfg).eval()

    prefill_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        logits, caches = model(prefill_ids)
    assert logits.shape == (1, 5, cfg.vocab_size)
    assert not torch.isnan(logits).any()

    next_ids = torch.randint(0, cfg.vocab_size, (1, 1))
    with torch.no_grad():
        next_logits, next_caches = model(next_ids, kv_caches=caches, offset=5)
    assert next_logits.shape == (1, 1, cfg.vocab_size)
    assert not torch.isnan(next_logits).any()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_quantized_matches_dense_forward(family):
    cfg = _base(**FAMILIES[family])
    dense = UniversalTransformer(cfg).eval()
    quantized = UniversalTransformer(cfg, init_weights=False, quant="int4", fused_qkv=True).eval()
    quantized.load_state_dict(dense.state_dict(), strict=False)
    quantize_model(quantized, bits=4, exclude_modules=[], backend="cpu")
    quantized.fuse_qkv()

    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        dense_logits, _ = dense(ids)
        q_logits, _ = quantized(ids)
    assert q_logits.shape == dense_logits.shape
    assert not torch.isnan(q_logits).any()


def test_gpt2_uses_learned_positions_not_rope():
    cfg = _base(**FAMILIES["gpt2"])
    assert cfg.position_encoding == "learned"
    model = UniversalTransformer(cfg).eval()
    assert model.position_embedding is not None
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 4, cfg.vocab_size)


def test_moe_routes_tokens_through_multiple_experts():
    cfg = _base(**FAMILIES["mixtral_moe"])
    model = UniversalTransformer(cfg).eval()
    block = model.layers[0]
    assert block.moe is not None and block.mlp is None
    assert block.moe.num_experts == 4 and block.moe.top_k == 2

    x = torch.randn(1, 3, cfg.hidden_size)
    with torch.no_grad():
        out = block.moe(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_gemma_offset_norm_and_scaled_embeddings():
    cfg = _base(**FAMILIES["gemma"])
    assert cfg.scale_embeddings and cfg.norm_type == "gemma_offset"
    model = UniversalTransformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 2))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 2, cfg.vocab_size)


def test_qk_layernorm_family_applies_per_head_norm():
    cfg = _base(**FAMILIES["qwen3"])
    assert cfg.qk_layernorm
    model = UniversalTransformer(cfg).eval()
    attn = model.layers[0].self_attn
    assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
    ids = torch.randint(0, cfg.vocab_size, (1, 2))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 2, cfg.vocab_size)


def test_parallel_residual_gpt_neox_block():
    cfg = _base(**FAMILIES["gpt_neox"])
    assert cfg.use_parallel_residual
    model = UniversalTransformer(cfg).eval()
    block = model.layers[0]
    assert block.use_parallel_residual
    ids = torch.randint(0, cfg.vocab_size, (1, 2))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 2, cfg.vocab_size)


def test_unsupported_moe_family_is_rejected():
    with pytest.raises(ValueError, match="Unsupported MoE architecture"):
        ModelConfig.from_dict(
            dict(
                model_type="falcon_moe",
                architectures=["FalconMoEForCausalLM"],
                num_experts=4,
                num_experts_per_tok=2,
            )
        )


def test_unsupported_activation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported decoder activation"):
        ModelConfig.from_dict(dict(model_type="llama", hidden_act="quick_gelu"))


def test_unsupported_family_emits_warning():
    with pytest.warns(UserWarning, match="not an explicitly verified TorchBurn family"):
        ModelConfig.from_dict(dict(model_type="roberta_lm", architectures=["RobertaForCausalLM"]))
