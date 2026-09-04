"""Configuration for NoCudaAI: Zero-CUDA, Pure-CPU Local LLM Assistant."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Transformer architecture configuration."""
    vocab_size: int = 50257
    hidden_size: int = 256
    intermediate_size: int = 684
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: Optional[int] = None  # Grouped-Query Attention if != num_attention_heads
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    qkv_bias: bool = False
    dropout: float = 0.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# Pre-configured profiles optimized for CPU inference
MODEL_PROFILES = {
    "pico": ModelConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=344,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    ),
    "micro": ModelConfig(
        vocab_size=8192,
        hidden_size=256,
        intermediate_size=684,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    ),
    "nano": ModelConfig(
        vocab_size=32000,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=6,
        num_key_value_heads=6,
        max_position_embeddings=2048,
    ),
    "smollm_tiny": ModelConfig(
        vocab_size=49152,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=9,
        num_key_value_heads=3,  # GQA 3:1
        max_position_embeddings=2048,
    ),
    "qwen_0_5b": ModelConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,  # GQA 7:1
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        qkv_bias=True,
    ),
    "qwen2_5_0_5b": ModelConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,  # GQA 7:1
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        qkv_bias=True,
    ),
}


@dataclass
class GenerationConfig:
    """Autoregressive text generation parameters."""
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    eos_token_id: Optional[int] = None
    stream: bool = True
    seed: Optional[int] = 42


@dataclass
class EngineConfig:
    """Inference engine configuration."""
    engine: Literal["torchburn", "igpu", "cpu", "eager"] = "torchburn"
    torchburn_engine: Optional[str] = None  # Auto-resolved to burn-wgpu for igpu, native_cpu for cpu
    quantization: Literal["none", "int8", "int4"] = "int8"
    use_static_kv_cache: bool = True
    num_threads: Optional[int] = 4
    warmup_steps: int = 2
    device: str = "auto"

