"""Configuration classes for TorchBurn LLM Inference Engine."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class ModelConfig:
    """Universal architecture configuration for Transformer LLMs."""

    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: Optional[int] = 2
    head_dim: Optional[int] = None
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    qkv_bias: bool = True
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    architectures: List[str] = field(default_factory=lambda: ["Qwen2ForCausalLM"])

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Constructs ModelConfig dynamically from a HuggingFace config.json."""
        hidden_size = data.get("hidden_size", 896)
        num_heads = data.get("num_attention_heads", 14)
        head_dim = data.get("head_dim", hidden_size // max(num_heads, 1))

        # Check rope parameters
        rope_theta = data.get("rope_theta")
        if rope_theta is None and "rope_parameters" in data:
            rope_theta = data["rope_parameters"].get("rope_theta")
        if rope_theta is None:
            rope_theta = 10000.0

        # Architecture-specific defaults
        arch = data.get("architectures", [""])[0] if data.get("architectures") else ""
        is_qwen = "qwen" in arch.lower() or "qwen" in str(data.get("model_type", "")).lower()

        qkv_bias = data.get("qkv_bias", data.get("attention_bias", is_qwen))

        return cls(
            vocab_size=data.get("vocab_size", 151936),
            hidden_size=hidden_size,
            intermediate_size=data.get("intermediate_size", hidden_size * 4),
            num_hidden_layers=data.get("num_hidden_layers", 24),
            num_attention_heads=num_heads,
            num_key_value_heads=data.get("num_key_value_heads", data.get("num_kv_heads", num_heads)),
            head_dim=head_dim,
            max_position_embeddings=data.get("max_position_embeddings", 32768),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            qkv_bias=bool(qkv_bias),
            tie_word_embeddings=data.get("tie_word_embeddings", is_qwen),
            hidden_act=data.get("hidden_act", "silu"),
            architectures=data.get("architectures", []),
        )


@dataclass
class GenerationConfig:
    """Hyperparameters governing text generation and sampling."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.05
    seed: Optional[int] = 42
    eos_token_id: Optional[int] = None
    stop_tokens: List[str] = field(default_factory=list)


@dataclass
class EngineConfig:
    """Settings controlling backend execution and device dispatch."""

    device: str = "auto"  # "auto", "cpu", "gpu", "igpu"
    quantization: str = "int4"  # "int4", "int8", "none"
    num_threads: Optional[int] = None
    use_static_kv_cache: bool = True
    torchburn_engine: Optional[str] = None
