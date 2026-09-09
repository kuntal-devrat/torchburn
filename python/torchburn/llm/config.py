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
    norm_type: str = "standard"  # "standard" or "gemma_offset" (1.0 + weight)
    scale_embeddings: bool = False  # True for Gemma (scales embeddings by sqrt(hidden_size))
    architectures: List[str] = field(default_factory=lambda: ["Qwen2ForCausalLM"])
    # Mixture-of-Experts support. num_experts == 0 means dense.
    num_experts: int = 0
    num_experts_per_tok: int = 0
    # Positional encoding flavor: "rope" or "learned" (GPT-2 style).
    position_encoding: str = "rope"
    use_parallel_residual: bool = False  # GPT-NeoX style parallel attention+MLP
    qk_layernorm: bool = False  # Qwen3 / DeepSeek-V3 style per-head Q/K norm

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
        model_type = str(data.get("model_type", "")).lower()
        arch_lower = arch.lower()

        # Check if family is recognized
        known_family_keywords = [
            "qwen", "llama", "mistral", "gemma", "gpt2", "gpt_neox", "gpt-neox", "pythia",
            "stablelm", "phi", "starcoder", "mixtral", "deepseek", "olmoe", "grok"
        ]
        is_known = any(k in model_type or k in arch_lower for k in known_family_keywords)
        if (model_type or arch) and not is_known:
            import warnings
            warnings.warn(
                f"Model architecture '{arch or model_type}' is not an explicitly verified TorchBurn family. "
                "Verified families include: Qwen, Llama, Mistral, Gemma, GPT-2, GPT-NeoX, StableLM, Mixtral, DeepSeek. "
                "Attempting execution via UniversalTransformer, but unsupported layers may require PyTorch eager fallback.",
                UserWarning,
                stacklevel=2,
            )

        is_qwen = "qwen" in arch_lower or "qwen" in model_type
        is_gemma = "gemma" in arch_lower or "gemma" in model_type
        is_gpt2 = "gpt2" in arch_lower or "gpt2" in model_type
        is_gpt_neox = "gpt_neox" in arch_lower or "gpt-neox" in model_type or "gpt_neox" in model_type
        is_moe = bool(data.get("num_experts", 0) or data.get("n_routed_experts", 0) or data.get("num_local_experts", 0))
        if is_moe:
            num_experts = int(data.get("num_experts") or data.get("n_routed_experts") or data.get("num_local_experts"))
            num_experts_per_tok = int(data.get("num_experts_per_tok") or data.get("num_experts_per_token") or 2)
            supported_moe_types = {"mixtral", "deepseek_v2", "deepseek_v3", "qwen3_moe", "olmoe", "grok1"}
            if model_type not in supported_moe_types:
                raise ValueError(
                    f"Unsupported MoE architecture '{model_type}'. Supported: "
                    + ", ".join(sorted(supported_moe_types))
                )
        else:
            num_experts = 0
            num_experts_per_tok = 0

        norm_type = "gemma_offset" if is_gemma else data.get("norm_type", "standard")
        scale_embeddings = is_gemma or data.get("scale_embeddings", False)
        qkv_bias = data.get("qkv_bias", data.get("attention_bias", is_qwen))
        hidden_act = str(data.get("hidden_act", "gelu_pytorch_tanh" if (is_gemma or is_gpt2) else "silu"))
        if hidden_act.lower() not in {
            "silu", "swish", "gelu", "gelu_new", "gelu_pytorch_tanh", "relu", "relu2"
        }:
            raise ValueError(
                f"Unsupported decoder activation '{hidden_act}'. "
                "Supported activations are SiLU, GELU, ReLU, and ReLU2."
            )

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
            tie_word_embeddings=data.get("tie_word_embeddings", is_qwen or is_gemma),
            hidden_act=hidden_act,
            norm_type=norm_type,
            scale_embeddings=scale_embeddings,
            architectures=data.get("architectures", []),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            position_encoding="learned" if (is_gpt2 or is_gpt_neox) else "rope",
            use_parallel_residual=bool(data.get("use_parallel_residual", is_gpt_neox)),
            qk_layernorm=bool(data.get("qk_layernorm", False)),
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

    # "auto" (cuda>metal>igpu>cpu), "cpu", "igpu", "dgpu", "gpu",
    # "cuda" (needs torchburn-cuda), "metal" (macOS only).
    device: str = "auto"
    quantization: str = "int4"  # "int4", "int8", "none"
    num_threads: Optional[int] = None
    use_static_kv_cache: bool = True
    torchburn_engine: Optional[str] = None
