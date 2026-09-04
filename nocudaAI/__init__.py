"""NoCudaAI: Zero-CUDA, Pure-CPU Local LLM Assistant powered by TorchBurn."""

from .config import ModelConfig, GenerationConfig, EngineConfig, MODEL_PROFILES
from .model import NoCudaModel, RMSNorm, RotaryEmbedding, CausalSelfAttention, TransformerBlock
from .tokenizer import NoCudaTokenizer
from .engine import NoCudaEngine
from .agent import NoCudaAgent

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "GenerationConfig",
    "EngineConfig",
    "MODEL_PROFILES",
    "NoCudaModel",
    "RMSNorm",
    "RotaryEmbedding",
    "CausalSelfAttention",
    "TransformerBlock",
    "NoCudaTokenizer",
    "NoCudaEngine",
    "NoCudaAgent",
]
