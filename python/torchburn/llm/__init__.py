"""TorchBurn LLM: Universal zero-CUDA language model inference engine."""

from .config import ModelConfig, GenerationConfig, EngineConfig
from .model import UniversalTransformer, StaticKVCache
from .tokenizer import UniversalTokenizer
from .loader import ModelLoader, resolve_hf_token
from .engine import UniversalEngine
from .api import LLM
from .cli import main as cli_main
from .speculative import SpeculativeDecoder, BatchedDecoder

__all__ = [
    "LLM",
    "ModelConfig",
    "GenerationConfig",
    "EngineConfig",
    "UniversalTransformer",
    "StaticKVCache",
    "UniversalTokenizer",
    "ModelLoader",
    "UniversalEngine",
    "SpeculativeDecoder",
    "BatchedDecoder",
    "resolve_hf_token",
    "cli_main",
]
