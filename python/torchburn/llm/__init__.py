"""TorchBurn LLM: Universal zero-CUDA language model inference engine."""

from .api import LLM
from .cli import main as cli_main
from .config import EngineConfig, GenerationConfig, ModelConfig
from .engine import UniversalEngine
from .loader import ModelLoader, resolve_hf_token
from .model import StaticKVCache, UniversalTransformer
from .speculative import BatchedDecoder, SpeculativeDecoder
from .tokenizer import UniversalTokenizer

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
