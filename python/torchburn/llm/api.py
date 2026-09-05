"""Top-level LLM API for TorchBurn."""

from __future__ import annotations
import os
import sys
from typing import Optional, Union, Generator, List, Dict, Any
import torch

from .config import ModelConfig, GenerationConfig, EngineConfig
from .model import UniversalTransformer
from .tokenizer import UniversalTokenizer
from .loader import ModelLoader
from .engine import UniversalEngine


class LLM:
    """Universal, zero-CUDA LLM inference interface.
    
    Usage::
        import torchburn as tb

        # Load from Hugging Face or local path in 1 line:
        model = tb.LLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", quant="int4")

        # Generate in 1 line:
        print(model.generate("Explain black holes in two sentences."))

        # Stream in 3 lines:
        for chunk in model.stream("Write a short poem:"):
            print(chunk, end="", flush=True)

        # Interactive chat session:
        model.chat()
    """

    def __init__(
        self,
        model: UniversalTransformer,
        tokenizer: UniversalTokenizer,
        config: Optional[EngineConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.engine_config = config or EngineConfig()
        self.engine = UniversalEngine(self.model, self.tokenizer, self.engine_config)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        quant: str = "int4",
        device: str = "auto",
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_threads: Optional[int] = None,
        local_files_only: bool = False,
    ) -> LLM:
        """Loads any model from Hugging Face Hub, local trained directory, or weight file.
        
        Args:
            model_id_or_path: Hugging Face repo ID or local checkpoint path.
            quant: Quantization precision ('int4', 'int8', or 'none'). Default: 'int4'.
            device: Target execution device ('auto', 'cpu', 'igpu', or 'gpu'). Default: 'auto'.
            token: Hugging Face authentication token (required for gated models like Llama 3 / Gemma 2).
            cache_dir: Custom directory for caching downloaded weights.
            num_threads: Number of CPU threads to utilize.
            local_files_only: Whether to exclusively look for local files without internet access.
        """
        engine_cfg = EngineConfig(device=device, quantization=quant, num_threads=num_threads)
        model, config, root_path = ModelLoader.load(
            model_id_or_path,
            quant=quant,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        tok_cand = root_path if (os.path.exists(root_path) and os.path.isfile(os.path.join(root_path, "tokenizer.json"))) else model_id_or_path
        tokenizer = UniversalTokenizer.from_pretrained(
            tok_cand,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls(model, tokenizer, engine_cfg)


    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        tokenizer: Any,
        quant: str = "int4",
        device: str = "auto",
        num_threads: Optional[int] = None,
    ) -> LLM:
        """Constructs an LLM directly from an in-memory PyTorch nn.Module (e.g. locally trained)."""
        engine_cfg = EngineConfig(device=device, quantization=quant, num_threads=num_threads)
        if not isinstance(tokenizer, UniversalTokenizer):
            tokenizer = UniversalTokenizer(tokenizer)
        return cls(model, tokenizer, engine_cfg)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = 42,
    ) -> str:
        """Generates text completion for the given prompt."""
        cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        return self.engine.generate(prompt, cfg)

    def stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = 42,
    ) -> Generator[str, None, None]:
        """Streams generated text tokens one by one."""
        cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        for pkt in self.engine.generate_stream(prompt, cfg):
            if pkt["type"] == "token":
                yield pkt["text"]

    def chat(
        self,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ):
        """Launches an interactive multi-turn terminal chat session with persistent memory."""
        history: List[Dict[str, str]] = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        cached_kv = None
        cached_tokens: List[int] = []

        cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        print("\n\033[92m=== TorchBurn LLM Interactive Chat Started ===\033[0m")
        print(f"Device: {self.engine_config.device} | Quantization: {self.engine_config.quantization}")
        print("Type 'exit', 'quit', or 'q' to end.\n")

        while True:
            try:
                user_msg = input("\033[1mYou:\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\033[93mChat session closed.\033[0m")
                break

            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", "q"):
                print("\033[93mGoodbye!\033[0m")
                break

            history.append({"role": "user", "content": user_msg})
            formatted_prompt = self.tokenizer.apply_chat_template(history, add_generation_prompt=True)

            print("\033[96m\033[1mAI\033[0m: ", end="", flush=True)

            collected = []
            summary = None
            in_think = False
            for pkt in self.engine.generate_stream(
                formatted_prompt,
                cfg,
                kv_caches=cached_kv,
                cached_token_ids=cached_tokens,
            ):
                if pkt["type"] == "token":
                    piece = pkt["text"]
                    collected.append(piece)
                    if "<think>" in piece:
                        in_think = True
                        print("\033[2m\033[3m[Thinking: ", end="", flush=True)
                        piece = piece.replace("<think>", "")
                    if "</think>" in piece:
                        in_think = False
                        piece = piece.replace("</think>", "")
                        print("]\033[0m\n\033[96m\033[1mAI\033[0m: ", end="", flush=True)

                    if not piece.startswith("<|"):
                        print(piece, end="", flush=True)
                elif pkt["type"] == "summary":
                    summary = pkt
                    cached_kv = pkt.get("kv_caches")
                    cached_tokens = pkt.get("all_token_ids", [])

            if in_think:
                print("]\033[0m", end="")
            print()
            if summary:
                tok_sec = summary.get("decode_tok_sec", 0)
                n_tok = summary.get("tokens_generated", 0)
                avg_ms = summary.get("avg_ms_per_token", 0)
                print(f"\033[2m⚡ [{n_tok} tokens | {avg_ms:.1f}ms/tok | {tok_sec:.1f} tok/s]\033[0m\n")

            response_text = "".join(collected).strip()
            history.append({"role": "assistant", "content": response_text})
