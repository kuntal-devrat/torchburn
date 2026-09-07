"""Universal Tokenizer for TorchBurn LLM."""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional, Union

from ._registry import fallback_eos_id, repo_snapshot_dir, resolve_repo_id


class UniversalTokenizer:
    """Universal Tokenizer wrapping HuggingFace AutoTokenizer or tokenizers.Tokenizer."""

    def __init__(self, tokenizer_obj: Any, eos_token_id: int = 151643, pad_token_id: Optional[int] = None):
        self._tok = tokenizer_obj
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self._chat_template = getattr(tokenizer_obj, "chat_template", None)
        # Cache whether encode() supports add_special_tokens (checked once)
        self._encode_supports_special_tokens: Optional[bool] = None
        try:
            import inspect
            params = inspect.signature(tokenizer_obj.encode).parameters
            self._encode_supports_special_tokens = "add_special_tokens" in params
        except Exception:
            self._encode_supports_special_tokens = False

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> UniversalTokenizer:
        """Loads a tokenizer from local directory or HuggingFace Hub."""
        # 1. Fast path: check for local tokenizer files in model_id_or_path or HF cache
        cand_dirs = []
        if os.path.isdir(model_id_or_path):
            cand_dirs.append(model_id_or_path)

        # Check HuggingFace cache snapshots
        hf_cache_snapshot_dir = repo_snapshot_dir(resolve_repo_id(model_id_or_path))
        if os.path.isdir(hf_cache_snapshot_dir):
            cand_dirs.append(hf_cache_snapshot_dir)

        for cand_dir in cand_dirs:
            tok_json = os.path.join(cand_dir, "tokenizer.json")
            if os.path.isfile(tok_json):
                # Try transformers AutoTokenizer locally first for full chat template support
                try:
                    from transformers import AutoTokenizer
                    hf_tok = AutoTokenizer.from_pretrained(cand_dir, local_files_only=True, trust_remote_code=True)
                    eos_id = hf_tok.eos_token_id or 151643
                    pad_id = hf_tok.pad_token_id or eos_id
                    return cls(hf_tok, eos_token_id=eos_id, pad_token_id=pad_id)
                except Exception:
                    pass

                # Fallback to fast tokenizers.Tokenizer
                try:
                    from tokenizers import Tokenizer
                    tok_fast = Tokenizer.from_file(tok_json)
                    cfg_cand = os.path.join(cand_dir, "tokenizer_config.json")
                    eos_id = fallback_eos_id(cand_dir)
                    if os.path.isfile(cfg_cand):
                        with open(cfg_cand, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                            eos_token = cfg.get("eos_token", cfg.get("eos_token_id", fallback_eos_id(cand_dir)))
                            if isinstance(eos_token, dict):
                                eos_token = eos_token.get("id", eos_token)
                            try:
                                eos_id = int(eos_token)
                            except (TypeError, ValueError):
                                eos_id = fallback_eos_id(cand_dir)
                    return cls(tok_fast, eos_token_id=eos_id)
                except Exception:
                    pass

        # 2. Online HuggingFace Hub resolution if not found locally
        try:
            from transformers import AutoTokenizer
            auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            hf_tok = AutoTokenizer.from_pretrained(
                model_id_or_path,
                token=auth_token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            eos_id = hf_tok.eos_token_id or 151643
            pad_id = hf_tok.pad_token_id or eos_id
            return cls(hf_tok, eos_token_id=eos_id, pad_token_id=pad_id)
        except Exception:
            pass


        # 3. Fallback: HuggingFace hub download of tokenizer.json
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            tok_json = hf_hub_download(model_id_or_path, "tokenizer.json", token=auth_token, cache_dir=cache_dir)
            tok_fast = Tokenizer.from_file(tok_json)
            return cls(tok_fast, eos_token_id=151643)
        except Exception as e:
            raise RuntimeError(f"Unable to load tokenizer for '{model_id_or_path}': {e}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string text into token IDs.

        ``add_special_tokens`` is honored when the underlying tokenizer supports
        it (transformers / tokenizers both expose the flag on ``encode``).
        """
        try:
            enc = self._tok.encode
            if self._encode_supports_special_tokens:
                res = enc(text, add_special_tokens=add_special_tokens)
            else:
                res = enc(text)
            if hasattr(res, "ids"):
                return list(res.ids)
            elif isinstance(res, list):
                return res
        except Exception:
            return []
        return []

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs into string text."""
        if hasattr(self._tok, "decode"):
            return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return ""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Applies chat template or falls back to ChatML standard."""
        if hasattr(self._tok, "apply_chat_template"):
            try:
                return self._tok.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

        # Standard ChatML fallback (<|im_start|>role\ncontent<|im_end|>)
        prompt_parts = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)
