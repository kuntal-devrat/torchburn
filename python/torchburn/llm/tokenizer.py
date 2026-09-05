"""Universal Tokenizer for TorchBurn LLM."""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional, Union


class UniversalTokenizer:
    """Universal Tokenizer wrapping HuggingFace AutoTokenizer or tokenizers.Tokenizer."""

    def __init__(self, tokenizer_obj: Any, eos_token_id: int = 151643, pad_token_id: Optional[int] = None):
        self._tok = tokenizer_obj
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self._chat_template = getattr(tokenizer_obj, "chat_template", None)

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
        repo_clean = model_id_or_path
        if repo_clean in ("qwen", "qwen_0_5b", "qwen2.5-0.5b", "default"):
            repo_clean = "Qwen/Qwen2.5-0.5B-Instruct"
        elif repo_clean in ("deepseek", "deepseek_1_5b", "deepseek-1.5b", "deepseek-r1", "r1"):
            repo_clean = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

        hf_cache_snapshot_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_clean.replace('/', '--')}/snapshots")
        if os.path.isdir(hf_cache_snapshot_dir):
            try:
                snaps = sorted(os.listdir(hf_cache_snapshot_dir))
                if snaps:
                    cand_dirs.append(os.path.join(hf_cache_snapshot_dir, snaps[-1]))
            except Exception:
                pass

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
                    eos_id = 151643
                    if os.path.isfile(cfg_cand):
                        with open(cfg_cand, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                            eos_id = cfg.get("eos_token_id", 151643)
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
        """Encodes string text into token IDs."""
        if hasattr(self._tok, "encode"):
            res = self._tok.encode(text)
            if hasattr(res, "ids"):
                return list(res.ids)
            elif isinstance(res, list):
                return res
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
