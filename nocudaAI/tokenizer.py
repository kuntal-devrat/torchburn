"""Self-contained, zero-dependency Tokenizer for NoCudaAI."""

from __future__ import annotations
from typing import List, Optional, Dict, Union


class NoCudaTokenizer:
    """Byte-fallback BPE/ASCII Tokenizer with agentic control tokens.
    
    Zero external dependencies required. Maps UTF-8 byte sequences
    into token IDs and includes special chat template tokens.
    """

    # Special tokens for conversational agentic interactions
    SPECIAL_TOKENS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool_call|>",
        "<|tool_result|>",
        "<|pad|>",
    ]

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self._special_to_id: Dict[str, int] = {}
        self._id_to_special: Dict[int, str] = {}
        
        # 1. Register special tokens at beginning
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self._special_to_id[token] = idx
            self._id_to_special[idx] = token
            
        self.special_offset = len(self.SPECIAL_TOKENS)
        
        # 2. Byte tokens: 0..255 mapped to special_offset .. special_offset + 255
        self.byte_offset = self.special_offset
        self.eos_token_id = self._special_to_id["<|endoftext|>"]
        self.pad_token_id = self._special_to_id["<|pad|>"]
        
        # 3. Common words / subwords vocabulary extension up to vocab_size
        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        self._init_common_subwords()

    def _init_common_subwords(self):
        """Pre-populate high-frequency English & programming tokens."""
        common = [
            " the", " The", " be", " to", " of", " and", " a", " in", " that", " have",
            " I", " it", " for", " not", " on", " with", " he", " as", " you", " do",
            " at", " this", " but", " his", " by", " from", " they", " we", " say",
            " her", " she", " or", " an", " will", " my", " one", " all", " would",
            " there", " their", " what", " so", " up", " out", " if", " about", " who",
            " get", " which", " go", " me", " when", " make", " can", " like", " time",
            " no", " just", " him", " know", " take", " people", " into", " year", " your",
            " good", " some", " could", " them", " see", " other", " than", " then",
            " now", " look", " only", " come", " its", " over", " think", " also",
            " back", " after", " use", " two", " how", " our", " work", " first",
            " well", " way", " even", " new", " want", " because", " any", " these",
            " give", " day", " most", " us", "\n", "\n\n", "    ", "  ",
            "def ", "class ", "return ", "import ", "from ", "print(", "self.",
            " = ", " == ", " != ", " + ", " - ", " * ", " / ", "->", "()", "[]", "{}",
            "TorchBurn", "NoCuda", "CPU", "CUDA", "PyTorch", "engine", "native_cpu"
        ]
        next_id = self.byte_offset + 256
        for w in common:
            if next_id < self.vocab_size:
                self._word_to_id[w] = next_id
                self._id_to_word[next_id] = w
                next_id += 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text to token IDs."""
        tokens: List[int] = []
        i = 0
        n = len(text)
        
        while i < n:
            # Check for special tokens first
            found_special = False
            for st, sid in self._special_to_id.items():
                if text.startswith(st, i):
                    tokens.append(sid)
                    i += len(st)
                    found_special = True
                    break
            if found_special:
                continue
                
            # Check for word/subword matches (longest first)
            found_word = False
            for w, wid in sorted(self._word_to_id.items(), key=lambda x: -len(x[0])):
                if text.startswith(w, i):
                    tokens.append(wid)
                    i += len(w)
                    found_word = True
                    break
            if found_word:
                continue
                
            # Fallback to UTF-8 bytes
            char = text[i]
            char_bytes = char.encode("utf-8")
            for b in char_bytes:
                tokens.append(self.byte_offset + b)
            i += 1
            
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back to a string."""
        byte_buffer = bytearray()
        result_parts: List[str] = []
        
        def flush_bytes():
            if byte_buffer:
                result_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()
                
        for t in tokens:
            if t in self._id_to_special:
                if not skip_special_tokens:
                    flush_bytes()
                    result_parts.append(self._id_to_special[t])
            elif t in self._id_to_word:
                flush_bytes()
                result_parts.append(self._id_to_word[t])
            elif self.byte_offset <= t < self.byte_offset + 256:
                byte_buffer.append(t - self.byte_offset)
            else:
                flush_bytes()
                result_parts.append(f"<unk:{t}>")
                
        flush_bytes()
        return "".join(result_parts)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        """Formats conversational messages into ChatML format."""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"
        return prompt


class PretrainedTokenizerWrapper:
    """Wraps Hugging Face AutoTokenizer with standard NoCudaTokenizer interface."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.vocab_size = len(tokenizer)
        self.eos_token_id = tokenizer.eos_token_id or 151645
        self.pad_token_id = tokenizer.pad_token_id or self.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(tokens, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        if hasattr(self._tok, "apply_chat_template") and getattr(self._tok, "chat_template", None):
            try:
                return self._tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"
        return prompt


def get_tokenizer(
    model_name_or_path: str = "pico",
    vocab_size: Optional[int] = None
) -> Union[NoCudaTokenizer, PretrainedTokenizerWrapper]:
    """Resolves or instantiates the appropriate tokenizer for a given model or profile."""
    import os
    local_weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    if os.path.isfile(os.path.join(local_weights_dir, "tokenizer.json")):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(local_weights_dir, local_files_only=True)
            return PretrainedTokenizerWrapper(tok)
        except Exception:
            pass

    is_qwen = any(q in model_name_or_path.lower() for q in ("qwen", "0.5b", "0_5b"))
    if is_qwen or (isinstance(model_name_or_path, str) and "/" in model_name_or_path):
        try:
            from transformers import AutoTokenizer
            repo_id = model_name_or_path
            if repo_id in ("qwen", "qwen_0_5b", "qwen2_5_0_5b", "default"):
                repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
            tok = AutoTokenizer.from_pretrained(repo_id)
            return PretrainedTokenizerWrapper(tok)
        except Exception as e:
            print(f"[NoCudaAI] Warning: could not load HF tokenizer for {model_name_or_path} ({e}), falling back to NoCudaTokenizer.")
    return NoCudaTokenizer(vocab_size=vocab_size or 8192)

