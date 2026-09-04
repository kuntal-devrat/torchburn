"""Micro-LLM Architecture for NoCudaAI.

Pure CPU-optimized Transformer featuring:
- RMSNorm with SIMD kernel acceleration
- Rotary Positional Embeddings (RoPE)
- SwiGLU Feed-Forward Network
- Causal Self-Attention with KV-Caching
- TorchBurn graph compile support for both prefill and decode phases
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class StaticKVCache:
    """Pre-allocated contiguous KV-cache buffer for zero-allocation token generation.
    Eliminates all torch.cat operations, memory copies, and heap reallocations.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        self.k = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.seq_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """In-place update without allocating new memory."""
        T = k.shape[2]
        self.k[:, :, offset : offset + T, :] = k
        self.v[:, :, offset : offset + T, :] = v
        self.seq_len = offset + T
        return self.k[:, :, : self.seq_len, :], self.v[:, :, : self.seq_len, :]

    def reset(self):
        self.seq_len = 0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.weight.shape[0],), self.weight, eps=self.eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._register_rope_cache()

    def _register_rope_cache(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        # Pre-replicate [cos, cos] and [sin, sin] as [1, 1, max_seq_len, dim]
        # Eliminates 48 unsqueeze and 48 cat allocations per decode step
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0).unsqueeze(1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0).unsqueeze(1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :, offset : offset + seq_len, :]
        sin = self.sin_cached[:, :, offset : offset + seq_len, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Attention with optional KV-cache."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or self.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_dim = self.num_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.v_dim = self.num_kv_heads * self.head_dim

        bias = getattr(config, "qkv_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.k_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.v_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def fuse_qkv(self):
        """Fuses q_proj, k_proj, and v_proj into a single qkv_proj Linear layer."""
        if hasattr(self, "qkv_proj"):
            return
        total_dim = self.q_dim + self.k_dim + self.v_dim
        has_bias = self.q_proj.bias is not None
        fused = nn.Linear(self.hidden_size, total_dim, bias=has_bias)
        fused.weight.data.copy_(
            torch.cat([self.q_proj.weight.data, self.k_proj.weight.data, self.v_proj.weight.data], dim=0)
        )
        if has_bias:
            fused.bias.data.copy_(
                torch.cat([self.q_proj.bias.data, self.k_proj.bias.data, self.v_proj.bias.data], dim=0)
            )
        self.qkv_proj = fused
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Union[Tuple[torch.Tensor, torch.Tensor], StaticKVCache]] = None,
        causal_mask: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Union[Tuple[torch.Tensor, torch.Tensor], StaticKVCache]]]:
        B, T, C = x.shape

        # Fast path: single token decode with StaticKVCache and quantized TorchBurn kernel
        if (
            T == 1
            and kv_cache is not None
            and isinstance(kv_cache, StaticKVCache)
            and hasattr(self, "qkv_proj")
            and hasattr(self.qkv_proj, "qweight")
            and hasattr(self.o_proj, "qweight")
        ):
            import torchburn
            attn_out = torchburn.fused_attention_step(
                x,
                self.qkv_proj,
                self.o_proj,
                kv_cache.k,
                kv_cache.v,
                cos,
                sin,
                offset=offset,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            kv_cache.seq_len = offset + 1
            return attn_out, kv_cache

        if hasattr(self, "qkv_proj"):
            qkv = self.qkv_proj(x)
            q, k, v = torch.split(qkv, [self.q_dim, self.k_dim, self.v_dim], dim=-1)
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV-cache for autoregressive decoding
        if kv_cache is not None:
            if isinstance(kv_cache, StaticKVCache):
                k, v = kv_cache.update(k, v, offset=offset)
                new_kv_cache = kv_cache
            else:
                prev_k, prev_v = kv_cache
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
                new_kv_cache = (k, v)
        else:
            new_kv_cache = (k, v)

        # Scaled dot-product attention
        if causal_mask is not None:
            if self.num_kv_heads < self.num_heads:
                repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + causal_mask
            probs = F.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
        else:
            enable_gqa = (self.num_kv_heads < self.num_heads)
            if T > 1 and offset > 0:
                # Continuation prefill: query row i attends to keys <= offset + i
                attn_mask = torch.full((T, offset + T), float("-inf"), dtype=q.dtype, device=q.device)
                for row_idx in range(T):
                    attn_mask[row_idx, : offset + row_idx + 1] = 0.0
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
            else:
                is_causal = (T > 1 and offset == 0)
                out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=enable_gqa)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv_cache


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If quantized with TorchBurn, use the zero-allocation fused native kernel
        if hasattr(self.gate_proj, "qweight") and hasattr(self.down_proj, "qweight"):
            import torchburn
            return torchburn.fused_swiglu_mlp(x, self.gate_proj, self.up_proj, self.down_proj)
        # Fused elementwise pattern: silu(gate) * up
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer decoder block with Pre-LayerNorm and SwiGLU."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Union[Tuple[torch.Tensor, torch.Tensor], StaticKVCache]] = None,
        causal_mask: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Union[Tuple[torch.Tensor, torch.Tensor], StaticKVCache]]]:
        # Self-attention branch
        normed = self.input_layernorm(x)
        attn_out, new_cache = self.self_attn(
            normed, cos=cos, sin=sin, kv_cache=kv_cache, causal_mask=causal_mask, offset=offset
        )
        x = x + attn_out

        # MLP branch
        mlp_out = self.mlp(self.post_attention_layernorm(x))
        x = x + mlp_out
        return x, new_cache


class NoCudaModel(nn.Module):
    """Complete Causal Language Model."""

    def __init__(self, config: ModelConfig, init_weights: bool = True):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        """Initializes weights with small normal distributions for stable CPU dynamics."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def create_static_kv_caches(
        self,
        max_batch_size: int = 1,
        max_seq_len: int = 2048,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> List[StaticKVCache]:
        """Creates pre-allocated StaticKVCache buffers for all transformer layers."""
        return [
            StaticKVCache(
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.config.num_key_value_heads or self.config.num_attention_heads,
                head_dim=self.config.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.config.num_hidden_layers)
        ]

    def fuse_qkv(self):
        """Fuses Q, K, V projections across all transformer blocks."""
        import gc
        for layer in self.layers:
            layer.self_attn.fuse_qkv()
        gc.collect()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[Union[List[Tuple[torch.Tensor, torch.Tensor]], List[StaticKVCache]]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Union[List[Tuple[torch.Tensor, torch.Tensor]], List[StaticKVCache]]]:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x, seq_len=T, offset=offset)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, cache = layer(x, cos=cos, sin=sin, kv_cache=layer_cache, offset=offset)
            new_caches.append(cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        weights_path: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> NoCudaModel:
        """Loads a pretrained model (e.g. Qwen2.5-0.5B) directly into NoCudaModel."""
        import os
        import json
        from pathlib import Path
        import safetensors.torch
        from .config import ModelConfig, MODEL_PROFILES

        # 1. Resolve weights file
        local_weights = [
            weights_path,
            model_name_or_path if isinstance(model_name_or_path, str) and model_name_or_path.endswith(".safetensors") else None,
            os.path.join(os.path.dirname(__file__), "weights", "qwen2.5-0.5b-model.safetensors"),
            os.path.join(os.path.dirname(__file__), "weights", "model.safetensors"),
        ]
        resolved_path = None
        for cand in local_weights:
            if cand and os.path.isfile(cand) and os.path.getsize(cand) > 500_000_000:
                resolved_path = cand
                break

        if not resolved_path:
            # Check Hugging Face hub cache
            try:
                from huggingface_hub import hf_hub_download
                repo_id = model_name_or_path
                if repo_id in ("qwen", "qwen_0_5b", "qwen2_5_0_5b", "default"):
                    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
                print(f"[NoCudaAI] Resolving weights for '{repo_id}' from Hugging Face...")
                resolved_path = hf_hub_download(repo_id, "model.safetensors")
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not locate Qwen weights locally or on Hugging Face: {e}. "
                    f"Please run `python -m nocudaAI.main download` or place 'qwen2.5-0.5b-model.safetensors' "
                    f"into '{os.path.join(os.path.dirname(__file__), 'weights')}'"
                )

        print(f"[NoCudaAI] Loading pretrained weights from: {resolved_path}")

        # 2. Resolve Config
        config = MODEL_PROFILES.get("qwen2_5_0_5b")
        cfg_path = os.path.join(os.path.dirname(resolved_path), "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                config = ModelConfig(
                    vocab_size=data.get("vocab_size", config.vocab_size),
                    hidden_size=data.get("hidden_size", config.hidden_size),
                    intermediate_size=data.get("intermediate_size", config.intermediate_size),
                    num_hidden_layers=data.get("num_hidden_layers", config.num_hidden_layers),
                    num_attention_heads=data.get("num_attention_heads", config.num_attention_heads),
                    num_key_value_heads=data.get("num_key_value_heads", config.num_key_value_heads),
                    max_position_embeddings=data.get("max_position_embeddings", config.max_position_embeddings),
                    rms_norm_eps=data.get("rms_norm_eps", config.rms_norm_eps),
                    rope_theta=data.get("rope_theta", data.get("rope_parameters", {}).get("rope_theta", config.rope_theta)),
                    tie_word_embeddings=data.get("tie_word_embeddings", True),
                    qkv_bias=True,
                )
            except Exception:
                pass

        # 3. Instantiate model without redundant random initialization
        model = cls(config, init_weights=False).to(device=device, dtype=dtype)

        # 4. Stream safetensors directly into model parameters to prevent RAM bloat
        import gc
        raw_sd = safetensors.torch.load_file(resolved_path, device="cpu")
        model_params = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())

        for k in list(raw_sd.keys()):
            tensor = raw_sd.pop(k)
            clean_k = k[len("model.") :] if k.startswith("model.") else k
            if clean_k in model_params:
                with torch.no_grad():
                    model_params[clean_k].copy_(tensor.to(dtype=dtype, device=device))
            elif clean_k in model_buffers:
                with torch.no_grad():
                    model_buffers[clean_k].copy_(tensor.to(dtype=dtype, device=device))
            del tensor

        del raw_sd
        del model_params
        del model_buffers

        if config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight

        gc.collect()

        print(f"[NoCudaAI] Successfully loaded {model.get_num_params() / 1e6:.2f}M parameter model ({dtype}).")
        return model.eval()

