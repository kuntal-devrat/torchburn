"""Universal Transformer Architecture for TorchBurn LLM."""

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class StaticKVCache:
    """Pre-allocated contiguous KV-cache buffer for zero-allocation token generation."""

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

    def __init__(self, dim: int, eps: float = 1e-6):
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

    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class UniversalAttention(nn.Module):
    """Universal Multi-Head / Grouped-Query Attention with optional QKV fusion."""

    def __init__(self, config: ModelConfig, quant: Optional[str] = None, fused_qkv: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or self.num_heads
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_dim = self.num_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.v_dim = self.num_kv_heads * self.head_dim

        bias = config.qkv_bias
        if quant in ("int4", "int8") and fused_qkv:
            from torchburn.quantization import QuantizedLinear
            bits = 4 if quant == "int4" else 8
            total_dim = self.q_dim + self.k_dim + self.v_dim
            self.qkv_proj = QuantizedLinear(self.hidden_size, total_dim, bias=bias, bits=bits, group_size=64)
            self.o_proj = QuantizedLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, bits=bits, group_size=64)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=bias)
            self.k_proj = nn.Linear(self.hidden_size, self.k_dim, bias=bias)
            self.v_proj = nn.Linear(self.hidden_size, self.v_dim, bias=bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


    def fuse_qkv(self):
        """Fuses q_proj, k_proj, and v_proj into a single qkv_proj layer."""
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
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Union[Tuple[torch.Tensor, torch.Tensor], StaticKVCache]]]:
        B, T, C = x.shape

        # Fast path: single token decode with StaticKVCache and quantized kernel
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

        # KV-cache update
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
        out = self.o_proj(out)
        return out, new_kv_cache


class UniversalMLP(nn.Module):
    """Universal SwiGLU / MLP feed-forward network."""

    def __init__(self, config: ModelConfig, quant: Optional[str] = None):
        super().__init__()
        self.config = config
        if quant in ("int4", "int8"):
            from torchburn.quantization import QuantizedLinear
            bits = 4 if quant == "int4" else 8
            self.gate_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False, bits=bits, group_size=64)
            self.up_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False, bits=bits, group_size=64)
            self.down_proj = QuantizedLinear(config.intermediate_size, config.hidden_size, bias=False, bits=bits, group_size=64)
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused SwiGLU check if quantized
        if (
            x.shape[1] == 1
            and hasattr(self.gate_proj, "qweight")
            and hasattr(self.up_proj, "qweight")
            and hasattr(self.down_proj, "qweight")
            and getattr(self.gate_proj, "backend", "cpu") != "igpu"
        ):
            import torchburn
            return torchburn.fused_swiglu_mlp(x, self.gate_proj, self.up_proj, self.down_proj)

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class UniversalTransformerBlock(nn.Module):
    """Transformer decoder block with pre-normalization and residual connections."""

    def __init__(self, config: ModelConfig, quant: Optional[str] = None, fused_qkv: bool = False):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = UniversalAttention(config, quant=quant, fused_qkv=fused_qkv)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = UniversalMLP(config, quant=quant)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Union[Tuple[torch.Tensor, torch.Tensor], StaticKVCache]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Union[Tuple[torch.Tensor, torch.Tensor], StaticKVCache]]]:
        residual = x
        normed = self.input_layernorm(x)
        attn_out, new_cache = self.self_attn(normed, cos=cos, sin=sin, kv_cache=kv_cache, offset=offset)
        x = residual + attn_out

        residual = x
        normed = self.post_attention_layernorm(x)
        mlp_out = self.mlp(normed)
        x = residual + mlp_out

        return x, new_cache


class UniversalTransformer(nn.Module):
    """Universal autoregressive language model adaptable to any Transformer architecture."""

    def __init__(
        self,
        config: ModelConfig,
        init_weights: bool = True,
        quant: Optional[str] = None,
        fused_qkv: bool = False,
    ):
        super().__init__()
        self.config = config
        self.quant = quant
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim or (config.hidden_size // config.num_attention_heads),
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            UniversalTransformerBlock(config, quant=quant, fused_qkv=fused_qkv)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if quant in ("int4", "int8"):
            from torchburn.quantization import QuantizedLinear
            bits = 4 if quant == "int4" else 8
            self.lm_head = QuantizedLinear(config.hidden_size, config.vocab_size, bias=False, bits=bits, group_size=64)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            if quant is None:
                self.lm_head.weight = self.embed_tokens.weight

        if init_weights and quant is None:
            self.apply(self._init_weights)


    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def create_static_kv_caches(
        self,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> List[StaticKVCache]:
        """Creates pre-allocated static KV-caches for all layers."""
        return [
            StaticKVCache(
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.config.num_key_value_heads or self.config.num_attention_heads,
                head_dim=self.config.head_dim or (self.config.hidden_size // self.config.num_attention_heads),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.config.num_hidden_layers)
        ]

    def fuse_qkv(self):
        """Fuses Q, K, V projections across all transformer blocks."""
        for layer in self.layers:
            layer.self_attn.fuse_qkv()
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
