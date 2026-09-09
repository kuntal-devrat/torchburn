"""Universal Transformer Architecture for TorchBurn LLM."""

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class LearnedPositionEmbedding(nn.Module):
    """GPT-2 style learned absolute position embeddings."""

    def __init__(self, max_seq_len: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(max_seq_len, hidden_size))

    def forward(self, seq_len: int, offset: int, device: torch.device) -> torch.Tensor:
        return self.weight[offset : offset + seq_len].to(device)


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
        """In-place update without allocating new memory.

        Uses ring-buffer indexing when offset >= max_seq_len so long contexts
        (>max_seq_len tokens) wrap around instead of OOM-ing.  The attention
        mask in the forward pass will see the ring-wrapped view, which is
        semantically equivalent to sliding-window KV reuse for decode.
        """
        T = k.shape[2]
        # Ring-buffer: write slots wrap around max_seq_len so we never OOM.
        slot = offset % self.k.shape[2]  # self.k.shape[2] == max_seq_len
        # For the common single-token decode case (T==1) this is a simple
        # indexed write.  For multi-token prefill that wraps the boundary we
        # fall back to a sequential loop over individual slots.
        if slot + T <= self.k.shape[2]:
            self.k[:, :, slot : slot + T, :] = k
            self.v[:, :, slot : slot + T, :] = v
        else:
            # Wrap-around: write in two parts
            first = self.k.shape[2] - slot
            self.k[:, :, slot:, :] = k[:, :, :first, :]
            self.k[:, :, :T - first, :] = k[:, :, first:, :]
            self.v[:, :, slot:, :] = v[:, :, :first, :]
            self.v[:, :, :T - first, :] = v[:, :, first:, :]
        self.seq_len = min(offset + T, self.k.shape[2])
        return self.k[:, :, : self.seq_len, :], self.v[:, :, : self.seq_len, :]

    def reset(self):
        self.seq_len = 0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with optional Gemma offset."""

    def __init__(self, dim: int, eps: float = 1e-6, norm_type: str = "standard"):
        super().__init__()
        self.eps = eps
        self.norm_type = norm_type
        # Gemma models initialize weight to 0.0 with formula (1.0 + weight) * x_norm
        # Standard models initialize weight to 1.0 with formula weight * x_norm
        init_val = 0.0 if norm_type == "gemma_offset" else 1.0
        self.weight = nn.Parameter(torch.full((dim,), init_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        if self.norm_type == "gemma_offset":
            return x_norm * (1.0 + self.weight)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.weight.shape[0],), self.weight, eps=self.eps)
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
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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

        if self.qk_layernorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE (skipped for learned absolute positions)
        if cos is not None and sin is not None:
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
        # expand k/v head-by-head when num_kv_heads does not divide num_heads
        # (e.g. GPT-2 variants where head_dim covers the full hidden size).
        if self.num_kv_heads != self.num_heads and self.num_heads % self.num_kv_heads != 0:
            reps = self.num_heads // self.num_kv_heads
            extra = self.num_heads - self.num_kv_heads * reps
            k = torch.cat([k] * reps + [k[:, :extra]], dim=1)
            v = torch.cat([v] * reps + [v[:, :extra]], dim=1)
            enable_gqa = False
        else:
            enable_gqa = (self.num_kv_heads < self.num_heads)
        if T > 1 and offset > 0:
            # Continuation prefill: query row i attends to keys <= offset + i.
            # Vectorized causal mask (bool => True means "attend"), no python loop.
            cols = torch.arange(offset + T, device=q.device)
            row_bounds = torch.arange(T, device=q.device) + offset
            attn_mask = cols.unsqueeze(0) <= row_bounds.unsqueeze(1)  # (T, offset+T)
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
        activation = str(self.config.hidden_act).lower()
        is_silu = activation in ("silu", "swish")

        # Fused SwiGLU check if quantized. The native kernel is only valid for
        # SiLU-gated MLPs; other supported activations use the reference path.
        if (
            x.shape[1] == 1
            and hasattr(self.gate_proj, "qweight")
            and hasattr(self.up_proj, "qweight")
            and hasattr(self.down_proj, "qweight")
            and getattr(self.gate_proj, "backend", "cpu") != "igpu"
        ):
            import torchburn
            return torchburn.fused_swiglu_mlp(x, self.gate_proj, self.up_proj, self.down_proj)

        # Batched prefill path: use the Rust GEMM kernel when seq_len > 1 and
        # the weights are quantized INT4 on the CPU backend.
        if (
            is_silu
            and x.shape[1] > 1
            and hasattr(self.gate_proj, "qweight")
            and hasattr(self.up_proj, "qweight")
            and hasattr(self.down_proj, "qweight")
            and getattr(self.gate_proj, "backend", "cpu") == "cpu"
        ):
            import torchburn
            try:
                return torchburn.fused_swiglu_mlp_batched(
                    x, self.gate_proj, self.up_proj, self.down_proj
                )
            except AttributeError:
                pass  # older build without fused_swiglu_mlp_batched — fall through

        if is_silu:
            activated = F.silu(self.gate_proj(x))
        elif activation in ("gelu_new", "gelu_pytorch_tanh"):
            activated = F.gelu(self.gate_proj(x), approximate="tanh")
        elif activation == "gelu":
            activated = F.gelu(self.gate_proj(x))
        elif activation == "relu":
            activated = F.relu(self.gate_proj(x))
        elif activation == "relu2":
            activated = F.relu(self.gate_proj(x)).square()
        else:
            raise ValueError(f"Unsupported decoder activation '{self.config.hidden_act}'")
        return self.down_proj(activated * self.up_proj(x))


class UniversalMoE(nn.Module):
    """Token-choice Mixture-of-Experts layer (Mixtral / DeepSeek / Qwen3-MoE style)."""

    def __init__(self, config: ModelConfig, quant: Optional[str] = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            UniversalMLP(config, quant=quant) for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        scores = F.softmax(self.gate(flat), dim=-1)
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            token_mask, slot = torch.where(topk_idx == expert_idx)
            if token_mask.numel() == 0:
                continue
            expert_in = flat[token_mask]
            expert_out = expert(expert_in.unsqueeze(1)).squeeze(1)
            out.index_add_(0, token_mask, expert_out * topk_scores[token_mask, slot].unsqueeze(-1))
        return out.reshape(B, T, C)


class UniversalTransformerBlock(nn.Module):
    """Transformer decoder block with pre-normalization and residual connections."""

    def __init__(self, config: ModelConfig, quant: Optional[str] = None, fused_qkv: bool = False):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_type=config.norm_type)
        self.self_attn = UniversalAttention(config, quant=quant, fused_qkv=fused_qkv)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_type=config.norm_type)
        if config.num_experts > 0:
            self.moe = UniversalMoE(config, quant=quant)
            self.mlp = None
        else:
            self.moe = None
            self.mlp = UniversalMLP(config, quant=quant)
        self.use_parallel_residual = config.use_parallel_residual

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
        if self.use_parallel_residual:
            mlp_out = (self.moe if self.moe is not None else self.mlp)(normed)
            x = residual + attn_out + mlp_out
            return x, new_cache
        x = residual + attn_out

        residual = x
        normed = self.post_attention_layernorm(x)
        mlp_out = (self.moe if self.moe is not None else self.mlp)(normed)
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
        self.layers = nn.ModuleList([
            UniversalTransformerBlock(config, quant=quant, fused_qkv=fused_qkv)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, norm_type=config.norm_type)
        self.position_encoding = config.position_encoding
        if self.position_encoding == "learned":
            self.position_embedding = LearnedPositionEmbedding(
                config.max_position_embeddings, config.hidden_size
            )
        else:
            self.position_embedding = None
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim or (config.hidden_size // config.num_attention_heads),
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        if quant in ("int4", "int8"):
            from torchburn.quantization import QuantizedLinear
            bits = 4 if quant == "int4" else 8
            self.lm_head = QuantizedLinear(config.hidden_size, config.vocab_size, bias=False, bits=bits, group_size=64)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            if quant is None:
                self.lm_head.weight = self.embed_tokens.weight

        if init_weights:
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
        if self.config.scale_embeddings:
            x = x * math.sqrt(self.config.hidden_size)
        if self.position_encoding == "learned":
            x = x + self.position_embedding(T, offset, x.device)
            cos = sin = None
        else:
            cos, sin = self.rotary_emb(x, seq_len=T, offset=offset)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, cache = layer(x, cos=cos, sin=sin, kv_cache=layer_cache, offset=offset)
            new_caches.append(cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches
