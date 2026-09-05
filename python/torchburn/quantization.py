"""TorchBurn Native Quantization Suite (INT8 and Grouped INT4).

Provides hardware-accelerated SIMD W8A32 (Weight INT8, Activation FP32)
and W4A32 (Grouped Weight INT4, Activation FP32) linear projections,
fused SwiGLU MLP kernels, drop-in `QuantizedLinear` modules, and full model
quantization utilities.
"""

from __future__ import annotations
import gc
import os
import math
from typing import Optional, Tuple, Union, List, Set
import torch
import torch.nn as nn

from . import _torchburn as _native


def quantize_weight_int8(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D weight matrix (N, K) to symmetric INT8 with per-channel scale (N,).
    
    Returns:
        w_q: torch.Tensor of dtype torch.int8 (1-byte storage), shape (N, K)
        scales: torch.Tensor of dtype torch.float32, shape (N,)
    """
    assert w.ndim == 2, f"Expected 2D weight, got {w.shape}"
    w_cap = torch.to_dlpack(w.detach().float().contiguous())
    qw_cap, qs_cap = _native.quantize_linear_int8(w_cap)
    return torch.from_dlpack(qw_cap), torch.from_dlpack(qs_cap)


def quantize_weight_int4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D weight matrix (N, K) to symmetric INT4 packed into bytes with per-channel scale (N,)."""
    assert w.ndim == 2, f"Expected 2D weight, got {w.shape}"
    w_cap = torch.to_dlpack(w.detach().float().contiguous())
    qw_cap, qs_cap = _native.quantize_linear_int4(w_cap)
    return torch.from_dlpack(qw_cap), torch.from_dlpack(qs_cap)


def quantize_weight_int4_grouped(
    weight: torch.Tensor,
    group_size: int = 64,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float weight matrix to signed 4-bit grouped format with bounded peak RAM.
    
    Args:
        weight: (N, K) float tensor
        group_size: number of elements per quantization group (default 64)
        chunk_size: maximum number of rows to process at once to prevent OOM / RAM freezing
        
    Returns:
        packed_weights: (N, (K + 1) // 2) uint8 tensor storing 2 nibbles per byte
        scales: (N, (K + group_size - 1) // group_size) float32 tensor
    """
    orig_device = weight.device
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size
    packed_cols = (K + 1) // 2

    # If matrix is small, quantize in a single shot
    if N <= chunk_size:
        w = weight.detach().to(device="cpu", dtype=torch.float32)
        w_grouped = w.view(N, num_groups, group_size)
        max_val = w_grouped.abs().amax(dim=-1, keepdim=True)
        scales = torch.clamp(max_val / 7.0, min=1e-8)
        q = torch.clamp(torch.round(w_grouped / scales), -8, 7).to(torch.int8)
        q_flat = q.view(N, K)
        u = (q_flat + 8).to(torch.uint8)
        low = u[:, 0::2] & 0x0F
        high = (u[:, 1::2] & 0x0F) << 4
        packed = (low | high).contiguous()
        scales_out = scales.view(N, num_groups).contiguous()
        return packed.to(orig_device), scales_out.to(orig_device)

    # For large matrices (e.g. lm_head with 151,936 rows), chunk rows to strictly bound peak memory
    packed = torch.empty((N, packed_cols), dtype=torch.uint8, device=orig_device)
    scales_out = torch.empty((N, num_groups), dtype=torch.float32, device=orig_device)

    for start_row in range(0, N, chunk_size):
        end_row = min(start_row + chunk_size, N)
        chunk_N = end_row - start_row
        w_chunk = weight[start_row:end_row].detach().to(device="cpu", dtype=torch.float32)
        w_grouped = w_chunk.view(chunk_N, num_groups, group_size)
        max_val = w_grouped.abs().amax(dim=-1, keepdim=True)
        chunk_scales = torch.clamp(max_val / 7.0, min=1e-8)
        q = torch.clamp(torch.round(w_grouped / chunk_scales), -8, 7).to(torch.int8)
        q_flat = q.view(chunk_N, K)
        u = (q_flat + 8).to(torch.uint8)
        low = u[:, 0::2] & 0x0F
        high = (u[:, 1::2] & 0x0F) << 4
        chunk_packed = (low | high).contiguous()
        packed[start_row:end_row].copy_(chunk_packed)
        scales_out[start_row:end_row].copy_(chunk_scales.view(chunk_N, num_groups))

    return packed, scales_out



def w8a32_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute native SIMD W8A32 linear projection."""
    x_orig_shape = x.shape
    K = x_orig_shape[-1]
    x_2d = x.reshape(-1, K).contiguous().float()
    w_1b = w.contiguous()
    scales_f32 = scales.contiguous().float()
    bias_f32 = bias.contiguous().float() if bias is not None else None

    x_cap = torch.to_dlpack(x_2d)
    w_cap = torch.to_dlpack(w_1b)
    s_cap = torch.to_dlpack(scales_f32)
    b_cap = torch.to_dlpack(bias_f32) if bias_f32 is not None else None

    out_cap = _native.w8a32_linear(x_cap, w_cap, s_cap, b_cap)
    out_2d = torch.from_dlpack(out_cap)

    out_shape = list(x_orig_shape[:-1]) + [w.shape[0]]
    return out_2d.reshape(out_shape)


def w4a32_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute native SIMD W4A32 linear projection (per-channel scales)."""
    x_orig_shape = x.shape
    K = x_orig_shape[-1]
    x_2d = x.reshape(-1, K).contiguous().float()
    w_1b = w_packed.contiguous()
    scales_f32 = scales.contiguous().float()
    bias_f32 = bias.contiguous().float() if bias is not None else None

    x_cap = torch.to_dlpack(x_2d)
    w_cap = torch.to_dlpack(w_1b)
    s_cap = torch.to_dlpack(scales_f32)
    b_cap = torch.to_dlpack(bias_f32) if bias_f32 is not None else None

    out_cap = _native.w4a32_linear(x_cap, w_cap, s_cap, b_cap)
    out_2d = torch.from_dlpack(out_cap)

    out_shape = list(x_orig_shape[:-1]) + [w_packed.shape[0]]
    return out_2d.reshape(out_shape)


def w4a32_grouped_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 64,
) -> torch.Tensor:
    """Execute native SIMD W4A32 linear projection with grouped scales."""
    x_orig_shape = x.shape
    K = x_orig_shape[-1]
    x_2d = x.reshape(-1, K).contiguous().float()
    w_1b = w_packed.contiguous()
    scales_f32 = scales.contiguous().float()
    bias_f32 = bias.contiguous().float() if bias is not None else None

    x_cap = torch.to_dlpack(x_2d)
    w_cap = torch.to_dlpack(w_1b)
    s_cap = torch.to_dlpack(scales_f32)
    b_cap = torch.to_dlpack(bias_f32) if bias_f32 is not None else None

    out_cap = _native.w4a32_grouped_linear(x_cap, w_cap, s_cap, b_cap, group_size)
    out_2d = torch.from_dlpack(out_cap)

    out_shape = list(x_orig_shape[:-1]) + [w_packed.shape[0]]
    return out_2d.reshape(out_shape)


def wgpu_w4a32_grouped_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 64,
) -> torch.Tensor:
    """Execute native INT4 Vulkan compute shader W4A32 linear projection on iGPU."""
    x_orig_shape = x.shape
    K = x_orig_shape[-1]
    x_2d = x.reshape(-1, K).contiguous().float()
    w_1b = w_packed.contiguous()
    scales_f32 = scales.contiguous().float()
    bias_f32 = bias.contiguous().float() if bias is not None else None

    x_cap = torch.to_dlpack(x_2d)
    w_cap = torch.to_dlpack(w_1b)
    s_cap = torch.to_dlpack(scales_f32)
    b_cap = torch.to_dlpack(bias_f32) if bias_f32 is not None else None

    out_cap = _native.wgpu_w4a32_grouped_linear(x_cap, w_cap, s_cap, b_cap, group_size)
    out_2d = torch.from_dlpack(out_cap)

    out_shape = list(x_orig_shape[:-1]) + [w_packed.shape[0]]
    return out_2d.reshape(out_shape)


def fused_swiglu_mlp(
    x: torch.Tensor,
    gate: QuantizedLinear,
    up: QuantizedLinear,
    down: QuantizedLinear,
) -> torch.Tensor:
    """Fused SwiGLU MLP: out = down(silu(gate(x)) * up(x)) executed in a single native pass."""
    x_orig_shape = x.shape
    K = x_orig_shape[-1]
    x_2d = x.reshape(-1, K).contiguous().float()
    x_cap = torch.to_dlpack(x_2d)

    gw_cap = torch.to_dlpack(gate.qweight.contiguous())
    gs_cap = torch.to_dlpack(gate.scales.contiguous().float())
    gb_cap = torch.to_dlpack(gate.bias.contiguous().float()) if gate.bias is not None else None

    uw_cap = torch.to_dlpack(up.qweight.contiguous())
    us_cap = torch.to_dlpack(up.scales.contiguous().float())
    ub_cap = torch.to_dlpack(up.bias.contiguous().float()) if up.bias is not None else None

    dw_cap = torch.to_dlpack(down.qweight.contiguous())
    ds_cap = torch.to_dlpack(down.scales.contiguous().float())
    db_cap = torch.to_dlpack(down.bias.contiguous().float()) if down.bias is not None else None

    if gate.bits == 8:
        out_cap = _native.fused_swiglu_mlp_w8a32(
            x_cap,
            gw_cap, gs_cap, gb_cap,
            uw_cap, us_cap, ub_cap,
            dw_cap, ds_cap, db_cap,
        )
    elif gate.bits == 4:
        out_cap = _native.fused_swiglu_mlp_w4a32(
            x_cap,
            gw_cap, gs_cap, gb_cap,
            uw_cap, us_cap, ub_cap,
            dw_cap, ds_cap, db_cap,
            gate.group_size,
        )
    else:
        raise ValueError(f"Unsupported bits={gate.bits}")

    out_2d = torch.from_dlpack(out_cap)
    out_shape = list(x_orig_shape[:-1]) + [down.out_features]
    return out_2d.reshape(out_shape)


def fused_attention_step(
    x: torch.Tensor,
    qkv: QuantizedLinear,
    o_proj: QuantizedLinear,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Zero-allocation Fused Attention decode step (T=1) in native Rust."""
    from . import _torchburn as _native

    x_orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous().float()

    x_cap = torch.to_dlpack(x_2d)
    qkv_w_cap = torch.to_dlpack(qkv.qweight.contiguous())
    qkv_s_cap = torch.to_dlpack(qkv.scales.contiguous().float())
    qkv_b_cap = torch.to_dlpack(qkv.bias.contiguous().float()) if qkv.bias is not None else None

    o_w_cap = torch.to_dlpack(o_proj.qweight.contiguous())
    o_s_cap = torch.to_dlpack(o_proj.scales.contiguous().float())
    o_b_cap = torch.to_dlpack(o_proj.bias.contiguous().float()) if o_proj.bias is not None else None

    k_cache_cap = torch.to_dlpack(k_cache.contiguous())
    v_cache_cap = torch.to_dlpack(v_cache.contiguous())

    cos_flat = cos.reshape(-1).contiguous().float()
    sin_flat = sin.reshape(-1).contiguous().float()
    cos_cap = torch.to_dlpack(cos_flat)
    sin_cap = torch.to_dlpack(sin_flat)

    if qkv.bits == 8:
        out_cap = _native.fused_attention_step_w8a32(
            x_cap,
            qkv_w_cap, qkv_s_cap, qkv_b_cap,
            o_w_cap, o_s_cap, o_b_cap,
            k_cache_cap, v_cache_cap,
            cos_cap, sin_cap,
            offset,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    elif qkv.bits == 4:
        out_cap = _native.fused_attention_step_w4a32(
            x_cap,
            qkv_w_cap, qkv_s_cap, qkv_b_cap,
            o_w_cap, o_s_cap, o_b_cap,
            k_cache_cap, v_cache_cap,
            cos_cap, sin_cap,
            offset,
            num_heads,
            num_kv_heads,
            head_dim,
            qkv.group_size,
        )
    else:
        raise ValueError(f"Unsupported bits={qkv.bits}")

    out_2d = torch.from_dlpack(out_cap)
    return out_2d.reshape(x_orig_shape)


def fused_transformer_layer_step(
    x: torch.Tensor,
    layer: nn.Module,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Zero-allocation single-pass transformer decoder block in native Rust."""
    from . import _torchburn as _native

    x_2d = x.reshape(-1, x.shape[-1]).contiguous().float()
    qkv = layer.self_attn.qkv_proj
    o_proj = layer.self_attn.o_proj
    gate = layer.mlp.gate_proj
    up = layer.mlp.up_proj
    down = layer.mlp.down_proj

    x_cap = torch.to_dlpack(x_2d)
    in_norm_cap = torch.to_dlpack(layer.input_layernorm.weight.contiguous().float())
    qkv_w_cap = torch.to_dlpack(qkv.qweight.contiguous())
    qkv_s_cap = torch.to_dlpack(qkv.scales.contiguous().float())
    qkv_b_cap = torch.to_dlpack(qkv.bias.contiguous().float()) if qkv.bias is not None else None

    o_w_cap = torch.to_dlpack(o_proj.qweight.contiguous())
    o_s_cap = torch.to_dlpack(o_proj.scales.contiguous().float())
    o_b_cap = torch.to_dlpack(o_proj.bias.contiguous().float()) if o_proj.bias is not None else None

    post_norm_cap = torch.to_dlpack(layer.post_attention_layernorm.weight.contiguous().float())
    gate_w_cap = torch.to_dlpack(gate.qweight.contiguous())
    gate_s_cap = torch.to_dlpack(gate.scales.contiguous().float())
    gate_b_cap = torch.to_dlpack(gate.bias.contiguous().float()) if gate.bias is not None else None

    up_w_cap = torch.to_dlpack(up.qweight.contiguous())
    up_s_cap = torch.to_dlpack(up.scales.contiguous().float())
    up_b_cap = torch.to_dlpack(up.bias.contiguous().float()) if up.bias is not None else None

    down_w_cap = torch.to_dlpack(down.qweight.contiguous())
    down_s_cap = torch.to_dlpack(down.scales.contiguous().float())
    down_b_cap = torch.to_dlpack(down.bias.contiguous().float()) if down.bias is not None else None

    k_cache_cap = torch.to_dlpack(k_cache.contiguous())
    v_cache_cap = torch.to_dlpack(v_cache.contiguous())

    cos_flat = cos.reshape(-1).contiguous().float()
    sin_flat = sin.reshape(-1).contiguous().float()
    cos_cap = torch.to_dlpack(cos_flat)
    sin_cap = torch.to_dlpack(sin_flat)

    _native.fused_transformer_layer_step_w4a32(
        x_cap,
        in_norm_cap,
        qkv_w_cap, qkv_s_cap, qkv_b_cap,
        o_w_cap, o_s_cap, o_b_cap,
        post_norm_cap,
        gate_w_cap, gate_s_cap, gate_b_cap,
        up_w_cap, up_s_cap, up_b_cap,
        down_w_cap, down_s_cap, down_b_cap,
        k_cache_cap, v_cache_cap,
        cos_cap, sin_cap,
        offset,
        num_heads,
        num_kv_heads,
        head_dim,
        qkv.group_size,
        eps,
    )
    return x_2d.reshape(x.shape)


def create_rust_qwen_decoder(model: nn.Module, max_seq_len: int = 4096) -> Any:
    """Instantiate zero-Python end-to-end RustQwenDecoder from a quantized NoCudaModel."""
    from . import _torchburn as _native

    embed_tokens_cap = torch.to_dlpack(model.embed_tokens.weight.detach().cpu().contiguous().float())
    final_norm_cap = torch.to_dlpack(model.norm.weight.detach().cpu().contiguous().float())
    lm_head_w_cap = torch.to_dlpack(model.lm_head.qweight.detach().cpu().contiguous())
    lm_head_s_cap = torch.to_dlpack(model.lm_head.scales.detach().cpu().contiguous().float())

    layers_caps = []
    for layer in model.layers:
        in_norm = torch.to_dlpack(layer.input_layernorm.weight.detach().cpu().contiguous().float())
        qkv_w = torch.to_dlpack(layer.self_attn.qkv_proj.qweight.detach().cpu().contiguous())
        qkv_s = torch.to_dlpack(layer.self_attn.qkv_proj.scales.detach().cpu().contiguous().float())
        o_w = torch.to_dlpack(layer.self_attn.o_proj.qweight.detach().cpu().contiguous())
        o_s = torch.to_dlpack(layer.self_attn.o_proj.scales.detach().cpu().contiguous().float())
        post_norm = torch.to_dlpack(layer.post_attention_layernorm.weight.detach().cpu().contiguous().float())
        gate_w = torch.to_dlpack(layer.mlp.gate_proj.qweight.detach().cpu().contiguous())
        gate_s = torch.to_dlpack(layer.mlp.gate_proj.scales.detach().cpu().contiguous().float())
        up_w = torch.to_dlpack(layer.mlp.up_proj.qweight.detach().cpu().contiguous())
        up_s = torch.to_dlpack(layer.mlp.up_proj.scales.detach().cpu().contiguous().float())
        down_w = torch.to_dlpack(layer.mlp.down_proj.qweight.detach().cpu().contiguous())
        down_s = torch.to_dlpack(layer.mlp.down_proj.scales.detach().cpu().contiguous().float())

        caps = [
            in_norm, qkv_w, qkv_s, o_w, o_s, post_norm,
            gate_w, gate_s, up_w, up_s, down_w, down_s,
        ]
        if hasattr(layer.self_attn.qkv_proj, "bias") and layer.self_attn.qkv_proj.bias is not None:
            caps.append(torch.to_dlpack(layer.self_attn.qkv_proj.bias.detach().cpu().contiguous().float()))
        layers_caps.append(caps)

    cfg = model.config
    decoder = _native.RustQwenDecoder(
        embed_tokens_cap,
        layers_caps,
        final_norm_cap,
        lm_head_w_cap,
        lm_head_s_cap,
        len(model.layers),
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        64,
        cfg.rms_norm_eps,
        max_seq_len,
        cfg.rope_theta,
    )
    return decoder


def create_wgpu_qwen_decoder(model: nn.Module, max_seq_len: int = 2048) -> Any:
    """Instantiate zero-Python end-to-end WgpuQwenDecoder for GPU (Vulkan/Metal/DX12)."""
    from . import _torchburn as _native

    if not hasattr(_native, "WgpuQwenDecoder"):
        raise RuntimeError("WgpuQwenDecoder requires torchburn compiled with the 'burn-wgpu' feature.")

    embed_tokens_cap = torch.to_dlpack(model.embed_tokens.weight.detach().cpu().contiguous().float())
    final_norm_cap = torch.to_dlpack(model.norm.weight.detach().cpu().contiguous().float())
    lm_head_w_cap = torch.to_dlpack(model.lm_head.qweight.detach().cpu().contiguous())
    lm_head_s_cap = torch.to_dlpack(model.lm_head.scales.detach().cpu().contiguous().float())

    layers_caps = []
    for layer in model.layers:
        in_norm = torch.to_dlpack(layer.input_layernorm.weight.detach().cpu().contiguous().float())
        qkv_w = torch.to_dlpack(layer.self_attn.qkv_proj.qweight.detach().cpu().contiguous())
        qkv_s = torch.to_dlpack(layer.self_attn.qkv_proj.scales.detach().cpu().contiguous().float())
        o_w = torch.to_dlpack(layer.self_attn.o_proj.qweight.detach().cpu().contiguous())
        o_s = torch.to_dlpack(layer.self_attn.o_proj.scales.detach().cpu().contiguous().float())
        post_norm = torch.to_dlpack(layer.post_attention_layernorm.weight.detach().cpu().contiguous().float())
        gate_w = torch.to_dlpack(layer.mlp.gate_proj.qweight.detach().cpu().contiguous())
        gate_s = torch.to_dlpack(layer.mlp.gate_proj.scales.detach().cpu().contiguous().float())
        up_w = torch.to_dlpack(layer.mlp.up_proj.qweight.detach().cpu().contiguous())
        up_s = torch.to_dlpack(layer.mlp.up_proj.scales.detach().cpu().contiguous().float())
        down_w = torch.to_dlpack(layer.mlp.down_proj.qweight.detach().cpu().contiguous())
        down_s = torch.to_dlpack(layer.mlp.down_proj.scales.detach().cpu().contiguous().float())

        caps = [
            in_norm, qkv_w, qkv_s, o_w, o_s, post_norm,
            gate_w, gate_s, up_w, up_s, down_w, down_s,
        ]
        if hasattr(layer.self_attn.qkv_proj, "bias") and layer.self_attn.qkv_proj.bias is not None:
            caps.append(torch.to_dlpack(layer.self_attn.qkv_proj.bias.detach().cpu().contiguous().float()))
        layers_caps.append(caps)

    cfg = model.config
    decoder = _native.WgpuQwenDecoder(
        embed_tokens_cap,
        layers_caps,
        final_norm_cap,
        lm_head_w_cap,
        lm_head_s_cap,
        len(model.layers),
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        64,
        cfg.rms_norm_eps,
        max_seq_len,
        cfg.rope_theta,
    )
    return decoder



class QuantizedLinear(nn.Module):
    """Drop-in replacement for `nn.Linear` using TorchBurn's native SIMD quantized kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
        group_size: int = 64,
        backend: str = "cpu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.backend = backend

        if bits == 8:
            self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scales", torch.ones(out_features, dtype=torch.float32))
        elif bits == 4:
            self.register_buffer("qweight", torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8))
            num_groups = (in_features + group_size - 1) // group_size
            self.register_buffer("scales", torch.ones(out_features, num_groups, dtype=torch.float32))
        else:
            raise ValueError(f"Supported quantization bits are 8 and 4, got {bits}")

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int = 8, group_size: int = 64, backend: str = "cpu") -> QuantizedLinear:
        """Create a QuantizedLinear module from a standard float nn.Linear."""
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, bias=has_bias, bits=bits, group_size=group_size, backend=backend)
        with torch.no_grad():
            if bits == 8:
                qw, sc = quantize_weight_int8(linear.weight)
            else:
                qw, sc = quantize_weight_int4_grouped(linear.weight, group_size=group_size)
            ql.qweight.copy_(qw)
            ql.scales.copy_(sc)
            if has_bias:
                ql.bias.copy_(linear.bias.float())
            # Free memory immediately!
            linear.weight = None
            if has_bias:
                linear.bias = None
        return ql

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 8:
            return w8a32_linear(x, self.qweight, self.scales, self.bias)
        else:
            if self.backend == "igpu" or os.environ.get("TORCHBURN_DEVICE", "").lower() in ("gpu", "igpu", "vulkan"):
                return wgpu_w4a32_grouped_linear(x, self.qweight, self.scales, self.bias, group_size=self.group_size)
            return w4a32_grouped_linear(x, self.qweight, self.scales, self.bias, group_size=self.group_size)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}, backend={self.backend}"


def quantize_model(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 64,
    exclude_modules: Optional[List[str]] = None,
    backend: str = "cpu",
) -> nn.Module:
    """Quantize all linear layers of a model to native TorchBurn SIMD INT8 or Grouped INT4."""
    exclude = set(exclude_modules if exclude_modules is not None else ["lm_head"])

    def _replace_linear(module: nn.Module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if name in exclude or full_name in exclude:
                    continue
                qlinear = QuantizedLinear.from_float(child, bits=bits, group_size=group_size, backend=backend)
                setattr(module, name, qlinear)
                del child
            else:
                _replace_linear(child, full_name)

    _replace_linear(model)
    gc.collect()
    return model
