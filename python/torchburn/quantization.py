"""TorchBurn Native Quantization Suite (INT8 and Grouped INT4).

Provides hardware-accelerated SIMD W8A32 (Weight INT8, Activation FP32)
and W4A32 (Grouped Weight INT4, Activation FP32) linear projections,
fused SwiGLU MLP kernels, drop-in `QuantizedLinear` modules, and full model
quantization utilities.
"""

from __future__ import annotations
import gc
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float weight matrix to signed 4-bit grouped format.
    
    Args:
        weight: (N, K) float tensor
        group_size: number of elements per quantization group (default 64)
        
    Returns:
        packed_weights: (N, (K + 1) // 2) uint8 tensor storing 2 nibbles per byte
        scales: (N, (K + group_size - 1) // group_size) float32 tensor
    """
    orig_device = weight.device
    w = weight.detach().cpu().float()
    N, K = w.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size

    # Reshape to (N, num_groups, group_size)
    w_grouped = w.view(N, num_groups, group_size)

    # Scale per group: max(|w|) / 7.0
    max_val = w_grouped.abs().amax(dim=-1, keepdim=True)
    scales = torch.clamp(max_val / 7.0, min=1e-8)  # (N, num_groups, 1)

    # Quantize to [-8, 7]
    q = torch.clamp(torch.round(w_grouped / scales), -8, 7).to(torch.int8)

    # Reshape back to (N, K)
    q_flat = q.view(N, K)

    # Offset by 8 to convert signed [-8, 7] to unsigned [0, 15]
    u = (q_flat + 8).to(torch.uint8)

    # Pack into (N, K // 2): Low nibble even cols, High nibble odd cols
    low = u[:, 0::2] & 0x0F
    high = (u[:, 1::2] & 0x0F) << 4
    packed = (low | high).contiguous()

    scales_out = scales.view(N, num_groups).contiguous()
    return packed.to(orig_device), scales_out.to(orig_device)


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


class QuantizedLinear(nn.Module):
    """Drop-in replacement for `nn.Linear` using TorchBurn's native SIMD quantized kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
        group_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

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
    def from_float(cls, linear: nn.Linear, bits: int = 8, group_size: int = 64) -> QuantizedLinear:
        """Create a QuantizedLinear module from a standard float nn.Linear."""
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, bias=has_bias, bits=bits, group_size=group_size)
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
            return w4a32_grouped_linear(x, self.qweight, self.scales, self.bias, group_size=self.group_size)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"


def quantize_model(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 64,
    exclude_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Quantize all linear layers of a model to native TorchBurn SIMD INT8 or Grouped INT4."""
    exclude = set(exclude_modules if exclude_modules is not None else ["lm_head"])

    def _replace_linear(module: nn.Module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if name in exclude or full_name in exclude:
                    continue
                qlinear = QuantizedLinear.from_float(child, bits=bits, group_size=group_size)
                setattr(module, name, qlinear)
                del child
            else:
                _replace_linear(child, full_name)

    _replace_linear(model)
    gc.collect()
    return model
