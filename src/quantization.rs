//! Universal Quantization & Low-Bit GEMM kernels (INT8, INT4, NF4, FP8).
//!
//! Provides native, memory-safe, hardware-agnostic low-bit tensor processing.

use crate::dlpack::{elem_count, unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;

pub(crate) unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

pub mod per_tensor;

pub use self::per_tensor::{
    dequantize_per_channel, dequantize_per_tensor, int4_unpack_dequantize, int8_gemm,
    nf4_dequantize, quantize_per_channel, quantize_per_tensor,
};

pub mod gemv;

pub use self::gemv::{dot_f32_i8, gemv_w4a32_grouped, w4a32_linear, w8a32_linear};
pub(crate) use self::gemv::{
    dot_f32_u4_group_scalar, gemv_w8a32, hsum256_ps_avx, quantize_activation_to_u8,
    swiglu_neuron_w4a32_group32_avx2, swiglu_neuron_w4a32_group32_avx512,
    swiglu_neuron_w4a32_group64_avx2, swiglu_neuron_w4a32_group64_avx512,
    swiglu_neuron_w4a8_group64_vnni_avx512, unpack_and_fma_32_avx2, unpack_and_fma_32_avx512,
};

pub mod packed_v2;

#[cfg(feature = "burn-wgpu")]
pub use self::packed_v2::wgpu_w4a32_grouped_linear;
pub use self::packed_v2::{
    f16_to_f32, f32_to_f16, gemv_w4a32_grouped_v2, pack_rows_w4a32_group64_v1_to_v2,
    w4a32_grouped_linear, w4a32_grouped_linear_v2,
};

pub mod fused_ops;

pub(crate) use self::fused_ops::fast_vector_add;
pub use self::fused_ops::{
    dot_f32_f32, fast_rms_norm, fused_attention_step_w4a32, fused_attention_step_w8a32,
    fused_swiglu_mlp_w4a32, fused_swiglu_mlp_w8a32, quantize_linear_weights_int4,
    quantize_linear_weights_int8,
};

pub mod fused_transformer;

pub use self::fused_transformer::fused_transformer_layer_step_w4a32;
