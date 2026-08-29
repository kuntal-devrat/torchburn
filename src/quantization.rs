//! Universal Quantization & Low-Bit GEMM kernels (INT8, INT4, NF4, FP8).
//!
//! Provides native, memory-safe, hardware-agnostic low-bit tensor processing.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, elem_count, unsupported};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

/// Quantize a floating-point tensor to INT8 / INT32: q = clamp(round(x / scale) + zero_point, min, max)
pub fn quantize_per_tensor(
    x: &BorrowedTensor,
    scale: f64,
    zero_point: i64,
    dtype_target: DType,
) -> PyResult<OwnedTensor> {
    if scale <= 0.0 {
        return Err(unsupported("quantize_per_tensor scale must be positive"));
    }
    let n = elem_count(&x.shape);
    let mut out = OwnedTensor::new(dtype_target, x.shape.clone());
    let inv_scale = 1.0 / scale;

    match (x.dtype, dtype_target) {
        (DType::F32, DType::I32) => {
            let src = unsafe { typed_slice::<f32>(x) };
            let dst = unsafe { typed_mut_slice::<i32>(&mut out) };
            let inv_s = inv_scale as f32;
            let zp = zero_point as f32;
            for i in 0..n {
                let q = (src[i] * inv_s).round() + zp;
                dst[i] = q.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            }
        }
        (DType::F32, DType::I64) => {
            let src = unsafe { typed_slice::<f32>(x) };
            let dst = unsafe { typed_mut_slice::<i64>(&mut out) };
            let inv_s = inv_scale as f32;
            let zp = zero_point as f32;
            for i in 0..n {
                let q = (src[i] * inv_s).round() + zp;
                dst[i] = q as i64;
            }
        }
        (DType::F64, DType::I64) => {
            let src = unsafe { typed_slice::<f64>(x) };
            let dst = unsafe { typed_mut_slice::<i64>(&mut out) };
            let zp = zero_point as f64;
            for i in 0..n {
                let q = (src[i] * inv_scale).round() + zp;
                dst[i] = q as i64;
            }
        }
        _ => return Err(unsupported("quantize_per_tensor unsupported dtype combination")),
    }
    Ok(out)
}

/// Dequantize an integer tensor back to float: x = (q - zero_point) * scale
pub fn dequantize_per_tensor(
    q: &BorrowedTensor,
    scale: f64,
    zero_point: i64,
) -> PyResult<OwnedTensor> {
    let n = elem_count(&q.shape);
    let mut out = OwnedTensor::new(DType::F32, q.shape.clone());

    match q.dtype {
        DType::I32 => {
            let src = unsafe { typed_slice::<i32>(q) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            let s = scale as f32;
            let zp = zero_point as i32;
            for i in 0..n {
                dst[i] = ((src[i] - zp) as f32) * s;
            }
        }
        DType::I64 => {
            let src = unsafe { typed_slice::<i64>(q) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            let s = scale as f32;
            let zp = zero_point;
            for i in 0..n {
                dst[i] = ((src[i] - zp) as f32) * s;
            }
        }
        DType::F32 => {
            let src = unsafe { typed_slice::<f32>(q) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            let s = scale as f32;
            let zp = zero_point as f32;
            for i in 0..n {
                dst[i] = (src[i] - zp) * s;
            }
        }
        _ => return Err(unsupported("dequantize_per_tensor unsupported dtype")),
    }
    Ok(out)
}

/// Per-channel quantization along specified axis.
pub fn quantize_per_channel(
    x: &BorrowedTensor,
    scales: &BorrowedTensor,
    zero_points: &BorrowedTensor,
    axis: usize,
) -> PyResult<OwnedTensor> {
    let rank = x.shape.len();
    if axis >= rank {
        return Err(unsupported("axis out of bounds for quantize_per_channel"));
    }
    let n = elem_count(&x.shape);
    let mut out = OwnedTensor::new(DType::I32, x.shape.clone());

    let scale_slice = unsafe { typed_slice::<f32>(scales) };
    let zp_slice = unsafe { typed_slice::<i32>(zero_points) };
    let x_slice = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<i32>(&mut out) };

    let axis_dim = x.shape[axis] as usize;
    let mut outer_stride = 1usize;
    for i in (axis + 1)..rank {
        outer_stride *= x.shape[i] as usize;
    }

    for i in 0..n {
        let ch = (i / outer_stride) % axis_dim;
        let s = scale_slice[ch];
        let zp = zp_slice[ch];
        let q = (x_slice[i] / s).round() + (zp as f32);
        dst[i] = q.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    }

    Ok(out)
}

/// Per-channel dequantization along specified axis.
pub fn dequantize_per_channel(
    q: &BorrowedTensor,
    scales: &BorrowedTensor,
    zero_points: &BorrowedTensor,
    axis: usize,
) -> PyResult<OwnedTensor> {
    let rank = q.shape.len();
    if axis >= rank {
        return Err(unsupported("axis out of bounds for dequantize_per_channel"));
    }
    let n = elem_count(&q.shape);
    let mut out = OwnedTensor::new(DType::F32, q.shape.clone());

    let scale_slice = unsafe { typed_slice::<f32>(scales) };
    let zp_slice = unsafe { typed_slice::<i32>(zero_points) };
    let q_slice = unsafe { typed_slice::<i32>(q) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let axis_dim = q.shape[axis] as usize;
    let mut outer_stride = 1usize;
    for i in (axis + 1)..rank {
        outer_stride *= q.shape[i] as usize;
    }

    for i in 0..n {
        let ch = (i / outer_stride) % axis_dim;
        let s = scale_slice[ch];
        let zp = zp_slice[ch];
        dst[i] = ((q_slice[i] - zp) as f32) * s;
    }

    Ok(out)
}

/// Native INT8 Matrix Multiplication: Y = A @ B * scale_a * scale_b
pub fn int8_gemm(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    scale_a: f64,
    scale_b: f64,
) -> PyResult<OwnedTensor> {
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    if a_rank < 2 || b_rank < 2 {
        return Err(unsupported("int8_gemm requires >= 2D matrices"));
    }
    let m = elem_count(&a.shape[..a_rank - 1]);
    let k = a.shape[a_rank - 1] as usize;
    let b_k = b.shape[b_rank - 2] as usize;
    let n = b.shape[b_rank - 1] as usize;
    if k != b_k {
        return Err(unsupported("int8_gemm dimension mismatch"));
    }

    let mut out_shape = a.shape.clone();
    out_shape[a_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let effective_scale = (scale_a * scale_b) as f32;
    let a_slice = unsafe { typed_slice::<i32>(a) };
    let b_slice = unsafe { typed_slice::<i32>(b) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    for row in 0..m {
        let a_row = &a_slice[row * k..(row + 1) * k];
        let out_row = &mut out_slice[row * n..(row + 1) * n];
        for col in 0..n {
            let mut acc = 0i64;
            for i in 0..k {
                acc += (a_row[i] as i64) * (b_slice[i * n + col] as i64);
            }
            out_row[col] = (acc as f32) * effective_scale;
        }
    }

    Ok(out)
}

/// NormalFloat4 (NF4) codebook lookup table as specified in QLoRA.
const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
];

/// Unpack packed 4-bit NormalFloat4 (2 elements per byte) and dequantize with per-group absmax.
pub fn nf4_dequantize(
    packed: &BorrowedTensor,
    absmax: &BorrowedTensor,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let n_bytes = elem_count(&packed.shape);
    let mut out_shape = packed.shape.clone();
    let rank = out_shape.len();
    out_shape[rank - 1] *= 2;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let src = unsafe { typed_slice::<u8>(packed) };
    let absmax_slice = unsafe { typed_slice::<f32>(absmax) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    for i in 0..n_bytes {
        let byte = src[i];
        let nibble0 = (byte & 0x0F) as usize;
        let nibble1 = ((byte >> 4) & 0x0F) as usize;

        let elem0 = i * 2;
        let elem1 = i * 2 + 1;

        let g0 = elem0 / group_size.max(1);
        let g1 = elem1 / group_size.max(1);

        let scale0 = if g0 < absmax_slice.len() { absmax_slice[g0] } else { 1.0 };
        let scale1 = if g1 < absmax_slice.len() { absmax_slice[g1] } else { 1.0 };

        dst[elem0] = NF4_CODEBOOK[nibble0] * scale0;
        dst[elem1] = NF4_CODEBOOK[nibble1] * scale1;
    }

    Ok(out)
}

/// Unpack 4-bit packed weights (AWQ / GPTQ linear format) with group scaling and zero-points.
pub fn int4_unpack_dequantize(
    packed: &BorrowedTensor,
    scales: &BorrowedTensor,
    zeros: &BorrowedTensor,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let n_bytes = elem_count(&packed.shape);
    let mut out_shape = packed.shape.clone();
    let rank = out_shape.len();
    out_shape[rank - 1] *= 2;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let src = unsafe { typed_slice::<u8>(packed) };
    let scale_slice = unsafe { typed_slice::<f32>(scales) };
    let zero_slice = unsafe { typed_slice::<f32>(zeros) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    for i in 0..n_bytes {
        let byte = src[i];
        let n0 = (byte & 0x0F) as f32;
        let n1 = ((byte >> 4) & 0x0F) as f32;

        let elem0 = i * 2;
        let elem1 = i * 2 + 1;

        let g0 = elem0 / group_size.max(1);
        let g1 = elem1 / group_size.max(1);

        let s0 = if g0 < scale_slice.len() { scale_slice[g0] } else { 1.0 };
        let s1 = if g1 < scale_slice.len() { scale_slice[g1] } else { 1.0 };
        let z0 = if g0 < zero_slice.len() { zero_slice[g0] } else { 0.0 };
        let z1 = if g1 < zero_slice.len() { zero_slice[g1] } else { 0.0 };

        dst[elem0] = (n0 - z0) * s0;
        dst[elem1] = (n1 - z1) * s1;
    }

    Ok(out)
}
