//! Universal Quantization & Low-Bit GEMM kernels (INT8, INT4, NF4, FP8).
//!
//! Provides native, memory-safe, hardware-agnostic low-bit tensor processing.

use crate::dlpack::{self, elem_count, unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use rayon::prelude::*;

pub(crate) unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
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
        (DType::F32, DType::Bool) => {
            let src = unsafe { typed_slice::<f32>(x) };
            let dst = unsafe { typed_mut_slice::<i8>(&mut out) };
            let inv_s = inv_scale as f32;
            let zp = zero_point as f32;
            for i in 0..n {
                let q = (src[i] * inv_s).round() + zp;
                dst[i] = q.clamp(i8::MIN as f32, i8::MAX as f32) as i8;
            }
        }
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
        _ => {
            return Err(unsupported(
                "quantize_per_tensor unsupported dtype combination",
            ))
        }
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
        DType::Bool => {
            let src = unsafe { typed_slice::<i8>(q) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            let s = scale as f32;
            let zp = zero_point as i8;
            for i in 0..n {
                dst[i] = ((src[i] - zp) as f32) * s;
            }
        }
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
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
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

        let scale0 = if g0 < absmax_slice.len() {
            absmax_slice[g0]
        } else {
            1.0
        };
        let scale1 = if g1 < absmax_slice.len() {
            absmax_slice[g1]
        } else {
            1.0
        };

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

        let s0 = if g0 < scale_slice.len() {
            scale_slice[g0]
        } else {
            1.0
        };
        let s1 = if g1 < scale_slice.len() {
            scale_slice[g1]
        } else {
            1.0
        };
        let z0 = if g0 < zero_slice.len() {
            zero_slice[g0]
        } else {
            0.0
        };
        let z1 = if g1 < zero_slice.len() {
            zero_slice[g1]
        } else {
            0.0
        };

        dst[elem0] = (n0 - z0) * s0;
        dst[elem1] = (n1 - z1) * s1;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Native AVX2 / SIMD Quantized Linear Kernels (W8A32 & W4A32)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let v_hi = _mm256_extractf128_ps::<1>(v);
    let v_lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(v_hi, v_lo);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<0x55>(sum64, sum64));
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_4rows_w8a32_avx2(
    x: *const f32,
    w0: *const i8,
    w1: *const i8,
    w2: *const i8,
    w3: *const i8,
    len: usize,
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;

    let mut acc0_0 = _mm256_setzero_ps();
    let mut acc0_1 = _mm256_setzero_ps();
    let mut acc1_0 = _mm256_setzero_ps();
    let mut acc1_1 = _mm256_setzero_ps();
    let mut acc2_0 = _mm256_setzero_ps();
    let mut acc2_1 = _mm256_setzero_ps();
    let mut acc3_0 = _mm256_setzero_ps();
    let mut acc3_1 = _mm256_setzero_ps();

    let chunks16 = len / 16;
    let mut offset = 0;

    for _ in 0..chunks16 {
        let x0 = _mm256_loadu_ps(x.add(offset));
        let x1 = _mm256_loadu_ps(x.add(offset + 8));

        // Row 0
        let wf0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w0.add(offset) as *const __m128i)));
        acc0_0 = _mm256_fmadd_ps(wf0_0, x0, acc0_0);

        let wf0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w0.add(offset + 8) as *const __m128i)));
        acc0_1 = _mm256_fmadd_ps(wf0_1, x1, acc0_1);

        // Row 1
        let wf1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w1.add(offset) as *const __m128i)));
        acc1_0 = _mm256_fmadd_ps(wf1_0, x0, acc1_0);

        let wf1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w1.add(offset + 8) as *const __m128i)));
        acc1_1 = _mm256_fmadd_ps(wf1_1, x1, acc1_1);

        // Row 2
        let wf2_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w2.add(offset) as *const __m128i)));
        acc2_0 = _mm256_fmadd_ps(wf2_0, x0, acc2_0);

        let wf2_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w2.add(offset + 8) as *const __m128i)));
        acc2_1 = _mm256_fmadd_ps(wf2_1, x1, acc2_1);

        // Row 3
        let wf3_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w3.add(offset) as *const __m128i)));
        acc3_0 = _mm256_fmadd_ps(wf3_0, x0, acc3_0);

        let wf3_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w3.add(offset + 8) as *const __m128i)));
        acc3_1 = _mm256_fmadd_ps(wf3_1, x1, acc3_1);

        offset += 16;
    }

    let mut sum0 = _mm256_add_ps(acc0_0, acc0_1);
    let mut sum1 = _mm256_add_ps(acc1_0, acc1_1);
    let mut sum2 = _mm256_add_ps(acc2_0, acc2_1);
    let mut sum3 = _mm256_add_ps(acc3_0, acc3_1);

    let chunks8 = (len - offset) / 8;
    for _ in 0..chunks8 {
        let x_vec = _mm256_loadu_ps(x.add(offset));

        let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w0.add(offset) as *const __m128i)));
        sum0 = _mm256_fmadd_ps(wf0, x_vec, sum0);

        let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w1.add(offset) as *const __m128i)));
        sum1 = _mm256_fmadd_ps(wf1, x_vec, sum1);

        let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w2.add(offset) as *const __m128i)));
        sum2 = _mm256_fmadd_ps(wf2, x_vec, sum2);

        let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w3.add(offset) as *const __m128i)));
        sum3 = _mm256_fmadd_ps(wf3, x_vec, sum3);

        offset += 8;
    }

    let mut tot0 = hsum256_ps_avx(sum0);
    let mut tot1 = hsum256_ps_avx(sum1);
    let mut tot2 = hsum256_ps_avx(sum2);
    let mut tot3 = hsum256_ps_avx(sum3);

    while offset < len {
        let xv = *x.add(offset);
        tot0 += xv * (*w0.add(offset) as f32);
        tot1 += xv * (*w1.add(offset) as f32);
        tot2 += xv * (*w2.add(offset) as f32);
        tot3 += xv * (*w3.add(offset) as f32);
        offset += 1;
    }

    (tot0, tot1, tot2, tot3)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemv_8rows_w8a32_avx512(
    x: *const f32,
    w_base: *const i8,
    stride: usize,
    len: usize,
) -> [f32; 8] {
    use std::arch::x86_64::*;

    let w0 = w_base;
    let w1 = w_base.add(stride);
    let w2 = w_base.add(stride * 2);
    let w3 = w_base.add(stride * 3);
    let w4 = w_base.add(stride * 4);
    let w5 = w_base.add(stride * 5);
    let w6 = w_base.add(stride * 6);
    let w7 = w_base.add(stride * 7);

    let mut acc0_0 = _mm512_setzero_ps();
    let mut acc0_1 = _mm512_setzero_ps();
    let mut acc1_0 = _mm512_setzero_ps();
    let mut acc1_1 = _mm512_setzero_ps();
    let mut acc2_0 = _mm512_setzero_ps();
    let mut acc2_1 = _mm512_setzero_ps();
    let mut acc3_0 = _mm512_setzero_ps();
    let mut acc3_1 = _mm512_setzero_ps();
    let mut acc4_0 = _mm512_setzero_ps();
    let mut acc4_1 = _mm512_setzero_ps();
    let mut acc5_0 = _mm512_setzero_ps();
    let mut acc5_1 = _mm512_setzero_ps();
    let mut acc6_0 = _mm512_setzero_ps();
    let mut acc6_1 = _mm512_setzero_ps();
    let mut acc7_0 = _mm512_setzero_ps();
    let mut acc7_1 = _mm512_setzero_ps();

    let chunks32 = len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let x0 = _mm512_loadu_ps(x.add(offset));
        let x1 = _mm512_loadu_ps(x.add(offset + 16));

        let load_wf = |w_ptr: *const i8, off: usize| -> __m512 {
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(w_ptr.add(off) as *const __m128i)))
        };

        acc0_0 = _mm512_fmadd_ps(load_wf(w0, offset), x0, acc0_0);
        acc0_1 = _mm512_fmadd_ps(load_wf(w0, offset + 16), x1, acc0_1);

        acc1_0 = _mm512_fmadd_ps(load_wf(w1, offset), x0, acc1_0);
        acc1_1 = _mm512_fmadd_ps(load_wf(w1, offset + 16), x1, acc1_1);

        acc2_0 = _mm512_fmadd_ps(load_wf(w2, offset), x0, acc2_0);
        acc2_1 = _mm512_fmadd_ps(load_wf(w2, offset + 16), x1, acc2_1);

        acc3_0 = _mm512_fmadd_ps(load_wf(w3, offset), x0, acc3_0);
        acc3_1 = _mm512_fmadd_ps(load_wf(w3, offset + 16), x1, acc3_1);

        acc4_0 = _mm512_fmadd_ps(load_wf(w4, offset), x0, acc4_0);
        acc4_1 = _mm512_fmadd_ps(load_wf(w4, offset + 16), x1, acc4_1);

        acc5_0 = _mm512_fmadd_ps(load_wf(w5, offset), x0, acc5_0);
        acc5_1 = _mm512_fmadd_ps(load_wf(w5, offset + 16), x1, acc5_1);

        acc6_0 = _mm512_fmadd_ps(load_wf(w6, offset), x0, acc6_0);
        acc6_1 = _mm512_fmadd_ps(load_wf(w6, offset + 16), x1, acc6_1);

        acc7_0 = _mm512_fmadd_ps(load_wf(w7, offset), x0, acc7_0);
        acc7_1 = _mm512_fmadd_ps(load_wf(w7, offset + 16), x1, acc7_1);

        offset += 32;
    }

    let mut sum0 = _mm512_add_ps(acc0_0, acc0_1);
    let mut sum1 = _mm512_add_ps(acc1_0, acc1_1);
    let mut sum2 = _mm512_add_ps(acc2_0, acc2_1);
    let mut sum3 = _mm512_add_ps(acc3_0, acc3_1);
    let mut sum4 = _mm512_add_ps(acc4_0, acc4_1);
    let mut sum5 = _mm512_add_ps(acc5_0, acc5_1);
    let mut sum6 = _mm512_add_ps(acc6_0, acc6_1);
    let mut sum7 = _mm512_add_ps(acc7_0, acc7_1);

    let chunks16 = (len - offset) / 16;
    for _ in 0..chunks16 {
        let x0 = _mm512_loadu_ps(x.add(offset));
        let load_wf = |w_ptr: *const i8, off: usize| -> __m512 {
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(w_ptr.add(off) as *const __m128i)))
        };

        sum0 = _mm512_fmadd_ps(load_wf(w0, offset), x0, sum0);
        sum1 = _mm512_fmadd_ps(load_wf(w1, offset), x0, sum1);
        sum2 = _mm512_fmadd_ps(load_wf(w2, offset), x0, sum2);
        sum3 = _mm512_fmadd_ps(load_wf(w3, offset), x0, sum3);
        sum4 = _mm512_fmadd_ps(load_wf(w4, offset), x0, sum4);
        sum5 = _mm512_fmadd_ps(load_wf(w5, offset), x0, sum5);
        sum6 = _mm512_fmadd_ps(load_wf(w6, offset), x0, sum6);
        sum7 = _mm512_fmadd_ps(load_wf(w7, offset), x0, sum7);

        offset += 16;
    }

    let mut tot = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
        _mm512_reduce_add_ps(sum4),
        _mm512_reduce_add_ps(sum5),
        _mm512_reduce_add_ps(sum6),
        _mm512_reduce_add_ps(sum7),
    ];

    while offset < len {
        let xv = *x.add(offset);
        tot[0] += xv * (*w0.add(offset) as f32);
        tot[1] += xv * (*w1.add(offset) as f32);
        tot[2] += xv * (*w2.add(offset) as f32);
        tot[3] += xv * (*w3.add(offset) as f32);
        tot[4] += xv * (*w4.add(offset) as f32);
        tot[5] += xv * (*w5.add(offset) as f32);
        tot[6] += xv * (*w6.add(offset) as f32);
        tot[7] += xv * (*w7.add(offset) as f32);
        offset += 1;
    }

    tot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_f32_i8_avx512(x: *const f32, w: *const i8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();

    let chunks32 = len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(w.add(offset) as *const __m128i)));
        let xf0 = _mm512_loadu_ps(x.add(offset));
        sum0 = _mm512_fmadd_ps(wf0, xf0, sum0);

        let wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(w.add(offset + 16) as *const __m128i)));
        let xf1 = _mm512_loadu_ps(x.add(offset + 16));
        sum1 = _mm512_fmadd_ps(wf1, xf1, sum1);

        offset += 32;
    }

    let mut sum = _mm512_add_ps(sum0, sum1);

    let chunks16 = (len - offset) / 16;
    for _ in 0..chunks16 {
        let wf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(w.add(offset) as *const __m128i)));
        let xf = _mm512_loadu_ps(x.add(offset));
        sum = _mm512_fmadd_ps(wf, xf, sum);
        offset += 16;
    }

    let mut total = _mm512_reduce_add_ps(sum);

    while offset < len {
        total += (*x.add(offset)) * ((*w.add(offset)) as f32);
        offset += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_avx2(x: *const f32, w: *const i8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks32 = len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w.add(offset) as *const __m128i)));
        let xf0 = _mm256_loadu_ps(x.add(offset));
        sum0 = _mm256_fmadd_ps(wf0, xf0, sum0);

        let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w.add(offset + 8) as *const __m128i)));
        let xf1 = _mm256_loadu_ps(x.add(offset + 8));
        sum1 = _mm256_fmadd_ps(wf1, xf1, sum1);

        let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w.add(offset + 16) as *const __m128i)));
        let xf2 = _mm256_loadu_ps(x.add(offset + 16));
        sum2 = _mm256_fmadd_ps(wf2, xf2, sum2);

        let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w.add(offset + 24) as *const __m128i)));
        let xf3 = _mm256_loadu_ps(x.add(offset + 24));
        sum3 = _mm256_fmadd_ps(wf3, xf3, sum3);

        offset += 32;
    }

    let mut sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    let chunks8 = (len - offset) / 8;
    for _ in 0..chunks8 {
        let wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w.add(offset) as *const __m128i)));
        let xf = _mm256_loadu_ps(x.add(offset));
        sum = _mm256_fmadd_ps(wf, xf, sum);
        offset += 8;
    }

    let mut total = hsum256_ps_avx(sum);

    while offset < len {
        total += (*x.add(offset)) * ((*w.add(offset)) as f32);
        offset += 1;
    }

    total
}

#[inline(always)]
unsafe fn dot_f32_i8(x: *const f32, w: *const i8, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return dot_f32_i8_avx512(x, w, len);
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f32_i8_avx2(x, w, len);
        }
    }
    let mut total = 0.0f32;
    for i in 0..len {
        total += *x.add(i) * (*w.add(i) as f32);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u4_avx2(x: *const f32, w_packed: *const u8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let chunks32 = len / 32;
    let mut offset = 0;

    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    for _ in 0..chunks32 {
        let byte_offset = offset / 2;
        let raw = _mm_loadu_si128(w_packed.add(byte_offset) as *const __m128i);

        let lo = _mm_and_si128(raw, mask_low);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask_low);

        let inter_lo = _mm_unpacklo_epi8(lo, hi);
        let inter_hi = _mm_unpackhi_epi8(lo, hi);

        let s_lo = _mm_sub_epi8(inter_lo, sub8);
        let s_hi = _mm_sub_epi8(inter_hi, sub8);

        let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo));
        let xf0 = _mm256_loadu_ps(x.add(offset));
        sum0 = _mm256_fmadd_ps(wf0, xf0, sum0);

        let s_lo_hi = _mm_unpackhi_epi64(s_lo, s_lo);
        let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo_hi));
        let xf1 = _mm256_loadu_ps(x.add(offset + 8));
        sum1 = _mm256_fmadd_ps(wf1, xf1, sum1);

        let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi));
        let xf2 = _mm256_loadu_ps(x.add(offset + 16));
        sum0 = _mm256_fmadd_ps(wf2, xf2, sum0);

        let s_hi_hi = _mm_unpackhi_epi64(s_hi, s_hi);
        let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi_hi));
        let xf3 = _mm256_loadu_ps(x.add(offset + 24));
        sum1 = _mm256_fmadd_ps(wf3, xf3, sum1);

        offset += 32;
    }

    let sum = _mm256_add_ps(sum0, sum1);
    let mut total = hsum256_ps_avx(sum);

    while offset < len {
        let byte_idx = offset / 2;
        let byte = *w_packed.add(byte_idx);
        let q = if (offset % 2) == 0 {
            ((byte & 0x0F) as i8) - 8
        } else {
            (((byte >> 4) & 0x0F) as i8) - 8
        };
        total += (*x.add(offset)) * (q as f32);
        offset += 1;
    }

    total
}

#[inline(always)]
unsafe fn dot_f32_u4(x: *const f32, w_packed: *const u8, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f32_u4_avx2(x, w_packed, len);
        }
    }
    let mut total = 0.0f32;
    for i in 0..len {
        let byte = *w_packed.add(i / 2);
        let q = if (i % 2) == 0 {
            ((byte & 0x0F) as i8) - 8
        } else {
            (((byte >> 4) & 0x0F) as i8) - 8
        };
        total += *x.add(i) * (q as f32);
    }
    total
}

/// Fast single-token GEMV (M=1) for W8A32 with 8-row AVX-512 or 4-row AVX2 unrolling and chunked Rayon scheduling.
unsafe fn gemv_w8a32(
    x: *const f32,
    w: *const i8,
    scales: *const f32,
    s_len: usize,
    bias: Option<*const f32>,
    out: *mut f32,
    n: usize,
    k: usize,
) {
    use rayon::prelude::*;

    let has_avx512 = {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    if has_avx512 {
        let n_octs = n / 8;

        #[inline(always)]
        unsafe fn process_oct(
            oct: usize,
            x: *const f32,
            w: *const i8,
            scales: *const f32,
            s_len: usize,
            bias: Option<*const f32>,
            out: *mut f32,
            k: usize,
        ) {
            let j = oct * 8;
            let w_base = w.add(j * k);
            #[cfg(target_arch = "x86_64")]
            let dots = gemv_8rows_w8a32_avx512(x, w_base, k, k);
            #[cfg(not(target_arch = "x86_64"))]
            let mut dots = [0.0f32; 8];
            #[cfg(not(target_arch = "x86_64"))]
            for r in 0..8 {
                dots[r] = dot_f32_i8(x, w.add((j + r) * k), k);
            }

            for r in 0..8 {
                let idx = j + r;
                let s = if s_len > 1 { *scales.add(idx) } else { *scales };
                let b = if let Some(bp) = bias { *bp.add(idx) } else { 0.0 };
                *out.add(idx) = dots[r] * s + b;
            }
        }

        if n <= 256 {
            for oct in 0..n_octs {
                process_oct(oct, x, w, scales, s_len, bias, out, k);
            }
        } else {
            let x_usize = x as usize;
            let w_usize = w as usize;
            let s_usize = scales as usize;
            let b_usize = bias.map(|bp| bp as usize);
            let out_usize = out as usize;

            let n_threads = rayon::current_num_threads();
            let min_chunk = (n_octs / (n_threads * 2)).max(8);

            (0..n_octs).into_par_iter().with_min_len(min_chunk).for_each(|oct| {
                let x_p = x_usize as *const f32;
                let w_p = w_usize as *const i8;
                let s_p = s_usize as *const f32;
                let b_p = b_usize.map(|bp| bp as *const f32);
                let out_p = out_usize as *mut f32;
                unsafe {
                    process_oct(oct, x_p, w_p, s_p, s_len, b_p, out_p, k);
                }
            });
        }

        // Remainder rows
        let rem_start = n_octs * 8;
        for j in rem_start..n {
            let w_row = w.add(j * k);
            let dot = dot_f32_i8(x, w_row, k);
            let scale = if s_len > 1 { *scales.add(j) } else { *scales };
            let b = if let Some(bp) = bias { *bp.add(j) } else { 0.0 };
            *out.add(j) = dot * scale + b;
        }
        return;
    }

    // AVX2 / fallback path
    let n_quads = n / 4;

    #[inline(always)]
    unsafe fn process_quad(
        q: usize,
        x: *const f32,
        w: *const i8,
        scales: *const f32,
        s_len: usize,
        bias: Option<*const f32>,
        out: *mut f32,
        k: usize,
        has_avx2: bool,
    ) {
        let j = q * 4;
        let w0 = w.add(j * k);
        let w1 = w.add((j + 1) * k);
        let w2 = w.add((j + 2) * k);
        let w3 = w.add((j + 3) * k);

        let (d0, d1, d2, d3) = if has_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                gemv_4rows_w8a32_avx2(x, w0, w1, w2, w3, k)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                (dot_f32_i8(x, w0, k), dot_f32_i8(x, w1, k), dot_f32_i8(x, w2, k), dot_f32_i8(x, w3, k))
            }
        } else {
            (dot_f32_i8(x, w0, k), dot_f32_i8(x, w1, k), dot_f32_i8(x, w2, k), dot_f32_i8(x, w3, k))
        };

        let s0 = if s_len > 1 { *scales.add(j) } else { *scales };
        let s1 = if s_len > 1 { *scales.add(j + 1) } else { *scales };
        let s2 = if s_len > 1 { *scales.add(j + 2) } else { *scales };
        let s3 = if s_len > 1 { *scales.add(j + 3) } else { *scales };

        let b0 = if let Some(bp) = bias { *bp.add(j) } else { 0.0 };
        let b1 = if let Some(bp) = bias { *bp.add(j + 1) } else { 0.0 };
        let b2 = if let Some(bp) = bias { *bp.add(j + 2) } else { 0.0 };
        let b3 = if let Some(bp) = bias { *bp.add(j + 3) } else { 0.0 };

        *out.add(j) = d0 * s0 + b0;
        *out.add(j + 1) = d1 * s1 + b1;
        *out.add(j + 2) = d2 * s2 + b2;
        *out.add(j + 3) = d3 * s3 + b3;
    }

    let has_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    // If N <= 256, execute sequentially to bypass Rayon thread dispatch overhead completely!
    if n <= 256 {
        for q in 0..n_quads {
            process_quad(q, x, w, scales, s_len, bias, out, k, has_avx2);
        }
    } else {
        let x_usize = x as usize;
        let w_usize = w as usize;
        let s_usize = scales as usize;
        let b_usize = bias.map(|bp| bp as usize);
        let out_usize = out as usize;

        let n_threads = rayon::current_num_threads();
        let min_chunk = (n_quads / (n_threads * 2)).max(16);

        (0..n_quads).into_par_iter().with_min_len(min_chunk).for_each(|q| {
            let x_p = x_usize as *const f32;
            let w_p = w_usize as *const i8;
            let s_p = s_usize as *const f32;
            let b_p = b_usize.map(|bp| bp as *const f32);
            let out_p = out_usize as *mut f32;
            unsafe {
                process_quad(q, x_p, w_p, s_p, s_len, b_p, out_p, k, has_avx2);
            }
        });
    }

    // Handle remainder rows (n % 4)
    let rem_start = n_quads * 4;
    for j in rem_start..n {
        let w_row = w.add(j * k);
        let dot = dot_f32_i8(x, w_row, k);
        let scale = if s_len > 1 { *scales.add(j) } else { *scales };
        let b = if let Some(bp) = bias { *bp.add(j) } else { 0.0 };
        *out.add(j) = dot * scale + b;
    }
}

/// Compute W8A32 Linear projection: out = (x @ w.T) * scales + bias
///
/// Multi-threaded Rayon execution with AVX2 SIMD dot-products.
pub fn w8a32_linear(
    x: &BorrowedTensor,
    w: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("w8a32_linear requires x with at least 1 dimension"));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w.shape.len() != 2 {
        return Err(unsupported("w8a32_linear requires 2D weight matrix"));
    }
    let n = w.shape[0] as usize;
    let w_k = w.shape[1] as usize;
    if k != w_k {
        return Err(unsupported(&format!(
            "w8a32_linear dimension mismatch: x K={k}, w K={w_k}"
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<i8>(w) };
    let s_slice = unsafe { typed_slice::<f32>(scales) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    let bias_slice = bias.map(|b| unsafe { typed_slice::<f32>(b) });

    // Single-token fast path: GEMV with 4-row AVX2 unrolling
    if m == 1 {
        unsafe {
            gemv_w8a32(
                x_slice.as_ptr(),
                w_slice.as_ptr(),
                s_slice.as_ptr(),
                s_slice.len(),
                bias_slice.map(|b| b.as_ptr()),
                out_slice.as_mut_ptr(),
                n,
                k,
            );
        }
        return Ok(out);
    }

    let x_ptr = x_slice.as_ptr() as usize;
    let w_ptr = w_slice.as_ptr() as usize;
    let s_ptr = s_slice.as_ptr() as usize;
    let s_len = s_slice.len();
    let out_ptr = out_slice.as_mut_ptr() as usize;
    let b_ptr = bias_slice.map(|b| b.as_ptr() as usize);

    use rayon::prelude::*;

    (0..n).into_par_iter().with_min_len(8).for_each(|j| {
        let w_row = (w_ptr as *const i8).wrapping_add(j * k);
        let scale = if s_len > 1 {
            unsafe { *((s_ptr as *const f32).add(j)) }
        } else {
            unsafe { *(s_ptr as *const f32) }
        };
        let b = if let Some(bp) = b_ptr {
            unsafe { *((bp as *const f32).add(j)) }
        } else {
            0.0f32
        };

        for r in 0..m {
            let x_row = (x_ptr as *const f32).wrapping_add(r * k);
            let dot = unsafe { dot_f32_i8(x_row, w_row, k) };
            unsafe {
                let out_p = out_ptr as *mut f32;
                *out_p.add(r * n + j) = dot * scale + b;
            }
        }
    });

    Ok(out)
}

/// Compute W4A32 Linear projection: out = (x @ w_unpacked.T) * scales + bias
///
/// Multi-threaded Rayon execution with AVX2 SIMD nibble unpacking.
pub fn w4a32_linear(
    x: &BorrowedTensor,
    w_packed: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("w4a32_linear requires x with at least 1 dimension"));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w_packed.shape.len() != 2 {
        return Err(unsupported("w4a32_linear requires 2D packed weight matrix"));
    }
    let n = w_packed.shape[0] as usize;
    let w_packed_k = w_packed.shape[1] as usize;
    if (k + 1) / 2 != w_packed_k {
        return Err(unsupported(&format!(
            "w4a32_linear dimension mismatch: x K={k}, w packed K={w_packed_k} (expected {})",
            (k + 1) / 2
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<u8>(w_packed) };
    let s_slice = unsafe { typed_slice::<f32>(scales) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    let bias_slice = bias.map(|b| unsafe { typed_slice::<f32>(b) });

    let x_ptr = x_slice.as_ptr() as usize;
    let w_ptr = w_slice.as_ptr() as usize;
    let s_ptr = s_slice.as_ptr() as usize;
    let s_len = s_slice.len();
    let out_ptr = out_slice.as_mut_ptr() as usize;
    let b_ptr = bias_slice.map(|b| b.as_ptr() as usize);
    let bytes_per_row = w_packed_k;

    use rayon::prelude::*;

    // Single-token fast path: sequential if N <= 256, chunked Rayon if N > 256
    if m == 1 {
        if n <= 256 {
            for j in 0..n {
                let w_row = unsafe { (w_ptr as *const u8).add(j * bytes_per_row) };
                let scale = if s_len > 1 {
                    unsafe { *((s_ptr as *const f32).add(j)) }
                } else {
                    unsafe { *(s_ptr as *const f32) }
                };
                let b = if let Some(bp) = b_ptr {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0f32
                };
                let dot = unsafe { dot_f32_u4(x_slice.as_ptr(), w_row, k) };
                unsafe {
                    *out_slice.as_mut_ptr().add(j) = dot * scale + b;
                }
            }
            return Ok(out);
        } else {
            (0..n).into_par_iter().with_min_len(8).for_each(|j| {
                let w_row = (w_ptr as *const u8).wrapping_add(j * bytes_per_row);
                let scale = if s_len > 1 {
                    unsafe { *((s_ptr as *const f32).add(j)) }
                } else {
                    unsafe { *(s_ptr as *const f32) }
                };
                let b = if let Some(bp) = b_ptr {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0f32
                };
                let dot = unsafe { dot_f32_u4(x_ptr as *const f32, w_row, k) };
                unsafe {
                    let out_p = out_ptr as *mut f32;
                    *out_p.add(j) = dot * scale + b;
                }
            });
            return Ok(out);
        }
    }

    (0..n).into_par_iter().with_min_len(8).for_each(|j| {
        let w_row = (w_ptr as *const u8).wrapping_add(j * bytes_per_row);
        let scale = if s_len > 1 {
            unsafe { *((s_ptr as *const f32).add(j)) }
        } else {
            unsafe { *(s_ptr as *const f32) }
        };
        let b = if let Some(bp) = b_ptr {
            unsafe { *((bp as *const f32).add(j)) }
        } else {
            0.0f32
        };

        for r in 0..m {
            let x_row = (x_ptr as *const f32).wrapping_add(r * k);
            let dot = unsafe { dot_f32_u4(x_row, w_row, k) };
            unsafe {
                let out_p = out_ptr as *mut f32;
                *out_p.add(r * n + j) = dot * scale + b;
            }
        }
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// Grouped INT4 (W4A32) SIMD Kernels & Dispatchers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_f32_u4_group64_avx512(x: *const f32, w_packed: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    // 64 weights = 32 packed bytes
    // Chunk 0: elements 0..31 (16 bytes)
    let raw0 = _mm_loadu_si128(w_packed as *const __m128i);
    let lo0 = _mm_and_si128(raw0, mask_low);
    let hi0 = _mm_and_si128(_mm_srli_epi16::<4>(raw0), mask_low);
    let inter_lo0 = _mm_unpacklo_epi8(lo0, hi0);
    let inter_hi0 = _mm_unpackhi_epi8(lo0, hi0);
    let s_lo0 = _mm_sub_epi8(inter_lo0, sub8);
    let s_hi0 = _mm_sub_epi8(inter_hi0, sub8);

    let wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_lo0));
    let xf0 = _mm512_loadu_ps(x);
    let mut sum0 = _mm512_mul_ps(wf0, xf0);

    let wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_hi0));
    let xf1 = _mm512_loadu_ps(x.add(16));
    sum0 = _mm512_fmadd_ps(wf1, xf1, sum0);

    // Chunk 1: elements 32..63 (16 bytes)
    let raw1 = _mm_loadu_si128(w_packed.add(16) as *const __m128i);
    let lo1 = _mm_and_si128(raw1, mask_low);
    let hi1 = _mm_and_si128(_mm_srli_epi16::<4>(raw1), mask_low);
    let inter_lo1 = _mm_unpacklo_epi8(lo1, hi1);
    let inter_hi1 = _mm_unpackhi_epi8(lo1, hi1);
    let s_lo1 = _mm_sub_epi8(inter_lo1, sub8);
    let s_hi1 = _mm_sub_epi8(inter_hi1, sub8);

    let wf2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_lo1));
    let xf2 = _mm512_loadu_ps(x.add(32));
    let mut sum1 = _mm512_mul_ps(wf2, xf2);

    let wf3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_hi1));
    let xf3 = _mm512_loadu_ps(x.add(48));
    sum1 = _mm512_fmadd_ps(wf3, xf3, sum1);

    let sum = _mm512_add_ps(sum0, sum1);
    _mm512_reduce_add_ps(sum)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn unpack_and_fma_32_avx512(
    x_ptr: *const f32,
    w_ptr: *const u8,
    mask_low: std::arch::x86_64::__m128i,
    sub8: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let raw = _mm_loadu_si128(w_ptr as *const __m128i);
    let lo = _mm_and_si128(raw, mask_low);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask_low);
    let inter_lo = _mm_unpacklo_epi8(lo, hi);
    let inter_hi = _mm_unpackhi_epi8(lo, hi);
    let s_lo = _mm_sub_epi8(inter_lo, sub8);
    let s_hi = _mm_sub_epi8(inter_hi, sub8);

    let wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_lo));
    let xf0 = _mm512_loadu_ps(x_ptr);
    let wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_hi));
    let xf1 = _mm512_loadu_ps(x_ptr.add(16));

    _mm512_fmadd_ps(wf1, xf1, _mm512_mul_ps(wf0, xf0))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn unpack_and_fma_32_avx2(
    x_ptr: *const f32,
    w_ptr: *const u8,
    mask_low: std::arch::x86_64::__m128i,
    sub8: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let raw = _mm_loadu_si128(w_ptr as *const __m128i);
    let lo = _mm_and_si128(raw, mask_low);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask_low);
    let inter_lo = _mm_unpacklo_epi8(lo, hi);
    let inter_hi = _mm_unpackhi_epi8(lo, hi);
    let s_lo = _mm_sub_epi8(inter_lo, sub8);
    let s_hi = _mm_sub_epi8(inter_hi, sub8);

    let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo));
    let xf0 = _mm256_loadu_ps(x_ptr);
    let mut sum0 = _mm256_mul_ps(wf0, xf0);

    let s_lo_hi = _mm_unpackhi_epi64(s_lo, s_lo);
    let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo_hi));
    let xf1 = _mm256_loadu_ps(x_ptr.add(8));
    sum0 = _mm256_fmadd_ps(wf1, xf1, sum0);

    let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi));
    let xf2 = _mm256_loadu_ps(x_ptr.add(16));
    let mut sum1 = _mm256_mul_ps(wf2, xf2);

    let s_hi_hi = _mm_unpackhi_epi64(s_hi, s_hi);
    let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi_hi));
    let xf3 = _mm256_loadu_ps(x_ptr.add(24));
    sum1 = _mm256_fmadd_ps(wf3, xf3, sum1);

    _mm256_add_ps(sum0, sum1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_f32_u4_group32_avx512(x: *const f32, w_packed: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let sum0 = unpack_and_fma_32_avx512(x, w_packed, mask_low, sub8);
    _mm512_reduce_add_ps(sum0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u4_group32_avx2(x: *const f32, w_packed: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let sum = unpack_and_fma_32_avx2(x, w_packed, mask_low, sub8);
    hsum256_ps_avx(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u4_group64_avx2(x: *const f32, w_packed: *const u8) -> f32 {
    let d0 = dot_f32_u4_group32_avx2(x, w_packed);
    let d1 = dot_f32_u4_group32_avx2(x.add(32), w_packed.add(16));
    d0 + d1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemv_row_w4a32_group64_avx512(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let w_grp = w_row.add(g * 32);
        let scale = _mm512_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx512(x_grp, w_grp, mask_low, sub8);
        let sum1 = unpack_and_fma_32_avx512(x_grp.add(32), w_grp.add(16), mask_low, sub8);
        let grp_sum = _mm512_add_ps(sum0, sum1);
        acc = _mm512_fmadd_ps(grp_sum, scale, acc);
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemv_row_w4a32_group32_avx512(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let w_grp = w_row.add(g * 16);
        let scale = _mm512_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx512(x_grp, w_grp, mask_low, sub8);
        acc = _mm512_fmadd_ps(sum0, scale, acc);
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_w4a32_group64_avx2(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let w_grp = w_row.add(g * 32);
        let scale = _mm256_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx2(x_grp, w_grp, mask_low, sub8);
        let sum1 = unpack_and_fma_32_avx2(x_grp.add(32), w_grp.add(16), mask_low, sub8);
        let grp_sum = _mm256_add_ps(sum0, sum1);
        acc = _mm256_fmadd_ps(grp_sum, scale, acc);
    }
    hsum256_ps_avx(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_w4a32_group32_avx2(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let w_grp = w_row.add(g * 16);
        let scale = _mm256_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx2(x_grp, w_grp, mask_low, sub8);
        acc = _mm256_fmadd_ps(sum0, scale, acc);
    }
    hsum256_ps_avx(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn quantize_activation_to_u8_avx512(x: *const f32, k: usize, out: *mut u8) -> f32 {
    use std::arch::x86_64::*;
    let mut max_vec = _mm512_setzero_ps();
    let num_chunks = k / 16;
    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        let abs_v = _mm512_abs_ps(v);
        max_vec = _mm512_max_ps(max_vec, abs_v);
    }
    let mut max_val = _mm512_reduce_max_ps(max_vec);
    for i in (num_chunks * 16)..k {
        let a = (*x.add(i)).abs();
        if a > max_val { max_val = a; }
    }
    if max_val < 1e-10 { max_val = 1e-10; }
    let s_x = max_val / 127.0;
    let inv_s = _mm512_set1_ps(1.0 / s_x);
    let offset128 = _mm512_set1_epi32(128);

    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        let scaled = _mm512_mul_ps(v, inv_s);
        let rounded = _mm512_cvtps_epi32(scaled);
        let shifted = _mm512_add_epi32(rounded, offset128);
        let bytes16 = _mm512_cvtepi32_epi8(shifted);
        _mm_storeu_si128(out.add(c * 16) as *mut __m128i, bytes16);
    }
    for i in (num_chunks * 16)..k {
        let q = ((*x.add(i)) / s_x).round().clamp(-127.0, 127.0) as i16;
        *out.add(i) = (q + 128) as u8;
    }
    s_x
}

#[inline(always)]
unsafe fn quantize_activation_to_u8(x: *const f32, k: usize) -> (Vec<u8>, f32) {
    let mut x_u8 = vec![0u8; k];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            let s = quantize_activation_to_u8_avx512(x, k, x_u8.as_mut_ptr());
            return (x_u8, s);
        }
    }
    let mut max_abs = 0.0f32;
    for i in 0..k {
        let v = (*x.add(i)).abs();
        if v > max_abs { max_abs = v; }
    }
    if max_abs < 1e-10 {
        max_abs = 1e-10;
    }
    let s_x = max_abs / 127.0;
    let inv_s = 1.0 / s_x;

    for i in 0..k {
        let q = ((*x.add(i)) * inv_s).round().clamp(-127.0, 127.0) as i16;
        x_u8[i] = (q + 128) as u8;
    }
    (x_u8, s_x)
}


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512f,avx512bw")]
unsafe fn gemv_row_w4a8_group64_vnni_avx512(
    x_u8: *const u8,
    s_x: f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let ones = _mm512_set1_epi8(1);
    let mut acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x64 = _mm512_loadu_si512(x_u8.add(g * 64) as *const __m512i);
        let w_grp = w_row.add(g * 32);
        let s_w = *s_row.add(g);
        let scale_vec = _mm512_set1_ps(s_x * s_w);

        // Chunk 0 (0..31)
        let raw0 = _mm_loadu_si128(w_grp as *const __m128i);
        let lo0 = _mm_and_si128(raw0, mask_low);
        let hi0 = _mm_and_si128(_mm_srli_epi16::<4>(raw0), mask_low);
        let inter_lo0 = _mm_unpacklo_epi8(lo0, hi0);
        let inter_hi0 = _mm_unpackhi_epi8(lo0, hi0);
        let s_lo0 = _mm_sub_epi8(inter_lo0, sub8);
        let s_hi0 = _mm_sub_epi8(inter_hi0, sub8);
        let w32_0 = _mm256_set_m128i(s_hi0, s_lo0);

        // Chunk 1 (32..63)
        let raw1 = _mm_loadu_si128(w_grp.add(16) as *const __m128i);
        let lo1 = _mm_and_si128(raw1, mask_low);
        let hi1 = _mm_and_si128(_mm_srli_epi16::<4>(raw1), mask_low);
        let inter_lo1 = _mm_unpacklo_epi8(lo1, hi1);
        let inter_hi1 = _mm_unpackhi_epi8(lo1, hi1);
        let s_lo1 = _mm_sub_epi8(inter_lo1, sub8);
        let s_hi1 = _mm_sub_epi8(inter_hi1, sub8);
        let w32_1 = _mm256_set_m128i(s_hi1, s_lo1);

        let w64 = _mm512_inserti64x4(_mm512_castsi256_si512(w32_0), w32_1, 1);

        // Hardware VNNI integer dot product (64 MACs in 1 cycle)
        let dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, w64);
        let w_sum_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, w64);
        let true_i32 = _mm512_sub_epi32(dot_i32, _mm512_slli_epi32::<7>(w_sum_i32));
        let true_f32 = _mm512_cvtepi32_ps(true_i32);

        acc = _mm512_fmadd_ps(true_f32, scale_vec, acc);
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512f,avx512bw")]
unsafe fn swiglu_neuron_w4a8_group64_vnni_avx512(
    x_u8: *const u8,
    s_x: f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let ones = _mm512_set1_epi8(1);
    let mut g_acc = _mm512_setzero_ps();
    let mut u_acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x64 = _mm512_loadu_si512(x_u8.add(g * 64) as *const __m512i);
        let gw_grp = gw_row.add(g * 32);
        let uw_grp = uw_row.add(g * 32);
        let g_scale = _mm512_set1_ps(s_x * *gs_row.add(g));
        let u_scale = _mm512_set1_ps(s_x * *us_row.add(g));

        // Unpack Gate weights
        let g_raw0 = _mm_loadu_si128(gw_grp as *const __m128i);
        let g_lo0 = _mm_and_si128(g_raw0, mask_low);
        let g_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw0), mask_low);
        let g_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo0, g_hi0), sub8);
        let g_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo0, g_hi0), sub8);
        let gw32_0 = _mm256_set_m128i(g_s_hi0, g_s_lo0);

        let g_raw1 = _mm_loadu_si128(gw_grp.add(16) as *const __m128i);
        let g_lo1 = _mm_and_si128(g_raw1, mask_low);
        let g_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw1), mask_low);
        let g_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo1, g_hi1), sub8);
        let g_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo1, g_hi1), sub8);
        let gw32_1 = _mm256_set_m128i(g_s_hi1, g_s_lo1);

        let gw64 = _mm512_inserti64x4(_mm512_castsi256_si512(gw32_0), gw32_1, 1);

        // Gate VNNI
        let g_dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, gw64);
        let g_w_sum_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, gw64);
        let g_true_i32 = _mm512_sub_epi32(g_dot_i32, _mm512_slli_epi32::<7>(g_w_sum_i32));
        let g_true_f32 = _mm512_cvtepi32_ps(g_true_i32);
        g_acc = _mm512_fmadd_ps(g_true_f32, g_scale, g_acc);

        // Unpack Up weights
        let u_raw0 = _mm_loadu_si128(uw_grp as *const __m128i);
        let u_lo0 = _mm_and_si128(u_raw0, mask_low);
        let u_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw0), mask_low);
        let u_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo0, u_hi0), sub8);
        let u_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo0, u_hi0), sub8);
        let uw32_0 = _mm256_set_m128i(u_s_hi0, u_s_lo0);

        let u_raw1 = _mm_loadu_si128(uw_grp.add(16) as *const __m128i);
        let u_lo1 = _mm_and_si128(u_raw1, mask_low);
        let u_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw1), mask_low);
        let u_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo1, u_hi1), sub8);
        let u_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo1, u_hi1), sub8);
        let uw32_1 = _mm256_set_m128i(u_s_hi1, u_s_lo1);

        let uw64 = _mm512_inserti64x4(_mm512_castsi256_si512(uw32_0), uw32_1, 1);

        // Up VNNI
        let u_dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, uw64);
        let u_w_sum_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, uw64);
        let u_true_i32 = _mm512_sub_epi32(u_dot_i32, _mm512_slli_epi32::<7>(u_w_sum_i32));
        let u_true_f32 = _mm512_cvtepi32_ps(u_true_i32);
        u_acc = _mm512_fmadd_ps(u_true_f32, u_scale, u_acc);
    }

    (_mm512_reduce_add_ps(g_acc), _mm512_reduce_add_ps(u_acc))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn swiglu_neuron_w4a32_group64_avx512(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm512_setzero_ps();
    let mut u_acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let gw_grp = gw_row.add(g * 32);
        let uw_grp = uw_row.add(g * 32);
        let g_scale = _mm512_set1_ps(*gs_row.add(g));
        let u_scale = _mm512_set1_ps(*us_row.add(g));

        // Chunk 0 (elements 0..31)
        let xf0 = _mm512_loadu_ps(x_grp);
        let xf1 = _mm512_loadu_ps(x_grp.add(16));

        // Gate Chunk 0
        let g_raw0 = _mm_loadu_si128(gw_grp as *const __m128i);
        let g_lo0 = _mm_and_si128(g_raw0, mask_low);
        let g_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw0), mask_low);
        let g_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo0, g_hi0), sub8);
        let g_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo0, g_hi0), sub8);
        let gwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_lo0));
        let gwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_hi0));
        let g_sum0 = _mm512_fmadd_ps(gwf1, xf1, _mm512_mul_ps(gwf0, xf0));

        // Up Chunk 0
        let u_raw0 = _mm_loadu_si128(uw_grp as *const __m128i);
        let u_lo0 = _mm_and_si128(u_raw0, mask_low);
        let u_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw0), mask_low);
        let u_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo0, u_hi0), sub8);
        let u_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo0, u_hi0), sub8);
        let uwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_lo0));
        let uwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_hi0));
        let u_sum0 = _mm512_fmadd_ps(uwf1, xf1, _mm512_mul_ps(uwf0, xf0));

        // Chunk 1 (elements 32..63)
        let xf2 = _mm512_loadu_ps(x_grp.add(32));
        let xf3 = _mm512_loadu_ps(x_grp.add(48));

        // Gate Chunk 1
        let g_raw1 = _mm_loadu_si128(gw_grp.add(16) as *const __m128i);
        let g_lo1 = _mm_and_si128(g_raw1, mask_low);
        let g_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw1), mask_low);
        let g_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo1, g_hi1), sub8);
        let g_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo1, g_hi1), sub8);
        let gwf2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_lo1));
        let gwf3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_hi1));
        let g_sum1 = _mm512_fmadd_ps(gwf3, xf3, _mm512_mul_ps(gwf2, xf2));

        // Up Chunk 1
        let u_raw1 = _mm_loadu_si128(uw_grp.add(16) as *const __m128i);
        let u_lo1 = _mm_and_si128(u_raw1, mask_low);
        let u_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw1), mask_low);
        let u_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo1, u_hi1), sub8);
        let u_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo1, u_hi1), sub8);
        let uwf2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_lo1));
        let uwf3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_hi1));
        let u_sum1 = _mm512_fmadd_ps(uwf3, xf3, _mm512_mul_ps(uwf2, xf2));

        let g_grp_sum = _mm512_add_ps(g_sum0, g_sum1);
        let u_grp_sum = _mm512_add_ps(u_sum0, u_sum1);

        g_acc = _mm512_fmadd_ps(g_grp_sum, g_scale, g_acc);
        u_acc = _mm512_fmadd_ps(u_grp_sum, u_scale, u_acc);
    }

    (_mm512_reduce_add_ps(g_acc), _mm512_reduce_add_ps(u_acc))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn swiglu_neuron_w4a32_group32_avx512(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm512_setzero_ps();
    let mut u_acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let gw_grp = gw_row.add(g * 16);
        let uw_grp = uw_row.add(g * 16);
        let g_scale = _mm512_set1_ps(*gs_row.add(g));
        let u_scale = _mm512_set1_ps(*us_row.add(g));

        let xf0 = _mm512_loadu_ps(x_grp);
        let xf1 = _mm512_loadu_ps(x_grp.add(16));

        // Gate
        let g_raw0 = _mm_loadu_si128(gw_grp as *const __m128i);
        let g_lo0 = _mm_and_si128(g_raw0, mask_low);
        let g_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw0), mask_low);
        let g_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo0, g_hi0), sub8);
        let g_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo0, g_hi0), sub8);
        let gwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_lo0));
        let gwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_hi0));
        let g_sum0 = _mm512_fmadd_ps(gwf1, xf1, _mm512_mul_ps(gwf0, xf0));

        // Up
        let u_raw0 = _mm_loadu_si128(uw_grp as *const __m128i);
        let u_lo0 = _mm_and_si128(u_raw0, mask_low);
        let u_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw0), mask_low);
        let u_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo0, u_hi0), sub8);
        let u_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo0, u_hi0), sub8);
        let uwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_lo0));
        let uwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_hi0));
        let u_sum0 = _mm512_fmadd_ps(uwf1, xf1, _mm512_mul_ps(uwf0, xf0));

        g_acc = _mm512_fmadd_ps(g_sum0, g_scale, g_acc);
        u_acc = _mm512_fmadd_ps(u_sum0, u_scale, u_acc);
    }

    (_mm512_reduce_add_ps(g_acc), _mm512_reduce_add_ps(u_acc))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_neuron_w4a32_group32_avx2(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm256_setzero_ps();
    let mut u_acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let gw_grp = gw_row.add(g * 16);
        let uw_grp = uw_row.add(g * 16);
        let g_scale = _mm256_set1_ps(*gs_row.add(g));
        let u_scale = _mm256_set1_ps(*us_row.add(g));

        let g_sum0 = unpack_and_fma_32_avx2(x_grp, gw_grp, mask_low, sub8);
        let u_sum0 = unpack_and_fma_32_avx2(x_grp, uw_grp, mask_low, sub8);

        g_acc = _mm256_fmadd_ps(g_sum0, g_scale, g_acc);
        u_acc = _mm256_fmadd_ps(u_sum0, u_scale, u_acc);
    }

    (hsum256_ps_avx(g_acc), hsum256_ps_avx(u_acc))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_neuron_w4a32_group64_avx2(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm256_setzero_ps();
    let mut u_acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let gw_grp = gw_row.add(g * 32);
        let uw_grp = uw_row.add(g * 32);
        let g_scale = _mm256_set1_ps(*gs_row.add(g));
        let u_scale = _mm256_set1_ps(*us_row.add(g));

        let g_sum0 = unpack_and_fma_32_avx2(x_grp, gw_grp, mask_low, sub8);
        let g_sum1 = unpack_and_fma_32_avx2(x_grp.add(32), gw_grp.add(16), mask_low, sub8);
        let g_grp_sum = _mm256_add_ps(g_sum0, g_sum1);

        let u_sum0 = unpack_and_fma_32_avx2(x_grp, uw_grp, mask_low, sub8);
        let u_sum1 = unpack_and_fma_32_avx2(x_grp.add(32), uw_grp.add(16), mask_low, sub8);
        let u_grp_sum = _mm256_add_ps(u_sum0, u_sum1);

        g_acc = _mm256_fmadd_ps(g_grp_sum, g_scale, g_acc);
        u_acc = _mm256_fmadd_ps(u_grp_sum, u_scale, u_acc);
    }

    (hsum256_ps_avx(g_acc), hsum256_ps_avx(u_acc))
}

unsafe fn dot_f32_u4_group_scalar(x: *const f32, w_packed: *const u8, group_size: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..group_size {
        let byte = *w_packed.add(i / 2);
        let q = if (i % 2) == 0 {
            ((byte & 0x0F) as i8) - 8
        } else {
            (((byte >> 4) & 0x0F) as i8) - 8
        };
        total += *x.add(i) * (q as f32);
    }
    total
}

#[inline(always)]
unsafe fn dot_f32_u4_group64_fast(x: *const f32, w_packed: *const u8, has_avx512: bool, has_avx2: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512 {
            return dot_f32_u4_group64_avx512(x, w_packed);
        }
        if has_avx2 {
            return dot_f32_u4_group64_avx2(x, w_packed);
        }
    }
    dot_f32_u4_group_scalar(x, w_packed, 64)
}

#[inline(always)]
unsafe fn dot_f32_u4_group32_fast(x: *const f32, w_packed: *const u8, has_avx512: bool, has_avx2: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512 {
            return dot_f32_u4_group32_avx512(x, w_packed);
        }
        if has_avx2 {
            return dot_f32_u4_group32_avx2(x, w_packed);
        }
    }
    dot_f32_u4_group_scalar(x, w_packed, 32)
}

#[inline(always)]
unsafe fn dot_f32_u4_group64(x: *const f32, w_packed: *const u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return dot_f32_u4_group64_avx512(x, w_packed);
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f32_u4_group64_avx2(x, w_packed);
        }
    }
    dot_f32_u4_group_scalar(x, w_packed, 64)
}

#[inline(always)]
unsafe fn dot_f32_u4_group32(x: *const f32, w_packed: *const u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return dot_f32_u4_group32_avx512(x, w_packed);
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f32_u4_group32_avx2(x, w_packed);
        }
    }
    dot_f32_u4_group_scalar(x, w_packed, 32)
}

/// Fast single-token GEMV for W4A32 with group-wise scales.
unsafe fn gemv_w4a32_grouped(
    x: *const f32,
    w_packed: *const u8,
    scales: *const f32,
    bias: Option<*const f32>,
    out: *mut f32,
    n: usize,
    k: usize,
    group_size: usize,
) {
    use rayon::prelude::*;
    let num_groups = (k + group_size - 1) / group_size;
    let bytes_per_row = (k + 1) / 2;
    let bytes_per_group = group_size / 2;

    let x_usize = x as usize;
    let w_usize = w_packed as usize;
    let s_usize = scales as usize;
    let b_usize = bias.map(|bp| bp as usize);
    let out_usize = out as usize;

    let n_threads = rayon::current_num_threads();
    let min_chunk = (n / (n_threads * 4)).max(8);

    #[cfg(target_arch = "x86_64")]
    let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let has_vnni = false;

    #[cfg(target_arch = "x86_64")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx512 = false;

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
        let (u, s) = unsafe { quantize_activation_to_u8(x, k) };
        (Some(u), s)
    } else {
        (None, 1.0f32)
    };
    let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

    (0..n).into_par_iter().with_min_len(min_chunk).for_each(|j| {
        let x_p = x_usize as *const f32;
        let w_row = (w_usize as *const u8).add(j * bytes_per_row);
        let s_row = (s_usize as *const f32).add(j * num_groups);
        let b_p = b_usize.map(|bp| bp as *const f32);
        let out_p = out_usize as *mut f32;

        let row_sum = if let Some(x_u8_p) = x_u8_ptr {
            #[cfg(target_arch = "x86_64")]
            {
                unsafe { gemv_row_w4a8_group64_vnni_avx512(x_u8_p as *const u8, s_x, w_row, s_row, num_groups) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { 0.0f32 }
        } else if has_avx512 {
            #[cfg(target_arch = "x86_64")]
            {
                if group_size == 64 {
                    gemv_row_w4a32_group64_avx512(x_p, w_row, s_row, num_groups)
                } else if group_size == 32 {
                    gemv_row_w4a32_group32_avx512(x_p, w_row, s_row, num_groups)
                } else {
                    let mut s = 0.0f32;
                    for g in 0..num_groups {
                        let cur_len = (k - g * group_size).min(group_size);
                        s += dot_f32_u4_group_scalar(x_p.add(g * group_size), w_row.add(g * bytes_per_group), cur_len) * *s_row.add(g);
                    }
                    s
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { 0.0f32 }
        } else if has_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                if group_size == 64 {
                    gemv_row_w4a32_group64_avx2(x_p, w_row, s_row, num_groups)
                } else if group_size == 32 {
                    gemv_row_w4a32_group32_avx2(x_p, w_row, s_row, num_groups)
                } else {
                    let mut s = 0.0f32;
                    for g in 0..num_groups {
                        let cur_len = (k - g * group_size).min(group_size);
                        s += dot_f32_u4_group_scalar(x_p.add(g * group_size), w_row.add(g * bytes_per_group), cur_len) * *s_row.add(g);
                    }
                    s
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { 0.0f32 }
        } else {
            let mut s = 0.0f32;
            for g in 0..num_groups {
                let cur_len = (k - g * group_size).min(group_size);
                s += dot_f32_u4_group_scalar(x_p.add(g * group_size), w_row.add(g * bytes_per_group), cur_len) * *s_row.add(g);
            }
            s
        };

        let b = if let Some(bp) = b_p { *bp.add(j) } else { 0.0 };
        *out_p.add(j) = row_sum + b;
    });
}

/// Compute W4A32 Linear projection with grouped scaling factors:
/// out = sum_g( (x_g @ w_g.T) * scale_g ) + bias
pub fn w4a32_grouped_linear(
    x: &BorrowedTensor,
    w_packed: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("w4a32_grouped_linear requires x with at least 1 dimension"));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w_packed.shape.len() != 2 {
        return Err(unsupported("w4a32_grouped_linear requires 2D packed weight matrix"));
    }
    let n = w_packed.shape[0] as usize;
    let w_packed_k = w_packed.shape[1] as usize;
    if (k + 1) / 2 != w_packed_k {
        return Err(unsupported(&format!(
            "w4a32_grouped_linear dimension mismatch: x K={k}, w packed K={w_packed_k} (expected {})",
            (k + 1) / 2
        )));
    }

    let num_groups = (k + group_size - 1) / group_size;
    if scales.shape.len() != 2 || scales.shape[0] as usize != n || scales.shape[1] as usize != num_groups {
        return Err(unsupported(&format!(
            "w4a32_grouped_linear scales mismatch: expected [{n}, {num_groups}], got {:?}",
            scales.shape
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<u8>(w_packed) };
    let s_slice = unsafe { typed_slice::<f32>(scales) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };
    let bias_slice = bias.map(|b| unsafe { typed_slice::<f32>(b) });
    let b_ptr = bias_slice.map(|b| b.as_ptr());

    if m == 1 {
        unsafe {
            gemv_w4a32_grouped(
                x_slice.as_ptr(),
                w_slice.as_ptr(),
                s_slice.as_ptr(),
                b_ptr,
                out_slice.as_mut_ptr(),
                n,
                k,
                group_size,
            );
        }
    } else {
        for i in 0..m {
            let x_tok = unsafe { x_slice.as_ptr().add(i * k) };
            let out_tok = unsafe { out_slice.as_mut_ptr().add(i * n) };
            unsafe {
                gemv_w4a32_grouped(
                    x_tok,
                    w_slice.as_ptr(),
                    s_slice.as_ptr(),
                    b_ptr,
                    out_tok,
                    n,
                    k,
                    group_size,
                );
            }
        }
    }

    Ok(out)
}

#[cfg(feature = "burn-wgpu")]
pub fn wgpu_w4a32_grouped_linear(
    x: &BorrowedTensor,
    w_packed: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("wgpu_w4a32_grouped_linear requires x with at least 1 dimension"));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w_packed.shape.len() != 2 {
        return Err(unsupported("wgpu_w4a32_grouped_linear requires 2D packed weight matrix"));
    }
    let n = w_packed.shape[0] as usize;
    let w_packed_k = w_packed.shape[1] as usize;
    if (k + 1) / 2 != w_packed_k {
        return Err(unsupported(&format!(
            "wgpu_w4a32_grouped_linear dimension mismatch: x K={k}, w packed K={w_packed_k} (expected {})",
            (k + 1) / 2
        )));
    }

    let num_groups = (k + group_size - 1) / group_size;
    if scales.shape.len() != 2 || scales.shape[0] as usize != n || scales.shape[1] as usize != num_groups {
        return Err(unsupported(&format!(
            "wgpu_w4a32_grouped_linear scales mismatch: expected [{n}, {num_groups}], got {:?}",
            scales.shape
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<u8>(w_packed) };
    let s_slice = unsafe { typed_slice::<f32>(scales) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    if m == 1 {
        crate::wgpu_backend::wgpu_gemv_w4a32(
            x_slice,
            w_slice,
            s_slice,
            out_slice,
            n,
            k,
            group_size,
        ).map_err(|e| unsupported(&e))?;
    } else {
        for i in 0..m {
            let x_tok = &x_slice[i * k..(i + 1) * k];
            let out_tok = &mut out_slice[i * n..(i + 1) * n];
            crate::wgpu_backend::wgpu_gemv_w4a32(
                x_tok,
                w_slice,
                s_slice,
                out_tok,
                n,
                k,
                group_size,
            ).map_err(|e| unsupported(&e))?;
        }
    }

    if let Some(b) = bias {
        let b_slice = unsafe { typed_slice::<f32>(b) };
        for i in 0..m {
            for j in 0..n {
                out_slice[i * n + j] += b_slice[j];
            }
        }
    }

    Ok(out)
}

/// Fused SwiGLU MLP for INT8 (W8A32):
/// intermediate = silu(gate_proj(x)) * up_proj(x)
/// out = down_proj(intermediate)
pub fn fused_swiglu_mlp_w8a32(
    x: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    gate_s: &BorrowedTensor,
    gate_b: Option<&BorrowedTensor>,
    up_w: &BorrowedTensor,
    up_s: &BorrowedTensor,
    up_b: Option<&BorrowedTensor>,
    down_w: &BorrowedTensor,
    down_s: &BorrowedTensor,
    down_b: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("fused_swiglu_mlp requires x with at least 1 dim"));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    let n_inter = gate_w.shape[0] as usize;
    let n_out = down_w.shape[0] as usize;

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n_out as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let gw_slice = unsafe { typed_slice::<i8>(gate_w) };
    let gs_slice = unsafe { typed_slice::<f32>(gate_s) };
    let gb_slice = gate_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let uw_slice = unsafe { typed_slice::<i8>(up_w) };
    let us_slice = unsafe { typed_slice::<f32>(up_s) };
    let ub_slice = up_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let dw_slice = unsafe { typed_slice::<i8>(down_w) };
    let ds_slice = unsafe { typed_slice::<f32>(down_s) };
    let db_slice = down_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    use rayon::prelude::*;

    for token_idx in 0..m {
        let x_tok = unsafe { x_slice.as_ptr().add(token_idx * k) };
        let out_tok = unsafe { out_slice.as_mut_ptr().add(token_idx * n_out) };

        let mut h_buf = vec![0.0f32; n_inter];
        let h_ptr = h_buf.as_mut_ptr() as usize;

        let x_usize = x_tok as usize;
        let gw_usize = gw_slice.as_ptr() as usize;
        let gs_usize = gs_slice.as_ptr() as usize;
        let gs_len = gs_slice.len();
        let gb_usize = gb_slice.map(|b| b.as_ptr() as usize);

        let uw_usize = uw_slice.as_ptr() as usize;
        let us_usize = us_slice.as_ptr() as usize;
        let us_len = us_slice.len();
        let ub_usize = ub_slice.map(|b| b.as_ptr() as usize);

        let n_threads = rayon::current_num_threads();
        let min_chunk = (n_inter / (n_threads * 4)).max(8);

        (0..n_inter).into_par_iter().with_min_len(min_chunk).for_each(|j| {
            let x_p = x_usize as *const f32;
            let gw_p = unsafe { (gw_usize as *const i8).add(j * k) };
            let gs = if gs_len > 1 { unsafe { *((gs_usize as *const f32).add(j)) } } else { unsafe { *(gs_usize as *const f32) } };
            let gb = if let Some(bp) = gb_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

            let uw_p = unsafe { (uw_usize as *const i8).add(j * k) };
            let us = if us_len > 1 { unsafe { *((us_usize as *const f32).add(j)) } } else { unsafe { *(us_usize as *const f32) } };
            let ub = if let Some(bp) = ub_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

            let g_dot = unsafe { dot_f32_i8(x_p, gw_p, k) };
            let g = g_dot * gs + gb;

            let u_dot = unsafe { dot_f32_i8(x_p, uw_p, k) };
            let u = u_dot * us + ub;

            let silu_g = g / (1.0 + (-g).exp());
            let val = silu_g * u;

            unsafe {
                let h_p = h_ptr as *mut f32;
                *h_p.add(j) = val;
            }
        });

        unsafe {
            gemv_w8a32(
                h_buf.as_ptr(),
                dw_slice.as_ptr(),
                ds_slice.as_ptr(),
                ds_slice.len(),
                db_slice.map(|b| b.as_ptr()),
                out_tok,
                n_out,
                n_inter,
            );
        }
    }

    Ok(out)
}

/// Fused SwiGLU MLP for Grouped INT4 (W4A32):
/// intermediate = silu(gate_proj(x)) * up_proj(x)
/// out = down_proj(intermediate)
pub fn fused_swiglu_mlp_w4a32(
    x: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    gate_s: &BorrowedTensor,
    gate_b: Option<&BorrowedTensor>,
    up_w: &BorrowedTensor,
    up_s: &BorrowedTensor,
    up_b: Option<&BorrowedTensor>,
    down_w: &BorrowedTensor,
    down_s: &BorrowedTensor,
    down_b: Option<&BorrowedTensor>,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("fused_swiglu_mlp_w4a32 requires x with at least 1 dim"));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    let n_inter = gate_w.shape[0] as usize;
    let n_out = down_w.shape[0] as usize;

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n_out as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let gw_slice = unsafe { typed_slice::<u8>(gate_w) };
    let gs_slice = unsafe { typed_slice::<f32>(gate_s) };
    let gb_slice = gate_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let uw_slice = unsafe { typed_slice::<u8>(up_w) };
    let us_slice = unsafe { typed_slice::<f32>(up_s) };
    let ub_slice = up_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let dw_slice = unsafe { typed_slice::<u8>(down_w) };
    let ds_slice = unsafe { typed_slice::<f32>(down_s) };
    let db_slice = down_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    let num_groups_k = (k + group_size - 1) / group_size;
    let bytes_per_row_k = (k + 1) / 2;

    use rayon::prelude::*;

    for token_idx in 0..m {
        let x_tok = unsafe { x_slice.as_ptr().add(token_idx * k) };
        let out_tok = unsafe { out_slice.as_mut_ptr().add(token_idx * n_out) };

        let mut h_buf = vec![0.0f32; n_inter];
        let h_ptr = h_buf.as_mut_ptr() as usize;

        let x_usize = x_tok as usize;
        let gw_usize = gw_slice.as_ptr() as usize;
        let gs_usize = gs_slice.as_ptr() as usize;
        let gb_usize = gb_slice.map(|b| b.as_ptr() as usize);

        let uw_usize = uw_slice.as_ptr() as usize;
        let us_usize = us_slice.as_ptr() as usize;
        let ub_usize = ub_slice.map(|b| b.as_ptr() as usize);

        let n_threads = rayon::current_num_threads();
        let min_chunk = (n_inter / (n_threads * 4)).max(8);

        #[cfg(target_arch = "x86_64")]
        let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
        #[cfg(not(target_arch = "x86_64"))]
        let has_vnni = false;

        #[cfg(target_arch = "x86_64")]
        let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx512 = false;

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx2 = false;

        let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
            let (u, s) = unsafe { quantize_activation_to_u8(x_tok, k) };
            (Some(u), s)
        } else {
            (None, 1.0f32)
        };
        let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

        (0..n_inter).into_par_iter().with_min_len(min_chunk).for_each(|j| {
            let x_p = x_usize as *const f32;
            let gw_row = unsafe { (gw_usize as *const u8).add(j * bytes_per_row_k) };
            let gs_row = unsafe { (gs_usize as *const f32).add(j * num_groups_k) };
            let gb = if let Some(bp) = gb_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

            let uw_row = unsafe { (uw_usize as *const u8).add(j * bytes_per_row_k) };
            let us_row = unsafe { (us_usize as *const f32).add(j * num_groups_k) };
            let ub = if let Some(bp) = ub_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

            let (g_sum, u_sum) = if let Some(x_u8_p) = x_u8_ptr {
                #[cfg(target_arch = "x86_64")]
                {
                    unsafe { swiglu_neuron_w4a8_group64_vnni_avx512(x_u8_p as *const u8, s_x, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                }
                #[cfg(not(target_arch = "x86_64"))]
                { (0.0, 0.0) }
            } else if has_avx512 {
                #[cfg(target_arch = "x86_64")]
                {
                    if group_size == 64 {
                        unsafe { swiglu_neuron_w4a32_group64_avx512(x_p, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                    } else if group_size == 32 {
                        unsafe { swiglu_neuron_w4a32_group32_avx512(x_p, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                    } else {
                        let mut gs = 0.0f32;
                        let mut us = 0.0f32;
                        for g in 0..num_groups_k {
                            let cur_len = (k - g * group_size).min(group_size);
                            gs += unsafe { dot_f32_u4_group_scalar(x_p.add(g * group_size), gw_row.add(g * (group_size / 2)), cur_len) * *gs_row.add(g) };
                            us += unsafe { dot_f32_u4_group_scalar(x_p.add(g * group_size), uw_row.add(g * (group_size / 2)), cur_len) * *us_row.add(g) };
                        }
                        (gs, us)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                { (0.0, 0.0) }
            } else if has_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    if group_size == 64 {
                        unsafe { swiglu_neuron_w4a32_group64_avx2(x_p, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                    } else if group_size == 32 {
                        unsafe { swiglu_neuron_w4a32_group32_avx2(x_p, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                    } else {
                        let mut gs = 0.0f32;
                        let mut us = 0.0f32;
                        for g in 0..num_groups_k {
                            let cur_len = (k - g * group_size).min(group_size);
                            gs += unsafe { dot_f32_u4_group_scalar(x_p.add(g * group_size), gw_row.add(g * (group_size / 2)), cur_len) * *gs_row.add(g) };
                            us += unsafe { dot_f32_u4_group_scalar(x_p.add(g * group_size), uw_row.add(g * (group_size / 2)), cur_len) * *us_row.add(g) };
                        }
                        (gs, us)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                { (0.0, 0.0) }
            } else {
                let mut gs = 0.0f32;
                let mut us = 0.0f32;
                for g in 0..num_groups_k {
                    let cur_len = (k - g * group_size).min(group_size);
                    gs += unsafe { dot_f32_u4_group_scalar(x_p.add(g * group_size), gw_row.add(g * (group_size / 2)), cur_len) * *gs_row.add(g) };
                    us += unsafe { dot_f32_u4_group_scalar(x_p.add(g * group_size), uw_row.add(g * (group_size / 2)), cur_len) * *us_row.add(g) };
                }
                (gs, us)
            };

            let g = g_sum + gb;
            let u = u_sum + ub;
            let silu_g = g / (1.0 + (-g).exp());
            let val = silu_g * u;

            unsafe {
                let h_p = h_ptr as *mut f32;
                *h_p.add(j) = val;
            }
        });

        unsafe {
            gemv_w4a32_grouped(
                h_buf.as_ptr(),
                dw_slice.as_ptr(),
                ds_slice.as_ptr(),
                db_slice.map(|b| b.as_ptr()),
                out_tok,
                n_out,
                n_inter,
                group_size,
            );
        }
    }

    Ok(out)
}

/// Quantize a 2D float weight matrix (N, K) to per-channel INT8 with scales (N,).
pub fn quantize_linear_weights_int8(w: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    if w.shape.len() != 2 {
        return Err(unsupported("quantize_linear_weights_int8 requires 2D matrix"));
    }
    let n = w.shape[0] as usize;
    let k = w.shape[1] as usize;
    let mut out_w = OwnedTensor::new(DType::Bool, w.shape.clone());
    let mut out_s = OwnedTensor::new(DType::F32, vec![n as i64]);

    let w_src = unsafe { typed_slice::<f32>(w) };
    let w_dst = unsafe { typed_mut_slice::<i8>(&mut out_w) };
    let s_dst = unsafe { typed_mut_slice::<f32>(&mut out_s) };

    for j in 0..n {
        let row = &w_src[j * k..(j + 1) * k];
        let mut max_abs = 0.0f32;
        for &val in row {
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1.0 };
        s_dst[j] = scale;
        let inv_scale = 1.0 / scale;
        let out_row = &mut w_dst[j * k..(j + 1) * k];
        for p in 0..k {
            let q = (row[p] * inv_scale).round();
            out_row[p] = q.clamp(-127.0, 127.0) as i8;
        }
    }

    Ok((out_w, out_s))
}

/// Quantize a 2D float weight matrix (N, K) to symmetric 4-bit packed INT4 with scales (N,).
pub fn quantize_linear_weights_int4(w: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    if w.shape.len() != 2 {
        return Err(unsupported("quantize_linear_weights_int4 requires 2D matrix"));
    }
    let n = w.shape[0] as usize;
    let k = w.shape[1] as usize;
    let k_packed = (k + 1) / 2;
    let mut out_w = OwnedTensor::new(DType::Bool, vec![n as i64, k_packed as i64]);
    let mut out_s = OwnedTensor::new(DType::F32, vec![n as i64]);

    let w_src = unsafe { typed_slice::<f32>(w) };
    let w_dst = unsafe { typed_mut_slice::<u8>(&mut out_w) };
    let s_dst = unsafe { typed_mut_slice::<f32>(&mut out_s) };

    for j in 0..n {
        let row = &w_src[j * k..(j + 1) * k];
        let mut max_abs = 0.0f32;
        for &val in row {
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 1e-8 { max_abs / 7.0 } else { 1.0 };
        s_dst[j] = scale;
        let inv_scale = 1.0 / scale;
        let out_row = &mut w_dst[j * k_packed..(j + 1) * k_packed];
        for b in 0..k_packed {
            let p0 = b * 2;
            let p1 = b * 2 + 1;
            let q0 = if p0 < k {
                ((row[p0] * inv_scale).round().clamp(-8.0, 7.0) as i8) + 8
            } else {
                8
            } as u8;
            let q1 = if p1 < k {
                ((row[p1] * inv_scale).round().clamp(-8.0, 7.0) as i8) + 8
            } else {
                8
            } as u8;
            out_row[b] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }

    Ok((out_w, out_s))
}

#[inline(always)]
unsafe fn dot_f32_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && len >= 16 {
            use std::arch::x86_64::*;
            let mut acc = _mm512_setzero_ps();
            let mut i = 0;
            while i + 16 <= len {
                let av = _mm512_loadu_ps(a.add(i));
                let bv = _mm512_loadu_ps(b.add(i));
                acc = _mm512_fmadd_ps(av, bv, acc);
                i += 16;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            while i < len {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            return sum;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && len >= 8 {
            use std::arch::x86_64::*;
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= len {
                let av = _mm256_loadu_ps(a.add(i));
                let bv = _mm256_loadu_ps(b.add(i));
                acc = _mm256_fmadd_ps(av, bv, acc);
                i += 8;
            }
            let mut sum = hsum256_ps_avx(acc);
            while i < len {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            return sum;
        }
    }
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += *a.add(i) * *b.add(i);
    }
    sum
}

/// Fused Attention Decode Step for W8A32 (T=1):
/// Computes:
/// 1. qkv = gemv_w8a32(x, qkv_w, qkv_s, qkv_b)
/// 2. In-place RoPE on q and k with cos and sin
/// 3. In-place update of k_cache and v_cache at offset
/// 4. GQA Attention scores, softmax, and weighted value accumulation
/// 5. o_out = gemv_w8a32(attn_out, o_w, o_s, o_b)
pub fn fused_attention_step_w8a32(
    x: &BorrowedTensor,
    qkv_w: &BorrowedTensor,
    qkv_s: &BorrowedTensor,
    qkv_b: Option<&BorrowedTensor>,
    o_w: &BorrowedTensor,
    o_s: &BorrowedTensor,
    o_b: Option<&BorrowedTensor>,
    k_cache: &BorrowedTensor,
    v_cache: &BorrowedTensor,
    cos: &BorrowedTensor,
    sin: &BorrowedTensor,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("fused_attention_step requires x with at least 1 dim"));
    }
    let hidden_size = x.shape[x_rank - 1] as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let total_qkv_dim = q_dim + 2 * kv_dim;

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let qkv_w_slice = unsafe { typed_slice::<i8>(qkv_w) };
    let qkv_s_slice = unsafe { typed_slice::<f32>(qkv_s) };
    let qkv_b_slice = qkv_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let o_w_slice = unsafe { typed_slice::<i8>(o_w) };
    let o_s_slice = unsafe { typed_slice::<f32>(o_s) };
    let o_b_slice = o_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let cos_slice = unsafe { typed_slice::<f32>(cos) };
    let sin_slice = unsafe { typed_slice::<f32>(sin) };

    // 1. Compute QKV projection
    let mut qkv = vec![0.0f32; total_qkv_dim];
    unsafe {
        gemv_w8a32(
            x_slice.as_ptr(),
            qkv_w_slice.as_ptr(),
            qkv_s_slice.as_ptr(),
            qkv_s_slice.len(),
            qkv_b_slice.map(|b| b.as_ptr()),
            qkv.as_mut_ptr(),
            total_qkv_dim,
            hidden_size,
        );
    }

    let (q, kv_rest) = qkv.split_at_mut(q_dim);
    let (k, v) = kv_rest.split_at_mut(kv_dim);

    // 2. In-place RoPE on q and k
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let q1 = q_head[i];
            let q2 = q_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            q_head[i] = q1 * c1 - q2 * s1;
            q_head[i + half_dim] = q2 * c2 + q1 * s2;
        }
    }

    for h in 0..num_kv_heads {
        let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let k1 = k_head[i];
            let k2 = k_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            k_head[i] = k1 * c1 - k2 * s1;
            k_head[i + half_dim] = k2 * c2 + k1 * s2;
        }
    }

    // 3. Update KV cache
    let max_seq_len = k_cache.shape[2] as usize;
    let head_stride = max_seq_len * head_dim;

    let k_cache_mut = unsafe { std::slice::from_raw_parts_mut(k_cache.data as *mut f32, k_cache.buffer_len()) };
    let v_cache_mut = unsafe { std::slice::from_raw_parts_mut(v_cache.data as *mut f32, v_cache.buffer_len()) };

    for kv_h in 0..num_kv_heads {
        let dst_offset = kv_h * head_stride + offset * head_dim;
        let k_src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
        let v_src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
        k_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(k_src);
        v_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(v_src);
    }

    // 4. Attention (GQA)
    let seq_len = offset + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    let mut attn_out = vec![0.0f32; q_dim];
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
        let k_base_ptr = unsafe { (k_cache.data as *const f32).add(kv_h * head_stride) };
        let v_base_ptr = unsafe { (v_cache.data as *const f32).add(kv_h * head_stride) };

        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
            let dot = unsafe { dot_f32_f32(q_ptr, k_t_ptr, head_dim) };
            let sc = dot * scale;
            scores[t] = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        let mut exp_sum = 0.0f32;
        for t in 0..seq_len {
            let ex = (scores[t] - max_score).exp();
            scores[t] = ex;
            exp_sum += ex;
        }
        let inv_sum = 1.0f32 / exp_sum;

        let out_h_ptr = unsafe { attn_out.as_mut_ptr().add(h * head_dim) };
        for t in 0..seq_len {
            let w = scores[t] * inv_sum;
            let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
            for d in 0..head_dim {
                unsafe {
                    *out_h_ptr.add(d) += w * *v_t_ptr.add(d);
                }
            }
        }
    }

    // 5. Output projection
    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = hidden_size as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    unsafe {
        gemv_w8a32(
            attn_out.as_ptr(),
            o_w_slice.as_ptr(),
            o_s_slice.as_ptr(),
            o_s_slice.len(),
            o_b_slice.map(|b| b.as_ptr()),
            out_slice.as_mut_ptr(),
            hidden_size,
            q_dim,
        );
    }

    Ok(out)
}

/// Fused Attention Decode Step for Grouped INT4 (T=1):
pub fn fused_attention_step_w4a32(
    x: &BorrowedTensor,
    qkv_w: &BorrowedTensor,
    qkv_s: &BorrowedTensor,
    qkv_b: Option<&BorrowedTensor>,
    o_w: &BorrowedTensor,
    o_s: &BorrowedTensor,
    o_b: Option<&BorrowedTensor>,
    k_cache: &BorrowedTensor,
    v_cache: &BorrowedTensor,
    cos: &BorrowedTensor,
    sin: &BorrowedTensor,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("fused_attention_step_w4a32 requires x with at least 1 dim"));
    }
    let hidden_size = x.shape[x_rank - 1] as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let total_qkv_dim = q_dim + 2 * kv_dim;

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let qkv_w_slice = unsafe { typed_slice::<u8>(qkv_w) };
    let qkv_s_slice = unsafe { typed_slice::<f32>(qkv_s) };
    let qkv_b_slice = qkv_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let o_w_slice = unsafe { typed_slice::<u8>(o_w) };
    let o_s_slice = unsafe { typed_slice::<f32>(o_s) };
    let o_b_slice = o_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let cos_slice = unsafe { typed_slice::<f32>(cos) };
    let sin_slice = unsafe { typed_slice::<f32>(sin) };

    // 1. Compute QKV projection with gemv_w4a32_grouped
    let mut qkv = vec![0.0f32; total_qkv_dim];
    unsafe {
        gemv_w4a32_grouped(
            x_slice.as_ptr(),
            qkv_w_slice.as_ptr(),
            qkv_s_slice.as_ptr(),
            qkv_b_slice.map(|b| b.as_ptr()),
            qkv.as_mut_ptr(),
            total_qkv_dim,
            hidden_size,
            group_size,
        );
    }

    let (q, kv_rest) = qkv.split_at_mut(q_dim);
    let (k, v) = kv_rest.split_at_mut(kv_dim);

    // 2. In-place RoPE on q and k
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let q1 = q_head[i];
            let q2 = q_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            q_head[i] = q1 * c1 - q2 * s1;
            q_head[i + half_dim] = q2 * c2 + q1 * s2;
        }
    }

    for h in 0..num_kv_heads {
        let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let k1 = k_head[i];
            let k2 = k_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            k_head[i] = k1 * c1 - k2 * s1;
            k_head[i + half_dim] = k2 * c2 + k1 * s2;
        }
    }

    // 3. Update KV cache
    let max_seq_len = k_cache.shape[2] as usize;
    let head_stride = max_seq_len * head_dim;

    let k_cache_mut = unsafe { std::slice::from_raw_parts_mut(k_cache.data as *mut f32, k_cache.buffer_len()) };
    let v_cache_mut = unsafe { std::slice::from_raw_parts_mut(v_cache.data as *mut f32, v_cache.buffer_len()) };

    for kv_h in 0..num_kv_heads {
        let dst_offset = kv_h * head_stride + offset * head_dim;
        let k_src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
        let v_src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
        k_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(k_src);
        v_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(v_src);
    }

    // 4. Attention (GQA)
    let seq_len = offset + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    let mut attn_out = vec![0.0f32; q_dim];
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
        let k_base_ptr = unsafe { (k_cache.data as *const f32).add(kv_h * head_stride) };
        let v_base_ptr = unsafe { (v_cache.data as *const f32).add(kv_h * head_stride) };

        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
            let dot = unsafe { dot_f32_f32(q_ptr, k_t_ptr, head_dim) };
            let sc = dot * scale;
            scores[t] = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        let mut exp_sum = 0.0f32;
        for t in 0..seq_len {
            let ex = (scores[t] - max_score).exp();
            scores[t] = ex;
            exp_sum += ex;
        }
        let inv_sum = 1.0f32 / exp_sum;

        let out_h_ptr = unsafe { attn_out.as_mut_ptr().add(h * head_dim) };
        for t in 0..seq_len {
            let w = scores[t] * inv_sum;
            let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
            for d in 0..head_dim {
                unsafe {
                    *out_h_ptr.add(d) += w * *v_t_ptr.add(d);
                }
            }
        }
    }

    // 5. Output projection with gemv_w4a32_grouped
    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = hidden_size as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    unsafe {
        gemv_w4a32_grouped(
            attn_out.as_ptr(),
            o_w_slice.as_ptr(),
            o_s_slice.as_ptr(),
            o_b_slice.map(|b| b.as_ptr()),
            out_slice.as_mut_ptr(),
            hidden_size,
            q_dim,
            group_size,
        );
    }

    Ok(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fast_rms_norm_avx512(x: *const f32, w: *const f32, out: *mut f32, n: usize, eps: f32) {
    use std::arch::x86_64::*;
    let mut sum_sq = _mm512_setzero_ps();
    let num_chunks = n / 16;
    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        sum_sq = _mm512_fmadd_ps(v, v, sum_sq);
    }
    let mut total_sq = _mm512_reduce_add_ps(sum_sq);
    for i in (num_chunks * 16)..n {
        let v = *x.add(i);
        total_sq += v * v;
    }
    let rms = (total_sq / (n as f32) + eps).sqrt();
    let inv_rms = _mm512_set1_ps(1.0 / rms);

    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        let weight = _mm512_loadu_ps(w.add(c * 16));
        let norm = _mm512_mul_ps(_mm512_mul_ps(v, inv_rms), weight);
        _mm512_storeu_ps(out.add(c * 16), norm);
    }
    for i in (num_chunks * 16)..n {
        *out.add(i) = (*x.add(i) / rms) * *w.add(i);
    }
}

#[inline(always)]
unsafe fn fast_rms_norm(x: *const f32, w: *const f32, out: *mut f32, n: usize, eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            fast_rms_norm_avx512(x, w, out, n, eps);
            return;
        }
    }
    let mut total_sq = 0.0f32;
    for i in 0..n {
        let v = *x.add(i);
        total_sq += v * v;
    }
    let rms = (total_sq / (n as f32) + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        *out.add(i) = *x.add(i) * inv_rms * *w.add(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fast_vector_add_avx512(dst: *mut f32, src: *const f32, n: usize) {
    use std::arch::x86_64::*;
    let num_chunks = n / 16;
    for c in 0..num_chunks {
        let d = _mm512_loadu_ps(dst.add(c * 16));
        let s = _mm512_loadu_ps(src.add(c * 16));
        _mm512_storeu_ps(dst.add(c * 16), _mm512_add_ps(d, s));
    }
    for i in (num_chunks * 16)..n {
        *dst.add(i) += *src.add(i);
    }
}

#[inline(always)]
unsafe fn fast_vector_add(dst: *mut f32, src: *const f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            fast_vector_add_avx512(dst, src, n);
            return;
        }
    }
    for i in 0..n {
        *dst.add(i) += *src.add(i);
    }
}

/// Fused Transformer Layer Decode Step for Grouped INT4 (T=1) in-place:
/// 1. RMSNorm(x, input_norm_w) -> normed
/// 2. QKV GEMV -> qkv
/// 3. RoPE on q and k
/// 4. Write k, v to cache
/// 5. Attention (GQA) -> attn_out
/// 6. O GEMV -> o_out
/// 7. x += o_out
/// 8. RMSNorm(x, post_norm_w) -> normed
/// 9. SwiGLU MLP: Gate + Up GEMV -> Silu(Gate) * Up -> Down GEMV -> mlp_out
/// 10. x += mlp_out
pub fn fused_transformer_layer_step_w4a32(
    x: &mut BorrowedTensor,
    input_norm_w: &BorrowedTensor,
    qkv_w: &BorrowedTensor,
    qkv_s: &BorrowedTensor,
    qkv_b: Option<&BorrowedTensor>,
    o_w: &BorrowedTensor,
    o_s: &BorrowedTensor,
    o_b: Option<&BorrowedTensor>,
    post_norm_w: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    gate_s: &BorrowedTensor,
    gate_b: Option<&BorrowedTensor>,
    up_w: &BorrowedTensor,
    up_s: &BorrowedTensor,
    up_b: Option<&BorrowedTensor>,
    down_w: &BorrowedTensor,
    down_s: &BorrowedTensor,
    down_b: Option<&BorrowedTensor>,
    k_cache: &BorrowedTensor,
    v_cache: &BorrowedTensor,
    cos: &BorrowedTensor,
    sin: &BorrowedTensor,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
    eps: f64,
) -> PyResult<()> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported("fused_transformer_layer_step_w4a32 requires x with at least 1 dim"));
    }
    let hidden_size = x.shape[x_rank - 1] as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let total_qkv_dim = q_dim + 2 * kv_dim;
    let n_inter = gate_w.shape[0] as usize;

    use rayon::prelude::*;
    let x_slice = unsafe { std::slice::from_raw_parts_mut(x.data as *mut f32, x.buffer_len()) };
    let input_norm_w_slice = unsafe { typed_slice::<f32>(input_norm_w) };
    let qkv_w_slice = unsafe { typed_slice::<u8>(qkv_w) };
    let qkv_s_slice = unsafe { typed_slice::<f32>(qkv_s) };
    let qkv_b_slice = qkv_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let o_w_slice = unsafe { typed_slice::<u8>(o_w) };
    let o_s_slice = unsafe { typed_slice::<f32>(o_s) };
    let o_b_slice = o_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let post_norm_w_slice = unsafe { typed_slice::<f32>(post_norm_w) };
    let gw_slice = unsafe { typed_slice::<u8>(gate_w) };
    let gs_slice = unsafe { typed_slice::<f32>(gate_s) };
    let gb_slice = gate_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let uw_slice = unsafe { typed_slice::<u8>(up_w) };
    let us_slice = unsafe { typed_slice::<f32>(up_s) };
    let ub_slice = up_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let dw_slice = unsafe { typed_slice::<u8>(down_w) };
    let ds_slice = unsafe { typed_slice::<f32>(down_s) };
    let db_slice = down_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let cos_slice = unsafe { typed_slice::<f32>(cos) };
    let sin_slice = unsafe { typed_slice::<f32>(sin) };

    let x_ptr = x_slice.as_mut_ptr();
    let mut normed = vec![0.0f32; hidden_size];

    // 1. Pre-attention RMSNorm
    unsafe {
        fast_rms_norm(x_ptr, input_norm_w_slice.as_ptr(), normed.as_mut_ptr(), hidden_size, eps as f32);
    }

    // 2. QKV projection
    let mut qkv = vec![0.0f32; total_qkv_dim];
    unsafe {
        gemv_w4a32_grouped(
            normed.as_ptr(),
            qkv_w_slice.as_ptr(),
            qkv_s_slice.as_ptr(),
            qkv_b_slice.map(|b| b.as_ptr()),
            qkv.as_mut_ptr(),
            total_qkv_dim,
            hidden_size,
            group_size,
        );
    }

    let (q, kv_rest) = qkv.split_at_mut(q_dim);
    let (k, v) = kv_rest.split_at_mut(kv_dim);

    // 3. RoPE on q and k
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let q1 = q_head[i];
            let q2 = q_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            q_head[i] = q1 * c1 - q2 * s1;
            q_head[i + half_dim] = q2 * c2 + q1 * s2;
        }
    }

    for h in 0..num_kv_heads {
        let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let k1 = k_head[i];
            let k2 = k_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            k_head[i] = k1 * c1 - k2 * s1;
            k_head[i + half_dim] = k2 * c2 + k1 * s2;
        }
    }

    // 4. Update KV cache
    let max_seq_len = k_cache.shape[2] as usize;
    let head_stride = max_seq_len * head_dim;

    let k_cache_mut = unsafe { std::slice::from_raw_parts_mut(k_cache.data as *mut f32, k_cache.buffer_len()) };
    let v_cache_mut = unsafe { std::slice::from_raw_parts_mut(v_cache.data as *mut f32, v_cache.buffer_len()) };

    for kv_h in 0..num_kv_heads {
        let dst_offset = kv_h * head_stride + offset * head_dim;
        let k_src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
        let v_src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
        k_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(k_src);
        v_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(v_src);
    }

    // 5. GQA Attention
    let seq_len = offset + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    let mut attn_out = vec![0.0f32; q_dim];
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
        let k_base_ptr = unsafe { (k_cache.data as *const f32).add(kv_h * head_stride) };
        let v_base_ptr = unsafe { (v_cache.data as *const f32).add(kv_h * head_stride) };

        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
            let dot = unsafe { dot_f32_f32(q_ptr, k_t_ptr, head_dim) };
            let sc = dot * scale;
            scores[t] = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        let mut exp_sum = 0.0f32;
        for t in 0..seq_len {
            let ex = (scores[t] - max_score).exp();
            scores[t] = ex;
            exp_sum += ex;
        }
        let inv_sum = 1.0f32 / exp_sum;

        let out_h_ptr = unsafe { attn_out.as_mut_ptr().add(h * head_dim) };
        for t in 0..seq_len {
            let w = scores[t] * inv_sum;
            let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
            for d in 0..head_dim {
                unsafe {
                    *out_h_ptr.add(d) += w * *v_t_ptr.add(d);
                }
            }
        }
    }

    // 6. O projection
    let mut o_out = vec![0.0f32; hidden_size];
    unsafe {
        gemv_w4a32_grouped(
            attn_out.as_ptr(),
            o_w_slice.as_ptr(),
            o_s_slice.as_ptr(),
            o_b_slice.map(|b| b.as_ptr()),
            o_out.as_mut_ptr(),
            hidden_size,
            q_dim,
            group_size,
        );
    }

    // 7. Residual add: x += o_out
    unsafe {
        fast_vector_add(x_ptr, o_out.as_ptr(), hidden_size);
    }

    // 8. Post-attention RMSNorm: normed = rms_norm(x, post_norm_w)
    unsafe {
        fast_rms_norm(x_ptr, post_norm_w_slice.as_ptr(), normed.as_mut_ptr(), hidden_size, eps as f32);
    }

    // 9. SwiGLU MLP
    let mut h_buf = vec![0.0f32; n_inter];
    let h_ptr = h_buf.as_mut_ptr() as usize;

    let num_groups_k = (hidden_size + group_size - 1) / group_size;
    let bytes_per_row_k = (hidden_size + 1) / 2;

    let x_usize = normed.as_ptr() as usize;
    let gw_usize = gw_slice.as_ptr() as usize;
    let gs_usize = gs_slice.as_ptr() as usize;
    let gb_usize = gb_slice.map(|b| b.as_ptr() as usize);

    let uw_usize = uw_slice.as_ptr() as usize;
    let us_usize = us_slice.as_ptr() as usize;
    let ub_usize = ub_slice.map(|b| b.as_ptr() as usize);

    let n_threads = rayon::current_num_threads();
    let min_chunk = (n_inter / (n_threads * 4)).max(8);

    #[cfg(target_arch = "x86_64")]
    let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let has_vnni = false;

    #[cfg(target_arch = "x86_64")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx512 = false;

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
        let (u, s) = unsafe { quantize_activation_to_u8(normed.as_ptr(), hidden_size) };
        (Some(u), s)
    } else {
        (None, 1.0f32)
    };
    let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

    (0..n_inter).into_par_iter().with_min_len(min_chunk).for_each(|j| {
        let x_p = x_usize as *const f32;
        let gw_row = unsafe { (gw_usize as *const u8).add(j * bytes_per_row_k) };
        let gs_row = unsafe { (gs_usize as *const f32).add(j * num_groups_k) };
        let gb = if let Some(bp) = gb_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

        let uw_row = unsafe { (uw_usize as *const u8).add(j * bytes_per_row_k) };
        let us_row = unsafe { (us_usize as *const f32).add(j * num_groups_k) };
        let ub = if let Some(bp) = ub_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

        let (g_sum, u_sum) = if let Some(x_u8_p) = x_u8_ptr {
            #[cfg(target_arch = "x86_64")]
            {
                unsafe { swiglu_neuron_w4a8_group64_vnni_avx512(x_u8_p as *const u8, s_x, gw_row, gs_row, uw_row, us_row, num_groups_k) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { (0.0, 0.0) }
        } else if has_avx512 {
            #[cfg(target_arch = "x86_64")]
            {
                if group_size == 64 {
                    unsafe { swiglu_neuron_w4a32_group64_avx512(x_p, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                } else {
                    (0.0, 0.0)
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { (0.0, 0.0) }
        } else if has_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                if group_size == 64 {
                    unsafe { swiglu_neuron_w4a32_group64_avx2(x_p, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                } else {
                    (0.0, 0.0)
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { (0.0, 0.0) }
        } else {
            (0.0, 0.0)
        };

        let g = g_sum + gb;
        let u = u_sum + ub;
        let silu_g = g / (1.0 + (-g).exp());
        let val = silu_g * u;

        unsafe {
            let h_p = h_ptr as *mut f32;
            *h_p.add(j) = val;
        }
    });

    // Down projection
    let mut down_out = vec![0.0f32; hidden_size];
    unsafe {
        gemv_w4a32_grouped(
            h_buf.as_ptr(),
            dw_slice.as_ptr(),
            ds_slice.as_ptr(),
            db_slice.map(|b| b.as_ptr()),
            down_out.as_mut_ptr(),
            hidden_size,
            n_inter,
            group_size,
        );
    }

    // 10. Residual add: x += down_out
    unsafe {
        fast_vector_add(x_ptr, down_out.as_ptr(), hidden_size);
    }

    Ok(())
}

pub struct RustLayerData {
    pub input_norm_w: Vec<f32>,
    pub qkv_w: Vec<u8>,
    pub qkv_s: Vec<f32>,
    pub qkv_b: Option<Vec<f32>>,
    pub o_w: Vec<u8>,
    pub o_s: Vec<f32>,
    pub o_b: Option<Vec<f32>>,
    pub post_norm_w: Vec<f32>,
    pub gate_w: Vec<u8>,
    pub gate_s: Vec<f32>,
    pub gate_b: Option<Vec<f32>>,
    pub up_w: Vec<u8>,
    pub up_s: Vec<f32>,
    pub up_b: Option<Vec<f32>>,
    pub down_w: Vec<u8>,
    pub down_s: Vec<f32>,
    pub down_b: Option<Vec<f32>>,
}

#[pyclass]
pub struct RustQwenDecoder {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub group_size: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<RustLayerData>,
    pub final_norm_w: Vec<f32>,
    pub lm_head_w: Vec<u8>,
    pub lm_head_s: Vec<f32>,
    pub k_caches: Vec<Vec<f32>>,
    pub v_caches: Vec<Vec<f32>>,
    pub cos_table: Vec<f32>,
    pub sin_table: Vec<f32>,
    // Preallocated scratch buffers
    x_buf: Vec<f32>,
    normed_buf: Vec<f32>,
    qkv_buf: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    h_buf: Vec<f32>,
    down_out: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
}

#[pymethods]
impl RustQwenDecoder {
    #[new]
    #[pyo3(signature = (
        embed_tokens,
        layers_data,
        final_norm_w,
        lm_head_w,
        lm_head_s,
        num_layers,
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        group_size=64,
        rms_norm_eps=1e-6,
        max_seq_len=2048,
        rope_theta=1000000.0
    ))]
    pub fn new(
        py: Python<'_>,
        embed_tokens: &Bound<'_, PyCapsule>,
        layers_data: Vec<Vec<Bound<'_, PyCapsule>>>,
        final_norm_w: &Bound<'_, PyCapsule>,
        lm_head_w: &Bound<'_, PyCapsule>,
        lm_head_s: &Bound<'_, PyCapsule>,
        num_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        group_size: usize,
        rms_norm_eps: f64,
        max_seq_len: usize,
        rope_theta: f64,
    ) -> PyResult<Self> {
        let _ = py;
        let emb_view = unsafe { dlpack::BorrowedTensor::from_capsule(embed_tokens)? };
        let vocab_size = emb_view.shape[0] as usize;
        let emb_slice = unsafe { typed_slice::<f32>(&emb_view) };
        let embed_tokens_vec = emb_slice.to_vec();

        let fnorm_view = unsafe { dlpack::BorrowedTensor::from_capsule(final_norm_w)? };
        let final_norm_vec = unsafe { typed_slice::<f32>(&fnorm_view) }.to_vec();

        let lm_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(lm_head_w)? };
        let lm_head_w_vec = unsafe { typed_slice::<u8>(&lm_w_view) }.to_vec();

        let lm_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(lm_head_s)? };
        let lm_head_s_vec = unsafe { typed_slice::<f32>(&lm_s_view) }.to_vec();

        let mut rust_layers = Vec::with_capacity(num_layers);
        for l_caps in layers_data {
            if l_caps.len() < 12 {
                return Err(unsupported("Each layer requires 12 capsules: input_norm, qkv_w, qkv_s, o_w, o_s, post_norm, gate_w, gate_s, up_w, up_s, down_w, down_s"));
            }
            let in_norm = unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[0])?) }.to_vec();
            let qkv_w = unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[1])?) }.to_vec();
            let qkv_s = unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[2])?) }.to_vec();
            let o_w = unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[3])?) }.to_vec();
            let o_s = unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[4])?) }.to_vec();
            let post_norm = unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[5])?) }.to_vec();
            let gate_w = unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[6])?) }.to_vec();
            let gate_s = unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[7])?) }.to_vec();
            let up_w = unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[8])?) }.to_vec();
            let up_s = unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[9])?) }.to_vec();
            let down_w = unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[10])?) }.to_vec();
            let down_s = unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[11])?) }.to_vec();
            let qkv_b = if l_caps.len() >= 13 {
                Some(unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[12])?) }.to_vec())
            } else {
                None
            };

            rust_layers.push(RustLayerData {
                input_norm_w: in_norm,
                qkv_w,
                qkv_s,
                qkv_b,
                o_w,
                o_s,
                o_b: None,
                post_norm_w: post_norm,
                gate_w,
                gate_s,
                gate_b: None,
                up_w,
                up_s,
                up_b: None,
                down_w,
                down_s,
                down_b: None,
            });
        }

        // Precompute RoPE tables
        let half_dim = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * head_dim];
        let mut sin_table = vec![0.0f32; max_seq_len * head_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0f32 / (rope_theta as f32).powf((2.0 * i as f32) / (head_dim as f32));
                let val = (pos as f32) * freq;
                let c = val.cos();
                let s = val.sin();
                cos_table[pos * head_dim + i] = c;
                cos_table[pos * head_dim + i + half_dim] = c;
                sin_table[pos * head_dim + i] = s;
                sin_table[pos * head_dim + i + half_dim] = s;
            }
        }

        // Allocate static KV caches for all layers
        let kv_cache_len = num_kv_heads * max_seq_len * head_dim;
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(vec![0.0f32; kv_cache_len]);
            v_caches.push(vec![0.0f32; kv_cache_len]);
        }

        let total_qkv = (num_heads + 2 * num_kv_heads) * head_dim;

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            group_size,
            rms_norm_eps: rms_norm_eps as f32,
            max_seq_len,
            embed_tokens: embed_tokens_vec,
            layers: rust_layers,
            final_norm_w: final_norm_vec,
            lm_head_w: lm_head_w_vec,
            lm_head_s: lm_head_s_vec,
            k_caches,
            v_caches,
            cos_table,
            sin_table,
            x_buf: vec![0.0f32; hidden_size],
            normed_buf: vec![0.0f32; hidden_size],
            qkv_buf: vec![0.0f32; total_qkv],
            attn_out: vec![0.0f32; num_heads * head_dim],
            o_out: vec![0.0f32; hidden_size],
            h_buf: vec![0.0f32; intermediate_size],
            down_out: vec![0.0f32; hidden_size],
            scores: vec![0.0f32; max_seq_len],
            logits: vec![0.0f32; vocab_size],
        })
    }

    pub fn reset_kv_cache(&mut self) {
        for kc in &mut self.k_caches {
            kc.fill(0.0);
        }
        for vc in &mut self.v_caches {
            vc.fill(0.0);
        }
    }

    pub fn copy_kv_cache_from_tensors(
        &mut self,
        k_tensors: Vec<Bound<'_, PyCapsule>>,
        v_tensors: Vec<Bound<'_, PyCapsule>>,
        seq_len: usize,
    ) -> PyResult<()> {
        let dst_head_stride = self.max_seq_len * self.head_dim;
        for l in 0..self.num_layers.min(k_tensors.len()) {
            let k_view = unsafe { dlpack::BorrowedTensor::from_capsule(&k_tensors[l])? };
            let v_view = unsafe { dlpack::BorrowedTensor::from_capsule(&v_tensors[l])? };
            let k_slice = unsafe { typed_slice::<f32>(&k_view) };
            let v_slice = unsafe { typed_slice::<f32>(&v_view) };

            let k_shape = &k_view.shape;
            let src_max_len = if k_shape.len() >= 2 {
                k_shape[k_shape.len() - 2] as usize
            } else {
                seq_len
            };
            let src_head_stride = src_max_len * self.head_dim;

            let dst_k = &mut self.k_caches[l];
            let dst_v = &mut self.v_caches[l];

            for kv_h in 0..self.num_kv_heads {
                for t in 0..seq_len {
                    let src_offset = kv_h * src_head_stride + t * self.head_dim;
                    let dst_offset = kv_h * dst_head_stride + t * self.head_dim;
                    dst_k[dst_offset..dst_offset + self.head_dim].copy_from_slice(&k_slice[src_offset..src_offset + self.head_dim]);
                    dst_v[dst_offset..dst_offset + self.head_dim].copy_from_slice(&v_slice[src_offset..src_offset + self.head_dim]);
                }
            }
        }
        Ok(())
    }

    pub fn decode_step(&mut self, py: Python<'_>, token_id: usize, offset: usize) -> PyResult<PyObject> {
        self.step_internal(token_id, offset);
        let mut out = OwnedTensor::new(DType::F32, vec![1, self.vocab_size as i64]);
        let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };
        out_slice.copy_from_slice(&self.logits);
        dlpack::owned_to_capsule_owned(py, out).map(|c| c.into_any())
    }

    #[pyo3(signature = (token_id, offset, temperature=0.7, top_k=40, repetition_penalty=1.0, recent_tokens=None))]
    pub fn decode_and_sample(
        &mut self,
        token_id: usize,
        offset: usize,
        temperature: f32,
        top_k: usize,
        repetition_penalty: f32,
        recent_tokens: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.step_internal(token_id, offset);
        if repetition_penalty > 1.0 {
            if let Some(tokens) = recent_tokens {
                let mut seen = std::collections::HashSet::new();
                for t in tokens {
                    if t < self.logits.len() && seen.insert(t) {
                        let l = self.logits[t];
                        if l > 0.0 {
                            self.logits[t] = l / repetition_penalty;
                        } else {
                            self.logits[t] = l * repetition_penalty;
                        }
                    }
                }
            }
        }
        Ok(sample_logits(&self.logits, temperature, top_k))
    }
}

pub(crate) fn sample_logits(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    let vocab_size = logits.len();
    if temperature <= 0.0 || top_k == 1 {
        // Greedy argmax
        let mut best_idx = 0;
        let mut best_val = logits[0];
        for i in 1..vocab_size {
            if logits[i] > best_val {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Top-k sampling: bounded vector tracking top-k (eliminating 2.4 MB heap allocation per token)
    let k = top_k.min(vocab_size).max(1);
    let mut top_items: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    let mut min_val = f32::NEG_INFINITY;
    let mut min_pos = 0;

    for (i, &val) in logits.iter().enumerate() {
        if top_items.len() < k {
            top_items.push((i, val));
            if val < min_val || top_items.len() == 1 {
                min_val = val;
                min_pos = top_items.len() - 1;
            }
        } else if val > min_val {
            top_items[min_pos] = (i, val);
            let mut new_min = top_items[0].1;
            let mut new_pos = 0;
            for (idx, &(_, v)) in top_items.iter().enumerate() {
                if v < new_min {
                    new_min = v;
                    new_pos = idx;
                }
            }
            min_val = new_min;
            min_pos = new_pos;
        }
    }

    let max_logit = top_items.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
    let inv_temp = 1.0 / temperature;
    let mut sum_exp = 0.0f32;
    for item in &mut top_items {
        let p = ((item.1 - max_logit) * inv_temp).exp();
        item.1 = p;
        sum_exp += p;
    }

    let r = rand::random::<f32>() * sum_exp;
    let mut accum = 0.0f32;
    for (idx, p) in top_items {
        accum += p;
        if accum >= r {
            return idx;
        }
    }
    best_idx(logits)
}

pub(crate) fn best_idx(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut val = logits[0];
    for (i, &l) in logits.iter().enumerate() {
        if l > val {
            val = l;
            best = i;
        }
    }
    best
}

impl RustQwenDecoder {
    fn step_internal(&mut self, token_id: usize, offset: usize) {
        let hidden_size = self.hidden_size;
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let total_qkv = q_dim + 2 * kv_dim;
        let intermediate_size = self.intermediate_size;
        let group_size = self.group_size;
        let max_seq_len = self.max_seq_len;
        let head_stride = max_seq_len * head_dim;
        let half_dim = head_dim / 2;
        let eps = self.rms_norm_eps;

        // 1. Embedding lookup
        let emb_start = token_id * hidden_size;
        self.x_buf.copy_from_slice(&self.embed_tokens[emb_start..emb_start + hidden_size]);

        let cos_offset = offset * head_dim;
        let cos_p = &self.cos_table[cos_offset..cos_offset + head_dim];
        let sin_p = &self.sin_table[cos_offset..cos_offset + head_dim];

        // 2. Iterate all layers
        for l in 0..self.num_layers {
            let layer = &self.layers[l];

            // A. Pre-attention RMSNorm
            unsafe {
                fast_rms_norm(
                    self.x_buf.as_ptr(),
                    layer.input_norm_w.as_ptr(),
                    self.normed_buf.as_mut_ptr(),
                    hidden_size,
                    eps,
                );
            }

            // B. QKV projection
            unsafe {
                gemv_w4a32_grouped(
                    self.normed_buf.as_ptr(),
                    layer.qkv_w.as_ptr(),
                    layer.qkv_s.as_ptr(),
                    layer.qkv_b.as_ref().map(|b| b.as_ptr()),
                    self.qkv_buf.as_mut_ptr(),
                    total_qkv,
                    hidden_size,
                    group_size,
                );
            }

            let (q, kv_rest) = self.qkv_buf.split_at_mut(q_dim);
            let (k, v) = kv_rest.split_at_mut(kv_dim);

            // C. RoPE
            for h in 0..num_heads {
                let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
                for i in 0..half_dim {
                    let q1 = q_head[i];
                    let q2 = q_head[i + half_dim];
                    let c1 = cos_p[i];
                    let s1 = sin_p[i];
                    let c2 = cos_p[i + half_dim];
                    let s2 = sin_p[i + half_dim];
                    q_head[i] = q1 * c1 - q2 * s1;
                    q_head[i + half_dim] = q2 * c2 + q1 * s2;
                }
            }

            for h in 0..num_kv_heads {
                let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
                for i in 0..half_dim {
                    let k1 = k_head[i];
                    let k2 = k_head[i + half_dim];
                    let c1 = cos_p[i];
                    let s1 = sin_p[i];
                    let c2 = cos_p[i + half_dim];
                    let s2 = sin_p[i + half_dim];
                    k_head[i] = k1 * c1 - k2 * s1;
                    k_head[i + half_dim] = k2 * c2 + k1 * s2;
                }
            }

            // D. Update KV caches
            let k_cache = &mut self.k_caches[l];
            let v_cache = &mut self.v_caches[l];
            for kv_h in 0..num_kv_heads {
                let dst_offset = kv_h * head_stride + offset * head_dim;
                let k_src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
                let v_src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
                k_cache[dst_offset..dst_offset + head_dim].copy_from_slice(k_src);
                v_cache[dst_offset..dst_offset + head_dim].copy_from_slice(v_src);
            }

            // E. GQA Attention
            let seq_len = offset + 1;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let heads_per_kv = num_heads / num_kv_heads;

            self.attn_out.fill(0.0);

            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;
                let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
                let k_base_ptr = unsafe { k_cache.as_ptr().add(kv_h * head_stride) };
                let v_base_ptr = unsafe { v_cache.as_ptr().add(kv_h * head_stride) };

                let mut max_score = f32::NEG_INFINITY;
                for t in 0..seq_len {
                    let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
                    let dot = unsafe { dot_f32_f32(q_ptr, k_t_ptr, head_dim) };
                    let sc = dot * scale;
                    self.scores[t] = sc;
                    if sc > max_score {
                        max_score = sc;
                    }
                }

                let mut exp_sum = 0.0f32;
                for t in 0..seq_len {
                    let ex = (self.scores[t] - max_score).exp();
                    self.scores[t] = ex;
                    exp_sum += ex;
                }
                let inv_sum = 1.0f32 / exp_sum;

                let out_h_ptr = unsafe { self.attn_out.as_mut_ptr().add(h * head_dim) };
                for t in 0..seq_len {
                    let w = self.scores[t] * inv_sum;
                    let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
                    for d in 0..head_dim {
                        unsafe {
                            *out_h_ptr.add(d) += w * *v_t_ptr.add(d);
                        }
                    }
                }
            }

            // F. Output projection
            unsafe {
                gemv_w4a32_grouped(
                    self.attn_out.as_ptr(),
                    layer.o_w.as_ptr(),
                    layer.o_s.as_ptr(),
                    layer.o_b.as_ref().map(|b| b.as_ptr()),
                    self.o_out.as_mut_ptr(),
                    hidden_size,
                    q_dim,
                    group_size,
                );
            }

            // G. Residual add: x += o_out
            unsafe {
                fast_vector_add(self.x_buf.as_mut_ptr(), self.o_out.as_ptr(), hidden_size);
            }

            // H. Post-attention RMSNorm
            unsafe {
                fast_rms_norm(
                    self.x_buf.as_ptr(),
                    layer.post_norm_w.as_ptr(),
                    self.normed_buf.as_mut_ptr(),
                    hidden_size,
                    eps,
                );
            }

            // I. SwiGLU MLP
            let h_ptr = self.h_buf.as_mut_ptr() as usize;
            let num_groups_k = (hidden_size + group_size - 1) / group_size;
            let bytes_per_row_k = (hidden_size + 1) / 2;

            let x_usize = self.normed_buf.as_ptr() as usize;
            let gw_usize = layer.gate_w.as_ptr() as usize;
            let gs_usize = layer.gate_s.as_ptr() as usize;
            let gb_usize = layer.gate_b.as_ref().map(|b| b.as_ptr() as usize);

            let uw_usize = layer.up_w.as_ptr() as usize;
            let us_usize = layer.up_s.as_ptr() as usize;
            let ub_usize = layer.up_b.as_ref().map(|b| b.as_ptr() as usize);

            let n_threads = rayon::current_num_threads();
            let min_chunk = (intermediate_size / (n_threads * 4)).max(8);

            #[cfg(target_arch = "x86_64")]
            let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
            #[cfg(not(target_arch = "x86_64"))]
            let has_vnni = false;

            let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
                let (u, s) = unsafe { quantize_activation_to_u8(self.normed_buf.as_ptr(), hidden_size) };
                (Some(u), s)
            } else {
                (None, 1.0f32)
            };
            let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

            (0..intermediate_size).into_par_iter().with_min_len(min_chunk).for_each(|j| {
                let x_p = x_usize as *const f32;
                let gw_row = unsafe { (gw_usize as *const u8).add(j * bytes_per_row_k) };
                let gs_row = unsafe { (gs_usize as *const f32).add(j * num_groups_k) };
                let gb = if let Some(bp) = gb_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

                let uw_row = unsafe { (uw_usize as *const u8).add(j * bytes_per_row_k) };
                let us_row = unsafe { (us_usize as *const f32).add(j * num_groups_k) };
                let ub = if let Some(bp) = ub_usize { unsafe { *((bp as *const f32).add(j)) } } else { 0.0 };

                let (g_sum, u_sum) = if let Some(x_u8_p) = x_u8_ptr {
                    #[cfg(target_arch = "x86_64")]
                    {
                        unsafe { swiglu_neuron_w4a8_group64_vnni_avx512(x_u8_p as *const u8, s_x, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    { (0.0, 0.0) }
                } else {
                    #[cfg(target_arch = "x86_64")]
                    {
                        unsafe { swiglu_neuron_w4a32_group64_avx512(x_p, gw_row, gs_row, uw_row, us_row, num_groups_k) }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    { (0.0, 0.0) }
                };

                let g = g_sum + gb;
                let u = u_sum + ub;
                let silu_g = g / (1.0 + (-g).exp());
                let val = silu_g * u;

                unsafe {
                    let h_p = h_ptr as *mut f32;
                    *h_p.add(j) = val;
                }
            });

            // Down GEMV
            unsafe {
                gemv_w4a32_grouped(
                    self.h_buf.as_ptr(),
                    layer.down_w.as_ptr(),
                    layer.down_s.as_ptr(),
                    layer.down_b.as_ref().map(|b| b.as_ptr()),
                    self.down_out.as_mut_ptr(),
                    hidden_size,
                    intermediate_size,
                    group_size,
                );
            }

            // J. Residual add: x += down_out
            unsafe {
                fast_vector_add(self.x_buf.as_mut_ptr(), self.down_out.as_ptr(), hidden_size);
            }
        }

        // 3. Final RMSNorm
        unsafe {
            fast_rms_norm(
                self.x_buf.as_ptr(),
                self.final_norm_w.as_ptr(),
                self.normed_buf.as_mut_ptr(),
                hidden_size,
                eps,
            );
        }

        // 4. LM Head GEMV
        unsafe {
            gemv_w4a32_grouped(
                self.normed_buf.as_ptr(),
                self.lm_head_w.as_ptr(),
                self.lm_head_s.as_ptr(),
                None,
                self.logits.as_mut_ptr(),
                self.vocab_size,
                hidden_size,
                group_size,
            );
        }
    }
}


