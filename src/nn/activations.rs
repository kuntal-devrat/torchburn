//! Activation functions — elementwise or reduction+elementwise.
//!
//! All activations support f32/f64, arbitrary strides, and are parallelized
//! via rayon for large tensors.

use crate::dlpack::{
    contiguous_strides, elem_count, unsupported, BorrowedTensor, DType, OwnedTensor,
};
use pyo3::prelude::*;
use wide::{f32x8, CmpGt};

const PAR_CHUNK: usize = 16 * 1024;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ---------------------------------------------------------------------------
// Generic elementwise activation helper with full closure inlining
// ---------------------------------------------------------------------------

#[inline(always)]
fn apply_elementwise_f32<F: Fn(f32) -> f32 + Sync + Send>(
    a: &BorrowedTensor,
    out: &mut OwnedTensor,
    f: F,
) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        if n >= PAR_CHUNK {
            use rayon::prelude::*;
            out_data
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * PAR_CHUNK;
                    let a_slice = &a_data[base..base + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = f(a_slice[i]);
                    }
                });
        } else {
            for i in 0..n {
                out_data[i] = f(a_data[i]);
            }
        }
    } else {
        let rank = a.shape.len();
        let mut coords = vec![0usize; rank];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                coords[d] = rem % (a.shape[d].max(1) as usize);
                rem /= a.shape[d].max(1) as usize;
            }
            let mut ai = 0usize;
            for d in 0..rank {
                if a.shape[d] > 1 {
                    ai += coords[d] * a.strides[d] as usize;
                }
            }
            out_data[i] = f(a_data[ai]);
        }
    }
}

#[inline(always)]
fn apply_elementwise_f64<F: Fn(f64) -> f64 + Sync + Send>(
    a: &BorrowedTensor,
    out: &mut OwnedTensor,
    f: F,
) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        if n >= PAR_CHUNK {
            use rayon::prelude::*;
            out_data
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * PAR_CHUNK;
                    let a_slice = &a_data[base..base + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = f(a_slice[i]);
                    }
                });
        } else {
            for i in 0..n {
                out_data[i] = f(a_data[i]);
            }
        }
    } else {
        let rank = a.shape.len();
        let mut coords = vec![0usize; rank];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                coords[d] = rem % (a.shape[d].max(1) as usize);
                rem /= a.shape[d].max(1) as usize;
            }
            let mut ai = 0usize;
            for d in 0..rank {
                if a.shape[d] > 1 {
                    ai += coords[d] * a.strides[d] as usize;
                }
            }
            out_data[i] = f(a_data[ai]);
        }
    }
}

#[inline(always)]
fn apply_elementwise<F32, F64>(
    a: &BorrowedTensor,
    f32_fn: F32,
    f64_fn: F64,
) -> PyResult<OwnedTensor>
where
    F32: Fn(f32) -> f32 + Sync + Send,
    F64: Fn(f64) -> f64 + Sync + Send,
{
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => apply_elementwise_f32(a, &mut out, f32_fn),
        DType::F64 => apply_elementwise_f64(a, &mut out, f64_fn),
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

/// High-precision rational Chebyshev erf approximation (< 1.5e-7 max absolute error, auto-vectorizable).
#[inline(always)]
pub fn fast_erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911f32;
    let t = 1.0 / (1.0 + p * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let r = 1.0 - poly * (-ax * ax).exp();
    sign * r
}

#[inline(always)]
pub fn fast_gelu_f32(x: f32) -> f32 {
    // PyTorch `approximate="tanh"` : 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    const C: f32 = 0.7978845608028654; // sqrt(2/pi)
    const B: f32 = 0.044715;
    let x3 = x * x * x;
    let inner = C * (x + B * x3);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline(always)]
pub fn fast_gelu_f64(x: f64) -> f64 {
    const C: f64 = 0.7978845608028654;
    const B: f64 = 0.044715;
    let x3 = x * x * x;
    let inner = C * (x + B * x3);
    0.5 * x * (1.0 + inner.tanh())
}

fn apply_elementwise_param_f32(
    a: &BorrowedTensor,
    out: &mut OwnedTensor,
    f: impl Fn(f32) -> f32 + Sync,
) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        if n >= PAR_CHUNK {
            use rayon::prelude::*;
            out_data
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * PAR_CHUNK;
                    let a_slice = &a_data[base..base + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = f(a_slice[i]);
                    }
                });
        } else {
            for i in 0..n {
                out_data[i] = f(a_data[i]);
            }
        }
    } else {
        let rank = a.shape.len();
        let mut coords = vec![0usize; rank];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                coords[d] = rem % (a.shape[d].max(1) as usize);
                rem /= a.shape[d].max(1) as usize;
            }
            let mut ai = 0usize;
            for d in 0..rank {
                if a.shape[d] > 1 {
                    ai += coords[d] * a.strides[d] as usize;
                }
            }
            out_data[i] = f(a_data[ai]);
        }
    }
}

#[inline(always)]
fn apply_elementwise_param_f64(
    a: &BorrowedTensor,
    out: &mut OwnedTensor,
    f: impl Fn(f64) -> f64 + Sync,
) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        if n >= PAR_CHUNK {
            use rayon::prelude::*;
            out_data
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * PAR_CHUNK;
                    let a_slice = &a_data[base..base + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = f(a_slice[i]);
                    }
                });
        } else {
            for i in 0..n {
                out_data[i] = f(a_data[i]);
            }
        }
    } else {
        let rank = a.shape.len();
        let mut coords = vec![0usize; rank];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                coords[d] = rem % (a.shape[d].max(1) as usize);
                rem /= a.shape[d].max(1) as usize;
            }
            let mut ai = 0usize;
            for d in 0..rank {
                if a.shape[d] > 1 {
                    ai += coords[d] * a.strides[d] as usize;
                }
            }
            out_data[i] = f(a_data[ai]);
        }
    }
}

// ---------------------------------------------------------------------------
// Simple elementwise activations
// ---------------------------------------------------------------------------

pub fn sigmoid(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    apply_elementwise(
        a,
        |x| 1.0 / (1.0 + (-x).exp()),
        |x| 1.0 / (1.0 + (-x).exp()),
    )
}

pub fn tanh_act(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    apply_elementwise(a, |x| x.tanh(), |x| x.tanh())
}

#[inline(always)]
pub fn fast_exp_f32x8(x: f32x8) -> f32x8 {
    let min_val = f32x8::splat(-87.3);
    let max_val = f32x8::splat(88.7);
    let xc = x.fast_max(min_val).fast_min(max_val);

    let log2e = f32x8::splat(std::f32::consts::LOG2_E);
    let ln2 = f32x8::splat(std::f32::consts::LN_2);
    let z = xc * log2e;

    let z_arr = z.to_array();
    let mut n_arr = [0.0f32; 8];
    let mut p2n_arr = [0.0f32; 8];
    for i in 0..8 {
        let ni = z_arr[i].round() as i32;
        n_arr[i] = ni as f32;
        p2n_arr[i] = f32::from_bits(((ni + 127) << 23) as u32);
    }
    let n = f32x8::new(n_arr);
    let p2n = f32x8::new(p2n_arr);
    let f = xc - n * ln2;

    let one = f32x8::splat(1.0);
    let c2 = f32x8::splat(0.5);
    let c3 = f32x8::splat(0.16666666666666666);
    let c4 = f32x8::splat(0.041666666666666664);
    let c5 = f32x8::splat(0.008333333333333333);

    let poly = one + f * (one + f * (c2 + f * (c3 + f * (c4 + f * c5))));
    poly * p2n
}

#[inline(always)]
pub fn fast_erf_f32x8(x: f32x8) -> f32x8 {
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let neg_one = f32x8::splat(-1.0);
    let sign = zero.cmp_gt(x).blend(neg_one, one);
    let ax = x.fast_max(-x);
    let p = f32x8::splat(0.3275911);
    let t = one / (one + p * ax);

    let c1 = f32x8::splat(0.254829592);
    let c2 = f32x8::splat(-0.284496736);
    let c3 = f32x8::splat(1.421413741);
    let c4 = f32x8::splat(-1.453152027);
    let c5 = f32x8::splat(1.061405429);
    let poly = t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));

    let exp_neg_x2 = fast_exp_f32x8(-ax * ax);
    let r = one - poly * exp_neg_x2;
    sign * r
}

#[inline(always)]
pub fn exact_gelu_f32x8(x: f32x8) -> f32x8 {
    let inv_sqrt2 = f32x8::splat(0.7071067811865475);
    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);
    half * x * (one + fast_erf_f32x8(x * inv_sqrt2))
}

#[inline(always)]
pub fn exact_gelu_f32(x: f32) -> f32 {
    const INV_SQRT2: f32 = 0.7071067811865475;
    0.5 * x * (1.0 + fast_erf_f32(x * INV_SQRT2))
}

#[inline(always)]
pub fn exact_gelu_f64(x: f64) -> f64 {
    const INV_SQRT2: f64 = 0.7071067811865475;
    0.5 * x * (1.0 + fast_erf_f32((x * INV_SQRT2) as f32) as f64)
}

pub fn gelu(a: &BorrowedTensor, approximate: &str) -> PyResult<OwnedTensor> {
    if a.dtype == DType::F32 && a.strides == contiguous_strides(&a.shape) {
        let a_data = unsafe { typed_slice::<f32>(a) };
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        let n = out_data.len();
        if approximate == "none" {
            if n >= PAR_CHUNK {
                use rayon::prelude::*;
                out_data
                    .par_chunks_mut(PAR_CHUNK)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let base = ci * PAR_CHUNK;
                        let a_slice = &a_data[base..base + chunk.len()];
                        let n_simd = chunk.len() / 8;
                        for j in 0..n_simd {
                            let offset = j * 8;
                            let v = f32x8::from(
                                *<&[f32; 8]>::try_from(&a_slice[offset..offset + 8]).unwrap(),
                            );
                            let res = exact_gelu_f32x8(v);
                            chunk[offset..offset + 8].copy_from_slice(&res.to_array());
                        }
                        for j in (n_simd * 8)..chunk.len() {
                            chunk[j] = exact_gelu_f32(a_slice[j]);
                        }
                    });
            } else {
                let n_simd = n / 8;
                for j in 0..n_simd {
                    let offset = j * 8;
                    let v =
                        f32x8::from(*<&[f32; 8]>::try_from(&a_data[offset..offset + 8]).unwrap());
                    let res = exact_gelu_f32x8(v);
                    out_data[offset..offset + 8].copy_from_slice(&res.to_array());
                }
                for j in (n_simd * 8)..n {
                    out_data[j] = exact_gelu_f32(a_data[j]);
                }
            }
            return Ok(out);
        }
    }
    if approximate == "none" {
        apply_elementwise(a, exact_gelu_f32, exact_gelu_f64)
    } else {
        apply_elementwise(a, fast_gelu_f32, fast_gelu_f64)
    }
}

pub fn silu(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // SiLU / Swish: x * sigmoid(x)
    apply_elementwise(a, |x| x / (1.0 + (-x).exp()), |x| x / (1.0 + (-x).exp()))
}

pub fn leaky_relu(a: &BorrowedTensor, negative_slope: f64) -> PyResult<OwnedTensor> {
    let ns = negative_slope;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            apply_elementwise_param_f32(a, &mut out, |x| if x > 0.0 { x } else { x * ns as f32 })
        }
        DType::F64 => {
            apply_elementwise_param_f64(a, &mut out, |x| if x > 0.0 { x } else { x * ns })
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

pub fn elu(a: &BorrowedTensor, alpha: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => apply_elementwise_param_f32(a, &mut out, |x| {
            if x > 0.0 {
                x
            } else {
                alpha as f32 * (x.exp() - 1.0)
            }
        }),
        DType::F64 => apply_elementwise_param_f64(a, &mut out, |x| {
            if x > 0.0 {
                x
            } else {
                alpha * (x.exp() - 1.0)
            }
        }),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

pub fn selu(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let lambda: f64 = 1.0507009873554805;
    let alpha: f64 = 1.6732632423543772;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => apply_elementwise_param_f32(a, &mut out, |x| {
            (if x > 0.0 {
                x
            } else {
                alpha as f32 * (x.exp() - 1.0)
            }) * lambda as f32
        }),
        DType::F64 => apply_elementwise_param_f64(a, &mut out, |x| {
            (if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }) * lambda
        }),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

pub fn softplus(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    apply_elementwise(a, |x| (1.0 + x.exp()).ln(), |x| (1.0 + x.exp()).ln())
}

pub fn hardswish(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    apply_elementwise(
        a,
        |x| x * (x + 3.0).max(0.0).min(6.0) / 6.0,
        |x| x * (x + 3.0).max(0.0).min(6.0) / 6.0,
    )
}

pub fn mish(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    apply_elementwise(
        a,
        |x| x * (1.0 + x.exp()).ln().tanh(),
        |x| x * (1.0 + x.exp()).ln().tanh(),
    )
}

// ---------------------------------------------------------------------------
// Softmax — requires reduction along a dim, then exp + normalize
// ---------------------------------------------------------------------------

fn softmax_f32(a: &BorrowedTensor, dim: isize, out: &mut OwnedTensor) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let shape = &a.shape;
    let rank = shape.len();
    let d = if dim < 0 {
        (rank as isize + dim) as usize
    } else {
        dim as usize
    };

    let dim_size = shape[d] as usize;
    let mut inner_stride = 1i64;
    for i in (d + 1)..rank {
        inner_stride *= shape[i];
    }
    let inner_size = inner_stride as usize;

    let chunk_size = dim_size * inner_size;
    if chunk_size == 0 || out_data.is_empty() {
        return;
    }

    use rayon::prelude::*;
    out_data
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(outer, out_chunk)| {
            let a_chunk = &a_data[outer * chunk_size..(outer + 1) * chunk_size];
            for inner in 0..inner_size {
                let mut max_val = f32::NEG_INFINITY;
                for i in 0..dim_size {
                    let idx = i * inner_size + inner;
                    if a_chunk[idx] > max_val {
                        max_val = a_chunk[idx];
                    }
                }
                let mut sum = 0.0f32;
                for i in 0..dim_size {
                    let idx = i * inner_size + inner;
                    let val = (a_chunk[idx] - max_val).exp();
                    out_chunk[idx] = val;
                    sum += val;
                }
                let inv_sum = 1.0 / sum;
                for i in 0..dim_size {
                    let idx = i * inner_size + inner;
                    out_chunk[idx] *= inv_sum;
                }
            }
        });
}

fn softmax_f64(a: &BorrowedTensor, dim: isize, out: &mut OwnedTensor) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    let shape = &a.shape;
    let rank = shape.len();
    let d = if dim < 0 {
        (rank as isize + dim) as usize
    } else {
        dim as usize
    };

    let dim_size = shape[d] as usize;
    let mut inner_stride = 1i64;
    for i in (d + 1)..rank {
        inner_stride *= shape[i];
    }
    let inner_size = inner_stride as usize;

    let chunk_size = dim_size * inner_size;
    if chunk_size == 0 || out_data.is_empty() {
        return;
    }

    use rayon::prelude::*;
    out_data
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(outer, out_chunk)| {
            let a_chunk = &a_data[outer * chunk_size..(outer + 1) * chunk_size];
            for inner in 0..inner_size {
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..dim_size {
                    let idx = i * inner_size + inner;
                    if a_chunk[idx] > max_val {
                        max_val = a_chunk[idx];
                    }
                }
                let mut sum = 0.0f64;
                for i in 0..dim_size {
                    let idx = i * inner_size + inner;
                    let val = (a_chunk[idx] - max_val).exp();
                    out_chunk[idx] = val;
                    sum += val;
                }
                let inv_sum = 1.0 / sum;
                for i in 0..dim_size {
                    let idx = i * inner_size + inner;
                    out_chunk[idx] *= inv_sum;
                }
            }
        });
}

pub fn softmax(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let _contig;
    let a = if !a.is_contiguous() {
        _contig = crate::shape_ops::to_contiguous(a)?;
        BorrowedTensor::from_owned(&_contig)
    } else {
        a.clone()
    };
    let a = &a;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => softmax_f32(a, dim, &mut out),
        DType::F64 => softmax_f64(a, dim, &mut out),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

pub fn log_softmax(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let _contig;
    let a = if !a.is_contiguous() {
        _contig = crate::shape_ops::to_contiguous(a)?;
        BorrowedTensor::from_owned(&_contig)
    } else {
        a.clone()
    };
    let a = &a;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let a_data = unsafe { typed_slice::<f32>(a) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let shape = &a.shape;
            let rank = shape.len();
            let d = if dim < 0 {
                (rank as isize + dim) as usize
            } else {
                dim as usize
            };
            let dim_size = shape[d] as usize;
            let mut outer_stride = 1i64;
            for i in 0..d {
                outer_stride *= shape[i];
            }
            let mut inner_stride = 1i64;
            for i in (d + 1)..rank {
                inner_stride *= shape[i];
            }
            let outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let mut max_val = f32::NEG_INFINITY;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        if a_data[idx] > max_val {
                            max_val = a_data[idx];
                        }
                    }
                    let mut sum = 0.0f32;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        sum += (a_data[idx] - max_val).exp();
                    }
                    let log_sum = max_val + sum.ln();
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        out_data[idx] = a_data[idx] - log_sum;
                    }
                }
            }
        }
        DType::F64 => {
            let a_data = unsafe { typed_slice::<f64>(a) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            let shape = &a.shape;
            let rank = shape.len();
            let d = if dim < 0 {
                (rank as isize + dim) as usize
            } else {
                dim as usize
            };
            let dim_size = shape[d] as usize;
            let mut outer_stride = 1i64;
            for i in 0..d {
                outer_stride *= shape[i];
            }
            let mut inner_stride = 1i64;
            for i in (d + 1)..rank {
                inner_stride *= shape[i];
            }
            let outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let mut max_val = f64::NEG_INFINITY;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        if a_data[idx] > max_val {
                            max_val = a_data[idx];
                        }
                    }
                    let mut sum = 0.0f64;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        sum += (a_data[idx] - max_val).exp();
                    }
                    let log_sum = max_val + sum.ln();
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        out_data[idx] = a_data[idx] - log_sum;
                    }
                }
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

pub fn threshold_backward(
    grad: &BorrowedTensor,
    x: &BorrowedTensor,
    threshold: f64,
) -> PyResult<OwnedTensor> {
    let n = elem_count(&grad.shape);
    let mut out = OwnedTensor::new(grad.dtype, grad.shape.clone());
    let xn = elem_count(&x.shape);
    match grad.dtype {
        DType::F32 => {
            let g = unsafe { typed_slice::<f32>(grad) };
            let xd = unsafe { typed_slice::<f32>(x) };
            let o = unsafe { typed_mut_slice::<f32>(&mut out) };
            let th = threshold as f32;
            for i in 0..n {
                let xi = if xn == 1 { 0 } else { i % xn };
                o[i] = if xd[xi] > th { g[i] } else { 0.0 };
            }
        }
        DType::F64 => {
            let g = unsafe { typed_slice::<f64>(grad) };
            let xd = unsafe { typed_slice::<f64>(x) };
            let o = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                let xi = if xn == 1 { 0 } else { i % xn };
                o[i] = if xd[xi] > threshold { g[i] } else { 0.0 };
            }
        }
        _ => return Err(unsupported("threshold_backward requires f32/f64")),
    }
    Ok(out)
}
