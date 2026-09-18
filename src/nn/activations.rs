//! Activation functions — elementwise or reduction+elementwise.
//!
//! All activations support f32/f64, arbitrary strides, and are parallelized
//! via rayon for large tensors.

use crate::dlpack::{elem_count, unsupported, BorrowedTensor, DType, OwnedTensor};
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
    let contig = a.is_contiguous();
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
        const MAX_RANK: usize = 8;
        let mut coords = [0usize; MAX_RANK];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                coords[d] = rem % (a.shape[d].max(1) as usize);
                rem /= a.shape[d].max(1) as usize;
            }
            let mut ai = 0usize;
            for d in 0..rank.min(MAX_RANK) {
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
    let contig = a.is_contiguous();
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
        const MAX_RANK: usize = 8;
        let mut coords = [0usize; MAX_RANK];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                coords[d] = rem % (a.shape[d].max(1) as usize);
                rem /= a.shape[d].max(1) as usize;
            }
            let mut ai = 0usize;
            for d in 0..rank.min(MAX_RANK) {
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
        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
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
    let contig = a.is_contiguous();
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
    let contig = a.is_contiguous();
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
    // SIMD fast path for contiguous f32
    if a.dtype == DType::F32 && a.is_contiguous() {
        let a_data = unsafe { typed_slice::<f32>(a) };
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        let n = out_data.len();
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
                        let res = fast_sigmoid_f32x8(v);
                        chunk[offset..offset + 8].copy_from_slice(&res.to_array());
                    }
                    for j in (n_simd * 8)..chunk.len() {
                        chunk[j] = 1.0 / (1.0 + (-a_slice[j]).exp());
                    }
                });
        } else {
            let n_simd = n / 8;
            for j in 0..n_simd {
                let offset = j * 8;
                let v = f32x8::from(*<&[f32; 8]>::try_from(&a_data[offset..offset + 8]).unwrap());
                let res = fast_sigmoid_f32x8(v);
                out_data[offset..offset + 8].copy_from_slice(&res.to_array());
            }
            for j in (n_simd * 8)..n {
                out_data[j] = 1.0 / (1.0 + (-a_data[j]).exp());
            }
        }
        return Ok(out);
    }
    apply_elementwise(
        a,
        |x| 1.0 / (1.0 + (-x).exp()),
        |x| 1.0 / (1.0 + (-x).exp()),
    )
}

pub fn tanh_act(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // SIMD fast path for contiguous f32
    if a.dtype == DType::F32 && a.is_contiguous() {
        let a_data = unsafe { typed_slice::<f32>(a) };
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        let n = out_data.len();
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
                        let res = fast_tanh_f32x8(v);
                        chunk[offset..offset + 8].copy_from_slice(&res.to_array());
                    }
                    for j in (n_simd * 8)..chunk.len() {
                        chunk[j] = a_slice[j].tanh();
                    }
                });
        } else {
            let n_simd = n / 8;
            for j in 0..n_simd {
                let offset = j * 8;
                let v = f32x8::from(*<&[f32; 8]>::try_from(&a_data[offset..offset + 8]).unwrap());
                let res = fast_tanh_f32x8(v);
                out_data[offset..offset + 8].copy_from_slice(&res.to_array());
            }
            for j in (n_simd * 8)..n {
                out_data[j] = a_data[j].tanh();
            }
        }
        return Ok(out);
    }
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

/// SIMD sigmoid: 1 / (1 + exp(-x))
#[inline(always)]
pub fn fast_sigmoid_f32x8(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    one / (one + fast_exp_f32x8(-x))
}

/// SIMD tanh via exp: (e^2x - 1) / (e^2x + 1)
#[inline(always)]
pub fn fast_tanh_f32x8(x: f32x8) -> f32x8 {
    let two_x = x + x;
    let exp2x = fast_exp_f32x8(two_x);
    let one = f32x8::splat(1.0);
    let neg_one = f32x8::splat(-1.0);
    (exp2x + neg_one) / (exp2x + one)
}

/// SIMD SiLU / Swish: x * sigmoid(x) = x / (1 + exp(-x))
#[inline(always)]
pub fn fast_silu_f32x8(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    x / (one + fast_exp_f32x8(-x))
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
    if a.dtype == DType::F32 && a.is_contiguous() {
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
    // SiLU / Swish: x * sigmoid(x) = x / (1 + exp(-x))
    // SIMD fast path for contiguous f32
    if a.dtype == DType::F32 && a.is_contiguous() {
        let a_data = unsafe { typed_slice::<f32>(a) };
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        let n = out_data.len();
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
                        let res = fast_silu_f32x8(v);
                        chunk[offset..offset + 8].copy_from_slice(&res.to_array());
                    }
                    for j in (n_simd * 8)..chunk.len() {
                        chunk[j] = a_slice[j] / (1.0 + (-a_slice[j]).exp());
                    }
                });
        } else {
            let n_simd = n / 8;
            for j in 0..n_simd {
                let offset = j * 8;
                let v = f32x8::from(*<&[f32; 8]>::try_from(&a_data[offset..offset + 8]).unwrap());
                let res = fast_silu_f32x8(v);
                out_data[offset..offset + 8].copy_from_slice(&res.to_array());
            }
            for j in (n_simd * 8)..n {
                out_data[j] = a_data[j] / (1.0 + (-a_data[j]).exp());
            }
        }
        return Ok(out);
    }
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

        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
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

        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
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

        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
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
    if inner_size == 1 && dim_size >= 8 {
        // SIMD fast path: contiguous reduction along last dim (common transformer case)
        out_data
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(outer, out_chunk)| {
                let a_chunk = &a_data[outer * chunk_size..(outer + 1) * chunk_size];
                // Phase 1: find max using SIMD
                let mut max_val = f32::NEG_INFINITY;
                let mut i = 0;
                while i + 8 <= dim_size {
                    let v = f32x8::from(&a_chunk[i..i + 8]);
                    let lane_maxs: [f32; 8] = v.into();
                    for &x in &lane_maxs {
                        if x > max_val {
                            max_val = x;
                        }
                    }
                    i += 8;
                }
                while i < dim_size {
                    if a_chunk[i] > max_val {
                        max_val = a_chunk[i];
                    }
                    i += 1;
                }
                // Phase 2: exp(x - max) + sum using SIMD
                let mut sum = 0.0f32;
                i = 0;
                while i + 8 <= dim_size {
                    let v = f32x8::from(&a_chunk[i..i + 8]);
                    let shifted = v - f32x8::splat(max_val);
                    let exp_vals = fast_exp_f32x8(shifted);
                    let exp_arr: [f32; 8] = exp_vals.into();
                    for (j, &x) in exp_arr.iter().enumerate() {
                        out_chunk[i + j] = x;
                    }
                    sum += exp_arr.iter().sum::<f32>();
                    i += 8;
                }
                while i < dim_size {
                    let val = (a_chunk[i] - max_val).exp();
                    out_chunk[i] = val;
                    sum += val;
                    i += 1;
                }
                // Phase 3: normalize using SIMD
                let inv_sum = 1.0 / sum;
                i = 0;
                while i + 8 <= dim_size {
                    let v = f32x8::from(&out_chunk[i..i + 8]);
                    let normalized = v * f32x8::splat(inv_sum);
                    let arr: [f32; 8] = normalized.into();
                    out_chunk[i..i + 8].copy_from_slice(&arr);
                    i += 8;
                }
                while i < dim_size {
                    out_chunk[i] *= inv_sum;
                    i += 1;
                }
            });
    } else {
        // Scalar path for non-contiguous inner stride
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

        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
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
            let _outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;
            let chunk_size = dim_size * inner_size;
            if chunk_size == 0 || out_data.is_empty() {
                return Ok(out);
            }
            use rayon::prelude::*;
            if inner_size == 1 && dim_size >= 8 {
                // SIMD fast path
                out_data
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(outer, out_chunk)| {
                        let a_chunk = &a_data[outer * chunk_size..(outer + 1) * chunk_size];
                        // Max via SIMD
                        let mut max_val = f32::NEG_INFINITY;
                        let mut i = 0;
                        while i + 8 <= dim_size {
                            let v = f32x8::from(&a_chunk[i..i + 8]);
                            let lane_maxs: [f32; 8] = v.into();
                            for &x in &lane_maxs {
                                if x > max_val {
                                    max_val = x;
                                }
                            }
                            i += 8;
                        }
                        while i < dim_size {
                            if a_chunk[i] > max_val {
                                max_val = a_chunk[i];
                            }
                            i += 1;
                        }
                        // exp sum via SIMD
                        let mut sum = 0.0f32;
                        i = 0;
                        while i + 8 <= dim_size {
                            let v = f32x8::from(&a_chunk[i..i + 8]);
                            let shifted = v - f32x8::splat(max_val);
                            let exp_vals = fast_exp_f32x8(shifted);
                            let exp_arr: [f32; 8] = exp_vals.into();
                            sum += exp_arr.iter().sum::<f32>();
                            i += 8;
                        }
                        while i < dim_size {
                            sum += (a_chunk[i] - max_val).exp();
                            i += 1;
                        }
                        let log_sum = max_val + sum.ln();
                        // Subtract log_sum via SIMD
                        i = 0;
                        while i + 8 <= dim_size {
                            let v = f32x8::from(&a_chunk[i..i + 8]);
                            let result = v - f32x8::splat(log_sum);
                            let arr: [f32; 8] = result.into();
                            out_chunk[i..i + 8].copy_from_slice(&arr);
                            i += 8;
                        }
                        while i < dim_size {
                            out_chunk[i] = a_chunk[i] - log_sum;
                            i += 1;
                        }
                    });
            } else {
                // Scalar path
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
                                sum += (a_chunk[idx] - max_val).exp();
                            }
                            let log_sum = max_val + sum.ln();
                            for i in 0..dim_size {
                                let idx = i * inner_size + inner;
                                out_chunk[idx] = a_chunk[idx] - log_sum;
                            }
                        }
                    });
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

        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
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

// ---------------------------------------------------------------------------
// Binary Elementwise Activation Backward Helpers (SIMD + Rayon + Strided)
// ---------------------------------------------------------------------------

#[inline(always)]
fn run_binary_f32_simd<
    S: Fn(f32x8, f32x8) -> f32x8 + Sync + Send,
    F: Fn(f32, f32) -> f32 + Sync + Send,
>(
    g_data: &[f32],
    x_data: &[f32],
    out_data: &mut [f32],
    simd_op: S,
    scalar_op: F,
) {
    let n = out_data.len();
    if n >= PAR_CHUNK {
        use rayon::prelude::*;
        out_data
            .par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * PAR_CHUNK;
                let g_slice = &g_data[base..base + chunk.len()];
                let x_slice = &x_data[base..base + chunk.len()];
                let n_simd = chunk.len() / 8;
                for j in 0..n_simd {
                    let offset = j * 8;
                    let g_v =
                        f32x8::from(*<&[f32; 8]>::try_from(&g_slice[offset..offset + 8]).unwrap());
                    let x_v =
                        f32x8::from(*<&[f32; 8]>::try_from(&x_slice[offset..offset + 8]).unwrap());
                    let res = simd_op(g_v, x_v);
                    chunk[offset..offset + 8].copy_from_slice(&res.to_array());
                }
                for j in (n_simd * 8)..chunk.len() {
                    chunk[j] = scalar_op(g_slice[j], x_slice[j]);
                }
            });
    } else {
        let n_simd = n / 8;
        for j in 0..n_simd {
            let offset = j * 8;
            let g_v = f32x8::from(*<&[f32; 8]>::try_from(&g_data[offset..offset + 8]).unwrap());
            let x_v = f32x8::from(*<&[f32; 8]>::try_from(&x_data[offset..offset + 8]).unwrap());
            let res = simd_op(g_v, x_v);
            out_data[offset..offset + 8].copy_from_slice(&res.to_array());
        }
        for j in (n_simd * 8)..n {
            out_data[j] = scalar_op(g_data[j], x_data[j]);
        }
    }
}

#[inline(always)]
fn apply_binary_elementwise_f32<F: Fn(f32, f32) -> f32 + Sync + Send>(
    g: &BorrowedTensor,
    x: &BorrowedTensor,
    out: &mut OwnedTensor,
    f: F,
) {
    let g_data = unsafe { typed_slice::<f32>(g) };
    let x_data = unsafe { typed_slice::<f32>(x) };
    let n = out.elem_count();
    let out_shape = out.shape.clone();
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    if g.is_contiguous() && x.is_contiguous() && g.shape == x.shape {
        if n >= PAR_CHUNK {
            use rayon::prelude::*;
            out_data
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * PAR_CHUNK;
                    let g_slice = &g_data[base..base + chunk.len()];
                    let x_slice = &x_data[base..base + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = f(g_slice[i], x_slice[i]);
                    }
                });
        } else {
            for i in 0..n {
                out_data[i] = f(g_data[i], x_data[i]);
            }
        }
    } else {
        let rank = out_shape.len();
        const MAX_RANK: usize = 8;
        let mut coords = [0usize; MAX_RANK];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                let dim_sz = out_shape[d].max(1) as usize;
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }
            let mut gi = 0usize;
            let g_rank = g.shape.len();
            for d in 0..g_rank.min(MAX_RANK) {
                let coord_d = if g_rank <= rank {
                    let offset = rank - g_rank;
                    coords[d + offset]
                } else {
                    coords[d]
                };
                if g.shape[d] > 1 {
                    gi += (coord_d % (g.shape[d] as usize)) * g.strides[d] as usize;
                }
            }
            let mut xi = 0usize;
            let x_rank = x.shape.len();
            for d in 0..x_rank.min(MAX_RANK) {
                let coord_d = if x_rank <= rank {
                    let offset = rank - x_rank;
                    coords[d + offset]
                } else {
                    coords[d]
                };
                if x.shape[d] > 1 {
                    xi += (coord_d % (x.shape[d] as usize)) * x.strides[d] as usize;
                }
            }
            out_data[i] = f(g_data[gi], x_data[xi]);
        }
    }
}

#[inline(always)]
fn apply_binary_elementwise_f64<F: Fn(f64, f64) -> f64 + Sync + Send>(
    g: &BorrowedTensor,
    x: &BorrowedTensor,
    out: &mut OwnedTensor,
    f: F,
) {
    let g_data = unsafe { typed_slice::<f64>(g) };
    let x_data = unsafe { typed_slice::<f64>(x) };
    let n = out.elem_count();
    let out_shape = out.shape.clone();
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    if g.is_contiguous() && x.is_contiguous() && g.shape == x.shape {
        if n >= PAR_CHUNK {
            use rayon::prelude::*;
            out_data
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * PAR_CHUNK;
                    let g_slice = &g_data[base..base + chunk.len()];
                    let x_slice = &x_data[base..base + chunk.len()];
                    for i in 0..chunk.len() {
                        chunk[i] = f(g_slice[i], x_slice[i]);
                    }
                });
        } else {
            for i in 0..n {
                out_data[i] = f(g_data[i], x_data[i]);
            }
        }
    } else {
        let rank = out_shape.len();
        const MAX_RANK: usize = 8;
        let mut coords = [0usize; MAX_RANK];
        for i in 0..n {
            let mut rem = i;
            for d in (0..rank).rev() {
                let dim_sz = out_shape[d].max(1) as usize;
                coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }
            let mut gi = 0usize;
            let g_rank = g.shape.len();
            for d in 0..g_rank.min(MAX_RANK) {
                let coord_d = if g_rank <= rank {
                    let offset = rank - g_rank;
                    coords[d + offset]
                } else {
                    coords[d]
                };
                if g.shape[d] > 1 {
                    gi += (coord_d % (g.shape[d] as usize)) * g.strides[d] as usize;
                }
            }
            let mut xi = 0usize;
            let x_rank = x.shape.len();
            for d in 0..x_rank.min(MAX_RANK) {
                let coord_d = if x_rank <= rank {
                    let offset = rank - x_rank;
                    coords[d + offset]
                } else {
                    coords[d]
                };
                if x.shape[d] > 1 {
                    xi += (coord_d % (x.shape[d] as usize)) * x.strides[d] as usize;
                }
            }
            out_data[i] = f(g_data[gi], x_data[xi]);
        }
    }
}

// ---------------------------------------------------------------------------
// GeLU Backward
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn gelu_backward_tanh_f32(g: f32, x: f32) -> f32 {
    const C: f32 = 0.7978845608028654;
    const B: f32 = 0.044715;
    let x3 = x * x * x;
    let inner = C * (x + B * x3);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = C * (1.0 + 3.0 * B * x * x);
    g * (0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner)
}

#[inline(always)]
pub fn gelu_backward_tanh_f32x8(g: f32x8, x: f32x8) -> f32x8 {
    let c = f32x8::splat(0.7978845608028654);
    let b = f32x8::splat(0.044715);
    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);
    let three = f32x8::splat(3.0);
    let x2 = x * x;
    let x3 = x2 * x;
    let inner = c * (x + b * x3);
    let tanh_inner = fast_tanh_f32x8(inner);
    let sech2 = one - tanh_inner * tanh_inner;
    let d_inner = c * (one + three * b * x2);
    g * (half * (one + tanh_inner) + half * x * sech2 * d_inner)
}

#[inline(always)]
pub fn gelu_backward_tanh_f64(g: f64, x: f64) -> f64 {
    const C: f64 = 0.7978845608028654;
    const B: f64 = 0.044715;
    let x3 = x * x * x;
    let inner = C * (x + B * x3);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = C * (1.0 + 3.0 * B * x * x);
    g * (0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner)
}

#[inline(always)]
pub fn gelu_backward_exact_f32(g: f32, x: f32) -> f32 {
    const INV_SQRT2: f32 = 0.7071067811865475;
    const INV_SQRT_2PI: f32 = 0.3989422804014327;
    let cdf = 0.5 * (1.0 + fast_erf_f32(x * INV_SQRT2));
    let pdf = INV_SQRT_2PI * (-0.5 * x * x).exp();
    g * (cdf + x * pdf)
}

#[inline(always)]
pub fn gelu_backward_exact_f32x8(g: f32x8, x: f32x8) -> f32x8 {
    let inv_sqrt2 = f32x8::splat(0.7071067811865475);
    let inv_sqrt_2pi = f32x8::splat(0.3989422804014327);
    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);
    let cdf = half * (one + fast_erf_f32x8(x * inv_sqrt2));
    let pdf = inv_sqrt_2pi * fast_exp_f32x8(-half * x * x);
    g * (cdf + x * pdf)
}

#[inline(always)]
pub fn gelu_backward_exact_f64(g: f64, x: f64) -> f64 {
    const INV_SQRT2: f64 = 0.7071067811865475;
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    let cdf = 0.5 * (1.0 + fast_erf_f32((x * INV_SQRT2) as f32) as f64);
    let pdf = INV_SQRT_2PI * (-0.5 * x * x).exp();
    g * (cdf + x * pdf)
}

pub fn gelu_backward(
    grad: &BorrowedTensor,
    input: &BorrowedTensor,
    approximate: &str,
) -> PyResult<OwnedTensor> {
    if grad.dtype != input.dtype {
        return Err(unsupported("gelu_backward dtype mismatch"));
    }
    let out_shape = crate::kernels::broadcast_shape(&grad.shape, &input.shape)?;
    let mut out = OwnedTensor::new(grad.dtype, out_shape.clone());
    let is_tanh = approximate == "tanh";

    if grad.dtype == DType::F32
        && grad.is_contiguous()
        && input.is_contiguous()
        && grad.shape == input.shape
    {
        let g_data = unsafe { typed_slice::<f32>(grad) };
        let x_data = unsafe { typed_slice::<f32>(input) };
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        if is_tanh {
            run_binary_f32_simd(
                g_data,
                x_data,
                out_data,
                gelu_backward_tanh_f32x8,
                gelu_backward_tanh_f32,
            );
        } else {
            run_binary_f32_simd(
                g_data,
                x_data,
                out_data,
                gelu_backward_exact_f32x8,
                gelu_backward_exact_f32,
            );
        }
        return Ok(out);
    }

    match grad.dtype {
        DType::F32 => {
            if is_tanh {
                apply_binary_elementwise_f32(grad, input, &mut out, gelu_backward_tanh_f32);
            } else {
                apply_binary_elementwise_f32(grad, input, &mut out, gelu_backward_exact_f32);
            }
        }
        DType::F64 => {
            if is_tanh {
                apply_binary_elementwise_f64(grad, input, &mut out, gelu_backward_tanh_f64);
            } else {
                apply_binary_elementwise_f64(grad, input, &mut out, gelu_backward_exact_f64);
            }
        }
        _ => return Err(unsupported("gelu_backward requires f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SiLU Backward
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn silu_backward_f32(g: f32, x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    g * (s * (1.0 + x * (1.0 - s)))
}

#[inline(always)]
pub fn silu_backward_f32x8(g: f32x8, x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let s = one / (one + fast_exp_f32x8(-x));
    g * (s * (one + x * (one - s)))
}

#[inline(always)]
pub fn silu_backward_f64(g: f64, x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    g * (s * (1.0 + x * (1.0 - s)))
}

pub fn silu_backward(grad: &BorrowedTensor, input: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if grad.dtype != input.dtype {
        return Err(unsupported("silu_backward dtype mismatch"));
    }
    let out_shape = crate::kernels::broadcast_shape(&grad.shape, &input.shape)?;
    let mut out = OwnedTensor::new(grad.dtype, out_shape.clone());

    if grad.dtype == DType::F32
        && grad.is_contiguous()
        && input.is_contiguous()
        && grad.shape == input.shape
    {
        let g_data = unsafe { typed_slice::<f32>(grad) };
        let x_data = unsafe { typed_slice::<f32>(input) };
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        run_binary_f32_simd(
            g_data,
            x_data,
            out_data,
            silu_backward_f32x8,
            silu_backward_f32,
        );
        return Ok(out);
    }

    match grad.dtype {
        DType::F32 => {
            apply_binary_elementwise_f32(grad, input, &mut out, silu_backward_f32);
        }
        DType::F64 => {
            apply_binary_elementwise_f64(grad, input, &mut out, silu_backward_f64);
        }
        _ => return Err(unsupported("silu_backward requires f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Sigmoid Backward (grad_output, output)
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn sigmoid_backward_f32(g: f32, y: f32) -> f32 {
    g * y * (1.0 - y)
}

#[inline(always)]
pub fn sigmoid_backward_f32x8(g: f32x8, y: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    g * y * (one - y)
}

#[inline(always)]
pub fn sigmoid_backward_f64(g: f64, y: f64) -> f64 {
    g * y * (1.0 - y)
}

pub fn sigmoid_backward(grad: &BorrowedTensor, output: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if grad.dtype != output.dtype {
        return Err(unsupported("sigmoid_backward dtype mismatch"));
    }
    let out_shape = crate::kernels::broadcast_shape(&grad.shape, &output.shape)?;
    let mut out = OwnedTensor::new(grad.dtype, out_shape.clone());

    if grad.dtype == DType::F32
        && grad.is_contiguous()
        && output.is_contiguous()
        && grad.shape == output.shape
    {
        let g_data = unsafe { typed_slice::<f32>(grad) };
        let y_data = unsafe { typed_slice::<f32>(output) };
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        run_binary_f32_simd(
            g_data,
            y_data,
            out_data,
            sigmoid_backward_f32x8,
            sigmoid_backward_f32,
        );
        return Ok(out);
    }

    match grad.dtype {
        DType::F32 => {
            apply_binary_elementwise_f32(grad, output, &mut out, sigmoid_backward_f32);
        }
        DType::F64 => {
            apply_binary_elementwise_f64(grad, output, &mut out, sigmoid_backward_f64);
        }
        _ => return Err(unsupported("sigmoid_backward requires f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tanh Backward (grad_output, output)
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn tanh_backward_f32(g: f32, y: f32) -> f32 {
    g * (1.0 - y * y)
}

#[inline(always)]
pub fn tanh_backward_f32x8(g: f32x8, y: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    g * (one - y * y)
}

#[inline(always)]
pub fn tanh_backward_f64(g: f64, y: f64) -> f64 {
    g * (1.0 - y * y)
}

pub fn tanh_backward(grad: &BorrowedTensor, output: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if grad.dtype != output.dtype {
        return Err(unsupported("tanh_backward dtype mismatch"));
    }
    let out_shape = crate::kernels::broadcast_shape(&grad.shape, &output.shape)?;
    let mut out = OwnedTensor::new(grad.dtype, out_shape.clone());

    if grad.dtype == DType::F32
        && grad.is_contiguous()
        && output.is_contiguous()
        && grad.shape == output.shape
    {
        let g_data = unsafe { typed_slice::<f32>(grad) };
        let y_data = unsafe { typed_slice::<f32>(output) };
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        run_binary_f32_simd(
            g_data,
            y_data,
            out_data,
            tanh_backward_f32x8,
            tanh_backward_f32,
        );
        return Ok(out);
    }

    match grad.dtype {
        DType::F32 => {
            apply_binary_elementwise_f32(grad, output, &mut out, tanh_backward_f32);
        }
        DType::F64 => {
            apply_binary_elementwise_f64(grad, output, &mut out, tanh_backward_f64);
        }
        _ => return Err(unsupported("tanh_backward requires f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// LeakyReLU Backward
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn leaky_relu_backward_f32(g: f32, x: f32, ns: f32) -> f32 {
    if x > 0.0 {
        g
    } else {
        g * ns
    }
}

#[inline(always)]
pub fn leaky_relu_backward_f32x8(g: f32x8, x: f32x8, ns: f32x8) -> f32x8 {
    let zero = f32x8::splat(0.0);
    let mask = x.cmp_gt(zero);
    mask.blend(g, g * ns)
}

#[inline(always)]
pub fn leaky_relu_backward_f64(g: f64, x: f64, ns: f64) -> f64 {
    if x > 0.0 {
        g
    } else {
        g * ns
    }
}

pub fn leaky_relu_backward(
    grad: &BorrowedTensor,
    input: &BorrowedTensor,
    negative_slope: f64,
) -> PyResult<OwnedTensor> {
    if grad.dtype != input.dtype {
        return Err(unsupported("leaky_relu_backward dtype mismatch"));
    }
    let out_shape = crate::kernels::broadcast_shape(&grad.shape, &input.shape)?;
    let mut out = OwnedTensor::new(grad.dtype, out_shape.clone());

    if grad.dtype == DType::F32
        && grad.is_contiguous()
        && input.is_contiguous()
        && grad.shape == input.shape
    {
        let g_data = unsafe { typed_slice::<f32>(grad) };
        let x_data = unsafe { typed_slice::<f32>(input) };
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        let ns_f32 = negative_slope as f32;
        let ns_v = f32x8::splat(ns_f32);
        run_binary_f32_simd(
            g_data,
            x_data,
            out_data,
            |gv, xv| leaky_relu_backward_f32x8(gv, xv, ns_v),
            |g, x| leaky_relu_backward_f32(g, x, ns_f32),
        );
        return Ok(out);
    }

    match grad.dtype {
        DType::F32 => {
            let ns = negative_slope as f32;
            apply_binary_elementwise_f32(grad, input, &mut out, |g, x| {
                leaky_relu_backward_f32(g, x, ns)
            });
        }
        DType::F64 => {
            apply_binary_elementwise_f64(grad, input, &mut out, |g, x| {
                leaky_relu_backward_f64(g, x, negative_slope)
            });
        }
        _ => return Err(unsupported("leaky_relu_backward requires f32/f64")),
    }
    Ok(out)
}
