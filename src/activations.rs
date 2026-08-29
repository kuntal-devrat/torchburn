//! Activation functions — elementwise or reduction+elementwise.
//!
//! All activations support f32/f64, arbitrary strides, and are parallelized
//! via rayon for large tensors.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, contiguous_strides, elem_count, unsupported};
use pyo3::prelude::*;

const PAR_CHUNK: usize = 16 * 1024;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ---------------------------------------------------------------------------
// Generic elementwise activation helper
// ---------------------------------------------------------------------------

fn apply_elementwise_f32(a: &BorrowedTensor, out: &mut OwnedTensor, f: fn(f32) -> f32) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        use rayon::prelude::*;
        out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
            let start = ci * PAR_CHUNK;
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = f(a_data[start + i]);
            }
        });
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

fn apply_elementwise_f64(a: &BorrowedTensor, out: &mut OwnedTensor, f: fn(f64) -> f64) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        use rayon::prelude::*;
        out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
            let start = ci * PAR_CHUNK;
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = f(a_data[start + i]);
            }
        });
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

fn apply_elementwise(a: &BorrowedTensor, f32_fn: fn(f32) -> f32, f64_fn: fn(f64) -> f64) -> PyResult<OwnedTensor> {
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

fn apply_elementwise_param_f32(a: &BorrowedTensor, out: &mut OwnedTensor, f: impl Fn(f32) -> f32 + Sync) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        use rayon::prelude::*;
        out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
            let start = ci * PAR_CHUNK;
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = f(a_data[start + i]);
            }
        });
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

fn apply_elementwise_param_f64(a: &BorrowedTensor, out: &mut OwnedTensor, f: impl Fn(f64) -> f64 + Sync) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let n = out.elem_count();
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    let contig = a.strides == contiguous_strides(&a.shape);
    if contig {
        use rayon::prelude::*;
        out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
            let start = ci * PAR_CHUNK;
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = f(a_data[start + i]);
            }
        });
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
    apply_elementwise(a, |x| 1.0 / (1.0 + (-x).exp()), |x| 1.0 / (1.0 + (-x).exp()))
}

pub fn tanh_act(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    apply_elementwise(a, |x| x.tanh(), |x| x.tanh())
}

pub fn gelu(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))) — exact, matches PyTorch
    apply_elementwise(
        a,
        |x| (0.5 * (x as f64) * (1.0 + libm::erf((x as f64) / std::f64::consts::SQRT_2))) as f32,
        |x| 0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2)),
    )
}

pub fn silu(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // SiLU / Swish: x * sigmoid(x)
    apply_elementwise(a, |x| x / (1.0 + (-x).exp()), |x| x / (1.0 + (-x).exp()))
}

pub fn leaky_relu(a: &BorrowedTensor, negative_slope: f64) -> PyResult<OwnedTensor> {
    let ns = negative_slope;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => apply_elementwise_param_f32(a, &mut out, |x| if x > 0.0 { x } else { x * ns as f32 }),
        DType::F64 => apply_elementwise_param_f64(a, &mut out, |x| if x > 0.0 { x } else { x * ns }),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
    Ok(out)
}

pub fn elu(a: &BorrowedTensor, alpha: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => apply_elementwise_param_f32(a, &mut out, |x| if x > 0.0 { x } else { alpha as f32 * (x.exp() - 1.0) }),
        DType::F64 => apply_elementwise_param_f64(a, &mut out, |x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }),

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
        DType::F32 => apply_elementwise_param_f32(a, &mut out, |x| (if x > 0.0 { x } else { alpha as f32 * (x.exp() - 1.0) }) * lambda as f32),
        DType::F64 => apply_elementwise_param_f64(a, &mut out, |x| (if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }) * lambda),

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
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };

    let dim_size = shape[d] as usize;
    let mut outer_stride = 1i64;
    for i in 0..d { outer_stride *= shape[i]; }
    let mut inner_stride = 1i64;
    for i in (d + 1)..rank { inner_stride *= shape[i]; }

    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    // 2-pass with wide f32x8 for exp + normalize fused
    use wide::f32x8;
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                if a_data[idx] > max_val { max_val = a_data[idx]; }
            }
            // wide exp for 8 at a time
            let mut sum = 0.0f32;
            let mut i = 0;
            while i + 8 <= dim_size {
                let mut vals = [0.0f32; 8];
                for j in 0..8 {
                    let idx = outer * (dim_size * inner_size) + (i+j) * inner_size + inner;
                    vals[j] = a_data[idx] - max_val;
                }
                let _av = f32x8::new(vals);
                // fast exp via libm per lane (wide doesn't have exp poly for all, use scalar fallback for now)
                let mut exp_vals = [0.0f32; 8];
                for j in 0..8 { exp_vals[j] = vals[j].exp(); }
                for j in 0..8 {
                    let idx = outer * (dim_size * inner_size) + (i+j) * inner_size + inner;
                    out_data[idx] = exp_vals[j];
                    sum += exp_vals[j];
                }
                i += 8;
            }
            while i < dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                let val = (a_data[idx] - max_val).exp();
                out_data[idx] = val;
                sum += val;
                i += 1;
            }
            let inv_sum = 1.0 / sum;
            // vectorized divide
            let mut i = 0;
            while i + 8 <= dim_size {
                let base = outer * (dim_size * inner_size) + i * inner_size + inner;
                // gather 8 strided values into contiguous for SIMD divide
                let mut tmp = [0.0f32; 8];
                for j in 0..8 { tmp[j] = out_data[base + j*inner_size]; }
                let av = f32x8::new(tmp);
                let rv = av * f32x8::splat(inv_sum);
                let arr = rv.to_array();
                for j in 0..8 {
                    let idx = outer * (dim_size * inner_size) + (i+j) * inner_size + inner;
                    out_data[idx] = arr[j];
                }
                i += 8;
            }
            while i < dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                out_data[idx] *= inv_sum;
                i += 1;
            }
        }
    }
}

fn softmax_f64(a: &BorrowedTensor, dim: isize, out: &mut OwnedTensor) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    let shape = &a.shape;
    let rank = shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };

    let dim_size = shape[d] as usize;
    let mut outer_stride = 1i64;
    for i in 0..d { outer_stride *= shape[i]; }
    let mut inner_stride = 1i64;
    for i in (d + 1)..rank { inner_stride *= shape[i]; }

    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                if a_data[idx] > max_val { max_val = a_data[idx]; }
            }
            let mut sum = 0.0f64;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                let val = (a_data[idx] - max_val).exp();
                out_data[idx] = val;
                sum += val;
            }
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                out_data[idx] /= sum;
            }
        }
    }
}

pub fn softmax(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
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
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let a_data = unsafe { typed_slice::<f32>(a) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let shape = &a.shape;
            let rank = shape.len();
            let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
            let dim_size = shape[d] as usize;
            let mut outer_stride = 1i64;
            for i in 0..d { outer_stride *= shape[i]; }
            let mut inner_stride = 1i64;
            for i in (d + 1)..rank { inner_stride *= shape[i]; }
            let outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let mut max_val = f32::NEG_INFINITY;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        if a_data[idx] > max_val { max_val = a_data[idx]; }
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
            let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
            let dim_size = shape[d] as usize;
            let mut outer_stride = 1i64;
            for i in 0..d { outer_stride *= shape[i]; }
            let mut inner_stride = 1i64;
            for i in (d + 1)..rank { inner_stride *= shape[i]; }
            let outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let mut max_val = f64::NEG_INFINITY;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        if a_data[idx] > max_val { max_val = a_data[idx]; }
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

pub fn threshold_backward(grad: &BorrowedTensor, x: &BorrowedTensor, threshold: f64) -> PyResult<OwnedTensor> {
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
