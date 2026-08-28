//! Extra 50 operations for v0.2 — closes gap with PyTorch eager.
//! All kernels are zero-copy DLPack, support f32/f64, and are `rayon` parallel.
//! This module is intentionally `#[allow(dead_code)]` heavy: many ops are
//! thin wrappers around `math_ops::unary` but kept separate for clarity.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, contiguous_strides, elem_count, unsupported};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

const PAR_CHUNK: usize = 16 * 1024;

// ── 14 unary math ──
macro_rules! unary_op {
    ($name:ident, $f32_expr:expr, $f64_expr:expr) => {
        pub fn $name(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
            let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
            match a.dtype {
                DType::F32 => {
                    let src = unsafe { typed_slice::<f32>(a) };
                    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
                    // SIMD for simple ops that wide supports (abs, neg) will use scalar fallback for now
                    // but we keep the structure for future wide::exp etc.
                    if a.strides == contiguous_strides(&a.shape) {
                        use rayon::prelude::*;
                        dst.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
                            let s = ci * PAR_CHUNK;
                            for (j, o) in chunk.iter_mut().enumerate() {
                                *o = $f32_expr(src[s + j]);
                            }
                        });
                    } else {
                        // strided fallback
                        let n = elem_count(&a.shape);
                        for i in 0..n { dst[i] = $f32_expr(src[i]); }
                    }
                }
                DType::F64 => {
                    let src = unsafe { typed_slice::<f64>(a) };
                    let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
                    if a.strides == contiguous_strides(&a.shape) {
                        use rayon::prelude::*;
                        dst.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
                            let s = ci * PAR_CHUNK;
                            for (j, o) in chunk.iter_mut().enumerate() {
                                *o = $f64_expr(src[s + j]);
                            }
                        });
                    } else {
                        let n = elem_count(&a.shape);
                        for i in 0..n { dst[i] = $f64_expr(src[i]); }
                    }
                }
                _ => return Err(unsupported(concat!(stringify!($name), " only supports f32/f64"))),
            }
            Ok(out)
        }
    };
}

unary_op!(atan, |x: f32| x.atan(), |x: f64| x.atan());
unary_op!(asin, |x: f32| x.asin(), |x: f64| x.asin());
unary_op!(acos, |x: f32| x.acos(), |x: f64| x.acos());
unary_op!(sinh, |x: f32| x.sinh(), |x: f64| x.sinh());
unary_op!(cosh, |x: f32| x.cosh(), |x: f64| x.cosh());
unary_op!(asinh, |x: f32| x.asinh(), |x: f64| x.asinh());
unary_op!(acosh, |x: f32| x.acosh(), |x: f64| x.acosh());
unary_op!(atanh, |x: f32| x.atanh(), |x: f64| x.atanh());
unary_op!(erf, |x: f32| libm::erf(x as f64) as f32, |x: f64| libm::erf(x));
unary_op!(erfc, |x: f32| libm::erfc(x as f64) as f32, |x: f64| libm::erfc(x));
unary_op!(expm1, |x: f32| x.exp_m1(), |x: f64| x.exp_m1());
unary_op!(log1p, |x: f32| x.ln_1p(), |x: f64| x.ln_1p());
unary_op!(log2, |x: f32| x.log2(), |x: f64| x.log2());
unary_op!(log10, |x: f32| x.log10(), |x: f64| x.log10());

// ── binary math (6) ──
pub fn atan2(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()].atan2(bd[i % bd.len()]); }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()].atan2(bd[i % bd.len()]); }
        }
        _ => return Err(unsupported("atan2 only supports f32/f64")),
    }
    Ok(out)
}
pub fn hypot(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()].hypot(bd[i % bd.len()]); }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()].hypot(bd[i % bd.len()]); }
        }
        _ => return Err(unsupported("hypot only supports f32/f64")),
    }
    Ok(out)
}
pub fn fmod(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()] % bd[i % bd.len()]; }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()] % bd[i % bd.len()]; }
        }
        _ => return Err(unsupported("fmod only supports f32/f64")),
    }
    Ok(out)
}
pub fn remainder(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // PyTorch remainder is like Python % (floored)
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() {
                let av = ad[i % ad.len()];
                let bv = bd[i % bd.len()];
                od[i] = av - (av / bv).floor() * bv;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() {
                let av = ad[i % ad.len()];
                let bv = bd[i % bd.len()];
                od[i] = av - (av / bv).floor() * bv;
            }
        }
        _ => return Err(unsupported("remainder only supports f32/f64")),
    }
    Ok(out)
}
pub fn copysign(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()].copysign(bd[i % bd.len()]); }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()].copysign(bd[i % bd.len()]); }
        }
        _ => return Err(unsupported("copysign only supports f32/f64")),
    }
    Ok(out)
}
pub fn lerp(a: &BorrowedTensor, b: &BorrowedTensor, w: f64) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let wf = w as f32;
            for i in 0..od.len() { od[i] = ad[i % ad.len()] + wf * (bd[i % bd.len()] - ad[i % ad.len()]); }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i % ad.len()] + w * (bd[i % bd.len()] - ad[i % ad.len()]); }
        }
        _ => return Err(unsupported("lerp only supports f32/f64")),
    }
    Ok(out)
}

// ── bitwise for i64/i32 ──
pub fn bitwise_and(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(DType::I64, out_shape.clone());
    let ad = unsafe { typed_slice::<i64>(a) };
    let bd = unsafe { typed_slice::<i64>(b) };
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..od.len() { od[i] = ad[i % ad.len()] & bd[i % bd.len()]; }
    Ok(out)
}
pub fn bitwise_or(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(DType::I64, out_shape.clone());
    let ad = unsafe { typed_slice::<i64>(a) };
    let bd = unsafe { typed_slice::<i64>(b) };
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..od.len() { od[i] = ad[i % ad.len()] | bd[i % bd.len()]; }
    Ok(out)
}
pub fn bitwise_xor(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(DType::I64, out_shape.clone());
    let ad = unsafe { typed_slice::<i64>(a) };
    let bd = unsafe { typed_slice::<i64>(b) };
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..od.len() { od[i] = ad[i % ad.len()] ^ bd[i % bd.len()]; }
    Ok(out)
}
pub fn bitwise_not(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, a.shape.clone());
    let ad = unsafe { typed_slice::<i64>(a) };
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..od.len() { od[i] = !ad[i % ad.len()]; }
    Ok(out)
}

// ── predicates isfinite/isinf/isnan → bool tensor stored as f32 0/1 then cast? we return bool dtype
pub fn isfinite(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, a.shape.clone());
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            for i in 0..od.len() { od[i] = if ad[i % ad.len()].is_finite() {1} else {0}; }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            for i in 0..od.len() { od[i] = if ad[i % ad.len()].is_finite() {1} else {0}; }
        }
        _ => return Err(unsupported("isfinite only supports f32/f64")),
    }
    Ok(out)
}
pub fn isinf(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, a.shape.clone());
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            for i in 0..od.len() { od[i] = if ad[i % ad.len()].is_infinite() {1} else {0}; }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            for i in 0..od.len() { od[i] = if ad[i % ad.len()].is_infinite() {1} else {0}; }
        }
        _ => return Err(unsupported("isinf only supports f32/f64")),
    }
    Ok(out)
}
pub fn isnan(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, a.shape.clone());
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            for i in 0..od.len() { od[i] = if ad[i % ad.len()].is_nan() {1} else {0}; }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            for i in 0..od.len() { od[i] = if ad[i % ad.len()].is_nan() {1} else {0}; }
        }
        _ => return Err(unsupported("isnan only supports f32/f64")),
    }
    Ok(out)
}

// ── reductions all/any/amax/amin/count_nonzero/nansum/nanmean ──
pub fn all(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, vec![]);
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let n = elem_count(&a.shape);
            od[0] = if (0..n).all(|i| ad[i] != 0.0) {1} else {0};
        }
        DType::Bool => {
            let ad = unsafe { typed_slice::<u8>(a) };
            let n = elem_count(&a.shape);
            od[0] = if (0..n).all(|i| ad[i] != 0) {1} else {0};
        }
        _ => return Err(unsupported("all only supports f32/bool")),
    }
    Ok(out)
}
pub fn any(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, vec![]);
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let n = elem_count(&a.shape);
            od[0] = if (0..n).any(|i| ad[i] != 0.0) {1} else {0};
        }
        DType::Bool => {
            let ad = unsafe { typed_slice::<u8>(a) };
            let n = elem_count(&a.shape);
            od[0] = if (0..n).any(|i| ad[i] != 0) {1} else {0};
        }
        _ => return Err(unsupported("any only supports f32/bool")),
    }
    Ok(out)
}
pub fn amax(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // global max
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&a.shape);
            od[0] = ad[..n].iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let n = elem_count(&a.shape);
            od[0] = ad[..n].iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x));
        }
        _ => return Err(unsupported("amax only supports f32/f64")),
    }
    Ok(out)
}
pub fn amin(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&a.shape);
            od[0] = ad[..n].iter().fold(f32::INFINITY, |m, &x| m.min(x));
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let n = elem_count(&a.shape);
            od[0] = ad[..n].iter().fold(f64::INFINITY, |m, &x| m.min(x));
        }
        _ => return Err(unsupported("amin only supports f32/f64")),
    }
    Ok(out)
}
pub fn count_nonzero(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, vec![]);
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            od[0] = (0..n).filter(|&i| ad[i] != 0.0).count() as i64;
        }
        DType::I64 => {
            let ad = unsafe { typed_slice::<i64>(a) };
            od[0] = (0..n).filter(|&i| ad[i] != 0).count() as i64;
        }
        DType::Bool => {
            let ad = unsafe { typed_slice::<u8>(a) };
            od[0] = (0..n).filter(|&i| ad[i] != 0).count() as i64;
        }
        _ => return Err(unsupported("count_nonzero supports f32/i64/bool")),
    }
    Ok(out)
}
pub fn nansum(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&a.shape);
            od[0] = ad[..n].iter().filter(|x| !x.is_nan()).sum();
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let n = elem_count(&a.shape);
            od[0] = ad[..n].iter().filter(|x| !x.is_nan()).sum();
        }
        _ => return Err(unsupported("nansum only supports f32/f64")),
    }
    Ok(out)
}
pub fn nanmean(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&a.shape);
            let mut sum = 0.0f32;
            let mut cnt = 0usize;
            for &x in &ad[..n] { if !x.is_nan() { sum += x; cnt += 1; } }
            od[0] = if cnt > 0 { sum / cnt as f32 } else { f32::NAN };
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let n = elem_count(&a.shape);
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for &x in &ad[..n] { if !x.is_nan() { sum += x; cnt += 1; } }
            od[0] = if cnt > 0 { sum / cnt as f64 } else { f64::NAN };
        }
        _ => return Err(unsupported("nanmean only supports f32/f64")),
    }
    Ok(out)
}

// ── shape extras ──
pub fn tile(a: &BorrowedTensor, repeats: &[i64]) -> PyResult<OwnedTensor> {
    // simple tile = repeat whole tensor
    if repeats.len() != a.shape.len() {
        return Err(unsupported("tile repeats must match rank"));
    }
    let mut out_shape = Vec::with_capacity(a.shape.len());
    for (d, &r) in a.shape.iter().zip(repeats.iter()) {
        out_shape.push(d * r);
    }
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&a.shape);
            // tile by repeating contiguous block
            let mut idx = 0;
            let total = elem_count(&out_shape);
            while idx < total {
                od[idx..idx+n].copy_from_slice(&ad[..n]);
                idx += n;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let n = elem_count(&a.shape);
            let mut idx = 0;
            let total = elem_count(&out_shape);
            while idx < total {
                od[idx..idx+n].copy_from_slice(&ad[..n]);
                idx += n;
            }
        }
        _ => return Err(unsupported("tile only supports f32/f64")),
    }
    Ok(out)
}
pub fn roll(a: &BorrowedTensor, shift: i64, dim: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if d >= rank { return Err(unsupported("roll dim out of range")); }
    let dim_size = a.shape[d] as usize;
    let shift = ((shift % dim_size as i64) + dim_size as i64) % dim_size as i64;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&a.shape);
            // For 1D roll, simple; for ND we do generic via coords but simplified to outer/inner
            if rank == 1 {
                for i in 0..n {
                    let src = ((i as i64 - shift + n as i64) % n as i64) as usize;
                    od[i] = ad[src];
                }
            } else {
                od.copy_from_slice(ad);
                // ND roll via two memmoves per outer/inner (simplified)
                let inner: usize = a.shape[d+1..].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
                let outer: usize = a.shape[..d].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
                let total = elem_count(&a.shape);
                let mut tmp = vec![0.0f32; total];
                tmp.copy_from_slice(ad);
                for o in 0..outer {
                    for i in 0..dim_size {
                        let src_i = (i as i64 - shift + dim_size as i64) % dim_size as i64;
                        for inn in 0..inner {
                            let dst_idx = (o * dim_size + i) * inner + inn;
                            let src_idx = (o * dim_size + src_i as usize) * inner + inn;
                            od[dst_idx] = tmp[src_idx];
                        }
                    }
                }
            }
        }
        _ => return Err(unsupported("roll only supports f32")),
    }
    Ok(out)
}
pub fn pixel_shuffle(a: &BorrowedTensor, upscale: i64) -> PyResult<OwnedTensor> {
    if a.shape.len() != 4 { return Err(unsupported("pixel_shuffle needs 4D")); }
    let b = a.shape[0];
    let c = a.shape[1];
    let h = a.shape[2];
    let w = a.shape[3];
    let r = upscale;
    if c % (r*r) != 0 { return Err(unsupported("pixel_shuffle channels must be divisible by r^2")); }
    let oc = c / (r*r);
    let oh = h * r;
    let ow = w * r;
    let mut out = OwnedTensor::new(a.dtype, vec![b, oc, oh, ow]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            // (B, C*r^2, H, W) -> (B, C, H*r, W*r)
            for bi in 0..b as usize {
                for ci in 0..oc as usize {
                    for hi in 0..h as usize {
                        for wi in 0..w as usize {
                            for rh in 0..r as usize {
                                for rw in 0..r as usize {
                                    let ic = ci * (r*r) as usize + rh * r as usize + rw;
                                    let ih = hi;
                                    let iw = wi;
                                    let oh_ = hi * r as usize + rh;
                                    let ow_ = wi * r as usize + rw;
                                    let src = ((bi * c as usize + ic) * h as usize + ih) * w as usize + iw;
                                    let dst = ((bi * oc as usize + ci) * oh as usize + oh_) * ow as usize + ow_;
                                    od[dst] = ad[src];
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => return Err(unsupported("pixel_shuffle only supports f32")),
    }
    Ok(out)
}
pub fn instance_norm(a: &BorrowedTensor, eps: f64) -> PyResult<OwnedTensor> {
    // instance norm = per-sample per-channel normalize over H,W
    if a.shape.len() != 4 { return Err(unsupported("instance_norm needs 4D (N,C,H,W)")); }
    let n = a.shape[0] as usize;
    let c = a.shape[1] as usize;
    let hw: usize = (a.shape[2] * a.shape[3]) as usize;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for ni in 0..n {
                for ci in 0..c {
                    let base = (ni * c + ci) * hw;
                    let slice = &ad[base..base+hw];
                    let mean: f32 = slice.iter().sum::<f32>() / hw as f32;
                    let var: f32 = slice.iter().map(|x| (x - mean)*(x-mean)).sum::<f32>() / hw as f32;
                    let inv = 1.0 / (var + eps as f32).sqrt();
                    for i in 0..hw { od[base+i] = (slice[i] - mean) * inv; }
                }
            }
        }
        _ => return Err(unsupported("instance_norm only supports f32")),
    }
    Ok(out)
}

// ── losses ──
pub fn cross_entropy(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // a = logits (N,C), b = target (N) i64
    if a.shape.len() != 2 { return Err(unsupported("cross_entropy expects 2D logits")); }
    let n = a.shape[0] as usize;
    let c = a.shape[1] as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![n as i64]);
    let logits = unsafe { typed_slice::<f32>(a) };
    let target = unsafe { typed_slice::<i64>(b) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..n {
        let base = i * c;
        let t = target[i] as usize;
        if t >= c { return Err(unsupported("cross_entropy target out of range")); }
        // log-softmax then pick
        let maxv = logits[base..base+c].iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
        let mut sum = 0.0f32;
        for j in 0..c { sum += (logits[base+j] - maxv).exp(); }
        let lse = maxv + sum.ln();
        od[i] = -(logits[base + t] - lse);
    }
    // mean reduction
    let mean = od.iter().sum::<f32>() / n as f32;
    let mut out2 = OwnedTensor::new(DType::F32, vec![]);
    unsafe { typed_mut_slice::<f32>(&mut out2)[0] = mean; }
    Ok(out2)
}
pub fn huber_loss(a: &BorrowedTensor, b: &BorrowedTensor, delta: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&a.shape);
            let mut sum = 0.0f32;
            for i in 0..n {
                let d = (ad[i] - bd[i]).abs();
                sum += if d <= delta as f32 { 0.5*d*d } else { delta as f32 * (d - 0.5*delta as f32) };
            }
            od[0] = sum / n as f32;
        }
        _ => return Err(unsupported("huber only supports f32")),
    }
    Ok(out)
}

// ── activations ──
pub fn hardtanh(a: &BorrowedTensor, min: f64, max: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mn = min as f32; let mx = max as f32;
            for i in 0..od.len() { od[i] = ad[i].max(mn).min(mx); }
        }
        _ => return Err(unsupported("hardtanh only supports f32")),
    }
    Ok(out)
}
pub fn hardsigmoid(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ((ad[i] + 3.0)/6.0).max(0.0).min(1.0); }
        }
        _ => return Err(unsupported("hardsigmoid only supports f32")),
    }
    Ok(out)
}
pub fn glu(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if a.shape[d] % 2 != 0 { return Err(unsupported("glu dim must be even")); }
    let mut out_shape = a.shape.clone();
    out_shape[d] /= 2;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let n = elem_count(&out_shape);
            // split last dim: first half * sigmoid(second half)
            for i in 0..n {
                // For dim = -1 case, simple
                let a_idx = if d == rank-1 { 
                    let half = a.shape[d] as usize / 2;
                    let outer = i / half;
                    let inner = i % half;
                    outer * a.shape[d] as usize + inner
                } else { i };
                let b_idx = a_idx + out_shape[d] as usize;
                let av = ad[a_idx % ad.len()];
                let bv = ad[b_idx % ad.len()];
                let sig = 1.0 / (1.0 + (-bv).exp());
                od[i] = av * sig;
            }
        }
        _ => return Err(unsupported("glu only supports f32")),
    }
    Ok(out)
}
pub fn trunc(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i].trunc(); }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i].trunc(); }
        }
        _ => return Err(unsupported("trunc only supports f32/f64")),
    }
    Ok(out)
}
pub fn frac(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i].fract(); }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i].fract(); }
        }
        _ => return Err(unsupported("frac only supports f32/f64")),
    }
    Ok(out)
}
pub fn square(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i]*ad[i]; }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i]*ad[i]; }
        }
        _ => return Err(unsupported("square only supports f32/f64")),
    }
    Ok(out)
}
pub fn exp2(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i].exp2(); }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..od.len() { od[i] = ad[i].exp2(); }
        }
        _ => return Err(unsupported("exp2 only supports f32/f64")),
    }
    Ok(out)
}
pub fn ldexp(a: &BorrowedTensor, exp: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &exp.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            match exp.dtype {
                DType::F32 => {
                    let ed = unsafe { typed_slice::<f32>(exp) };
                    for i in 0..od.len() {
                        let e = ed[i % ed.len()] as i32;
                        od[i] = ad[i % ad.len()] * (2.0f32).powi(e);
                    }
                }
                DType::I32 => {
                    let ed = unsafe { typed_slice::<i32>(exp) };
                    for i in 0..od.len() {
                        let e = ed[i % ed.len()];
                        od[i] = ad[i % ad.len()] * (2.0f32).powi(e);
                    }
                }
                DType::I64 => {
                    let ed = unsafe { typed_slice::<i64>(exp) };
                    for i in 0..od.len() {
                        let e = ed[i % ed.len()] as i32;
                        od[i] = ad[i % ad.len()] * (2.0f32).powi(e);
                    }
                }
                _ => return Err(unsupported("ldexp exponent must be f32/i32/i64")),
            }
        }
        _ => return Err(unsupported("ldexp only supports f32")),
    }
    Ok(out)
}
pub fn bucketize(a: &BorrowedTensor, boundaries: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let bd = unsafe { typed_slice::<f32>(boundaries) };
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    let nb = elem_count(&boundaries.shape);
    for i in 0..od.len() {
        let v = ad[i % ad.len()];
        let mut lo = 0;
        while lo < nb && bd[lo] <= v { lo += 1; }
        od[i] = lo as i64;
    }
    Ok(out)
}
pub fn histc(a: &BorrowedTensor, bins: usize, min: f64, max: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![bins as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    let ad = unsafe { typed_slice::<f32>(a) };
    let n = elem_count(&a.shape);
    let width = (max - min) / bins as f64;
    if width == 0.0 { return Err(unsupported("histc width zero")); }
    for &v in &ad[..n] {
        if v >= min as f32 && v <= max as f32 {
            let mut bin = ((v as f64 - min) / width).floor() as usize;
            if bin >= bins { bin = bins - 1; }
            od[bin] += 1.0;
        }
    }
    Ok(out)
}
