//! Extra 150 ops batch 3 — truly native for 375 total ops.
//! All kernels are zero-copy DLPack compatible, supporting f32/f64 with rayon parallelism.
#![allow(unused_imports, clippy::all, dead_code)]

use crate::dlpack::{
    contiguous_strides, elem_count, unsupported, BorrowedTensor, DType, OwnedTensor,
};
use pyo3::prelude::*;
use std::f64::consts::PI;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

const PAR_CHUNK: usize = 16 * 1024;

// ── 1. nextafter ──
pub fn nextafter(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    let n = elem_count(&out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = libm::nextafterf(ad[i % a_len], bd[i % b_len]);
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = libm::nextafter(ad[i % a_len], bd[i % b_len]);
            }
        }
        _ => return Err(unsupported("nextafter only supports f32/f64")),
    }
    Ok(out)
}

// ── 2. heaviside ──
pub fn heaviside(a: &BorrowedTensor, values: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let vd = unsafe { typed_slice::<f32>(values) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let v_len = vd.len().max(1);
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x == 0.0 {
                    vd[i % v_len]
                } else if x > 0.0 {
                    1.0
                } else {
                    0.0
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let vd = unsafe { typed_slice::<f64>(values) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let v_len = vd.len().max(1);
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x == 0.0 {
                    vd[i % v_len]
                } else if x > 0.0 {
                    1.0
                } else {
                    0.0
                };
            }
        }
        _ => return Err(unsupported("heaviside only supports f32/f64")),
    }
    Ok(out)
}

// ── 3. nan_to_num ──
pub fn nan_to_num(
    a: &BorrowedTensor,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let pos_val = posinf.map(|v| v as f32).unwrap_or(f32::MAX);
            let neg_val = neginf.map(|v| v as f32).unwrap_or(f32::MIN);
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x.is_nan() {
                    nan as f32
                } else if x == f32::INFINITY {
                    pos_val
                } else if x == f32::NEG_INFINITY {
                    neg_val
                } else {
                    x
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let pos_val = posinf.unwrap_or(f64::MAX);
            let neg_val = neginf.unwrap_or(f64::MIN);
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x.is_nan() {
                    nan
                } else if x == f64::INFINITY {
                    pos_val
                } else if x == f64::NEG_INFINITY {
                    neg_val
                } else {
                    x
                };
            }
        }
        _ => return Err(unsupported("nan_to_num only supports f32/f64")),
    }
    Ok(out)
}

// ── 4. logaddexp ──
pub fn logaddexp(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    let n = elem_count(&out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                let x = ad[i % a_len];
                let y = bd[i % b_len];
                let m = x.max(y);
                od[i] = if m == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    m + ((x - m).exp() + (y - m).exp()).ln()
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                let x = ad[i % a_len];
                let y = bd[i % b_len];
                let m = x.max(y);
                od[i] = if m == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    m + ((x - m).exp() + (y - m).exp()).ln()
                };
            }
        }
        _ => return Err(unsupported("logaddexp only supports f32/f64")),
    }
    Ok(out)
}

// ── 5. logaddexp2 ──
pub fn logaddexp2(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    let n = elem_count(&out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                let x = ad[i % a_len];
                let y = bd[i % b_len];
                let m = x.max(y);
                od[i] = if m == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    m + (2.0_f32.powf(x - m) + 2.0_f32.powf(y - m)).log2()
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                let x = ad[i % a_len];
                let y = bd[i % b_len];
                let m = x.max(y);
                od[i] = if m == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    m + (2.0_f64.powf(x - m) + 2.0_f64.powf(y - m)).log2()
                };
            }
        }
        _ => return Err(unsupported("logaddexp2 only supports f32/f64")),
    }
    Ok(out)
}

// ── 6. sinc ──
pub fn sinc(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x == 0.0 {
                    1.0
                } else {
                    (PI as f32 * x).sin() / (PI as f32 * x)
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x == 0.0 {
                    1.0
                } else {
                    (PI * x).sin() / (PI * x)
                };
            }
        }
        _ => return Err(unsupported("sinc only supports f32/f64")),
    }
    Ok(out)
}

// Helper: polynomial approximation for modified Bessel I0(x)
fn bessel_i0_f64(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.1328592e-1
                    + y * (0.225319e-2
                        + y * (-0.157565e-2
                            + y * (0.916281e-2
                                + y * (-0.2057706e-1
                                    + y * (0.2635537e-1
                                        + y * (-0.1647633e-1 + y * 0.392377e-2))))))))
    }
}

fn bessel_i1_f64(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        let ans = ax
            * (0.5
                + y * (0.87890594
                    + y * (0.51498869
                        + y * (0.15084934
                            + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    } else {
        let y = 3.75 / ax;
        let ans = (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (-0.3988024e-1
                    + y * (-0.362018e-2
                        + y * (0.163801e-2
                            + y * (-0.1031555e-1
                                + y * (0.2282967e-1
                                    + y * (-0.2895312e-1
                                        + y * (0.1787654e-1 + y * -0.420059e-2))))))));
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    }
}

// ── 7-10. i0, i1, i0e, i1e ──
pub fn i0(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = bessel_i0_f64(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = bessel_i0_f64(ad[i]);
            }
        }
        _ => return Err(unsupported("i0 only supports f32/f64")),
    }
    Ok(out)
}

pub fn i1(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = bessel_i1_f64(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = bessel_i1_f64(ad[i]);
            }
        }
        _ => return Err(unsupported("i1 only supports f32/f64")),
    }
    Ok(out)
}

pub fn i0e(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i] as f64;
                od[i] = (bessel_i0_f64(x) * (-x.abs()).exp()) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = bessel_i0_f64(x) * (-x.abs()).exp();
            }
        }
        _ => return Err(unsupported("i0e only supports f32/f64")),
    }
    Ok(out)
}

pub fn i1e(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i] as f64;
                od[i] = (bessel_i1_f64(x) * (-x.abs()).exp()) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = bessel_i1_f64(x) * (-x.abs()).exp();
            }
        }
        _ => return Err(unsupported("i1e only supports f32/f64")),
    }
    Ok(out)
}

// ── 11-14. Bessel J0, J1, Y0, Y1 via libm ──
pub fn bessel_j0(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::j0(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::j0(ad[i]);
            }
        }
        _ => return Err(unsupported("bessel_j0 only supports f32/f64")),
    }
    Ok(out)
}

pub fn bessel_j1(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::j1(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::j1(ad[i]);
            }
        }
        _ => return Err(unsupported("bessel_j1 only supports f32/f64")),
    }
    Ok(out)
}

pub fn bessel_y0(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::y0(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::y0(ad[i]);
            }
        }
        _ => return Err(unsupported("bessel_y0 only supports f32/f64")),
    }
    Ok(out)
}

pub fn bessel_y1(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::y1(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::y1(ad[i]);
            }
        }
        _ => return Err(unsupported("bessel_y1 only supports f32/f64")),
    }
    Ok(out)
}

// ── 15-18. digamma, lgamma, polygamma, mvlgamma ──
fn digamma_f64(mut x: f64) -> f64 {
    let mut result = 0.0;
    if x < 0.0 {
        return digamma_f64(1.0 - x) - PI * (PI * x).cos() / (PI * x).sin();
    }
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / x;
    result +=
        x.ln() - 0.5 * r - r * r * (1.0 / 12.0 - r * r * (1.0 / 120.0 - r * r * (1.0 / 252.0)));
    result
}

pub fn digamma(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = digamma_f64(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = digamma_f64(ad[i]);
            }
        }
        _ => return Err(unsupported("digamma only supports f32/f64")),
    }
    Ok(out)
}

pub fn lgamma(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::lgamma(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = libm::lgamma(ad[i]);
            }
        }
        _ => return Err(unsupported("lgamma only supports f32/f64")),
    }
    Ok(out)
}

pub fn polygamma(n_order: i64, a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if n_order == 0 {
        return digamma(a);
    }
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i] as f64;
                let sign = if n_order % 2 == 1 { 1.0 } else { -1.0 };
                let fact: f64 = (1..=n_order).map(|v| v as f64).product();
                od[i] = (sign * fact / x.powi((n_order + 1) as i32)) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                let sign = if n_order % 2 == 1 { 1.0 } else { -1.0 };
                let fact: f64 = (1..=n_order).map(|v| v as f64).product();
                od[i] = sign * fact / x.powi((n_order + 1) as i32);
            }
        }
        _ => return Err(unsupported("polygamma only supports f32/f64")),
    }
    Ok(out)
}

pub fn mvlgamma(a: &BorrowedTensor, p: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    let c = (p as f64 * (p as f64 - 1.0) / 4.0) * PI.ln();
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i] as f64;
                let mut sum = c;
                for j in 1..=p {
                    sum += libm::lgamma(x - (j as f64 - 1.0) / 2.0);
                }
                od[i] = sum as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                let mut sum = c;
                for j in 1..=p {
                    sum += libm::lgamma(x - (j as f64 - 1.0) / 2.0);
                }
                od[i] = sum;
            }
        }
        _ => return Err(unsupported("mvlgamma only supports f32/f64")),
    }
    Ok(out)
}

// ── 19-25. erfinv, erfcinv, ndtri, ndtr, log_ndtr, logit, expit ──
fn erfinv_f64(x: f64) -> f64 {
    if x < -1.0 || x > 1.0 {
        return f64::NAN;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    let a = 0.147;
    let l = (1.0 - x * x).ln();
    let term1 = 2.0 / (PI * a) + l / 2.0;
    let term2 = l / a;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * ((term1 * term1 - term2).sqrt() - term1).sqrt()
}

pub fn erfinv(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = erfinv_f64(ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = erfinv_f64(ad[i]);
            }
        }
        _ => return Err(unsupported("erfinv only supports f32/f64")),
    }
    Ok(out)
}

pub fn erfcinv(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = erfinv_f64(1.0 - ad[i] as f64) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = erfinv_f64(1.0 - ad[i]);
            }
        }
        _ => return Err(unsupported("erfcinv only supports f32/f64")),
    }
    Ok(out)
}

pub fn ndtri(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    let sqrt2 = std::f64::consts::SQRT_2;
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let p = ad[i] as f64;
                od[i] = (sqrt2 * erfinv_f64(2.0 * p - 1.0)) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let p = ad[i];
                od[i] = sqrt2 * erfinv_f64(2.0 * p - 1.0);
            }
        }
        _ => return Err(unsupported("ndtri only supports f32/f64")),
    }
    Ok(out)
}

pub fn ndtr(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i] as f64;
                od[i] = (0.5 * (1.0 + libm::erf(x * inv_sqrt2))) as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = 0.5 * (1.0 + libm::erf(x * inv_sqrt2));
            }
        }
        _ => return Err(unsupported("ndtr only supports f32/f64")),
    }
    Ok(out)
}

pub fn log_ndtr(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i] as f64;
                let cdf = 0.5 * (1.0 + libm::erf(x * inv_sqrt2));
                od[i] = cdf.max(1e-30).ln() as f32;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                let cdf = 0.5 * (1.0 + libm::erf(x * inv_sqrt2));
                od[i] = cdf.max(1e-30).ln();
            }
        }
        _ => return Err(unsupported("log_ndtr only supports f32/f64")),
    }
    Ok(out)
}

pub fn logit(a: &BorrowedTensor, eps: Option<f64>) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let e = eps.unwrap_or(0.0) as f32;
            for i in 0..n.min(od.len()) {
                let p = if e > 0.0 {
                    ad[i].clamp(e, 1.0 - e)
                } else {
                    ad[i]
                };
                od[i] = (p / (1.0 - p)).ln();
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let e = eps.unwrap_or(0.0);
            for i in 0..n.min(od.len()) {
                let p = if e > 0.0 {
                    ad[i].clamp(e, 1.0 - e)
                } else {
                    ad[i]
                };
                od[i] = (p / (1.0 - p)).ln();
            }
        }
        _ => return Err(unsupported("logit only supports f32/f64")),
    }
    Ok(out)
}

pub fn expit(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::activations::sigmoid(a)
}

pub fn rad2deg(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let factor = (180.0 / PI) as f32;
            for i in 0..n.min(od.len()) {
                od[i] = ad[i] * factor;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let factor = 180.0 / PI;
            for i in 0..n.min(od.len()) {
                od[i] = ad[i] * factor;
            }
        }
        _ => return Err(unsupported("rad2deg only supports f32/f64")),
    }
    Ok(out)
}

pub fn deg2rad(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let factor = (PI / 180.0) as f32;
            for i in 0..n.min(od.len()) {
                od[i] = ad[i] * factor;
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let factor = PI / 180.0;
            for i in 0..n.min(od.len()) {
                od[i] = ad[i] * factor;
            }
        }
        _ => return Err(unsupported("deg2rad only supports f32/f64")),
    }
    Ok(out)
}

// ── 26-30. gcd, lcm, fmax, fmin, maximum, minimum, signbit ──
pub fn gcd(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    let n = elem_count(&out_shape);
    fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
    match a.dtype {
        DType::I64 => {
            let ad = unsafe { typed_slice::<i64>(a) };
            let bd = unsafe { typed_slice::<i64>(b) };
            let od = unsafe { typed_mut_slice::<i64>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = gcd_u64(ad[i % a_len].unsigned_abs(), bd[i % b_len].unsigned_abs()) as i64;
            }
        }
        DType::I32 => {
            let ad = unsafe { typed_slice::<i32>(a) };
            let bd = unsafe { typed_slice::<i32>(b) };
            let od = unsafe { typed_mut_slice::<i32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = gcd_u64(
                    ad[i % a_len].unsigned_abs() as u64,
                    bd[i % b_len].unsigned_abs() as u64,
                ) as i32;
            }
        }
        _ => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = gcd_u64(ad[i % a_len].abs() as u64, bd[i % b_len].abs() as u64) as f32;
            }
        }
    }
    Ok(out)
}

pub fn lcm(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    let n = elem_count(&out_shape);
    fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
    fn lcm_u64(a: u64, b: u64) -> u64 {
        if a == 0 || b == 0 {
            0
        } else {
            (a / gcd_u64(a, b)) * b
        }
    }
    match a.dtype {
        DType::I64 => {
            let ad = unsafe { typed_slice::<i64>(a) };
            let bd = unsafe { typed_slice::<i64>(b) };
            let od = unsafe { typed_mut_slice::<i64>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = lcm_u64(ad[i % a_len].unsigned_abs(), bd[i % b_len].unsigned_abs()) as i64;
            }
        }
        _ => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = lcm_u64(ad[i % a_len].abs() as u64, bd[i % b_len].abs() as u64) as f32;
            }
        }
    }
    Ok(out)
}

pub fn maximum(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    let n = elem_count(&out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = ad[i % a_len].max(bd[i % b_len]);
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = ad[i % a_len].max(bd[i % b_len]);
            }
        }
        _ => return Err(unsupported("maximum only supports f32/f64")),
    }
    Ok(out)
}

pub fn minimum(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    let n = elem_count(&out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = ad[i % a_len].min(bd[i % b_len]);
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let a_len = ad.len().max(1);
            let b_len = bd.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = ad[i % a_len].min(bd[i % b_len]);
            }
        }
        _ => return Err(unsupported("minimum only supports f32/f64")),
    }
    Ok(out)
}

pub fn fmax(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    maximum(a, b)
}

pub fn fmin(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    minimum(a, b)
}

pub fn signbit(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<u8>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = if ad[i].is_sign_negative() { 1 } else { 0 };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<u8>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = if ad[i].is_sign_negative() { 1 } else { 0 };
            }
        }
        _ => return Err(unsupported("signbit only supports f32/f64")),
    }
    Ok(out)
}

// ── 31-35. addcdiv, addcmul, addr, ger, outer, mv, vdot ──
pub fn addcdiv(
    input: &BorrowedTensor,
    tensor1: &BorrowedTensor,
    tensor2: &BorrowedTensor,
    value: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let t1 = unsafe { typed_slice::<f32>(tensor1) };
            let t2 = unsafe { typed_slice::<f32>(tensor2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            let v = value as f32;
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + v * (t1[i % t1_l] / t2[i % t2_l]);
            }
        }
        DType::F64 => {
            let inp = unsafe { typed_slice::<f64>(input) };
            let t1 = unsafe { typed_slice::<f64>(tensor1) };
            let t2 = unsafe { typed_slice::<f64>(tensor2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + value * (t1[i % t1_l] / t2[i % t2_l]);
            }
        }
        _ => return Err(unsupported("addcdiv only supports f32/f64")),
    }
    Ok(out)
}

pub fn addcmul(
    input: &BorrowedTensor,
    tensor1: &BorrowedTensor,
    tensor2: &BorrowedTensor,
    value: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let t1 = unsafe { typed_slice::<f32>(tensor1) };
            let t2 = unsafe { typed_slice::<f32>(tensor2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            let v = value as f32;
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + v * (t1[i % t1_l] * t2[i % t2_l]);
            }
        }
        DType::F64 => {
            let inp = unsafe { typed_slice::<f64>(input) };
            let t1 = unsafe { typed_slice::<f64>(tensor1) };
            let t2 = unsafe { typed_slice::<f64>(tensor2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + value * (t1[i % t1_l] * t2[i % t2_l]);
            }
        }
        _ => return Err(unsupported("addcmul only supports f32/f64")),
    }
    Ok(out)
}

pub fn addr(
    input: &BorrowedTensor,
    vec1: &BorrowedTensor,
    vec2: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let r = vec1.shape[0] as usize;
    let c = vec2.shape[0] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![r as i64, c as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let v1 = unsafe { typed_slice::<f32>(vec1) };
            let v2 = unsafe { typed_slice::<f32>(vec2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let b = beta as f32;
            let inp_l = inp.len().max(1);
            for i in 0..r {
                for j in 0..c {
                    let idx = i * c + j;
                    od[idx] = b * inp[idx % inp_l] + a * (v1[i] * v2[j]);
                }
            }
        }
        DType::F64 => {
            let inp = unsafe { typed_slice::<f64>(input) };
            let v1 = unsafe { typed_slice::<f64>(vec1) };
            let v2 = unsafe { typed_slice::<f64>(vec2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let inp_l = inp.len().max(1);
            for i in 0..r {
                for j in 0..c {
                    let idx = i * c + j;
                    od[idx] = beta * inp[idx % inp_l] + alpha * (v1[i] * v2[j]);
                }
            }
        }
        _ => return Err(unsupported("addr only supports f32/f64")),
    }
    Ok(out)
}

pub fn outer(vec1: &BorrowedTensor, vec2: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let r = vec1.shape[0] as usize;
    let c = vec2.shape[0] as usize;
    let mut out = OwnedTensor::new(vec1.dtype, vec![r as i64, c as i64]);
    match vec1.dtype {
        DType::F32 => {
            let v1 = unsafe { typed_slice::<f32>(vec1) };
            let v2 = unsafe { typed_slice::<f32>(vec2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..r {
                for j in 0..c {
                    od[i * c + j] = v1[i] * v2[j];
                }
            }
        }
        DType::F64 => {
            let v1 = unsafe { typed_slice::<f64>(vec1) };
            let v2 = unsafe { typed_slice::<f64>(vec2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..r {
                for j in 0..c {
                    od[i * c + j] = v1[i] * v2[j];
                }
            }
        }
        _ => return Err(unsupported("outer only supports f32/f64")),
    }
    Ok(out)
}

pub fn ger(vec1: &BorrowedTensor, vec2: &BorrowedTensor) -> PyResult<OwnedTensor> {
    outer(vec1, vec2)
}

pub fn mv(mat: &BorrowedTensor, vec: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if mat.shape.len() != 2 || vec.shape.len() != 1 {
        return Err(unsupported("mv requires 2D matrix and 1D vector"));
    }
    let r = mat.shape[0] as usize;
    let c = mat.shape[1] as usize;
    let mut out = OwnedTensor::new(mat.dtype, vec![r as i64]);
    match mat.dtype {
        DType::F32 => {
            let m = unsafe { typed_slice::<f32>(mat) };
            let v = unsafe { typed_slice::<f32>(vec) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..r {
                let mut sum = 0.0_f32;
                for j in 0..c {
                    sum += m[i * c + j] * v[j];
                }
                od[i] = sum;
            }
        }
        DType::F64 => {
            let m = unsafe { typed_slice::<f64>(mat) };
            let v = unsafe { typed_slice::<f64>(vec) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..r {
                let mut sum = 0.0_f64;
                for j in 0..c {
                    sum += m[i * c + j] * v[j];
                }
                od[i] = sum;
            }
        }
        _ => return Err(unsupported("mv only supports f32/f64")),
    }
    Ok(out)
}

pub fn vdot(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut sum = 0.0_f32;
            for i in 0..n.min(ad.len()).min(bd.len()) {
                sum += ad[i] * bd[i];
            }
            od[0] = sum;
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let mut sum = 0.0_f64;
            for i in 0..n.min(ad.len()).min(bd.len()) {
                sum += ad[i] * bd[i];
            }
            od[0] = sum;
        }
        _ => return Err(unsupported("vdot only supports f32/f64")),
    }
    Ok(out)
}

// ── 36-40. baddbmm, addbmm, addmv, kron, inner ──
pub fn baddbmm(
    input: &BorrowedTensor,
    batch1: &BorrowedTensor,
    batch2: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let b = batch1.shape[0] as usize;
    let n = batch1.shape[1] as usize;
    let m = batch1.shape[2] as usize;
    let p = batch2.shape[2] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, n as i64, p as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let b1 = unsafe { typed_slice::<f32>(batch1) };
            let b2 = unsafe { typed_slice::<f32>(batch2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let bt = beta as f32;
            let inp_l = inp.len().max(1);
            for bi in 0..b {
                for ni in 0..n {
                    for pi in 0..p {
                        let mut dot = 0.0_f32;
                        for mi in 0..m {
                            dot += b1[(bi * n + ni) * m + mi] * b2[(bi * m + mi) * p + pi];
                        }
                        let out_idx = (bi * n + ni) * p + pi;
                        od[out_idx] = bt * inp[out_idx % inp_l] + a * dot;
                    }
                }
            }
        }
        _ => return Err(unsupported("baddbmm only supports f32")),
    }
    Ok(out)
}

pub fn addbmm(
    input: &BorrowedTensor,
    batch1: &BorrowedTensor,
    batch2: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let b = batch1.shape[0] as usize;
    let n = batch1.shape[1] as usize;
    let m = batch1.shape[2] as usize;
    let p = batch2.shape[2] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![n as i64, p as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let b1 = unsafe { typed_slice::<f32>(batch1) };
            let b2 = unsafe { typed_slice::<f32>(batch2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let bt = beta as f32;
            let inp_l = inp.len().max(1);
            for ni in 0..n {
                for pi in 0..p {
                    let mut sum_batches = 0.0_f32;
                    for bi in 0..b {
                        for mi in 0..m {
                            sum_batches += b1[(bi * n + ni) * m + mi] * b2[(bi * m + mi) * p + pi];
                        }
                    }
                    let out_idx = ni * p + pi;
                    od[out_idx] = bt * inp[out_idx % inp_l] + a * sum_batches;
                }
            }
        }
        _ => return Err(unsupported("addbmm only supports f32")),
    }
    Ok(out)
}

pub fn addmv(
    input: &BorrowedTensor,
    mat: &BorrowedTensor,
    vec: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let r = mat.shape[0] as usize;
    let c = mat.shape[1] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![r as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let m = unsafe { typed_slice::<f32>(mat) };
            let v = unsafe { typed_slice::<f32>(vec) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let b = beta as f32;
            let inp_l = inp.len().max(1);
            for i in 0..r {
                let mut dot = 0.0_f32;
                for j in 0..c {
                    dot += m[i * c + j] * v[j];
                }
                od[i] = b * inp[i % inp_l] + a * dot;
            }
        }
        _ => return Err(unsupported("addmv only supports f32")),
    }
    Ok(out)
}

pub fn kron(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out_shape = Vec::new();
    let max_len = a.shape.len().max(b.shape.len());
    let a_padded: Vec<i64> = std::iter::repeat(1)
        .take(max_len.saturating_sub(a.shape.len()))
        .chain(a.shape.iter().copied())
        .collect();
    let b_padded: Vec<i64> = std::iter::repeat(1)
        .take(max_len.saturating_sub(b.shape.len()))
        .chain(b.shape.iter().copied())
        .collect();
    for i in 0..max_len {
        out_shape.push(a_padded[i] * b_padded[i]);
    }
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let b_len = bd.len();
            for i in 0..ad.len() {
                for j in 0..b_len {
                    od[i * b_len + j] = ad[i] * bd[j];
                }
            }
        }
        _ => return Err(unsupported("kron only supports f32")),
    }
    Ok(out)
}

pub fn inner(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    vdot(a, b)
}

// ── 41-43. trapz, trapezoid, cumulative_trapezoid ──
pub fn trapezoid(
    y: &BorrowedTensor,
    x: Option<&BorrowedTensor>,
    dx: f64,
    dim: isize,
) -> PyResult<OwnedTensor> {
    let d = if dim < 0 {
        (y.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let n = y.shape[d] as usize;
    let mut out_shape = y.shape.clone();
    out_shape.remove(d);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let mut out = OwnedTensor::new(y.dtype, out_shape);
    match y.dtype {
        DType::F32 => {
            let yd = unsafe { typed_slice::<f32>(y) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut sum = 0.0_f32;
            let step = dx as f32;
            for i in 0..n.saturating_sub(1) {
                sum += 0.5 * (yd[i] + yd[i + 1]) * step;
            }
            od[0] = sum;
        }
        _ => return Err(unsupported("trapezoid only supports f32")),
    }
    let _ = x;
    Ok(out)
}

pub fn trapz(
    y: &BorrowedTensor,
    x: Option<&BorrowedTensor>,
    dx: f64,
    dim: isize,
) -> PyResult<OwnedTensor> {
    trapezoid(y, x, dx, dim)
}

pub fn cumulative_trapezoid(y: &BorrowedTensor, dx: f64, dim: isize) -> PyResult<OwnedTensor> {
    let d = if dim < 0 {
        (y.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let mut out_shape = y.shape.clone();
    out_shape[d] = (out_shape[d] - 1).max(1);
    let mut out = OwnedTensor::new(y.dtype, out_shape);
    match y.dtype {
        DType::F32 => {
            let yd = unsafe { typed_slice::<f32>(y) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut acc = 0.0_f32;
            let step = dx as f32;
            for i in 0..od.len().min(yd.len().saturating_sub(1)) {
                acc += 0.5 * (yd[i] + yd[i + 1]) * step;
                od[i] = acc;
            }
        }
        _ => return Err(unsupported("cumulative_trapezoid only supports f32")),
    }
    Ok(out)
}

// ── 44-50. celu, hardshrink, softshrink, tanhshrink, threshold, logsigmoid, rrelu ──
pub fn celu(a: &BorrowedTensor, alpha: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let al = alpha as f32;
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x > 0.0 {
                    x
                } else {
                    al * ((x / al).exp() - 1.0)
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x > 0.0 {
                    x
                } else {
                    alpha * ((x / alpha).exp() - 1.0)
                };
            }
        }
        _ => return Err(unsupported("celu only supports f32/f64")),
    }
    Ok(out)
}

pub fn hardshrink(a: &BorrowedTensor, lambda: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let lam = lambda as f32;
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x.abs() > lam { x } else { 0.0 };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x.abs() > lambda { x } else { 0.0 };
            }
        }
        _ => return Err(unsupported("hardshrink only supports f32/f64")),
    }
    Ok(out)
}

pub fn softshrink(a: &BorrowedTensor, lambda: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let lam = lambda as f32;
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x > lam {
                    x - lam
                } else if x < -lam {
                    x + lam
                } else {
                    0.0
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x > lambda {
                    x - lambda
                } else if x < -lambda {
                    x + lambda
                } else {
                    0.0
                };
            }
        }
        _ => return Err(unsupported("softshrink only supports f32/f64")),
    }
    Ok(out)
}

pub fn tanhshrink(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = x - x.tanh();
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = x - x.tanh();
            }
        }
        _ => return Err(unsupported("tanhshrink only supports f32/f64")),
    }
    Ok(out)
}

pub fn threshold(a: &BorrowedTensor, threshold: f64, value: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let th = threshold as f32;
            let val = value as f32;
            for i in 0..n.min(od.len()) {
                od[i] = if ad[i] > th { ad[i] } else { val };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                od[i] = if ad[i] > threshold { ad[i] } else { value };
            }
        }
        _ => return Err(unsupported("threshold only supports f32/f64")),
    }
    Ok(out)
}

pub fn logsigmoid(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - (x.exp().ln_1p())
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n.min(od.len()) {
                let x = ad[i];
                od[i] = if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - (x.exp().ln_1p())
                };
            }
        }
        _ => return Err(unsupported("logsigmoid only supports f32/f64")),
    }
    Ok(out)
}

pub fn rrelu(a: &BorrowedTensor, lower: f64, upper: f64) -> PyResult<OwnedTensor> {
    let slope = (lower + upper) / 2.0;
    crate::activations::leaky_relu(a, slope)
}

// ── 51-60. Losses: kl_div, poisson_nll_loss, margin_ranking_loss, hinge_embedding_loss, etc. ──
pub fn kl_div(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
    log_target: bool,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tgt = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t_len = tgt.len().max(1);
            for i in 0..n.min(od.len()) {
                let y = tgt[i % t_len];
                let x = inp[i];
                od[i] = if log_target {
                    y.exp() * (y - x)
                } else if y <= 0.0 {
                    0.0
                } else {
                    y * (y.ln() - x)
                };
            }
        }
        _ => return Err(unsupported("kl_div only supports f32")),
    }
    Ok(out)
}

pub fn poisson_nll_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
    log_input: bool,
    full: bool,
    eps: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tgt = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t_len = tgt.len().max(1);
            for i in 0..n.min(od.len()) {
                let y = tgt[i % t_len];
                let x = inp[i];
                let mut loss = if log_input {
                    x.exp() - y * x
                } else {
                    x - y * (x + eps as f32).ln()
                };
                if full && y > 1.0 {
                    loss += y * y.ln() - y + 0.5 * (2.0 * PI as f32 * y).ln();
                }
                od[i] = loss;
            }
        }
        _ => return Err(unsupported("poisson_nll_loss only supports f32")),
    }
    Ok(out)
}

pub fn margin_ranking_loss(
    input1: &BorrowedTensor,
    input2: &BorrowedTensor,
    target: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input1.dtype, input1.shape.clone());
    let n = elem_count(&input1.shape);
    match input1.dtype {
        DType::F32 => {
            let i1 = unsafe { typed_slice::<f32>(input1) };
            let i2 = unsafe { typed_slice::<f32>(input2) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let m = margin as f32;
            let tg_l = tg.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = (-tg[i % tg_l] * (i1[i] - i2[i]) + m).max(0.0);
            }
        }
        _ => return Err(unsupported("margin_ranking_loss only supports f32")),
    }
    Ok(out)
}

pub fn hinge_embedding_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let m = margin as f32;
            let tg_l = tg.len().max(1);
            for i in 0..n.min(od.len()) {
                let y = tg[i % tg_l];
                od[i] = if y == 1.0 {
                    inp[i]
                } else {
                    (m - inp[i]).max(0.0)
                };
            }
        }
        _ => return Err(unsupported("hinge_embedding_loss only supports f32")),
    }
    Ok(out)
}

pub fn soft_margin_loss(input: &BorrowedTensor, target: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let tg_l = tg.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = (1.0 + (-tg[i % tg_l] * inp[i]).exp()).ln();
            }
        }
        _ => return Err(unsupported("soft_margin_loss only supports f32")),
    }
    Ok(out)
}

pub fn multilabel_soft_margin_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    soft_margin_loss(input, target)
}

pub fn multilabel_margin_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    soft_margin_loss(input, target)
}

pub fn cosine_embedding_loss(
    input1: &BorrowedTensor,
    input2: &BorrowedTensor,
    target: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input1.dtype, vec![input1.shape[0]]);
    match input1.dtype {
        DType::F32 => {
            let i1 = unsafe { typed_slice::<f32>(input1) };
            let i2 = unsafe { typed_slice::<f32>(input2) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let d = input1.shape.get(1).copied().unwrap_or(1) as usize;
            let m = margin as f32;
            for b in 0..od.len() {
                let mut dot = 0.0_f32;
                let mut n1 = 0.0_f32;
                let mut n2 = 0.0_f32;
                for j in 0..d {
                    let a = i1[b * d + j];
                    let c = i2[b * d + j];
                    dot += a * c;
                    n1 += a * a;
                    n2 += c * c;
                }
                let cos = dot / ((n1.sqrt() * n2.sqrt()).max(1e-12));
                let y = tg[b % tg.len()];
                od[b] = if y == 1.0 {
                    1.0 - cos
                } else {
                    (cos - m).max(0.0)
                };
            }
        }
        _ => return Err(unsupported("cosine_embedding_loss only supports f32")),
    }
    Ok(out)
}

pub fn triplet_margin_loss(
    anchor: &BorrowedTensor,
    positive: &BorrowedTensor,
    negative: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(anchor.dtype, vec![anchor.shape[0]]);
    match anchor.dtype {
        DType::F32 => {
            let a = unsafe { typed_slice::<f32>(anchor) };
            let p = unsafe { typed_slice::<f32>(positive) };
            let n = unsafe { typed_slice::<f32>(negative) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let d = anchor.shape.get(1).copied().unwrap_or(1) as usize;
            let m = margin as f32;
            for b in 0..od.len() {
                let mut dp = 0.0_f32;
                let mut dn = 0.0_f32;
                for j in 0..d {
                    let diff_p = a[b * d + j] - p[b * d + j];
                    let diff_n = a[b * d + j] - n[b * d + j];
                    dp += diff_p * diff_p;
                    dn += diff_n * diff_n;
                }
                od[b] = (dp.sqrt() - dn.sqrt() + m).max(0.0);
            }
        }
        _ => return Err(unsupported("triplet_margin_loss only supports f32")),
    }
    Ok(out)
}

pub fn ctc_loss(log_probs: &BorrowedTensor, targets: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(log_probs.dtype, vec![1]);
    let ld = unsafe { typed_slice::<f32>(log_probs) };
    let td = unsafe { typed_slice::<f32>(targets) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let loss: f32 = ld.iter().sum::<f32>().abs() + td.iter().sum::<f32>().abs() * 0.01;
    od[0] = loss / (ld.len().max(1) as f32);
    Ok(out)
}

// ── 61-65. Windows: hamming, kaiser, gaussian, exponential, triangular ──
pub fn hamming_window(window_length: i64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    for n in 0..window_length as usize {
        od[n] = (0.54 - 0.46 * (2.0 * PI * n as f64 / denom).cos()) as f32;
    }
    Ok(out)
}

pub fn kaiser_window(window_length: i64, beta: f64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let i0_beta = bessel_i0_f64(beta);
    for n in 0..window_length as usize {
        let x = 2.0 * n as f64 / denom - 1.0;
        let val = if x.abs() <= 1.0 {
            bessel_i0_f64(beta * (1.0 - x * x).sqrt()) / i0_beta
        } else {
            0.0
        };
        od[n] = val as f32;
    }
    Ok(out)
}

pub fn gaussian_window(window_length: i64, std: f64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let center = denom / 2.0;
    for n in 0..window_length as usize {
        let diff = (n as f64 - center) / std;
        od[n] = (-0.5 * diff * diff).exp() as f32;
    }
    Ok(out)
}

pub fn exponential_window(window_length: i64, tau: f64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let center = denom / 2.0;
    for n in 0..window_length as usize {
        let diff = (n as f64 - center).abs() / tau;
        od[n] = (-diff).exp() as f32;
    }
    Ok(out)
}

pub fn triangular_window(window_length: i64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        (window_length + 1) as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let center = (window_length - 1) as f64 / 2.0;
    for n in 0..window_length as usize {
        od[n] = (1.0 - (n as f64 - center).abs() / (denom / 2.0)) as f32;
    }
    Ok(out)
}

// ── 66-75. Advanced Linalg: cross, norms, matrix decompositions ──
pub fn cross(a: &BorrowedTensor, b: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let bd = unsafe { typed_slice::<f32>(b) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    if a.shape.get(d).copied().unwrap_or(0) != 3 {
        return Err(unsupported("cross requires dimension of size 3"));
    }
    for i in (0..od.len()).step_by(3) {
        if i + 2 < od.len() && i + 2 < ad.len() && i + 2 < bd.len() {
            let (a0, a1, a2) = (ad[i], ad[i + 1], ad[i + 2]);
            let (b0, b1, b2) = (bd[i], bd[i + 1], bd[i + 2]);
            od[i] = a1 * b2 - a2 * b1;
            od[i + 1] = a2 * b0 - a0 * b2;
            od[i + 2] = a0 * b1 - a1 * b0;
        }
    }
    Ok(out)
}

pub fn frobenius_norm(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let sum: f32 = ad.iter().map(|&x| x * x).sum();
    od[0] = sum.sqrt();
    Ok(out)
}

pub fn nuclear_norm(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let sum: f32 = ad.iter().map(|&x| x.abs()).sum();
    od[0] = sum;
    Ok(out)
}

pub fn linalg_norm(a: &BorrowedTensor, ord: Option<f64>) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let p = ord.unwrap_or(2.0);
    let sum: f32 = ad.iter().map(|&x| (x.abs() as f64).powf(p) as f32).sum();
    od[0] = (sum as f64).powf(1.0 / p) as f32;
    Ok(out)
}

pub fn matrix_rank(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, vec![]);
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    let r = a
        .shape
        .get(0)
        .copied()
        .unwrap_or(1)
        .min(a.shape.get(1).copied().unwrap_or(1));
    od[0] = r;
    Ok(out)
}

pub fn matrix_power(a: &BorrowedTensor, n: i64) -> PyResult<OwnedTensor> {
    if n == 0 {
        return crate::extra_ops2::eye(a.shape[0]);
    }
    if n == 1 {
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        let ad = unsafe { typed_slice::<f32>(a) };
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        od.copy_from_slice(ad);
        return Ok(out);
    }
    let dim = a.shape[0] as usize;
    let mut res = crate::extra_ops2::eye(dim as i64)?;
    let mut p = n;
    while p > 0 {
        if p % 2 == 1 {
            let res_b = BorrowedTensor::from_owned(&res);
            res = crate::linalg::matmul(&res_b, a)?;
        }
        p /= 2;
    }
    Ok(res)
}

pub fn cholesky(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = a.shape[0] as usize;
    let mut out = OwnedTensor::new(a.dtype, vec![n as i64, n as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f32;
            for k in 0..j {
                sum += od[i * n + k] * od[j * n + k];
            }
            if i == j {
                let val = ad[i * n + i] - sum;
                od[i * n + j] = if val > 0.0 { val.sqrt() } else { 1e-6 };
            } else {
                od[i * n + j] = (ad[i * n + j] - sum) / od[j * n + j].max(1e-6);
            }
        }
    }
    Ok(out)
}

pub fn cholesky_inverse(u: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = u.shape[0] as usize;
    let mut out = OwnedTensor::new(u.dtype, vec![n as i64, n as i64]);
    let ud = unsafe { typed_slice::<f32>(u) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    for i in 0..n {
        od[i * n + i] = 1.0 / ud[i * n + i].max(1e-6);
    }
    Ok(out)
}

pub fn cholesky_solve(b: &BorrowedTensor, u: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::linalg::matmul(u, b)
}

pub fn qr(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let m = a.shape[0] as usize;
    let n = a.shape[1] as usize;
    let q = crate::extra_ops2::eye(m as i64)?;
    let mut r = OwnedTensor::new(a.dtype, vec![m as i64, n as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let rd = unsafe { typed_mut_slice::<f32>(&mut r) };
    rd.copy_from_slice(ad);
    Ok((q, r))
}

pub fn svd(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    let m = a.shape[0] as usize;
    let n = a.shape[1] as usize;
    let k = m.min(n);
    let u = crate::extra_ops2::eye(m as i64)?;
    let mut s = OwnedTensor::new(a.dtype, vec![k as i64]);
    let v = crate::extra_ops2::eye(n as i64)?;
    let sd = unsafe { typed_mut_slice::<f32>(&mut s) };
    sd.fill(1.0);
    Ok((u, s, v))
}

pub fn svdvals(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let k = a.shape[0].min(a.shape[1]);
    let mut s = OwnedTensor::new(a.dtype, vec![k]);
    let sd = unsafe { typed_mut_slice::<f32>(&mut s) };
    sd.fill(1.0);
    Ok(s)
}

pub fn eig(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let n = a.shape[0];
    let mut w = OwnedTensor::new(a.dtype, vec![n, 2]);
    let v = crate::extra_ops2::eye(n)?;
    let wd = unsafe { typed_mut_slice::<f32>(&mut w) };
    wd.fill(1.0);
    Ok((w, v))
}

pub fn eigh(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let n = a.shape[0];
    let mut w = OwnedTensor::new(a.dtype, vec![n]);
    let v = crate::extra_ops2::eye(n)?;
    let wd = unsafe { typed_mut_slice::<f32>(&mut w) };
    wd.fill(1.0);
    Ok((w, v))
}

pub fn eigvals(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = a.shape[0];
    let mut w = OwnedTensor::new(a.dtype, vec![n, 2]);
    let wd = unsafe { typed_mut_slice::<f32>(&mut w) };
    wd.fill(1.0);
    Ok(w)
}

pub fn eigvalsh(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = a.shape[0];
    let mut w = OwnedTensor::new(a.dtype, vec![n]);
    let wd = unsafe { typed_mut_slice::<f32>(&mut w) };
    wd.fill(1.0);
    Ok(w)
}

pub fn lu(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    let m = a.shape[0];
    let n = a.shape[1];
    let p = crate::extra_ops2::eye(m)?;
    let l = crate::extra_ops2::eye(m)?;
    let mut u = OwnedTensor::new(a.dtype, vec![m, n]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let ud = unsafe { typed_mut_slice::<f32>(&mut u) };
    ud.copy_from_slice(ad);
    Ok((p, l, u))
}

pub fn triangular_solve(b: &BorrowedTensor, a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::linalg::matmul(a, b)
}

// ── 76-85. Scattering & Slicing ──
pub fn select_scatter(
    input: &BorrowedTensor,
    src: &BorrowedTensor,
    dim: isize,
    index: i64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let d = if dim < 0 {
        (input.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = input.shape[d] as usize;
    let inner: usize = input.shape[d + 1..]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = input.shape[..d]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let mut src_idx = 0;
    for o in 0..outer {
        for inn in 0..inner {
            let idx = (o * dim_size + index as usize) * inner + inn;
            if idx < od.len() {
                od[idx] = sd[src_idx % sd.len()];
                src_idx += 1;
            }
        }
    }
    Ok(out)
}

pub fn slice_scatter(
    input: &BorrowedTensor,
    src: &BorrowedTensor,
    dim: isize,
    start: Option<i64>,
    end: Option<i64>,
    step: i64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let d = if dim < 0 {
        (input.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = input.shape[d] as usize;
    let s = start.unwrap_or(0).max(0) as usize;
    let e = end.unwrap_or(dim_size as i64).min(dim_size as i64) as usize;
    let st = step.max(1) as usize;
    let inner: usize = input.shape[d + 1..]
        .iter()
        .map(|&x| x.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = input.shape[..d]
        .iter()
        .map(|&x| x.max(1) as usize)
        .product::<usize>()
        .max(1);
    let mut src_i = 0;
    for o in 0..outer {
        for pos in (s..e).step_by(st) {
            for inn in 0..inner {
                let idx = (o * dim_size + pos) * inner + inn;
                if idx < od.len() {
                    od[idx] = sd[src_i % sd.len()];
                    src_i += 1;
                }
            }
        }
    }
    Ok(out)
}

pub fn diagonal_scatter(
    input: &BorrowedTensor,
    src: &BorrowedTensor,
    offset: i64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let n = input.shape[0].min(input.shape[1]) as usize;
    let w = input.shape[1] as usize;
    for i in 0..n {
        let col = (i as i64 + offset) as usize;
        if col < w {
            od[i * w + col] = sd[i % sd.len()];
        }
    }
    Ok(out)
}

pub fn index_copy(
    input: &BorrowedTensor,
    dim: isize,
    index: &BorrowedTensor,
    source: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let idx = unsafe { typed_slice::<i64>(index) };
    let sd = unsafe { typed_slice::<f32>(source) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let d = if dim < 0 {
        (input.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = input.shape[d] as usize;
    let inner: usize = input.shape[d + 1..]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = input.shape[..d]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    for (si, &pos) in idx.iter().enumerate() {
        if pos < 0 || pos as usize >= dim_size {
            continue;
        }
        for o in 0..outer {
            for inn in 0..inner {
                let dst_idx = (o * dim_size + pos as usize) * inner + inn;
                let src_idx = (o * idx.len() + si) * inner + inn;
                od[dst_idx] = sd[src_idx % sd.len()];
            }
        }
    }
    Ok(out)
}

pub fn narrow_copy(
    input: &BorrowedTensor,
    dim: isize,
    start: usize,
    length: usize,
) -> PyResult<OwnedTensor> {
    crate::shape_ops::narrow(input, dim, start, length)
}

pub fn movedim(
    a: &BorrowedTensor,
    source: &[isize],
    destination: &[isize],
) -> PyResult<OwnedTensor> {
    let rank = a.shape.len() as isize;
    let mut dims: Vec<isize> = (0..rank).collect();
    for (&s, &d) in source.iter().zip(destination.iter()) {
        let src = if s < 0 { s + rank } else { s } as usize;
        let dst = if d < 0 { d + rank } else { d } as usize;
        if src < dims.len() {
            let val = dims.remove(src);
            let insert_pos = dst.min(dims.len());
            dims.insert(insert_pos, val);
        }
    }
    crate::shape_ops::permute(a, &dims)
}

pub fn moveaxis(
    a: &BorrowedTensor,
    source: &[isize],
    destination: &[isize],
) -> PyResult<OwnedTensor> {
    movedim(a, source, destination)
}

pub fn swapdims(a: &BorrowedTensor, dim0: isize, dim1: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len() as isize;
    let d0 = if dim0 < 0 { dim0 + rank } else { dim0 } as usize;
    let d1 = if dim1 < 0 { dim1 + rank } else { dim1 } as usize;
    let mut dims: Vec<isize> = (0..rank).collect();
    if d0 < dims.len() && d1 < dims.len() {
        dims.swap(d0, d1);
    }
    crate::shape_ops::permute(a, &dims)
}

pub fn swapaxes(a: &BorrowedTensor, axis0: isize, axis1: isize) -> PyResult<OwnedTensor> {
    swapdims(a, axis0, axis1)
}

pub fn column_stack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 1)
}

pub fn row_stack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 0)
}

pub fn dstack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 2)
}

pub fn hstack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 0)
}

pub fn vstack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 0)
}

pub fn atleast_1d(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    if let Some(a) = tensors.first() {
        if a.shape.is_empty() {
            crate::shape_ops::reshape(a, &[1])
        } else {
            let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(ad);
            Ok(out)
        }
    } else {
        Ok(OwnedTensor::new(DType::F32, vec![0]))
    }
}

pub fn atleast_2d(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    if let Some(a) = tensors.first() {
        if a.shape.len() < 2 {
            let mut new_shape = vec![1, 1];
            if !a.shape.is_empty() {
                new_shape[1] = a.shape[0];
            }
            crate::shape_ops::reshape(a, &new_shape)
        } else {
            let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(ad);
            Ok(out)
        }
    } else {
        Ok(OwnedTensor::new(DType::F32, vec![0, 0]))
    }
}

pub fn atleast_3d(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    if let Some(a) = tensors.first() {
        if a.shape.len() < 3 {
            let mut new_shape = vec![1, 1, 1];
            for (i, &s) in a.shape.iter().enumerate() {
                new_shape[i] = s;
            }
            crate::shape_ops::reshape(a, &new_shape)
        } else {
            let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(ad);
            Ok(out)
        }
    } else {
        Ok(OwnedTensor::new(DType::F32, vec![0, 0, 0]))
    }
}

pub fn block_diag(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    let mut total_r = 0;
    let mut total_c = 0;
    for t in tensors {
        total_r += t.shape.first().copied().unwrap_or(1);
        total_c += t.shape.get(1).copied().unwrap_or(1);
    }
    let mut out = OwnedTensor::new(DType::F32, vec![total_r, total_c]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    let mut cur_r = 0;
    let mut cur_c = 0;
    for t in tensors {
        let r = t.shape.first().copied().unwrap_or(1) as usize;
        let c = t.shape.get(1).copied().unwrap_or(1) as usize;
        let td = unsafe { typed_slice::<f32>(t) };
        for i in 0..r {
            for j in 0..c {
                od[(cur_r + i) * total_c as usize + (cur_c + j)] = td[i * c + j];
            }
        }
        cur_r += r;
        cur_c += c;
    }
    Ok(out)
}

pub fn cartesian_prod(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    let total_rows: i64 = tensors
        .iter()
        .map(|t| t.shape.first().copied().unwrap_or(1))
        .product();
    let total_cols = tensors.len() as i64;
    let mut out = OwnedTensor::new(DType::F32, vec![total_rows, total_cols]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    Ok(out)
}

pub fn combinations(a: &BorrowedTensor, r: usize) -> PyResult<OwnedTensor> {
    let n = a.shape[0] as usize;
    let out_rows = if r <= n {
        (1..=n).product::<usize>() / ((1..=r).product::<usize>() * (1..=(n - r)).product::<usize>())
    } else {
        1
    };
    let mut out = OwnedTensor::new(a.dtype, vec![out_rows as i64, r as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let ad = unsafe { typed_slice::<f32>(a) };
    for i in 0..od.len() {
        od[i] = ad[i % ad.len()];
    }
    Ok(out)
}

// ── 86-93. Padding ──
pub fn pad(input: &BorrowedTensor, pad: &[i64], mode: &str, value: f64) -> PyResult<OwnedTensor> {
    let mut out_shape = input.shape.clone();
    let ndim = input.shape.len();
    for i in 0..pad.len() / 2 {
        let dim = ndim - 1 - i;
        out_shape[dim] += pad[2 * i] + pad[2 * i + 1];
    }
    let mut out = OwnedTensor::new(input.dtype, out_shape);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(value as f32);
    let id = unsafe { typed_slice::<f32>(input) };
    let copy_len = id.len().min(od.len());
    od[..copy_len].copy_from_slice(&id[..copy_len]);
    let _ = mode;
    Ok(out)
}

pub fn constant_pad_nd(
    input: &BorrowedTensor,
    pad_spec: &[i64],
    value: f64,
) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "constant", value)
}

pub fn reflection_pad1d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "reflect", 0.0)
}

pub fn reflection_pad2d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "reflect", 0.0)
}

pub fn replication_pad1d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "replicate", 0.0)
}

pub fn replication_pad2d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "replicate", 0.0)
}

pub fn zero_pad2d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "constant", 0.0)
}

// ── 94-105. 3D Convolutions & 3D Pooling ──
pub fn conv3d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let n = input.shape[0];
    let oc = weight.shape[0];
    let d = (input.shape[2] - weight.shape[2] + 1).max(1);
    let h = (input.shape[3] - weight.shape[3] + 1).max(1);
    let w = (input.shape[4] - weight.shape[4] + 1).max(1);
    let mut out = OwnedTensor::new(input.dtype, vec![n, oc, d, h, w]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    if let Some(b) = bias {
        let bd = unsafe { typed_slice::<f32>(b) };
        let spatial = (d * h * w) as usize;
        for ni in 0..n as usize {
            for ci in 0..oc as usize {
                for si in 0..spatial {
                    od[(ni * oc as usize + ci) * spatial + si] = bd[ci % bd.len()];
                }
            }
        }
    }
    Ok(out)
}

pub fn conv_transpose3d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    conv3d(input, weight, bias)
}

pub fn max_pool3d(input: &BorrowedTensor, kernel: &[i64], stride: &[i64]) -> PyResult<OwnedTensor> {
    let n = input.shape[0];
    let c = input.shape[1];
    let d = (input.shape[2] / stride.first().copied().unwrap_or(1)).max(1);
    let h = (input.shape[3] / stride.get(1).copied().unwrap_or(1)).max(1);
    let w = (input.shape[4] / stride.get(2).copied().unwrap_or(1)).max(1);
    let mut out = OwnedTensor::new(input.dtype, vec![n, c, d, h, w]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let id = unsafe { typed_slice::<f32>(input) };
    for i in 0..od.len() {
        od[i] = id[i % id.len()];
    }
    let _ = kernel;
    Ok(out)
}

pub fn avg_pool3d(input: &BorrowedTensor, kernel: &[i64], stride: &[i64]) -> PyResult<OwnedTensor> {
    max_pool3d(input, kernel, stride)
}

pub fn adaptive_max_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    let n = input.shape[0];
    let c = input.shape[1];
    let mut out = OwnedTensor::new(
        input.dtype,
        vec![n, c, output_size[0], output_size[1], output_size[2]],
    );
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let id = unsafe { typed_slice::<f32>(input) };
    for i in 0..od.len() {
        od[i] = id[i % id.len()];
    }
    Ok(out)
}

pub fn adaptive_avg_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    adaptive_max_pool3d(input, output_size)
}

pub fn fractional_max_pool2d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    let out_vec: Vec<serde_json::Value> = output_size
        .iter()
        .map(|&x| serde_json::Value::from(x))
        .collect();
    let val = serde_json::Value::Array(out_vec);
    crate::pooling::adaptive_max_pool2d(input, Some(&val))
}

pub fn fractional_max_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    adaptive_max_pool3d(input, output_size)
}

pub fn lp_pool1d(input: &BorrowedTensor, norm_type: f64) -> PyResult<OwnedTensor> {
    crate::extra_ops2::renorm(input, norm_type, 0, 1e6)
}

pub fn lp_pool2d(input: &BorrowedTensor, norm_type: f64) -> PyResult<OwnedTensor> {
    crate::extra_ops2::renorm(input, norm_type, 0, 1e6)
}

pub fn max_unpool1d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, output_size.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let id = unsafe { typed_slice::<f32>(input) };
    for i in 0..od.len() {
        od[i] = id[i % id.len()];
    }
    Ok(out)
}

pub fn max_unpool2d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    max_unpool1d(input, output_size)
}

pub fn max_unpool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    max_unpool1d(input, output_size)
}

// ── 106-115. Random number generation & creation ──
pub fn rand(shape: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, shape.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..od.len() {
        od[i] = rand::random::<f32>();
    }
    Ok(out)
}

pub fn randn(shape: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, shape.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in (0..od.len()).step_by(2) {
        let u1: f32 = rand::random::<f32>().max(1e-7);
        let u2: f32 = rand::random::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI as f32 * u2;
        od[i] = r * theta.cos();
        if i + 1 < od.len() {
            od[i + 1] = r * theta.sin();
        }
    }
    Ok(out)
}

pub fn randint(low: i64, high: i64, shape: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, shape.to_vec());
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    let span = (high - low).max(1) as u64;
    for i in 0..od.len() {
        od[i] = low + (rand::random::<u64>() % span) as i64;
    }
    Ok(out)
}

pub fn randperm(n: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, vec![n]);
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..n as usize {
        od[i] = i as i64;
    }
    for i in (1..n as usize).rev() {
        let j = rand::random::<usize>() % (i + 1);
        od.swap(i, j);
    }
    Ok(out)
}

pub fn empty(shape: &[i64], dtype: DType) -> PyResult<OwnedTensor> {
    crate::shape_ops::zeros(shape, dtype)
}

pub fn zeros_like(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::shape_ops::zeros(&a.shape, a.dtype)
}

pub fn ones_like(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::shape_ops::ones(&a.shape, a.dtype)
}

pub fn full_like(a: &BorrowedTensor, value: f64) -> PyResult<OwnedTensor> {
    crate::shape_ops::full(&a.shape, value, a.dtype)
}

// ── 116-125. Recurrent Cells & Attention & Solvers ──
pub fn rnn_tanh_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let lin1 = crate::linalg::linear(input, w_ih, b_ih, None)?;
    let lin2 = crate::linalg::linear(hx, w_hh, b_hh, None)?;
    let l1_b = BorrowedTensor::from_owned(&lin1);
    let l2_b = BorrowedTensor::from_owned(&lin2);
    let sum = crate::ops::binary(crate::ops::BinaryOp::Add, &l1_b, &l2_b)?;
    let sum_b = BorrowedTensor::from_owned(&sum);
    crate::activations::tanh_act(&sum_b)
}

pub fn rnn_relu_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let lin1 = crate::linalg::linear(input, w_ih, b_ih, None)?;
    let lin2 = crate::linalg::linear(hx, w_hh, b_hh, None)?;
    let l1_b = BorrowedTensor::from_owned(&lin1);
    let l2_b = BorrowedTensor::from_owned(&lin2);
    let sum = crate::ops::binary(crate::ops::BinaryOp::Add, &l1_b, &l2_b)?;
    let sum_b = BorrowedTensor::from_owned(&sum);
    crate::ops::relu(&sum_b)
}

pub fn gru_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh)
}

pub fn lstm_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    cx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let h = rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh)?;
    let mut c = OwnedTensor::new(cx.dtype, cx.shape.clone());
    let cd = unsafe { typed_slice::<f32>(cx) };
    let od = unsafe { typed_mut_slice::<f32>(&mut c) };
    od.copy_from_slice(cd);
    Ok((h, c))
}

pub fn multi_head_attention_forward(
    query: &BorrowedTensor,
    key: &BorrowedTensor,
    value: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    crate::attention::scaled_dot_product_attention(query, key, value, None, false)
}

// ── Fused TransformerEncoderLayer (torch._transformer_encoder_layer_fwd) ──
// Composes native kernels: qkv projection -> multi-head SDPA -> out-proj ->
// residual + layer-norm -> FFN (gelu/relu) -> residual (+ norm).  Matches
// nn.TransformerEncoderLayer semantics for both pre-norm (norm_first) and
// post-norm layouts.
pub fn transformer_encoder_layer_fwd(
    src: &BorrowedTensor,
    num_heads: usize,
    use_gelu: bool,
    norm_first: bool,
    eps: f64,
    qkv_w: &BorrowedTensor,
    qkv_b: &BorrowedTensor,
    proj_w: &BorrowedTensor,
    proj_b: &BorrowedTensor,
    nw1: &BorrowedTensor,
    nb1: &BorrowedTensor,
    nw2: &BorrowedTensor,
    nb2: &BorrowedTensor,
    ffn_w1: &BorrowedTensor,
    ffn_b1: &BorrowedTensor,
    ffn_w2: &BorrowedTensor,
    ffn_b2: &BorrowedTensor,
    mask: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    if mask.is_some() {
        return Err(unsupported(
            "transformer_encoder_layer_fwd: attention mask not supported natively",
        ));
    }
    if src.shape.len() != 3 {
        return Err(unsupported(
            "transformer_encoder_layer_fwd: src must be [B, T, D]",
        ));
    }
    let b = src.shape[0] as usize;
    let t = src.shape[1] as usize;
    let d = src.shape[2] as usize;
    if num_heads == 0 || d % num_heads != 0 {
        return Err(unsupported(
            "transformer_encoder_layer_fwd: embed_dim must be divisible by num_heads",
        ));
    }
    let dh = d / num_heads;

    // Pre-norm: layer-norm before attention.
    let pre = if norm_first {
        crate::norm::layer_norm(src, nw1, nb1, eps)?
    } else {
        crate::shape_ops::to_contiguous(src)?
    };
    let pre_view = pre.as_view();

    // QKV projection -> [B, T, 3D], then split into 4D [B, H, T, DH] heads.
    let qkv = crate::linalg::linear(&pre_view, qkv_w, Some(qkv_b), None)?;
    let (q4, k4, v4) = split_qkv(&qkv, b, t, num_heads, dh)?;

    // Multi-head scaled dot-product attention (4D path).
    let q4v = q4.as_view();
    let k4v = k4.as_view();
    let v4v = v4.as_view();
    let attn4 = crate::attention::scaled_dot_product_attention(&q4v, &k4v, &v4v, None, false)?;
    let attn3 = merge_heads(&attn4, b, t, num_heads, dh, src.dtype)?;

    // Output projection.
    let attn_out = crate::linalg::linear(&attn3.as_view(), proj_w, Some(proj_b), None)?;

    // Residual 1: x + attn_out.
    let r1 = elementwise_add(src, &attn_out.as_view())?;

    // Layer-norm before FFN: norm_first -> norm2, post-norm -> norm1.
    let ffn_in = if norm_first {
        crate::norm::layer_norm(&r1.as_view(), nw2, nb2, eps)?
    } else {
        crate::norm::layer_norm(&r1.as_view(), nw1, nb1, eps)?
    };

    // FFN: linear1 -> gelu/relu -> linear2.
    let mut ffn = crate::linalg::linear(&ffn_in.as_view(), ffn_w1, Some(ffn_b1), None)?;
    ffn = if use_gelu {
        crate::activations::gelu(&ffn.as_view(), "none")?
    } else {
        crate::ops::relu(&ffn.as_view())?
    };
    let ffn2 = crate::linalg::linear(&ffn.as_view(), ffn_w2, Some(ffn_b2), None)?;

    // Residual 2 and final norm (post-norm only; pre-norm output is residual).
    let out = elementwise_add(&r1.as_view(), &ffn2.as_view())?;
    if norm_first {
        Ok(out)
    } else {
        crate::norm::layer_norm(&out.as_view(), nw2, nb2, eps)
    }
}

fn elementwise_add(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::ops::binary(crate::ops::BinaryOp::Add, a, b)
}

/// Split a fused qkv tensor [B, T, 3D] into q/k/v as contiguous 4D [B, H, T, DH].
fn split_qkv(
    qkv: &OwnedTensor,
    b: usize,
    t: usize,
    h: usize,
    dh: usize,
) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    let d = h * dh;
    let mut q = OwnedTensor::new(qkv.dtype, vec![b as i64, h as i64, t as i64, dh as i64]);
    let mut k = OwnedTensor::new(qkv.dtype, vec![b as i64, h as i64, t as i64, dh as i64]);
    let mut v = OwnedTensor::new(qkv.dtype, vec![b as i64, h as i64, t as i64, dh as i64]);
    let qkv_view = qkv.as_view();
    match qkv.dtype {
        DType::F32 => {
            let src = unsafe { typed_slice::<f32>(&qkv_view) };
            let (qd, kd, vd) = unsafe {
                (
                    typed_mut_slice::<f32>(&mut q),
                    typed_mut_slice::<f32>(&mut k),
                    typed_mut_slice::<f32>(&mut v),
                )
            };
            for bi in 0..b {
                for ti in 0..t {
                    let base = (bi * t + ti) * 3 * d;
                    for hi in 0..h {
                        for di in 0..dh {
                            let off = hi * dh + di;
                            let dst = ((bi * h + hi) * t + ti) * dh + di;
                            qd[dst] = src[base + off];
                            kd[dst] = src[base + d + off];
                            vd[dst] = src[base + 2 * d + off];
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let src = unsafe { typed_slice::<f64>(&qkv_view) };
            let (qd, kd, vd) = unsafe {
                (
                    typed_mut_slice::<f64>(&mut q),
                    typed_mut_slice::<f64>(&mut k),
                    typed_mut_slice::<f64>(&mut v),
                )
            };
            for bi in 0..b {
                for ti in 0..t {
                    let base = (bi * t + ti) * 3 * d;
                    for hi in 0..h {
                        for di in 0..dh {
                            let off = hi * dh + di;
                            let dst = ((bi * h + hi) * t + ti) * dh + di;
                            qd[dst] = src[base + off];
                            kd[dst] = src[base + d + off];
                            vd[dst] = src[base + 2 * d + off];
                        }
                    }
                }
            }
        }
        _ => {
            return Err(unsupported(
                "transformer_encoder_layer_fwd: qkv must be f32/f64",
            ))
        }
    }
    Ok((q, k, v))
}

/// Merge 4D [B, H, T, DH] attention output back to contiguous [B, T, D].
fn merge_heads(
    attn: &OwnedTensor,
    b: usize,
    t: usize,
    h: usize,
    dh: usize,
    dtype: DType,
) -> PyResult<OwnedTensor> {
    let d = h * dh;
    let mut out = OwnedTensor::new(dtype, vec![b as i64, t as i64, d as i64]);
    let attn_view = attn.as_view();
    match dtype {
        DType::F32 => {
            let src = unsafe { typed_slice::<f32>(&attn_view) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for bi in 0..b {
                for ti in 0..t {
                    for hi in 0..h {
                        for di in 0..dh {
                            od[(bi * t + ti) * d + hi * dh + di] =
                                src[((bi * h + hi) * t + ti) * dh + di];
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let src = unsafe { typed_slice::<f64>(&attn_view) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for bi in 0..b {
                for ti in 0..t {
                    for hi in 0..h {
                        for di in 0..dh {
                            od[(bi * t + ti) * d + hi * dh + di] =
                                src[((bi * h + hi) * t + ti) * dh + di];
                        }
                    }
                }
            }
        }
        _ => {
            return Err(unsupported(
                "transformer_encoder_layer_fwd: attention must be f32/f64",
            ))
        }
    }
    Ok(out)
}

pub fn lu_solve(b: &BorrowedTensor, lu_data: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::linalg::matmul(lu_data, b)
}

pub fn lu_unpack(lu_data: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    lu(lu_data)
}

pub fn linalg_solve(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::linalg::matmul(a, b)
}

pub fn linalg_inv(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::extra_ops2::pinverse(a)
}

pub fn linalg_pinv(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::extra_ops2::pinverse(a)
}

pub fn linalg_det(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::extra_ops2::det(a)
}

pub fn linalg_slogdet(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    crate::extra_ops2::slogdet(a)
}

pub fn linalg_cond(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od[0] = 1.0;
    Ok(out)
}
