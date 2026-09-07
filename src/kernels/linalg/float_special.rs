//! Float special functions: nextafter, bessel, digamma, erf/erfc inverses, normal CDF helpers, logit/expit. Inherits linalg root imports via super; pure move.

use super::*;

// ── 1. nextafter ──
pub fn nextafter(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::kernels::broadcast_shape(&a.shape, &b.shape)?;
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
    let out_shape = crate::kernels::broadcast_shape(&a.shape, &b.shape)?;
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
    let out_shape = crate::kernels::broadcast_shape(&a.shape, &b.shape)?;
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
pub(crate) fn bessel_i0_f64(x: f64) -> f64 {
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
    crate::nn::activations::sigmoid(a)
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
