//! Extra activations: celu, hardshrink/softshrink, tanhshrink, threshold, logsigmoid, rrelu. Inherits linalg root imports via super; pure move.

use super::*;

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
    crate::nn::activations::leaky_relu(a, slope)
}
