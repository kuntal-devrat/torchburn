//! Integer and elementwise extensions: gcd/lcm, maximum/minimum, fmax/fmin, signbit. Inherits linalg root imports via super; pure move.

use super::*;

// ── 26-30. gcd, lcm, fmax, fmin, maximum, minimum, signbit ──
pub fn gcd(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::kernels::broadcast_shape(&a.shape, &b.shape)?;
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
    let out_shape = crate::kernels::broadcast_shape(&a.shape, &b.shape)?;
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
