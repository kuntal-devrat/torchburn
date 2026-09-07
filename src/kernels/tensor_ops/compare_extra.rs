//! Comparisons and predicates: isclose, allclose, equal, isreal, is_complex, is_nonzero. Inherits tensor_ops root imports via super; pure move.

use super::*;

// 1. isclose
pub fn isclose(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<OwnedTensor> {
    let out_shape = crate::kernels::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(DType::Bool, out_shape.clone());
    let n = elem_count(&out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<u8>(&mut out) };
            let al = ad.len().max(1);
            let bl = bd.len().max(1);
            let rt = rtol as f32;
            let at = atol as f32;
            for i in 0..n {
                let av = ad[i % al];
                let bv = bd[i % bl];
                let close = if av.is_nan() || bv.is_nan() {
                    equal_nan && av.is_nan() && bv.is_nan()
                } else if av.is_infinite() || bv.is_infinite() {
                    av == bv
                } else {
                    (av - bv).abs() <= at + rt * bv.abs()
                };
                od[i] = if close { 1 } else { 0 };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<u8>(&mut out) };
            let al = ad.len().max(1);
            let bl = bd.len().max(1);
            for i in 0..n {
                let av = ad[i % al];
                let bv = bd[i % bl];
                let close = if av.is_nan() || bv.is_nan() {
                    equal_nan && av.is_nan() && bv.is_nan()
                } else if av.is_infinite() || bv.is_infinite() {
                    av == bv
                } else {
                    (av - bv).abs() <= atol + rtol * bv.abs()
                };
                od[i] = if close { 1 } else { 0 };
            }
        }
        _ => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<u8>(&mut out) };
            let al = ad.len().max(1);
            let bl = bd.len().max(1);
            for i in 0..n {
                od[i] = if ad[i % al] == bd[i % bl] { 1 } else { 0 };
            }
        }
    }
    Ok(out)
}

// 2. allclose -> scalar bool
pub fn allclose(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<OwnedTensor> {
    let tmp = isclose(a, b, rtol, atol, equal_nan)?;
    let n = elem_count(&tmp.shape);
    let data = unsafe { std::slice::from_raw_parts(tmp.data.as_ptr() as *const u8, n) };
    let all = data.iter().all(|&v| v != 0);
    let mut out = OwnedTensor::new(DType::Bool, vec![]);
    let od = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut u8, 1) };
    od[0] = if all { 1 } else { 0 };
    Ok(out)
}

// 3. equal -> scalar bool exact
pub fn equal(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.shape != b.shape {
        let mut out = OwnedTensor::new(DType::Bool, vec![]);
        let od = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut u8, 1) };
        od[0] = 0;
        return Ok(out);
    }
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(DType::Bool, vec![]);
    let mut is_eq = true;
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            for i in 0..n {
                if ad[i] != bd[i] {
                    is_eq = false;
                    break;
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            for i in 0..n {
                if ad[i] != bd[i] {
                    is_eq = false;
                    break;
                }
            }
        }
        DType::I64 => {
            let ad = unsafe { typed_slice::<i64>(a) };
            let bd = unsafe { typed_slice::<i64>(b) };
            for i in 0..n {
                if ad[i] != bd[i] {
                    is_eq = false;
                    break;
                }
            }
        }
        DType::I32 => {
            let ad = unsafe { typed_slice::<i32>(a) };
            let bd = unsafe { typed_slice::<i32>(b) };
            for i in 0..n {
                if ad[i] != bd[i] {
                    is_eq = false;
                    break;
                }
            }
        }
        DType::Bool => {
            let ad = unsafe { typed_slice::<u8>(a) };
            let bd = unsafe { typed_slice::<u8>(b) };
            for i in 0..n {
                if ad[i] != bd[i] {
                    is_eq = false;
                    break;
                }
            }
        }
    }
    let od = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut u8, 1) };
    od[0] = if is_eq { 1 } else { 0 };
    Ok(out)
}

// 4. isreal -> bool tensor all true for real dtypes
pub fn isreal(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, a.shape.clone());
    let n = elem_count(&a.shape);
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    for i in 0..n {
        od[i] = 1;
    }
    Ok(out)
}
pub fn is_complex(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, a.shape.clone());
    let n = elem_count(&a.shape);
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    for i in 0..n {
        od[i] = 0;
    }
    Ok(out)
}
pub fn is_nonzero(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut has = false;
    match a.dtype {
        DType::F32 => {
            let d = unsafe { typed_slice::<f32>(a) };
            for &v in d.iter().take(n) {
                if v != 0.0 {
                    has = true;
                    break;
                }
            }
        }
        DType::F64 => {
            let d = unsafe { typed_slice::<f64>(a) };
            for &v in d.iter().take(n) {
                if v != 0.0 {
                    has = true;
                    break;
                }
            }
        }
        DType::I64 => {
            let d = unsafe { typed_slice::<i64>(a) };
            for &v in d.iter().take(n) {
                if v != 0 {
                    has = true;
                    break;
                }
            }
        }
        DType::I32 => {
            let d = unsafe { typed_slice::<i32>(a) };
            for &v in d.iter().take(n) {
                if v != 0 {
                    has = true;
                    break;
                }
            }
        }
        DType::Bool => {
            let d = unsafe { typed_slice::<u8>(a) };
            for &v in d.iter().take(n) {
                if v != 0 {
                    has = true;
                    break;
                }
            }
        }
    }
    let mut out = OwnedTensor::new(DType::Bool, vec![]);
    let od = unsafe { typed_mut_slice::<u8>(&mut out) };
    od[0] = if has { 1 } else { 0 };
    Ok(out)
}
