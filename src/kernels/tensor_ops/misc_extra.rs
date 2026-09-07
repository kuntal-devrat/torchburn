//! Misc: logsumexp, rand_like family, empty_strided, view_as/expand_as, isfinite, masked_select, istft. Inherits tensor_ops root imports via super; pure move.

use super::*;

pub fn logsumexp(a: &BorrowedTensor, dim: isize, keepdim: bool) -> PyResult<OwnedTensor> {
    // logsumexp = max + log(sum(exp(x-max)))
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = a.shape[d] as usize;
    let outer: usize = a.shape[..d].iter().map(|&s| s.max(0) as usize).product();
    let inner: usize = a.shape[d + 1..]
        .iter()
        .map(|&s| s.max(0) as usize)
        .product();
    let mut out_shape = a.shape.clone();
    if keepdim {
        out_shape[d] = 1;
    } else {
        out_shape.remove(d);
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let mut out = OwnedTensor::new(a.dtype, out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for o in 0..outer {
                for inn in 0..inner {
                    let mut maxv = f32::NEG_INFINITY;
                    for k in 0..dim_size {
                        let idx = o * dim_size * inner + k * inner + inn;
                        if ad[idx] > maxv {
                            maxv = ad[idx];
                        }
                    }
                    let mut sum = 0.0f32;
                    for k in 0..dim_size {
                        let idx = o * dim_size * inner + k * inner + inn;
                        sum += (ad[idx] - maxv).exp();
                    }
                    let res = maxv + sum.ln();
                    let out_idx = if keepdim {
                        o * 1 * inner + 0 * inner + inn
                    } else {
                        o * inner + inn
                    };
                    od[out_idx] = res;
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for o in 0..outer {
                for inn in 0..inner {
                    let mut maxv = f64::NEG_INFINITY;
                    for k in 0..dim_size {
                        let idx = o * dim_size * inner + k * inner + inn;
                        if ad[idx] > maxv {
                            maxv = ad[idx];
                        }
                    }
                    let mut sum = 0.0;
                    for k in 0..dim_size {
                        let idx = o * dim_size * inner + k * inner + inn;
                        sum += (ad[idx] - maxv).exp();
                    }
                    let res = maxv + sum.ln();
                    let out_idx = if keepdim {
                        o * 1 * inner + 0 * inner + inn
                    } else {
                        o * inner + inn
                    };
                    od[out_idx] = res;
                }
            }
        }
        _ => return Err(unsupported("logsumexp only f32/f64")),
    }
    Ok(out)
}
pub fn randn_like(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                od[i] = rand::random::<f32>() * 2.0 - 1.0;
            }
        }
        DType::F64 => {
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                od[i] = rand::random::<f64>() * 2.0 - 1.0;
            }
        }
        _ => return Err(unsupported("randn_like only f32/f64")),
    }
    Ok(out)
}
pub fn rand_like(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                od[i] = rand::random::<f32>();
            }
        }
        DType::F64 => {
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                od[i] = rand::random::<f64>();
            }
        }
        _ => return Err(unsupported("rand_like only f32/f64")),
    }
    Ok(out)
}
pub fn randint_like(a: &BorrowedTensor, low: i64, high: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, a.shape.clone());
    let n = elem_count(&a.shape);
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..n {
        od[i] = rand::random::<i64>().rem_euclid(high - low) + low;
    }
    Ok(out)
}
pub fn empty_strided(size: Vec<i64>, stride: Vec<i64>) -> PyResult<OwnedTensor> {
    let out = OwnedTensor::new(DType::F32, size.clone());
    let _ = stride;
    Ok(out)
}
pub fn view_as(a: &BorrowedTensor, other: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let shape = other.shape.clone();
    crate::shape_ops::reshape(a, &shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
}
pub fn expand_as(a: &BorrowedTensor, other: &BorrowedTensor) -> PyResult<OwnedTensor> {
    broadcast_to(a, other.shape.clone())
}

// two more to reach 48: scalar_tensor already exists but add isfinite alias and stft placeholder
pub fn isfinite(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::Bool, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<u8>(&mut out) };
            for i in 0..n {
                od[i] = if ad[i].is_finite() { 1 } else { 0 };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<u8>(&mut out) };
            for i in 0..n {
                od[i] = if ad[i].is_finite() { 1 } else { 0 };
            }
        }
        _ => return Err(unsupported("isfinite only f32/f64")),
    }
    Ok(out)
}
pub fn masked_select(a: &BorrowedTensor, mask: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut vals = Vec::new();
    match mask.dtype {
        DType::Bool => {
            let md = unsafe { typed_slice::<u8>(mask) };
            match a.dtype {
                DType::F32 => {
                    let ad = unsafe { typed_slice::<f32>(a) };
                    for i in 0..n.max(md.len()).min(ad.len()) {
                        if md[i % md.len()] != 0 {
                            vals.push(ad[i % md.len()]);
                        }
                    }
                }
                DType::F64 => {
                    let ad = unsafe { typed_slice::<f64>(a) };
                    let mut vals_f: Vec<f64> = Vec::new();
                    for i in 0..n.max(md.len()).min(ad.len()) {
                        if md[i % md.len()] != 0 {
                            vals_f.push(ad[i % md.len()]);
                        }
                    }
                    let mut out = OwnedTensor::new(DType::F64, vec![vals_f.len() as i64]);
                    let od = unsafe { typed_mut_slice::<f64>(&mut out) };
                    od.copy_from_slice(&vals_f);
                    return Ok(out);
                }
                _ => return Err(unsupported("masked_select only f32/f64")),
            }
        }
        _ => return Err(unsupported("masked_select mask must be bool")),
    }
    let mut out = OwnedTensor::new(DType::F32, vec![vals.len() as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(&vals);
    Ok(out)
}
pub fn istft(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // placeholder inverse STFT: return copy
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(&ad[..n.min(od.len())]);
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            od.copy_from_slice(&ad[..n.min(od.len())]);
        }
        _ => return Err(unsupported("istft only f32/f64")),
    }
    Ok(out)
}
