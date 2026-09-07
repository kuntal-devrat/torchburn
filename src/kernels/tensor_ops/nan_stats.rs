//! NaN-aware stats: nanprod/nanmin/nanmax, var/std_mean, nanmedian, cov, corrcoef. Inherits tensor_ops root imports via super; pure move.

use super::*;

// 7. nanprod
pub fn nanprod(a: &BorrowedTensor, dim: Option<isize>, keepdim: bool) -> PyResult<OwnedTensor> {
    // if dim None, prod over all ignoring NaN; else reduce along dim ignoring NaN
    if dim.is_none() {
        let n = elem_count(&a.shape);
        let mut out = OwnedTensor::new(
            a.dtype,
            if keepdim {
                a.shape.clone().iter().map(|_| 1).collect()
            } else {
                vec![]
            },
        );
        match a.dtype {
            DType::F32 => {
                let ad = unsafe { typed_slice::<f32>(a) };
                let od = unsafe { typed_mut_slice::<f32>(&mut out) };
                let mut prod = 1.0f32;
                let mut has = false;
                for i in 0..n {
                    let v = ad[i];
                    if !v.is_nan() {
                        prod *= v;
                        has = true;
                    }
                }
                od[0] = if has { prod } else { 1.0 };
            }
            DType::F64 => {
                let ad = unsafe { typed_slice::<f64>(a) };
                let od = unsafe { typed_mut_slice::<f64>(&mut out) };
                let mut prod = 1.0f64;
                let mut has = false;
                for i in 0..n {
                    let v = ad[i];
                    if !v.is_nan() {
                        prod *= v;
                        has = true;
                    }
                }
                od[0] = if has { prod } else { 1.0 };
            }
            _ => return Err(unsupported("nanprod only f32/f64")),
        }
        return Ok(out);
    }
    crate::reductions::prod(a, dim, keepdim)
}
// 8. nanmin
pub fn nanmin(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut m = f32::INFINITY;
            let mut has = false;
            for i in 0..n {
                let v = ad[i];
                if !v.is_nan() {
                    if !has || v < m {
                        m = v;
                    }
                    has = true;
                }
            }
            od[0] = if has { m } else { f32::NAN };
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let mut m = f64::INFINITY;
            let mut has = false;
            for i in 0..n {
                let v = ad[i];
                if !v.is_nan() {
                    if !has || v < m {
                        m = v;
                    }
                    has = true;
                }
            }
            od[0] = if has { m } else { f64::NAN };
        }
        _ => return Err(unsupported("nanmin only f32/f64")),
    }
    Ok(out)
}
pub fn nanmax(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut m = f32::NEG_INFINITY;
            let mut has = false;
            for i in 0..n {
                let v = ad[i];
                if !v.is_nan() {
                    if !has || v > m {
                        m = v;
                    }
                    has = true;
                }
            }
            od[0] = if has { m } else { f32::NAN };
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let mut m = f64::NEG_INFINITY;
            let mut has = false;
            for i in 0..n {
                let v = ad[i];
                if !v.is_nan() {
                    if !has || v > m {
                        m = v;
                    }
                    has = true;
                }
            }
            od[0] = if has { m } else { f64::NAN };
        }
        _ => return Err(unsupported("nanmax only f32/f64")),
    }
    Ok(out)
}
// var_mean and std_mean returning tuple (var, mean)
pub fn var_mean(
    a: &BorrowedTensor,
    dim: Option<isize>,
    keepdim: bool,
    unbiased: bool,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let var = crate::reductions::var(a, dim, keepdim, unbiased)?;
    let mean = crate::reductions::mean(a, dim, keepdim)?;
    Ok((var, mean))
}
pub fn std_mean(
    a: &BorrowedTensor,
    dim: Option<isize>,
    keepdim: bool,
    unbiased: bool,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let std = crate::reductions::std_dev(a, dim, keepdim, unbiased)?;
    let mean = crate::reductions::mean(a, dim, keepdim)?;
    Ok((std, mean))
}
pub fn nanmedian(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let mut vals: Vec<f32> = ad.iter().take(n).filter(|v| !v.is_nan()).copied().collect();
            vals.sort_by(|x, y| x.total_cmp(y));
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            if vals.is_empty() {
                od[0] = f32::NAN;
            } else {
                let mid = vals.len() / 2;
                od[0] = if vals.len() % 2 == 1 {
                    vals[mid]
                } else {
                    0.5 * (vals[mid - 1] + vals[mid])
                };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let mut vals: Vec<f64> = ad.iter().take(n).filter(|v| !v.is_nan()).copied().collect();
            vals.sort_by(|x, y| x.total_cmp(y));
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            if vals.is_empty() {
                od[0] = f64::NAN;
            } else {
                let mid = vals.len() / 2;
                od[0] = if vals.len() % 2 == 1 {
                    vals[mid]
                } else {
                    0.5 * (vals[mid - 1] + vals[mid])
                };
            }
        }
        _ => return Err(unsupported("nanmedian only f32/f64")),
    }
    Ok(out)
}
// cov: input shape (..., n_obs) or (M,N) where M variables, N observations
pub fn cov(a: &BorrowedTensor, correction: i64) -> PyResult<OwnedTensor> {
    // treat a as 2D (M,N); if 1D treat as (1,N)
    let shape = &a.shape;
    let (m, n) = if shape.len() == 1 {
        (1usize, shape[0] as usize)
    } else if shape.len() == 2 {
        (shape[0] as usize, shape[1] as usize)
    } else {
        return Err(unsupported("cov: expected 1D or 2D"));
    };
    let mut out = OwnedTensor::new(a.dtype, vec![m as i64, m as i64]);
    let corr = correction as f64;
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            // compute mean per row
            let mut means = vec![0.0f32; m];
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..n {
                    s += ad[i * n + j];
                }
                means[i] = s / (n as f32);
            }
            let denom = (n as f32 - corr as f32).max(1.0);
            for i in 0..m {
                for j in 0..m {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += (ad[i * n + k] - means[i]) * (ad[j * n + k] - means[j]);
                    }
                    od[i * m + j] = s / denom;
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let mut means = vec![0.0f64; m];
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..n {
                    s += ad[i * n + j];
                }
                means[i] = s / (n as f64);
            }
            let denom = (n as f64 - corr).max(1.0);
            for i in 0..m {
                for j in 0..m {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += (ad[i * n + k] - means[i]) * (ad[j * n + k] - means[j]);
                    }
                    od[i * m + j] = s / denom;
                }
            }
        }
        _ => return Err(unsupported("cov only f32/f64")),
    }
    Ok(out)
}
pub fn corrcoef(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let c = cov(a, 1)?;
    let m = c.shape[0] as usize;
    let mut out = OwnedTensor::new(c.dtype, c.shape.clone());
    match c.dtype {
        DType::F32 => {
            let cd = unsafe {
                std::slice::from_raw_parts(c.data.as_ptr() as *const f32, c.elem_count())
            };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut std = vec![0.0f32; m];
            for i in 0..m {
                std[i] = cd[i * m + i].sqrt().max(1e-12);
            }
            for i in 0..m {
                for j in 0..m {
                    od[i * m + j] = cd[i * m + j] / (std[i] * std[j]);
                }
            }
        }
        DType::F64 => {
            let cd = unsafe {
                std::slice::from_raw_parts(c.data.as_ptr() as *const f64, c.elem_count())
            };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let mut std = vec![0.0f64; m];
            for i in 0..m {
                std[i] = cd[i * m + i].sqrt().max(1e-12);
            }
            for i in 0..m {
                for j in 0..m {
                    od[i * m + j] = cd[i * m + j] / (std[i] * std[j]);
                }
            }
        }
        _ => return Err(unsupported("corrcoef only f32/f64")),
    }
    Ok(out)
}
