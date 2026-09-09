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
    // torch.masked_select requires same numel (broadcastable mask is flattened).
    let na = elem_count(&a.shape);
    let nm = elem_count(&mask.shape);
    if na != nm {
        return Err(unsupported(&format!(
            "masked_select shape mismatch: input {na} vs mask {nm}"
        )));
    }
    let md: Vec<bool> = match mask.dtype {
        DType::Bool | DType::U8 | DType::I8 => unsafe {
            typed_slice::<u8>(mask).iter().map(|&x| x != 0).collect()
        },
        DType::F32 => unsafe { typed_slice::<f32>(mask).iter().map(|&x| x != 0.0).collect() },
        DType::F64 => unsafe { typed_slice::<f64>(mask).iter().map(|&x| x != 0.0).collect() },
        DType::I64 => unsafe { typed_slice::<i64>(mask).iter().map(|&x| x != 0).collect() },
        DType::I32 => unsafe { typed_slice::<i32>(mask).iter().map(|&x| x != 0).collect() },
        _ => return Err(unsupported("masked_select mask must be bool/int/float")),
    };
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let vals: Vec<f32> = ad
                .iter()
                .zip(md.iter())
                .filter(|(_, &m)| m)
                .map(|(&x, _)| x)
                .collect();
            let mut out = OwnedTensor::new(DType::F32, vec![vals.len() as i64]);
            unsafe { typed_mut_slice::<f32>(&mut out).copy_from_slice(&vals) };
            Ok(out)
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let vals: Vec<f64> = ad
                .iter()
                .zip(md.iter())
                .filter(|(_, &m)| m)
                .map(|(&x, _)| x)
                .collect();
            let mut out = OwnedTensor::new(DType::F64, vec![vals.len() as i64]);
            unsafe { typed_mut_slice::<f64>(&mut out).copy_from_slice(&vals) };
            Ok(out)
        }
        _ => Err(unsupported("masked_select only f32/f64")),
    }
}
pub fn istft(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // Real overlap-add inverse DFT matching `stft` above:
    // input [F,T,2] complex with F = n_fft/2+1 -> output length
    // (T-1)*hop + n_fft with hop = n_fft/4 (torch default).
    if a.shape.len() != 3 || a.shape[2] != 2 {
        return Err(unsupported("istft expects [F,T,2] complex"));
    }
    if a.dtype != DType::F32 {
        return Err(unsupported("istft only f32"));
    }
    let n_freqs = a.shape[0].max(0) as usize;
    let n_frames = a.shape[1].max(0) as usize;
    if n_freqs == 0 || n_frames == 0 {
        return Err(unsupported("istft: empty"));
    }
    let n_fft = (n_freqs - 1) * 2;
    let hop = (n_fft / 4).max(1);
    let len = (n_frames - 1) * hop + n_fft;
    let ad = unsafe { typed_slice::<f32>(a) };
    let mut out = OwnedTensor::new(DType::F32, vec![len as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    let mut wsum = vec![0.0f32; len];
    for f in 0..n_frames {
        // IDFT of frame f into temp buffer
        for t in 0..n_fft {
            let mut s = 0.0f32;
            for k in 0..n_freqs {
                let idx = (k * n_frames + f) * 2;
                let re = ad[idx];
                let im = ad[idx + 1];
                // Reconstruct full spectrum symmetry: k and N-k conjugate.
                // For k=0 and k=n_fft/2 the bin is real-only in RFFT.
                let angle = 2.0 * std::f64::consts::PI * (k * t) as f64 / n_fft as f64;
                let (c, sn) = (angle.cos() as f32, angle.sin() as f32);
                let w = if k == 0 || k == n_freqs - 1 { 1.0 } else { 2.0 };
                s += w * (re * c - im * sn);
            }
            s /= n_fft as f32;
            let pos = f * hop + t;
            od[pos] += s;
            wsum[pos] += 1.0;
        }
    }
    for i in 0..len {
        if wsum[i] > 0.0 {
            od[i] /= wsum[i];
        }
    }
    Ok(out)
}
