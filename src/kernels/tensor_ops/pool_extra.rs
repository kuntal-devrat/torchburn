//! 1D pooling: local_response_norm, adaptive avg/max pool1d, lp_pool3d. Inherits tensor_ops root imports via super; pure move.

use super::*;

pub fn local_response_norm(
    a: &BorrowedTensor,
    size: usize,
    alpha: f64,
    beta: f64,
    k: f64,
) -> PyResult<OwnedTensor> {
    // input assumed NCHW
    if a.shape.len() != 4 {
        return Err(unsupported("lrn requires 4D"));
    }
    let n = a.shape[0] as usize;
    let c = a.shape[1] as usize;
    let h = a.shape[2] as usize;
    let w = a.shape[3] as usize;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let alpha_f = alpha as f32;
            let beta_f = beta as f32;
            let k_f = k as f32;
            for nn in 0..n {
                for cc in 0..c {
                    for hh in 0..h {
                        for ww in 0..w {
                            let mut sum = 0.0f32;
                            let start = (cc as isize - size as isize / 2).max(0) as usize;
                            let end = (cc + size / 2 + 1).min(c);
                            for ci in start..end {
                                let idx = ((nn * c + ci) * h + hh) * w + ww;
                                sum += ad[idx] * ad[idx];
                            }
                            let idx = ((nn * c + cc) * h + hh) * w + ww;
                            od[idx] = ad[idx] / (k_f + alpha_f * sum).powf(beta_f);
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for nn in 0..n {
                for cc in 0..c {
                    for hh in 0..h {
                        for ww in 0..w {
                            let mut sum = 0.0;
                            let start = (cc as isize - size as isize / 2).max(0) as usize;
                            let end = (cc + size / 2 + 1).min(c);
                            for ci in start..end {
                                let idx = ((nn * c + ci) * h + hh) * w + ww;
                                sum += ad[idx] * ad[idx];
                            }
                            let idx = ((nn * c + cc) * h + hh) * w + ww;
                            od[idx] = ad[idx] / (k + alpha * sum).powf(beta);
                        }
                    }
                }
            }
        }
        _ => return Err(unsupported("lrn only f32/f64")),
    }
    Ok(out)
}
pub fn adaptive_avg_pool1d(a: &BorrowedTensor, out_sz: usize) -> PyResult<OwnedTensor> {
    // NCL -> N C Lout average
    if a.shape.len() != 3 {
        return Err(unsupported("adaptive_avg_pool1d requires 3D"));
    }
    let n = a.shape[0] as usize;
    let c = a.shape[1] as usize;
    let l = a.shape[2] as usize;
    let mut out = OwnedTensor::new(a.dtype, vec![n as i64, c as i64, out_sz as i64]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for nn in 0..n {
                for cc in 0..c {
                    for o in 0..out_sz {
                        let start = (o * l) / out_sz;
                        let end = ((o + 1) * l) / out_sz;
                        let mut s = 0.0;
                        for k in start..end {
                            s += ad[(nn * c + cc) * l + k];
                        }
                        od[(nn * c + cc) * out_sz + o] = if end > start {
                            s / ((end - start) as f32)
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for nn in 0..n {
                for cc in 0..c {
                    for o in 0..out_sz {
                        let start = (o * l) / out_sz;
                        let end = ((o + 1) * l) / out_sz;
                        let mut s = 0.0;
                        for k in start..end {
                            s += ad[(nn * c + cc) * l + k];
                        }
                        od[(nn * c + cc) * out_sz + o] = if end > start {
                            s / ((end - start) as f64)
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        _ => return Err(unsupported("adaptive_avg_pool1d only f32/f64")),
    }
    Ok(out)
}
pub fn adaptive_max_pool1d(a: &BorrowedTensor, out_sz: usize) -> PyResult<OwnedTensor> {
    if a.shape.len() != 3 {
        return Err(unsupported("adaptive_max_pool1d requires 3D"));
    }
    let n = a.shape[0] as usize;
    let c = a.shape[1] as usize;
    let l = a.shape[2] as usize;
    let mut out = OwnedTensor::new(a.dtype, vec![n as i64, c as i64, out_sz as i64]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for nn in 0..n {
                for cc in 0..c {
                    for o in 0..out_sz {
                        let start = (o * l) / out_sz;
                        let end = ((o + 1) * l) / out_sz;
                        let mut m = f32::NEG_INFINITY;
                        for k in start..end {
                            let v = ad[(nn * c + cc) * l + k];
                            if v > m {
                                m = v;
                            }
                        }
                        od[(nn * c + cc) * out_sz + o] = m;
                    }
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for nn in 0..n {
                for cc in 0..c {
                    for o in 0..out_sz {
                        let start = (o * l) / out_sz;
                        let end = ((o + 1) * l) / out_sz;
                        let mut m = f64::NEG_INFINITY;
                        for k in start..end {
                            let v = ad[(nn * c + cc) * l + k];
                            if v > m {
                                m = v;
                            }
                        }
                        od[(nn * c + cc) * out_sz + o] = m;
                    }
                }
            }
        }
        _ => return Err(unsupported("adaptive_max_pool1d only f32/f64")),
    }
    Ok(out)
}
pub fn lp_pool3d(
    a: &BorrowedTensor,
    p: f64,
    kernel: usize,
    stride: usize,
) -> PyResult<OwnedTensor> {
    // naive 3D pooling: input NCDHW
    if a.shape.len() != 5 {
        return Err(unsupported("lp_pool3d requires 5D"));
    }
    let n = a.shape[0] as usize;
    let c = a.shape[1] as usize;
    let d = a.shape[2] as usize;
    let h = a.shape[3] as usize;
    let w = a.shape[4] as usize;
    let od = (d - kernel) / stride + 1;
    let oh = (h - kernel) / stride + 1;
    let ow = (w - kernel) / stride + 1;
    let mut out = OwnedTensor::new(
        a.dtype,
        vec![n as i64, c as i64, od as i64, oh as i64, ow as i64],
    );
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let odat = unsafe { typed_mut_slice::<f32>(&mut out) };
            for nn in 0..n {
                for cc in 0..c {
                    for odz in 0..od {
                        for ohh in 0..oh {
                            for oww in 0..ow {
                                let mut s = 0.0;
                                for kz in 0..kernel {
                                    for ky in 0..kernel {
                                        for kx in 0..kernel {
                                            let iz = odz * stride + kz;
                                            let iy = ohh * stride + ky;
                                            let ix = oww * stride + kx;
                                            let val =
                                                ad[(((nn * c + cc) * d + iz) * h + iy) * w + ix];
                                            s += val.abs().powf(p as f32);
                                        }
                                    }
                                }
                                let out_idx = (((nn * c + cc) * od + odz) * oh + ohh) * ow + oww;
                                odat[out_idx] = s.powf(1.0 / p as f32);
                            }
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let odat = unsafe { typed_mut_slice::<f64>(&mut out) };
            for nn in 0..n {
                for cc in 0..c {
                    for odz in 0..od {
                        for ohh in 0..oh {
                            for oww in 0..ow {
                                let mut s = 0.0;
                                for kz in 0..kernel {
                                    for ky in 0..kernel {
                                        for kx in 0..kernel {
                                            let iz = odz * stride + kz;
                                            let iy = ohh * stride + ky;
                                            let ix = oww * stride + kx;
                                            let val =
                                                ad[(((nn * c + cc) * d + iz) * h + iy) * w + ix];
                                            s += val.abs().powf(p);
                                        }
                                    }
                                }
                                let out_idx = (((nn * c + cc) * od + odz) * oh + ohh) * ow + oww;
                                odat[out_idx] = s.powf(1.0 / p);
                            }
                        }
                    }
                }
            }
        }
        _ => return Err(unsupported("lp_pool3d only f32/f64")),
    }
    Ok(out)
}
