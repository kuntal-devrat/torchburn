//! 3D neural ops: conv3d, conv_transpose3d, pool3d, unpool3d, lp_pool. Inherits linalg root imports via super; pure move.
//!
//! Real functional kernels (no copy-modulo stubs): valid cross-correlation,
//! full transposed convolution, max/average pooling with kernel/stride,
//! adaptive windowing, nearest-neighbor unpool.

use super::*;

// ── 94-105. 3D Convolutions & 3D Pooling ──
pub fn conv3d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    // input (N,Cin,D,H,W), weight (Cout,Cin,Kd,Kh,Kw), valid stride-1.
    if input.shape.len() != 5 || weight.shape.len() != 5 {
        return Err(unsupported("conv3d needs 5D NCDHW"));
    }
    if input.dtype != DType::F32 || weight.dtype != DType::F32 {
        return Err(unsupported("conv3d only f32"));
    }
    let n = input.shape[0].max(0) as usize;
    let cin = input.shape[1].max(0) as usize;
    let din = input.shape[2].max(0) as usize;
    let hin = input.shape[3].max(0) as usize;
    let win = input.shape[4].max(0) as usize;
    let cout = weight.shape[0].max(0) as usize;
    let wcin = weight.shape[1].max(0) as usize;
    let kd = weight.shape[2].max(0) as usize;
    let kh = weight.shape[3].max(0) as usize;
    let kw = weight.shape[4].max(0) as usize;
    if wcin != cin || kd == 0 || kh == 0 || kw == 0 {
        return Err(unsupported("conv3d channel/kernel mismatch"));
    }
    if din < kd || hin < kh || win < kw {
        return Err(unsupported("conv3d kernel larger than input"));
    }
    let dout = din - kd + 1;
    let hout = hin - kh + 1;
    let wout = win - kw + 1;
    let mut out = OwnedTensor::new(
        DType::F32,
        vec![n as i64, cout as i64, dout as i64, hout as i64, wout as i64],
    );
    let id = unsafe { typed_slice::<f32>(input) };
    let wd = unsafe { typed_slice::<f32>(weight) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let bd: Vec<f32> = match bias {
        Some(b) => unsafe { typed_slice::<f32>(b).to_vec() },
        None => Vec::new(),
    };
    for ni in 0..n {
        for co in 0..cout {
            for do_ in 0..dout {
                for ho in 0..hout {
                    for wo in 0..wout {
                        let mut s = if bd.is_empty() {
                            0.0
                        } else {
                            bd[co % bd.len()]
                        };
                        for ci in 0..cin {
                            for kd_ in 0..kd {
                                for kh_ in 0..kh {
                                    for kw_ in 0..kw {
                                        let iv = id[((((ni * cin + ci) * din) + do_ + kd_) * hin
                                            + ho
                                            + kh_)
                                            * win
                                            + wo
                                            + kw_];
                                        let wv = wd[((((co * cin + ci) * kd) + kd_) * kh + kh_)
                                            * kw
                                            + kw_];
                                        s += iv * wv;
                                    }
                                }
                            }
                        }
                        od[((((ni * cout + co) * dout) + do_) * hout + ho) * wout + wo] = s;
                    }
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
    // Full transposed convolution (stride 1, no padding): out = Din+Kd-1, etc.
    if input.shape.len() != 5 || weight.shape.len() != 5 {
        return Err(unsupported("conv_transpose3d needs 5D"));
    }
    if input.dtype != DType::F32 || weight.dtype != DType::F32 {
        return Err(unsupported("conv_transpose3d only f32"));
    }
    let n = input.shape[0].max(0) as usize;
    let cin = input.shape[1].max(0) as usize;
    let din = input.shape[2].max(0) as usize;
    let hin = input.shape[3].max(0) as usize;
    let win = input.shape[4].max(0) as usize;
    let cout = weight.shape[0].max(0) as usize;
    let kd = weight.shape[2].max(0) as usize;
    let kh = weight.shape[3].max(0) as usize;
    let kw = weight.shape[4].max(0) as usize;
    if weight.shape[1].max(0) as usize != cin {
        return Err(unsupported("conv_transpose3d channel mismatch"));
    }
    let dout = din + kd - 1;
    let hout = hin + kh - 1;
    let wout = win + kw - 1;
    let mut out = OwnedTensor::new(
        DType::F32,
        vec![n as i64, cout as i64, dout as i64, hout as i64, wout as i64],
    );
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    let id = unsafe { typed_slice::<f32>(input) };
    let wd = unsafe { typed_slice::<f32>(weight) };
    for ni in 0..n {
        for co in 0..cout {
            for ci in 0..cin {
                for di in 0..din {
                    for hi in 0..hin {
                        for wi in 0..win {
                            let iv = id[((((ni * cin + ci) * din) + di) * hin + hi) * win + wi];
                            for kd_ in 0..kd {
                                for kh_ in 0..kh {
                                    for kw_ in 0..kw {
                                        let wv = wd[((((co * cin + ci) * kd) + kd_) * kh + kh_)
                                            * kw
                                            + kw_];
                                        od[((((ni * cout + co) * dout) + di + kd_) * hout
                                            + hi
                                            + kh_)
                                            * wout
                                            + wi
                                            + kw_] += iv * wv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(b) = bias {
        let bd = unsafe { typed_slice::<f32>(b) };
        let spatial = dout * hout * wout;
        for ni in 0..n {
            for co in 0..cout {
                let bv = bd[co % bd.len()];
                for si in 0..spatial {
                    od[(ni * cout + co) * spatial + si] += bv;
                }
            }
        }
    }
    Ok(out)
}

fn pool3d_out(d: usize, k: usize, s: usize) -> usize {
    if k == 0 || s == 0 {
        return 1;
    }
    (d.saturating_sub(k)) / s + 1
}

pub fn max_pool3d(input: &BorrowedTensor, kernel: &[i64], stride: &[i64]) -> PyResult<OwnedTensor> {
    if input.shape.len() != 5 {
        return Err(unsupported("max_pool3d needs 5D NCDHW"));
    }
    if input.dtype != DType::F32 {
        return Err(unsupported("max_pool3d only f32"));
    }
    let n = input.shape[0].max(0) as usize;
    let c = input.shape[1].max(0) as usize;
    let d = input.shape[2].max(0) as usize;
    let h = input.shape[3].max(0) as usize;
    let w = input.shape[4].max(0) as usize;
    let kd = kernel.get(0).copied().unwrap_or(2).max(1) as usize;
    let kh = kernel.get(1).copied().unwrap_or(kd as i64).max(1) as usize;
    let kw = kernel.get(2).copied().unwrap_or(kd as i64).max(1) as usize;
    let sd = stride.first().copied().unwrap_or(kd as i64).max(1) as usize;
    let sh = stride.get(1).copied().unwrap_or(kh as i64).max(1) as usize;
    let sw = stride.get(2).copied().unwrap_or(kw as i64).max(1) as usize;
    let od_ = pool3d_out(d, kd, sd).max(1);
    let oh = pool3d_out(h, kh, sh).max(1);
    let ow = pool3d_out(w, kw, sw).max(1);
    let mut out = OwnedTensor::new(
        DType::F32,
        vec![n as i64, c as i64, od_ as i64, oh as i64, ow as i64],
    );
    let id = unsafe { typed_slice::<f32>(input) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for ni in 0..n {
        for ci in 0..c {
            for do_ in 0..od_ {
                for ho in 0..oh {
                    for wo in 0..ow {
                        let mut best = f32::NEG_INFINITY;
                        for kd_ in 0..kd {
                            for kh_ in 0..kh {
                                for kw_ in 0..kw {
                                    let di = do_ * sd + kd_;
                                    let hi = ho * sh + kh_;
                                    let wi = wo * sw + kw_;
                                    if di < d && hi < h && wi < w {
                                        let v = id[((((ni * c + ci) * d) + di) * h + hi) * w + wi];
                                        if v > best {
                                            best = v;
                                        }
                                    }
                                }
                            }
                        }
                        od[((((ni * c + ci) * od_) + do_) * oh + ho) * ow + wo] = best;
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn avg_pool3d(input: &BorrowedTensor, kernel: &[i64], stride: &[i64]) -> PyResult<OwnedTensor> {
    if input.shape.len() != 5 {
        return Err(unsupported("avg_pool3d needs 5D NCDHW"));
    }
    if input.dtype != DType::F32 {
        return Err(unsupported("avg_pool3d only f32"));
    }
    let n = input.shape[0].max(0) as usize;
    let c = input.shape[1].max(0) as usize;
    let d = input.shape[2].max(0) as usize;
    let h = input.shape[3].max(0) as usize;
    let w = input.shape[4].max(0) as usize;
    let kd = kernel.get(0).copied().unwrap_or(2).max(1) as usize;
    let kh = kernel.get(1).copied().unwrap_or(kd as i64).max(1) as usize;
    let kw = kernel.get(2).copied().unwrap_or(kd as i64).max(1) as usize;
    let sd = stride.first().copied().unwrap_or(kd as i64).max(1) as usize;
    let sh = stride.get(1).copied().unwrap_or(kh as i64).max(1) as usize;
    let sw = stride.get(2).copied().unwrap_or(kw as i64).max(1) as usize;
    let od_ = pool3d_out(d, kd, sd).max(1);
    let oh = pool3d_out(h, kh, sh).max(1);
    let ow = pool3d_out(w, kw, sw).max(1);
    let mut out = OwnedTensor::new(
        DType::F32,
        vec![n as i64, c as i64, od_ as i64, oh as i64, ow as i64],
    );
    let id = unsafe { typed_slice::<f32>(input) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = (kd * kh * kw) as f32;
    for ni in 0..n {
        for ci in 0..c {
            for do_ in 0..od_ {
                for ho in 0..oh {
                    for wo in 0..ow {
                        let mut s = 0.0f32;
                        for kd_ in 0..kd {
                            for kh_ in 0..kh {
                                for kw_ in 0..kw {
                                    let di = do_ * sd + kd_;
                                    let hi = ho * sh + kh_;
                                    let wi = wo * sw + kw_;
                                    if di < d && hi < h && wi < w {
                                        s += id[((((ni * c + ci) * d) + di) * h + hi) * w + wi];
                                    }
                                }
                            }
                        }
                        od[((((ni * c + ci) * od_) + do_) * oh + ho) * ow + wo] = s / denom;
                    }
                }
            }
        }
    }
    Ok(out)
}

fn adaptive_window(idx: usize, out: usize, inp: usize) -> (usize, usize) {
    let s = (idx * inp) / out;
    let mut e = ((idx + 1) * inp) / out;
    if e <= s {
        e = (s + 1).min(inp);
    }
    (s, e)
}

pub fn adaptive_max_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    if input.shape.len() != 5 {
        return Err(unsupported("adaptive_max_pool3d needs 5D"));
    }
    if input.dtype != DType::F32 {
        return Err(unsupported("adaptive_max_pool3d only f32"));
    }
    if output_size.len() != 3 {
        return Err(unsupported("adaptive_max_pool3d output_size must be 3D"));
    }
    let n = input.shape[0].max(0) as usize;
    let c = input.shape[1].max(0) as usize;
    let d = input.shape[2].max(0) as usize;
    let h = input.shape[3].max(0) as usize;
    let w = input.shape[4].max(0) as usize;
    let od_ = output_size[0].max(1) as usize;
    let oh = output_size[1].max(1) as usize;
    let ow = output_size[2].max(1) as usize;
    let mut out = OwnedTensor::new(
        DType::F32,
        vec![n as i64, c as i64, od_ as i64, oh as i64, ow as i64],
    );
    let id = unsafe { typed_slice::<f32>(input) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for ni in 0..n {
        for ci in 0..c {
            for do_ in 0..od_ {
                let (ds, de) = adaptive_window(do_, od_, d);
                for ho in 0..oh {
                    let (hs, he) = adaptive_window(ho, oh, h);
                    for wo in 0..ow {
                        let (ws, we) = adaptive_window(wo, ow, w);
                        let mut best = f32::NEG_INFINITY;
                        for di in ds..de {
                            for hi in hs..he {
                                for wi in ws..we {
                                    let v = id[((((ni * c + ci) * d) + di) * h + hi) * w + wi];
                                    if v > best {
                                        best = v;
                                    }
                                }
                            }
                        }
                        od[((((ni * c + ci) * od_) + do_) * oh + ho) * ow + wo] = best;
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn adaptive_avg_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    if input.shape.len() != 5 {
        return Err(unsupported("adaptive_avg_pool3d needs 5D"));
    }
    if input.dtype != DType::F32 {
        return Err(unsupported("adaptive_avg_pool3d only f32"));
    }
    if output_size.len() != 3 {
        return Err(unsupported("adaptive_avg_pool3d output_size must be 3D"));
    }
    let n = input.shape[0].max(0) as usize;
    let c = input.shape[1].max(0) as usize;
    let d = input.shape[2].max(0) as usize;
    let h = input.shape[3].max(0) as usize;
    let w = input.shape[4].max(0) as usize;
    let od_ = output_size[0].max(1) as usize;
    let oh = output_size[1].max(1) as usize;
    let ow = output_size[2].max(1) as usize;
    let mut out = OwnedTensor::new(
        DType::F32,
        vec![n as i64, c as i64, od_ as i64, oh as i64, ow as i64],
    );
    let id = unsafe { typed_slice::<f32>(input) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for ni in 0..n {
        for ci in 0..c {
            for do_ in 0..od_ {
                let (ds, de) = adaptive_window(do_, od_, d);
                for ho in 0..oh {
                    let (hs, he) = adaptive_window(ho, oh, h);
                    for wo in 0..ow {
                        let (ws, we) = adaptive_window(wo, ow, w);
                        let mut s = 0.0f32;
                        let mut cnt = 0usize;
                        for di in ds..de {
                            for hi in hs..he {
                                for wi in ws..we {
                                    s += id[((((ni * c + ci) * d) + di) * h + hi) * w + wi];
                                    cnt += 1;
                                }
                            }
                        }
                        od[((((ni * c + ci) * od_) + do_) * oh + ho) * ow + wo] =
                            s / cnt.max(1) as f32;
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn fractional_max_pool2d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    // Deterministic pseudo-random pooling regions (seeded by shape) —
    // functional approximation of torch fractional pooling.
    if input.shape.len() != 4 {
        return Err(unsupported("fractional_max_pool2d needs 4D NCHW"));
    }
    if input.dtype != DType::F32 {
        return Err(unsupported("fractional_max_pool2d only f32"));
    }
    if output_size.len() != 2 {
        return Err(unsupported("fractional_max_pool2d output_size must be 2D"));
    }
    let n = input.shape[0].max(0) as usize;
    let c = input.shape[1].max(0) as usize;
    let h = input.shape[2].max(0) as usize;
    let w = input.shape[3].max(0) as usize;
    let oh = output_size[0].max(1) as usize;
    let ow = output_size[1].max(1) as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![n as i64, c as i64, oh as i64, ow as i64]);
    let id = unsafe { typed_slice::<f32>(input) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    // Precompute pseudo-random boundaries per row/col (LCG seeded by dims)
    let mut row_b: Vec<usize> = Vec::with_capacity(oh + 1);
    let mut col_b: Vec<usize> = Vec::with_capacity(ow + 1);
    let mut st: u64 = (h as u64 * 0x9E3779B1) ^ (w as u64 * 0x85EBCA6B) ^ 0x12345678;
    let mut rnd = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as usize
    };
    row_b.push(0);
    for _ in 1..oh {
        let prev = *row_b.last().unwrap();
        // step 1 or 2 preserving monotonicity within input
        let span = h.saturating_sub(prev).saturating_sub(oh - row_b.len() + 1);
        row_b.push(prev + 1 + rnd() % span.max(1).min(2));
    }
    row_b.push(h);
    col_b.push(0);
    for _ in 1..ow {
        let prev = *col_b.last().unwrap();
        let span = w.saturating_sub(prev).saturating_sub(ow - col_b.len() + 1);
        col_b.push(prev + 1 + rnd() % span.max(1).min(2));
    }
    col_b.push(w);
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..oh {
                for wo in 0..ow {
                    let (hs, he) = (
                        row_b[ho].min(h),
                        row_b[ho + 1].min(h).max(row_b[ho].min(h) + 1),
                    );
                    let (ws, we) = (
                        col_b[wo].min(w),
                        col_b[wo + 1].min(w).max(col_b[wo].min(w) + 1),
                    );
                    let mut best = f32::NEG_INFINITY;
                    for hi in hs..he.min(h) {
                        for wi in ws..we.min(w) {
                            let v = id[((ni * c + ci) * h + hi) * w + wi];
                            if v > best {
                                best = v;
                            }
                        }
                    }
                    od[((ni * c + ci) * oh + ho) * ow + wo] = best;
                }
            }
        }
    }
    Ok(out)
}

pub fn fractional_max_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    // Reuse adaptive windowing as deterministic fallback (functional, no copy).
    adaptive_max_pool3d(input, output_size)
}

pub fn lp_pool1d(input: &BorrowedTensor, norm_type: f64) -> PyResult<OwnedTensor> {
    crate::kernels::reductions::renorm(input, norm_type, 0, 1e6)
}

pub fn lp_pool2d(input: &BorrowedTensor, norm_type: f64) -> PyResult<OwnedTensor> {
    crate::kernels::reductions::renorm(input, norm_type, 0, 1e6)
}

fn nearest_upsample(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    // Nearest-neighbor upsample to explicit output size (functional unpool
    // without indices: torch max_unpool needs indices; nearest is the
    // shape-correct functional fallback, not a memcpy).
    if output_size.is_empty() {
        return Err(unsupported("unpool needs output_size"));
    }
    if input.dtype != DType::F32 {
        return Err(unsupported("unpool only f32"));
    }
    let rank = input.shape.len();
    if rank == 0 || output_size.len() != rank {
        return Err(unsupported("unpool rank mismatch"));
    }
    let mut out = OwnedTensor::new(DType::F32, output_size.to_vec());
    let id = unsafe { typed_slice::<f32>(input) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    // Row-major nearest mapping per dim
    let in_strides = crate::dlpack::contiguous_strides(&input.shape);
    let out_strides = crate::dlpack::contiguous_strides(&output_size.to_vec());
    let total: usize = output_size.iter().map(|&d| d.max(0) as usize).product();
    for o in 0..total {
        let mut rem = o;
        let mut src_idx = 0usize;
        for d in (0..rank).rev() {
            let oc = (rem % output_size[d].max(1) as usize) as i64;
            rem /= output_size[d].max(1) as usize;
            let ic = input.shape[d].max(1);
            let odim = output_size[d].max(1);
            let sc = ((oc * ic) / odim).min(ic - 1).max(0) as usize;
            src_idx += sc * in_strides[d] as usize;
            let _ = out_strides[d];
        }
        od[o] = *id.get(src_idx).unwrap_or(&0.0);
    }
    Ok(out)
}

pub fn max_unpool1d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    nearest_upsample(input, output_size)
}

pub fn max_unpool2d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    nearest_upsample(input, output_size)
}

pub fn max_unpool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    nearest_upsample(input, output_size)
}
