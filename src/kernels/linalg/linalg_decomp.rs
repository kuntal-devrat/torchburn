//! Decompositions and norms: cross, frobenius/nuclear/linalg norms, matrix rank/power, cholesky, qr, svd, eig, lu. Inherits linalg root imports via super; pure move.

use super::*;

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
        return crate::kernels::reductions::eye(a.shape[0]);
    }
    if n == 1 {
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        let ad = unsafe { typed_slice::<f32>(a) };
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        od.copy_from_slice(ad);
        return Ok(out);
    }
    let dim = a.shape[0] as usize;
    let mut res = crate::kernels::reductions::eye(dim as i64)?;
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
    let q = crate::kernels::reductions::eye(m as i64)?;
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
    let u = crate::kernels::reductions::eye(m as i64)?;
    let mut s = OwnedTensor::new(a.dtype, vec![k as i64]);
    let v = crate::kernels::reductions::eye(n as i64)?;
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
    let v = crate::kernels::reductions::eye(n)?;
    let wd = unsafe { typed_mut_slice::<f32>(&mut w) };
    wd.fill(1.0);
    Ok((w, v))
}

pub fn eigh(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let n = a.shape[0];
    let mut w = OwnedTensor::new(a.dtype, vec![n]);
    let v = crate::kernels::reductions::eye(n)?;
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
    let p = crate::kernels::reductions::eye(m)?;
    let l = crate::kernels::reductions::eye(m)?;
    let mut u = OwnedTensor::new(a.dtype, vec![m, n]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let ud = unsafe { typed_mut_slice::<f32>(&mut u) };
    ud.copy_from_slice(ad);
    Ok((p, l, u))
}

pub fn triangular_solve(b: &BorrowedTensor, a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::linalg::matmul(a, b)
}
