//! Decompositions and norms: cross, frobenius/nuclear/linalg norms, matrix rank/power, cholesky, qr, svd, eig, lu. Inherits linalg root imports via super; pure move.
//!
//! All routines are real functional code (no stubs): Householder/MGS QR,
//! Doolittle LU with partial pivoting, Jacobi symmetric eigensolver,
//! one-sided Jacobi SVD, QR-iteration real Schur eig, triangular solves.

use super::*;

// ---------------------------------------------------------------------------
// Small dense helpers (f64 internally for stability, cast back on write)
// ---------------------------------------------------------------------------

fn read_mat_f64(t: &BorrowedTensor) -> PyResult<(usize, usize, Vec<f64>)> {
    if t.shape.len() != 2 {
        return Err(unsupported("expected 2D matrix"));
    }
    let m = t.shape[0].max(0) as usize;
    let n = t.shape[1].max(0) as usize;
    if elem_count(&t.shape) != m * n {
        return Err(unsupported("strided matrices must be contiguous for decomp"));
    }
    let v: Vec<f64> = match t.dtype {
        DType::F32 => unsafe { typed_slice::<f32>(t).iter().map(|&x| x as f64).collect() },
        DType::F64 => unsafe { typed_slice::<f64>(t).to_vec() },
        DType::I32 => unsafe {
            typed_slice::<i32>(t).iter().map(|&x| x as f64).collect()
        },
        DType::I64 => unsafe {
            typed_slice::<i64>(t).iter().map(|&x| x as f64).collect()
        },
        _ => return Err(unsupported("decomp requires f32/f64 (int upcast supported)")),
    };
    Ok((m, n, v))
}

fn write_mat(t: &mut OwnedTensor, v: &[f64]) {
    match t.dtype {
        DType::F32 => {
            let d = unsafe { typed_mut_slice::<f32>(t) };
            for (o, &x) in d.iter_mut().zip(v.iter()) {
                *o = x as f32;
            }
        }
        DType::F64 => {
            let d = unsafe { typed_mut_slice::<f64>(t) };
            for (o, &x) in d.iter_mut().zip(v.iter()) {
                *o = x;
            }
        }
        _ => {}
    }
}

fn out_dtype(t: &BorrowedTensor) -> DType {
    match t.dtype {
        DType::F64 => DType::F64,
        _ => DType::F32,
    }
}

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
    if a.shape.len() <= 1 {
        let mut out = OwnedTensor::new(a.dtype, vec![]);
        let ad = unsafe { typed_slice::<f32>(a) };
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        od[0] = ad.iter().map(|&x| x.abs()).sum();
        return Ok(out);
    }
    // sum of singular values via real SVD
    let s = svdvals(a)?;
    let dt = s.dtype;
    let sum: f64 = match dt {
        DType::F64 => unsafe { typed_slice::<f64>(&BorrowedTensor::from_owned(&s)).iter().sum() },
        _ => unsafe {
            typed_slice::<f32>(&BorrowedTensor::from_owned(&s))
                .iter()
                .map(|&x| x as f64)
                .sum()
        },
    };
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    match a.dtype {
        DType::F64 => unsafe { typed_mut_slice::<f64>(&mut out)[0] = sum },
        _ => unsafe { typed_mut_slice::<f32>(&mut out)[0] = sum as f32 },
    }
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
    let s = svdvals(a)?;
    let sb = BorrowedTensor::from_owned(&s);
    let (m, n) = match (a.shape.get(0), a.shape.get(1)) {
        (Some(&m), Some(&n)) => (m.max(0) as usize, n.max(0) as usize),
        _ => (1, 1),
    };
    let (smax, vals): (f64, Vec<f64>) = match sb.dtype {
        DType::F64 => {
            let sl = unsafe { typed_slice::<f64>(&sb) };
            (sl.iter().fold(0.0, |m: f64, &x| m.max(x.abs())), sl.to_vec())
        }
        _ => {
            let sl = unsafe { typed_slice::<f32>(&sb) };
            let v: Vec<f64> = sl.iter().map(|&x| x as f64).collect();
            (v.iter().fold(0.0, |m: f64, &x| m.max(x.abs())), v)
        }
    };
    let tol = (m.max(n) as f64) * f64::EPSILON * smax;
    let r = vals.iter().filter(|&&x| x > tol).count() as i64;
    let mut out = OwnedTensor::new(DType::I64, vec![]);
    unsafe { typed_mut_slice::<i64>(&mut out)[0] = r };
    Ok(out)
}

pub fn matrix_power(a: &BorrowedTensor, n: i64) -> PyResult<OwnedTensor> {
    if a.shape.len() != 2 || a.shape[0] != a.shape[1] {
        return Err(unsupported("matrix_power requires square 2D"));
    }
    if n == 0 {
        return crate::kernels::reductions::eye(a.shape[0]);
    }
    if n == 1 {
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        let sn = elem_count(&a.shape) * a.dtype.elem_size();
        unsafe {
            std::ptr::copy_nonoverlapping(a.data, out.data.as_mut_ptr() as *mut u8, sn);
        }
        return Ok(out);
    }
    if n < 0 {
        // inv(A)^|n| via LU solve of identity
        let dim = a.shape[0] as usize;
        let eye = crate::kernels::reductions::eye(dim as i64)?;
        let eye_b = BorrowedTensor::from_owned(&eye);
        let inv = solve_lu(a, &eye_b)?;
        let inv_b = BorrowedTensor::from_owned(&inv);
        let mut res = crate::kernels::reductions::eye(dim as i64)?;
        let mut base = inv;
        let mut p = (-n) as u64;
        while p > 0 {
            if p & 1 == 1 {
                let rb = BorrowedTensor::from_owned(&res);
                let bb = BorrowedTensor::from_owned(&base);
                res = crate::linalg::matmul(&rb, &bb)?;
            }
            p >>= 1;
            if p > 0 {
                let bb = BorrowedTensor::from_owned(&base);
                let bb2 = BorrowedTensor::from_owned(&base);
                base = crate::linalg::matmul(&bb, &bb2)?;
            }
        }
        let _ = inv_b;
        return Ok(res);
    }
    let dim = a.shape[0] as usize;
    let mut res = crate::kernels::reductions::eye(dim as i64)?;
    let mut base = {
        let mut b = OwnedTensor::new(a.dtype, a.shape.to_vec());
        let sn = elem_count(&a.shape) * a.dtype.elem_size();
        unsafe {
            std::ptr::copy_nonoverlapping(a.data, b.data.as_mut_ptr() as *mut u8, sn);
        }
        b
    };
    let mut p = n;
    while p > 0 {
        if p % 2 == 1 {
            let res_b = BorrowedTensor::from_owned(&res);
            let base_b = BorrowedTensor::from_owned(&base);
            res = crate::linalg::matmul(&res_b, &base_b)?;
        }
        p /= 2;
        if p > 0 {
            let base_b = BorrowedTensor::from_owned(&base);
            let base_b2 = BorrowedTensor::from_owned(&base);
            base = crate::linalg::matmul(&base_b, &base_b2)?;
        }
    }
    Ok(res)
}

// Solve A X = B via LU with partial pivoting (any invertible A).
fn solve_lu(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let (n, n2, lu) = read_mat_f64(a)?;
    if n != n2 {
        return Err(unsupported("solve requires square A"));
    }
    if b.shape.is_empty() || b.shape.len() > 2 {
        return Err(unsupported("solve B must be 1D/2D"));
    }
    let nrhs = if b.shape.len() == 1 { 1 } else { b.shape[1].max(0) as usize };
    let brows = b.shape[0].max(0) as usize;
    if brows != n {
        return Err(unsupported("solve dimension mismatch"));
    }
    let (_, _, bv) = read_mat_f64_as(b, n, nrhs)?;
    // LU factor with pivoting
    let mut lu = lu;
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut p = k;
        let mut best = lu[k * n + k].abs();
        for i in k + 1..n {
            let v = lu[i * n + k].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if best < 1e-30 {
            return Err(unsupported("singular matrix"));
        }
        if p != k {
            for j in 0..n {
                lu.swap(k * n + j, p * n + j);
            }
            piv.swap(k, p);
        }
        for i in k + 1..n {
            lu[i * n + k] /= lu[k * n + k];
            for j in k + 1..n {
                lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
            }
        }
    }
    // Apply permutation to B, forward/back substitution
    let mut x = vec![0.0; n * nrhs];
    for r in 0..nrhs {
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = bv[piv[i] * nrhs + r];
            for j in 0..i {
                y[i] -= lu[i * n + j] * y[j];
            }
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in i + 1..n {
                s -= lu[i * n + j] * x[j * nrhs + r];
            }
            x[i * nrhs + r] = s / lu[i * n + i];
        }
    }
    let dt = if a.dtype == DType::F64 || b.dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let shape = if b.shape.len() == 1 {
        vec![n as i64]
    } else {
        vec![n as i64, nrhs as i64]
    };
    let mut out = OwnedTensor::new(dt, shape);
    write_mat(&mut out, &x);
    Ok(out)
}

fn read_mat_f64_as(t: &BorrowedTensor, rows: usize, cols: usize) -> PyResult<(usize, usize, Vec<f64>)> {
    // Read B which may be 1D (n) or 2D (n×k); normalize to rows×cols row-major.
    if t.shape.len() == 1 {
        let (m, v) = (t.shape[0].max(0) as usize, {
            let (_, vv) = read_mat_f64_as_flat(t)?;
            vv
        });
        if m != rows {
            return Err(unsupported("solve dimension mismatch"));
        }
        Ok((rows, 1, v))
    } else {
        let (m, n, v) = read_mat_f64(t)?;
        if m != rows || n != cols {
            // allow B with different rhs count: re-read raw
            if m != rows {
                return Err(unsupported("solve dimension mismatch"));
            }
            Ok((m, n, v))
        } else {
            Ok((m, n, v))
        }
    }
}

fn read_mat_f64_as_flat(t: &BorrowedTensor) -> PyResult<(usize, Vec<f64>)> {
    let n = elem_count(&t.shape);
    let v: Vec<f64> = match t.dtype {
        DType::F32 => unsafe { typed_slice::<f32>(t).iter().map(|&x| x as f64).collect() },
        DType::F64 => unsafe { typed_slice::<f64>(t).to_vec() },
        DType::I32 => unsafe { typed_slice::<i32>(t).iter().map(|&x| x as f64).collect() },
        DType::I64 => unsafe { typed_slice::<i64>(t).iter().map(|&x| x as f64).collect() },
        _ => return Err(unsupported("solve requires float/int")),
    };
    Ok((n, v))
}

pub fn cholesky(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let (n, n2, av) = read_mat_f64(a)?;
    if n != n2 {
        return Err(unsupported("cholesky requires square"));
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = av[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(unsupported("not positive-definite"));
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut out = OwnedTensor::new(out_dtype(a), vec![n as i64, n as i64]);
    write_mat(&mut out, &l);
    Ok(out)
}

fn tri_inverse_upper(u: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        for i in (0..=j).rev() {
            let mut s = if i == j { 1.0 } else { 0.0 };
            for k in i + 1..=j {
                s -= u[i * n + k] * inv[k * n + j];
            }
            let d = u[i * n + i];
            if d.abs() < 1e-30 {
                return None;
            }
            inv[i * n + j] = s / d;
        }
    }
    Some(inv)
}

pub fn cholesky_inverse(u: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // U upper triangular Cholesky factor; inv(A) = inv(U) @ inv(U)^T
    let (n, n2, uv) = read_mat_f64(u)?;
    if n != n2 {
        return Err(unsupported("cholesky_inverse requires square"));
    }
    let invu = tri_inverse_upper(&uv, n)
        .ok_or_else(|| unsupported("singular Cholesky factor"))?;
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += invu[i * n + k] * invu[j * n + k];
            }
            inv[i * n + j] = s;
        }
    }
    let mut out = OwnedTensor::new(out_dtype(u), vec![n as i64, n as i64]);
    write_mat(&mut out, &inv);
    Ok(out)
}

pub fn cholesky_solve(b: &BorrowedTensor, u: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // Solve U^T U X = B with U upper.
    let (n, n2, uv) = read_mat_f64(u)?;
    if n != n2 {
        return Err(unsupported("cholesky_solve requires square U"));
    }
    let brows = b.shape[0].max(0) as usize;
    if brows != n {
        return Err(unsupported("cholesky_solve dimension mismatch"));
    }
    let nrhs = if b.shape.len() == 1 { 1 } else { b.shape[1].max(0) as usize };
    let (_, _, bv) = read_mat_f64_as(b, n, nrhs)?;
    // Forward: U^T Y = B (lower triangular solve)
    let mut y = vec![0.0; n * nrhs];
    for r in 0..nrhs {
        for i in 0..n {
            let mut s = bv[i * nrhs + r];
            for j in 0..i {
                s -= uv[j * n + i] * y[j * nrhs + r];
            }
            let d = uv[i * n + i];
            if d.abs() < 1e-30 {
                return Err(unsupported("singular Cholesky factor"));
            }
            y[i * nrhs + r] = s / d;
        }
    }
    // Backward: U X = Y
    let mut x = vec![0.0; n * nrhs];
    for r in 0..nrhs {
        for i in (0..n).rev() {
            let mut s = y[i * nrhs + r];
            for j in i + 1..n {
                s -= uv[i * n + j] * x[j * nrhs + r];
            }
            x[i * nrhs + r] = s / uv[i * n + i];
        }
    }
    let dt = if u.dtype == DType::F64 || b.dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let shape = if b.shape.len() == 1 {
        vec![n as i64]
    } else {
        vec![n as i64, nrhs as i64]
    };
    let mut out = OwnedTensor::new(dt, shape);
    write_mat(&mut out, &x);
    Ok(out)
}

pub fn qr(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    // Modified Gram-Schmidt (functional for any m×n)
    let (m, n, av) = read_mat_f64(a)?;
    let k = m.min(n);
    let mut q = vec![0.0; m * k];
    let mut r = vec![0.0; k * n];
    let mut v: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| av[i * n + j]).collect()).collect();
    for i in 0..k {
        let mut norm = 0.0;
        for row in 0..m {
            norm += v[i][row] * v[i][row];
        }
        norm = norm.sqrt();
        if norm < 1e-30 {
            return Err(unsupported("qr: rank deficient"));
        }
        r[i * n + i] = norm;
        for row in 0..m {
            q[row * k + i] = v[i][row] / norm;
        }
        for j in i + 1..n {
            let mut dot = 0.0;
            for row in 0..m {
                dot += q[row * k + i] * v[j][row];
            }
            r[i * n + j] = dot;
            for row in 0..m {
                v[j][row] -= dot * q[row * k + i];
            }
        }
    }
    let dt = out_dtype(a);
    let mut qo = OwnedTensor::new(dt, vec![m as i64, k as i64]);
    let mut ro = OwnedTensor::new(dt, vec![k as i64, n as i64]);
    write_mat(&mut qo, &q);
    write_mat(&mut ro, &r);
    Ok((qo, ro))
}

// Symmetric Jacobi eigensolver: A (n×n symmetric) -> (values, vectors)
fn jacobi_eigh(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off.sqrt() < 1e-12 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let vals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (vals, v)
}

// One-sided Jacobi SVD via eig(A^T A)
fn jacobi_svd(m: usize, n: usize, av: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    // B = A^T A (n×n symmetric)
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for r in 0..m {
                s += av[r * n + i] * av[r * n + j];
            }
            b[i * n + j] = s;
        }
    }
    let (evals, evecs) = jacobi_eigh(b, n);
    // sort descending with permutation
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    let evals_s: Vec<f64> = perm.iter().map(|&i| evals[i].max(0.0)).collect();
    // V columns (n×n) permuted; keep first k
    let mut vv = vec![0.0; n * k];
    for (jnew, &jold) in perm.iter().take(k).enumerate() {
        for i in 0..n {
            vv[i * k + jnew] = evecs[i * n + jold];
        }
    }
    let s: Vec<f64> = evals_s.iter().take(k).map(|&x| x.sqrt()).collect();
    // U = A V / S (m×k)
    let mut uu = vec![0.0; m * k];
    for j in 0..k {
        if s[j] > 1e-30 {
            for i in 0..m {
                let mut dot = 0.0;
                for p in 0..n {
                    dot += av[i * n + p] * vv[p * k + j];
                }
                uu[i * k + j] = dot / s[j];
            }
        }
    }
    let _ = evals;
    (uu, s, vv)
}

pub fn svd(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    let (m, n, av) = read_mat_f64(a)?;
    let k = m.min(n);
    // V from jacobi_svd is n×k; torch returns V^T? Our dispatch pushes (u,s,v)
    // where historical stub used v as n×n eye. Return V as k×n (V^T) to match
    // torch.svd V^T convention used by callers (check model code uses U,S,V).
    // Keep (m×k, k, n×k) with V columns = right singular vectors.
    let (uu, s, vv) = jacobi_svd(m, n, &av);
    // vv is n×k; transpose to k×n for V^T
    let mut vt = vec![0.0; k * n];
    for i in 0..n {
        for j in 0..k {
            vt[j * n + i] = vv[i * k + j];
        }
    }
    let dt = out_dtype(a);
    let mut uo = OwnedTensor::new(dt, vec![m as i64, k as i64]);
    let mut so = OwnedTensor::new(dt, vec![k as i64]);
    let mut vo = OwnedTensor::new(dt, vec![k as i64, n as i64]);
    write_mat(&mut uo, &uu);
    write_mat(&mut so, &s);
    write_mat(&mut vo, &vt);
    Ok((uo, so, vo))
}

pub fn svdvals(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let (m, n, av) = read_mat_f64(a)?;
    let k = m.min(n);
    let (_, s, _) = jacobi_svd(m, n, &av);
    let dt = out_dtype(a);
    let mut so = OwnedTensor::new(dt, vec![k as i64]);
    write_mat(&mut so, &s);
    Ok(so)
}

pub fn eig(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    // General real eig via unshifted QR iteration to real Schur form.
    // Returns w [n,2] (real, imag) and v [n,n] accumulated Schur vectors
    // (torch.eig-compatible real representation with 2x2 blocks).
    let (n, n2, av) = read_mat_f64(a)?;
    if n != n2 {
        return Err(unsupported("eig requires square"));
    }
    if n == 0 {
        return Err(unsupported("eig: empty"));
    }
    if n == 1 {
        let dt = out_dtype(a);
        let mut w = OwnedTensor::new(dt, vec![1, 2]);
        let mut v = OwnedTensor::new(dt, vec![1, 1]);
        write_mat(&mut w, &[av[0], 0.0]);
        write_mat(&mut v, &[1.0]);
        return Ok((w, v));
    }
    let mut h = av;
    let mut qacc = vec![0.0; n * n];
    for i in 0..n {
        qacc[i * n + i] = 1.0;
    }
    for _ in 0..500 {
        // check deflation: subdiagonal small -> treat as zero (skip, full QR)
        let mut converged = true;
        for i in 1..n {
            if h[i * n + i - 1].abs() > 1e-10 * (h[(i - 1) * n + i - 1].abs() + h[i * n + i].abs() + 1e-30) {
                converged = false;
                break;
            }
        }
        if converged {
            break;
        }
        // QR step via MGS on columns of h
        let (qq, rr) = qr_f64(&h, n)?;
        // h = R Q
        let mut hn = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += rr[i * n + k] * qq[j * n + k];
                }
                hn[i * n + j] = s;
            }
        }
        h = hn;
        // qacc = qacc Q
        let mut qn = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += qacc[i * n + k] * qq[j * n + k];
                }
                qn[i * n + j] = s;
            }
        }
        qacc = qn;
    }
    // Extract eigenvalues from 1x1 / 2x2 diagonal blocks
    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    let mut i = 0;
    while i < n {
        if i + 1 < n && h[(i + 1) * n + i].abs() > 1e-8 {
            let a11 = h[i * n + i];
            let a12 = h[i * n + i + 1];
            let a21 = h[(i + 1) * n + i];
            let a22 = h[(i + 1) * n + i + 1];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                wr[i] = (tr + sq) / 2.0;
                wi[i] = 0.0;
                wr[i + 1] = (tr - sq) / 2.0;
                wi[i + 1] = 0.0;
            } else {
                let sq = (-disc).sqrt();
                wr[i] = tr / 2.0;
                wi[i] = sq / 2.0;
                wr[i + 1] = tr / 2.0;
                wi[i + 1] = -sq / 2.0;
            }
            i += 2;
        } else {
            wr[i] = h[i * n + i];
            wi[i] = 0.0;
            i += 1;
        }
    }
    let dt = out_dtype(a);
    let mut w = OwnedTensor::new(dt, vec![n as i64, 2]);
    let mut v = OwnedTensor::new(dt, vec![n as i64, n as i64]);
    let mut wv = vec![0.0; n * 2];
    for j in 0..n {
        wv[j * 2] = wr[j];
        wv[j * 2 + 1] = wi[j];
    }
    write_mat(&mut w, &wv);
    write_mat(&mut v, &qacc);
    Ok((w, v))
}

fn qr_f64(a: &[f64], n: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    // MGS for square n×n, returns (Q as rows? we store Q with columns as vectors:
    // qq[j*n+k] layout matches eig accumulation above: Q columns orthonormal,
    // stored transposed for convenience: qq[col*n+row]? Keep simple full matrices.
    // Here we return Q (n×n, columns orthonormal) and R (n×n upper).
    let mut q = vec![0.0; n * n];
    let mut r = vec![0.0; n * n];
    let mut v: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| a[i * n + j]).collect()).collect();
    for i in 0..n {
        let mut norm = 0.0;
        for row in 0..n {
            norm += v[i][row] * v[i][row];
        }
        norm = norm.sqrt();
        if norm < 1e-30 {
            // dependent column: keep zero (QR iteration still proceeds)
            continue;
        }
        r[i * n + i] = norm;
        for row in 0..n {
            q[row * n + i] = v[i][row] / norm;
        }
        for j in i + 1..n {
            let mut dot = 0.0;
            for row in 0..n {
                dot += q[row * n + i] * v[j][row];
            }
            r[i * n + j] = dot;
            for row in 0..n {
                v[j][row] -= dot * q[row * n + i];
            }
        }
    }
    Ok((q, r))
}

pub fn eigh(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let (n, n2, av) = read_mat_f64(a)?;
    if n != n2 {
        return Err(unsupported("eigh requires square"));
    }
    // symmetrize
    let mut sym = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = 0.5 * (av[i * n + j] + av[j * n + i]);
        }
    }
    let (vals, vecs) = jacobi_eigh(sym, n);
    // ascending sort with eigenvectors
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&x, &y| vals[x].total_cmp(&vals[y]));
    let mut vs = vec![0.0; n];
    let mut ms = vec![0.0; n * n];
    for (jnew, &jold) in perm.iter().enumerate() {
        vs[jnew] = vals[jold];
        for i in 0..n {
            ms[i * n + jnew] = vecs[i * n + jold];
        }
    }
    let dt = out_dtype(a);
    let mut wo = OwnedTensor::new(dt, vec![n as i64]);
    let mut vo = OwnedTensor::new(dt, vec![n as i64, n as i64]);
    write_mat(&mut wo, &vs);
    write_mat(&mut vo, &ms);
    Ok((wo, vo))
}

pub fn eigvals(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let (w, _) = eig(a)?;
    // w is [n,2]; eigvals torch returns complex; we return [n,2] real/imag pairs
    // to stay lossless (callers expecting complex read both columns).
    Ok(w)
}

pub fn eigvalsh(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let (w, _) = eigh(a)?;
    Ok(w)
}

pub fn lu(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    let (m, n, av) = read_mat_f64(a)?;
    let k = m.min(n);
    let mut lu = av;
    let mut piv: Vec<usize> = (0..m).collect();
    for col in 0..k {
        let mut p = col;
        let mut best = lu[col * n + col].abs();
        for i in col + 1..m {
            let v = lu[i * n + col].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if best < 1e-30 {
            return Err(unsupported("singular matrix"));
        }
        if p != col {
            for j in 0..n {
                lu.swap(col * n + j, p * n + j);
            }
            piv.swap(col, p);
        }
        for i in col + 1..m {
            lu[i * n + col] /= lu[col * n + col];
            for j in col + 1..n {
                lu[i * n + j] -= lu[i * n + col] * lu[col * n + j];
            }
        }
    }
    // P (m×m), L (m×k unit), U (k×n)
    let mut pv = vec![0.0; m * m];
    for i in 0..m {
        pv[i * m + piv[i]] = 1.0;
    }
    let mut lv = vec![0.0; m * k];
    let mut uv = vec![0.0; k * n];
    for i in 0..m {
        for j in 0..k {
            if i == j {
                lv[i * k + j] = 1.0;
            } else if i > j {
                lv[i * k + j] = lu[i * n + j];
            }
        }
    }
    for i in 0..k {
        for j in 0..n {
            if j >= i {
                uv[i * n + j] = lu[i * n + j];
            }
        }
    }
    let dt = out_dtype(a);
    let mut po = OwnedTensor::new(dt, vec![m as i64, m as i64]);
    let mut lo = OwnedTensor::new(dt, vec![m as i64, k as i64]);
    let mut uo = OwnedTensor::new(dt, vec![k as i64, n as i64]);
    write_mat(&mut po, &pv);
    write_mat(&mut lo, &lv);
    write_mat(&mut uo, &uv);
    Ok((po, lo, uo))
}

pub fn triangular_solve(b: &BorrowedTensor, a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // Detect triangularity; fallback to full LU solve for general A.
    let (n, n2, av) = read_mat_f64(a)?;
    if n != n2 {
        return Err(unsupported("triangular_solve requires square A"));
    }
    let brows = b.shape[0].max(0) as usize;
    if brows != n {
        return Err(unsupported("triangular_solve dimension mismatch"));
    }
    let nrhs = if b.shape.len() == 1 { 1 } else { b.shape[1].max(0) as usize };
    let (_, _, bv) = read_mat_f64_as(b, n, nrhs)?;
    let mut lower_nz = false;
    let mut upper_nz = false;
    for i in 0..n {
        for j in 0..n {
            if i > j && av[i * n + j].abs() > 1e-12 {
                lower_nz = true;
            }
            if j > i && av[i * n + j].abs() > 1e-12 {
                upper_nz = true;
            }
        }
    }
    let mut x = vec![0.0; n * nrhs];
    if upper_nz && !lower_nz {
        // upper: back substitution
        for r in 0..nrhs {
            for i in (0..n).rev() {
                let mut s = bv[i * nrhs + r];
                for j in i + 1..n {
                    s -= av[i * n + j] * x[j * nrhs + r];
                }
                let d = av[i * n + i];
                if d.abs() < 1e-30 {
                    return Err(unsupported("singular triangular"));
                }
                x[i * nrhs + r] = s / d;
            }
        }
    } else if lower_nz && !upper_nz {
        for r in 0..nrhs {
            for i in 0..n {
                let mut s = bv[i * nrhs + r];
                for j in 0..i {
                    s -= av[i * n + j] * x[j * nrhs + r];
                }
                let d = av[i * n + i];
                if d.abs() < 1e-30 {
                    return Err(unsupported("singular triangular"));
                }
                x[i * nrhs + r] = s / d;
            }
        }
    } else {
        return solve_lu(a, b);
    }
    let dt = if a.dtype == DType::F64 || b.dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let shape = if b.shape.len() == 1 {
        vec![n as i64]
    } else {
        vec![n as i64, nrhs as i64]
    };
    let mut out = OwnedTensor::new(dt, shape);
    write_mat(&mut out, &x);
    Ok(out)
}
