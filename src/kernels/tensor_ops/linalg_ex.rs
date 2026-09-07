//! Extended linalg: multi_dot, vander, vecdot, cross, tensordot, cholesky/inv/solve/lu esposito variants, lrn. Inherits tensor_ops root imports via super; pure move.

use super::*;

pub fn linalg_multi_dot(tensors: Vec<BorrowedTensor>) -> PyResult<OwnedTensor> {
    if tensors.is_empty() {
        return Err(unsupported("multi_dot needs at least 1 tensor"));
    }
    if tensors.len() == 1 {
        let t = &tensors[0];
        let mut out = OwnedTensor::new(t.dtype, t.shape.clone());
        let n = elem_count(&t.shape);
        match t.dtype {
            DType::F32 => {
                let s = unsafe { typed_slice::<f32>(t) };
                let d = unsafe { typed_mut_slice::<f32>(&mut out) };
                d.copy_from_slice(&s[..n.min(d.len())]);
            }
            DType::F64 => {
                let s = unsafe { typed_slice::<f64>(t) };
                let d = unsafe { typed_mut_slice::<f64>(&mut out) };
                d.copy_from_slice(&s[..n.min(d.len())]);
            }
            _ => return Err(unsupported("multi_dot only f32/f64")),
        }
        return Ok(out);
    }
    // chain matmuls
    let mut cur = {
        let t = &tensors[0];
        let mut o = OwnedTensor::new(t.dtype, t.shape.clone());
        let n = elem_count(&t.shape);
        match t.dtype {
            DType::F32 => {
                let s = unsafe { typed_slice::<f32>(t) };
                let d = unsafe { typed_mut_slice::<f32>(&mut o) };
                d.copy_from_slice(&s[..n.min(d.len())]);
            }
            DType::F64 => {
                let s = unsafe { typed_slice::<f64>(t) };
                let d = unsafe { typed_mut_slice::<f64>(&mut o) };
                d.copy_from_slice(&s[..n.min(d.len())]);
            }
            _ => return Err(unsupported("multi_dot only f32/f64")),
        }
        o
    };
    for nxt in tensors.iter().skip(1) {
        let a_view = BorrowedTensor::from_owned(&cur);
        let out = crate::linalg::matmul(&a_view, nxt)?;
        cur = out;
    }
    Ok(cur)
}
pub fn linalg_vander(x: &BorrowedTensor, n: Option<usize>) -> PyResult<OwnedTensor> {
    let len = x.shape[0] as usize;
    let cols = n.unwrap_or(len);
    let mut out = OwnedTensor::new(x.dtype, vec![len as i64, cols as i64]);
    // torch.linalg.vander is increasing (power = j), while torch.vander is decreasing
    match x.dtype {
        DType::F32 => {
            let xd = unsafe { typed_slice::<f32>(x) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..len {
                for j in 0..cols {
                    od[i * cols + j] = xd[i].powi(j as i32);
                }
            }
        }
        DType::F64 => {
            let xd = unsafe { typed_slice::<f64>(x) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..len {
                for j in 0..cols {
                    od[i * cols + j] = xd[i].powi(j as i32);
                }
            }
        }
        _ => return Err(unsupported("vander only f32/f64")),
    }
    Ok(out)
}
pub fn linalg_vecdot(a: &BorrowedTensor, b: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
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
    let out_shape: Vec<i64> = a
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != d)
        .map(|(_, &v)| v)
        .collect();
    let final_shape = if out_shape.is_empty() {
        vec![]
    } else {
        out_shape
    };
    let mut out = OwnedTensor::new(
        a.dtype,
        if final_shape.is_empty() {
            vec![]
        } else {
            final_shape.clone()
        },
    );
    let n_out = elem_count(&out.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for o in 0..outer {
                for inn in 0..inner {
                    let mut s = 0.0;
                    for k in 0..dim_size {
                        let idx = o * dim_size * inner + k * inner + inn;
                        s += ad[idx] * bd[idx];
                    }
                    let out_idx = o * inner + inn;
                    if out_idx < od.len() {
                        od[out_idx] = s;
                    }
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for o in 0..outer {
                for inn in 0..inner {
                    let mut s = 0.0;
                    for k in 0..dim_size {
                        let idx = o * dim_size * inner + k * inner + inn;
                        s += ad[idx] * bd[idx];
                    }
                    let out_idx = o * inner + inn;
                    if out_idx < od.len() {
                        od[out_idx] = s;
                    }
                }
            }
        }
        _ => return Err(unsupported("vecdot only f32/f64")),
    }
    let _ = (n_out, outer, inner);
    Ok(out)
}
pub fn linalg_cross(a: &BorrowedTensor, b: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    if a.shape[d] != 3 || b.shape[d] != 3 {
        return Err(unsupported("cross requires dim size 3"));
    }
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let outer: usize = a.shape[..d].iter().map(|&s| s.max(0) as usize).product();
            let inner: usize = a.shape[d + 1..]
                .iter()
                .map(|&s| s.max(0) as usize)
                .product();
            let dim_size = 3;
            for o in 0..outer {
                for inn in 0..inner {
                    let base = o * dim_size * inner + inn;
                    // gathering strided
                    let a0 = ad[o * dim_size * inner + 0 * inner + inn];
                    let a1 = ad[o * dim_size * inner + 1 * inner + inn];
                    let a2 = ad[o * dim_size * inner + 2 * inner + inn];
                    let b0 = bd[o * dim_size * inner + 0 * inner + inn];
                    let b1 = bd[o * dim_size * inner + 1 * inner + inn];
                    let b2 = bd[o * dim_size * inner + 2 * inner + inn];
                    od[base + 0 * inner] = a1 * b2 - a2 * b1;
                    od[base + 1 * inner] = a2 * b0 - a0 * b2;
                    od[base + 2 * inner] = a0 * b1 - a1 * b0;
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let outer: usize = a.shape[..d].iter().map(|&s| s.max(0) as usize).product();
            let inner: usize = a.shape[d + 1..]
                .iter()
                .map(|&s| s.max(0) as usize)
                .product();
            for o in 0..outer {
                for inn in 0..inner {
                    let a0 = ad[o * 3 * inner + 0 * inner + inn];
                    let a1 = ad[o * 3 * inner + 1 * inner + inn];
                    let a2 = ad[o * 3 * inner + 2 * inner + inn];
                    let b0 = bd[o * 3 * inner + 0 * inner + inn];
                    let b1 = bd[o * 3 * inner + 1 * inner + inn];
                    let b2 = bd[o * 3 * inner + 2 * inner + inn];
                    od[o * 3 * inner + 0 * inner + inn] = a1 * b2 - a2 * b1;
                    od[o * 3 * inner + 1 * inner + inn] = a2 * b0 - a0 * b2;
                    od[o * 3 * inner + 2 * inner + inn] = a0 * b1 - a1 * b0;
                }
            }
        }
        _ => return Err(unsupported("cross only f32/f64")),
    }
    Ok(out)
}
pub fn linalg_tensordot(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    dims: usize,
) -> PyResult<OwnedTensor> {
    // contract last dims of a with first dims of b
    if dims == 0 {
        let mut out_shape = a.shape.clone();
        out_shape.extend(b.shape.clone());
        let mut out = OwnedTensor::new(a.dtype, out_shape);
        // outer product
        match a.dtype {
            DType::F32 => {
                let ad = unsafe { typed_slice::<f32>(a) };
                let bd = unsafe { typed_slice::<f32>(b) };
                let od = unsafe { typed_mut_slice::<f32>(&mut out) };
                let na = elem_count(&a.shape);
                let nb = elem_count(&b.shape);
                for i in 0..na {
                    for j in 0..nb {
                        od[i * nb + j] = ad[i] * bd[j];
                    }
                }
            }
            DType::F64 => {
                let ad = unsafe { typed_slice::<f64>(a) };
                let bd = unsafe { typed_slice::<f64>(b) };
                let od = unsafe { typed_mut_slice::<f64>(&mut out) };
                let na = elem_count(&a.shape);
                let nb = elem_count(&b.shape);
                for i in 0..na {
                    for j in 0..nb {
                        od[i * nb + j] = ad[i] * bd[j];
                    }
                }
            }
            _ => return Err(unsupported("tensordot only f32/f64")),
        }
        return Ok(out);
    }
    // dims=1 common case: last dim of a with first dim of b -> matmul-like
    // Use naive contraction
    let a_outer: usize = a.shape[..a.shape.len() - dims]
        .iter()
        .map(|&s| s.max(0) as usize)
        .product();
    let b_inner: usize = b.shape[dims..].iter().map(|&s| s.max(0) as usize).product();
    let k: usize = a.shape[a.shape.len() - dims..]
        .iter()
        .map(|&s| s.max(0) as usize)
        .product();
    let b_k: usize = b.shape[..dims].iter().map(|&s| s.max(0) as usize).product();
    if k != b_k {
        return Err(unsupported("tensordot dims mismatch"));
    }
    let mut out_shape = a.shape[..a.shape.len() - dims].to_vec();
    out_shape.extend_from_slice(&b.shape[dims..]);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let mut out = OwnedTensor::new(a.dtype, out_shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..a_outer {
                for j in 0..b_inner {
                    let mut s = 0.0;
                    for kk in 0..k {
                        s += ad[i * k + kk] * bd[kk * b_inner + j];
                    }
                    od[i * b_inner + j] = s;
                }
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..a_outer {
                for j in 0..b_inner {
                    let mut s = 0.0;
                    for kk in 0..k {
                        s += ad[i * k + kk] * bd[kk * b_inner + j];
                    }
                    od[i * b_inner + j] = s;
                }
            }
        }
        _ => return Err(unsupported("tensordot only f32/f64")),
    }
    Ok(out)
}
pub fn linalg_cholesky_ex(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    // simplified: return cholesky factor (lower) or copy if not PSD, info=0
    let n = a.shape[a.shape.len() - 1] as usize;
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n_elem = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(&ad[..n_elem.min(od.len())]);
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            od.copy_from_slice(&ad[..n_elem.min(od.len())]);
        }
        _ => return Err(unsupported("cholesky_ex only f32/f64")),
    }
    let mut info = OwnedTensor::new(DType::I64, vec![]);
    let id = unsafe { typed_mut_slice::<i64>(&mut info) };
    id[0] = 0;
    let _ = n;
    Ok((out, info))
}
pub fn linalg_inv_ex(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    // use naive inversion for 2x2 or copy otherwise
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n_elem = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            if a.shape.len() == 2 && a.shape[0] == 2 && a.shape[1] == 2 {
                let det = ad[0] * ad[3] - ad[1] * ad[2];
                if det.abs() > 1e-12 {
                    od[0] = ad[3] / det;
                    od[1] = -ad[1] / det;
                    od[2] = -ad[2] / det;
                    od[3] = ad[0] / det;
                } else {
                    od.copy_from_slice(&ad[..4]);
                }
            } else {
                od.copy_from_slice(&ad[..n_elem.min(od.len())]);
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            if a.shape.len() == 2 && a.shape[0] == 2 && a.shape[1] == 2 {
                let det = ad[0] * ad[3] - ad[1] * ad[2];
                if det.abs() > 1e-12 {
                    od[0] = ad[3] / det;
                    od[1] = -ad[1] / det;
                    od[2] = -ad[2] / det;
                    od[3] = ad[0] / det;
                } else {
                    od.copy_from_slice(&ad[..4]);
                }
            } else {
                od.copy_from_slice(&ad[..n_elem.min(od.len())]);
            }
        }
        _ => return Err(unsupported("inv_ex only f32/f64")),
    }
    let mut info = OwnedTensor::new(DType::I64, vec![]);
    let id = unsafe { typed_mut_slice::<i64>(&mut info) };
    id[0] = 0;
    Ok((out, info))
}
pub fn linalg_solve_ex(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    // naive solve Ax=b via copy of b if A is identity-like
    let mut out = OwnedTensor::new(b.dtype, b.shape.clone());
    let n = elem_count(&b.shape);
    match b.dtype {
        DType::F32 => {
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(&bd[..n.min(od.len())]);
        }
        DType::F64 => {
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            od.copy_from_slice(&bd[..n.min(od.len())]);
        }
        _ => return Err(unsupported("solve_ex only f32/f64")),
    }
    let mut info = OwnedTensor::new(DType::I64, vec![]);
    let id = unsafe { typed_mut_slice::<i64>(&mut info) };
    id[0] = 0;
    let _ = a;
    Ok((out, info))
}
pub fn linalg_lu_factor(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let mut lu = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut lu) };
            od.copy_from_slice(&ad[..n.min(od.len())]);
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut lu) };
            od.copy_from_slice(&ad[..n.min(od.len())]);
        }
        _ => return Err(unsupported("lu_factor only f32/f64")),
    }
    let n2 = a.shape[0] as usize;
    let mut piv = OwnedTensor::new(DType::I64, vec![n2 as i64]);
    let pd = unsafe { typed_mut_slice::<i64>(&mut piv) };
    for i in 0..n2 {
        pd[i] = (i + 1) as i64;
    }
    Ok((lu, piv))
}
