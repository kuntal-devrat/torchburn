//! BLAS-like extensions: addr/outer/ger, mv/vdot, addbmm/addmv, kron/inner. Inherits linalg root imports via super; pure move.

use super::*;

// ── 31-35. addcdiv, addcmul, addr, ger, outer, mv, vdot ──
pub fn addcdiv(
    input: &BorrowedTensor,
    tensor1: &BorrowedTensor,
    tensor2: &BorrowedTensor,
    value: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let t1 = unsafe { typed_slice::<f32>(tensor1) };
            let t2 = unsafe { typed_slice::<f32>(tensor2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            let v = value as f32;
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + v * (t1[i % t1_l] / t2[i % t2_l]);
            }
        }
        DType::F64 => {
            let inp = unsafe { typed_slice::<f64>(input) };
            let t1 = unsafe { typed_slice::<f64>(tensor1) };
            let t2 = unsafe { typed_slice::<f64>(tensor2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + value * (t1[i % t1_l] / t2[i % t2_l]);
            }
        }
        _ => return Err(unsupported("addcdiv only supports f32/f64")),
    }
    Ok(out)
}

pub fn addcmul(
    input: &BorrowedTensor,
    tensor1: &BorrowedTensor,
    tensor2: &BorrowedTensor,
    value: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let t1 = unsafe { typed_slice::<f32>(tensor1) };
            let t2 = unsafe { typed_slice::<f32>(tensor2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            let v = value as f32;
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + v * (t1[i % t1_l] * t2[i % t2_l]);
            }
        }
        DType::F64 => {
            let inp = unsafe { typed_slice::<f64>(input) };
            let t1 = unsafe { typed_slice::<f64>(tensor1) };
            let t2 = unsafe { typed_slice::<f64>(tensor2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let t1_l = t1.len().max(1);
            let t2_l = t2.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = inp[i] + value * (t1[i % t1_l] * t2[i % t2_l]);
            }
        }
        _ => return Err(unsupported("addcmul only supports f32/f64")),
    }
    Ok(out)
}

pub fn addr(
    input: &BorrowedTensor,
    vec1: &BorrowedTensor,
    vec2: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let r = vec1.shape[0] as usize;
    let c = vec2.shape[0] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![r as i64, c as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let v1 = unsafe { typed_slice::<f32>(vec1) };
            let v2 = unsafe { typed_slice::<f32>(vec2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let b = beta as f32;
            let inp_l = inp.len().max(1);
            for i in 0..r {
                for j in 0..c {
                    let idx = i * c + j;
                    od[idx] = b * inp[idx % inp_l] + a * (v1[i] * v2[j]);
                }
            }
        }
        DType::F64 => {
            let inp = unsafe { typed_slice::<f64>(input) };
            let v1 = unsafe { typed_slice::<f64>(vec1) };
            let v2 = unsafe { typed_slice::<f64>(vec2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let inp_l = inp.len().max(1);
            for i in 0..r {
                for j in 0..c {
                    let idx = i * c + j;
                    od[idx] = beta * inp[idx % inp_l] + alpha * (v1[i] * v2[j]);
                }
            }
        }
        _ => return Err(unsupported("addr only supports f32/f64")),
    }
    Ok(out)
}

pub fn outer(vec1: &BorrowedTensor, vec2: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let r = vec1.shape[0] as usize;
    let c = vec2.shape[0] as usize;
    let mut out = OwnedTensor::new(vec1.dtype, vec![r as i64, c as i64]);
    match vec1.dtype {
        DType::F32 => {
            let v1 = unsafe { typed_slice::<f32>(vec1) };
            let v2 = unsafe { typed_slice::<f32>(vec2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..r {
                for j in 0..c {
                    od[i * c + j] = v1[i] * v2[j];
                }
            }
        }
        DType::F64 => {
            let v1 = unsafe { typed_slice::<f64>(vec1) };
            let v2 = unsafe { typed_slice::<f64>(vec2) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..r {
                for j in 0..c {
                    od[i * c + j] = v1[i] * v2[j];
                }
            }
        }
        _ => return Err(unsupported("outer only supports f32/f64")),
    }
    Ok(out)
}

pub fn ger(vec1: &BorrowedTensor, vec2: &BorrowedTensor) -> PyResult<OwnedTensor> {
    outer(vec1, vec2)
}

pub fn mv(mat: &BorrowedTensor, vec: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if mat.shape.len() != 2 || vec.shape.len() != 1 {
        return Err(unsupported("mv requires 2D matrix and 1D vector"));
    }
    let r = mat.shape[0] as usize;
    let c = mat.shape[1] as usize;
    let mut out = OwnedTensor::new(mat.dtype, vec![r as i64]);
    match mat.dtype {
        DType::F32 => {
            let m = unsafe { typed_slice::<f32>(mat) };
            let v = unsafe { typed_slice::<f32>(vec) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..r {
                let mut sum = 0.0_f32;
                for j in 0..c {
                    sum += m[i * c + j] * v[j];
                }
                od[i] = sum;
            }
        }
        DType::F64 => {
            let m = unsafe { typed_slice::<f64>(mat) };
            let v = unsafe { typed_slice::<f64>(vec) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..r {
                let mut sum = 0.0_f64;
                for j in 0..c {
                    sum += m[i * c + j] * v[j];
                }
                od[i] = sum;
            }
        }
        _ => return Err(unsupported("mv only supports f32/f64")),
    }
    Ok(out)
}

pub fn vdot(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut sum = 0.0_f32;
            for i in 0..n.min(ad.len()).min(bd.len()) {
                sum += ad[i] * bd[i];
            }
            od[0] = sum;
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let bd = unsafe { typed_slice::<f64>(b) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let mut sum = 0.0_f64;
            for i in 0..n.min(ad.len()).min(bd.len()) {
                sum += ad[i] * bd[i];
            }
            od[0] = sum;
        }
        _ => return Err(unsupported("vdot only supports f32/f64")),
    }
    Ok(out)
}

// ── 36-40. baddbmm, addbmm, addmv, kron, inner ──
pub fn baddbmm(
    input: &BorrowedTensor,
    batch1: &BorrowedTensor,
    batch2: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let b = batch1.shape[0] as usize;
    let n = batch1.shape[1] as usize;
    let m = batch1.shape[2] as usize;
    let p = batch2.shape[2] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, n as i64, p as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let b1 = unsafe { typed_slice::<f32>(batch1) };
            let b2 = unsafe { typed_slice::<f32>(batch2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let bt = beta as f32;
            let inp_l = inp.len().max(1);
            for bi in 0..b {
                for ni in 0..n {
                    for pi in 0..p {
                        let mut dot = 0.0_f32;
                        for mi in 0..m {
                            dot += b1[(bi * n + ni) * m + mi] * b2[(bi * m + mi) * p + pi];
                        }
                        let out_idx = (bi * n + ni) * p + pi;
                        od[out_idx] = bt * inp[out_idx % inp_l] + a * dot;
                    }
                }
            }
        }
        _ => return Err(unsupported("baddbmm only supports f32")),
    }
    Ok(out)
}

pub fn addbmm(
    input: &BorrowedTensor,
    batch1: &BorrowedTensor,
    batch2: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let b = batch1.shape[0] as usize;
    let n = batch1.shape[1] as usize;
    let m = batch1.shape[2] as usize;
    let p = batch2.shape[2] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![n as i64, p as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let b1 = unsafe { typed_slice::<f32>(batch1) };
            let b2 = unsafe { typed_slice::<f32>(batch2) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let bt = beta as f32;
            let inp_l = inp.len().max(1);
            for ni in 0..n {
                for pi in 0..p {
                    let mut sum_batches = 0.0_f32;
                    for bi in 0..b {
                        for mi in 0..m {
                            sum_batches += b1[(bi * n + ni) * m + mi] * b2[(bi * m + mi) * p + pi];
                        }
                    }
                    let out_idx = ni * p + pi;
                    od[out_idx] = bt * inp[out_idx % inp_l] + a * sum_batches;
                }
            }
        }
        _ => return Err(unsupported("addbmm only supports f32")),
    }
    Ok(out)
}

pub fn addmv(
    input: &BorrowedTensor,
    mat: &BorrowedTensor,
    vec: &BorrowedTensor,
    beta: f64,
    alpha: f64,
) -> PyResult<OwnedTensor> {
    let r = mat.shape[0] as usize;
    let c = mat.shape[1] as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![r as i64]);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let m = unsafe { typed_slice::<f32>(mat) };
            let v = unsafe { typed_slice::<f32>(vec) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let a = alpha as f32;
            let b = beta as f32;
            let inp_l = inp.len().max(1);
            for i in 0..r {
                let mut dot = 0.0_f32;
                for j in 0..c {
                    dot += m[i * c + j] * v[j];
                }
                od[i] = b * inp[i % inp_l] + a * dot;
            }
        }
        _ => return Err(unsupported("addmv only supports f32")),
    }
    Ok(out)
}

pub fn kron(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out_shape = Vec::new();
    let max_len = a.shape.len().max(b.shape.len());
    let a_padded: Vec<i64> = std::iter::repeat(1)
        .take(max_len.saturating_sub(a.shape.len()))
        .chain(a.shape.iter().copied())
        .collect();
    let b_padded: Vec<i64> = std::iter::repeat(1)
        .take(max_len.saturating_sub(b.shape.len()))
        .chain(b.shape.iter().copied())
        .collect();
    for i in 0..max_len {
        out_shape.push(a_padded[i] * b_padded[i]);
    }
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let bd = unsafe { typed_slice::<f32>(b) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let b_len = bd.len();
            for i in 0..ad.len() {
                for j in 0..b_len {
                    od[i * b_len + j] = ad[i] * bd[j];
                }
            }
        }
        _ => return Err(unsupported("kron only supports f32")),
    }
    Ok(out)
}

pub fn inner(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    vdot(a, b)
}
