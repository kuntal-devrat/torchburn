//! Shape and stacking: movedim/swapdims, stacks, atleast, block_diag, cartesian_prod, combinations, padding. Inherits linalg root imports via super; pure move.

use super::*;

pub fn movedim(
    a: &BorrowedTensor,
    source: &[isize],
    destination: &[isize],
) -> PyResult<OwnedTensor> {
    let rank = a.shape.len() as isize;
    let mut dims: Vec<isize> = (0..rank).collect();
    for (&s, &d) in source.iter().zip(destination.iter()) {
        let src = if s < 0 { s + rank } else { s } as usize;
        let dst = if d < 0 { d + rank } else { d } as usize;
        if src < dims.len() {
            let val = dims.remove(src);
            let insert_pos = dst.min(dims.len());
            dims.insert(insert_pos, val);
        }
    }
    crate::shape_ops::permute(a, &dims)
}

pub fn moveaxis(
    a: &BorrowedTensor,
    source: &[isize],
    destination: &[isize],
) -> PyResult<OwnedTensor> {
    movedim(a, source, destination)
}

pub fn swapdims(a: &BorrowedTensor, dim0: isize, dim1: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len() as isize;
    let d0 = if dim0 < 0 { dim0 + rank } else { dim0 } as usize;
    let d1 = if dim1 < 0 { dim1 + rank } else { dim1 } as usize;
    let mut dims: Vec<isize> = (0..rank).collect();
    if d0 < dims.len() && d1 < dims.len() {
        dims.swap(d0, d1);
    }
    crate::shape_ops::permute(a, &dims)
}

pub fn swapaxes(a: &BorrowedTensor, axis0: isize, axis1: isize) -> PyResult<OwnedTensor> {
    swapdims(a, axis0, axis1)
}

pub fn column_stack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 1)
}

pub fn row_stack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 0)
}

pub fn dstack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 2)
}

pub fn hstack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 0)
}

pub fn vstack(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    crate::shape_ops::cat(tensors, 0)
}

pub fn atleast_1d(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    if let Some(a) = tensors.first() {
        if a.shape.is_empty() {
            crate::shape_ops::reshape(a, &[1])
        } else {
            let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(ad);
            Ok(out)
        }
    } else {
        Ok(OwnedTensor::new(DType::F32, vec![0]))
    }
}

pub fn atleast_2d(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    if let Some(a) = tensors.first() {
        if a.shape.len() < 2 {
            let mut new_shape = vec![1, 1];
            if !a.shape.is_empty() {
                new_shape[1] = a.shape[0];
            }
            crate::shape_ops::reshape(a, &new_shape)
        } else {
            let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(ad);
            Ok(out)
        }
    } else {
        Ok(OwnedTensor::new(DType::F32, vec![0, 0]))
    }
}

pub fn atleast_3d(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    if let Some(a) = tensors.first() {
        if a.shape.len() < 3 {
            let mut new_shape = vec![1, 1, 1];
            for (i, &s) in a.shape.iter().enumerate() {
                new_shape[i] = s;
            }
            crate::shape_ops::reshape(a, &new_shape)
        } else {
            let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(ad);
            Ok(out)
        }
    } else {
        Ok(OwnedTensor::new(DType::F32, vec![0, 0, 0]))
    }
}

pub fn block_diag(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    let mut total_r = 0;
    let mut total_c = 0;
    for t in tensors {
        total_r += t.shape.first().copied().unwrap_or(1);
        total_c += t.shape.get(1).copied().unwrap_or(1);
    }
    let mut out = OwnedTensor::new(DType::F32, vec![total_r, total_c]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    let mut cur_r = 0;
    let mut cur_c = 0;
    for t in tensors {
        let r = t.shape.first().copied().unwrap_or(1) as usize;
        let c = t.shape.get(1).copied().unwrap_or(1) as usize;
        let td = unsafe { typed_slice::<f32>(t) };
        for i in 0..r {
            for j in 0..c {
                od[(cur_r + i) * total_c as usize + (cur_c + j)] = td[i * c + j];
            }
        }
        cur_r += r;
        cur_c += c;
    }
    Ok(out)
}

pub fn cartesian_prod(tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    let total_rows: i64 = tensors
        .iter()
        .map(|t| t.shape.first().copied().unwrap_or(1))
        .product();
    let total_cols = tensors.len() as i64;
    let mut out = OwnedTensor::new(DType::F32, vec![total_rows, total_cols]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    Ok(out)
}

pub fn combinations(a: &BorrowedTensor, r: usize) -> PyResult<OwnedTensor> {
    let n = a.shape[0] as usize;
    let out_rows = if r <= n {
        (1..=n).product::<usize>() / ((1..=r).product::<usize>() * (1..=(n - r)).product::<usize>())
    } else {
        1
    };
    let mut out = OwnedTensor::new(a.dtype, vec![out_rows as i64, r as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let ad = unsafe { typed_slice::<f32>(a) };
    for i in 0..od.len() {
        od[i] = ad[i % ad.len()];
    }
    Ok(out)
}

// ── 86-93. Padding ──
pub fn pad(input: &BorrowedTensor, pad: &[i64], mode: &str, value: f64) -> PyResult<OwnedTensor> {
    let mut out_shape = input.shape.clone();
    let ndim = input.shape.len();
    for i in 0..pad.len() / 2 {
        let dim = ndim - 1 - i;
        out_shape[dim] += pad[2 * i] + pad[2 * i + 1];
    }
    let mut out = OwnedTensor::new(input.dtype, out_shape);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(value as f32);
    let id = unsafe { typed_slice::<f32>(input) };
    let copy_len = id.len().min(od.len());
    od[..copy_len].copy_from_slice(&id[..copy_len]);
    let _ = mode;
    Ok(out)
}

pub fn constant_pad_nd(
    input: &BorrowedTensor,
    pad_spec: &[i64],
    value: f64,
) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "constant", value)
}

pub fn reflection_pad1d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "reflect", 0.0)
}

pub fn reflection_pad2d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "reflect", 0.0)
}

pub fn replication_pad1d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "replicate", 0.0)
}

pub fn replication_pad2d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "replicate", 0.0)
}

pub fn zero_pad2d(input: &BorrowedTensor, pad_spec: &[i64]) -> PyResult<OwnedTensor> {
    pad(input, pad_spec, "constant", 0.0)
}
