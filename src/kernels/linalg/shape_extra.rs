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

pub fn cartesian_prod(_tensors: &[BorrowedTensor]) -> PyResult<OwnedTensor> {
    Err(unsupported("cartesian_prod requires eager fallback"))
}

pub fn combinations(_a: &BorrowedTensor, _r: usize) -> PyResult<OwnedTensor> {
    Err(unsupported("combinations requires eager fallback"))
}

// ── 86-93. Padding ──
pub fn pad(input: &BorrowedTensor, pad: &[i64], mode: &str, value: f64) -> PyResult<OwnedTensor> {
    if mode != "constant" {
        return Err(unsupported(&format!(
            "pad mode '{mode}' requires eager fallback"
        )));
    }
    let ndim = input.shape.len();
    let num_padded_dims = pad.len() / 2;
    if num_padded_dims > ndim {
        return Err(unsupported("padding dimensions exceed tensor rank"));
    }
    let mut out_shape = input.shape.clone();
    for i in 0..num_padded_dims {
        let dim = ndim - 1 - i;
        out_shape[dim] += pad[2 * i] + pad[2 * i + 1];
    }
    let out_strides = crate::dlpack::contiguous_strides(&out_shape);
    let mut out = OwnedTensor::new(input.dtype, out_shape);
    match input.dtype {
        DType::F32 => {
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.fill(value as f32);
            let id = unsafe { typed_slice::<f32>(input) };
            let in_n = elem_count(&input.shape);
            let mut in_coords = vec![0usize; ndim];
            for i in 0..in_n {
                let mut rem = i;
                for d in (0..ndim).rev() {
                    let s = input.shape[d].max(1) as usize;
                    in_coords[d] = rem % s;
                    rem /= s;
                }
                let mut out_idx = 0usize;
                for d in 0..ndim {
                    let pad_before = if d >= ndim - num_padded_dims {
                        let pad_idx = (ndim - 1 - d) * 2;
                        pad[pad_idx].max(0) as usize
                    } else {
                        0
                    };
                    out_idx += (in_coords[d] + pad_before) * (out_strides[d] as usize);
                }
                if out_idx < od.len() {
                    od[out_idx] = id[i];
                }
            }
        }
        DType::F64 => {
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            od.fill(value);
            let id = unsafe { typed_slice::<f64>(input) };
            let in_n = elem_count(&input.shape);
            let mut in_coords = vec![0usize; ndim];
            for i in 0..in_n {
                let mut rem = i;
                for d in (0..ndim).rev() {
                    let s = input.shape[d].max(1) as usize;
                    in_coords[d] = rem % s;
                    rem /= s;
                }
                let mut out_idx = 0usize;
                for d in 0..ndim {
                    let pad_before = if d >= ndim - num_padded_dims {
                        let pad_idx = (ndim - 1 - d) * 2;
                        pad[pad_idx].max(0) as usize
                    } else {
                        0
                    };
                    out_idx += (in_coords[d] + pad_before) * (out_strides[d] as usize);
                }
                if out_idx < od.len() {
                    od[out_idx] = id[i];
                }
            }
        }
        _ => return Err(unsupported("pad only supports f32/f64")),
    }
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
