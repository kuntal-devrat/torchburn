//! Splitting and indexing: as_strided, broadcast, split/vsplit/hsplit/dsplit, tensor_split, take_along_dim, index_reduce, scatter_max/min. Inherits tensor_ops root imports via super; pure move.

use super::*;

pub fn as_strided(
    a: &BorrowedTensor,
    size: Vec<i64>,
    stride: Vec<i64>,
    storage_offset: usize,
) -> PyResult<OwnedTensor> {
    // create new tensor with given size/stride view semantics but copy data accordingly (simplified: copy from strided view)
    let n = size.iter().map(|&d| d.max(0) as usize).product::<usize>();
    let mut out = OwnedTensor::new(a.dtype, size.clone());
    let elem = a.dtype.elem_size();
    // For simplicity, use naive copy respecting stride+offset
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            // ad is contiguous view of original buffer; use linear indexing with stride
            for idx in 0..n {
                let mut rem = idx;
                let mut coords = vec![0usize; size.len()];
                for d in (0..size.len()).rev() {
                    let dim = size[d] as usize;
                    coords[d] = rem % dim.max(1);
                    rem /= dim.max(1);
                }
                let mut src_idx = storage_offset;
                for d in 0..size.len() {
                    src_idx += coords[d] * (stride[d] as usize);
                }
                od[idx] = if src_idx < ad.len() { ad[src_idx] } else { 0.0 };
            }
        }
        DType::F64 => {
            let ad = unsafe { typed_slice::<f64>(a) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for idx in 0..n {
                let mut rem = idx;
                let mut coords = vec![0usize; size.len()];
                for d in (0..size.len()).rev() {
                    let dim = size[d] as usize;
                    coords[d] = rem % dim.max(1);
                    rem /= dim.max(1);
                }
                let mut src_idx = storage_offset;
                for d in 0..size.len() {
                    src_idx += coords[d] * (stride[d] as usize);
                }
                od[idx] = if src_idx < ad.len() { ad[src_idx] } else { 0.0 };
            }
        }
        _ => return Err(unsupported("as_strided only f32/f64")),
    }
    let _ = elem;
    Ok(out)
}
pub fn broadcast_to(a: &BorrowedTensor, shape: Vec<i64>) -> PyResult<OwnedTensor> {
    crate::shape_ops::expand(a, &shape)
}
pub fn broadcast_tensors(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let out_shape = crate::kernels::broadcast_shape(&a.shape, &b.shape)?;
    let ea = broadcast_to(a, out_shape.clone())?;
    let eb = broadcast_to(b, out_shape.clone())?;
    Ok((ea, eb))
}
pub fn split(a: &BorrowedTensor, split_size: usize, dim: isize) -> PyResult<Vec<OwnedTensor>> {
    let rank = a.shape.len() as isize;
    let d = if dim < 0 {
        (rank + dim) as usize
    } else {
        dim as usize
    };
    if d >= a.shape.len() {
        return Err(unsupported("split dim oob"));
    }
    let dim_size = a.shape[d] as usize;
    let mut res = Vec::new();
    let mut start = 0;
    while start < dim_size {
        let len = (split_size).min(dim_size - start);
        res.push(crate::shape_ops::narrow(a, dim as isize, start, len)?);
        start += len;
    }
    Ok(res)
}
pub fn vsplit(a: &BorrowedTensor, sections: usize) -> PyResult<Vec<OwnedTensor>> {
    split(a, (a.shape[0] as usize + sections - 1) / sections, 0)
}
pub fn hsplit(a: &BorrowedTensor, sections: usize) -> PyResult<Vec<OwnedTensor>> {
    let dim = if a.shape.len() >= 2 { 1 } else { 0 };
    split(
        a,
        (a.shape[dim] as usize + sections - 1) / sections,
        dim as isize,
    )
}
pub fn dsplit(a: &BorrowedTensor, sections: usize) -> PyResult<Vec<OwnedTensor>> {
    let dim = if a.shape.len() >= 3 { 2 } else { 0 };
    split(
        a,
        (a.shape[dim] as usize + sections - 1) / sections,
        dim as isize,
    )
}
pub fn tensor_split(
    a: &BorrowedTensor,
    indices: Vec<usize>,
    dim: isize,
) -> PyResult<Vec<OwnedTensor>> {
    if indices.is_empty() {
        return Ok(vec![crate::shape_ops::to_contiguous(a)?]);
    }
    let rank = a.shape.len() as isize;
    let d = if dim < 0 {
        (rank + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = a.shape[d] as usize;
    let mut sorted = indices.clone();
    sorted.sort();
    let mut res = Vec::new();
    let mut prev = 0;
    for &idx in &sorted {
        let end = idx.min(dim_size);
        if end > prev {
            res.push(crate::shape_ops::narrow(a, dim, prev, end - prev)?);
        }
        prev = end;
    }
    if prev < dim_size {
        res.push(crate::shape_ops::narrow(a, dim, prev, dim_size - prev)?);
    }
    Ok(res)
}
pub fn take_along_dim(
    a: &BorrowedTensor,
    indices: &BorrowedTensor,
    dim: isize,
) -> PyResult<OwnedTensor> {
    // indices shape must equal output shape
    let mut out = OwnedTensor::new(a.dtype, indices.shape.clone());
    let n = elem_count(&indices.shape);
    let a_rank = a.shape.len() as isize;
    let d = if dim < 0 {
        (a_rank + dim) as usize
    } else {
        dim as usize
    };
    let a_dim = a.shape[d] as usize;
    match indices.dtype {
        DType::I64 => {
            let idx = unsafe { typed_slice::<i64>(indices) };
            match a.dtype {
                DType::F32 => {
                    let ad = unsafe { typed_slice::<f32>(a) };
                    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
                    let inner: usize = a.shape[d + 1..]
                        .iter()
                        .map(|&s| s.max(0) as usize)
                        .product();
                    let idx_dim = indices.shape[d] as usize;
                    for i in 0..n {
                        let outer_idx = i / (idx_dim * inner.max(1));
                        let inner_idx = i % inner.max(1);
                        let k = idx[i] as usize;
                        let kk = k.min(a_dim - 1);
                        let src = outer_idx * a_dim * inner + kk * inner + inner_idx;
                        od[i] = ad[src.min(ad.len() - 1)];
                    }
                }
                _ => return Err(unsupported("take_along_dim only f32 for now")),
            }
        }
        _ => return Err(unsupported("take_along_dim indices must be i64")),
    }
    Ok(out)
}
pub fn index_reduce(
    a: &BorrowedTensor,
    dim: isize,
    index: &BorrowedTensor,
    source: &BorrowedTensor,
    reduce: &str,
) -> PyResult<OwnedTensor> {
    // dest = a.clone then reduce source into dest at index
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            od.copy_from_slice(&ad[..n.min(od.len())]);
            let idx = unsafe { typed_slice::<i64>(index) };
            let src = unsafe { typed_slice::<f32>(source) };
            let d = if dim < 0 {
                (a.shape.len() as isize + dim) as usize
            } else {
                dim as usize
            };
            let inner: usize = a.shape[d + 1..]
                .iter()
                .map(|&s| s.max(0) as usize)
                .product();
            let outer: usize = a.shape[..d].iter().map(|&s| s.max(0) as usize).product();
            for i in 0..idx.len() {
                let dest_idx = idx[i] as usize;
                for k in 0..inner {
                    for o in 0..outer {
                        let dst = o * a.shape[d] as usize * inner + dest_idx * inner + k;
                        let sidx = o * source.shape[d] as usize * inner + i * inner + k;
                        if dst < od.len() && sidx < src.len() {
                            match reduce {
                                "amax" => od[dst] = od[dst].max(src[sidx]),
                                "amin" => od[dst] = od[dst].min(src[sidx]),
                                "prod" => od[dst] *= src[sidx],
                                _ => od[dst] += src[sidx],
                            }
                        }
                    }
                }
            }
        }
        _ => return Err(unsupported("index_reduce only f32")),
    }
    Ok(out)
}
pub fn scatter_max(
    a: &BorrowedTensor,
    dim: isize,
    index: &BorrowedTensor,
    src: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    index_reduce(a, dim, index, src, "amax")
}
pub fn scatter_min(
    a: &BorrowedTensor,
    dim: isize,
    index: &BorrowedTensor,
    src: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    index_reduce(a, dim, index, src, "amin")
}
