//! Shape and indexing operations: cat, stack, reshape, permute, expand,
//! index_select, gather, where, masked_fill, repeat, flip, narrow.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, contiguous_strides, elem_count, unsupported};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ---------------------------------------------------------------------------
// Cat — concatenate tensors along a dimension
// ---------------------------------------------------------------------------

pub fn cat(tensors: &[BorrowedTensor], dim: isize) -> PyResult<OwnedTensor> {
    if tensors.is_empty() {
        return Err(unsupported("cat: no tensors provided"));
    }
    let rank = tensors[0].shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if d >= rank {
        return Err(unsupported(&format!("cat: dim {} out of range for rank {}", d, rank)));
    }

    // Verify all tensors have same shape except along dim, and same dtype
    let dtype = tensors[0].dtype;
    let mut out_shape = tensors[0].shape.clone();
    for t in tensors[1..].iter() {
        if t.shape.len() != rank {
            return Err(unsupported("cat: all tensors must have same rank"));
        }
        if t.dtype != dtype {
            return Err(unsupported("cat: all tensors must have the same dtype"));
        }
        for i in 0..rank {
            if i != d && t.shape[i] != tensors[0].shape[i] {
                return Err(unsupported("cat: shapes mismatch outside concat dim"));
            }
        }
        out_shape[d] += t.shape[d];
    }

    let elem_size = dtype.elem_size();
    let mut out = OwnedTensor::new(dtype, out_shape.clone());

    // outer = product of dims 0..d
    // inner = product of dims d+1..rank
    let outer: usize = out_shape[..d].iter().map(|&s| s.max(0) as usize).product();
    let inner: usize = out_shape[d+1..].iter().map(|&s| s.max(0) as usize).product();

    // For each outer index, copy slices from each tensor one after another along dim d
    let mut dim_offset = 0usize; // cumulative offset in the output along dim d
    for t in tensors {
        let t_dim_size = t.shape[d] as usize;
        let t_contig = t.strides == contiguous_strides(&t.shape);

        if t_contig {
            let chunk_bytes = inner * elem_size;
            for o in 0..outer {
                for t_i in 0..t_dim_size {
                    let out_idx = o * out_shape[d] as usize * inner + (dim_offset + t_i) * inner;
                    let src_idx = o * t_dim_size * inner + t_i * inner;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (t.data as *const u8).add(src_idx * elem_size),
                            (out.data.as_mut_ptr() as *mut u8).add(out_idx * elem_size),
                            chunk_bytes,
                        );
                    }
                }
            }
        } else {
            for o in 0..outer {
                for t_i in 0..t_dim_size {
                    for inn in 0..inner {
                        let out_idx = o * out_shape[d] as usize * inner + (dim_offset + t_i) * inner + inn;
                        // Compute strided source index using the tensor's actual strides.
                        // Decompose o → outer per-dim coords, t_i → dim-d coord, inn → inner per-dim coords.
                        let mut src_flat = 0usize;
                        // Decode outer index into per-dim coords
                        let mut rem_o = o;
                        let mut outer_coords = vec![0usize; d];
                        for dd in (0..d).rev() {
                            outer_coords[dd] = rem_o % (t.shape[dd].max(1) as usize);
                            rem_o /= t.shape[dd].max(1) as usize;
                        }
                        for dd in 0..d {
                            src_flat += outer_coords[dd] * t.strides[dd] as usize;
                        }
                        src_flat += t_i * t.strides[d] as usize;
                        // Decode inner index into per-dim coords
                        let mut rem_i = inn;
                        let mut inner_coords = vec![0usize; rank - d - 1];
                        for dd in (0..rank - d - 1).rev() {
                            inner_coords[dd] = rem_i % (t.shape[d + 1 + dd].max(1) as usize);
                            rem_i /= t.shape[d + 1 + dd].max(1) as usize;
                        }
                        for dd in 0..rank - d - 1 {
                            src_flat += inner_coords[dd] * t.strides[d + 1 + dd] as usize;
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                (t.data as *const u8).add(src_flat * elem_size),
                                (out.data.as_mut_ptr() as *mut u8).add(out_idx * elem_size),
                                elem_size,
                            );
                        }
                    }
                }
            }
        }
        dim_offset += t_dim_size;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Stack — concatenate along a new dimension
// ---------------------------------------------------------------------------

pub fn stack(tensors: &[BorrowedTensor], dim: isize) -> PyResult<OwnedTensor> {
    if tensors.is_empty() {
        return Err(unsupported("stack: no tensors provided"));
    }
    let rank = tensors[0].shape.len();
    let d = if dim < 0 { (rank as isize + dim + 1) as usize } else { dim as usize };

    // Unsqueeze each tensor, then cat
    let mut unsqueezed = Vec::with_capacity(tensors.len());
    for t in tensors {
        let mut new_shape = t.shape.clone();
        new_shape.insert(d, 1);
        let strides = contiguous_strides(&new_shape);
        unsqueezed.push(BorrowedTensor {
            data: t.data,
            shape: new_shape,
            strides,
            dtype: t.dtype,
        });
    }
    cat(&unsqueezed, d as isize)
}

// ---------------------------------------------------------------------------
// Flatten — merge dims [start, end] into one (torch.flatten semantics)
// ---------------------------------------------------------------------------

fn norm_dim(dim: isize, rank: usize) -> usize {
    if dim < 0 { (rank as isize + dim).max(0) as usize } else { dim as usize }
}

pub fn flatten(a: &BorrowedTensor, start_dim: isize, end_dim: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    if rank == 0 {
        return Err(unsupported("flatten: cannot flatten a 0-D tensor"));
    }
    let s = norm_dim(start_dim, rank);
    let e = norm_dim(end_dim, rank);
    if s > e || e >= rank {
        return Err(unsupported(&format!(
            "flatten: invalid dims [{s}, {e}] for rank {rank}"
        )));
    }
    let mut out_shape: Vec<i64> = Vec::with_capacity(rank - (e - s));
    out_shape.extend_from_slice(&a.shape[..s]);
    let merged: i64 = a.shape[s..=e].iter().map(|&d| d.max(0)).product();
    out_shape.push(merged);
    out_shape.extend_from_slice(&a.shape[e + 1..]);
    reshape(a, &out_shape)
}

// ---------------------------------------------------------------------------
// Reshape — change shape, preserve data
// ---------------------------------------------------------------------------

pub fn resolve_shape(old_shape: &[i64], new_shape: &[i64]) -> PyResult<Vec<i64>> {
    let old_size = elem_count(old_shape) as i64;

    // Resolve any -1 dim (at most one allowed)
    let neg_count = new_shape.iter().filter(|&&d| d < 0).count();
    if neg_count > 1 {
        return Err(unsupported("reshape: at most one dimension can be -1"));
    }
    let resolved: Vec<i64> = if neg_count == 1 {
        let known: i64 = new_shape.iter().filter(|&&d| d >= 0).product();
        if known == 0 {
            return Err(unsupported("reshape: cannot infer -1 dim when other dims include 0"));
        }
        if old_size % known != 0 {
            return Err(unsupported(&format!(
                "reshape: size {} not divisible by known dims product {}",
                old_size, known
            )));
        }
        new_shape.iter().map(|&d| if d < 0 { old_size / known } else { d }).collect()
    } else {
        new_shape.to_vec()
    };

    let new_size: usize = resolved.iter().map(|&d| d.max(0) as usize).product();
    if old_size as usize != new_size {
        return Err(unsupported(&format!(
            "reshape: size mismatch {} -> {}",
            old_size, new_size
        )));
    }
    Ok(resolved)
}

pub fn reshape(a: &BorrowedTensor, new_shape: &[i64]) -> PyResult<OwnedTensor> {
    let resolved = resolve_shape(&a.shape, new_shape)?;
    let old_size = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, resolved);
    let bytes = old_size * a.dtype.elem_size();
    // If contiguous, bitwise copy suffices; otherwise materialize via index walk.
    if a.strides == contiguous_strides(&a.shape) {
        unsafe {
            std::ptr::copy_nonoverlapping(a.data, out.data.as_mut_ptr() as *mut u8, bytes);
        }
    } else {
        // Non-contiguous: copy element-by-element in logical order
        match a.dtype {
            DType::F32 => {
                let src = unsafe { typed_slice::<f32>(a) };
                let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
                let n = elem_count(&a.shape);
                let rank = a.shape.len();
                let mut coords = vec![0usize; rank];
                for i in 0..n {
                    let mut rem = i;
                    for d in (0..rank).rev() {
                        coords[d] = rem % (a.shape[d].max(1) as usize);
                        rem /= a.shape[d].max(1) as usize;
                    }
                    let mut ai = 0usize;
                    for d in 0..rank { ai += coords[d] * a.strides[d] as usize; }
                    dst[i] = src[ai];
                }
            }
            DType::F64 => {
                let src = unsafe { typed_slice::<f64>(a) };
                let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
                let n = elem_count(&a.shape);
                let rank = a.shape.len();
                let mut coords = vec![0usize; rank];
                for i in 0..n {
                    let mut rem = i;
                    for d in (0..rank).rev() {
                        coords[d] = rem % (a.shape[d].max(1) as usize);
                        rem /= a.shape[d].max(1) as usize;
                    }
                    let mut ai = 0usize;
                    for d in 0..rank { ai += coords[d] * a.strides[d] as usize; }
                    dst[i] = src[ai];
                }
            }

            DType::I64 | DType::I32 | DType::Bool => {
                return Err(unsupported("this kernel only supports f32/f64 tensors"));
            }

        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Permute — reorder dimensions
// ---------------------------------------------------------------------------

/// Materialize a borrowed tensor into an owned contiguous tensor.
pub fn to_contiguous(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    reshape(a, &a.shape.clone())
}

/// Compute (shape, strides) for a permute view without data copying.
pub fn permute_view(a: &BorrowedTensor, dims: &[isize]) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let rank = a.shape.len();
    if dims.len() != rank {
        return Err(unsupported(&format!("permute: dims length {} != rank {}", dims.len(), rank)));
    }
    let new_shape: Vec<i64> = dims.iter().map(|&d| {
        let dd = if d < 0 { (rank as isize + d) as usize } else { d as usize };
        a.shape[dd]
    }).collect();
    let new_strides: Vec<i64> = dims.iter().map(|&d| {
        let dd = if d < 0 { (rank as isize + d) as usize } else { d as usize };
        a.strides[dd]
    }).collect();
    Ok((new_shape, new_strides))
}

/// Compute (shape, strides) for a transpose view without data copying.
pub fn transpose_view(a: &BorrowedTensor, d0: isize, d1: isize) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let rank = a.shape.len();
    let d0_norm = if d0 < 0 { rank as isize + d0 } else { d0 } as usize;
    let d1_norm = if d1 < 0 { rank as isize + d1 } else { d1 } as usize;
    if d0_norm >= rank || d1_norm >= rank {
        return Err(unsupported(&format!(
            "transpose: dims {d0},{d1} out of range for rank {rank}"
        )));
    }
    let mut dims: Vec<isize> = (0..rank as isize).collect();
    dims.swap(d0_norm, d1_norm);
    permute_view(a, &dims)
}

/// Compute (shape, strides) for a squeeze view without data copying.
pub fn squeeze_view(a: &BorrowedTensor, dim: isize) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let rank = a.shape.len();
    if rank == 0 {
        return Ok((a.shape.clone(), a.strides.clone()));
    }
    let norm_dim = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if norm_dim >= rank {
        return Err(unsupported(&format!("squeeze: dim {} out of range for rank {}", dim, rank)));
    }
    if a.shape[norm_dim] == 1 {
        let mut new_shape = a.shape.clone();
        let mut new_strides = a.strides.clone();
        new_shape.remove(norm_dim);
        new_strides.remove(norm_dim);
        Ok((new_shape, new_strides))
    } else {
        Ok((a.shape.clone(), a.strides.clone()))
    }
}

/// Compute (shape, strides) for an unsqueeze view without data copying.
pub fn unsqueeze_view(a: &BorrowedTensor, dim: isize) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let rank = a.shape.len();
    let norm_dim = if dim < 0 { ((rank + 1) as isize + dim) as usize } else { dim as usize };
    if norm_dim > rank {
        return Err(unsupported(&format!("unsqueeze: dim {} out of range for rank {}", dim, rank)));
    }
    let mut new_shape = a.shape.clone();
    let mut new_strides = a.strides.clone();
    let stride = if norm_dim < rank {
        new_strides[norm_dim] * new_shape[norm_dim]
    } else {
        1
    };
    new_shape.insert(norm_dim, 1);
    new_strides.insert(norm_dim, stride);
    Ok((new_shape, new_strides))
}

/// Swap two dims (aten.transpose semantics), expressed as a full permute.
pub fn transpose(a: &BorrowedTensor, d0: isize, d1: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    let d0 = if d0 < 0 { rank as isize + d0 } else { d0 } as usize;
    let d1 = if d1 < 0 { rank as isize + d1 } else { d1 } as usize;
    if d0 >= rank || d1 >= rank {
        return Err(unsupported(&format!(
            "transpose: dims {d0},{d1} out of range for rank {rank}"
        )));
    }
    let mut dims: Vec<isize> = (0..rank as isize).collect();
    dims.swap(d0, d1);
    permute(a, &dims)
}

pub fn permute(a: &BorrowedTensor, dims: &[isize]) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    if dims.len() != rank {
        return Err(unsupported(&format!("permute: dims length {} != rank {}", dims.len(), rank)));
    }

    let new_shape: Vec<i64> = dims.iter().map(|&d| {
        let dd = if d < 0 { (rank as isize + d) as usize } else { d as usize };
        a.shape[dd]
    }).collect();

    let new_strides: Vec<i64> = dims.iter().map(|&d| {
        let dd = if d < 0 { (rank as isize + d) as usize } else { d as usize };
        a.strides[dd]
    }).collect();

    // Copy data with new stride layout
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, new_shape.clone());
    let out_shape = new_shape;

    match a.dtype {
        DType::F32 => {
            let src = unsafe { typed_slice::<f32>(a) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                // Compute source index using original strides
                let mut src_idx = 0;
                let mut rem = i;
                for dd in (0..rank).rev() {
                    let dim_size = out_shape[dd] as usize;
                    src_idx += (rem % dim_size) * new_strides[dd] as usize;
                    rem /= dim_size;
                }
                dst[i] = src[src_idx];
            }
        }
        DType::F64 => {
            let src = unsafe { typed_slice::<f64>(a) };
            let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                let mut src_idx = 0;
                let mut rem = i;
                for dd in (0..rank).rev() {
                    let dim_size = out_shape[dd] as usize;
                    src_idx += (rem % dim_size) * new_strides[dd] as usize;
                    rem /= dim_size;
                }
                dst[i] = src[src_idx];
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// t — transpose the last two dims (aten.t), 2D-only per torch semantics
// ---------------------------------------------------------------------------

pub fn t(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    if rank != 2 {
        return Err(unsupported(&format!("t: expected 2D tensor, got rank {}", rank)));
    }
    permute(a, &[1, 0])
}

// ---------------------------------------------------------------------------
// Expand — broadcast to a larger shape
// ---------------------------------------------------------------------------

pub fn expand(a: &BorrowedTensor, target_shape: &[i64]) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    let target_rank = target_shape.len();
    if target_rank < rank {
        return Err(unsupported("expand: target shape has fewer dims than input"));
    }

    // Pad input shape with 1s on the left
    let mut padded_shape = vec![1i64; target_rank - rank];
    padded_shape.extend_from_slice(&a.shape);

    // Resolve -1 dimensions: -1 means "keep the source size"
    let resolved_shape: Vec<i64> = target_shape.iter().enumerate().map(|(i, &d)| {
        if d == -1 {
            padded_shape[i]
        } else {
            d
        }
    }).collect();

    // Verify broadcast compatibility
    for i in 0..target_rank {
        if resolved_shape[i] != padded_shape[i] && padded_shape[i] != 1 {
            return Err(unsupported(&format!(
                "expand: cannot expand dim {} from {} to {}",
                i, padded_shape[i], resolved_shape[i]
            )));
        }
    }

    let total: usize = resolved_shape.iter().map(|&d| d.max(0) as usize).product();
    let mut out = OwnedTensor::new(a.dtype, resolved_shape.clone());

    // Materialize the expansion: decompose each output linear index into
    // per-dimension coordinates, then map to the source using padded strides.
    // For dimensions padded on the left (input rank < target rank), the
    // source coordinate is always 0 (broadcast).  For dimensions where the
    // source size is 1, the source coordinate is also clamped to 0.
    let pad = target_rank - rank;

    macro_rules! expand_typed {
        ($T:ty) => {{
            let src = unsafe { typed_slice::<$T>(a) };
            let dst = unsafe { typed_mut_slice::<$T>(&mut out) };
            let mut coords = vec![0usize; target_rank];
            for i in 0..total {
                // Decode i into per-dimension output coordinates.
                let mut rem = i;
                for d in (0..target_rank).rev() {
                    coords[d] = rem % (resolved_shape[d].max(1) as usize);
                    rem /= resolved_shape[d].max(1) as usize;
                }
                // Map output coords → source linear index.
                let mut src_flat = 0usize;
                for d in 0..target_rank {
                    if d < pad {
                        // Padded dim — always broadcast from index 0.
                        continue;
                    }
                    let sd = d - pad; // dimension in input
                    let src_size = a.shape[sd] as usize;
                    if src_size > 1 {
                        src_flat += coords[d] * a.strides[sd] as usize;
                    }
                    // src_size == 1 → broadcast, src_flat unchanged for this dim.
                }
                dst[i] = src[src_flat];
            }
        }};
    }

    match a.dtype {
        DType::F32 => expand_typed!(f32),
        DType::F64 => expand_typed!(f64),
        DType::I64 => expand_typed!(i64),
        DType::I32 => expand_typed!(i32),
        DType::Bool => expand_typed!(u8),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Where — elementwise select between two tensors by a condition
// ---------------------------------------------------------------------------

pub fn where_op(condition: &BorrowedTensor, x: &BorrowedTensor, y: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // condition is f32 (1.0 = true, 0.0 = false) OR bool (0/1 bytes)
    let out_shape = crate::ops::broadcast_shape(&crate::ops::broadcast_shape(&condition.shape, &x.shape)?, &y.shape)?;
    let mut out = OwnedTensor::new(x.dtype, out_shape.clone());
    let n = elem_count(&out_shape);

    // Predicate helper: value of the condition at broadcast position i.
    // Bool conditions are 1-byte each; f32/f64 are non-zero float tests.
    let cond_len = elem_count(&condition.shape);
    let cond_bool = condition.dtype == DType::Bool;
    let cond_bytes = unsafe { typed_slice::<u8>(condition) };
    let cond_f32 = unsafe { typed_slice::<f32>(condition) };
    let cond_true = |i: usize| -> bool {
        if cond_bool {
            cond_bytes[i % cond_len] != 0
        } else {
            cond_f32[i % cond_len] != 0.0
        }
    };

    match x.dtype {
        DType::F32 => {
            let x_data = unsafe { typed_slice::<f32>(x) };
            let y_data = unsafe { typed_slice::<f32>(y) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                out_data[i] = if cond_true(i) {
                    x_data[i % x_data.len()]
                } else {
                    y_data[i % y_data.len()]
                };
            }
        }
        DType::F64 => {
            let x_data = unsafe { typed_slice::<f64>(x) };
            let y_data = unsafe { typed_slice::<f64>(y) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                out_data[i] = if cond_true(i) {
                    x_data[i % x_data.len()]
                } else {
                    y_data[i % y_data.len()]
                };
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Masked Fill — fill elements where mask is nonzero
// ---------------------------------------------------------------------------

pub fn masked_fill(a: &BorrowedTensor, mask: &BorrowedTensor, value: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let n = a.elem_count();

    // Mask may be bool (1 byte per element) or f32/f64 (non-zero = true).
    let mask_bool = mask.dtype == DType::Bool;
    let mask_len = elem_count(&mask.shape);
    let mask_bytes = unsafe { typed_slice::<u8>(mask) };
    let mask_f32 = unsafe { typed_slice::<f32>(mask) };
    let mask_true = |i: usize| -> bool {
        if mask_bool {
            mask_bytes[i % mask_len] != 0
        } else {
            mask_f32[i % mask_len] != 0.0
        }
    };

    match a.dtype {
        DType::F32 => {
            let src = unsafe { typed_slice::<f32>(a) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                dst[i] = if mask_true(i) { value as f32 } else { src[i] };
            }
        }
        DType::F64 => {
            let src = unsafe { typed_slice::<f64>(a) };
            let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                dst[i] = if mask_true(i) { value } else { src[i] };
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Flip — reverse elements along the given dims
// ---------------------------------------------------------------------------

pub fn flip(a: &BorrowedTensor, dims: &[isize]) -> PyResult<OwnedTensor> {
    let a_contig;
    let a = if a.strides == contiguous_strides(&a.shape) {
        a
    } else {
        a_contig = to_contiguous(a)?;
        &a_contig.as_view()
    };
    let rank = a.shape.len();
    let mut flip_dims = Vec::new();
    for &d in dims {
        let dd = if d < 0 { (rank as isize + d) as usize } else { d as usize };
        if dd >= rank {
            return Err(unsupported(&format!("flip: dim {d} out of range for rank {rank}")));
        }
        flip_dims.push(dd);
    }

    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let out_shape = a.shape.clone();
    let elem = a.dtype.elem_size();

    for i in 0..n {
        let mut rem = i;
        let mut coords = vec![0usize; rank];
        for k in (0..rank).rev() {
            let s = out_shape[k].max(0) as usize;
            coords[k] = rem % s;
            rem /= s;
        }
        let mut src_coords = coords.clone();
        for &d in &flip_dims {
            src_coords[d] = out_shape[d].max(0) as usize - 1 - coords[d];
        }
        let mut src = 0usize;
        for k in 0..rank {
            src = src * a.shape[k].max(0) as usize + src_coords[k];
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                (a.data as *const u8).add(src * elem),
                (out.data.as_mut_ptr() as *mut u8).add(i * elem),
                elem,
            );
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Index Select — select indices along a dim
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Narrow — extract a slice along a dim
// ---------------------------------------------------------------------------

pub fn narrow(a: &BorrowedTensor, dim: isize, start: usize, length: usize) -> PyResult<OwnedTensor> {
    let a_contig;
    let a = if a.strides == contiguous_strides(&a.shape) {
        a
    } else {
        a_contig = to_contiguous(a)?;
        &a_contig.as_view()
    };
    let rank = a.shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if d >= rank {
        return Err(unsupported("narrow: dim out of range"));
    }

    let mut out_shape = a.shape.clone();
    out_shape[d] = length as i64;

    // Copy the slice
    let mut out = OwnedTensor::new(a.dtype, out_shape);
    let elem_size = a.dtype.elem_size();
    let outer: usize = a.shape[..d].iter().map(|&s| s.max(0) as usize).product();
    let inner: usize = a.shape[d+1..].iter().map(|&s| s.max(0) as usize).product();

    for o in 0..outer {
        for i in 0..length {
            for inn in 0..inner {
                let src_idx = o * a.shape[d] as usize * inner + (start + i) * inner + inn;
                let dst_idx = o * length * inner + i * inner + inn;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (a.data as *const u8).add(src_idx * elem_size),
                        (out.data.as_mut_ptr() as *mut u8).add(dst_idx * elem_size),
                        elem_size,
                    );
                }
            }
        }
    }
    Ok(out)
}

/// select(dim, index): drop dim ``dim`` at position ``index`` (aten.select).
/// Equivalent to ``narrow(dim, index, 1)`` followed by ``squeeze(dim)``.
pub fn select(a: &BorrowedTensor, dim: isize, index: usize) -> PyResult<OwnedTensor> {
    let a_contig;
    let a = if a.strides == contiguous_strides(&a.shape) {
        a
    } else {
        a_contig = to_contiguous(a)?;
        &a_contig.as_view()
    };
    let rank = a.shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if d >= rank {
        return Err(unsupported("select: dim out of range"));
    }
    if index >= a.shape[d].max(0) as usize {
        return Err(unsupported("select: index out of range"));
    }

    let mut out_shape = a.shape.clone();
    out_shape.remove(d);
    let mut out = OwnedTensor::new(a.dtype, out_shape);
    let elem_size = a.dtype.elem_size();
    let outer: usize = a.shape[..d].iter().map(|&s| s.max(0) as usize).product();
    let inner: usize = a.shape[d + 1..].iter().map(|&s| s.max(0) as usize).product();
    let dim_len = a.shape[d] as usize;

    for o in 0..outer {
        for inn in 0..inner {
            let src_idx = o * dim_len * inner + index * inner + inn;
            let dst_idx = o * inner + inn;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (a.data as *const u8).add(src_idx * elem_size),
                    (out.data.as_mut_ptr() as *mut u8).add(dst_idx * elem_size),
                    elem_size,
                );
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// index_select: select along a dim at int64 index positions (Phase 4)
// ---------------------------------------------------------------------------

pub fn index_select(a: &BorrowedTensor, dim: isize, index: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let a_contig;
    let a = if a.strides == contiguous_strides(&a.shape) {
        a
    } else {
        a_contig = to_contiguous(a)?;
        &a_contig.as_view()
    };
    if index.dtype != DType::I64 && index.dtype != DType::I32 {
        return Err(unsupported("index_select index must be int64/int32"));
    }
    if index.shape.len() != 1 {
        return Err(unsupported("index_select index must be 1D"));
    }
    let rank = a.shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if d >= rank {
        return Err(unsupported("index_select dim out of range"));
    }
    let idx_len = index.shape[0] as usize;
    let dim_size = a.shape[d] as usize;
    let outer: usize = a.shape[..d].iter().map(|&s| s.max(0) as usize).product();
    let inner: usize = a.shape[d + 1..].iter().map(|&s| s.max(0) as usize).product();

    let mut out_shape = a.shape.clone();
    out_shape[d] = idx_len as i64;
    let mut out = OwnedTensor::new(a.dtype, out_shape);

    let elem = a.dtype.elem_size();
    match index.dtype {
        DType::I64 => {
            let idx = unsafe { typed_slice::<i64>(index) };
            for o in 0..outer {
                for (j, &ix) in idx.iter().enumerate() {
                    if ix < 0 || ix as usize >= dim_size {
                        return Err(unsupported(&format!(
                            "index_select index {ix} out of range [0, {dim_size})"
                        )));
                    }
                    let src = o * dim_size * inner + ix as usize * inner;
                    let dst = o * idx_len * inner + j * inner;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (a.data as *const u8).add(src * elem),
                            (out.data.as_mut_ptr() as *mut u8).add(dst * elem),
                            inner * elem,
                        );
                    }
                }
            }
        }
        DType::I32 => {
            let idx = unsafe { typed_slice::<i32>(index) };
            for o in 0..outer {
                for (j, &ix) in idx.iter().enumerate() {
                    if ix < 0 || ix as usize >= dim_size {
                        return Err(unsupported(&format!(
                            "index_select index {ix} out of range [0, {dim_size})"
                        )));
                    }
                    let src = o * dim_size * inner + ix as usize * inner;
                    let dst = o * idx_len * inner + j * inner;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (a.data as *const u8).add(src * elem),
                            (out.data.as_mut_ptr() as *mut u8).add(dst * elem),
                            inner * elem,
                        );
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// gather: out[i...] = input[..., index[i...], ...] at `dim` (Phase 4)
// ---------------------------------------------------------------------------

pub fn gather(a: &BorrowedTensor, dim: isize, index: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if index.dtype != DType::I64 && index.dtype != DType::I32 {
        return Err(unsupported("gather index must be int64/int32"));
    }
    let rank = a.shape.len();
    if index.shape.len() != rank {
        return Err(unsupported(&format!(
            "gather index rank {} does not match input rank {rank}",
            index.shape.len()
        )));
    }
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
    if d >= rank {
        return Err(unsupported("gather dim out of range"));
    }
    let dim_size = a.shape[d] as usize;
    let n = elem_count(&index.shape);

    // Out-of-range indices are UB in torch; we reject them cleanly.
    let idx_i64: Option<&[i64]> = if index.dtype == DType::I64 {
        Some(unsafe { typed_slice::<i64>(index) })
    } else {
        None
    };
    let idx_i32: Option<&[i32]> = if index.dtype == DType::I32 {
        Some(unsafe { typed_slice::<i32>(index) })
    } else {
        None
    };

    let mut out = OwnedTensor::new(a.dtype, index.shape.clone());
    let elem = a.dtype.elem_size();
    let out_shape = index.shape.clone();

    // For each flat output position, decompose into coords, take the index
    // value at dim, and read the corresponding input element.
    for i in 0..n {
        let mut rem = i;
        let mut coords = vec![0usize; rank];
        for k in (0..rank).rev() {
            let s = out_shape[k].max(0) as usize;
            coords[k] = rem % s;
            rem /= s;
        }
        let ix: i64 = if let Some(idx) = idx_i64 {
            idx[i]
        } else {
            idx_i32.expect("index must be i64 or i32")[i] as i64
        };
        if ix < 0 || ix as usize >= dim_size {
            return Err(unsupported(&format!(
                "gather index {ix} out of range [0, {dim_size}) at dim {d}"
            )));
        }
        coords[d] = ix as usize;
        // flat input offset
        let mut src = 0usize;
        for k in 0..rank {
            src = src * a.shape[k].max(0) as usize + coords[k];
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                (a.data as *const u8).add(src * elem),
                (out.data.as_mut_ptr() as *mut u8).add(i * elem),
                elem,
            );
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tensor creation ops
// ---------------------------------------------------------------------------

/// Create a tensor filled with a constant value.
pub fn full(shape: &[i64], value: f64, dtype: crate::dlpack::DType) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(dtype, shape.to_vec());
    let n = out.elem_count();
    match dtype {
        DType::F32 => {
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, n) };
            d.iter_mut().for_each(|x| *x = value as f32);
        }
        DType::F64 => {
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, n) };
            d.iter_mut().for_each(|x| *x = value);
        }
        DType::I64 => {
            let v = value as i64;
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut i64, n) };
            d.iter_mut().for_each(|x| *x = v);
        }
        DType::I32 => {
            let v = value as i32;
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut i32, n) };
            d.iter_mut().for_each(|x| *x = v);
        }
        DType::Bool => {
            let v = if value != 0.0 { 1u8 } else { 0u8 };
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut u8, n) };
            d.iter_mut().for_each(|x| *x = v);
        }
    }
    Ok(out)
}

/// Create a tensor of zeros.
pub fn zeros(shape: &[i64], dtype: crate::dlpack::DType) -> PyResult<OwnedTensor> {
    full(shape, 0.0, dtype)
}

/// Create a tensor of ones.
pub fn ones(shape: &[i64], dtype: crate::dlpack::DType) -> PyResult<OwnedTensor> {
    full(shape, 1.0, dtype)
}

/// Create a 1-D tensor of evenly spaced values.
pub fn arange(start: f64, end: f64, step: f64, dtype: crate::dlpack::DType) -> PyResult<OwnedTensor> {
    if step == 0.0 {
        return Err(unsupported("arange: step must be non-zero"));
    }
    let n = ((end - start) / step).ceil().max(0.0) as usize;
    let mut out = OwnedTensor::new(dtype, vec![n as i64]);
    match dtype {
        DType::F32 => {
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, n) };
            for i in 0..n {
                d[i] = (start + i as f64 * step) as f32;
            }
        }
        DType::F64 => {
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                d[i] = start + i as f64 * step;
            }
        }
        _ => {
            let v_start = start as i64;
            let v_step = step as i64;
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                d[i] = v_start + i as i64 * v_step;
            }
        }
    }
    Ok(out)
}

/// Create a 1-D tensor of evenly spaced values between start and end (inclusive).
pub fn linspace(start: f64, end: f64, steps: usize, dtype: crate::dlpack::DType) -> PyResult<OwnedTensor> {
    if steps == 0 {
        return Err(unsupported("linspace: steps must be > 0"));
    }
    let mut out = OwnedTensor::new(dtype, vec![steps as i64]);
    let incr = if steps > 1 { (end - start) / (steps - 1) as f64 } else { 0.0 };
    match dtype {
        DType::F32 => {
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, steps) };
            for i in 0..steps {
                d[i] = (start + i as f64 * incr) as f32;
            }
        }
        DType::F64 => {
            let d = unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, steps) };
            for i in 0..steps {
                d[i] = start + i as f64 * incr;
            }
        }
        _ => {
            return Err(unsupported("linspace: only supports f32/f64"));
        }
    }
    Ok(out)
}

/// Split a tensor into chunks along a dimension.
pub fn chunk(a: &BorrowedTensor, chunks: usize, dim: isize) -> PyResult<Vec<OwnedTensor>> {
    let d = if dim < 0 { (a.shape.len() as isize + dim) as usize } else { dim as usize };
    if d >= a.shape.len() {
        return Err(unsupported(&format!("chunk: dim {} out of range for rank {}", dim, a.shape.len())));
    }
    let dim_size = a.shape[d] as usize;
    let chunk_size = (dim_size + chunks - 1) / chunks;
    let mut result = Vec::new();
    let mut start = 0;
    while start < dim_size {
        let end = (start + chunk_size).min(dim_size);
        let mut new_shape = a.shape.clone();
        new_shape[d] = (end - start) as i64;
        let mut out = OwnedTensor::new(a.dtype, new_shape);
        // Copy elements — for each slice along dim d, copy contiguous blocks
        let elem = a.dtype.elem_size();
        let outer: usize = a.shape[..d].iter().map(|&s| s.max(1) as usize).product();
        let inner: usize = a.shape[d+1..].iter().map(|&s| s.max(1) as usize).product();
        let src_offset = start * inner;
        let count = (end - start) * inner;
        for o in 0..outer {
            unsafe {
                let src = (a.data as *const u8)
                    .add((o * dim_size * inner + src_offset) * elem);
                let dst = (out.data.as_mut_ptr() as *mut u8)
                    .add(o * count * elem);
                std::ptr::copy_nonoverlapping(src, dst, count * elem);
            }
        }
        result.push(out);
        start = end;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Squeeze / unsqueeze / unflatten
// ---------------------------------------------------------------------------

/// Remove a dimension of size 1.
pub fn squeeze(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len() as isize;
    let d = if dim < 0 { rank + dim } else { dim };
    if d < 0 || d as usize >= a.shape.len() {
        return Err(unsupported(&format!("squeeze: dim {dim} out of range for rank {}", a.shape.len())));
    }
    if a.shape[d as usize] != 1 {
        return Err(unsupported(&format!("squeeze: dim {dim} has size {}, not 1", a.shape[d as usize])));
    }
    let mut new_shape = Vec::with_capacity(a.shape.len() - 1);
    for (i, &s) in a.shape.iter().enumerate() {
        if i as isize != d {
            new_shape.push(s);
        }
    }
    reshape(a, &new_shape)
}

/// Add a dimension of size 1 at the given position.
pub fn unsqueeze(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let rank = a.shape.len() as isize;
    let d = if dim < 0 { rank + dim + 1 } else { dim };
    if d < 0 || d as usize > a.shape.len() {
        return Err(unsupported(&format!("unsqueeze: dim {dim} out of range for rank {}", a.shape.len())));
    }
    let mut new_shape = Vec::with_capacity(a.shape.len() + 1);
    for (i, &s) in a.shape.iter().enumerate() {
        if i as isize == d {
            new_shape.push(1);
        }
        new_shape.push(s);
    }
    if d as usize == a.shape.len() {
        new_shape.push(1);
    }
    reshape(a, &new_shape)
}

/// Unflatten a dimension into a known shape.
pub fn unflatten(a: &BorrowedTensor, dim: isize, sizes: &[i64]) -> PyResult<OwnedTensor> {
    let rank = a.shape.len() as isize;
    let d = if dim < 0 { rank + dim } else { dim };
    if d < 0 || d as usize >= a.shape.len() {
        return Err(unsupported(&format!("unflatten: dim {dim} out of range for rank {}", a.shape.len())));
    }
    let dim_size = a.shape[d as usize];
    let expected: i64 = sizes.iter().product();
    if dim_size != expected {
        return Err(unsupported(&format!(
            "unflatten: dim size {} does not match product of sizes {:?}",
            dim_size, sizes
        )));
    }
    let mut new_shape = Vec::with_capacity(a.shape.len() - 1 + sizes.len());
    for (i, &s) in a.shape.iter().enumerate() {
        if i as isize == d {
            new_shape.extend_from_slice(sizes);
        } else {
            new_shape.push(s);
        }
    }
    reshape(a, &new_shape)
}
