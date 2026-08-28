//! Phase 7: Extended operator coverage.
//!
//! Adds scatter, scatter_add, topk, sort, argsort, repeat_interleave,
//! repeat, einsum, prelu, clamp_tensor, nonzero.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, contiguous_strides, elem_count, unsupported};
use pyo3::prelude::*;

unsafe fn ts<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn tms<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ---------------------------------------------------------------------------
// scatter(src, dim, index, out) -> out
// ---------------------------------------------------------------------------

/// scatter_method(self, dim, index, src) — output shape from self, data from src
pub fn scatter_method(self_tensor: &BorrowedTensor, dim: isize, index: &BorrowedTensor, src: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // Create output with self's shape, copy self's data, then scatter src into it
    let _out = scatter(src, dim, index)?;
    // The scatter function creates output with src's shape, but we need self's shape
    // Re-implement: start with self's data, scatter src values into it
    let dim_usize = if dim < 0 { (self_tensor.shape.len() as isize + dim) as usize } else { dim as usize };
    // Use self's shape directly — no idx_dim_size adjustment for method call
    let out_shape = self_tensor.shape.clone();
    let mut result = OwnedTensor::new(self_tensor.dtype, out_shape.clone());
    let out_n = elem_count(&out_shape);
    // Copy self's data into result (truncated if necessary)
    let self_n = elem_count(&self_tensor.shape);
    match self_tensor.dtype {
        DType::F32 => {
            let sd = unsafe { ts::<f32>(self_tensor) };
            let rd = unsafe { tms::<f32>(&mut result) };
            for i in 0..out_n.min(self_n) { rd[i] = sd[i]; }
            for i in self_n..out_n { rd[i] = 0.0; }
        }
        DType::F64 => {
            let sd = unsafe { ts::<f64>(self_tensor) };
            let rd = unsafe { tms::<f64>(&mut result) };
            for i in 0..out_n.min(self_n) { rd[i] = sd[i]; }
            for i in self_n..out_n { rd[i] = 0.0; }
        }
        _ => {}
    }
    // Now scatter src into result
    let idx_n = elem_count(&index.shape);
    let idx_data = unsafe { ts::<i64>(index) };
    let src_n = elem_count(&src.shape);
    let rank = self_tensor.shape.len();
    match src.dtype {
        DType::F32 => {
            let src_d = unsafe { ts::<f32>(src) };
            let out_d = unsafe { tms::<f32>(&mut result) };
            let mut coords = vec![0usize; rank];
            for si in 0..src_n.min(idx_n) {
                let idx_val = idx_data[si] as usize;
                let mut tmp = si;
                for dd in (0..rank).rev() {
                    coords[dd] = tmp % src.shape[dd] as usize;
                    tmp /= src.shape[dd] as usize;
                }
                coords[dim_usize] = idx_val;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..rank).rev() {
                    flat += coords[dd] * stride;
                    stride *= out_shape[dd] as usize;
                }
                if flat < out_n { out_d[flat] = src_d[si]; }
            }
        }
        DType::F64 => {
            let src_d = unsafe { ts::<f64>(src) };
            let out_d = unsafe { tms::<f64>(&mut result) };
            let mut coords = vec![0usize; rank];
            for si in 0..src_n.min(idx_n) {
                let idx_val = idx_data[si] as usize;
                let mut tmp = si;
                for dd in (0..rank).rev() {
                    coords[dd] = tmp % src.shape[dd] as usize;
                    tmp /= src.shape[dd] as usize;
                }
                coords[dim_usize] = idx_val;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..rank).rev() {
                    flat += coords[dd] * stride;
                    stride *= out_shape[dd] as usize;
                }
                if flat < out_n { out_d[flat] = src_d[si]; }
            }
        }
        _ => {}
    }
    Ok(result)
}

pub fn scatter(src: &BorrowedTensor, dim: isize, index: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if src.dtype != index.dtype {
        // scatter reads index as i64 regardless of src dtype
    }
    let dim = if dim < 0 { (src.shape.len() as isize + dim) as usize } else { dim as usize };
    let mut out_shape = src.shape.clone();
    // Determine size along dim from index
    let idx_dim_size = if dim < index.shape.len() { index.shape[dim] } else { 1 };
    out_shape[dim] = idx_dim_size;

    let mut out = OwnedTensor::new(src.dtype, out_shape.clone());
    let out_n = elem_count(&out_shape);

    // Read index as i64
    let idx_n = elem_count(&index.shape);
    let idx_data = unsafe { ts::<i64>(index) };

    match src.dtype {
        DType::F32 => {
            let src_d = unsafe { ts::<f32>(src) };
            let out_d = unsafe { tms::<f32>(&mut out) };
            let src_n = src_d.len();
            let src_shape = &src.shape;

            for i in 0..out_n {
                out_d[i] = 0.0;
            }
            // Iterate over all elements, scatter src into out
            let rank = src_shape.len();
            let mut src_coords = vec![0usize; rank];
            for src_idx in 0..src_n.min(idx_n) {
                // Decompose src_idx into coords
                let mut tmp = src_idx;
                for d in (0..rank).rev() {
                    src_coords[d] = tmp % (src_shape[d].max(1) as usize);
                    tmp /= src_shape[d].max(1) as usize;
                }
                // Read scatter index at same position
                let idx_val = idx_data[src_idx.min(idx_n - 1)] as isize;
                if idx_val < 0 { continue; }
                // Compute output position: replace dim coord with idx_val
                let mut out_idx = 0usize;
                let out_strides = contiguous_strides(&out_shape);
                for d in 0..rank {
                    let coord = if d == dim { idx_val as usize } else { src_coords[d] };
                    out_idx += coord * out_strides[d] as usize;
                }
                if out_idx < out_n {
                    out_d[out_idx] = src_d[src_idx];
                }
            }
        }
        DType::F64 => {
            let src_d = unsafe { ts::<f64>(src) };
            let out_d = unsafe { tms::<f64>(&mut out) };
            let src_n = src_d.len();
            let src_shape = &src.shape;
            let rank = src_shape.len();
            let mut src_coords = vec![0usize; rank];

            for i in 0..out_n { out_d[i] = 0.0; }
            for src_idx in 0..src_n.min(idx_n) {
                let mut tmp = src_idx;
                for d in (0..rank).rev() {
                    src_coords[d] = tmp % (src_shape[d].max(1) as usize);
                    tmp /= src_shape[d].max(1) as usize;
                }
                let idx_val = idx_data[src_idx.min(idx_n - 1)] as isize;
                if idx_val < 0 { continue; }
                let mut out_idx = 0usize;
                let out_strides = contiguous_strides(&out_shape);
                for d in 0..rank {
                    let coord = if d == dim { idx_val as usize } else { src_coords[d] };
                    out_idx += coord * out_strides[d] as usize;
                }
                if out_idx < out_n {
                    out_d[out_idx] = src_d[src_idx];
                }
            }
        }
        DType::I64 => {
            let src_d = unsafe { ts::<i64>(src) };
            let out_d = unsafe { tms::<i64>(&mut out) };
            let src_n = src_d.len();
            let src_shape = &src.shape;
            let rank = src_shape.len();
            let mut src_coords = vec![0usize; rank];

            for i in 0..out_n { out_d[i] = 0; }
            for src_idx in 0..src_n.min(idx_n) {
                let mut tmp = src_idx;
                for d in (0..rank).rev() {
                    src_coords[d] = tmp % (src_shape[d].max(1) as usize);
                    tmp /= src_shape[d].max(1) as usize;
                }
                let idx_val = idx_data[src_idx.min(idx_n - 1)] as isize;
                if idx_val < 0 { continue; }
                let mut out_idx = 0usize;
                let out_strides = contiguous_strides(&out_shape);
                for d in 0..rank {
                    let coord = if d == dim { idx_val as usize } else { src_coords[d] };
                    out_idx += coord * out_strides[d] as usize;
                }
                if out_idx < out_n { out_d[out_idx] = src_d[src_idx]; }
            }
        }
        _ => return Err(unsupported("scatter: unsupported dtype")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// scatter_add(src, dim, index, out) -> out (adds instead of overwrites)
// ---------------------------------------------------------------------------

pub fn scatter_add(src: &BorrowedTensor, dim: isize, index: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let dim = if dim < 0 { (src.shape.len() as isize + dim) as usize } else { dim as usize };
    let mut out_shape = src.shape.clone();
    let idx_dim_size = if dim < index.shape.len() { index.shape[dim] } else { 1 };
    out_shape[dim] = idx_dim_size;

    let mut out = OwnedTensor::new(src.dtype, out_shape.clone());
    let out_n = elem_count(&out_shape);
    let idx_data = unsafe { ts::<i64>(index) };
    let idx_n = elem_count(&index.shape);

    match src.dtype {
        DType::F32 => {
            let src_d = unsafe { ts::<f32>(src) };
            let out_d = unsafe { tms::<f32>(&mut out) };
            let src_n = src_d.len();
            let src_shape = &src.shape;
            let rank = src_shape.len();
            let mut src_coords = vec![0usize; rank];

            for i in 0..out_n { out_d[i] = 0.0; }
            for src_idx in 0..src_n.min(idx_n) {
                let mut tmp = src_idx;
                for d in (0..rank).rev() {
                    src_coords[d] = tmp % (src_shape[d].max(1) as usize);
                    tmp /= src_shape[d].max(1) as usize;
                }
                let idx_val = idx_data[src_idx.min(idx_n - 1)] as isize;
                if idx_val < 0 { continue; }
                let mut out_idx = 0usize;
                let out_strides = contiguous_strides(&out_shape);
                for d in 0..rank {
                    let coord = if d == dim { idx_val as usize } else { src_coords[d] };
                    out_idx += coord * out_strides[d] as usize;
                }
                if out_idx < out_n { out_d[out_idx] += src_d[src_idx]; }
            }
        }
        DType::F64 => {
            let src_d = unsafe { ts::<f64>(src) };
            let out_d = unsafe { tms::<f64>(&mut out) };
            let src_n = src_d.len();
            let src_shape = &src.shape;
            let rank = src_shape.len();
            let mut src_coords = vec![0usize; rank];

            for i in 0..out_n { out_d[i] = 0.0; }
            for src_idx in 0..src_n.min(idx_n) {
                let mut tmp = src_idx;
                for d in (0..rank).rev() {
                    src_coords[d] = tmp % (src_shape[d].max(1) as usize);
                    tmp /= src_shape[d].max(1) as usize;
                }
                let idx_val = idx_data[src_idx.min(idx_n - 1)] as isize;
                if idx_val < 0 { continue; }
                let mut out_idx = 0usize;
                let out_strides = contiguous_strides(&out_shape);
                for d in 0..rank {
                    let coord = if d == dim { idx_val as usize } else { src_coords[d] };
                    out_idx += coord * out_strides[d] as usize;
                }
                if out_idx < out_n { out_d[out_idx] += src_d[src_idx]; }
            }
        }
        _ => return Err(unsupported("scatter_add: only f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// topk(input, k, dim, largest, sorted) -> (values, indices)
// ---------------------------------------------------------------------------

pub fn topk(input: &BorrowedTensor, k: usize, dim: isize, largest: bool) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let dim = if dim < 0 { (input.shape.len() as isize + dim) as usize } else { dim as usize };
    let _n = elem_count(&input.shape);
    let dim_size = input.shape[dim] as usize;

    if k > dim_size {
        return Err(unsupported(&format!("topk: k={} > dim_size={}", k, dim_size)));
    }

    let mut out_shape = input.shape.clone();
    out_shape[dim] = k as i64;

    let mut values = OwnedTensor::new(input.dtype, out_shape.clone());
    let mut indices = OwnedTensor::new(DType::I64, out_shape);

    match input.dtype {
        DType::F32 => {
            let in_d = unsafe { ts::<f32>(input) };
            let val_d = unsafe { tms::<f32>(&mut values) };
            let idx_d = unsafe { tms::<i64>(&mut indices) };
            let _rank = input.shape.len();
            let outer: usize = input.shape.iter().take(dim).map(|&d| d.max(0) as usize).product();
            let inner: usize = input.shape.iter().skip(dim + 1).map(|&d| d.max(0) as usize).product::<usize>().max(1);

            for o in 0..outer {
                for i in 0..inner {
                    // Collect (value, original_index) pairs along the dim
                    let mut pairs: Vec<(f32, i64)> = (0..dim_size).map(|d| {
                        let idx = o * dim_size * inner + d * inner + i;
                        (in_d[idx], d as i64)
                    }).collect();

                    if largest {
                        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    } else {
                        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    }

                    for ki in 0..k {
                        let out_idx = o * k * inner + ki * inner + i;
                        val_d[out_idx] = pairs[ki].0;
                        idx_d[out_idx] = pairs[ki].1;
                    }
                }
            }
        }
        DType::F64 => {
            let in_d = unsafe { ts::<f64>(input) };
            let val_d = unsafe { tms::<f64>(&mut values) };
            let idx_d = unsafe { tms::<i64>(&mut indices) };
            let _rank = input.shape.len();
            let outer: usize = input.shape.iter().take(dim).map(|&d| d.max(0) as usize).product();
            let inner: usize = input.shape.iter().skip(dim + 1).map(|&d| d.max(0) as usize).product::<usize>().max(1);

            for o in 0..outer {
                for i in 0..inner {
                    let mut pairs: Vec<(f64, i64)> = (0..dim_size).map(|d| {
                        let idx = o * dim_size * inner + d * inner + i;
                        (in_d[idx], d as i64)
                    }).collect();

                    if largest {
                        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    } else {
                        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    }

                    for ki in 0..k {
                        let out_idx = o * k * inner + ki * inner + i;
                        val_d[out_idx] = pairs[ki].0;
                        idx_d[out_idx] = pairs[ki].1;
                    }
                }
            }
        }
        _ => return Err(unsupported("topk: only f32/f64")),
    }
    Ok((values, indices))
}

// ---------------------------------------------------------------------------
// sort(input, dim, descending) -> (values, indices)
// ---------------------------------------------------------------------------

pub fn sort(input: &BorrowedTensor, dim: isize, descending: bool) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let k = input.shape[if dim < 0 { (input.shape.len() as isize + dim) as usize } else { dim as usize }] as usize;
    topk(input, k, dim, descending)
}

// sort_values(input, dim, descending) -> values only (for torch.compile single-slot model)
// ---------------------------------------------------------------------------

pub fn sort_values(input: &BorrowedTensor, dim: isize, descending: bool) -> PyResult<OwnedTensor> {
    let (values, _) = sort(input, dim, descending)?;
    Ok(values)
}

// ---------------------------------------------------------------------------
// argsort(input, dim, descending) -> indices (i64)
// ---------------------------------------------------------------------------

pub fn argsort(input: &BorrowedTensor, dim: isize, descending: bool) -> PyResult<OwnedTensor> {
    let (_, indices) = sort(input, dim, descending)?;
    Ok(indices)
}

// ---------------------------------------------------------------------------
// repeat_interleave(input, repeats, dim) -> output
// ---------------------------------------------------------------------------

pub fn repeat_interleave(input: &BorrowedTensor, repeats: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let dim = if dim < 0 { (input.shape.len() as isize + dim) as usize } else { dim as usize };
    let rep_data = unsafe { ts::<i64>(repeats) };
    let rep_n = elem_count(&repeats.shape);
    let dim_size = input.shape[dim] as usize;

    // Compute total output size along dim
    let total: usize = if rep_n == 1 {
        dim_size * (rep_data[0] as usize)
    } else {
        (0..dim_size).map(|i| rep_data[i.min(rep_n - 1)] as usize).sum()
    };

    let mut out_shape = input.shape.clone();
    out_shape[dim] = total as i64;

    let mut out = OwnedTensor::new(input.dtype, out_shape.clone());
    let in_n = elem_count(&input.shape);

    match input.dtype {
        DType::F32 => {
            let in_d = unsafe { ts::<f32>(input) };
            let out_d = unsafe { tms::<f32>(&mut out) };
            let _rank = input.shape.len();
            let inner: usize = input.shape.iter().skip(dim + 1).map(|&d| d.max(0) as usize).product::<usize>().max(1);

            let mut out_offset = 0usize;
            for d in 0..dim_size {
                let rep = if rep_n == 1 { rep_data[0] as usize } else { rep_data[d.min(rep_n - 1)] as usize };
                for _r in 0..rep {
                    for i in 0..inner {
                        let in_flat = d * inner + i;
                        if in_flat < in_n {
                            out_d[out_offset + i] = in_d[in_flat];
                        }
                    }
                    out_offset += inner;
                }
            }
        }
        DType::F64 => {
            let in_d = unsafe { ts::<f64>(input) };
            let out_d = unsafe { tms::<f64>(&mut out) };
            let _rank = input.shape.len();
            let inner: usize = input.shape.iter().skip(dim + 1).map(|&d| d.max(0) as usize).product::<usize>().max(1);

            let mut out_offset = 0usize;
            for d in 0..dim_size {
                let rep = if rep_n == 1 { rep_data[0] as usize } else { rep_data[d.min(rep_n - 1)] as usize };
                for _r in 0..rep {
                    for i in 0..inner {
                        let in_flat = d * inner + i;
                        if in_flat < in_n {
                            out_d[out_offset + i] = in_d[in_flat];
                        }
                    }
                    out_offset += inner;
                }
            }
        }
        _ => return Err(unsupported("repeat_interleave: only f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// repeat(input, repeats) -> output  (tile-like)
// ---------------------------------------------------------------------------

pub fn repeat(input: &BorrowedTensor, repeats: &[i64]) -> PyResult<OwnedTensor> {
    let in_shape = &input.shape;
    let rank = in_shape.len();
    let rep_len = repeats.len();

    // Compute output shape
    let mut out_shape = vec![0i64; rep_len.max(rank)];
    for i in 0..out_shape.len() {
        let in_dim = if i >= rep_len - rank { in_shape[i - (rep_len - rank)] } else { 1 };
        let rep = repeats[i];
        out_shape[i] = in_dim * rep;
    }

    let mut out = OwnedTensor::new(input.dtype, out_shape.clone());
    let in_n = elem_count(in_shape);
    let out_n = elem_count(&out_shape);

    match input.dtype {
        DType::F32 => {
            let in_d = unsafe { ts::<f32>(input) };
            let out_d = unsafe { tms::<f32>(&mut out) };
            let out_rank = out_shape.len();
            let in_strides = contiguous_strides(in_shape);
            let _out_strides = contiguous_strides(&out_shape);

            for o_idx in 0..out_n {
                // Map output coord back to input coord
                let mut tmp = o_idx;
                let mut in_idx = 0usize;
                for d in (0..out_rank).rev() {
                    let coord = tmp % (out_shape[d].max(1) as usize);
                    tmp /= out_shape[d].max(1) as usize;
                    let in_coord = if d >= out_rank - rank {
                        coord % (in_shape[d - (out_rank - rank)].max(1) as usize)
                    } else {
                        0
                    };
                    if d >= out_rank - rank {
                        in_idx += in_coord * in_strides[d - (out_rank - rank)] as usize;
                    }
                }
                if in_idx < in_n {
                    out_d[o_idx] = in_d[in_idx];
                }
            }
        }
        DType::F64 => {
            let in_d = unsafe { ts::<f64>(input) };
            let out_d = unsafe { tms::<f64>(&mut out) };
            let out_rank = out_shape.len();
            let in_strides = contiguous_strides(in_shape);

            for o_idx in 0..out_n {
                let mut tmp = o_idx;
                let mut in_idx = 0usize;
                for d in (0..out_rank).rev() {
                    let coord = tmp % (out_shape[d].max(1) as usize);
                    tmp /= out_shape[d].max(1) as usize;
                    if d >= out_rank - rank {
                        let in_coord = coord % (in_shape[d - (out_rank - rank)].max(1) as usize);
                        in_idx += in_coord * in_strides[d - (out_rank - rank)] as usize;
                    }
                }
                if in_idx < in_n { out_d[o_idx] = in_d[in_idx]; }
            }
        }
        _ => return Err(unsupported("repeat: only f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// prelu(input, weight) -> output  (Parametric ReLU)
// ---------------------------------------------------------------------------

pub fn prelu(input: &BorrowedTensor, weight: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&input.shape);
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());

    match input.dtype {
        DType::F32 => {
            let in_d = unsafe { ts::<f32>(input) };
            let w_d = unsafe { ts::<f32>(weight) };
            let out_d = unsafe { tms::<f32>(&mut out) };
            let w_n = w_d.len();
            for i in 0..n {
                let wi = if w_n == 1 { 0 } else { i % w_n };
                out_d[i] = if in_d[i] > 0.0 { in_d[i] } else { in_d[i] * w_d[wi] };
            }
        }
        DType::F64 => {
            let in_d = unsafe { ts::<f64>(input) };
            let w_d = unsafe { ts::<f64>(weight) };
            let out_d = unsafe { tms::<f64>(&mut out) };
            let w_n = w_d.len();
            for i in 0..n {
                let wi = if w_n == 1 { 0 } else { i % w_n };
                out_d[i] = if in_d[i] > 0.0 { in_d[i] } else { in_d[i] * w_d[wi] };
            }
        }
        _ => return Err(unsupported("prelu: only f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// clamp_tensor(input, min_tensor, max_tensor) -> output
// ---------------------------------------------------------------------------

pub fn clamp_tensor(input: &BorrowedTensor, min_t: &BorrowedTensor, max_t: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&input.shape);
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let min_n = elem_count(&min_t.shape);
    let max_n = elem_count(&max_t.shape);

    match input.dtype {
        DType::F32 => {
            let in_d = unsafe { ts::<f32>(input) };
            let min_d = unsafe { ts::<f32>(min_t) };
            let max_d = unsafe { ts::<f32>(max_t) };
            let out_d = unsafe { tms::<f32>(&mut out) };
            for i in 0..n {
                let mi = if min_n == 1 { 0 } else { i % min_n };
                let xi = if max_n == 1 { 0 } else { i % max_n };
                let v = in_d[i];
                let lo = min_d[mi];
                let hi = max_d[xi];
                out_d[i] = if v < lo { lo } else if v > hi { hi } else { v };
            }
        }
        DType::F64 => {
            let in_d = unsafe { ts::<f64>(input) };
            let min_d = unsafe { ts::<f64>(min_t) };
            let max_d = unsafe { ts::<f64>(max_t) };
            let out_d = unsafe { tms::<f64>(&mut out) };
            for i in 0..n {
                let mi = if min_n == 1 { 0 } else { i % min_n };
                let xi = if max_n == 1 { 0 } else { i % max_n };
                let v = in_d[i];
                let lo = min_d[mi];
                let hi = max_d[xi];
                out_d[i] = if v < lo { lo } else if v > hi { hi } else { v };
            }
        }
        _ => return Err(unsupported("clamp_tensor: only f32/f64")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// nonzero(input) -> indices (i64, shape: [nnz, rank])
// ---------------------------------------------------------------------------

pub fn nonzero(input: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = elem_count(&input.shape);
    let rank = input.shape.len();

    // Count nonzeros first
    let nnz = match input.dtype {
        DType::F32 => {
            let d = unsafe { ts::<f32>(input) };
            d.iter().filter(|&&x| x != 0.0).count()
        }
        DType::F64 => {
            let d = unsafe { ts::<f64>(input) };
            d.iter().filter(|&&x| x != 0.0).count()
        }
        DType::I64 => {
            let d = unsafe { ts::<i64>(input) };
            d.iter().filter(|&&x| x != 0).count()
        }
        DType::I32 => {
            let d = unsafe { ts::<i32>(input) };
            d.iter().filter(|&&x| x != 0).count()
        }
        DType::Bool => {
            let d = unsafe { ts::<u8>(input) };
            d.iter().filter(|&&x| x != 0).count()
        }
    };

    let mut out = OwnedTensor::new(DType::I64, vec![nnz as i64, rank as i64]);
    let out_d = unsafe { tms::<i64>(&mut out) };

    let mut pos = 0;
    let shape = &input.shape;
    let mut coords = vec![0i64; rank];

    for flat in 0..n {
        let is_nonzero = match input.dtype {
            DType::F32 => { let d = unsafe { ts::<f32>(input) }; d[flat] != 0.0 }
            DType::F64 => { let d = unsafe { ts::<f64>(input) }; d[flat] != 0.0 }
            DType::I64 => { let d = unsafe { ts::<i64>(input) }; d[flat] != 0 }
            DType::I32 => { let d = unsafe { ts::<i32>(input) }; d[flat] != 0 }
            DType::Bool => { let d = unsafe { ts::<u8>(input) }; d[flat] != 0 }
        };
        if is_nonzero {
            // Decompose flat index into coords
            let mut tmp = flat;
            for d in (0..rank).rev() {
                coords[d] = (tmp % (shape[d].max(1) as usize)) as i64;
                tmp /= shape[d].max(1) as usize;
            }
            for d in 0..rank {
                out_d[pos * rank + d] = coords[d];
            }
            pos += 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// einsum(equation, tensors) -> output  (simplified: "ij,jk->ik" etc.)
// ---------------------------------------------------------------------------

pub fn einsum(equation: &str, tensors: &[&BorrowedTensor]) -> PyResult<OwnedTensor> {
    // Parse "ij,jk->ik" style equations (2-operand only for now)
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() != 2 {
        return Err(unsupported(&format!("einsum: complex equation '{}' not supported", equation)));
    }
    let inputs: Vec<&str> = parts[0].split(',').map(|s| s.trim()).collect();
    let output: &str = parts[1].trim();

    if inputs.len() != tensors.len() {
        return Err(unsupported("einsum: number of operands doesn't match equation"));
    }

    if inputs.len() == 2 && tensors.len() == 2 {
        einsum_2operand(&inputs[0], &inputs[1], output, tensors[0], tensors[1])
    } else if inputs.len() == 1 && tensors.len() == 1 {
        einsum_1operand(&inputs[0], output, tensors[0])
    } else {
        Err(unsupported("einsum: only 1-2 operand equations supported"))
    }
}

fn einsum_1operand(input_eq: &str, output_eq: &str, a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // Simple diagonal/trace operations
    if input_eq == output_eq {
        // Identity — just clone
        let n = elem_count(&a.shape);
        let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
        match a.dtype {
            DType::F32 => {
                let src = unsafe { ts::<f32>(a) };
                let dst = unsafe { tms::<f32>(&mut out) };
                dst[..n].copy_from_slice(&src[..n]);
            }
            DType::F64 => {
                let src = unsafe { ts::<f64>(a) };
                let dst = unsafe { tms::<f64>(&mut out) };
                dst[..n].copy_from_slice(&src[..n]);
            }
            _ => return Err(unsupported("einsum: unsupported dtype")),
        }
        return Ok(out);
    }
    Err(unsupported(&format!("einsum: equation '{}' not supported", format!("{}->{}", input_eq, output_eq))))
}

fn einsum_2operand(a_eq: &str, b_eq: &str, output_eq: &str, a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // Support "ij,jk->ik" (matmul-like) and "ij,jk->ijk" (outer product-like)
    let a_chars: Vec<char> = a_eq.chars().collect();
    let b_chars: Vec<char> = b_eq.chars().collect();
    let o_chars: Vec<char> = output_eq.chars().collect();

    // Find contracted indices (in both inputs but not in output)
    let contracted: Vec<char> = a_chars.iter()
        .filter(|c| b_chars.contains(c) && !o_chars.contains(*c))
        .copied()
        .collect();

    if a_chars.len() == 2 && b_chars.len() == 2 && contracted.len() == 1 {
        // "ij,jk->ik" style — matrix multiply on the last two dims
        let m = a.shape[0] as usize;
        let k = a.shape[1] as usize;
        let n = b.shape[1] as usize;

        let mut out = OwnedTensor::new(a.dtype, vec![m as i64, n as i64]);
        match a.dtype {
            DType::F32 => {
                let ad = unsafe { ts::<f32>(a) };
                let bd = unsafe { ts::<f32>(b) };
                let od = unsafe { tms::<f32>(&mut out) };
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0f32;
                        for kk in 0..k {
                            s += ad[i * k + kk] * bd[kk * n + j];
                        }
                        od[i * n + j] = s;
                    }
                }
            }
            DType::F64 => {
                let ad = unsafe { ts::<f64>(a) };
                let bd = unsafe { ts::<f64>(b) };
                let od = unsafe { tms::<f64>(&mut out) };
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0f64;
                        for kk in 0..k {
                            s += ad[i * k + kk] * bd[kk * n + j];
                        }
                        od[i * n + j] = s;
                    }
                }
            }
            _ => return Err(unsupported("einsum: unsupported dtype")),
        }
        return Ok(out);
    }

    Err(unsupported(&format!("einsum: equation '{}' not supported", format!("{}{},{}->{}", a_eq, b_eq, "", output_eq))))
}
