//! Reduction operations along one or more dimensions.
//!
//! Supports sum, mean, max, min, argmax, argmin, std, var, cumsum, prod, norm.
//! All reductions support optional `dim` and `keepdim` parameters.

use crate::dlpack::{elem_count, unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

/// Normalize a dim argument (handle negative dims).
fn norm_dim(dim: isize, rank: usize) -> usize {
    if dim < 0 {
        (rank as isize + dim) as usize
    } else {
        dim as usize
    }
}

/// Compute output shape after reducing along `dim` with optional `keepdim`.
fn reduce_shape(shape: &[i64], dim: usize, keepdim: bool) -> Vec<i64> {
    if keepdim {
        let mut out = shape.to_vec();
        out[dim] = 1;
        out
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

fn sum_f32(a: &BorrowedTensor, dim: usize, keepdim: bool) -> OwnedTensor {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_shape = reduce_shape(&a.shape, dim, keepdim);
    let mut out = OwnedTensor::new(DType::F32, out_shape.clone());
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let shape = &a.shape;
    let rank = shape.len();
    let dim_size = shape[dim] as usize;

    // Compute strides for iterating over all dimensions except dim
    let mut outer_stride = 1i64;
    for i in 0..dim {
        outer_stride *= shape[i];
    }
    let mut inner_stride = 1i64;
    for i in (dim + 1)..rank {
        inner_stride *= shape[i];
    }

    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut sum = 0.0f32;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                sum += a_data[idx];
            }
            let out_idx = outer * inner_size + inner;
            out_data[out_idx] = sum;
        }
    }
    out
}

fn sum_f64(a: &BorrowedTensor, dim: usize, keepdim: bool) -> OwnedTensor {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_shape = reduce_shape(&a.shape, dim, keepdim);
    let mut out = OwnedTensor::new(DType::F64, out_shape.clone());
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let shape = &a.shape;
    let rank = shape.len();
    let dim_size = shape[dim] as usize;
    let mut outer_stride = 1i64;
    for i in 0..dim {
        outer_stride *= shape[i];
    }
    let mut inner_stride = 1i64;
    for i in (dim + 1)..rank {
        inner_stride *= shape[i];
    }

    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut sum = 0.0f64;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                sum += a_data[idx];
            }
            let out_idx = if keepdim {
                outer * inner_size + inner
            } else {
                outer * inner_size + inner
            };
            out_data[out_idx] = sum;
        }
    }
    out
}

pub fn sum(a: &BorrowedTensor, dim: Option<isize>, keepdim: bool) -> PyResult<OwnedTensor> {
    match dim {
        Some(d) => {
            let d = norm_dim(d, a.shape.len());
            Ok(match a.dtype {
                DType::F32 => sum_f32(a, d, keepdim),
                DType::F64 => sum_f64(a, d, keepdim),
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"))
                }
            })
        }
        None => {
            // Reduce all dims — scalar output
            let total: f64 = match a.dtype {
                DType::F32 => unsafe { typed_slice::<f32>(a) }
                    .iter()
                    .map(|&x| x as f64)
                    .sum(),
                DType::F64 => unsafe { typed_slice::<f64>(a) }.iter().copied().sum(),
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"))
                }
            };
            let mut out = OwnedTensor::new(a.dtype, vec![]);
            match a.dtype {
                DType::F32 => {
                    let d = unsafe { typed_mut_slice::<f32>(&mut out) };
                    d[0] = total as f32;
                }
                DType::F64 => {
                    let d = unsafe { typed_mut_slice::<f64>(&mut out) };
                    d[0] = total;
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Mean
// ---------------------------------------------------------------------------

pub fn mean(a: &BorrowedTensor, dim: Option<isize>, keepdim: bool) -> PyResult<OwnedTensor> {
    match dim {
        Some(d) => {
            let d = norm_dim(d, a.shape.len());
            let dim_size = a.shape[d] as f64;
            let mut out = sum(a, Some(d as isize), keepdim)?;
            // Divide by dim_size
            match a.dtype {
                DType::F32 => {
                    let data = unsafe { typed_mut_slice::<f32>(&mut out) };
                    for v in data.iter_mut() {
                        *v /= dim_size as f32;
                    }
                }
                DType::F64 => {
                    let data = unsafe { typed_mut_slice::<f64>(&mut out) };
                    for v in data.iter_mut() {
                        *v /= dim_size;
                    }
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
            Ok(out)
        }
        None => {
            let total_elems = elem_count(&a.shape) as f64;
            let total: f64 = match a.dtype {
                DType::F32 => unsafe { typed_slice::<f32>(a) }
                    .iter()
                    .map(|&x| x as f64)
                    .sum(),
                DType::F64 => unsafe { typed_slice::<f64>(a) }.iter().copied().sum(),
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"))
                }
            };
            let mean_val = total / total_elems;
            let mut out = OwnedTensor::new(a.dtype, vec![]);
            match a.dtype {
                DType::F32 => {
                    let d = unsafe { typed_mut_slice::<f32>(&mut out) };
                    d[0] = mean_val as f32;
                }
                DType::F64 => {
                    let d = unsafe { typed_mut_slice::<f64>(&mut out) };
                    d[0] = mean_val;
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Max / Min
// ---------------------------------------------------------------------------

fn max_f32(a: &BorrowedTensor, dim: usize, keepdim: bool) -> (OwnedTensor, OwnedTensor) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_shape = reduce_shape(&a.shape, dim, keepdim);
    let mut out_val = OwnedTensor::new(DType::F32, out_shape.clone());
    let mut out_idx = OwnedTensor::new(DType::I64, out_shape);
    let out_val_data = unsafe { typed_mut_slice::<f32>(&mut out_val) };
    let out_idx_data = unsafe { typed_mut_slice::<i64>(&mut out_idx) };

    let shape = &a.shape;
    let rank = shape.len();
    let dim_size = shape[dim] as usize;
    let mut outer_stride = 1i64;
    for i in 0..dim {
        outer_stride *= shape[i];
    }
    let mut inner_stride = 1i64;
    for i in (dim + 1)..rank {
        inner_stride *= shape[i];
    }

    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_i = 0usize;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                if a_data[idx] > max_val {
                    max_val = a_data[idx];
                    max_i = i;
                }
            }
            let out_idx_pos = outer * inner_size + inner;
            out_val_data[out_idx_pos] = max_val;
            out_idx_data[out_idx_pos] = max_i as i64;
        }
    }
    (out_val, out_idx)
}

fn max_f64(a: &BorrowedTensor, dim: usize, keepdim: bool) -> (OwnedTensor, OwnedTensor) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_shape = reduce_shape(&a.shape, dim, keepdim);
    let mut out_val = OwnedTensor::new(DType::F64, out_shape.clone());
    let mut out_idx = OwnedTensor::new(DType::I64, out_shape);
    let out_val_data = unsafe { typed_mut_slice::<f64>(&mut out_val) };
    let out_idx_data = unsafe { typed_mut_slice::<i64>(&mut out_idx) };

    let shape = &a.shape;
    let rank = shape.len();
    let dim_size = shape[dim] as usize;
    let mut outer_stride = 1i64;
    for i in 0..dim {
        outer_stride *= shape[i];
    }
    let mut inner_stride = 1i64;
    for i in (dim + 1)..rank {
        inner_stride *= shape[i];
    }

    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_i = 0usize;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                if a_data[idx] > max_val {
                    max_val = a_data[idx];
                    max_i = i;
                }
            }
            let out_idx_pos = outer * inner_size + inner;
            out_val_data[out_idx_pos] = max_val;
            out_idx_data[out_idx_pos] = max_i as i64;
        }
    }
    (out_val, out_idx)
}

pub fn max_reduce(
    a: &BorrowedTensor,
    dim: Option<isize>,
    keepdim: bool,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    match dim {
        Some(d) => {
            let d = norm_dim(d, a.shape.len());
            Ok(match a.dtype {
                DType::F32 => max_f32(a, d, keepdim),
                DType::F64 => max_f64(a, d, keepdim),
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"))
                }
            })
        }
        None => {
            // Reduce all — scalar
            match a.dtype {
                DType::F32 => {
                    let data = unsafe { typed_slice::<f32>(a) };
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_i = 0usize;
                    for (i, &v) in data.iter().enumerate() {
                        if v > max_val {
                            max_val = v;
                            max_i = i;
                        }
                    }
                    let mut out_val = OwnedTensor::new(DType::F32, vec![]);
                    let mut out_idx = OwnedTensor::new(DType::I64, vec![]);
                    {
                        let v = unsafe { typed_mut_slice::<f32>(&mut out_val) };
                        v[0] = max_val;
                    }
                    {
                        let i = unsafe { typed_mut_slice::<i64>(&mut out_idx) };
                        i[0] = max_i as i64;
                    }
                    Ok((out_val, out_idx))
                }
                DType::F64 => {
                    let data = unsafe { typed_slice::<f64>(a) };
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_i = 0usize;
                    for (i, &v) in data.iter().enumerate() {
                        if v > max_val {
                            max_val = v;
                            max_i = i;
                        }
                    }
                    let mut out_val = OwnedTensor::new(DType::F64, vec![]);
                    let mut out_idx = OwnedTensor::new(DType::I64, vec![]);
                    {
                        let v = unsafe { typed_mut_slice::<f64>(&mut out_val) };
                        v[0] = max_val;
                    }
                    {
                        let i = unsafe { typed_mut_slice::<i64>(&mut out_idx) };
                        i[0] = max_i as i64;
                    }
                    Ok((out_val, out_idx))
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Min — symmetric counterpart to max_reduce
// ---------------------------------------------------------------------------

fn min_f32(a: &BorrowedTensor, dim: usize, keepdim: bool) -> (OwnedTensor, OwnedTensor) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_shape = reduce_shape(&a.shape, dim, keepdim);
    let mut out_val = OwnedTensor::new(DType::F32, out_shape.clone());
    let mut out_idx = OwnedTensor::new(DType::I64, out_shape);
    let out_val_data = unsafe { typed_mut_slice::<f32>(&mut out_val) };
    let out_idx_data = unsafe { typed_mut_slice::<i64>(&mut out_idx) };

    let shape = &a.shape;
    let rank = shape.len();
    let dim_size = shape[dim] as usize;
    let mut outer_stride = 1i64;
    for i in 0..dim {
        outer_stride *= shape[i];
    }
    let mut inner_stride = 1i64;
    for i in (dim + 1)..rank {
        inner_stride *= shape[i];
    }
    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut min_val = f32::INFINITY;
            let mut min_i = 0usize;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                if a_data[idx] < min_val {
                    min_val = a_data[idx];
                    min_i = i;
                }
            }
            let out_pos = outer * inner_size + inner;
            out_val_data[out_pos] = min_val;
            out_idx_data[out_pos] = min_i as i64;
        }
    }
    (out_val, out_idx)
}

fn min_f64(a: &BorrowedTensor, dim: usize, keepdim: bool) -> (OwnedTensor, OwnedTensor) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_shape = reduce_shape(&a.shape, dim, keepdim);
    let mut out_val = OwnedTensor::new(DType::F64, out_shape.clone());
    let mut out_idx = OwnedTensor::new(DType::I64, out_shape);
    let out_val_data = unsafe { typed_mut_slice::<f64>(&mut out_val) };
    let out_idx_data = unsafe { typed_mut_slice::<i64>(&mut out_idx) };

    let shape = &a.shape;
    let rank = shape.len();
    let dim_size = shape[dim] as usize;
    let mut outer_stride = 1i64;
    for i in 0..dim {
        outer_stride *= shape[i];
    }
    let mut inner_stride = 1i64;
    for i in (dim + 1)..rank {
        inner_stride *= shape[i];
    }
    let outer_size = outer_stride as usize;
    let inner_size = inner_stride as usize;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut min_val = f64::INFINITY;
            let mut min_i = 0usize;
            for i in 0..dim_size {
                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                if a_data[idx] < min_val {
                    min_val = a_data[idx];
                    min_i = i;
                }
            }
            let out_pos = outer * inner_size + inner;
            out_val_data[out_pos] = min_val;
            out_idx_data[out_pos] = min_i as i64;
        }
    }
    (out_val, out_idx)
}

pub fn min_reduce(
    a: &BorrowedTensor,
    dim: Option<isize>,
    keepdim: bool,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    match dim {
        Some(d) => {
            let d = norm_dim(d, a.shape.len());
            Ok(match a.dtype {
                DType::F32 => min_f32(a, d, keepdim),
                DType::F64 => min_f64(a, d, keepdim),
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"))
                }
            })
        }
        None => match a.dtype {
            DType::F32 => {
                let data = unsafe { typed_slice::<f32>(a) };
                let mut min_val = f32::INFINITY;
                let mut min_i = 0usize;
                for (i, &v) in data.iter().enumerate() {
                    if v < min_val {
                        min_val = v;
                        min_i = i;
                    }
                }
                let mut out_val = OwnedTensor::new(DType::F32, vec![]);
                let mut out_idx = OwnedTensor::new(DType::I64, vec![]);
                {
                    let v = unsafe { typed_mut_slice::<f32>(&mut out_val) };
                    v[0] = min_val;
                }
                {
                    let i = unsafe { typed_mut_slice::<i64>(&mut out_idx) };
                    i[0] = min_i as i64;
                }
                Ok((out_val, out_idx))
            }
            DType::F64 => {
                let data = unsafe { typed_slice::<f64>(a) };
                let mut min_val = f64::INFINITY;
                let mut min_i = 0usize;
                for (i, &v) in data.iter().enumerate() {
                    if v < min_val {
                        min_val = v;
                        min_i = i;
                    }
                }
                let mut out_val = OwnedTensor::new(DType::F64, vec![]);
                let mut out_idx = OwnedTensor::new(DType::I64, vec![]);
                {
                    let v = unsafe { typed_mut_slice::<f64>(&mut out_val) };
                    v[0] = min_val;
                }
                {
                    let i = unsafe { typed_mut_slice::<i64>(&mut out_idx) };
                    i[0] = min_i as i64;
                }
                Ok((out_val, out_idx))
            }
            DType::I64 | DType::I32 | DType::Bool => {
                return Err(unsupported("this kernel only supports f32/f64 tensors"));
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Argmax / Argmin
// ---------------------------------------------------------------------------

pub fn argmax(a: &BorrowedTensor, dim: Option<isize>, keepdim: bool) -> PyResult<OwnedTensor> {
    // argmax returns int64 indices
    let (_, idx) = max_reduce(a, dim, keepdim)?;
    Ok(idx)
}

pub fn argmin(a: &BorrowedTensor, dim: Option<isize>, keepdim: bool) -> PyResult<OwnedTensor> {
    // argmin is negated max
    let negated = match a.dtype {
        DType::F32 => {
            let data = unsafe { typed_slice::<f32>(a) };
            let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = -v;
            }
            out
        }
        DType::F64 => {
            let data = unsafe { typed_slice::<f64>(a) };
            let mut out = OwnedTensor::new(DType::F64, a.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = -v;
            }
            out
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"))
        }
    };
    let (_, idx) = max_reduce(&BorrowedTensor::from_owned(&negated), dim, keepdim)?;
    Ok(idx)
}

// ---------------------------------------------------------------------------
// Std / Var
// ---------------------------------------------------------------------------

pub fn std_dev(
    a: &BorrowedTensor,
    dim: Option<isize>,
    keepdim: bool,
    unbiased: bool,
) -> PyResult<OwnedTensor> {
    let var = variance(a, dim, keepdim, unbiased)?;
    // sqrt(var)
    match var.dtype {
        DType::F32 => {
            let view = var.as_view();
            let data = unsafe { typed_slice::<f32>(&view) };
            let mut out = OwnedTensor::new(DType::F32, var.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.sqrt();
            }
            Ok(out)
        }
        DType::F64 => {
            let view = var.as_view();
            let data = unsafe { typed_slice::<f64>(&view) };
            let mut out = OwnedTensor::new(DType::F64, var.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.sqrt();
            }
            Ok(out)
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

fn variance(
    a: &BorrowedTensor,
    dim: Option<isize>,
    keepdim: bool,
    unbiased: bool,
) -> PyResult<OwnedTensor> {
    let m = mean(a, dim, keepdim)?;
    // var = mean((x - mean)^2), then divide by n or n-1
    match dim {
        Some(d) => {
            let d = norm_dim(d, a.shape.len());
            let dim_size = a.shape[d] as f64;
            let n = if unbiased {
                (dim_size - 1.0).max(1.0)
            } else {
                dim_size
            };
            let out_shape = reduce_shape(&a.shape, d, keepdim);
            let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
            let shape = &a.shape;
            let rank = shape.len();
            let mut outer_stride = 1i64;
            for i in 0..d {
                outer_stride *= shape[i];
            }
            let mut inner_stride = 1i64;
            for i in (d + 1)..rank {
                inner_stride *= shape[i];
            }
            let outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;

            match a.dtype {
                DType::F32 => {
                    let m_view = m.as_view();
                    let m_data = unsafe { typed_slice::<f32>(&m_view) };
                    let a_data = unsafe { typed_slice::<f32>(a) };
                    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mean_val = m_data[outer * inner_size + inner];
                            let mut sum_sq = 0.0f32;
                            for i in 0..dim_size as usize {
                                let idx = outer * (dim_size as usize * inner_size)
                                    + i * inner_size
                                    + inner;
                                let diff = a_data[idx] - mean_val;
                                sum_sq += diff * diff;
                            }
                            out_data[outer * inner_size + inner] = sum_sq / n as f32;
                        }
                    }
                }
                DType::F64 => {
                    let m_view = m.as_view();
                    let m_data = unsafe { typed_slice::<f64>(&m_view) };
                    let a_data = unsafe { typed_slice::<f64>(a) };
                    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mean_val = m_data[outer * inner_size + inner];
                            let mut sum_sq = 0.0f64;
                            for i in 0..dim_size as usize {
                                let idx = outer * (dim_size as usize * inner_size)
                                    + i * inner_size
                                    + inner;
                                let diff = a_data[idx] - mean_val;
                                sum_sq += diff * diff;
                            }
                            out_data[outer * inner_size + inner] = sum_sq / n;
                        }
                    }
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
            Ok(out)
        }
        None => {
            let total_elems = elem_count(&a.shape) as f64;
            let n = if unbiased {
                (total_elems - 1.0).max(1.0)
            } else {
                total_elems
            };
            let mut out = OwnedTensor::new(a.dtype, vec![]);
            match a.dtype {
                DType::F32 => {
                    let m_view = m.as_view();
                    let m_val = unsafe { typed_slice::<f32>(&m_view) }[0];
                    let a_data = unsafe { typed_slice::<f32>(a) };
                    let mut sum_sq = 0.0f32;
                    for &v in a_data.iter() {
                        let diff = v - m_val;
                        sum_sq += diff * diff;
                    }
                    let d = unsafe { typed_mut_slice::<f32>(&mut out) };
                    d[0] = sum_sq / n as f32;
                }
                DType::F64 => {
                    let m_view = m.as_view();
                    let m_val = unsafe { typed_slice::<f64>(&m_view) }[0];
                    let a_data = unsafe { typed_slice::<f64>(a) };
                    let mut sum_sq = 0.0f64;
                    for &v in a_data.iter() {
                        let diff = v - m_val;
                        sum_sq += diff * diff;
                    }
                    let d = unsafe { typed_mut_slice::<f64>(&mut out) };
                    d[0] = sum_sq / n;
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
            Ok(out)
        }
    }
}

pub fn var(
    a: &BorrowedTensor,
    dim: Option<isize>,
    keepdim: bool,
    unbiased: bool,
) -> PyResult<OwnedTensor> {
    variance(a, dim, keepdim, unbiased)
}

// ---------------------------------------------------------------------------
// Cumsum
// ---------------------------------------------------------------------------

pub fn cumsum(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let d = norm_dim(dim, a.shape.len());
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());

    match a.dtype {
        DType::F32 => {
            let a_data = unsafe { typed_slice::<f32>(a) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let shape = &a.shape;
            let rank = shape.len();
            let dim_size = shape[d] as usize;
            let mut outer_stride = 1i64;
            for i in 0..d {
                outer_stride *= shape[i];
            }
            let mut inner_stride = 1i64;
            for i in (d + 1)..rank {
                inner_stride *= shape[i];
            }

            let outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;

            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let mut cum = 0.0f32;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        cum += a_data[idx];
                        out_data[idx] = cum;
                    }
                }
            }
        }
        DType::F64 => {
            let a_data = unsafe { typed_slice::<f64>(a) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            let shape = &a.shape;
            let rank = shape.len();
            let dim_size = shape[d] as usize;
            let mut outer_stride = 1i64;
            for i in 0..d {
                outer_stride *= shape[i];
            }
            let mut inner_stride = 1i64;
            for i in (d + 1)..rank {
                inner_stride *= shape[i];
            }

            let outer_size = outer_stride as usize;
            let inner_size = inner_stride as usize;

            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let mut cum = 0.0f64;
                    for i in 0..dim_size {
                        let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                        cum += a_data[idx];
                        out_data[idx] = cum;
                    }
                }
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Prod
// ---------------------------------------------------------------------------

pub fn prod(a: &BorrowedTensor, dim: Option<isize>, keepdim: bool) -> PyResult<OwnedTensor> {
    match dim {
        Some(d) => {
            let d = norm_dim(d, a.shape.len());
            let out_shape = reduce_shape(&a.shape, d, keepdim);
            let mut out = OwnedTensor::new(a.dtype, out_shape);

            match a.dtype {
                DType::F32 => {
                    let a_data = unsafe { typed_slice::<f32>(a) };
                    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
                    let shape = &a.shape;
                    let rank = shape.len();
                    let dim_size = shape[d] as usize;
                    let mut outer_stride = 1i64;
                    for i in 0..d {
                        outer_stride *= shape[i];
                    }
                    let mut inner_stride = 1i64;
                    for i in (d + 1)..rank {
                        inner_stride *= shape[i];
                    }
                    let outer_size = outer_stride as usize;
                    let inner_size = inner_stride as usize;
                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mut prod = 1.0f32;
                            for i in 0..dim_size {
                                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                                prod *= a_data[idx];
                            }
                            out_data[outer * inner_size + inner] = prod;
                        }
                    }
                }
                DType::F64 => {
                    let a_data = unsafe { typed_slice::<f64>(a) };
                    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
                    let shape = &a.shape;
                    let rank = shape.len();
                    let dim_size = shape[d] as usize;
                    let mut outer_stride = 1i64;
                    for i in 0..d {
                        outer_stride *= shape[i];
                    }
                    let mut inner_stride = 1i64;
                    for i in (d + 1)..rank {
                        inner_stride *= shape[i];
                    }
                    let outer_size = outer_stride as usize;
                    let inner_size = inner_stride as usize;
                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            let mut prod = 1.0f64;
                            for i in 0..dim_size {
                                let idx = outer * (dim_size * inner_size) + i * inner_size + inner;
                                prod *= a_data[idx];
                            }
                            out_data[outer * inner_size + inner] = prod;
                        }
                    }
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
            Ok(out)
        }
        None => {
            let total: f64 = match a.dtype {
                DType::F32 => unsafe { typed_slice::<f32>(a) }
                    .iter()
                    .map(|&x| x as f64)
                    .product(),
                DType::F64 => unsafe { typed_slice::<f64>(a) }.iter().copied().product(),
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"))
                }
            };
            let mut out = OwnedTensor::new(a.dtype, vec![]);
            match a.dtype {
                DType::F32 => {
                    let d = unsafe { typed_mut_slice::<f32>(&mut out) };
                    d[0] = total as f32;
                }
                DType::F64 => {
                    let d = unsafe { typed_mut_slice::<f64>(&mut out) };
                    d[0] = total;
                }
                DType::I64 | DType::I32 | DType::Bool => {
                    return Err(unsupported("this kernel only supports f32/f64 tensors"));
                }
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Norm (L2 / Frobenius)
// ---------------------------------------------------------------------------

pub fn norm(a: &BorrowedTensor, dim: Option<isize>, keepdim: bool) -> PyResult<OwnedTensor> {
    // L2 norm: sqrt(sum(x^2))
    let squared = match a.dtype {
        DType::F32 => {
            let data = unsafe { typed_slice::<f32>(a) };
            let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v * v;
            }
            out
        }
        DType::F64 => {
            let data = unsafe { typed_slice::<f64>(a) };
            let mut out = OwnedTensor::new(DType::F64, a.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v * v;
            }
            out
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"))
        }
    };
    let summed = sum(&BorrowedTensor::from_owned(&squared), dim, keepdim)?;
    // sqrt
    match summed.dtype {
        DType::F32 => {
            let view = summed.as_view();
            let data = unsafe { typed_slice::<f32>(&view) };
            let mut out = OwnedTensor::new(DType::F32, summed.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.sqrt();
            }
            Ok(out)
        }
        DType::F64 => {
            let view = summed.as_view();
            let data = unsafe { typed_slice::<f64>(&view) };
            let mut out = OwnedTensor::new(DType::F64, summed.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.sqrt();
            }
            Ok(out)
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

// ---------------------------------------------------------------------------
// General p-norm: ||x||_p = (sum(|x|^p))^(1/p)
// Supports p=1 (L1), p=2 (L2), p=infinity (Linf), and any positive float.
// ---------------------------------------------------------------------------

pub fn p_norm(
    a: &BorrowedTensor,
    p: f64,
    dim: Option<isize>,
    keepdim: bool,
) -> PyResult<OwnedTensor> {
    // Handle special cases
    if (p - 1.0).abs() < 1e-9 {
        // L1 norm: sum(|x|)
        let abs_typed = match a.dtype {
            DType::F32 => {
                let data = unsafe { typed_slice::<f32>(a) };
                let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
                let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
                for (i, &v) in data.iter().enumerate() {
                    out_data[i] = v.abs();
                }
                out
            }
            DType::F64 => {
                let data = unsafe { typed_slice::<f64>(a) };
                let mut out = OwnedTensor::new(DType::F64, a.shape.clone());
                let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
                for (i, &v) in data.iter().enumerate() {
                    out_data[i] = v.abs();
                }
                out
            }
            _ => return Err(unsupported("p_norm: only f32/f64 supported")),
        };
        return sum(&BorrowedTensor::from_owned(&abs_typed), dim, keepdim);
    }
    if (p - 2.0).abs() < 1e-9 {
        return norm(a, dim, keepdim);
    }
    if p.is_infinite() && p > 0.0 {
        // Linf norm: max(|x|)
        let abs_typed = match a.dtype {
            DType::F32 => {
                let data = unsafe { typed_slice::<f32>(a) };
                let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
                let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
                for (i, &v) in data.iter().enumerate() {
                    out_data[i] = v.abs();
                }
                out
            }
            DType::F64 => {
                let data = unsafe { typed_slice::<f64>(a) };
                let mut out = OwnedTensor::new(DType::F64, a.shape.clone());
                let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
                for (i, &v) in data.iter().enumerate() {
                    out_data[i] = v.abs();
                }
                out
            }
            _ => return Err(unsupported("p_norm: only f32/f64 supported")),
        };
        let (val, _) = max_reduce(&BorrowedTensor::from_owned(&abs_typed), dim, keepdim)?;
        return Ok(val);
    }
    if p == 0.0 {
        // L0: count of non-zero elements
        let indicator = match a.dtype {
            DType::F32 => {
                let data = unsafe { typed_slice::<f32>(a) };
                let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
                let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
                for (i, v) in data.iter().enumerate() {
                    out_data[i] = if *v != 0.0 { 1.0 } else { 0.0 };
                }
                out
            }
            DType::F64 => {
                let data = unsafe { typed_slice::<f64>(a) };
                let mut out = OwnedTensor::new(DType::F64, a.shape.clone());
                let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
                for (i, v) in data.iter().enumerate() {
                    out_data[i] = if *v != 0.0 { 1.0 } else { 0.0 };
                }
                out
            }
            _ => return Err(unsupported("p_norm: only f32/f64 supported")),
        };
        return sum(&BorrowedTensor::from_owned(&indicator), dim, keepdim);
    }
    // General p-norm: (sum(|x|^p))^(1/p)
    let inv_p = 1.0 / p;
    let pow_abs = match a.dtype {
        DType::F32 => {
            let data = unsafe { typed_slice::<f32>(a) };
            let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.abs().powf(p as f32);
            }
            out
        }
        DType::F64 => {
            let data = unsafe { typed_slice::<f64>(a) };
            let mut out = OwnedTensor::new(DType::F64, a.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.abs().powf(p);
            }
            out
        }
        _ => return Err(unsupported("p_norm: only f32/f64 supported")),
    };
    let summed = sum(&BorrowedTensor::from_owned(&pow_abs), dim, keepdim)?;
    match summed.dtype {
        DType::F32 => {
            let view = summed.as_view();
            let data = unsafe { typed_slice::<f32>(&view) };
            let mut out = OwnedTensor::new(DType::F32, summed.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.powf(inv_p as f32);
            }
            Ok(out)
        }
        DType::F64 => {
            let view = summed.as_view();
            let data = unsafe { typed_slice::<f64>(&view) };
            let mut out = OwnedTensor::new(DType::F64, summed.shape.clone());
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            for (i, &v) in data.iter().enumerate() {
                out_data[i] = v.powf(inv_p);
            }
            Ok(out)
        }
        _ => Err(unsupported("p_norm: only f32/f64 supported")),
    }
}
