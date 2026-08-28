//! Embedding lookup (Phase 4): row gather from a weight table.
//!
//! `embedding(weight, indices)` where `weight` is `[num_embeddings, D]` and
//! `indices` is an int64/int32 tensor of any shape `[...]`.  The output has
//! shape `indices.shape + [D]` and each output row is `weight[indices[...]]`.
//!
//! Requires indices in range `[0, num_embeddings)`; out-of-range indices
//! raise `TB_UNSUPPORTED` (delegates to eager, mirroring torch's error).

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, unsupported};
use pyo3::prelude::*;

/// Read a tensor's elements as a typed slice.
unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

/// Write typed data into an owned tensor.
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

pub fn embedding(weight: &BorrowedTensor, indices: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if weight.dtype != DType::F32 && weight.dtype != DType::F64 {
        return Err(unsupported("embedding weight must be f32/f64"));
    }
    if weight.shape.len() != 2 {
        return Err(unsupported("embedding weight must be 2D [num_embeddings, D]"));
    }
    if indices.dtype != DType::I64 && indices.dtype != DType::I32 {
        return Err(unsupported("embedding indices must be int64/int32"));
    }

    let num_embeddings = weight.shape[0] as usize;
    let d = weight.shape[1] as usize;

    // Output shape: indices.shape + [D]
    let mut out_shape = indices.shape.clone();
    out_shape.push(d as i64);
    let mut out = OwnedTensor::new(weight.dtype, out_shape);

    // Flattened index tensor: gather rows densely.
    match weight.dtype {
        DType::F32 => {
            let w = unsafe { typed_slice::<f32>(weight) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            match indices.dtype {
                DType::I64 => {
                    let idx = unsafe { typed_slice::<i64>(indices) };
                    for (i, &ix) in idx.iter().enumerate() {
                        let row = ix as usize;
                        if row >= num_embeddings {
                            return Err(unsupported(&format!(
                                "embedding index {ix} out of range [0, {num_embeddings})"
                            )));
                        }
                        out_data[i * d..(i + 1) * d].copy_from_slice(&w[row * d..(row + 1) * d]);
                    }
                }
                DType::I32 => {
                    let idx = unsafe { typed_slice::<i32>(indices) };
                    for (i, &ix) in idx.iter().enumerate() {
                        let row = ix as usize;
                        if row >= num_embeddings {
                            return Err(unsupported(&format!(
                                "embedding index {ix} out of range [0, {num_embeddings})"
                            )));
                        }
                        out_data[i * d..(i + 1) * d].copy_from_slice(&w[row * d..(row + 1) * d]);
                    }
                }
                _ => unreachable!("index dtype checked above"),
            }
        }
        DType::F64 => {
            let w = unsafe { typed_slice::<f64>(weight) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            match indices.dtype {
                DType::I64 => {
                    let idx = unsafe { typed_slice::<i64>(indices) };
                    for (i, &ix) in idx.iter().enumerate() {
                        let row = ix as usize;
                        if row >= num_embeddings {
                            return Err(unsupported(&format!(
                                "embedding index {ix} out of range [0, {num_embeddings})"
                            )));
                        }
                        out_data[i * d..(i + 1) * d].copy_from_slice(&w[row * d..(row + 1) * d]);
                    }
                }
                DType::I32 => {
                    let idx = unsafe { typed_slice::<i32>(indices) };
                    for (i, &ix) in idx.iter().enumerate() {
                        let row = ix as usize;
                        if row >= num_embeddings {
                            return Err(unsupported(&format!(
                                "embedding index {ix} out of range [0, {num_embeddings})"
                            )));
                        }
                        out_data[i * d..(i + 1) * d].copy_from_slice(&w[row * d..(row + 1) * d]);
                    }
                }
                _ => unreachable!("index dtype checked above"),
            }
        }
        _ => unreachable!("weight dtype checked above"),
    }
    Ok(out)
}
