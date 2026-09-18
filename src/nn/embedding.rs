//! Embedding lookup (Phase 4): row gather from a weight table.
//!
//! `embedding(weight, indices)` where `weight` is `[num_embeddings, D]` and
//! `indices` is an int64/int32 tensor of any shape `[...]`.  The output has
//! shape `indices.shape + [D]` and each output row is `weight[indices[...]]`.
//!
//! Requires indices in range `[0, num_embeddings)`; out-of-range indices
//! raise `TB_UNSUPPORTED` (delegates to eager, mirroring torch's error).

use crate::dlpack::{elem_count, unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;
use rayon::prelude::*;

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
        return Err(unsupported(
            "embedding weight must be 2D [num_embeddings, D]",
        ));
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

    let num_indices = elem_count(&indices.shape);
    let use_par = num_indices >= 16 * 1024;

    // Flattened index tensor: gather rows densely.
    match weight.dtype {
        DType::F32 => {
            let w = unsafe { typed_slice::<f32>(weight) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            match indices.dtype {
                DType::I64 => {
                    let idx = unsafe { typed_slice::<i64>(indices) };
                    if use_par {
                        for &ix in idx.iter() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                        }
                        // Process output in chunks — each chunk writes to disjoint [i*d .. (i+1)*d]
                        let chunk_size = 4096; // elements per chunk
                        out_data.par_chunks_mut(chunk_size).enumerate().for_each(
                            |(chunk_idx, chunk)| {
                                let offset = chunk_idx * chunk_size;
                                for (j, out_elem) in chunk.iter_mut().enumerate() {
                                    let flat_idx = offset + j;
                                    let row_idx = flat_idx / d;
                                    let col_idx = flat_idx % d;
                                    *out_elem = w[idx[row_idx] as usize * d + col_idx];
                                }
                            },
                        );
                    } else {
                        for (i, &ix) in idx.iter().enumerate() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                            out_data[i * d..(i + 1) * d]
                                .copy_from_slice(&w[row * d..(row + 1) * d]);
                        }
                    }
                }
                DType::I32 => {
                    let idx = unsafe { typed_slice::<i32>(indices) };
                    if use_par {
                        for &ix in idx.iter() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                        }
                        let chunk_size = 4096;
                        out_data.par_chunks_mut(chunk_size).enumerate().for_each(
                            |(chunk_idx, chunk)| {
                                let offset = chunk_idx * chunk_size;
                                for (j, out_elem) in chunk.iter_mut().enumerate() {
                                    let flat_idx = offset + j;
                                    let row_idx = flat_idx / d;
                                    let col_idx = flat_idx % d;
                                    *out_elem = w[idx[row_idx] as usize * d + col_idx];
                                }
                            },
                        );
                    } else {
                        for (i, &ix) in idx.iter().enumerate() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                            out_data[i * d..(i + 1) * d]
                                .copy_from_slice(&w[row * d..(row + 1) * d]);
                        }
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
                    if use_par {
                        for &ix in idx.iter() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                        }
                        let chunk_size = 4096;
                        out_data.par_chunks_mut(chunk_size).enumerate().for_each(
                            |(chunk_idx, chunk)| {
                                let offset = chunk_idx * chunk_size;
                                for (j, out_elem) in chunk.iter_mut().enumerate() {
                                    let flat_idx = offset + j;
                                    let row_idx = flat_idx / d;
                                    let col_idx = flat_idx % d;
                                    *out_elem = w[idx[row_idx] as usize * d + col_idx];
                                }
                            },
                        );
                    } else {
                        for (i, &ix) in idx.iter().enumerate() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                            out_data[i * d..(i + 1) * d]
                                .copy_from_slice(&w[row * d..(row + 1) * d]);
                        }
                    }
                }
                DType::I32 => {
                    let idx = unsafe { typed_slice::<i32>(indices) };
                    if use_par {
                        for &ix in idx.iter() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                        }
                        let chunk_size = 4096;
                        out_data.par_chunks_mut(chunk_size).enumerate().for_each(
                            |(chunk_idx, chunk)| {
                                let offset = chunk_idx * chunk_size;
                                for (j, out_elem) in chunk.iter_mut().enumerate() {
                                    let flat_idx = offset + j;
                                    let row_idx = flat_idx / d;
                                    let col_idx = flat_idx % d;
                                    *out_elem = w[idx[row_idx] as usize * d + col_idx];
                                }
                            },
                        );
                    } else {
                        for (i, &ix) in idx.iter().enumerate() {
                            let row = ix as usize;
                            if row >= num_embeddings {
                                return Err(unsupported(&format!(
                                    "embedding index {ix} out of range [0, {num_embeddings})"
                                )));
                            }
                            out_data[i * d..(i + 1) * d]
                                .copy_from_slice(&w[row * d..(row + 1) * d]);
                        }
                    }
                }
                _ => unreachable!("index dtype checked above"),
            }
        }
        _ => unreachable!("weight dtype checked above"),
    }
    Ok(out)
}

pub fn embedding_backward(
    grad_output: &BorrowedTensor,
    indices: &BorrowedTensor,
    num_weights: usize,
    padding_idx: i64,
    scale_grad_by_freq: bool,
) -> PyResult<OwnedTensor> {
    if grad_output.dtype != DType::F32 && grad_output.dtype != DType::F64 {
        return Err(unsupported("embedding_backward grad_output must be f32/f64"));
    }
    if indices.dtype != DType::I64 && indices.dtype != DType::I32 {
        return Err(unsupported("embedding_backward indices must be int64/int32"));
    }
    let grad_shape = &grad_output.shape;
    if grad_shape.is_empty() {
        return Err(unsupported("embedding_backward grad_output cannot be 0-D"));
    }
    let d = *grad_shape.last().unwrap() as usize;
    let n_indices = elem_count(&indices.shape);

    let num_weights = if num_weights > 0 {
        num_weights
    } else {
        let max_idx = match indices.dtype {
            DType::I64 => {
                let idx = unsafe { typed_slice::<i64>(indices) };
                idx.iter().copied().max().unwrap_or(0).max(0) as usize + 1
            }
            DType::I32 => {
                let idx = unsafe { typed_slice::<i32>(indices) };
                idx.iter().copied().max().unwrap_or(0).max(0) as usize + 1
            }
            _ => 1,
        };
        max_idx
    };

    let mut grad_weight =
        OwnedTensor::new_zeroed(grad_output.dtype, vec![num_weights as i64, d as i64]);

    let freqs = if scale_grad_by_freq {
        let mut f = vec![0usize; num_weights];
        match indices.dtype {
            DType::I64 => {
                let idx = unsafe { typed_slice::<i64>(indices) };
                for &ix in idx {
                    if ix >= 0 && (ix as usize) < num_weights && ix != padding_idx {
                        f[ix as usize] += 1;
                    }
                }
            }
            DType::I32 => {
                let idx = unsafe { typed_slice::<i32>(indices) };
                for &ix in idx {
                    if ix >= 0 && (ix as usize) < num_weights && (ix as i64) != padding_idx {
                        f[ix as usize] += 1;
                    }
                }
            }
            _ => {}
        }
        Some(f)
    } else {
        None
    };

    let get_idx = |i: usize| -> i64 {
        match indices.dtype {
            DType::I64 => unsafe { *(indices.data as *const i64).add(i) },
            DType::I32 => unsafe { *(indices.data as *const i32).add(i) as i64 },
            _ => -1,
        }
    };

    let use_par = n_indices * d >= 16 * 1024 && d >= 32;

    match grad_output.dtype {
        DType::F32 => {
            let g = unsafe { typed_slice::<f32>(grad_output) };
            let gw = unsafe { typed_mut_slice::<f32>(&mut grad_weight) };
            if use_par {
                const CHUNK_D: usize = 32;
                (0..d).into_par_iter().step_by(CHUNK_D).for_each(|col_start| {
                    let col_end = (col_start + CHUNK_D).min(d);
                    let col_width = col_end - col_start;
                    for i in 0..n_indices {
                        let ix = get_idx(i);
                        if ix < 0 || ix == padding_idx {
                            continue;
                        }
                        let row = ix as usize;
                        if row >= num_weights {
                            continue;
                        }
                        let scale = if let Some(ref fr) = freqs {
                            let cnt = fr[row];
                            if cnt > 0 {
                                1.0 / cnt as f32
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        };
                        let g_offset = i * d + col_start;
                        let gw_offset = row * d + col_start;
                        unsafe {
                            let gw_ptr = gw.as_ptr().add(gw_offset) as *mut f32;
                            let g_ptr = g.as_ptr().add(g_offset);
                            if scale == 1.0 {
                                for c in 0..col_width {
                                    *gw_ptr.add(c) += *g_ptr.add(c);
                                }
                            } else {
                                for c in 0..col_width {
                                    *gw_ptr.add(c) += *g_ptr.add(c) * scale;
                                }
                            }
                        }
                    }
                });
            } else {
                for i in 0..n_indices {
                    let ix = get_idx(i);
                    if ix < 0 || ix == padding_idx {
                        continue;
                    }
                    let row = ix as usize;
                    if row >= num_weights {
                        return Err(unsupported(&format!(
                            "embedding index {ix} out of range [0, {num_weights})"
                        )));
                    }
                    let scale = if let Some(ref fr) = freqs {
                        let cnt = fr[row];
                        if cnt > 0 {
                            1.0 / cnt as f32
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };
                    let g_row = &g[i * d..(i + 1) * d];
                    let gw_row = &mut gw[row * d..(row + 1) * d];
                    if scale == 1.0 {
                        for c in 0..d {
                            gw_row[c] += g_row[c];
                        }
                    } else {
                        for c in 0..d {
                            gw_row[c] += g_row[c] * scale;
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let g = unsafe { typed_slice::<f64>(grad_output) };
            let gw = unsafe { typed_mut_slice::<f64>(&mut grad_weight) };
            if use_par {
                const CHUNK_D: usize = 32;
                (0..d).into_par_iter().step_by(CHUNK_D).for_each(|col_start| {
                    let col_end = (col_start + CHUNK_D).min(d);
                    let col_width = col_end - col_start;
                    for i in 0..n_indices {
                        let ix = get_idx(i);
                        if ix < 0 || ix == padding_idx {
                            continue;
                        }
                        let row = ix as usize;
                        if row >= num_weights {
                            continue;
                        }
                        let scale = if let Some(ref fr) = freqs {
                            let cnt = fr[row];
                            if cnt > 0 {
                                1.0 / cnt as f64
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        };
                        let g_offset = i * d + col_start;
                        let gw_offset = row * d + col_start;
                        unsafe {
                            let gw_ptr = gw.as_ptr().add(gw_offset) as *mut f64;
                            let g_ptr = g.as_ptr().add(g_offset);
                            if scale == 1.0 {
                                for c in 0..col_width {
                                    *gw_ptr.add(c) += *g_ptr.add(c);
                                }
                            } else {
                                for c in 0..col_width {
                                    *gw_ptr.add(c) += *g_ptr.add(c) * scale;
                                }
                            }
                        }
                    }
                });
            } else {
                for i in 0..n_indices {
                    let ix = get_idx(i);
                    if ix < 0 || ix == padding_idx {
                        continue;
                    }
                    let row = ix as usize;
                    if row >= num_weights {
                        return Err(unsupported(&format!(
                            "embedding index {ix} out of range [0, {num_weights})"
                        )));
                    }
                    let scale = if let Some(ref fr) = freqs {
                        let cnt = fr[row];
                        if cnt > 0 {
                            1.0 / cnt as f64
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };
                    let g_row = &g[i * d..(i + 1) * d];
                    let gw_row = &mut gw[row * d..(row + 1) * d];
                    if scale == 1.0 {
                        for c in 0..d {
                            gw_row[c] += g_row[c];
                        }
                    } else {
                        for c in 0..d {
                            gw_row[c] += g_row[c] * scale;
                        }
                    }
                }
            }
        }
        _ => return Err(unsupported("embedding_backward unsupported dtype")),
    }

    Ok(grad_weight)
}

