//! Shape backward: sum, reshape, permute, cat. Inherits autograd root via super-super; pure move.

use super::super::*;

// --- sum(x, dim, keepdim) ---

#[allow(dead_code)]
struct SumBackward {
    input_id: usize,
    #[allow(dead_code)]
    input_shape: Vec<i64>,
    #[allow(dead_code)]
    dim: Option<isize>,
    #[allow(dead_code)]
    keepdim: bool,
}

impl BackwardOp for SumBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        _saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        // To undo a sum, broadcast the gradient back to the input shape.
        let grad = broadcast_to(upstream, &self.input_shape);
        vec![(self.input_id, grad)]
    }
}

pub fn record_sum(
    input_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    input_shape: &[i64],
    dim: Option<isize>,
    keepdim: bool,
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    record(
        Box::new(SumBackward {
            input_id,
            input_shape: input_shape.to_vec(),
            dim,
            keepdim,
        }),
        &[out_id],
        &[input_id],
    );
}

/// Broadcast a tensor to a target shape (right-aligned, numpy-style).
fn broadcast_to(t: &OwnedTensor, target_shape: &[i64]) -> OwnedTensor {
    let mut out = OwnedTensor::new(t.dtype, target_shape.to_vec());
    let out_n = elem_count(target_shape);
    let in_n = elem_count(&t.shape);
    if in_n == 0 || out_n == 0 {
        return out;
    }

    match t.dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f32, in_n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, out_n) };
            if in_n == 1 {
                dst.fill(src[0]);
            } else {
                // Simple linear broadcast for contiguous tensors
                for i in 0..out_n {
                    dst[i] = src[i % in_n];
                }
            }
        }
        DType::F64 => {
            let src = unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f64, in_n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, out_n) };
            if in_n == 1 {
                dst.fill(src[0]);
            } else {
                for i in 0..out_n {
                    dst[i] = src[i % in_n];
                }
            }
        }
        _ => {}
    }
    out
}

// --- reshape backward ---

#[allow(dead_code)]
struct ReshapeBackward {
    input_id: usize,
    #[allow(dead_code)]
    input_shape: Vec<i64>,
}

impl BackwardOp for ReshapeBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        _saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        vec![(self.input_id, broadcast_to(upstream, &self.input_shape))]
    }
}

pub fn record_reshape(
    input_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    input_shape: &[i64],
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    record(
        Box::new(ReshapeBackward {
            input_id,
            input_shape: input_shape.to_vec(),
        }),
        &[out_id],
        &[input_id],
    );
}

// --- permute backward ---

#[allow(dead_code)]
struct PermuteBackward {
    input_id: usize,
    #[allow(dead_code)]
    dims: Vec<isize>,
    #[allow(dead_code)]
    input_shape: Vec<i64>,
}

impl BackwardOp for PermuteBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        _saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        // Inverse permutation
        let n = self.dims.len();
        let mut inv = vec![0isize; n];
        for (i, &d) in self.dims.iter().enumerate() {
            inv[d as usize] = i as isize;
        }
        // Transpose the upstream using the inverse permutation
        vec![(self.input_id, permute_tensor(upstream, &inv))]
    }
}

fn permute_tensor(t: &OwnedTensor, dims: &[isize]) -> OwnedTensor {
    let rank = t.shape.len();
    assert_eq!(dims.len(), rank);
    let mut new_shape = vec![0i64; rank];
    for i in 0..rank {
        new_shape[i] = t.shape[dims[i] as usize];
    }
    let n = elem_count(&t.shape);
    let mut out = OwnedTensor::new(t.dtype, new_shape.clone());

    // For contiguous tensors, compute source index from output index
    match t.dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f32, n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, n) };
            let in_strides = contiguous_strides(&t.shape);
            let _out_strides = contiguous_strides(&new_shape);
            for i in 0..n {
                // Decompose output index into coords, then map through inverse perm
                let mut src_idx = 0usize;
                let mut tmp = i;
                for d in (0..rank).rev() {
                    let coord = tmp % (new_shape[d].max(1) as usize);
                    tmp /= new_shape[d].max(1) as usize;
                    // This coord maps to dim dims[d] in source
                    src_idx += coord * in_strides[dims[d] as usize] as usize;
                }
                dst[i] = src[src_idx];
            }
        }
        DType::F64 => {
            let src = unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f64, n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, n) };
            let in_strides = contiguous_strides(&t.shape);
            for i in 0..n {
                let mut src_idx = 0usize;
                let mut tmp = i;
                for d in (0..rank).rev() {
                    let coord = tmp % (new_shape[d].max(1) as usize);
                    tmp /= new_shape[d].max(1) as usize;
                    src_idx += coord * in_strides[dims[d] as usize] as usize;
                }
                dst[i] = src[src_idx];
            }
        }
        _ => {}
    }
    out
}

pub fn record_permute(input_id: usize, out_id: usize, input_data: &OwnedTensor, dims: &[isize]) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    record(
        Box::new(PermuteBackward {
            input_id,
            dims: dims.to_vec(),
            input_shape: input_data.shape.clone(),
        }),
        &[out_id],
        &[input_id],
    );
}

// --- cat backward ---

#[allow(dead_code)]
struct CatBackward {
    input_ids: Vec<usize>,
    #[allow(dead_code)]
    input_shapes: Vec<Vec<i64>>,
    #[allow(dead_code)]
    dim: isize,
}

impl BackwardOp for CatBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        _saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        // Split upstream along dim into chunks matching input sizes.
        let d = if self.dim < 0 {
            (upstream.shape.len() as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        let mut grads = Vec::new();
        let mut offset = 0usize;
        for shape in &self.input_shapes {
            let size = shape[d] as usize;
            let mut grad_shape = upstream.shape.clone();
            grad_shape[d] = size as i64;
            let n = elem_count(&grad_shape);
            let mut grad = OwnedTensor::new(upstream.dtype, grad_shape);

            match upstream.dtype {
                DType::F32 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f32,
                            elem_count(&upstream.shape),
                        )
                    };
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let _out_stride = contiguous_strides(&upstream.shape);
                    let chunk_size: usize = shape
                        .iter()
                        .skip(d + 1)
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let block = size * chunk_size;
                    for base in 0..(n / block) {
                        let src_start = base
                            * contiguous_strides(&upstream.shape)[d.max(0) as usize].max(1)
                                as usize
                            + offset * chunk_size;
                        // Copy block from src to dst
                        for i in 0..block {
                            let s = src_start + i;
                            if s < src.len() {
                                dst[base * block + i] = src[s];
                            }
                        }
                    }
                }
                DType::F64 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f64,
                            elem_count(&upstream.shape),
                        )
                    };
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let chunk_size: usize = shape
                        .iter()
                        .skip(d + 1)
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let block = size * chunk_size;
                    for base in 0..(n / block) {
                        let src_start = base
                            * contiguous_strides(&upstream.shape)[d.max(0) as usize].max(1)
                                as usize
                            + offset * chunk_size;
                        for i in 0..block {
                            let s = src_start + i;
                            if s < src.len() {
                                dst[base * block + i] = src[s];
                            }
                        }
                    }
                }
                _ => {}
            }
            offset += size;
            grads.push(grad);
        }

        self.input_ids
            .iter()
            .zip(grads.into_iter())
            .map(|(&id, g)| (id, g))
            .collect()
    }
}

pub fn record_cat(
    input_ids: &[usize],
    out_id: usize,
    input_data_list: &[&OwnedTensor],
    dim: isize,
) {
    if !is_enabled() {
        return;
    }
    for &_id in input_ids {
        // We can't clone here since we don't have OwnedTensor refs.
        // The caller must call save_data for each input before this.
    }
    let shapes: Vec<Vec<i64>> = input_data_list.iter().map(|t| t.shape.clone()).collect();
    record(
        Box::new(CatBackward {
            input_ids: input_ids.to_vec(),
            input_shapes: shapes,
            dim,
        }),
        &[out_id],
        input_ids,
    );
}
