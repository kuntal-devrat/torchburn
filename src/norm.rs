//! Normalization layers: layer_norm, batch_norm, group_norm, rms_norm.
//!
//! These are reduction + elementwise operations over the normalized dimensions.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, unsupported};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ---------------------------------------------------------------------------
// Layer Normalization: normalize over the last `normalized_shape` dims
// ---------------------------------------------------------------------------

pub fn layer_norm(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: &BorrowedTensor,
    eps: f64,
) -> PyResult<OwnedTensor> {
    if input.dtype != weight.dtype || input.dtype != bias.dtype {
        return Err(unsupported("layer_norm: dtype mismatch between input, weight, bias"));
    }

    let shape = &input.shape;
    let rank = shape.len();
    let normalized_dims = weight.shape.len(); // weight shape = normalized_shape
    let normalized_size: usize = weight.shape.iter().map(|&d| d.max(0) as usize).product();

    // The normalized dims are the last `normalized_dims` dims of input
    if rank < normalized_dims {
        return Err(unsupported("layer_norm: input rank < weight rank"));
    }

    let _batch_size: usize = shape[..rank - normalized_dims].iter().map(|&d| d.max(0) as usize).product();

    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());

    match input.dtype {
        DType::F32 => {
            let in_data = unsafe { typed_slice::<f32>(input) };
            let w_data = unsafe { typed_slice::<f32>(weight) };
            let b_data = unsafe { typed_slice::<f32>(bias) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let eps_f32 = eps as f32;
            let norm_f = normalized_size as f32;

            use rayon::prelude::*;
            out_data.par_chunks_mut(normalized_size).enumerate().for_each(|(b_idx, out_row)| {
                let base = b_idx * normalized_size;
                let row_in = &in_data[base..base + normalized_size];
                let mut sum = 0.0f32;
                for &v in row_in {
                    sum += v;
                }
                let mean = sum / norm_f;
                let mut var = 0.0f32;
                for &v in row_in {
                    let diff = v - mean;
                    var += diff * diff;
                }
                var /= norm_f;
                let inv_std = 1.0 / (var + eps_f32).sqrt();
                for i in 0..normalized_size {
                    out_row[i] = (row_in[i] - mean) * inv_std * w_data[i] + b_data[i];
                }
            });
        }
        DType::F64 => {
            let in_data = unsafe { typed_slice::<f64>(input) };
            let w_data = unsafe { typed_slice::<f64>(weight) };
            let b_data = unsafe { typed_slice::<f64>(bias) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            let norm_f = normalized_size as f64;

            use rayon::prelude::*;
            out_data.par_chunks_mut(normalized_size).enumerate().for_each(|(b_idx, out_row)| {
                let base = b_idx * normalized_size;
                let row_in = &in_data[base..base + normalized_size];
                let mut sum = 0.0f64;
                for &v in row_in {
                    sum += v;
                }
                let mean = sum / norm_f;
                let mut var = 0.0f64;
                for &v in row_in {
                    let diff = v - mean;
                    var += diff * diff;
                }
                var /= norm_f;
                let inv_std = 1.0 / (var + eps).sqrt();
                for i in 0..normalized_size {
                    out_row[i] = (row_in[i] - mean) * inv_std * w_data[i] + b_data[i];
                }
            });
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Batch Normalization
// ---------------------------------------------------------------------------

pub fn batch_norm(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: &BorrowedTensor,
    running_mean: &BorrowedTensor,
    running_var: &BorrowedTensor,
    eps: f64,
    training: bool,
) -> PyResult<OwnedTensor> {
    if input.dtype != weight.dtype || input.dtype != bias.dtype {
        return Err(unsupported("batch_norm: dtype mismatch"));
    }

    let shape = &input.shape;
    let rank = shape.len();
    if rank < 2 {
        return Err(unsupported("batch_norm: input must be at least 2D (N, C, ...)"));
    }

    let n = shape[0] as usize; // batch size
    let c = shape[1] as usize; // channels
    let spatial_size: usize = shape[2..].iter().map(|&d| d.max(0) as usize).product();

    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());

    match input.dtype {
        DType::F32 => {
            let in_data = unsafe { typed_slice::<f32>(input) };
            let w_data = unsafe { typed_slice::<f32>(weight) };
            let b_data = unsafe { typed_slice::<f32>(bias) };
            let rm_data = unsafe { typed_slice::<f32>(running_mean) };
            let rv_data = unsafe { typed_slice::<f32>(running_var) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let eps_f32 = eps as f32;

            for ch in 0..c {
                let mean = if training {
                    // Compute batch mean for this channel
                    let mut sum = 0.0f32;
                    for i in 0..n {
                        for s in 0..spatial_size {
                            let idx = i * c * spatial_size + ch * spatial_size + s;
                            sum += in_data[idx];
                        }
                    }
                    sum / (n * spatial_size) as f32
                } else {
                    rm_data[ch]
                };

                let var = if training {
                    let mut sum = 0.0f32;
                    for i in 0..n {
                        for s in 0..spatial_size {
                            let idx = i * c * spatial_size + ch * spatial_size + s;
                            let diff = in_data[idx] - mean;
                            sum += diff * diff;
                        }
                    }
                    sum / (n * spatial_size) as f32
                } else {
                    rv_data[ch]
                };

                let inv_std = 1.0 / (var + eps_f32).sqrt();

                for i in 0..n {
                    for s in 0..spatial_size {
                        let idx = i * c * spatial_size + ch * spatial_size + s;
                        let normalized = (in_data[idx] - mean) * inv_std;
                        out_data[idx] = normalized * w_data[ch] + b_data[ch];
                    }
                }
            }
        }
        DType::F64 => {
            let in_data = unsafe { typed_slice::<f64>(input) };
            let w_data = unsafe { typed_slice::<f64>(weight) };
            let b_data = unsafe { typed_slice::<f64>(bias) };
            let rm_data = unsafe { typed_slice::<f64>(running_mean) };
            let rv_data = unsafe { typed_slice::<f64>(running_var) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

            for ch in 0..c {
                let mean = if training {
                    let mut sum = 0.0f64;
                    for i in 0..n {
                        for s in 0..spatial_size {
                            let idx = i * c * spatial_size + ch * spatial_size + s;
                            sum += in_data[idx];
                        }
                    }
                    sum / (n * spatial_size) as f64
                } else {
                    rm_data[ch]
                };

                let var = if training {
                    let mut sum = 0.0f64;
                    for i in 0..n {
                        for s in 0..spatial_size {
                            let idx = i * c * spatial_size + ch * spatial_size + s;
                            let diff = in_data[idx] - mean;
                            sum += diff * diff;
                        }
                    }
                    sum / (n * spatial_size) as f64
                } else {
                    rv_data[ch]
                };

                let inv_std = 1.0 / (var + eps).sqrt();

                for i in 0..n {
                    for s in 0..spatial_size {
                        let idx = i * c * spatial_size + ch * spatial_size + s;
                        let normalized = (in_data[idx] - mean) * inv_std;
                        out_data[idx] = normalized * w_data[ch] + b_data[ch];
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
// Group Normalization
// ---------------------------------------------------------------------------

pub fn group_norm(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: &BorrowedTensor,
    num_groups: usize,
    eps: f64,
) -> PyResult<OwnedTensor> {
    if input.dtype != weight.dtype || input.dtype != bias.dtype {
        return Err(unsupported("group_norm: dtype mismatch"));
    }

    let shape = &input.shape;
    let rank = shape.len();
    if rank < 3 {
        return Err(unsupported("group_norm: input must be at least 3D (N, C, ...)"));
    }

    let n = shape[0] as usize;
    let c = shape[1] as usize;
    if c % num_groups != 0 {
        return Err(unsupported(&format!("group_norm: channels {} not divisible by groups {}", c, num_groups)));
    }

    let channels_per_group = c / num_groups;
    let spatial_size: usize = shape[2..].iter().map(|&d| d.max(0) as usize).product();
    let group_size = channels_per_group * spatial_size;

    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());

    match input.dtype {
        DType::F32 => {
            let in_data = unsafe { typed_slice::<f32>(input) };
            let w_data = unsafe { typed_slice::<f32>(weight) };
            let b_data = unsafe { typed_slice::<f32>(bias) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let eps_f32 = eps as f32;

            for batch in 0..n {
                for g in 0..num_groups {
                    let base = batch * c * spatial_size + g * channels_per_group * spatial_size;
                    // Compute group mean
                    let mut mean = 0.0f32;
                    for i in 0..group_size {
                        mean += in_data[base + i];
                    }
                    mean /= group_size as f32;
                    // Compute group variance
                    let mut var = 0.0f32;
                    for i in 0..group_size {
                        let diff = in_data[base + i] - mean;
                        var += diff * diff;
                    }
                    var /= group_size as f32;
                    let inv_std = 1.0 / (var + eps_f32).sqrt();
                    // Normalize and apply affine
                    for i in 0..group_size {
                        let ch = g * channels_per_group + (i / spatial_size);
                        let normalized = (in_data[base + i] - mean) * inv_std;
                        out_data[base + i] = normalized * w_data[ch] + b_data[ch];
                    }
                }
            }
        }
        DType::F64 => {
            let in_data = unsafe { typed_slice::<f64>(input) };
            let w_data = unsafe { typed_slice::<f64>(weight) };
            let b_data = unsafe { typed_slice::<f64>(bias) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

            for batch in 0..n {
                for g in 0..num_groups {
                    let base = batch * c * spatial_size + g * channels_per_group * spatial_size;
                    let mut mean = 0.0f64;
                    for i in 0..group_size { mean += in_data[base + i]; }
                    mean /= group_size as f64;
                    let mut var = 0.0f64;
                    for i in 0..group_size {
                        let diff = in_data[base + i] - mean;
                        var += diff * diff;
                    }
                    var /= group_size as f64;
                    let inv_std = 1.0 / (var + eps).sqrt();
                    for i in 0..group_size {
                        let ch = g * channels_per_group + (i / spatial_size);
                        let normalized = (in_data[base + i] - mean) * inv_std;
                        out_data[base + i] = normalized * w_data[ch] + b_data[ch];
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
// RMS Normalization: x / sqrt(mean(x^2) + eps) * weight
// ---------------------------------------------------------------------------

pub fn rms_norm(input: &BorrowedTensor, weight: &BorrowedTensor, eps: f64) -> PyResult<OwnedTensor> {
    if input.dtype != weight.dtype {
        return Err(unsupported("rms_norm: dtype mismatch"));
    }

    let shape = &input.shape;
    let normalized_size: usize = weight.shape.iter().map(|&d| d.max(0) as usize).product();
    let total_elements: usize = shape.iter().map(|&d| d.max(0) as usize).product::<usize>();
    let batch_size: usize = total_elements / normalized_size;

    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());

    match input.dtype {
        DType::F32 => {
            let in_data = unsafe { typed_slice::<f32>(input) };
            let w_data = unsafe { typed_slice::<f32>(weight) };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let eps_f32 = eps as f32;

            for b_idx in 0..batch_size {
                let base = b_idx * normalized_size;
                // Compute RMS
                let mut sum_sq = 0.0f32;
                for i in 0..normalized_size {
                    let v = in_data[base + i];
                    sum_sq += v * v;
                }
                let rms = (sum_sq / normalized_size as f32 + eps_f32).sqrt();
                // Normalize and scale
                for i in 0..normalized_size {
                    out_data[base + i] = (in_data[base + i] / rms) * w_data[i];
                }
            }
        }
        DType::F64 => {
            let in_data = unsafe { typed_slice::<f64>(input) };
            let w_data = unsafe { typed_slice::<f64>(weight) };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

            for b_idx in 0..batch_size {
                let base = b_idx * normalized_size;
                let mut sum_sq = 0.0f64;
                for i in 0..normalized_size {
                    let v = in_data[base + i];
                    sum_sq += v * v;
                }
                let rms = (sum_sq / normalized_size as f64 + eps).sqrt();
                for i in 0..normalized_size {
                    out_data[base + i] = (in_data[base + i] / rms) * w_data[i];
                }
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
    Ok(out)
}
