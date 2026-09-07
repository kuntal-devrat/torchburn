//! Softmax, layer-norm and dropout backward. Inherits autograd root via super-super; pure move.

use super::super::*;

// --- softmax(a, dim) → jacobian-based backward ---

#[allow(dead_code)]
struct SoftmaxBackward {
    input_id: usize,
    #[allow(dead_code)]
    dim: isize,
}

impl BackwardOp for SoftmaxBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        let output = saved[0]; // saved output = softmax(input)
        let n = elem_count(&upstream.shape);
        let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());

        // For each row along the softmax dim: grad_input = output * (upstream - dot(output, upstream))
        // Simplified: since we saved the softmax output, compute the Jacobian.
        match upstream.dtype {
            DType::F32 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n) };
                let s =
                    unsafe { std::slice::from_raw_parts(output.data.as_ptr() as *const f32, n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                };

                // Simplified: treat last dim as softmax dim (most common case)
                let dim_size = *upstream.shape.last().unwrap_or(&1) as usize;
                if dim_size > 0 {
                    for base in (0..n).step_by(dim_size) {
                        // dot = sum(s[base..] * g[base..])
                        let mut dot = 0.0f32;
                        for j in 0..dim_size {
                            dot += s[base + j] * g[base + j];
                        }
                        for j in 0..dim_size {
                            out[base + j] = s[base + j] * (g[base + j] - dot);
                        }
                    }
                }
            }
            DType::F64 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n) };
                let s =
                    unsafe { std::slice::from_raw_parts(output.data.as_ptr() as *const f64, n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                };

                let dim_size = *upstream.shape.last().unwrap_or(&1) as usize;
                if dim_size > 0 {
                    for base in (0..n).step_by(dim_size) {
                        let mut dot = 0.0f64;
                        for j in 0..dim_size {
                            dot += s[base + j] * g[base + j];
                        }
                        for j in 0..dim_size {
                            out[base + j] = s[base + j] * (g[base + j] - dot);
                        }
                    }
                }
            }
            _ => {}
        }

        vec![(self.input_id, grad)]
    }
}

pub fn record_softmax(
    input_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    output_data: &OwnedTensor,
    dim: isize,
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    save_data(out_id, output_data); // save the softmax output
    record(
        Box::new(SoftmaxBackward { input_id, dim }),
        &[out_id],
        &[input_id, out_id], // input_id is the one we grad w.r.t.
    );
}

// --- layer_norm(input, weight, bias, eps) ---

struct LayerNormBackward {
    input_id: usize,
    weight_id: usize,
    bias_id: usize,
}

impl BackwardOp for LayerNormBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        // saved[0] = input, saved[1] = normalized (unused), saved[2] = weight
        let input = saved[0];
        let weight = saved[2];

        let n = elem_count(&upstream.shape);
        let last_dim = *input.shape.last().unwrap_or(&1) as usize;
        let batch: usize = if last_dim > 0 { n / last_dim } else { 1 };
        let eps = 1e-5f64;

        let mut grad_input = OwnedTensor::new(upstream.dtype, input.shape.clone());
        let mut grad_weight = OwnedTensor::new(weight.dtype, weight.shape.clone());
        let mut grad_bias = if self.bias_id > 0 {
            Some(OwnedTensor::new(upstream.dtype, weight.shape.clone()))
        } else {
            None
        };

        match upstream.dtype {
            DType::F32 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n) };
                let xd =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n) };
                let wd = unsafe {
                    std::slice::from_raw_parts(weight.data.as_ptr() as *const f32, last_dim)
                };
                let gi = unsafe {
                    std::slice::from_raw_parts_mut(grad_input.data.as_mut_ptr() as *mut f32, n)
                };
                let gw = unsafe {
                    std::slice::from_raw_parts_mut(
                        grad_weight.data.as_mut_ptr() as *mut f32,
                        last_dim,
                    )
                };
                gw.fill(0.0);

                for b in 0..batch {
                    let base = b * last_dim;
                    // Compute mean
                    let mut mu = 0.0f32;
                    for j in 0..last_dim {
                        mu += xd[base + j];
                    }
                    mu /= last_dim as f32;
                    // Compute variance
                    let mut var = 0.0f32;
                    for j in 0..last_dim {
                        let d = xd[base + j] - mu;
                        var += d * d;
                    }
                    var /= last_dim as f32;
                    let inv_std = 1.0f32 / (var + eps as f32).sqrt();
                    // grad_x_hat = g * weight
                    let mut ghat_mean = 0.0f32;
                    for j in 0..last_dim {
                        ghat_mean += g[base + j] * wd[j];
                    }
                    ghat_mean /= last_dim as f32;
                    // mean(grad_x_hat * x_hat)
                    let mut ghat_xhat_mean = 0.0f32;
                    for j in 0..last_dim {
                        let xh = (xd[base + j] - mu) * inv_std;
                        ghat_xhat_mean += g[base + j] * wd[j] * xh;
                    }
                    ghat_xhat_mean /= last_dim as f32;
                    for j in 0..last_dim {
                        let xh = (xd[base + j] - mu) * inv_std;
                        gi[base + j] =
                            inv_std * (g[base + j] * wd[j] - ghat_mean - xh * ghat_xhat_mean);
                        gw[j] += g[base + j] * xh;
                    }
                }
            }
            DType::F64 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n) };
                let xd =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n) };
                let wd = unsafe {
                    std::slice::from_raw_parts(weight.data.as_ptr() as *const f64, last_dim)
                };
                let gi = unsafe {
                    std::slice::from_raw_parts_mut(grad_input.data.as_mut_ptr() as *mut f64, n)
                };
                let gw = unsafe {
                    std::slice::from_raw_parts_mut(
                        grad_weight.data.as_mut_ptr() as *mut f64,
                        last_dim,
                    )
                };
                gw.fill(0.0);

                for b in 0..batch {
                    let base = b * last_dim;
                    let mut mu = 0.0f64;
                    for j in 0..last_dim {
                        mu += xd[base + j];
                    }
                    mu /= last_dim as f64;
                    let mut var = 0.0f64;
                    for j in 0..last_dim {
                        let d = xd[base + j] - mu;
                        var += d * d;
                    }
                    var /= last_dim as f64;
                    let inv_std = 1.0f64 / (var + eps).sqrt();
                    let mut ghat_mean = 0.0f64;
                    for j in 0..last_dim {
                        ghat_mean += g[base + j] * wd[j];
                    }
                    ghat_mean /= last_dim as f64;
                    let mut ghat_xhat_mean = 0.0f64;
                    for j in 0..last_dim {
                        let xh = (xd[base + j] - mu) * inv_std;
                        ghat_xhat_mean += g[base + j] * wd[j] * xh;
                    }
                    ghat_xhat_mean /= last_dim as f64;
                    for j in 0..last_dim {
                        let xh = (xd[base + j] - mu) * inv_std;
                        gi[base + j] =
                            inv_std * (g[base + j] * wd[j] - ghat_mean - xh * ghat_xhat_mean);
                        gw[j] += g[base + j] * xh;
                    }
                }
            }
            _ => {}
        }

        // grad_bias = sum_over_batch(grad_output)
        if let Some(ref mut gb) = grad_bias {
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let gbb = unsafe {
                        std::slice::from_raw_parts_mut(gb.data.as_mut_ptr() as *mut f32, last_dim)
                    };
                    gbb.fill(0.0);
                    for b in 0..batch {
                        for j in 0..last_dim {
                            gbb[j] += g[b * last_dim + j];
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let gbb = unsafe {
                        std::slice::from_raw_parts_mut(gb.data.as_mut_ptr() as *mut f64, last_dim)
                    };
                    gbb.fill(0.0);
                    for b in 0..batch {
                        for j in 0..last_dim {
                            gbb[j] += g[b * last_dim + j];
                        }
                    }
                }
                _ => {}
            }
        }

        let mut result = vec![(self.input_id, grad_input), (self.weight_id, grad_weight)];
        if let Some(gb) = grad_bias {
            result.push((self.bias_id, gb));
        }
        result
    }
}

pub fn record_layer_norm(
    input_id: usize,
    weight_id: usize,
    bias_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    weight_data: &OwnedTensor,
    _bias_data: &OwnedTensor,
    normed_data: &OwnedTensor,
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    save_data(out_id, normed_data); // save normalized for grad_weight
    save_data(weight_id, weight_data);
    record(
        Box::new(LayerNormBackward {
            input_id,
            weight_id,
            bias_id,
        }),
        &[out_id],
        &[input_id, weight_id, bias_id],
    );
}

// --- dropout(x, p, training) ---

struct DropoutBackward {
    input_id: usize,
    mask: Vec<bool>,
    p: f64,
}

impl BackwardOp for DropoutBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        _saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        let n = elem_count(&upstream.shape);
        let scale = 1.0 / (1.0 - self.p);
        let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());

        match upstream.dtype {
            DType::F32 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                };
                for i in 0..n {
                    out[i] = if self.mask.get(i).copied().unwrap_or(false) {
                        g[i] * scale as f32
                    } else {
                        0.0
                    };
                }
            }
            DType::F64 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                };
                for i in 0..n {
                    out[i] = if self.mask.get(i).copied().unwrap_or(false) {
                        g[i] * scale
                    } else {
                        0.0
                    };
                }
            }
            _ => {}
        }

        vec![(self.input_id, grad)]
    }
}

pub fn record_dropout(
    input_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    mask: Vec<bool>,
    p: f64,
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    record(
        Box::new(DropoutBackward { input_id, mask, p }),
        &[out_id],
        &[input_id],
    );
}
