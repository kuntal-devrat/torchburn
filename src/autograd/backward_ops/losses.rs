//! NLL and MSE loss backward. Inherits autograd root via super-super; pure move.

use super::super::*;

// --- nll_loss backward ---

#[allow(dead_code)]
struct NllLossBackward {
    input_id: usize,
    #[allow(dead_code)]
    target_id: usize,
    #[allow(dead_code)]
    reduction: i64,
    #[allow(dead_code)]
    ignore_index: i64,
}

impl BackwardOp for NllLossBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        let input = saved[0]; // shape (N, C) or (C,)
        let target = saved[1]; // shape (N,) or scalar

        let n_classes = *input.shape.last().unwrap_or(&1) as usize;
        let n = elem_count(&input.shape);
        let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());

        match input.dtype {
            DType::F32 => {
                let _inp =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n) };
                let tgt = unsafe {
                    std::slice::from_raw_parts(
                        target.data.as_ptr() as *const f64,
                        elem_count(&target.shape),
                    )
                };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                };
                out.fill(0.0);

                let scale = match self.reduction {
                    0 => 1.0f32,                          // none
                    1 => 1.0f32 / (n / n_classes) as f32, // mean
                    _ => 1.0f32,                          // sum
                };
                let up = unsafe {
                    std::slice::from_raw_parts(
                        upstream.data.as_ptr() as *const f32,
                        elem_count(&upstream.shape),
                    )
                };

                let n_batch = n / n_classes;
                for b in 0..n_batch {
                    let t = tgt[b] as i64;
                    if t == self.ignore_index {
                        continue;
                    }
                    if t >= 0 && (t as usize) < n_classes {
                        out[b * n_classes + t as usize] = -scale
                            * if self.reduction == 1 || self.reduction == 2 {
                                up[0]
                            } else {
                                up[b]
                            };
                    }
                }
            }
            DType::F64 => {
                let _inp =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n) };
                let tgt = unsafe {
                    std::slice::from_raw_parts(
                        target.data.as_ptr() as *const f64,
                        elem_count(&target.shape),
                    )
                };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                };
                out.fill(0.0);

                let scale = match self.reduction {
                    0 => 1.0f64,
                    1 => 1.0f64 / (n / n_classes) as f64,
                    _ => 1.0f64,
                };
                let up = unsafe {
                    std::slice::from_raw_parts(
                        upstream.data.as_ptr() as *const f64,
                        elem_count(&upstream.shape),
                    )
                };

                let n_batch = n / n_classes;
                for b in 0..n_batch {
                    let t = tgt[b] as i64;
                    if t == self.ignore_index {
                        continue;
                    }
                    if t >= 0 && (t as usize) < n_classes {
                        out[b * n_classes + t as usize] = -scale
                            * if self.reduction == 1 || self.reduction == 2 {
                                up[0]
                            } else {
                                up[b]
                            };
                    }
                }
            }
            _ => {}
        }

        vec![(self.input_id, grad)]
    }
}

pub fn record_nll_loss(
    input_id: usize,
    target_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    target_data: &OwnedTensor,
    reduction: i64,
    ignore_index: i64,
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    save_data(target_id, target_data);
    record(
        Box::new(NllLossBackward {
            input_id,
            target_id,
            reduction,
            ignore_index,
        }),
        &[out_id],
        &[input_id, target_id],
    );
}

// --- mse_loss backward ---

#[allow(dead_code)]
struct MseLossBackward {
    input_id: usize,
    #[allow(dead_code)]
    target_id: usize,
    #[allow(dead_code)]
    reduction: i64,
}

impl BackwardOp for MseLossBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        let input = saved[0];
        let target = saved[1];
        let n = elem_count(&input.shape);
        let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());

        let scale = match self.reduction {
            1 => 2.0f64 / n as f64, // mean
            2 => 2.0f64,            // sum
            _ => 2.0f64,            // none
        };

        let up = unsafe {
            std::slice::from_raw_parts(
                upstream.data.as_ptr() as *const f64,
                elem_count(&upstream.shape),
            )
        };
        let up_val = if up.is_empty() { 1.0 } else { up[0] };

        match input.dtype {
            DType::F32 => {
                let inp =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n) };
                let tgt =
                    unsafe { std::slice::from_raw_parts(target.data.as_ptr() as *const f32, n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                };
                for i in 0..n {
                    out[i] = (scale * up_val) as f32 * (inp[i] - tgt[i]);
                }
            }
            DType::F64 => {
                let inp =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n) };
                let tgt =
                    unsafe { std::slice::from_raw_parts(target.data.as_ptr() as *const f64, n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                };
                for i in 0..n {
                    out[i] = scale * up_val * (inp[i] - tgt[i]);
                }
            }
            _ => {}
        }

        vec![(self.input_id, grad)]
    }
}

pub fn record_mse_loss(
    input_id: usize,
    target_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    target_data: &OwnedTensor,
    reduction: i64,
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    save_data(target_id, target_data);
    record(
        Box::new(MseLossBackward {
            input_id,
            target_id,
            reduction,
        }),
        &[out_id],
        &[input_id, target_id],
    );
}
