//! Elementwise binary backward: add/sub/mul/div. Inherits autograd root via super-super; pure move.

use super::super::*;

// ---------------------------------------------------------------------------
// Backward implementations for each op
// ---------------------------------------------------------------------------

// --- add(a, b) → grad_a = grad_out, grad_b = grad_out ---
/// Record an elementwise add for autograd.
pub fn record_add(
    a_id: usize,
    b_id: usize,
    out_id: usize,
    a_data: &OwnedTensor,
    b_data: &OwnedTensor,
) {
    if !is_enabled() {
        return;
    }
    save_data(a_id, a_data);
    save_data(b_id, b_data);
    record(
        Box::new(ElemwiseBinaryBackward {
            op_type: BinOpType::Add,
            a_id,
            b_id,
        }),
        &[out_id],
        &[a_id, b_id],
    );
}

pub fn record_sub(
    a_id: usize,
    b_id: usize,
    out_id: usize,
    a_data: &OwnedTensor,
    b_data: &OwnedTensor,
) {
    if !is_enabled() {
        return;
    }
    save_data(a_id, a_data);
    save_data(b_id, b_data);
    record(
        Box::new(ElemwiseBinaryBackward {
            op_type: BinOpType::Sub,
            a_id,
            b_id,
        }),
        &[out_id],
        &[a_id, b_id],
    );
}

pub fn record_mul(
    a_id: usize,
    b_id: usize,
    out_id: usize,
    a_data: &OwnedTensor,
    b_data: &OwnedTensor,
) {
    if !is_enabled() {
        return;
    }
    save_data(a_id, a_data);
    save_data(b_id, b_data);
    record(
        Box::new(ElemwiseBinaryBackward {
            op_type: BinOpType::Mul,
            a_id,
            b_id,
        }),
        &[out_id],
        &[a_id, b_id],
    );
}

pub fn record_div(
    a_id: usize,
    b_id: usize,
    out_id: usize,
    a_data: &OwnedTensor,
    b_data: &OwnedTensor,
) {
    if !is_enabled() {
        return;
    }
    save_data(a_id, a_data);
    save_data(b_id, b_data);
    record(
        Box::new(ElemwiseBinaryBackward {
            op_type: BinOpType::Div,
            a_id,
            b_id,
        }),
        &[out_id],
        &[a_id, b_id],
    );
}

#[derive(Clone, Copy)]
pub(crate) enum BinOpType {
    Add,
    Sub,
    Mul,
    Div,
}

struct ElemwiseBinaryBackward {
    // dead code — used via record_add/sub/mul/div
    #[allow(dead_code)]
    op_type: BinOpType,
    a_id: usize,
    b_id: usize,
}

impl BackwardOp for ElemwiseBinaryBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        assert!(saved.len() >= 2);
        let a = saved[0];
        let b = saved[1];
        let n = elem_count(&upstream.shape);

        let mut grad_a = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
        let mut grad_b = OwnedTensor::new(upstream.dtype, upstream.shape.clone());

        match upstream.dtype {
            DType::F32 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n) };
                let ad = unsafe {
                    std::slice::from_raw_parts(a.data.as_ptr() as *const f32, elem_count(&a.shape))
                };
                let bd = unsafe {
                    std::slice::from_raw_parts(b.data.as_ptr() as *const f32, elem_count(&b.shape))
                };
                let ga = unsafe {
                    std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f32, n)
                };
                let gb = unsafe {
                    std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f32, n)
                };

                // Simple case: both same shape as upstream
                let a_n = elem_count(&a.shape);
                let b_n = elem_count(&b.shape);
                for i in 0..n {
                    let ai = if a_n == 1 { 0 } else { i % a_n };
                    let bi = if b_n == 1 { 0 } else { i % b_n };
                    match self.op_type {
                        BinOpType::Add => {
                            ga[i] = g[i];
                            gb[i] = g[i];
                        }
                        BinOpType::Sub => {
                            ga[i] = g[i];
                            gb[i] = -g[i];
                        }
                        BinOpType::Mul => {
                            ga[i] = g[i] * bd[bi];
                            gb[i] = g[i] * ad[ai];
                        }
                        BinOpType::Div => {
                            ga[i] = g[i] / bd[bi];
                            gb[i] = -g[i] * ad[ai] / (bd[bi] * bd[bi]);
                        }
                    }
                }
            }
            DType::F64 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n) };
                let ad = unsafe {
                    std::slice::from_raw_parts(a.data.as_ptr() as *const f64, elem_count(&a.shape))
                };
                let bd = unsafe {
                    std::slice::from_raw_parts(b.data.as_ptr() as *const f64, elem_count(&b.shape))
                };
                let ga = unsafe {
                    std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f64, n)
                };
                let gb = unsafe {
                    std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f64, n)
                };

                let a_n = elem_count(&a.shape);
                let b_n = elem_count(&b.shape);
                for i in 0..n {
                    let ai = if a_n == 1 { 0 } else { i % a_n };
                    let bi = if b_n == 1 { 0 } else { i % b_n };
                    match self.op_type {
                        BinOpType::Add => {
                            ga[i] = g[i];
                            gb[i] = g[i];
                        }
                        BinOpType::Sub => {
                            ga[i] = g[i];
                            gb[i] = -g[i];
                        }
                        BinOpType::Mul => {
                            ga[i] = g[i] * bd[bi];
                            gb[i] = g[i] * ad[ai];
                        }
                        BinOpType::Div => {
                            ga[i] = g[i] / bd[bi];
                            gb[i] = -g[i] * ad[ai] / (bd[bi] * bd[bi]);
                        }
                    }
                }
            }
            _ => {}
        }

        vec![(self.a_id, grad_a), (self.b_id, grad_b)]
    }
}
