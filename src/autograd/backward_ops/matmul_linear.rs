//! Matmul and linear backward. Inherits autograd root via super-super; pure move.

use super::super::*;

// --- matmul(a, b) → grad_a = grad_out @ b^T, grad_b = a^T @ grad_out ---

struct MatMulBackward {
    #[allow(dead_code)]
    a_id: usize,
    b_id: usize,
}

impl BackwardOp for MatMulBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        assert!(saved.len() >= 2);
        let a = saved[0];
        let b = saved[1];

        // grad_a = upstream @ b^T
        let b_t = transpose_2d(b);
        let grad_a = matmul_2d_same(upstream, &b_t);

        // grad_b = a^T @ upstream
        let a_t = transpose_2d(a);
        let grad_b = matmul_2d_same(&a_t, upstream);

        vec![(self.a_id, grad_a), (self.b_id, grad_b)]
    }
}

pub fn record_matmul(
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
        Box::new(MatMulBackward { a_id, b_id }),
        &[out_id],
        &[a_id, b_id],
    );
}

fn transpose_2d(t: &OwnedTensor) -> OwnedTensor {
    assert_eq!(t.shape.len(), 2);
    let m = t.shape[0] as usize;
    let n = t.shape[1] as usize;
    let mut out = OwnedTensor::new(t.dtype, vec![n as i64, m as i64]);
    match t.dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f32, m * n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, m * n) };
            for i in 0..m {
                for j in 0..n {
                    dst[j * m + i] = src[i * n + j];
                }
            }
        }
        DType::F64 => {
            let src = unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f64, m * n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, m * n) };
            for i in 0..m {
                for j in 0..n {
                    dst[j * m + i] = src[i * n + j];
                }
            }
        }
        _ => {}
    }
    out
}

/// Simple 2D matmul: (M,K) x (K,N) → (M,N).  Used inside backward only.
fn matmul_2d_same(a: &OwnedTensor, b: &OwnedTensor) -> OwnedTensor {
    let m = a.shape[0] as usize;
    let k = a.shape[1] as usize;
    let n = b.shape[1] as usize;
    let mut out = OwnedTensor::new(a.dtype, vec![m as i64, n as i64]);

    match a.dtype {
        DType::F32 => {
            let ad = unsafe { std::slice::from_raw_parts(a.data.as_ptr() as *const f32, m * k) };
            let bd = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const f32, k * n) };
            let od =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, m * n) };
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
            let ad = unsafe { std::slice::from_raw_parts(a.data.as_ptr() as *const f64, m * k) };
            let bd = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const f64, k * n) };
            let od =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, m * n) };
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
        _ => {}
    }
    out
}

// --- linear(input, weight, bias) → grad_input, grad_weight, grad_bias ---

struct LinearBackward {
    #[allow(dead_code)]
    input_id: usize,
    weight_id: usize,
    bias_id: Option<usize>,
}

impl BackwardOp for LinearBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        let input = saved[0];
        let weight = saved[1];

        // grad_input = upstream @ weight (weight is (O,I), no transpose needed)
        let grad_input = matmul_2d_same(upstream, weight);

        // grad_weight = input^T @ upstream
        let input_t = transpose_2d(input);
        let grad_weight = matmul_2d_same(&input_t, upstream);

        let mut result = vec![(self.input_id, grad_input), (self.weight_id, grad_weight)];

        // grad_bias = sum(upstream, dim=0)
        if let Some(bias_id) = self.bias_id {
            let grad_bias = sum_dim0(upstream);
            result.push((bias_id, grad_bias));
        }

        result
    }
}

pub fn record_linear(
    input_id: usize,
    weight_id: usize,
    bias_id: Option<usize>,
    out_id: usize,
    input_data: &OwnedTensor,
    weight_data: &OwnedTensor,
    bias_data: Option<&OwnedTensor>,
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    save_data(weight_id, weight_data);
    if let Some(bd) = bias_data {
        if let Some(bid) = bias_id {
            save_data(bid, bd);
        }
    }
    record(
        Box::new(LinearBackward {
            input_id,
            weight_id,
            bias_id,
        }),
        &[out_id],
        &[input_id, weight_id],
    );
}

/// Sum along dim=0, collapsing that dimension.  Used for bias grad.
fn sum_dim0(t: &OwnedTensor) -> OwnedTensor {
    assert!(t.shape.len() >= 2);
    let n = t.shape.len();
    let outer: usize = t.shape[..n - 1]
        .iter()
        .map(|&d| d.max(0) as usize)
        .product();
    let inner = *t
        .shape
        .last()
        .expect("shape guaranteed non-empty by assert") as usize;
    let mut out = OwnedTensor::new(t.dtype, vec![inner as i64]);

    match t.dtype {
        DType::F32 => {
            let src =
                unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f32, outer * inner) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, inner) };
            dst.fill(0.0);
            for i in 0..outer {
                for j in 0..inner {
                    dst[j] += src[i * inner + j];
                }
            }
        }
        DType::F64 => {
            let src =
                unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f64, outer * inner) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, inner) };
            dst.fill(0.0);
            for i in 0..outer {
                for j in 0..inner {
                    dst[j] += src[i * inner + j];
                }
            }
        }
        _ => {}
    }
    out
}
