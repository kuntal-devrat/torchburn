//! Unary backward: neg/sqrt/exp and friends. Inherits autograd root via super-super; pure move.

use super::super::*;

// --- Unary backward ops: relu, sigmoid, tanh, gelu, abs, neg, exp, log, sqrt, etc. ---

#[derive(Clone, Copy)]
pub(crate) enum UnaryOpType {
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Abs,
    Neg,
    Sign,
    Sqrt,
    Exp,
    Log,
    Reciprocal,
    Ceil,
    Floor,
    Silu,
    Elu,
    Selu,
    Softplus,
    Hardswish,
    Mish,
    Rsqrt,
    LeakyRelu,
    Pow,
}

struct UnaryBackward {
    #[allow(dead_code)]
    op_type: UnaryOpType,
    input_id: usize,
    /// Op parameters (e.g. negative_slope for leaky_relu, exp for pow).
    params: [f64; 2],
}

impl BackwardOp for UnaryBackward {
    fn backward(
        &self,
        upstream: &OwnedTensor,
        saved: &[&OwnedTensor],
    ) -> Vec<(usize, OwnedTensor)> {
        assert!(!saved.is_empty());
        let input = saved[0];
        let n = elem_count(&upstream.shape);
        let in_n = elem_count(&input.shape);
        let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());

        match upstream.dtype {
            DType::F32 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n) };
                let x =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f32, in_n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                };
                for i in 0..n {
                    let xi = if in_n == 1 { 0 } else { i % in_n };
                    out[i] = g[i] * self.unary_grad_f32(x[xi]);
                }
            }
            DType::F64 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n) };
                let x =
                    unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f64, in_n) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                };
                for i in 0..n {
                    let xi = if in_n == 1 { 0 } else { i % in_n };
                    out[i] = g[i] * self.unary_grad_f64(x[xi]);
                }
            }
            _ => {}
        }

        vec![(self.input_id, grad)]
    }
}

impl UnaryBackward {
    fn unary_grad_f32(&self, x: f32) -> f32 {
        match self.op_type {
            UnaryOpType::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            UnaryOpType::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            UnaryOpType::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            UnaryOpType::Gelu => {
                // tanh approximation derivative
                let c = 0.7978845608028654f32;
                let b = 0.044715f32;
                let x3 = x * x * x;
                let inner = c * (x + b * x3);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let d_inner = c * (1.0 + 3.0 * b * x * x);
                0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
            }
            UnaryOpType::Silu => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 + x * (1.0 - s))
            }
            UnaryOpType::Abs => {
                if x >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            UnaryOpType::Neg => -1.0,
            UnaryOpType::Sign => 0.0, // subgradient: 0 at 0
            UnaryOpType::Sqrt => 0.5 / x.sqrt(),
            UnaryOpType::Rsqrt => -0.5 * x.powf(-1.5),
            UnaryOpType::Exp => x.exp(),
            UnaryOpType::Log => 1.0 / x,
            UnaryOpType::Reciprocal => -1.0 / (x * x),
            UnaryOpType::Ceil => 0.0f32, // subgradient
            UnaryOpType::Floor => 0.0f32,
            UnaryOpType::Elu => {
                if x > 0.0 {
                    1.0f32
                } else {
                    self.params[0] as f32 * x.exp()
                }
            }
            UnaryOpType::Selu => {
                let alpha = 1.6732632423543772f32;
                let scale = 1.0507009873554805f32;
                if x > 0.0 {
                    scale
                } else {
                    scale * alpha * x.exp()
                }
            }
            UnaryOpType::Softplus => {
                let beta = self.params[0] as f32;
                let sig = 1.0 / (1.0 + (-beta * x).exp());
                sig
            }
            UnaryOpType::Hardswish => {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    1.0
                } else {
                    (2.0 * x + 3.0) / 6.0
                }
            }
            UnaryOpType::Mish => {
                let _sp = (1.0 + x.exp()).ln().tanh();
                let sig = 1.0 / (1.0 + (-x).exp());
                let omega = 1.0 + x * (1.0 - sig);
                sig * (omega + x * sig * (1.0 - omega * omega / (1.0 + x.exp())))
            }
            UnaryOpType::LeakyRelu => {
                if x > 0.0 {
                    1.0
                } else {
                    self.params[0] as f32
                }
            }
            UnaryOpType::Pow => {
                let e = self.params[0] as f32;
                e * x.powf(e - 1.0)
            }
        }
    }

    fn unary_grad_f64(&self, x: f64) -> f64 {
        match self.op_type {
            UnaryOpType::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            UnaryOpType::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            UnaryOpType::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            UnaryOpType::Gelu => {
                let c = 0.7978845608028654f64;
                let b = 0.044715f64;
                let x3 = x * x * x;
                let inner = c * (x + b * x3);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let d_inner = c * (1.0 + 3.0 * b * x * x);
                0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
            }
            UnaryOpType::Silu => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 + x * (1.0 - s))
            }
            UnaryOpType::Abs => {
                if x >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            UnaryOpType::Neg => -1.0,
            UnaryOpType::Sign => 0.0,
            UnaryOpType::Sqrt => 0.5 / x.sqrt(),
            UnaryOpType::Rsqrt => -0.5 * x.powf(-1.5),
            UnaryOpType::Exp => x.exp(),
            UnaryOpType::Log => 1.0 / x,
            UnaryOpType::Reciprocal => -1.0 / (x * x),
            UnaryOpType::Ceil => 0.0,
            UnaryOpType::Floor => 0.0,
            UnaryOpType::Elu => {
                if x > 0.0 {
                    1.0
                } else {
                    self.params[0] * x.exp()
                }
            }
            UnaryOpType::Selu => {
                let alpha = 1.6732632423543772f64;
                let scale = 1.0507009873554805f64;
                if x > 0.0 {
                    scale
                } else {
                    scale * alpha * x.exp()
                }
            }
            UnaryOpType::Softplus => {
                let beta = self.params[0];
                1.0 / (1.0 + (-beta * x).exp())
            }
            UnaryOpType::Hardswish => {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    1.0
                } else {
                    (2.0 * x + 3.0) / 6.0
                }
            }
            UnaryOpType::Mish => {
                let _sp = (1.0 + x.exp()).ln().tanh();
                let sig = 1.0 / (1.0 + (-x).exp());
                let omega = 1.0 + x * (1.0 - sig);
                sig * (omega + x * sig * (1.0 - omega * omega / (1.0 + x.exp())))
            }
            UnaryOpType::LeakyRelu => {
                if x > 0.0 {
                    1.0
                } else {
                    self.params[0]
                }
            }
            UnaryOpType::Pow => {
                let e = self.params[0];
                e * x.powf(e - 1.0)
            }
        }
    }
}

// Convenience recording functions for unary ops
#[allow(dead_code)]
pub(crate) fn record_unary(
    op_type: UnaryOpType,
    input_id: usize,
    out_id: usize,
    input_data: &OwnedTensor,
    params: [f64; 2],
) {
    if !is_enabled() {
        return;
    }
    save_data(input_id, input_data);
    record(
        Box::new(UnaryBackward {
            op_type,
            input_id,
            params,
        }),
        &[out_id],
        &[input_id],
    );
}
