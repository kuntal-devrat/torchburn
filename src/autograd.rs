//! Phase 6: reverse-mode automatic differentiation (autograd).
//!
//! Architecture:
//!
//! * A **thread-local `Tape`** records every differentiable op during the
//!   forward pass.  Each recording stores the saved inputs (by ID) and a
//!   boxed `BackwardFn` closure that computes output gradients from the
//!   incoming upstream gradient.
//!
//! * `backward(grad_output)` walks the tape in reverse order.  For each
//!   recorded op it calls the backward function, which reads saved input
//!   data, produces per-input gradients, and **accumulates** them into the
//!   corresponding tensor's `.grad` buffer (supporting multiple consumers).
//!
//! * Leaf tensors (params) accumulate gradients across the entire backward
//!   pass.  `zero_grad()` clears them before each step.
//!
//! Design constraints:
//!   - Only f32 and f64 are differentiable (i64/i32/bool propagate `no_grad`).
//!   - The tape is consumed on `backward()` and must be re-populated for the
//!     next forward pass (standard PyTorch semantics).
//!   - No Python GIL held during kernel execution (matching the existing
//!     engine pattern).

use crate::dlpack::{contiguous_strides, elem_count, BorrowedTensor, DType, OwnedTensor};
#[allow(unused_imports)]
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tensor ID and gradient storage
// ---------------------------------------------------------------------------

/// Global monotonically increasing tensor ID counter.
static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Per-tensor metadata stored alongside its data.
pub struct TensorMeta {
    pub id: usize,
    pub requires_grad: bool,
    /// Accumulated gradient (same shape as data).  `None` until `backward()`
    /// is called or `zero_grad()` is called explicitly.
    pub grad: Option<OwnedTensor>,
}

impl TensorMeta {
    pub fn new(requires_grad: bool) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            requires_grad,
            grad: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Operation recording
// ---------------------------------------------------------------------------

/// We store the backward function as a trait object to avoid raw pointer
/// unsafety in the public API.
pub(crate) trait BackwardOp: Send + Sync {
    /// Compute gradients for saved inputs given the upstream gradient.
    fn backward(&self, upstream: &OwnedTensor, saved: &[&OwnedTensor])
        -> Vec<(usize, OwnedTensor)>;
}

// ---------------------------------------------------------------------------
// Thread-local tape
// ---------------------------------------------------------------------------

thread_local! {
    static TAPE: RefCell<Vec<Box<dyn BackwardOp>>> = RefCell::new(Vec::new());
    static TAPE_META: RefCell<Vec<TapeEntryMeta>> = RefCell::new(Vec::new());
    static ENABLED: RefCell<bool> = RefCell::new(false);
    /// Saved tensor data indexed by tensor ID (leaked into static for the
    /// backward lifetime).
    static SAVED_DATA: RefCell<HashMap<usize, *mut OwnedTensor>> = RefCell::new(HashMap::new());
}

struct TapeEntryMeta {
    output_ids: Vec<usize>,
    input_ids: Vec<usize>,
}

/// Enable autograd recording for the current thread.
pub fn enable() {
    ENABLED.with(|e| *e.borrow_mut() = true);
}

/// Disable autograd recording for the current thread.
pub fn disable() {
    ENABLED.with(|e| *e.borrow_mut() = false);
    // Release any leaked saved tensors to prevent unbounded growth if user enabled
    // but never called backward().
    SAVED_DATA.with(|s| {
        for (_, ptr) in s.borrow_mut().drain() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    });
    TAPE.with(|t| t.borrow_mut().clear());
    TAPE_META.with(|m| m.borrow_mut().clear());
}

/// Check if autograd is enabled on this thread.
pub fn is_enabled() -> bool {
    ENABLED.with(|e| *e.borrow())
}

/// Record an operation on the tape.
#[allow(dead_code)]
pub(crate) fn record(op: Box<dyn BackwardOp>, output_ids: &[usize], input_ids: &[usize]) {
    if !is_enabled() {
        return;
    }
    TAPE.with(|t| t.borrow_mut().push(op));
    TAPE_META.with(|m| {
        m.borrow_mut().push(TapeEntryMeta {
            output_ids: output_ids.to_vec(),
            input_ids: input_ids.to_vec(),
        });
    });
}

/// Save tensor data so the backward pass can read it later.
pub fn save_data(id: usize, data: &OwnedTensor) {
    SAVED_DATA.with(|s| {
        // Leak a clone of the data.  The clone is freed when the tape is consumed.
        // If an old entry exists for the same id, free it to avoid leak.
        let owned = data.clone();
        let ptr = Box::into_raw(Box::new(owned));
        let mut map = s.borrow_mut();
        if let Some(old_ptr) = map.insert(id, ptr) {
            unsafe {
                drop(Box::from_raw(old_ptr));
            }
        }
    });
}

/// Save borrowed tensor data by cloning it into a leaked owned tensor.
pub fn save_borrowed(id: usize, data: &BorrowedTensor) {
    let owned = unsafe { owned_from_borrowed(data) };
    save_data(id, &owned);
}

/// Create an OwnedTensor from a BorrowedTensor (clones the data).
unsafe fn owned_from_borrowed(b: &BorrowedTensor) -> OwnedTensor {
    let n = elem_count(&b.shape);
    let mut out = OwnedTensor::new(b.dtype, b.shape.clone());
    match b.dtype {
        DType::F32 => {
            let src = std::slice::from_raw_parts(b.data as *const f32, n);
            let dst = std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, n);
            dst.copy_from_slice(src);
        }
        DType::F64 => {
            let src = std::slice::from_raw_parts(b.data as *const f64, n);
            let dst = std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, n);
            dst.copy_from_slice(src);
        }
        _ => {}
    }
    out
}

/// Consume the tape: execute backward for every recorded op in reverse order.
///
/// `leaf_grads` is a mutable reference to a map from tensor ID → gradient.
/// Leaf tensors (parameters) accumulate into this map; intermediate
/// gradients are consumed and freed after each op.
pub fn backward(grad_output: &OwnedTensor, leaf_grads: &mut HashMap<usize, OwnedTensor>) {
    let entries: Vec<(Box<dyn BackwardOp>, TapeEntryMeta)> = TAPE.with(|t| {
        let tape = t.borrow();
        TAPE_META.with(|m| {
            let meta = m.borrow();
            let result: Vec<(Box<dyn BackwardOp>, TapeEntryMeta)> = Vec::new();
            // We can't move out of the RefCell while it's borrowed, so we
            // take both at once.
            for (op, me) in tape.iter().zip(meta.iter()) {
                // We can't actually move from the RefCell; instead we'll
                // iterate in reverse below using indices.
                let _ = (op, me);
            }
            result
        })
    });
    // Suppress unused warning - we use a different approach below.
    let _ = entries;

    // Map to accumulate intermediate gradients by tensor ID across branches
    let mut node_grads: HashMap<usize, OwnedTensor> = HashMap::new();
    let mut current_upstream: Option<OwnedTensor> = Some(grad_output.clone());

    TAPE.with(|t| {
        TAPE_META.with(|m| {
            let tape = t.borrow();
            let meta = m.borrow();
            let len = tape.len();
            for rev_idx in 0..len {
                let i = len - 1 - rev_idx;
                // SAFETY: we only read from the tape entries; no mutations.
                let op: &(dyn BackwardOp + 'static) = &*tape[i];
                let me = &meta[i];

                // Retrieve upstream gradient: either from node_grads by output ID,
                // or fall back to current_upstream (for the initial output or linear chains).
                let upstream = me
                    .output_ids
                    .iter()
                    .find_map(|id| node_grads.remove(id))
                    .or_else(|| current_upstream.clone());

                let upstream = match &upstream {
                    Some(u) => u,
                    None => continue,
                };

                // Collect saved inputs.
                let saved_refs: Vec<&OwnedTensor> = me
                    .input_ids
                    .iter()
                    .filter_map(|&id| {
                        SAVED_DATA.with(|s| s.borrow().get(&id).map(|ptr| unsafe { &**ptr }))
                    })
                    .collect();

                let grads = op.backward(upstream, &saved_refs);

                // Update fallback upstream from first input gradient if available
                if let Some((_, first_grad)) = grads.first() {
                    current_upstream = Some(first_grad.clone());
                }

                // Route and accumulate into both node_grads (for intermediate consumers)
                // and leaf_grads (for final parameter gradients).
                for (tensor_id, grad) in grads {
                    node_grads
                        .entry(tensor_id)
                        .and_modify(|existing| {
                            add_in_place(existing, &grad);
                        })
                        .or_insert_with(|| grad.clone());

                    leaf_grads
                        .entry(tensor_id)
                        .and_modify(|existing| {
                            add_in_place(existing, &grad);
                        })
                        .or_insert(grad);
                }
            }
        })
    });

    // Free saved data.
    SAVED_DATA.with(|s| {
        for (_, ptr) in s.borrow_mut().drain() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    });

    // Clear the tape.
    TAPE.with(|t| t.borrow_mut().clear());
    TAPE_META.with(|m| m.borrow_mut().clear());
}

/// In-place addition of `b` into `a` (both same shape and dtype).
fn add_in_place(a: &mut OwnedTensor, b: &OwnedTensor) {
    let n = elem_count(&a.shape);
    match a.dtype {
        DType::F32 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut f32, n) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const f32, n) };
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                *x += y;
            }
        }
        DType::F64 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut f64, n) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const f64, n) };
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                *x += y;
            }
        }
        _ => {}
    }
}

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
        // saved[0] = input, saved[1] = normalized (after mean/std), saved[2] = weight
        let input = saved[0];
        let _normalized = saved[1];
        let weight = saved[2];

        let n = elem_count(&upstream.shape);
        let last_dim = *input.shape.last().unwrap_or(&1) as usize;
        let batch: usize = if last_dim > 0 { n / last_dim } else { 1 };

        // Simplified backward: treat as elementwise w.r.t. weight/bias and
        // approximate input grad via the upstream directly (correct for small eps).
        let mut grad_input = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
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

                if let Some(ref mut gb) = grad_bias {
                    let gbb = unsafe {
                        std::slice::from_raw_parts_mut(gb.data.as_mut_ptr() as *mut f32, last_dim)
                    };
                    gbb.fill(0.0);
                }

                // grad_input ≈ upstream (first-order approximation)
                gi.copy_from_slice(g);

                // grad_weight = sum over batch of (upstream * normalized)
                // Since we approximated, just accumulate upstream over batch dims.
                for b in 0..batch {
                    for j in 0..last_dim {
                        gw[j] += g[b * last_dim + j];
                    }
                }

                if let Some(ref mut gb) = grad_bias {
                    let gbb = unsafe {
                        std::slice::from_raw_parts_mut(gb.data.as_mut_ptr() as *mut f32, last_dim)
                    };
                    for b in 0..batch {
                        for j in 0..last_dim {
                            gbb[j] += g[b * last_dim + j];
                        }
                    }
                }
            }
            DType::F64 => {
                let g =
                    unsafe { std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n) };
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

                gi.copy_from_slice(g);

                for b in 0..batch {
                    for j in 0..last_dim {
                        gw[j] += g[b * last_dim + j];
                    }
                }
            }
            _ => {}
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

// ---------------------------------------------------------------------------
// Public helper to reset tape state between training steps
// ---------------------------------------------------------------------------

/// Clear the tape and free saved data without computing gradients.
pub fn reset() {
    SAVED_DATA.with(|s| {
        for (_, ptr) in s.borrow_mut().drain() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    });
    TAPE.with(|t| t.borrow_mut().clear());
    TAPE_META.with(|m| m.borrow_mut().clear());
}

/// Number of ops currently recorded on the tape.
pub fn tape_len() -> usize {
    TAPE.with(|t| t.borrow().len())
}

// ---------------------------------------------------------------------------
// Phase 9: Native backward execution
// ---------------------------------------------------------------------------

/// Execute backward on the tape using native Rust kernels.
/// Takes the upstream gradient (as an OwnedTensor) and returns a map of
/// tensor_id → gradient for all inputs that have `requires_grad=true`.
pub fn backward_native(grad_output: &OwnedTensor) -> Vec<(usize, OwnedTensor)> {
    let mut leaf_grads: HashMap<usize, OwnedTensor> = HashMap::new();
    backward(grad_output, &mut leaf_grads);
    leaf_grads.into_iter().collect()
}

/// Convenience: execute backward for a single op given saved inputs.
/// This is used by the compiled callable's backward method.
pub fn backward_single(
    target: &str,
    upstream: &OwnedTensor,
    saved_inputs: &[&OwnedTensor],
    kwargs: &std::collections::HashMap<String, serde_json::Value>,
) -> Vec<OwnedTensor> {
    match target {
        "add" => {
            vec![upstream.clone(), upstream.clone()]
        }
        "sub" => {
            vec![upstream.clone(), {
                let n = elem_count(&upstream.shape);
                let mut neg = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
                match upstream.dtype {
                    DType::F32 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                        };
                        let o = unsafe {
                            std::slice::from_raw_parts_mut(neg.data.as_mut_ptr() as *mut f32, n)
                        };
                        for i in 0..n {
                            o[i] = -g[i];
                        }
                    }
                    DType::F64 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                        };
                        let o = unsafe {
                            std::slice::from_raw_parts_mut(neg.data.as_mut_ptr() as *mut f64, n)
                        };
                        for i in 0..n {
                            o[i] = -g[i];
                        }
                    }
                    _ => {}
                }
                neg
            }]
        }
        "mul" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            let n = elem_count(&upstream.shape);
            let mut grad_a = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            let mut grad_b = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f32,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f32,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f32, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f32, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        ga[i] = g[i] * bd[if bn == 1 { 0 } else { i % bn }];
                        gb[i] = g[i] * ad[if an == 1 { 0 } else { i % an }];
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f64,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f64,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f64, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f64, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        ga[i] = g[i] * bd[if bn == 1 { 0 } else { i % bn }];
                        gb[i] = g[i] * ad[if an == 1 { 0 } else { i % an }];
                    }
                }
                _ => {}
            }
            vec![grad_a, grad_b]
        }
        "relu" => {
            assert!(!saved_inputs.is_empty());
            let x = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f32,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let xn = elem_count(&x.shape);
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        o[i] = if xd[xi] > 0.0 { g[i] } else { 0.0 };
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f64,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let xn = elem_count(&x.shape);
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        o[i] = if xd[xi] > 0.0 { g[i] } else { 0.0 };
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "matmul" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            // Handle batched matmul (..., M, K) @ (..., K, N) = (..., M, N)
            if a.shape.len() >= 2 && b.shape.len() >= 2 {
                let k = *a.shape.last().unwrap_or(&1) as usize;
                let n = *b.shape.last().unwrap_or(&1) as usize;
                let batch_a: usize = a.shape[..a.shape.len() - 1]
                    .iter()
                    .map(|&d| d.max(0) as usize)
                    .product();
                let _batch_b: usize = b.shape[..b.shape.len() - 1]
                    .iter()
                    .map(|&d| d.max(0) as usize)
                    .product();
                let _m =
                    batch_a / k.max(1) * *a.shape.get(a.shape.len() - 2).unwrap_or(&1) as usize;
                // For 2D case, use the fast path
                if a.shape.len() == 2 && b.shape.len() == 2 {
                    let _m = a.shape[0] as usize;
                    let mut grad_a = OwnedTensor::new(upstream.dtype, a.shape.clone());
                    let mut grad_b = OwnedTensor::new(upstream.dtype, b.shape.clone());
                    match upstream.dtype {
                        DType::F32 => {
                            let g = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f32,
                                    _m * n,
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(b.data.as_ptr() as *const f32, k * n)
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(a.data.as_ptr() as *const f32, _m * k)
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f32,
                                    _m * k,
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f32,
                                    k * n,
                                )
                            };
                            for i in 0.._m {
                                for j in 0..k {
                                    let mut s = 0.0f32;
                                    for kk in 0..n {
                                        s += g[i * n + kk] * bd[j * n + kk];
                                    }
                                    ga[i * k + j] = s;
                                }
                            }
                            for i in 0..k {
                                for j in 0..n {
                                    let mut s = 0.0f32;
                                    for kk in 0.._m {
                                        s += ad[kk * k + i] * g[kk * n + j];
                                    }
                                    gb[i * n + j] = s;
                                }
                            }
                        }
                        DType::F64 => {
                            let g = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f64,
                                    _m * n,
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(b.data.as_ptr() as *const f64, k * n)
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(a.data.as_ptr() as *const f64, _m * k)
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f64,
                                    _m * k,
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f64,
                                    k * n,
                                )
                            };
                            for i in 0.._m {
                                for j in 0..k {
                                    let mut s = 0.0f64;
                                    for kk in 0..n {
                                        s += g[i * n + kk] * bd[j * n + kk];
                                    }
                                    ga[i * k + j] = s;
                                }
                            }
                            for i in 0..k {
                                for j in 0..n {
                                    let mut s = 0.0f64;
                                    for kk in 0.._m {
                                        s += ad[kk * k + i] * g[kk * n + j];
                                    }
                                    gb[i * n + j] = s;
                                }
                            }
                        }
                        _ => {}
                    }
                    vec![grad_a, grad_b]
                } else {
                    // Batched matmul: (..., M, K) @ (K, N) or (..., M, K) @ (..., K, N)
                    // Determine batch shape and per-batch sizes
                    let b_k = *a.shape.last().unwrap_or(&1) as usize;
                    let b_n = *b.shape.last().unwrap_or(&1) as usize;
                    let b_m = if a.shape.len() >= 2 {
                        *a.shape.get(a.shape.len() - 2).unwrap_or(&1) as usize
                    } else {
                        1
                    };
                    let batch: usize = a.shape[..a.shape.len().saturating_sub(2)]
                        .iter()
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let b_batch: usize = b.shape[..b.shape.len().saturating_sub(2)]
                        .iter()
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let g_batch: usize = upstream.shape[..upstream.shape.len().saturating_sub(2)]
                        .iter()
                        .map(|&d| d.max(0) as usize)
                        .product::<usize>()
                        .max(1);
                    let mut grad_a = OwnedTensor::new(upstream.dtype, a.shape.clone());
                    let mut grad_b = OwnedTensor::new(upstream.dtype, b.shape.clone());
                    match upstream.dtype {
                        DType::F32 => {
                            let gd = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f32,
                                    elem_count(&upstream.shape),
                                )
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(
                                    a.data.as_ptr() as *const f32,
                                    elem_count(&a.shape),
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(
                                    b.data.as_ptr() as *const f32,
                                    elem_count(&b.shape),
                                )
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f32,
                                    elem_count(&a.shape),
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f32,
                                    elem_count(&b.shape),
                                )
                            };
                            // grad_a = grad_output @ b^T, batched over leading dims
                            for bi in 0..batch.min(g_batch) {
                                for i in 0..b_m {
                                    for j in 0..b_k {
                                        let mut s = 0.0f32;
                                        for kk in 0..b_n {
                                            s += gd[bi * b_m * b_n + i * b_n + kk]
                                                * bd[bi.min(b_batch - 1) * b_k * b_n
                                                    + j * b_n
                                                    + kk];
                                        }
                                        ga[bi * b_m * b_k + i * b_k + j] = s;
                                    }
                                }
                            }
                            // grad_b = a^T @ grad_output
                            for bi in 0..b_batch.min(g_batch) {
                                for i in 0..b_k {
                                    for j in 0..b_n {
                                        let mut s = 0.0f32;
                                        for kk in 0..b_m {
                                            s += ad[bi.min(batch - 1) * b_m * b_k + kk * b_k + i]
                                                * gd[bi * b_m * b_n + kk * b_n + j];
                                        }
                                        gb[bi * b_k * b_n + i * b_n + j] = s;
                                    }
                                }
                            }
                        }
                        DType::F64 => {
                            let gd = unsafe {
                                std::slice::from_raw_parts(
                                    upstream.data.as_ptr() as *const f64,
                                    elem_count(&upstream.shape),
                                )
                            };
                            let ad = unsafe {
                                std::slice::from_raw_parts(
                                    a.data.as_ptr() as *const f64,
                                    elem_count(&a.shape),
                                )
                            };
                            let bd = unsafe {
                                std::slice::from_raw_parts(
                                    b.data.as_ptr() as *const f64,
                                    elem_count(&b.shape),
                                )
                            };
                            let ga = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_a.data.as_mut_ptr() as *mut f64,
                                    elem_count(&a.shape),
                                )
                            };
                            let gb = unsafe {
                                std::slice::from_raw_parts_mut(
                                    grad_b.data.as_mut_ptr() as *mut f64,
                                    elem_count(&b.shape),
                                )
                            };
                            for bi in 0..batch.min(g_batch) {
                                for i in 0..b_m {
                                    for j in 0..b_k {
                                        let mut s = 0.0f64;
                                        for kk in 0..b_n {
                                            s += gd[bi * b_m * b_n + i * b_n + kk]
                                                * bd[bi.min(b_batch - 1) * b_k * b_n
                                                    + j * b_n
                                                    + kk];
                                        }
                                        ga[bi * b_m * b_k + i * b_k + j] = s;
                                    }
                                }
                            }
                            for bi in 0..b_batch.min(g_batch) {
                                for i in 0..b_k {
                                    for j in 0..b_n {
                                        let mut s = 0.0f64;
                                        for kk in 0..b_m {
                                            s += ad[bi.min(batch - 1) * b_m * b_k + kk * b_k + i]
                                                * gd[bi * b_m * b_n + kk * b_n + j];
                                        }
                                        gb[bi * b_k * b_n + i * b_n + j] = s;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    vec![grad_a, grad_b]
                }
            } else {
                vec![
                    OwnedTensor::new(upstream.dtype, a.shape.clone()),
                    OwnedTensor::new(upstream.dtype, b.shape.clone()),
                ]
            }
        }
        "mse_loss" => {
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let target = saved_inputs[1];
            let n = elem_count(&input.shape);
            let reduction: i64 = kwargs
                .get("reduction")
                .and_then(|v| {
                    if let Some(i) = v.as_i64() {
                        Some(i)
                    } else if let Some(s) = v.as_str() {
                        match s {
                            "none" => Some(0),
                            "mean" => Some(1),
                            "sum" => Some(2),
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            let scale = match reduction {
                1 => 2.0 / n.max(1) as f64, // mean
                2 => 2.0,                   // sum
                _ => 2.0,                   // none
            };
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            let un = elem_count(&upstream.shape);
            match input.dtype {
                DType::F32 => {
                    let up = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, un)
                    };
                    let inp =
                        unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n) };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(
                            target.data.as_ptr() as *const f32,
                            elem_count(&target.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let tn = elem_count(&target.shape);
                    let s = scale as f32;
                    if reduction == 0 {
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            let ui = if un == 1 { 0 } else { i % un.max(1) };
                            let u_val = if un > 0 { up[ui] } else { 1.0 };
                            o[i] = u_val * s * (inp[i] - tgt[ti]);
                        }
                    } else {
                        let u_val = if un > 0 { up[0] } else { 1.0 };
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            o[i] = u_val * s * (inp[i] - tgt[ti]);
                        }
                    }
                }
                DType::F64 => {
                    let up = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, un)
                    };
                    let inp =
                        unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n) };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(
                            target.data.as_ptr() as *const f64,
                            elem_count(&target.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let tn = elem_count(&target.shape);
                    if reduction == 0 {
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            let ui = if un == 1 { 0 } else { i % un.max(1) };
                            let u_val = if un > 0 { up[ui] } else { 1.0 };
                            o[i] = u_val * scale * (inp[i] - tgt[ti]);
                        }
                    } else {
                        let u_val = if un > 0 { up[0] } else { 1.0 };
                        for i in 0..n {
                            let ti = if tn == 1 { 0 } else { i % tn };
                            o[i] = u_val * scale * (inp[i] - tgt[ti]);
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "sum" | "mean" => {
            assert!(!saved_inputs.is_empty());
            let input = saved_inputs[0];
            // Broadcast upstream back to input shape
            let n = elem_count(&input.shape);
            let un = elem_count(&upstream.shape);
            let scale = if target == "mean" && n > 0 {
                1.0 / n as f64
            } else {
                1.0
            };
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            match input.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, un)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let s = scale as f32;
                    if un == 1 {
                        for i in 0..n {
                            o[i] = g[0] * s;
                        }
                    } else {
                        for i in 0..n.min(un) {
                            o[i] = g[i] * s;
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, un)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    if un == 1 {
                        for i in 0..n {
                            o[i] = g[0] * scale;
                        }
                    } else {
                        for i in 0..n.min(un) {
                            o[i] = g[i] * scale;
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "div" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            let n = elem_count(&upstream.shape);
            let mut grad_a = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            let mut grad_b = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f32,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f32,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f32, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f32, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        ga[i] = g[i] / bd[bi];
                        gb[i] = -g[i] * ad[ai] / (bd[bi] * bd[bi]);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f64,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f64,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f64, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f64, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        ga[i] = g[i] / bd[bi];
                        gb[i] = -g[i] * ad[ai] / (bd[bi] * bd[bi]);
                    }
                }
                _ => {}
            }
            vec![grad_a, grad_b]
        }
        "pow" => {
            assert!(saved_inputs.len() >= 2);
            let a = saved_inputs[0];
            let b = saved_inputs[1];
            let n = elem_count(&upstream.shape);
            let mut grad_a = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            let mut grad_b = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f32,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f32,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f32, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f32, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        let aval = ad[ai];
                        let bval = bd[bi];
                        ga[i] = g[i] * bval * aval.powf(bval - 1.0);
                        gb[i] = if aval > 0.0 {
                            g[i] * aval.powf(bval) * aval.ln()
                        } else {
                            0.0
                        };
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let ad = unsafe {
                        std::slice::from_raw_parts(
                            a.data.as_ptr() as *const f64,
                            elem_count(&a.shape),
                        )
                    };
                    let bd = unsafe {
                        std::slice::from_raw_parts(
                            b.data.as_ptr() as *const f64,
                            elem_count(&b.shape),
                        )
                    };
                    let ga = unsafe {
                        std::slice::from_raw_parts_mut(grad_a.data.as_mut_ptr() as *mut f64, n)
                    };
                    let gb = unsafe {
                        std::slice::from_raw_parts_mut(grad_b.data.as_mut_ptr() as *mut f64, n)
                    };
                    let an = elem_count(&a.shape);
                    let bn = elem_count(&b.shape);
                    for i in 0..n {
                        let ai = if an == 1 { 0 } else { i % an };
                        let bi = if bn == 1 { 0 } else { i % bn };
                        let aval = ad[ai];
                        let bval = bd[bi];
                        ga[i] = g[i] * bval * aval.powf(bval - 1.0);
                        gb[i] = if aval > 0.0 {
                            g[i] * aval.powf(bval) * aval.ln()
                        } else {
                            0.0
                        };
                    }
                }
                _ => {}
            }
            vec![grad_a, grad_b]
        }
        "sigmoid" => {
            assert!(!saved_inputs.is_empty());
            // saved_inputs[0] is the OUTPUT of sigmoid (not input)
            let s = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f32,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * sd[si] * (1.0 - sd[si]);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f64,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * sd[si] * (1.0 - sd[si]);
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "tanh" => {
            assert!(!saved_inputs.is_empty());
            // saved_inputs[0] is the OUTPUT of tanh
            let s = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f32,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * (1.0 - sd[si] * sd[si]);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f64,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let sn = elem_count(&s.shape);
                    for i in 0..n {
                        let si = if sn == 1 { 0 } else { i % sn };
                        o[i] = g[i] * (1.0 - sd[si] * sd[si]);
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "gelu" => {
            assert!(!saved_inputs.is_empty());
            // saved_inputs[0] is the INPUT to gelu
            let x = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f32,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let xn = elem_count(&x.shape);
                    let c = 0.7978845608028654f32;
                    let b = 0.044715f32;
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        let v = xd[xi];
                        let x3 = v * v * v;
                        let inner = c * (v + b * x3);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0 - tanh_inner * tanh_inner;
                        let d_inner = c * (1.0 + 3.0 * b * v * v);
                        o[i] = g[i] * (0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner);
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let xd = unsafe {
                        std::slice::from_raw_parts(
                            x.data.as_ptr() as *const f64,
                            elem_count(&x.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let xn = elem_count(&x.shape);
                    let c = 0.7978845608028654f64;
                    let b = 0.044715f64;
                    for i in 0..n {
                        let xi = if xn == 1 { 0 } else { i % xn };
                        let v = xd[xi];
                        let x3 = v * v * v;
                        let inner = c * (v + b * x3);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0 - tanh_inner * tanh_inner;
                        let d_inner = c * (1.0 + 3.0 * b * v * v);
                        o[i] = g[i] * (0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner);
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "linear" => {
            // saved_inputs: [input, weight, optional_bias]
            // grad_output is (..., out_features)
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let weight = saved_inputs[1];
            // weight is (out_features, in_features)
            // grad_input = grad_output @ weight
            let m = upstream.shape[upstream.shape.len() - 1] as usize; // out_features
            let k = weight.shape[1] as usize; // in_features
            let batch: usize = upstream.shape[..upstream.shape.len() - 1]
                .iter()
                .map(|&d| d.max(0) as usize)
                .product();
            let mut grad_input = OwnedTensor::new(input.dtype, input.shape.clone());
            let mut grad_weight = OwnedTensor::new(weight.dtype, weight.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, batch * m)
                    };
                    let wd = unsafe {
                        std::slice::from_raw_parts(weight.data.as_ptr() as *const f32, m * k)
                    };
                    let gi = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_input.data.as_mut_ptr() as *mut f32,
                            batch * k,
                        )
                    };
                    let gw = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_weight.data.as_mut_ptr() as *mut f32,
                            m * k,
                        )
                    };
                    // grad_input = g @ W
                    for b in 0..batch {
                        for j in 0..k {
                            let mut s = 0.0f32;
                            for i in 0..m {
                                s += g[b * m + i] * wd[i * k + j];
                            }
                            gi[b * k + j] = s;
                        }
                    }
                    // grad_weight = grad_output^T @ input (use input data, not grad_input)
                    let id = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f32, batch * k)
                    };
                    for i in 0..m {
                        for j in 0..k {
                            let mut s = 0.0f32;
                            for bi in 0..batch {
                                s += g[bi * m + i] * id[bi * k + j];
                            }
                            gw[i * k + j] = s;
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, batch * m)
                    };
                    let wd = unsafe {
                        std::slice::from_raw_parts(weight.data.as_ptr() as *const f64, m * k)
                    };
                    let gi = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_input.data.as_mut_ptr() as *mut f64,
                            batch * k,
                        )
                    };
                    let gw = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad_weight.data.as_mut_ptr() as *mut f64,
                            m * k,
                        )
                    };
                    for b in 0..batch {
                        for j in 0..k {
                            let mut s = 0.0f64;
                            for i in 0..m {
                                s += g[b * m + i] * wd[i * k + j];
                            }
                            gi[b * k + j] = s;
                        }
                    }
                    // grad_weight = grad_output^T @ input (use input data, not grad_input)
                    let id = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f64, batch * k)
                    };
                    for i in 0..m {
                        for j in 0..k {
                            let mut s = 0.0f64;
                            for bi in 0..batch {
                                s += g[bi * m + i] * id[bi * k + j];
                            }
                            gw[i * k + j] = s;
                        }
                    }
                }
                _ => {}
            }
            let mut result = vec![grad_input, grad_weight];
            if saved_inputs.len() > 2 {
                // grad_bias = sum(g, dim=batch_dims)
                let bias = saved_inputs[2];
                let mut grad_bias = OwnedTensor::new(bias.dtype, bias.shape.clone());
                match upstream.dtype {
                    DType::F32 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(
                                upstream.data.as_ptr() as *const f32,
                                batch * m,
                            )
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f32,
                                m,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for i in 0..m {
                                gb[i] += g[b * m + i];
                            }
                        }
                    }
                    DType::F64 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(
                                upstream.data.as_ptr() as *const f64,
                                batch * m,
                            )
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f64,
                                m,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for i in 0..m {
                                gb[i] += g[b * m + i];
                            }
                        }
                    }
                    _ => {}
                }
                result.push(grad_bias);
            }
            result
        }
        "layer_norm" => {
            // saved_inputs: [input, weight, optional_bias]
            // Proper layer_norm backward:
            //   x_hat = (x - mean) / sqrt(var + eps)
            //   grad_x_hat = grad_output * weight
            //   grad_x = inv_std * (grad_x_hat - mean(grad_x_hat) - x_hat * mean(grad_x_hat * x_hat))
            //   grad_weight = sum_over_batch(grad_output * x_hat)
            //   grad_bias = sum_over_batch(grad_output)
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let weight = saved_inputs[1];
            let last_dim = *input.shape.last().unwrap_or(&1) as usize;
            let n = elem_count(&input.shape);
            let batch = if last_dim > 0 { n / last_dim } else { 1 };
            let mut grad_input = OwnedTensor::new(input.dtype, input.shape.clone());
            let mut grad_weight = OwnedTensor::new(weight.dtype, weight.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
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
                        let mut mu = 0.0f32;
                        for j in 0..last_dim {
                            mu += xd[base + j];
                        }
                        mu /= last_dim as f32;
                        let mut var = 0.0f32;
                        for j in 0..last_dim {
                            let d = xd[base + j] - mu;
                            var += d * d;
                        }
                        var /= last_dim as f32;
                        let inv_std = 1.0f32 / (var + 1e-5f32).sqrt();
                        // grad_x_hat = g * weight
                        // mean(grad_x_hat)
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
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
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
                        let inv_std = 1.0f64 / (var + 1e-5f64).sqrt();
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
            let mut result = vec![grad_input, grad_weight];
            if saved_inputs.len() > 2 {
                let bias = saved_inputs[2];
                let mut grad_bias = OwnedTensor::new(bias.dtype, bias.shape.clone());
                match upstream.dtype {
                    DType::F32 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f32,
                                last_dim,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for j in 0..last_dim {
                                gb[j] += g[b * last_dim + j];
                            }
                        }
                    }
                    DType::F64 => {
                        let g = unsafe {
                            std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                        };
                        let gb = unsafe {
                            std::slice::from_raw_parts_mut(
                                grad_bias.data.as_mut_ptr() as *mut f64,
                                last_dim,
                            )
                        };
                        gb.fill(0.0);
                        for b in 0..batch {
                            for j in 0..last_dim {
                                gb[j] += g[b * last_dim + j];
                            }
                        }
                    }
                    _ => {}
                }
                result.push(grad_bias);
            }
            result
        }
        "softmax" => {
            // saved_inputs[0] = output of softmax
            assert!(!saved_inputs.is_empty());
            let s = saved_inputs[0];
            let n = elem_count(&upstream.shape);
            let mut grad = OwnedTensor::new(upstream.dtype, upstream.shape.clone());
            match upstream.dtype {
                DType::F32 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f32, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f32,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n)
                    };
                    let dim_size = *upstream.shape.last().unwrap_or(&1) as usize;
                    if dim_size > 0 {
                        for base in (0..n).step_by(dim_size) {
                            let mut dot = 0.0f32;
                            for j in 0..dim_size {
                                dot += sd[base + j] * g[base + j];
                            }
                            for j in 0..dim_size {
                                o[base + j] = sd[base + j] * (g[base + j] - dot);
                            }
                        }
                    }
                }
                DType::F64 => {
                    let g = unsafe {
                        std::slice::from_raw_parts(upstream.data.as_ptr() as *const f64, n)
                    };
                    let sd = unsafe {
                        std::slice::from_raw_parts(
                            s.data.as_ptr() as *const f64,
                            elem_count(&s.shape),
                        )
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n)
                    };
                    let dim_size = *upstream.shape.last().unwrap_or(&1) as usize;
                    if dim_size > 0 {
                        for base in (0..n).step_by(dim_size) {
                            let mut dot = 0.0f64;
                            for j in 0..dim_size {
                                dot += sd[base + j] * g[base + j];
                            }
                            for j in 0..dim_size {
                                o[base + j] = sd[base + j] * (g[base + j] - dot);
                            }
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "nll_loss" => {
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let target = saved_inputs[1];
            let n_batch = input.shape[0] as usize;
            let n_classes = if input.shape.len() > 1 {
                *input.shape.last().unwrap_or(&1) as usize
            } else {
                1
            };
            let reduction_str = kwargs
                .get("reduction")
                .and_then(|v| v.as_str())
                .unwrap_or("mean");
            let scale = match reduction_str {
                "mean" => 1.0 / n_batch as f64,
                "sum" => 1.0,
                _ => 1.0,
            };
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            match input.dtype {
                DType::F32 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f32,
                            elem_count(&upstream.shape),
                        )
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad.data.as_mut_ptr() as *mut f32,
                            n_batch * n_classes,
                        )
                    };
                    o.fill(0.0);
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 0.0 };
                    for b in 0..n_batch {
                        let t = tgt[b] as usize;
                        if t < n_classes {
                            o[b * n_classes + t] = -(scale as f32) * g_val;
                        }
                    }
                }
                DType::F64 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f64,
                            elem_count(&upstream.shape),
                        )
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(
                            grad.data.as_mut_ptr() as *mut f64,
                            n_batch * n_classes,
                        )
                    };
                    o.fill(0.0);
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 0.0 };
                    for b in 0..n_batch {
                        let t = tgt[b] as usize;
                        if t < n_classes {
                            o[b * n_classes + t] = -scale * g_val;
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        "cross_entropy" => {
            // cross_entropy(input, target) = nll_loss(log_softmax(input), target)
            // grad = scale * upstream * (softmax(input) - one_hot(target))
            assert!(saved_inputs.len() >= 2);
            let input = saved_inputs[0];
            let target = saved_inputs[1];
            let n_batch = input.shape[0] as usize;
            let n_classes = if input.shape.len() > 1 {
                *input.shape.last().unwrap_or(&1) as usize
            } else {
                1
            };
            let reduction_str = kwargs
                .get("reduction")
                .and_then(|v| v.as_str())
                .unwrap_or("mean");
            let scale = match reduction_str {
                "mean" => 1.0 / n_batch as f64,
                "sum" => 1.0,
                _ => 1.0,
            };
            let n_total = n_batch * n_classes;
            let mut grad = OwnedTensor::new(input.dtype, input.shape.clone());
            match input.dtype {
                DType::F32 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f32,
                            elem_count(&upstream.shape),
                        )
                    };
                    let inp = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f32, n_total)
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f32, n_total)
                    };
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 1.0f32 };
                    for b in 0..n_batch {
                        // compute softmax for this row
                        let mut max_val = f32::NEG_INFINITY;
                        for c in 0..n_classes {
                            let v = inp[b * n_classes + c];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut sum_exp = 0.0f32;
                        for c in 0..n_classes {
                            sum_exp += (inp[b * n_classes + c] - max_val).exp();
                        }
                        for c in 0..n_classes {
                            let prob = (inp[b * n_classes + c] - max_val).exp() / sum_exp;
                            let target_one_hot = if c == tgt[b] as usize { 1.0f32 } else { 0.0f32 };
                            o[b * n_classes + c] = (scale as f32) * g_val * (prob - target_one_hot);
                        }
                    }
                }
                DType::F64 => {
                    let g_raw = unsafe {
                        std::slice::from_raw_parts(
                            upstream.data.as_ptr() as *const f64,
                            elem_count(&upstream.shape),
                        )
                    };
                    let inp = unsafe {
                        std::slice::from_raw_parts(input.data.as_ptr() as *const f64, n_total)
                    };
                    let tgt = unsafe {
                        std::slice::from_raw_parts(target.data.as_ptr() as *const i64, n_batch)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(grad.data.as_mut_ptr() as *mut f64, n_total)
                    };
                    let g_val = if g_raw.len() == 1 { g_raw[0] } else { 1.0f64 };
                    for b in 0..n_batch {
                        let mut max_val = f64::NEG_INFINITY;
                        for c in 0..n_classes {
                            let v = inp[b * n_classes + c];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut sum_exp = 0.0f64;
                        for c in 0..n_classes {
                            sum_exp += (inp[b * n_classes + c] - max_val).exp();
                        }
                        for c in 0..n_classes {
                            let prob = (inp[b * n_classes + c] - max_val).exp() / sum_exp;
                            let target_one_hot = if c == tgt[b] as usize { 1.0f64 } else { 0.0f64 };
                            o[b * n_classes + c] = scale * g_val * (prob - target_one_hot);
                        }
                    }
                }
                _ => {}
            }
            vec![grad]
        }
        _ => {
            // Unsupported backward: return zero gradients
            let mut grads = Vec::new();
            for input in saved_inputs {
                grads.push(OwnedTensor::new(input.dtype, input.shape.clone()));
            }
            grads
        }
    }
}

// ---------------------------------------------------------------------------
// Batch backward — process entire tape in one FFI call
// ---------------------------------------------------------------------------

/// A single entry in the batch tape, mirroring what the Python autograd
/// records during the forward pass.  Each entry is passed to
/// `backward_single` internally, but all DLPack conversions happen
/// *outside* the per-op loop, eliminating per-op FFI overhead.
pub struct BatchTapeEntry {
    pub target: String,
    pub saved_inputs: Vec<OwnedTensor>,
    pub kwargs: std::collections::HashMap<String, serde_json::Value>,
    pub output_id: usize,
    pub input_ids: Vec<usize>,
    /// Original shapes of saved inputs (before padding to capsules).
    /// Used to reduce broadcast gradients to match input shapes.
    pub saved_shapes: Vec<Vec<i64>>,
}

/// Reduce a gradient tensor to match a target shape (broadcast fix).
/// When an op like add(a, b) was broadcast, backward_single returns
/// grad with the upstream shape, but the input may have a smaller shape.
/// We sum over extra leading dimensions and broadcast dims.
fn reduce_to_shape(grad: &OwnedTensor, target: &[i64]) -> OwnedTensor {
    let mut result = grad.clone();

    // Step 1: trim leading dimensions
    while result.shape.len() > target.len() {
        let n = elem_count(&result.shape);
        let dim0 = result.shape[0] as usize;
        let trimmed_len = n / dim0;
        let mut trimmed = OwnedTensor::new(result.dtype, result.shape[1..].to_vec());
        match result.dtype {
            DType::F32 => {
                let src =
                    unsafe { std::slice::from_raw_parts(result.data.as_ptr() as *const f32, n) };
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        trimmed.data.as_mut_ptr() as *mut f32,
                        trimmed_len,
                    )
                };
                for i in 0..trimmed_len {
                    let mut s = 0.0f32;
                    for j in 0..dim0 {
                        s += src[j * trimmed_len + i];
                    }
                    dst[i] = s;
                }
            }
            DType::F64 => {
                let src =
                    unsafe { std::slice::from_raw_parts(result.data.as_ptr() as *const f64, n) };
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        trimmed.data.as_mut_ptr() as *mut f64,
                        trimmed_len,
                    )
                };
                for i in 0..trimmed_len {
                    let mut s = 0.0f64;
                    for j in 0..dim0 {
                        s += src[j * trimmed_len + i];
                    }
                    dst[i] = s;
                }
            }
            _ => {}
        }
        result = trimmed;
    }

    // Step 2: sum over broadcast dims (target is 1 but grad > 1)
    for i in 0..target.len() {
        if target[i] == 1 && result.shape[i] > 1 {
            let n = elem_count(&result.shape);
            let dim_size = result.shape[i] as usize;
            let outer: usize = result.shape[..i]
                .iter()
                .map(|&d| d.max(0) as usize)
                .product();
            let inner: usize = result.shape[i + 1..]
                .iter()
                .map(|&d| d.max(0) as usize)
                .product();
            let mut out_shape = result.shape.clone();
            out_shape[i] = 1;
            let mut out = OwnedTensor::new(result.dtype, out_shape);
            match result.dtype {
                DType::F32 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(result.data.as_ptr() as *const f32, n)
                    };
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            out.data.as_mut_ptr() as *mut f32,
                            outer * inner,
                        )
                    };
                    dst.fill(0.0);
                    for o in 0..outer {
                        for d in 0..dim_size {
                            for k in 0..inner {
                                dst[o * inner + k] += src[o * dim_size * inner + d * inner + k];
                            }
                        }
                    }
                }
                DType::F64 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(result.data.as_ptr() as *const f64, n)
                    };
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            out.data.as_mut_ptr() as *mut f64,
                            outer * inner,
                        )
                    };
                    dst.fill(0.0);
                    for o in 0..outer {
                        for d in 0..dim_size {
                            for k in 0..inner {
                                dst[o * inner + k] += src[o * dim_size * inner + d * inner + k];
                            }
                        }
                    }
                }
                _ => {}
            }
            result = out;
        }
    }

    // Step 3: reshape to exact target
    if result.shape != target && elem_count(&result.shape) == elem_count(target) {
        result.shape = target.to_vec();
    }

    result
}

/// Process the entire autograd tape in a single call.
///
/// Walks the tape in reverse order, computes gradients via
/// `backward_single`, and accumulates them by tensor ID.  Returns a
/// flat list of `(tensor_id, gradient)` pairs — one per leaf tensor
/// that received a non-zero gradient.
///
/// The big win is that *all* DLPack→OwnedTensor conversions happen
/// once at the start, and all OwnedTensor→DLPack conversions happen
/// once at the end — instead of doing them per-op.
pub fn backward_batch(
    tape: &[BatchTapeEntry],
    initial_upstream: &OwnedTensor,
    initial_output_id: usize,
) -> Vec<(usize, OwnedTensor)> {
    // Map: tensor_id -> accumulated gradient
    let mut grads: HashMap<usize, OwnedTensor> = HashMap::new();
    grads.insert(initial_output_id, initial_upstream.clone());

    // Walk tape in reverse (last recorded op first)
    for entry in tape.iter().rev() {
        let upstream = match grads.remove(&entry.output_id) {
            Some(g) => g,
            None => continue, // no gradient flows through this op
        };

        let saved_refs: Vec<&OwnedTensor> = entry.saved_inputs.iter().collect();
        let per_input = backward_single(&entry.target, &upstream, &saved_refs, &entry.kwargs);

        // Accumulate gradients into the input tensor IDs
        for (i, tid) in entry.input_ids.iter().enumerate() {
            if i < per_input.len() {
                let mut pg = per_input[i].clone();
                // Skip zero-valued gradients (common for unsupported ops)
                if pg.data.iter().all(|&b| b == 0) {
                    continue;
                }

                // Broadcast shape reduction: if the saved input had a
                // different shape than the upstream (e.g. b=(4,) was
                // broadcast to (3,4)), reduce the gradient back.
                if i < entry.saved_shapes.len() {
                    let target = &entry.saved_shapes[i];
                    let pg_shape: Vec<i64> = pg.shape.iter().map(|&d| d as i64).collect();
                    if pg_shape != *target {
                        pg = reduce_to_shape(&pg, target);
                    }
                }

                if let Some(existing) = grads.get_mut(tid) {
                    // In-place addition: existing += pg
                    let n = elem_count(&existing.shape);
                    match existing.dtype {
                        DType::F32 => {
                            let e = unsafe {
                                std::slice::from_raw_parts_mut(
                                    existing.data.as_mut_ptr() as *mut f32,
                                    n,
                                )
                            };
                            let p = unsafe {
                                std::slice::from_raw_parts(
                                    pg.data.as_ptr() as *const f32,
                                    n.min(elem_count(&pg.shape)),
                                )
                            };
                            for j in 0..n.min(p.len()) {
                                e[j] += p[j];
                            }
                        }
                        DType::F64 => {
                            let e = unsafe {
                                std::slice::from_raw_parts_mut(
                                    existing.data.as_mut_ptr() as *mut f64,
                                    n,
                                )
                            };
                            let p = unsafe {
                                std::slice::from_raw_parts(
                                    pg.data.as_ptr() as *const f64,
                                    n.min(elem_count(&pg.shape)),
                                )
                            };
                            for j in 0..n.min(p.len()) {
                                e[j] += p[j];
                            }
                        }
                        _ => {}
                    }
                } else {
                    grads.insert(*tid, pg);
                }
            }
        }
    }

    grads.into_iter().collect()
}
