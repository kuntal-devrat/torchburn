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

pub mod tape;

pub use self::tape::{
    backward, disable, enable, is_enabled, reset, save_borrowed, save_data, tape_len, TensorMeta,
};
pub(crate) use self::tape::{record, BackwardOp};

pub mod backward_ops;

pub use self::backward_ops::{
    record_add, record_cat, record_div, record_dropout, record_layer_norm, record_linear,
    record_matmul, record_mse_loss, record_mul, record_nll_loss, record_permute, record_reshape,
    record_softmax, record_sub, record_sum,
};

pub mod batch;

pub use self::batch::{backward_batch, backward_native, backward_single, BatchTapeEntry};
