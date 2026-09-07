//! Batch and native backward entry points.
//!
//! backward_native / backward_single / backward_batch plus
//! BatchTapeEntry. Inherits root imports via super; pure move.

use super::*;
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
pub mod single;

pub use self::single::backward_single;

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
