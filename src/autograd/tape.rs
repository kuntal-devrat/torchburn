//! Autograd tape: recording infrastructure and gradient storage.
//!
//! ## Thread-Safety Contract
//!
//! Autograd recording (`enable()`, `record()`, `save_data()`) and the reverse-mode
//! pass (`backward()`) operate on thread-local tape storage. This design ensures
//! that concurrent Python threads training separate models or graphs never corrupt
//! each other's gradient tapes.
//!
//! **Concurrency Guidelines:**
//! 1. A single model forward + backward execution must happen on the same thread
//!    where `torchburn.autograd.enable()` was called.
//! 2. Multi-threaded worker pools (such as Rayon workers used internally by TorchBurn
//!    for parallel matrix multiplications and reductions) do NOT record ops to the
//!    tape, preventing worker threads from scattering operations across separate tapes.
//! 3. If training multiple models concurrently, each training loop should run on its
//!    own dedicated thread with its own `enable()` / `disable()` cycle.

use super::*;
// ---------------------------------------------------------------------------
// Tensor ID and gradient storage
// ---------------------------------------------------------------------------

/// Global monotonically increasing tensor ID counter.
static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Global counter of active recording threads to diagnose cross-thread usage.
static ACTIVE_RECORDING_THREADS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

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
    /// Saved tensor data indexed by tensor ID.
    static SAVED_DATA: RefCell<HashMap<usize, Box<OwnedTensor>>> = RefCell::new(HashMap::new());
}

struct TapeEntryMeta {
    output_ids: Vec<usize>,
    input_ids: Vec<usize>,
}

/// Enable autograd recording for the current thread.
pub fn enable() {
    ENABLED.with(|e| {
        let mut en = e.borrow_mut();
        if !*en {
            *en = true;
            ACTIVE_RECORDING_THREADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// Disable autograd recording for the current thread.
pub fn disable() {
    ENABLED.with(|e| {
        let mut en = e.borrow_mut();
        if *en {
            *en = false;
            ACTIVE_RECORDING_THREADS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
    // Release any saved tensors to prevent unbounded growth if user enabled
    // but never called backward().
    SAVED_DATA.with(|s| s.borrow_mut().clear());
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
/// No-op when tape disabled (prevents unbounded growth if user forgets backward).
pub fn save_data(id: usize, data: &OwnedTensor) {
    if !is_enabled() {
        return;
    }
    SAVED_DATA.with(|s| {
        s.borrow_mut().insert(id, Box::new(data.clone()));
    });
}

/// Save borrowed tensor data by cloning it into a safe owned tensor.
pub fn save_borrowed(id: usize, data: &BorrowedTensor) {
    if !is_enabled() {
        return;
    }
    let owned = unsafe { owned_from_borrowed(data) };
    SAVED_DATA.with(|s| {
        s.borrow_mut().insert(id, Box::new(owned));
    });
}

/// Create an OwnedTensor from a BorrowedTensor (strided-safe, all dtypes).
unsafe fn owned_from_borrowed(b: &BorrowedTensor) -> OwnedTensor {
    let n = elem_count(&b.shape);
    let mut out = OwnedTensor::new(b.dtype, b.shape.to_vec());
    if n == 0 {
        return out;
    }
    if b.is_contiguous() {
        let bytes = n * b.dtype.elem_size();
        std::ptr::copy_nonoverlapping(b.data, out.data.as_mut_ptr() as *mut u8, bytes);
        return out;
    }
    let elem = b.dtype.elem_size();
    let ndim = b.shape.len();
    let mut idx = vec![0i64; ndim];
    for out_off in 0..n {
        let mut rem = out_off;
        for d in (0..ndim).rev() {
            let dim = b.shape[d].max(1) as usize;
            idx[d] = (rem % dim) as i64;
            rem /= dim;
        }
        let mut phys: i64 = 0;
        for d in 0..ndim {
            phys += idx[d] * b.strides[d];
        }
        let src = b.data.add((phys.max(0) as usize) * elem);
        let dst = (out.data.as_mut_ptr() as *mut u8).add(out_off * elem);
        std::ptr::copy_nonoverlapping(src, dst, elem);
    }
    out
}

/// Consume the tape: execute backward for every recorded op in reverse order.
///
/// `leaf_grads` is a mutable reference to a map from tensor ID → gradient.
/// Leaf tensors (parameters) accumulate into this map; intermediate
/// gradients are consumed and freed after each op.
pub fn backward(grad_output: &OwnedTensor, leaf_grads: &mut HashMap<usize, OwnedTensor>) {
    if !is_enabled() && ACTIVE_RECORDING_THREADS.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        eprintln!(
            "torchburn: autograd: backward() called on a thread where autograd is not enabled, \
             but another thread has an active recording session. \
             Autograd forward and backward must be executed on the same thread."
        );
    }
    // Map to accumulate intermediate gradients by tensor ID across branches.
    let mut node_grads: HashMap<usize, OwnedTensor> = HashMap::new();

    // Seed the initial gradient by the last op's output tensor ID.
    // This correctly handles DAG-shaped graphs (multiple roots, branching)
    // where the old positional approach (rev_idx == 0) would miss ops.
    TAPE_META.with(|m| {
        let meta = m.borrow();
        if let Some(last_meta) = meta.last() {
            for id in &last_meta.output_ids {
                match node_grads.entry(*id) {
                    std::collections::hash_map::Entry::Occupied(mut o) => {
                        add_in_place(o.get_mut(), grad_output);
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert(grad_output.clone());
                    }
                }
            }
        }
    });

    SAVED_DATA.with(|s| {
        let saved_map = s.borrow();
        TAPE.with(|t| {
            TAPE_META.with(|m| {
                let tape = t.borrow();
                let meta = m.borrow();
                let len = tape.len();
                for rev_idx in 0..len {
                    let i = len - 1 - rev_idx;
                    let op: &(dyn BackwardOp + 'static) = &*tape[i];
                    let me = &meta[i];

                    // Sum all output grads (fan-out: same tensor consumed twice)
                    let mut upstream_opt: Option<OwnedTensor> = None;
                    for id in me.output_ids.iter() {
                        if let Some(g) = node_grads.remove(id) {
                            match upstream_opt.take() {
                                None => upstream_opt = Some(g),
                                Some(mut acc) => {
                                    add_in_place(&mut acc, &g);
                                    upstream_opt = Some(acc);
                                }
                            }
                        }
                    }
                    let upstream = match upstream_opt {
                        Some(u) => u,
                        None => continue,
                    };

                    // Collect saved inputs safely from the immutably borrowed map.
                    let saved_refs: Vec<&OwnedTensor> = me
                        .input_ids
                        .iter()
                        .filter_map(|&id| saved_map.get(&id).map(|b| b.as_ref()))
                        .collect();

                    let grads = op.backward(&upstream, &saved_refs);

                    for (tensor_id, grad) in grads {
                        // Accumulate into leaf_grads first (always needed).
                        match leaf_grads.entry(tensor_id) {
                            std::collections::hash_map::Entry::Occupied(mut o) => {
                                add_in_place(o.get_mut(), &grad);
                            }
                            std::collections::hash_map::Entry::Vacant(v) => {
                                v.insert(grad.clone());
                            }
                        }

                        // Accumulate into node_grads (for upstream routing).
                        // Move the grad here to avoid a second clone.
                        match node_grads.entry(tensor_id) {
                            std::collections::hash_map::Entry::Occupied(mut o) => {
                                add_in_place(o.get_mut(), &grad);
                            }
                            std::collections::hash_map::Entry::Vacant(v) => {
                                v.insert(grad); // move, no clone
                            }
                        }
                    }
                }
            })
        });
    });

    // Free saved data.
    SAVED_DATA.with(|s| s.borrow_mut().clear());

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
            if n >= 256 * 1024 {
                use rayon::prelude::*;
                a_data
                    .par_iter_mut()
                    .zip(b_data.par_iter())
                    .for_each(|(x, y)| {
                        *x += y;
                    });
            } else {
                for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                    *x += y;
                }
            }
        }
        DType::F64 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut f64, n) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const f64, n) };
            if n >= 256 * 1024 {
                use rayon::prelude::*;
                a_data
                    .par_iter_mut()
                    .zip(b_data.par_iter())
                    .for_each(|(x, y)| {
                        *x += y;
                    });
            } else {
                for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                    *x += y;
                }
            }
        }
        DType::F16 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut half::f16, n) };
            let b_data =
                unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const half::f16, n) };
            if n >= 64 * 1024 {
                use rayon::prelude::*;
                a_data
                    .par_iter_mut()
                    .zip(b_data.par_iter())
                    .for_each(|(x, y)| {
                        *x = half::f16::from_f32(x.to_f32() + y.to_f32());
                    });
            } else {
                for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                    *x = half::f16::from_f32(x.to_f32() + y.to_f32());
                }
            }
        }
        DType::BF16 => {
            let a_data = unsafe {
                std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut half::bf16, n)
            };
            let b_data =
                unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const half::bf16, n) };
            if n >= 64 * 1024 {
                use rayon::prelude::*;
                a_data
                    .par_iter_mut()
                    .zip(b_data.par_iter())
                    .for_each(|(x, y)| {
                        *x = half::bf16::from_f32(x.to_f32() + y.to_f32());
                    });
            } else {
                for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                    *x = half::bf16::from_f32(x.to_f32() + y.to_f32());
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Public helper to reset tape state between training steps
// ---------------------------------------------------------------------------

/// Clear the tape and free saved data without computing gradients.
pub fn reset() {
    SAVED_DATA.with(|s| s.borrow_mut().clear());
    TAPE.with(|t| t.borrow_mut().clear());
    TAPE_META.with(|m| m.borrow_mut().clear());
}

/// Number of ops currently recorded on the tape.
pub fn tape_len() -> usize {
    TAPE.with(|t| t.borrow().len())
}
