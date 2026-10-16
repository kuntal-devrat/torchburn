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

/// Sentinel stored in thread-local storage that decrements `ACTIVE_RECORDING_THREADS`
/// when the thread terminates, preventing diagnostic counter leaks even if `disable()`
/// was not explicitly called.
struct ThreadAutogradSentinel {
    enabled: bool,
}

impl Drop for ThreadAutogradSentinel {
    fn drop(&mut self) {
        if self.enabled {
            self.enabled = false;
            ACTIVE_RECORDING_THREADS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

thread_local! {
    static TAPE: RefCell<Vec<Box<dyn BackwardOp>>> = RefCell::new(Vec::new());
    static TAPE_META: RefCell<Vec<TapeEntryMeta>> = RefCell::new(Vec::new());
    static SENTINEL: RefCell<ThreadAutogradSentinel> =
        RefCell::new(ThreadAutogradSentinel { enabled: false });
    /// Saved tensor data indexed by tensor ID.
    static SAVED_DATA: RefCell<HashMap<usize, Box<OwnedTensor>>> = RefCell::new(HashMap::new());
}

struct TapeEntryMeta {
    output_ids: Vec<usize>,
    input_ids: Vec<usize>,
}

/// Enable autograd recording for the current thread.
pub fn enable() {
    SENTINEL.with(|s| {
        let mut sentinel = s.borrow_mut();
        if !sentinel.enabled {
            sentinel.enabled = true;
            ACTIVE_RECORDING_THREADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// Disable autograd recording for the current thread.
pub fn disable() {
    SENTINEL.with(|s| {
        let mut sentinel = s.borrow_mut();
        if sentinel.enabled {
            sentinel.enabled = false;
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
    SENTINEL.with(|s| s.borrow().enabled)
}

/// Number of threads currently recording autograd operations.
pub fn active_recording_threads() -> usize {
    ACTIVE_RECORDING_THREADS.load(std::sync::atomic::Ordering::Relaxed)
}

/// RAII guard that enables autograd recording on creation and disables it on drop.
pub struct AutogradGuard {
    active: bool,
}

impl AutogradGuard {
    /// Create a new RAII guard and enable autograd recording for the current thread.
    pub fn new() -> Self {
        enable();
        Self { active: true }
    }

    /// Explicitly disable recording before drop.
    pub fn disable(&mut self) {
        if self.active {
            disable();
            self.active = false;
        }
    }

    /// Check if this guard is actively holding an enabled recording session.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Default for AutogradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutogradGuard {
    fn drop(&mut self) {
        self.disable();
    }
}

/// Create an RAII drop guard that enables autograd recording and disables it when dropped.
pub fn record_guard() -> AutogradGuard {
    AutogradGuard::new()
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
///
/// ## Single-loss semantics
///
/// `grad_output` is seeded into the **last recorded tape entry only**. This is
/// correct for the standard single-loss training loop (one final loss node).
/// For multi-root graphs — e.g. a model with several independent outputs —
/// only the last root is differentiated; a diagnostic warning is emitted when
/// additional unclaimed roots are detected. Passing per-output gradients is
/// not supported by this API.
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

    // Diagnose multi-root tapes (see the `backward` docs): any earlier entry
    // whose outputs are not consumed by a later entry is an additional graph
    // root that will silently receive no gradient. Warn instead of staying
    // silent — this is the classic "one output trained, the rest quietly
    // ignored" training bug.
    TAPE_META.with(|m| {
        let meta = m.borrow();
        if meta.len() > 1 {
            let consumed: std::collections::HashSet<usize> = meta
                .iter()
                .flat_map(|e| e.input_ids.iter().copied())
                .collect();
            let extra_roots = meta[..meta.len() - 1]
                .iter()
                .filter(|e| e.output_ids.iter().any(|id| !consumed.contains(id)))
                .count();
            if extra_roots > 0 {
                eprintln!(
                    "torchburn: autograd: backward() received a single grad_output but the \
                     tape has {extra_roots} additional unclaimed graph root(s). Only the last \
                     recorded output is differentiated; per-output gradients are not supported. \
                     Sum losses into a single scalar output before backward() for full-graph \
                     gradient flow."
                );
            }
        }
    });

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
        DType::I8 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut i8, n) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const i8, n) };
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                *x = x.wrapping_add(*y);
            }
        }
        DType::U8 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut u8, n) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const u8, n) };
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                *x = x.wrapping_add(*y);
            }
        }
        DType::I32 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut i32, n) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const i32, n) };
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                *x = x.wrapping_add(*y);
            }
        }
        DType::I64 => {
            let a_data =
                unsafe { std::slice::from_raw_parts_mut(a.data.as_mut_ptr() as *mut i64, n) };
            let b_data = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const i64, n) };
            for (x, y) in a_data.iter_mut().zip(b_data.iter()) {
                *x = x.wrapping_add(*y);
            }
        }
        DType::Bool => {
            // Boolean "gradients" are not meaningful; refuse loudly rather
            // than silently dropping the accumulation (the old `_ => {}`
            // behavior) or panicking across the FFI boundary.
            eprintln!(
                "torchburn: autograd: gradient accumulation for bool tensors is not supported; \
                 the accumulation was skipped"
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_enable_disable_lifecycle() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // survive poisoned lock from a sibling failure
        let initial = active_recording_threads();
        enable();
        assert!(is_enabled());
        assert_eq!(active_recording_threads(), initial + 1);

        disable();
        assert!(!is_enabled());
        assert_eq!(active_recording_threads(), initial);
    }

    #[test]
    fn test_thread_exit_cleans_up_active_recording_threads() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // survive poisoned lock from a sibling failure
        let initial = active_recording_threads();
        let handle = std::thread::spawn(move || {
            enable();
            assert!(is_enabled());
            assert_eq!(active_recording_threads(), initial + 1);
            // Thread terminates without explicit disable()
        });
        handle.join().unwrap();
        // The thread-local sentinel drop implementation MUST have decremented ACTIVE_RECORDING_THREADS.
        assert_eq!(active_recording_threads(), initial);
    }

    #[test]
    fn test_autograd_guard_raii() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // survive poisoned lock from a sibling failure
        let initial = active_recording_threads();
        {
            let guard = AutogradGuard::new();
            assert!(is_enabled());
            assert!(guard.is_active());
            assert_eq!(active_recording_threads(), initial + 1);
        }
        // Guard dropped
        assert!(!is_enabled());
        assert_eq!(active_recording_threads(), initial);
    }

    #[test]
    fn test_add_in_place_integer_dtypes_accumulate() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        fn fill<T: Copy>(t: &mut OwnedTensor, vals: &[T]) {
            unsafe {
                let v = std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, vals.len());
                v.copy_from_slice(vals);
            }
        }
        fn read<T: Copy>(t: &OwnedTensor, n: usize) -> Vec<T> {
            unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const T, n).to_vec() }
        }

        // Regression: integer gradients used to be SILENTLY DROPPED by the
        // old `_ => {}` catch-all in add_in_place.
        let mut a = OwnedTensor::new(DType::I32, vec![3]);
        let mut b = OwnedTensor::new(DType::I32, vec![3]);
        fill(&mut a, &[1i32, -2, 3]);
        fill(&mut b, &[10i32, 20, -30]);
        add_in_place(&mut a, &b);
        assert_eq!(read::<i32>(&a, 3), vec![11, 18, -27]);

        // Overflowing integer gradients wrap (no panic across the FFI).
        let mut x = OwnedTensor::new(DType::I64, vec![2]);
        let mut y = OwnedTensor::new(DType::I64, vec![2]);
        fill(&mut x, &[i64::MAX, 5i64]);
        fill(&mut y, &[1i64, -5i64]);
        add_in_place(&mut x, &y);
        assert_eq!(read::<i64>(&x, 2), vec![i64::MIN, 0]);
    }

    #[test]
    fn test_concurrent_threads_active_count() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // survive poisoned lock from a sibling failure
        let initial = active_recording_threads();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                let _guard = AutogradGuard::new();
                b.wait();
                b.wait();
            }));
        }

        // Wait until all 4 threads have enabled autograd
        barrier.wait();
        assert_eq!(active_recording_threads(), initial + 4);

        // Release threads
        barrier.wait();
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(active_recording_threads(), initial);
    }
}
