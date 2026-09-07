//! Memory pool for intermediate tensor reuse.
//!
//! Reuses `Vec<u64>` allocations across kernel calls to avoid repeated
//! malloc/free overhead on the hot path. Two tiers:
//!
//! * **Thread pool** (`RefCell`, lock-free): intermediates recycled back into
//!   the executing thread (see `engine::collect_outputs`) are returned here.
//! * **Global free list** (`Mutex`): output tensors exported to Python come
//!   back when the Python side drops them — the DLPack capsule deleter runs on
//!   whatever thread GC happened to use, so a process-wide list (not the
//!   thread pool) is where those buffers land.  Repeated same-shape calls
//!   (the steady-state inference loop) then hit the free list instead of
//!   paying a fresh malloc + zero-fill per call.
//!
//! Reused buffers are **not** re-zeroed: every kernel is expected to fully
//! overwrite its output (the same contract the thread pool has always used;
//! `u64` has no trap representations, and the memory was initialised when the
//! buffer was first allocated).

use crate::dlpack::{DType, OwnedTensor};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Maximum number of free buffers to retain in the thread pool.
const MAX_FREE: usize = 64;
/// Maximum number of free buffers to retain in the global (deleter-fed) list.
const MAX_GLOBAL_FREE: usize = 32;

static GLOBAL_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_HIT_COUNT: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_RECYCLE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug)]
pub struct PoolStats {
    pub alloc_count: usize,
    pub hit_count: usize,
    pub recycle_count: usize,
    pub cached_buffers: usize,
    pub cached_words: usize,
}

/// Per-thread pool of reusable buffers: (dtype, capacity in u64s, buffer).
thread_local! {
    static POOL: RefCell<Vec<(DType, usize, Vec<u64>)>> = RefCell::new(Vec::new());
}

/// Process-wide free list fed by the DLPack capsule deleter (Python GC thread)
/// and drained by `take_buffer` (execution thread).
static GLOBAL_POOL: Mutex<Vec<(DType, usize, Vec<u64>)>> = Mutex::new(Vec::new());

/// Large buffers (in u64 words, i.e. >64MB of f32) are not pooled to avoid
/// unbounded memory retention.
fn poolable(words: usize) -> bool {
    words <= 10_000_000
}

/// Borrow a buffer from the pool, or allocate a new one.
/// The returned `Vec<u64>` has length `words` and is zeroed only when newly
/// allocated; pooled buffers keep their previous contents (kernels must write
/// their full output, see the module docs).
pub fn take_buffer(dtype: DType, words: usize) -> Vec<u64> {
    GLOBAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    // Avoid pooling extremely large buffers (>64MB) to prevent memory bloat
    if !poolable(words) {
        return vec![0u64; words];
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        // Best-fit search: smallest buffer that fits to reduce fragmentation
        let mut best_idx: Option<usize> = None;
        let mut best_cap = usize::MAX;
        for (i, (d, cap, _)) in pool.iter().enumerate() {
            if *d == dtype && *cap >= words && *cap < best_cap {
                best_cap = *cap;
                best_idx = Some(i);
            }
        }
        if let Some(idx) = best_idx {
            GLOBAL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
            let mut buf = pool.remove(idx).2;
            // Reuse allocation without zeroing memory (see module docs).
            unsafe {
                buf.set_len(words);
            }
            return buf;
        }
        drop(pool);
        take_buffer_global(dtype, words)
    })
}

/// Best-fit drain of the global free list (fed by the capsule deleter).
fn take_buffer_global(dtype: DType, words: usize) -> Vec<u64> {
    let mut global = GLOBAL_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut best_idx: Option<usize> = None;
    let mut best_cap = usize::MAX;
    for (i, (d, cap, _)) in global.iter().enumerate() {
        if *d == dtype && *cap >= words && *cap < best_cap {
            best_cap = *cap;
            best_idx = Some(i);
        }
    }
    if let Some(idx) = best_idx {
        GLOBAL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
        let mut buf = global.remove(idx).2;
        drop(global);
        unsafe {
            buf.set_len(words);
        }
        return buf;
    }
    drop(global);
    vec![0u64; words]
}

/// Return a buffer to the pool for reuse.
pub fn give_buffer(dtype: DType, capacity: usize, mut buf: Vec<u64>) {
    GLOBAL_RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed);
    // Don't pool huge buffers
    if !poolable(capacity) {
        return;
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < MAX_FREE {
            buf.clear();
            // Ensure capacity is preserved for next reuse; don't shrink
            pool.push((dtype, capacity, buf));
        }
    })
}

/// Return a buffer to the process-wide free list. Called from the DLPack
/// capsule deleter (Python GC thread); bounded so memory does not grow
/// without limit under many distinct shapes.
pub fn give_buffer_global(dtype: DType, mut buf: Vec<u64>) {
    let capacity = buf.capacity();
    GLOBAL_RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed);
    if !poolable(capacity) {
        return;
    }
    let mut global = GLOBAL_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if global.len() < MAX_GLOBAL_FREE {
        buf.clear();
        global.push((dtype, capacity, buf));
    }
}

/// Allocate an `OwnedTensor` using the pool.
pub fn pooled_tensor(dtype: DType, shape: Vec<i64>) -> OwnedTensor {
    OwnedTensor::new(dtype, shape)
}

/// Return an `OwnedTensor`'s buffer to the pool.
pub fn recycle_tensor(tensor: OwnedTensor) {
    let capacity = tensor.data.capacity();
    let dtype = tensor.dtype;
    let data = tensor.into_pool_buffer();
    give_buffer(dtype, capacity, data);
}

/// Retrieve snapshot of global and thread-local pool metrics.
pub fn get_pool_stats() -> PoolStats {
    let (cached_buffers, cached_words) = POOL.with(|pool| {
        let pool = pool.borrow();
        let bufs = pool.len();
        let words: usize = pool.iter().map(|(_, cap, _)| *cap).sum();
        (bufs, words)
    });
    let (g_bufs, g_words) = {
        let global = GLOBAL_POOL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let bufs = global.len();
        let words: usize = global.iter().map(|(_, cap, _)| *cap).sum();
        (bufs, words)
    };
    PoolStats {
        alloc_count: GLOBAL_ALLOC_COUNT.load(Ordering::Relaxed),
        hit_count: GLOBAL_HIT_COUNT.load(Ordering::Relaxed),
        recycle_count: GLOBAL_RECYCLE_COUNT.load(Ordering::Relaxed),
        cached_buffers: cached_buffers + g_bufs,
        cached_words: cached_words + g_words,
    }
}

/// Reset pool metric counters.
pub fn reset_pool_stats() {
    GLOBAL_ALLOC_COUNT.store(0, Ordering::Relaxed);
    GLOBAL_HIT_COUNT.store(0, Ordering::Relaxed);
    GLOBAL_RECYCLE_COUNT.store(0, Ordering::Relaxed);
}

/// Clear the thread-local pool (useful for memory pressure relief).
pub fn clear_pool() {
    POOL.with(|pool| {
        pool.borrow_mut().clear();
    });
    let mut global = GLOBAL_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    global.clear();
}
