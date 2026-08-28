//! Memory pool for intermediate tensor reuse.
//!
//! Reuses `Vec<u64>` allocations across kernel calls to avoid repeated
//! malloc/free overhead on the hot path. The pool is per-thread to avoid
//! lock contention in multi-threaded execution.

use crate::dlpack::{DType, OwnedTensor};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum number of free buffers to retain in the thread pool.
const MAX_FREE: usize = 64;

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

/// Borrow a buffer from the pool, or allocate a new one.
/// The returned `Vec<u64>` has length `words` and is zeroed.
pub fn take_buffer(dtype: DType, words: usize) -> Vec<u64> {
    GLOBAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for i in 0..pool.len() {
            if pool[i].0 == dtype && pool[i].1 >= words {
                GLOBAL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                let mut buf = pool.remove(i).2;
                buf.resize(words, 0u64);
                buf.fill(0u64);
                return buf;
            }
        }
        vec![0u64; words]
    })
}

/// Return a buffer to the pool for reuse.
pub fn give_buffer(dtype: DType, capacity: usize, mut buf: Vec<u64>) {
    GLOBAL_RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed);
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < MAX_FREE {
            buf.clear();
            pool.push((dtype, capacity, buf));
        }
    })
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
    PoolStats {
        alloc_count: GLOBAL_ALLOC_COUNT.load(Ordering::Relaxed),
        hit_count: GLOBAL_HIT_COUNT.load(Ordering::Relaxed),
        recycle_count: GLOBAL_RECYCLE_COUNT.load(Ordering::Relaxed),
        cached_buffers,
        cached_words,
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
}
