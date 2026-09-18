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

/// Maximum number of free buffers to retain per bucket in the thread pool.
const MAX_FREE_PER_BUCKET: usize = 8;
/// Maximum number of free buffers to retain in the global (deleter-fed) list.
const MAX_GLOBAL_FREE: usize = 48;
/// Number of size-class buckets (powers of 2 from 128 to 16M words).
const NUM_BUCKETS: usize = 18;

static GLOBAL_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_HIT_COUNT: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_RECYCLE_COUNT: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of total cached words (thread-local + global) for
/// production memory pressure diagnostics.
static GLOBAL_PEAK_CACHED_WORDS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug)]
pub struct PoolStats {
    pub alloc_count: usize,
    pub hit_count: usize,
    pub recycle_count: usize,
    pub cached_buffers: usize,
    pub cached_words: usize,
    /// Peak total cached words observed since last reset.
    pub peak_cached_words: usize,
}

type BucketList = Vec<(usize, Vec<u64>)>;
const EMPTY_BUCKET_LIST: BucketList = Vec::new();
const EMPTY_DTYPE_BUCKETS: [BucketList; NUM_BUCKETS] = [EMPTY_BUCKET_LIST; NUM_BUCKETS];

/// Per-thread pool of reusable buffers segregated by DType and size class (O(1) lookup).
thread_local! {
    static POOL: RefCell<[[BucketList; NUM_BUCKETS]; DType::NUM_DTYPES]> =
        RefCell::new([EMPTY_DTYPE_BUCKETS; DType::NUM_DTYPES]);
}

/// Process-wide free list fed by the DLPack capsule deleter (Python GC thread)
/// and drained by `take_buffer` (execution thread).
static GLOBAL_POOL: Mutex<Vec<(DType, usize, Vec<u64>)>> = Mutex::new(Vec::new());

/// Large buffers (in u64 words, i.e. >128MB of f32) are not pooled to avoid
/// unbounded memory retention.
#[inline(always)]
fn poolable(words: usize) -> bool {
    words <= 16_000_000 // ~128MB
}

/// Map a word count to a size-class bucket index.
/// Buckets are powers of 2: [128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K,
///                           64K, 128K, 256K, 512K, 1M, 2M, 4M, 8M, 16M]
#[inline(always)]
fn bucket_index(words: usize) -> usize {
    if words <= 128 {
        return 0;
    }
    let bits = usize::BITS - (words - 1).leading_zeros();
    // bucket 0 = 128 words (2^7), bucket 1 = 256 (2^8), ...
    let idx = (bits as usize).saturating_sub(7);
    idx.min(NUM_BUCKETS - 1)
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
        let dtype_buckets = &mut pool[dtype.index()];
        let want_idx = bucket_index(words);

        // Check exact size-class bucket first, then next larger class (want_idx + 1)
        for b_idx in want_idx..=(want_idx + 1).min(NUM_BUCKETS - 1) {
            let bucket = &mut dtype_buckets[b_idx];
            for i in (0..bucket.len()).rev() {
                let (cap, _) = bucket[i];
                if cap >= words && cap <= words.saturating_mul(4).max(128) {
                    GLOBAL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                    let mut buf = bucket.swap_remove(i).1;
                    if words <= buf.capacity() {
                        unsafe {
                            buf.set_len(words);
                        }
                    } else {
                        buf.resize(words, 0u64);
                    }
                    #[cfg(debug_assertions)]
                    buf.fill(0);
                    return buf;
                }
            }
        }
        drop(pool);
        take_buffer_global(dtype, words)
    })
}

/// Best-fit drain of the global free list (fed by the capsule deleter).
/// Only hit on thread-local miss, so the `Mutex` is off the fast path.
fn take_buffer_global(dtype: DType, words: usize) -> Vec<u64> {
    let mut global = GLOBAL_POOL.lock().unwrap_or_else(|e| e.into_inner());
    let mut best_idx: Option<usize> = None;
    let mut best_cap = usize::MAX;
    for (i, (d, cap, _)) in global.iter().enumerate() {
        if *d != dtype || *cap < words {
            continue;
        }
        if *cap > words.saturating_mul(4).max(128) {
            continue;
        }
        if *cap < best_cap {
            best_cap = *cap;
            best_idx = Some(i);
            if *cap == words {
                break;
            }
        }
    }
    if let Some(idx) = best_idx {
        GLOBAL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
        let mut buf = global.swap_remove(idx).2;
        drop(global);
        // SAFETY: the best-fit scan above guarantees cap >= words, but
        // we verify defensively to prevent UB on logic errors.
        if words <= buf.capacity() {
            unsafe {
                buf.set_len(words);
            }
        } else {
            buf.resize(words, 0u64);
        }
        #[cfg(debug_assertions)]
        buf.fill(0);
        return buf;
    }
    drop(global);
    vec![0u64; words]
}

/// Return a buffer to the pool for reuse.
///
/// Enforces `MAX_FREE_PER_BUCKET` **per size-class bucket** (previously the
/// cap was global, so one shape could evict all others and steady-state
/// inference with 2-3 live shapes thrashed the allocator). Buffers whose
/// capacity exceeds 4x the bucket floor are dropped to bound waste.
pub fn give_buffer(dtype: DType, capacity: usize, buf: Vec<u64>) {
    GLOBAL_RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed);
    // Use actual Vec capacity (caller may pass stale capacity after set_len).
    let capacity = buf.capacity().max(capacity);
    // Don't pool huge buffers
    if !poolable(capacity) {
        return;
    }
    // Bound waste: drop buffers far larger than their nominal bucket.
    // bucket floor = 128 << idx; allow up to 4x before dropping.
    let idx = bucket_index(capacity);
    let floor = 128usize << idx.min(24);
    if capacity > floor.saturating_mul(4) {
        return;
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let bucket = &mut pool[dtype.index()][idx];
        if bucket.len() >= MAX_FREE_PER_BUCKET {
            return;
        }
        // Don't clear: kernels must fully overwrite output (pool contract).
        // The buffer keeps its capacity for reuse; set_len restores it in take_buffer.
        bucket.push((capacity, buf));
    })
}

/// Return a buffer to the process-wide free list. Called from the DLPack
/// capsule deleter (Python GC thread); bounded so memory does not grow
/// without limit under many distinct shapes.
pub fn give_buffer_global(dtype: DType, buf: Vec<u64>) {
    let capacity = buf.capacity();
    GLOBAL_RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed);
    if !poolable(capacity) {
        return;
    }
    let mut global = GLOBAL_POOL.lock().unwrap_or_else(|e| e.into_inner());
    if global.len() < MAX_GLOBAL_FREE {
        // Don't clear: kernels must fully overwrite output (pool contract).
        global.push((dtype, capacity, buf));
        // Update peak-cached-words high-water mark
        let total: usize = global.iter().map(|(_, cap, _)| *cap).sum();
        let _ = GLOBAL_PEAK_CACHED_WORDS.fetch_max(total, Ordering::Relaxed);
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
        let mut bufs = 0;
        let mut words = 0;
        for dtype_buckets in pool.iter() {
            for bucket in dtype_buckets.iter() {
                bufs += bucket.len();
                words += bucket.iter().map(|&(cap, _)| cap).sum::<usize>();
            }
        }
        (bufs, words)
    });
    let (g_bufs, g_words) = {
        let global = GLOBAL_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let bufs = global.len();
        let words: usize = global.iter().map(|(_, cap, _)| *cap).sum();
        (bufs, words)
    };
    let total_words = cached_words + g_words;
    let peak = GLOBAL_PEAK_CACHED_WORDS
        .fetch_max(total_words, Ordering::Relaxed)
        .max(total_words);
    PoolStats {
        alloc_count: GLOBAL_ALLOC_COUNT.load(Ordering::Relaxed),
        hit_count: GLOBAL_HIT_COUNT.load(Ordering::Relaxed),
        recycle_count: GLOBAL_RECYCLE_COUNT.load(Ordering::Relaxed),
        cached_buffers: cached_buffers + g_bufs,
        cached_words: total_words,
        peak_cached_words: peak,
    }
}

/// Reset pool metric counters.
pub fn reset_pool_stats() {
    GLOBAL_ALLOC_COUNT.store(0, Ordering::Relaxed);
    GLOBAL_HIT_COUNT.store(0, Ordering::Relaxed);
    GLOBAL_RECYCLE_COUNT.store(0, Ordering::Relaxed);
    GLOBAL_PEAK_CACHED_WORDS.store(0, Ordering::Relaxed);
}

/// Clear the thread-local pool (useful for memory pressure relief).
pub fn clear_pool() {
    POOL.with(|pool| {
        for dtype_buckets in pool.borrow_mut().iter_mut() {
            for bucket in dtype_buckets.iter_mut() {
                bucket.clear();
            }
        }
    });
    let mut global = GLOBAL_POOL.lock().unwrap_or_else(|e| e.into_inner());
    global.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_take_and_recycle_buffer() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_pool();
        reset_pool_stats();

        // Fresh allocation
        let buf1 = take_buffer(DType::F32, 100);
        assert_eq!(buf1.len(), 100);
        assert!(buf1.capacity() >= 100);
        let initial_allocs = GLOBAL_ALLOC_COUNT.load(Ordering::Relaxed);
        assert!(initial_allocs >= 1);

        // Recycle it
        let cap = buf1.capacity();
        give_buffer(DType::F32, cap, buf1);
        let recycles = GLOBAL_RECYCLE_COUNT.load(Ordering::Relaxed);
        assert!(recycles >= 1);

        // Take again - should hit the pool
        let buf2 = take_buffer(DType::F32, 80);
        assert_eq!(buf2.len(), 80);
        assert!(buf2.capacity() >= cap);
        let hits = GLOBAL_HIT_COUNT.load(Ordering::Relaxed);
        assert!(hits >= 1);
    }

    #[test]
    fn test_clear_pool() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_pool();
        let buf = take_buffer(DType::F32, 64);
        let cap = buf.capacity();
        give_buffer(DType::F32, cap, buf);

        let stats_before = get_pool_stats();
        assert!(stats_before.cached_buffers >= 1);

        clear_pool();
        let stats_after = get_pool_stats();
        assert_eq!(stats_after.cached_buffers, 0);
        assert_eq!(stats_after.cached_words, 0);
    }

    #[test]
    fn test_dtype_isolation() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_pool();
        let buf_f32 = take_buffer(DType::F32, 50);
        let cap = buf_f32.capacity();
        give_buffer(DType::F32, cap, buf_f32);

        // Asking for F64 should not take from F32 pool bucket
        let hits_before = GLOBAL_HIT_COUNT.load(Ordering::Relaxed);
        let buf_f64 = take_buffer(DType::F64, 50);
        let hits_after = GLOBAL_HIT_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            hits_before, hits_after,
            "F64 request must not hit F32 bucket"
        );
        assert_eq!(buf_f64.len(), 50);
    }

    #[test]
    fn test_edge_case_sizes() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_pool();
        // 0 words
        let buf0 = take_buffer(DType::F32, 0);
        assert_eq!(buf0.len(), 0);
        give_buffer(DType::F32, buf0.capacity(), buf0);

        clear_pool();

        // Allocation larger than 16M words (~128MB) should not be pooled
        let huge_words = 16_000_001;
        let buf_huge = take_buffer(DType::F32, huge_words);
        assert_eq!(buf_huge.len(), huge_words);
        give_buffer(DType::F32, buf_huge.capacity(), buf_huge);
        let stats = get_pool_stats();
        assert_eq!(
            stats.cached_buffers, 0,
            "Huge buffers exceeding 16M words should not be cached"
        );
    }

    #[test]
    fn test_pooled_tensor_roundtrip() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_pool();
        let tensor = pooled_tensor(DType::F32, vec![2, 3, 4]);
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.elem_count(), 24);
        recycle_tensor(tensor);
        let stats = get_pool_stats();
        assert!(stats.cached_buffers >= 1);
    }
}
