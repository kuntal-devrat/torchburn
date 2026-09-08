//! BLAKE3 structural graph cache (REQ-004).
//!
//! The *structural signature* of a graph is a BLAKE3 hash of its canonical
//! JSON payload — node sequence, operator targets, input shapes/dtypes —
//! deliberately excluding any tensor *data*. Identical graph structures
//! therefore share one compiled plan and skip re-tracing.

use pyo3::prelude::*;
use serde_json::Value;
use std::sync::LazyLock;
use std::sync::RwLock;

/// Thread-safe LRU cache: BLAKE3 signature -> canonical JSON payload string.
///
/// Perf notes (hot path: one lookup per `execute` call):
/// * Values are stored pre-serialized (`String`) so a hit is a single
///   `clone()` — no `serde_json::to_string` per hit (was O(payload) serialize
///   on every warm run).
/// * Backed by `ahash` (fast DoS-resistant hash) instead of std SipHash.
/// * `get` fast path: `RwLock::read` for the common hit-at-MRU case with no
///   promotion write; only a miss-of-MRU takes the write lock for LRU
///   reordering. This removes the global write-serialization on steady-state
///   inference (previously every `cache_get` took `write()` + O(1024)
///   `order.retain`).
/// * LRU promotion uses positional `remove` (single shift) instead of
///   `retain` closure over the whole queue.
struct LruCache {
    map: std::collections::HashMap<String, String, ahash::RandomState>,
    order: std::collections::VecDeque<String>,
}

impl LruCache {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::with_hasher(ahash::RandomState::new()),
            order: std::collections::VecDeque::new(),
        }
    }
    /// Read-only probe (no LRU promotion). Returns cloned payload on hit.
    #[inline]
    fn get_readonly(&self, key: &str) -> Option<String> {
        self.map.get(key).cloned()
    }
    /// Check whether `key` is already MRU (no promotion needed).
    #[inline]
    fn is_mru(&self, key: &str) -> bool {
        self.order.back().is_some_and(|k| k == key)
    }
    /// Promote `key` to MRU. Caller must hold the write lock and `key` must
    /// exist in `map`.
    fn promote(&mut self, key: &str) {
        if self.is_mru(key) {
            return;
        }
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.to_string());
    }
    fn insert(&mut self, key: String, value: String) {
        if self.map.contains_key(&key) {
            return; // first-write-wins
        }
        if self.map.len() >= 1024 {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }
    fn len(&self) -> usize {
        self.map.len()
    }
    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

static GRAPH_CACHE: LazyLock<RwLock<LruCache>> = LazyLock::new(|| RwLock::new(LruCache::new()));

static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// BLAKE3 structural signature over the canonical payload string.
pub fn structural_signature(payload: &str) -> String {
    blake3::hash(payload.as_bytes()).to_hex().to_string()
}

/// Maximum payload size accepted (DoS protection).
pub const MAX_PAYLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Cache lookup: returns the canonicalized payload JSON on a hit.
#[pyfunction]
pub fn cache_get(signature: &str) -> Option<String> {
    // Fast path: read lock only. If the entry is already MRU no promotion
    // write is needed at all (steady-state inference hits here).
    let (hit, needs_promote) = {
        let cache = GRAPH_CACHE.read().unwrap_or_else(|e| e.into_inner());
        match cache.get_readonly(signature) {
            Some(v) => (Some(v), !cache.is_mru(signature)),
            None => (None, false),
        }
    };
    if let Some(v) = hit {
        if needs_promote {
            // Slow path: single write lock to record LRU recency.
            let mut cache = GRAPH_CACHE.write().unwrap_or_else(|e| e.into_inner());
            // Re-check existence (may have been evicted/cleared).
            if cache.map.contains_key(signature) {
                cache.promote(signature);
            }
        }
        HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    } else {
        MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }
}

/// Cache insert: stores the parsed payload under its signature (first write wins).
#[pyfunction]
pub fn cache_put(signature: &str, payload: &str) {
    if payload.len() > MAX_PAYLOAD_BYTES {
        return;
    }
    // Canonicalize once at insert (validates JSON); hits then skip serialization.
    if let Ok(value) = serde_json::from_str::<Value>(payload) {
        let canonical = serde_json::to_string(&value).unwrap_or_default();
        if canonical.is_empty() {
            return;
        }
        let mut cache = GRAPH_CACHE.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(signature.to_string(), canonical);
    }
}

/// (size, hits, misses) — used by `torchburn.cache_stats()`.
#[pyfunction]
pub fn cache_stats() -> (usize, u64, u64) {
    let size = GRAPH_CACHE.read().unwrap_or_else(|e| e.into_inner()).len();
    let hits = HITS.load(std::sync::atomic::Ordering::Relaxed);
    let misses = MISSES.load(std::sync::atomic::Ordering::Relaxed);
    (size, hits, misses)
}

/// Reset the graph cache (mostly useful for tests).
#[pyfunction]
pub fn cache_clear() {
    GRAPH_CACHE
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    MISSES.store(0, std::sync::atomic::Ordering::Relaxed);
}
