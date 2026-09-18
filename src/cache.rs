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
struct LruNode {
    key: String,
    value: String,
    prev: usize,
    next: usize,
}

const NIL: usize = usize::MAX;

struct LruCache {
    map: std::collections::HashMap<String, usize, ahash::RandomState>,
    nodes: Vec<LruNode>,
    free: Vec<usize>,
    head: usize, // LRU
    tail: usize, // MRU
    capacity: usize,
}

impl LruCache {
    fn new() -> Self {
        let capacity = std::env::var("TORCHBURN_CACHE_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1024);
        Self {
            map: std::collections::HashMap::with_hasher(ahash::RandomState::new()),
            nodes: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            capacity,
        }
    }

    /// Read-only probe (no LRU promotion). Returns cloned payload on hit.
    #[inline]
    fn get_readonly(&self, key: &str) -> Option<String> {
        let &idx = self.map.get(key)?;
        Some(self.nodes[idx].value.clone())
    }

    /// Check whether `key` is already MRU (no promotion needed).
    #[inline]
    fn is_mru(&self, key: &str) -> bool {
        if self.tail == NIL {
            return false;
        }
        self.nodes[self.tail].key == key
    }

    /// Promote `key` to MRU in O(1). Caller must hold the write lock and `key`
    /// must exist in `map`.
    fn promote(&mut self, key: &str) {
        let &idx = match self.map.get(key) {
            Some(i) => i,
            None => return,
        };
        if idx == self.tail {
            return;
        }

        // Unlink from current position
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;
        if prev != NIL {
            self.nodes[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.nodes[next].prev = prev;
        }

        // Attach to tail (MRU)
        self.nodes[idx].prev = self.tail;
        self.nodes[idx].next = NIL;
        if self.tail != NIL {
            self.nodes[self.tail].next = idx;
        }
        self.tail = idx;
        if self.head == NIL {
            self.head = idx;
        }
    }

    fn insert(&mut self, key: String, value: String) {
        if self.map.contains_key(&key) {
            return; // first-write-wins
        }

        // Evict LRU (head) if at capacity
        if self.map.len() >= self.capacity && self.head != NIL {
            let evict_idx = self.head;
            let evict_key = self.nodes[evict_idx].key.clone();
            self.map.remove(&evict_key);

            let next = self.nodes[evict_idx].next;
            self.head = next;
            if next != NIL {
                self.nodes[next].prev = NIL;
            } else {
                self.tail = NIL;
            }
            self.free.push(evict_idx);
        }

        // Allocate slot for new node
        let idx = if let Some(free_idx) = self.free.pop() {
            self.nodes[free_idx] = LruNode {
                key: key.clone(),
                value,
                prev: self.tail,
                next: NIL,
            };
            free_idx
        } else {
            let new_idx = self.nodes.len();
            self.nodes.push(LruNode {
                key: key.clone(),
                value,
                prev: self.tail,
                next: NIL,
            });
            new_idx
        };

        if self.tail != NIL {
            self.nodes[self.tail].next = idx;
        }
        self.tail = idx;
        if self.head == NIL {
            self.head = idx;
        }
        self.map.insert(key, idx);
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.nodes.clear();
        self.free.clear();
        self.head = NIL;
        self.tail = NIL;
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
