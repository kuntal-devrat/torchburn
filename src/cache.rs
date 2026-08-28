//! BLAKE3 structural graph cache (REQ-004).
//!
//! The *structural signature* of a graph is a BLAKE3 hash of its canonical
//! JSON payload — node sequence, operator targets, input shapes/dtypes —
//! deliberately excluding any tensor *data*. Identical graph structures
//! therefore share one compiled plan and skip re-tracing.

use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::LazyLock;

/// Thread-safe LRU cache: BLAKE3 signature -> canonical parsed payload.
/// Uses RwLock for better read concurrency (cache lookups are read-only hot path).
/// Maintains insertion order for true LRU eviction.
struct LruCache {
    map: HashMap<String, Value>,
    order: std::collections::VecDeque<String>,
}

impl LruCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }
    fn get(&mut self, key: &str) -> Option<&Value> {
        if self.map.contains_key(key) {
            // Move to back (most recent)
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_string());
            self.map.get(key)
        } else {
            None
        }
    }
    fn insert(&mut self, key: String, value: Value) {
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

static GRAPH_CACHE: LazyLock<RwLock<LruCache>> =
    LazyLock::new(|| RwLock::new(LruCache::new()));

static HITS: LazyLock<RwLock<u64>> = LazyLock::new(|| RwLock::new(0));
static MISSES: LazyLock<RwLock<u64>> = LazyLock::new(|| RwLock::new(0));

/// BLAKE3 structural signature over the canonical payload string.
pub fn structural_signature(payload: &str) -> String {
    blake3::hash(payload.as_bytes()).to_hex().to_string()
}

/// Maximum payload size accepted (DoS protection).
pub const MAX_PAYLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Cache lookup: returns the canonicalized payload JSON on a hit.
#[pyfunction]
pub fn cache_get(signature: &str) -> Option<String> {
    let result = {
        let mut cache = GRAPH_CACHE.write().unwrap_or_else(|e| e.into_inner());
        cache.get(signature).map(|v| serde_json::to_string(v).expect("canonical payload is JSON"))
    };
    if result.is_some() {
        *HITS.write().unwrap_or_else(|e| e.into_inner()) += 1;
    } else {
        *MISSES.write().unwrap_or_else(|e| e.into_inner()) += 1;
    }
    result
}

/// Cache insert: stores the parsed payload under its signature (first write wins).
#[pyfunction]
pub fn cache_put(signature: &str, payload: &str) {
    if payload.len() > MAX_PAYLOAD_BYTES {
        return;
    }
    if let Ok(value) = serde_json::from_str::<Value>(payload) {
        let mut cache = GRAPH_CACHE.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(signature.to_string(), value);
    }
}

/// (size, hits, misses) — used by `torchburn.cache_stats()`.
#[pyfunction]
pub fn cache_stats() -> (usize, u64, u64) {
    let size = GRAPH_CACHE.read().unwrap_or_else(|e| e.into_inner()).len();
    let hits = *HITS.read().unwrap_or_else(|e| e.into_inner());
    let misses = *MISSES.read().unwrap_or_else(|e| e.into_inner());
    (size, hits, misses)
}

/// Reset the graph cache (mostly useful for tests).
#[pyfunction]
pub fn cache_clear() {
    GRAPH_CACHE.write().unwrap_or_else(|e| e.into_inner()).clear();
    *HITS.write().unwrap_or_else(|e| e.into_inner()) = 0;
    *MISSES.write().unwrap_or_else(|e| e.into_inner()) = 0;
}
