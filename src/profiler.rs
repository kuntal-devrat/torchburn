//! First-party op-level profiling for the TorchBurn dispatch loop.
//!
//! Provides microsecond-resolution per-op timing that mirrors `torch.profiler`
//! and `llama.cpp`'s `LLAMA_PERF` output. Zero overhead when disabled.
//!
//! # Usage (Python)
//!
//! ```python
//! import torchburn._torchburn as _tb
//!
//! _tb.profiler_enable()
//! model(inputs)                       # traced call
//! report = _tb.profiler_report()      # list of (op, calls, total_us, min_us, max_us)
//! _tb.profiler_reset()
//! _tb.profiler_disable()
//! ```
//!
//! # Design
//!
//! * Enabled/disabled via a `static AtomicBool` — the hot path is a single
//!   relaxed load, adding ~1 ns when disabled.
//! * Per-op stats are accumulated in a process-global `Mutex<HashMap>`.
//!   The mutex is only acquired at the *end* of each op dispatch, after the
//!   kernel has already run, so it does not add latency to the critical path.
//! * Wall-clock time is measured with `std::time::Instant` (monotonic).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static PROFILER_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, Default)]
pub struct OpStats {
    pub calls: u64,
    pub total_us: u64,
    pub min_us: u64,
    pub max_us: u64,
}

fn op_stats() -> &'static Mutex<HashMap<String, OpStats>> {
    static INSTANCE: OnceLock<Mutex<HashMap<String, OpStats>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// Core API (used internally from dispatch_node)
// ---------------------------------------------------------------------------

/// Returns true if profiling is currently active. O(1) relaxed atomic load.
#[inline(always)]
pub fn is_enabled() -> bool {
    PROFILER_ENABLED.load(Ordering::Relaxed)
}

/// Record elapsed microseconds for one op call. Called by dispatch_node.
pub fn record(op: &str, elapsed_us: u64) {
    if let Ok(mut map) = op_stats().lock() {
        let entry = map.entry(op.to_string()).or_insert_with(|| OpStats {
            calls: 0,
            total_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        });
        entry.calls += 1;
        entry.total_us += elapsed_us;
        if elapsed_us < entry.min_us {
            entry.min_us = elapsed_us;
        }
        if elapsed_us > entry.max_us {
            entry.max_us = elapsed_us;
        }
    }
}

use std::borrow::Cow;

/// Guard type: records elapsed time when dropped.
pub struct OpTimer {
    op: Cow<'static, str>,
    start: Instant,
}

impl OpTimer {
    #[inline(always)]
    pub fn start(op: &'static str) -> Self {
        Self {
            op: Cow::Borrowed(op),
            start: Instant::now(),
        }
    }

    #[inline(always)]
    pub fn start_str(op: &str) -> Self {
        Self {
            op: Cow::Owned(op.to_string()),
            start: Instant::now(),
        }
    }

    #[inline(always)]
    pub fn start_owned(op: String) -> Self {
        Self {
            op: Cow::Owned(op),
            start: Instant::now(),
        }
    }
}

impl Drop for OpTimer {
    #[inline]
    fn drop(&mut self) {
        let us = self.start.elapsed().as_micros() as u64;
        record(&self.op, us);
    }
}

/// Begin timing an op by static target string. Returns `None` when profiling is off.
#[inline(always)]
pub fn maybe_time(target: &'static str) -> Option<OpTimer> {
    if is_enabled() {
        Some(OpTimer::start(target))
    } else {
        None
    }
}

/// Begin timing an op by dynamic target string. Returns `None` when profiling is off.
#[inline(always)]
pub fn maybe_time_str(target: &str) -> Option<OpTimer> {
    if is_enabled() {
        Some(OpTimer::start_str(target))
    } else {
        None
    }
}

#[inline(always)]
pub fn maybe_time_dyn(target: &str, start: Instant) {
    if is_enabled() {
        let us = start.elapsed().as_micros() as u64;
        record(target, us);
    }
}

// ---------------------------------------------------------------------------
// Python-facing API
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn profiler_enable() {
    PROFILER_ENABLED.store(true, Ordering::Relaxed);
}

#[pyfunction]
pub fn profiler_disable() {
    PROFILER_ENABLED.store(false, Ordering::Relaxed);
}

#[pyfunction]
pub fn profiler_reset() {
    if let Ok(mut map) = op_stats().lock() {
        map.clear();
    }
}

/// Returns a list of `(op_name, calls, total_us, min_us, max_us)` sorted by
/// total_us descending — the slowest ops first, matching torch.profiler style.
#[pyfunction]
pub fn profiler_report() -> Vec<(String, u64, u64, u64, u64)> {
    let map = match op_stats().lock() {
        Ok(m) => m,
        Err(_) => return vec![],
    };
    let mut rows: Vec<(String, u64, u64, u64, u64)> = map
        .iter()
        .map(|(name, s)| {
            (
                name.clone(),
                s.calls,
                s.total_us,
                if s.min_us == u64::MAX { 0 } else { s.min_us },
                s.max_us,
            )
        })
        .collect();
    // Sort by total_us descending (hottest ops first)
    rows.sort_unstable_by_key(|b| std::cmp::Reverse(b.2));
    rows
}

/// Pretty-print the profiler report, matching llama.cpp LLAMA_PERF style.
#[pyfunction]
pub fn profiler_print() {
    let rows = profiler_report();
    if rows.is_empty() {
        eprintln!("[torchburn profiler] No data recorded.");
        return;
    }
    eprintln!(
        "\n[torchburn profiler] Op timing report ({} ops):",
        rows.len()
    );
    eprintln!(
        "{:<40} {:>8} {:>12} {:>10} {:>10}",
        "op", "calls", "total_us", "min_us", "max_us"
    );
    eprintln!("{}", "-".repeat(84));
    for (op, calls, total, min, max) in &rows {
        eprintln!(
            "{:<40} {:>8} {:>12} {:>10} {:>10}",
            op, calls, total, min, max
        );
    }
    eprintln!();
}
