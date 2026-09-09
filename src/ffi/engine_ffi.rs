//! Core engine FFI: graph execution, caching, memory pool, and dropout.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::{cache, dlpack, engine, memory_pool};

/// Execute a payload (JSON node plan) over DLPack capsule inputs.
///
/// Returns one capsule per entry in `payload["outputs"]`.
#[pyfunction]
pub fn execute(
    py: Python<'_>,
    payload: &str,
    inputs: Vec<Bound<'_, PyCapsule>>,
) -> PyResult<Vec<Py<PyCapsule>>> {
    engine::execute_plan(py, payload, &inputs)
}

/// Execute a payload directly from a Python dict, bypassing JSON.
#[pyfunction]
pub fn execute_from_dict(
    py: Python<'_>,
    dict: &Bound<'_, pyo3::types::PyDict>,
    inputs: Vec<Bound<'_, PyCapsule>>,
) -> PyResult<Vec<Py<PyCapsule>>> {
    engine::execute_from_dict(py, dict, &inputs)
}

/// Parse a graph dict once and cache it in Rust. Returns a handle.
#[pyfunction]
pub fn prepare_graph(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<i64> {
    engine::prepare_graph(dict)
}

/// Execute a previously prepared graph with new input tensors.
#[pyfunction]
pub fn execute_prepared(
    py: Python<'_>,
    handle: i64,
    inputs: Vec<Bound<'_, PyCapsule>>,
) -> PyResult<Vec<Py<PyCapsule>>> {
    engine::execute_prepared(py, handle, &inputs)
}

/// Release a prepared graph from the cache.
#[pyfunction]
pub fn release_graph(handle: i64) {
    engine::release_graph(handle)
}

/// BLAKE3 structural signature of a graph payload (REQ-004).
#[pyfunction]
pub fn signature(payload: &str) -> String {
    cache::structural_signature(payload)
}

/// Canonical names of the operators the engine can execute natively.
#[pyfunction]
pub fn supported_targets() -> Vec<String> {
    engine::supported_targets()
}

/// Name of the active execution engine.
#[pyfunction]
pub fn active_engine() -> &'static str {
    engine::engine_name()
}

/// Number of worker threads rayon will use (debug helper).
#[pyfunction]
pub fn rayon_threads() -> usize {
    rayon::current_num_threads()
}

/// Dropout forward pass: apply dropout mask and return output capsule.
/// If training=false, returns input unchanged (still a copy for capsule ownership).
#[pyfunction]
pub fn dropout_forward(
    py: Python<'_>,
    input: &Bound<'_, PyCapsule>,
    p: f64,
    training: bool,
) -> PyResult<Py<PyCapsule>> {
    use dlpack::DType;
    if !training || p <= 0.0 {
        let view = unsafe { dlpack::BorrowedTensor::from_capsule(input)? };
        let owned = unsafe { super::capsule_to_owned(&view) };
        return dlpack::owned_to_capsule_owned(py, owned);
    }
    if !(0.0..1.0).contains(&p) || !p.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dropout p must be in [0,1), got {p}"
        )));
    }
    let view = unsafe { dlpack::BorrowedTensor::from_capsule(input)? };
    match view.dtype {
        DType::F32 | DType::F64 => {}
        _ => {
            // TB_UNSUPPORTED so the Python interpreter can fallback cleanly
            return Err(crate::dlpack::unsupported("dropout only supports f32/f64"));
        }
    }
    let dtype = view.dtype;
    let shape = view.shape.to_vec();
    // Clone into owned src (handles strided) on GIL, compute off GIL
    let src_owned = unsafe { super::capsule_to_owned(&view) };
    let out = py.allow_threads(move || {
        use rayon::prelude::*;
        #[inline(always)]
        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        #[inline(always)]
        fn next_f64(state: &mut u64) -> f64 {
            // 53-bit uniform in [0,1)
            const DIV: f64 = (1u64 << 53) as f64;
            ((splitmix64(state) >> 11) as f64) / DIV
        }
        let n = dlpack::elem_count(&shape);
        let mut out = dlpack::OwnedTensor::new(dtype, shape);
        let scale = 1.0 / (1.0 - p);
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ (n as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .unwrap_or(0x1234_5678_9ABC_DEF0);
        match dtype {
            DType::F32 => {
                let src =
                    unsafe { std::slice::from_raw_parts(src_owned.data.as_ptr() as *const f32, n) };
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, n) };
                dst.par_chunks_mut(16_384)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let mut st = seed.wrapping_add(ci as u64 * 0x9E3779B9).max(1);
                        let base = ci * 16_384;
                        for (j, d) in chunk.iter_mut().enumerate() {
                            let keep = next_f64(&mut st) >= p;
                            *d = if keep {
                                src[base + j] * scale as f32
                            } else {
                                0.0
                            };
                        }
                    });
            }
            DType::F64 => {
                let src =
                    unsafe { std::slice::from_raw_parts(src_owned.data.as_ptr() as *const f64, n) };
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, n) };
                dst.par_chunks_mut(16_384)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let mut st = seed.wrapping_add(ci as u64 * 0x9E3779B9).max(1);
                        let base = ci * 16_384;
                        for (j, d) in chunk.iter_mut().enumerate() {
                            let keep = next_f64(&mut st) >= p;
                            *d = if keep { src[base + j] * scale } else { 0.0 };
                        }
                    });
            }
            _ => unreachable!(),
        }
        out
    });
    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
pub fn memory_pool_stats(py: Python<'_>) -> PyResult<PyObject> {
    let stats = memory_pool::get_pool_stats();
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("alloc_count", stats.alloc_count)?;
    dict.set_item("hit_count", stats.hit_count)?;
    dict.set_item("recycle_count", stats.recycle_count)?;
    dict.set_item("cached_buffers", stats.cached_buffers)?;
    dict.set_item("cached_words", stats.cached_words)?;
    let hit_rate = if stats.alloc_count > 0 {
        stats.hit_count as f64 / stats.alloc_count as f64
    } else {
        0.0
    };
    dict.set_item("hit_rate", hit_rate)?;
    Ok(dict.into())
}

#[pyfunction]
pub fn clear_memory_pool() -> PyResult<()> {
    memory_pool::clear_pool();
    memory_pool::reset_pool_stats();
    Ok(())
}
