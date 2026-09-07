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
/// If training=false, returns input unchanged.
#[pyfunction]
pub fn dropout_forward(
    py: Python<'_>,
    input: &Bound<'_, PyCapsule>,
    p: f64,
    training: bool,
) -> PyResult<Py<PyCapsule>> {
    let view = unsafe { dlpack::BorrowedTensor::from_capsule(input)? };
    if !training || p == 0.0 {
        let owned = unsafe { super::capsule_to_owned(&view) };
        return dlpack::owned_to_capsule_owned(py, owned);
    }
    let n = dlpack::elem_count(&view.shape);
    let mut out = unsafe { dlpack::OwnedTensor::new(view.dtype, view.shape.clone()) };
    let scale = 1.0 / (1.0 - p);

    use dlpack::DType;
    match view.dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(view.data as *const f32, n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, n) };
            for i in 0..n {
                let keep: bool = rand::random::<f64>() >= p;
                dst[i] = if keep { src[i] * scale as f32 } else { 0.0 };
            }
        }
        DType::F64 => {
            let src = unsafe { std::slice::from_raw_parts(view.data as *const f64, n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                let keep: bool = rand::random::<f64>() >= p;
                dst[i] = if keep { src[i] * scale } else { 0.0 };
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "dropout only supports f32/f64",
            ));
        }
    }

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
