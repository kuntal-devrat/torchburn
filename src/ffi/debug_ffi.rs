//! Debug and diagnostic FFI wrappers.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::{dispatch, dlpack};

/// Reports the resolved CPU feature tier (Phase 0.3 observability).
///
/// Returns a dict with keys: tier, avx2, avx512f, avx512bw, avx512vnni, neon.
#[pyfunction]
pub fn cpu_features_report(py: Python<'_>) -> PyResult<pyo3::PyObject> {
    let feats = dispatch::cpu_features();
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("tier", feats.tier_name())?;
    dict.set_item("avx2", feats.avx2)?;
    dict.set_item("avx512f", feats.avx512f)?;
    dict.set_item("avx512bw", feats.avx512bw)?;
    dict.set_item("avx512vnni", feats.avx512vnni)?;
    dict.set_item("neon", feats.neon)?;
    Ok(dict.into())
}

/// Debug/verification helper: absolute address of the buffer behind a capsule.
#[pyfunction]
pub fn data_ptr(capsule: &Bound<'_, PyCapsule>) -> PyResult<usize> {
    dlpack::capsule_data_ptr(capsule)
}

/// Debug helper: dump the raw DLPack fields behind a capsule.
#[pyfunction]
pub fn capsule_dump(capsule: &Bound<'_, PyCapsule>) -> PyResult<String> {
    dlpack::capsule_debug_dump(capsule)
}
