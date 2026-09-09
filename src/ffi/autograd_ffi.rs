//! Autograd FFI: enable/disable, backward passes (single, native, batch).

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::dlpack;

#[pyfunction]
pub fn autograd_enable() {
    crate::autograd::enable();
}

#[pyfunction]
pub fn autograd_disable() {
    crate::autograd::disable();
}

#[pyfunction]
pub fn autograd_is_enabled() -> bool {
    crate::autograd::is_enabled()
}

/// Execute backward on the autograd tape.  `grad_output` is a DLPack capsule
/// of the upstream gradient.  Returns a dict mapping tensor_id → capsule of
/// the accumulated gradient for each leaf tensor.
#[pyfunction]
pub fn autograd_backward(
    py: Python<'_>,
    grad_output: &Bound<'_, PyCapsule>,
) -> PyResult<Vec<(usize, Py<PyCapsule>)>> {
    let grad_view = unsafe { dlpack::BorrowedTensor::from_capsule(grad_output)? };
    match grad_view.dtype {
        dlpack::DType::F32 | dlpack::DType::F64 => {}
        _ => {
            return Err(crate::dlpack::unsupported(
                "autograd_backward only supports f32/f64 upstream",
            ))
        }
    }
    let upstream = unsafe { super::capsule_to_owned(&grad_view) };

    let mut leaf_grads = std::collections::HashMap::new();
    py.allow_threads(|| crate::autograd::backward(&upstream, &mut leaf_grads));

    let mut result = Vec::new();
    for (id, owned) in leaf_grads {
        let cap = dlpack::owned_to_capsule_owned(py, owned)?;
        result.push((id, cap));
    }
    Ok(result)
}

#[pyfunction]
pub fn autograd_reset() {
    crate::autograd::reset();
}

#[pyfunction]
pub fn autograd_tape_len() -> usize {
    crate::autograd::tape_len()
}

/// Execute backward on the native autograd tape.
/// Returns a list of (tensor_id, capsule) pairs for all input gradients.
#[pyfunction]
pub fn backward_native(
    py: Python<'_>,
    grad_output: &Bound<'_, PyCapsule>,
) -> PyResult<Vec<(usize, Py<PyCapsule>)>> {
    let grad_view = unsafe { dlpack::BorrowedTensor::from_capsule(grad_output)? };
    match grad_view.dtype {
        dlpack::DType::F32 | dlpack::DType::F64 => {}
        _ => {
            return Err(crate::dlpack::unsupported(
                "backward_native only supports f32/f64 upstream",
            ))
        }
    }
    let upstream = unsafe { super::capsule_to_owned(&grad_view) };

    let grads = py.allow_threads(|| crate::autograd::backward_native(&upstream));
    let mut result = Vec::new();
    for (id, owned) in grads {
        let cap = dlpack::owned_to_capsule_owned(py, owned)?;
        result.push((id, cap));
    }
    Ok(result)
}

/// Execute a single backward step given an op target, upstream gradient,
/// and saved input tensors.
#[pyfunction]
pub fn backward_single(
    py: Python<'_>,
    target: &str,
    grad_output: &Bound<'_, PyCapsule>,
    saved_inputs: Vec<Bound<'_, PyCapsule>>,
    kwargs_json: &str,
) -> PyResult<Vec<Py<PyCapsule>>> {
    let grad_view = unsafe { dlpack::BorrowedTensor::from_capsule(grad_output)? };
    match grad_view.dtype {
        dlpack::DType::F32 | dlpack::DType::F64 => {}
        _ => {
            return Err(crate::dlpack::unsupported(
                "backward_single only supports f32/f64 upstream",
            ))
        }
    }
    if grad_view.shape.iter().any(|&d| d < 0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "negative dim in upstream shape",
        ));
    }
    let upstream = unsafe { super::capsule_to_owned(&grad_view) };

    let kwargs: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(kwargs_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid kwargs_json: {e}"))
        })?;

    let saved_owned: Vec<dlpack::OwnedTensor> = saved_inputs
        .iter()
        .map(|c| unsafe {
            let b = dlpack::BorrowedTensor::from_capsule(c)?;
            Ok(super::capsule_to_owned(&b))
        })
        .collect::<PyResult<_>>()?;
    let saved_refs: Vec<&dlpack::OwnedTensor> = saved_owned.iter().collect();

    let grads = py.allow_threads(|| {
        crate::autograd::backward_single(target, &upstream, &saved_refs, &kwargs)
    });
    let mut result = Vec::new();
    for owned in grads {
        result.push(dlpack::owned_to_capsule_owned(py, owned)?);
    }
    Ok(result)
}

/// Batch backward: process the entire autograd tape in a single FFI call.
///
/// Instead of calling backward_single once per op (each paying DLPack
/// capsule creation + FFI boundary crossing), this sends the entire tape
/// at once.  Rust does all backward computation and accumulation
/// internally, returning only the final accumulated gradients.
///
/// Returns a list of (tensor_id, gradient_capsule) pairs.
#[pyfunction]
pub fn backward_batch(
    py: Python<'_>,
    targets: Vec<String>,
    all_inputs: Vec<Vec<Bound<'_, PyCapsule>>>,
    all_kwargs: Vec<String>,
    output_ids: Vec<usize>,
    input_ids_all: Vec<Vec<usize>>,
    saved_shapes_all: Vec<Vec<Vec<i64>>>,
    upstream_capsule: &Bound<'_, PyCapsule>,
    initial_output_id: usize,
) -> PyResult<Vec<(usize, Py<PyCapsule>)>> {
    // 1. Convert initial upstream capsule -> OwnedTensor
    let init_view = unsafe { dlpack::BorrowedTensor::from_capsule(upstream_capsule)? };
    let init_owned = unsafe { super::capsule_to_owned(&init_view) };

    // 2. Build batch tape entries (all capsule->OwnedTensor conversions here)
    if !(targets.len() == all_kwargs.len()
        && targets.len() == output_ids.len()
        && targets.len() == input_ids_all.len()
        && targets.len() == all_inputs.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "backward_batch: targets/kwargs/ids/inputs length mismatch",
        ));
    }
    let mut tape = Vec::with_capacity(targets.len());
    for i in 0..targets.len() {
        // Saved input capsules -> Vec<OwnedTensor>
        let mut saved_owned = Vec::with_capacity(all_inputs[i].len());
        for c in &all_inputs[i] {
            let view = unsafe { dlpack::BorrowedTensor::from_capsule(c)? };
            saved_owned.push(unsafe { super::capsule_to_owned(&view) });
        }

        let kwargs: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&all_kwargs[i]).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid kwargs_json[{}]: {e}", i))
            })?;

        let saved_shapes = saved_shapes_all.get(i).cloned().unwrap_or_default();
        if saved_shapes.len() != saved_owned.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "backward_batch[{}]: saved_shapes len {} != inputs len {}",
                i,
                saved_shapes.len(),
                saved_owned.len()
            )));
        }
        tape.push(crate::autograd::BatchTapeEntry {
            target: targets[i].clone(),
            saved_inputs: saved_owned,
            kwargs,
            output_id: output_ids[i],
            input_ids: input_ids_all[i].clone(),
            saved_shapes,
        });
    }

    // 3. Run batch backward -- zero FFI overhead per op (GIL released)
    let grads =
        py.allow_threads(|| crate::autograd::backward_batch(&tape, &init_owned, initial_output_id));

    // 4. Convert accumulated grads -> DLPack capsules (zero-copy)
    let mut result = Vec::with_capacity(grads.len());
    for (tid, owned) in grads {
        let capsule = dlpack::owned_to_capsule_owned(py, owned)?;
        result.push((tid, capsule));
    }
    Ok(result)
}
