//! TorchBurn — a hardware-agnostic PyTorch compilation backend.
//!
//! This crate is the PyO3 FFI layer between PyTorch's Python frontend and a
//! Rust execution engine. Tensors cross the boundary zero-copy via DLPack
//! capsules (REQ-003); graph structure is hashed with BLAKE3 (REQ-004); and
//! unsupported operators are flagged so the Python interpreter can route them
//! to native PyTorch eager execution (REQ-002).

#![warn(clippy::all)]
// Kernel dispatch needs many params (tensor shapes, strides, dtypes)
#![allow(clippy::too_many_arguments)]
// DLPack FFI safety is documented per-callsite; raw pointers are inherent
// to the zero-copy boundary and cannot be abstracted away.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
// Iterating with index-based access is clearer for tensor math kernels
// where the loop variable maps to a spatial dimension.
#![allow(clippy::needless_range_loop)]
// Manual `return` aids readability in long dispatch arms.
#![allow(clippy::needless_return)]
// `*const u8` → `*const T` casts are inherent to the DLPack type-erasure layer.
#![allow(clippy::unnecessary_cast)]
// Excessive float precision is intentional for kernel constants.
#![allow(clippy::excessive_precision)]
// is_multiple_of is not stable on all Rust versions; % 2 is idiomatic.
#![allow(clippy::manual_is_multiple_of)]
// Thread-local lazy init can't be const in all cases.
#![allow(clippy::missing_const_for_thread_local)]
// Dead code in autograd module — these types are used via trait objects.
#![allow(dead_code)]
// Accessing first element via get(0) is clear in the context of node args.
#![allow(clippy::get_first)]
// Explicit into_iter() is clearer for intended ownership semantics.
#![allow(clippy::explicit_into_iter_loop)]
// Manual clamp patterns are clearer in kernel code.
#![allow(clippy::manual_clamp)]
// Auto-deref is intentional in DLPack FFI layer.
#![allow(clippy::needless_borrow)]
// if let is used for readability in dispatch arms.
#![allow(clippy::single_match)]
// format! nesting is clearer than intermediate variables.
#![allow(clippy::to_string_in_format_args)]
// DLPack raw pointer casts are inherent to the FFI boundary.
#![allow(unused_unsafe)]
// Explicit lifetimes improve clarity in the DLPack borrow chain.
#![allow(clippy::needless_lifetimes)]
// Manual range checks are clearer in kernel dispatch code.
#![allow(clippy::manual_range_contains)]
// let-binding returns are used for readability.
#![allow(clippy::let_and_return)]
// if-identical-blocks is intentional for symmetry in kernel dispatch.
#![allow(clippy::if_same_then_else)]
// Manual checked division is clearer in the DLPack bounds checking.
#![allow(clippy::manual_div_ceil)]
// Privacy warnings are intentional — these types are used via trait objects.
#![allow(clippy::type_repetition_in_bounds)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::unnecessary_map_or)]
#![allow(unused_doc_comments)]
#![allow(clippy::map_flatten)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::unnecessary_min_or_max)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]

mod activations;
mod attention;
pub mod autograd;
mod cache;
mod convolution;
mod dlpack;
mod embedding;
mod engine;
mod extra_ops;
mod extra_ops2;
mod extra_ops3;
mod extra_ops4;
mod fft_complex;
mod fusion;
mod linalg;
mod losses;
mod math_ops;
mod norm;
mod ops;
mod ops_phase7;
mod pool;
mod pooling;
mod quantization;
mod reductions;
mod shape_ops;
mod upsample;

#[cfg(feature = "openblas")]
pub mod blas;

#[cfg(feature = "burn")]
mod burn_engine;

#[cfg(feature = "burn-wgpu")]
mod wgpu_backend;

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Execute a payload (JSON node plan) over DLPack capsule inputs.
///
/// Returns one capsule per entry in `payload["outputs"]`.
#[pyfunction]
fn execute(
    py: Python<'_>,
    payload: &str,
    inputs: Vec<Bound<'_, PyCapsule>>,
) -> PyResult<Vec<Py<PyCapsule>>> {
    engine::execute_plan(py, payload, &inputs)
}

/// Execute a payload directly from a Python dict, bypassing JSON.
#[pyfunction]
fn execute_from_dict(
    py: Python<'_>,
    dict: &Bound<'_, pyo3::types::PyDict>,
    inputs: Vec<Bound<'_, PyCapsule>>,
) -> PyResult<Vec<Py<PyCapsule>>> {
    engine::execute_from_dict(py, dict, &inputs)
}

/// Parse a graph dict once and cache it in Rust. Returns a handle.
#[pyfunction]
fn prepare_graph(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<i64> {
    engine::prepare_graph(dict)
}

/// Execute a previously prepared graph with new input tensors.
#[pyfunction]
fn execute_prepared(
    py: Python<'_>,
    handle: i64,
    inputs: Vec<Bound<'_, PyCapsule>>,
) -> PyResult<Vec<Py<PyCapsule>>> {
    engine::execute_prepared(py, handle, &inputs)
}

/// Release a prepared graph from the cache.
#[pyfunction]
fn release_graph(handle: i64) {
    engine::release_graph(handle)
}

/// BLAKE3 structural signature of a graph payload (REQ-004).
#[pyfunction]
fn signature(payload: &str) -> String {
    cache::structural_signature(payload)
}

/// Canonical names of the operators the engine can execute natively.
#[pyfunction]
fn supported_targets() -> Vec<String> {
    engine::supported_targets()
}

/// Name of the active execution engine.
#[pyfunction]
fn active_engine() -> &'static str {
    engine::engine_name()
}

/// Number of worker threads rayon will use (debug helper).
#[pyfunction]
fn rayon_threads() -> usize {
    rayon::current_num_threads()
}

/// Returns GPU adapter information.
///
/// Returns a dict with keys: available, adapter_name, backend, vram_bytes.
#[pyfunction]
fn gpu_info(py: Python<'_>) -> PyResult<pyo3::PyObject> {
    #[cfg(feature = "burn-wgpu")]
    {
        let (available, name, backend, vram) = crate::wgpu_backend::gpu_info();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("available", available)?;
        dict.set_item("adapter_name", &name)?;
        dict.set_item("backend", &backend)?;
        dict.set_item("vram_bytes", vram)?;
        dict.set_item(
            "device_override",
            crate::wgpu_backend::device_override().unwrap_or_default(),
        )?;
        Ok(dict.into())
    }
    #[cfg(not(feature = "burn-wgpu"))]
    {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("available", false)?;
        dict.set_item("adapter_name", "burn-wgpu feature not compiled")?;
        dict.set_item("backend", "none")?;
        dict.set_item("vram_bytes", 0u64)?;
        dict.set_item(
            "device_override",
            std::env::var("TORCHBURN_DEVICE").unwrap_or_default(),
        )?;
        Ok(dict.into())
    }
}

/// Returns the name of the active GPU backend (e.g. "Metal", "Vulkan", "none").
#[pyfunction]
fn gpu_backend() -> String {
    #[cfg(feature = "burn-wgpu")]
    {
        if crate::wgpu_backend::gpu_available() {
            crate::wgpu_backend::gpu_info().2
        } else {
            "none".to_string()
        }
    }
    #[cfg(not(feature = "burn-wgpu"))]
    {
        "none".to_string()
    }
}

/// Check if a GPU adapter is available.
#[pyfunction]
fn gpu_available() -> bool {
    #[cfg(feature = "burn-wgpu")]
    {
        crate::wgpu_backend::gpu_available()
    }
    #[cfg(not(feature = "burn-wgpu"))]
    {
        false
    }
}

/// Debug/verification helper: absolute address of the buffer behind a capsule.
#[pyfunction]
fn data_ptr(capsule: &Bound<'_, PyCapsule>) -> PyResult<usize> {
    dlpack::capsule_data_ptr(capsule)
}

/// Debug helper: dump the raw DLPack fields behind a capsule.
#[pyfunction]
fn capsule_dump(capsule: &Bound<'_, PyCapsule>) -> PyResult<String> {
    dlpack::capsule_debug_dump(capsule)
}

#[pyfunction]
fn autograd_enable() {
    crate::autograd::enable();
}

#[pyfunction]
fn autograd_disable() {
    crate::autograd::disable();
}

#[pyfunction]
fn autograd_is_enabled() -> bool {
    crate::autograd::is_enabled()
}

/// Execute backward on the autograd tape.  `grad_output` is a DLPack capsule
/// of the upstream gradient.  Returns a dict mapping tensor_id -> capsule of
/// the accumulated gradient for each leaf tensor.
#[pyfunction]
fn autograd_backward(
    py: Python<'_>,
    grad_output: &Bound<'_, PyCapsule>,
) -> PyResult<Vec<(usize, Py<PyCapsule>)>> {
    // Read the upstream gradient from the capsule.
    let grad_view = unsafe { dlpack::BorrowedTensor::from_capsule(grad_output)? };
    let upstream = unsafe {
        let n = dlpack::elem_count(&grad_view.shape);
        let mut owned = crate::dlpack::OwnedTensor::new(grad_view.dtype, grad_view.shape.clone());
        match grad_view.dtype {
            dlpack::DType::F32 => {
                let src = std::slice::from_raw_parts(grad_view.data as *const f32, n);
                let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n);
                dst.copy_from_slice(src);
            }
            dlpack::DType::F64 => {
                let src = std::slice::from_raw_parts(grad_view.data as *const f64, n);
                let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n);
                dst.copy_from_slice(src);
            }
            _ => {}
        }
        owned
    };

    let mut leaf_grads = std::collections::HashMap::new();
    py.allow_threads(|| crate::autograd::backward(&upstream, &mut leaf_grads));

    // Convert leaf grads to DLPack capsules (zero-copy move).
    let mut result = Vec::new();
    for (id, owned) in leaf_grads {
        let cap = dlpack::owned_to_capsule_owned(py, owned)?;
        result.push((id, cap));
    }
    Ok(result)
}

#[pyfunction]
fn autograd_reset() {
    crate::autograd::reset();
}

#[pyfunction]
fn autograd_tape_len() -> usize {
    crate::autograd::tape_len()
}

/// Execute backward on the native autograd tape.
/// `grad_output` is a DLPack capsule of the upstream gradient.
/// Returns a list of (tensor_id, capsule) pairs for all input gradients.
#[pyfunction]
fn backward_native(
    py: Python<'_>,
    grad_output: &Bound<'_, PyCapsule>,
) -> PyResult<Vec<(usize, Py<PyCapsule>)>> {
    let grad_view = unsafe { dlpack::BorrowedTensor::from_capsule(grad_output)? };
    let upstream = unsafe {
        let n = dlpack::elem_count(&grad_view.shape);
        let mut owned = crate::dlpack::OwnedTensor::new(grad_view.dtype, grad_view.shape.clone());
        match grad_view.dtype {
            dlpack::DType::F32 => {
                let src = std::slice::from_raw_parts(grad_view.data as *const f32, n);
                let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n);
                dst.copy_from_slice(src);
            }
            dlpack::DType::F64 => {
                let src = std::slice::from_raw_parts(grad_view.data as *const f64, n);
                let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n);
                dst.copy_from_slice(src);
            }
            _ => {}
        }
        owned
    };

    let grads = crate::autograd::backward_native(&upstream);
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
fn backward_single(
    py: Python<'_>,
    target: &str,
    grad_output: &Bound<'_, PyCapsule>,
    saved_inputs: Vec<Bound<'_, PyCapsule>>,
    kwargs_json: &str,
) -> PyResult<Vec<Py<PyCapsule>>> {
    let grad_view = unsafe { dlpack::BorrowedTensor::from_capsule(grad_output)? };
    let upstream = unsafe {
        let n = dlpack::elem_count(&grad_view.shape);
        let mut owned = crate::dlpack::OwnedTensor::new(grad_view.dtype, grad_view.shape.clone());
        match grad_view.dtype {
            dlpack::DType::F32 => {
                let src = std::slice::from_raw_parts(grad_view.data as *const f32, n);
                let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n);
                dst.copy_from_slice(src);
            }
            dlpack::DType::F64 => {
                let src = std::slice::from_raw_parts(grad_view.data as *const f64, n);
                let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n);
                dst.copy_from_slice(src);
            }
            _ => {}
        }
        owned
    };

    let kwargs: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(kwargs_json).unwrap_or_default();

    let saved_borrowed: Vec<dlpack::BorrowedTensor> = saved_inputs
        .iter()
        .map(|c| unsafe { dlpack::BorrowedTensor::from_capsule(c) })
        .collect::<PyResult<_>>()?;
    // Convert BorrowedTensor → OwnedTensor for the backward functions
    let saved_owned: Vec<dlpack::OwnedTensor> = saved_borrowed
        .iter()
        .map(|b| unsafe {
            let n = dlpack::elem_count(&b.shape);
            let mut o = dlpack::OwnedTensor::new(b.dtype, b.shape.clone());
            match b.dtype {
                dlpack::DType::F32 => {
                    let src = std::slice::from_raw_parts(b.data as *const f32, n);
                    let dst = std::slice::from_raw_parts_mut(o.data.as_mut_ptr() as *mut f32, n);
                    dst.copy_from_slice(src);
                }
                dlpack::DType::F64 => {
                    let src = std::slice::from_raw_parts(b.data as *const f64, n);
                    let dst = std::slice::from_raw_parts_mut(o.data.as_mut_ptr() as *mut f64, n);
                    dst.copy_from_slice(src);
                }
                dlpack::DType::I64 => {
                    let src = std::slice::from_raw_parts(b.data as *const i64, n);
                    let dst = std::slice::from_raw_parts_mut(o.data.as_mut_ptr() as *mut i64, n);
                    dst.copy_from_slice(src);
                }
                dlpack::DType::I32 => {
                    let src = std::slice::from_raw_parts(b.data as *const i32, n);
                    let dst = std::slice::from_raw_parts_mut(o.data.as_mut_ptr() as *mut i32, n);
                    dst.copy_from_slice(src);
                }
                dlpack::DType::Bool => {
                    let src = std::slice::from_raw_parts(b.data as *const u8, n);
                    let dst = std::slice::from_raw_parts_mut(o.data.as_mut_ptr() as *mut u8, n);
                    dst.copy_from_slice(src);
                }
            }
            o
        })
        .collect();
    let saved_refs: Vec<&dlpack::OwnedTensor> = saved_owned.iter().collect();

    let grads = crate::autograd::backward_single(target, &upstream, &saved_refs, &kwargs);
    let mut result = Vec::new();
    for owned in grads {
        result.push(dlpack::owned_to_capsule_owned(py, owned)?);
    }
    Ok(result)
}

/// Helper: copy data from a BorrowedTensor (DLPack capsule) into an OwnedTensor.
unsafe fn capsule_to_owned(view: &dlpack::BorrowedTensor) -> dlpack::OwnedTensor {
    let n = dlpack::elem_count(&view.shape);
    let mut owned = dlpack::OwnedTensor::new(view.dtype, view.shape.clone());
    match view.dtype {
        dlpack::DType::F32 => {
            let src = std::slice::from_raw_parts(view.data as *const f32, n);
            let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n);
            dst.copy_from_slice(src);
        }
        dlpack::DType::F64 => {
            let src = std::slice::from_raw_parts(view.data as *const f64, n);
            let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n);
            dst.copy_from_slice(src);
        }
        dlpack::DType::I64 => {
            let src = std::slice::from_raw_parts(view.data as *const i64, n);
            let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut i64, n);
            dst.copy_from_slice(src);
        }
        dlpack::DType::I32 => {
            let src = std::slice::from_raw_parts(view.data as *const i32, n);
            let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut i32, n);
            dst.copy_from_slice(src);
        }
        dlpack::DType::Bool => {
            let src = std::slice::from_raw_parts(view.data as *const u8, n);
            let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut u8, n);
            dst.copy_from_slice(src);
        }
    }
    owned
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
fn backward_batch(
    py: Python<'_>,
    targets: Vec<String>,
    saved_all: Vec<Vec<Bound<'_, PyCapsule>>>,
    all_kwargs: Vec<String>,
    output_ids: Vec<usize>,
    input_ids_all: Vec<Vec<usize>>,
    saved_shapes_all: Vec<Vec<Vec<i64>>>,
    initial_upstream: &Bound<'_, PyCapsule>,
    initial_output_id: usize,
) -> PyResult<Vec<(usize, Py<PyCapsule>)>> {
    // 1. Convert initial upstream capsule -> OwnedTensor
    let init_view = unsafe { dlpack::BorrowedTensor::from_capsule(initial_upstream)? };
    let init_owned = unsafe { capsule_to_owned(&init_view) };

    // 2. Build batch tape entries (all capsule->OwnedTensor conversions here)
    let mut tape = Vec::with_capacity(targets.len());
    for i in 0..targets.len() {
        // Saved input capsules -> Vec<OwnedTensor>
        let mut saved_owned = Vec::with_capacity(saved_all[i].len());
        for c in &saved_all[i] {
            let view = unsafe { dlpack::BorrowedTensor::from_capsule(c)? };
            saved_owned.push(unsafe { capsule_to_owned(&view) });
        }

        let kwargs: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&all_kwargs[i]).unwrap_or_default();

        tape.push(crate::autograd::BatchTapeEntry {
            target: targets[i].clone(),
            saved_inputs: saved_owned,
            kwargs,
            output_id: output_ids[i],
            input_ids: input_ids_all[i].clone(),
            saved_shapes: saved_shapes_all.get(i).cloned().unwrap_or_default(),
        });
    }

    // 3. Run batch backward -- zero FFI overhead per op
    let grads = crate::autograd::backward_batch(&tape, &init_owned, initial_output_id);

    // 4. Convert accumulated grads -> DLPack capsules (zero-copy)
    let mut result = Vec::with_capacity(grads.len());
    for (tid, owned) in grads {
        let capsule = dlpack::owned_to_capsule_owned(py, owned)?;
        result.push((tid, capsule));
    }
    Ok(result)
}

/// Dropout forward pass: apply dropout mask and return output capsule.
/// If training=false, returns input unchanged.
#[pyfunction]
fn dropout_forward(
    py: Python<'_>,
    input: &Bound<'_, PyCapsule>,
    p: f64,
    training: bool,
) -> PyResult<Py<PyCapsule>> {
    let view = unsafe { dlpack::BorrowedTensor::from_capsule(input)? };
    if !training || p == 0.0 {
        // No-op: clone the input into an owned tensor and return.
        let owned = unsafe {
            let mut o = crate::dlpack::OwnedTensor::new(view.dtype, view.shape.clone());
            let n = dlpack::elem_count(&view.shape);
            match view.dtype {
                dlpack::DType::F32 => {
                    let src = std::slice::from_raw_parts(view.data as *const f32, n);
                    let dst = std::slice::from_raw_parts_mut(o.data.as_mut_ptr() as *mut f32, n);
                    dst.copy_from_slice(src);
                }
                dlpack::DType::F64 => {
                    let src = std::slice::from_raw_parts(view.data as *const f64, n);
                    let dst = std::slice::from_raw_parts_mut(o.data.as_mut_ptr() as *mut f64, n);
                    dst.copy_from_slice(src);
                }
                _ => {}
            }
            o
        };
        return dlpack::owned_to_capsule_owned(py, owned);
    }
    let n = dlpack::elem_count(&view.shape);
    let mut out = unsafe { crate::dlpack::OwnedTensor::new(view.dtype, view.shape.clone()) };
    let scale = 1.0 / (1.0 - p);
    let mut mask = Vec::with_capacity(n);

    use crate::dlpack::DType;
    match view.dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(view.data as *const f32, n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, n) };
            for i in 0..n {
                let keep: bool = rand::random::<f64>() >= p;
                mask.push(keep);
                dst[i] = if keep { src[i] * scale as f32 } else { 0.0 };
            }
        }
        DType::F64 => {
            let src = unsafe { std::slice::from_raw_parts(view.data as *const f64, n) };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                let keep: bool = rand::random::<f64>() >= p;
                mask.push(keep);
                dst[i] = if keep { src[i] * scale } else { 0.0 };
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "dropout only supports f32/f64",
            ));
        }
    }

    // Note: dropout autograd recording is handled by the Python wrapper,
    // not from this raw capsule-level function.

    dlpack::owned_to_capsule_owned(py, out)
}

#[pyfunction]
fn memory_pool_stats(py: Python<'_>) -> PyResult<PyObject> {
    let stats = pool::get_pool_stats();
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
fn clear_memory_pool() -> PyResult<()> {
    pool::clear_pool();
    pool::reset_pool_stats();
    Ok(())
}

#[pymodule]
fn _torchburn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    m.add_function(wrap_pyfunction!(execute_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_graph, m)?)?;
    m.add_function(wrap_pyfunction!(execute_prepared, m)?)?;
    m.add_function(wrap_pyfunction!(release_graph, m)?)?;
    m.add_function(wrap_pyfunction!(signature, m)?)?;
    m.add_function(wrap_pyfunction!(supported_targets, m)?)?;
    m.add_function(wrap_pyfunction!(active_engine, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_threads, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_backend, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(capsule_dump, m)?)?;
    m.add_function(wrap_pyfunction!(autograd_enable, m)?)?;
    m.add_function(wrap_pyfunction!(autograd_disable, m)?)?;
    m.add_function(wrap_pyfunction!(autograd_is_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(autograd_backward, m)?)?;
    m.add_function(wrap_pyfunction!(autograd_reset, m)?)?;
    m.add_function(wrap_pyfunction!(autograd_tape_len, m)?)?;
    m.add_function(wrap_pyfunction!(backward_native, m)?)?;
    m.add_function(wrap_pyfunction!(backward_single, m)?)?;
    m.add_function(wrap_pyfunction!(backward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(dropout_forward, m)?)?;
    m.add_function(wrap_pyfunction!(cache::cache_get, m)?)?;
    m.add_function(wrap_pyfunction!(cache::cache_put, m)?)?;
    m.add_function(wrap_pyfunction!(cache::cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(cache::cache_clear, m)?)?;
    m.add_function(wrap_pyfunction!(memory_pool_stats, m)?)?;
    m.add_function(wrap_pyfunction!(clear_memory_pool, m)?)?;
    Ok(())
}
