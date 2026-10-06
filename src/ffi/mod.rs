//! PyO3 FFI boundary layer — all `#[pyfunction]` wrappers live here.
//!
//! Submodules are organized by domain; the crate root (`lib.rs`) only
//! declares modules and registers them in the `#[pymodule]`.

pub mod autograd_ffi;
pub mod debug_ffi;
pub mod engine_ffi;
pub mod gguf_ffi;
pub mod gpu_ffi;
pub mod quantization_ffi;

// ---------------------------------------------------------------------------
// Shared helper: capsule → OwnedTensor copy
// ---------------------------------------------------------------------------

use crate::dlpack;

/// Convert a caught Rust panic payload into a fallback-eligible Python error.
///
/// PyO3 converts uncaught panics into `PanicException`, which derives from
/// `BaseException` (like `SystemExit`) and therefore *not* caught by the
/// Python interpreter's `except Exception` fallback net. Wrapping engine entry
/// points in `catch_unwind` + this mapper turns a would-be process-killing
/// panic into a `TB_UNSUPPORTED` `RuntimeError`, so the Python side degrades
/// to eager PyTorch execution instead of crashing the host application.
pub(crate) fn panic_to_pyerr(payload: Box<dyn std::any::Any + Send>) -> pyo3::PyErr {
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_string());
    dlpack::unsupported(&format!("native engine panic: {msg}"))
}

/// Copy data from a `BorrowedTensor` into an owned allocation.
/// Handles non-contiguous (strided) views via gather; contiguous fast-path memcpys.
pub(crate) unsafe fn capsule_to_owned(view: &dlpack::BorrowedTensor) -> dlpack::OwnedTensor {
    let n = dlpack::elem_count(&view.shape);
    let mut owned = dlpack::OwnedTensor::new(view.dtype, view.shape.to_vec());
    if n == 0 {
        return owned;
    }
    let elem = view.dtype.elem_size();
    if view.is_contiguous() {
        let bytes = n * elem;
        std::ptr::copy_nonoverlapping(view.data, owned.data.as_mut_ptr() as *mut u8, bytes);
        return owned;
    }
    // Fast path: 2D row-contiguous strided tensor (e.g. sliced rows)
    if view.shape.len() == 2 && view.strides[1] == 1 && view.strides[0] >= 0 {
        let rows = view.shape[0].max(0) as usize;
        let cols = view.shape[1].max(0) as usize;
        let row_stride = view.strides[0] as usize;
        let row_bytes = cols * elem;
        let dst_base = owned.data.as_mut_ptr() as *mut u8;
        for r in 0..rows {
            let src = view.data.add(r * row_stride * elem);
            let dst = dst_base.add(r * cols * elem);
            std::ptr::copy_nonoverlapping(src, dst, row_bytes);
        }
        return owned;
    }
    let ndim = view.shape.len();
    let mut idx = vec![0i64; ndim];
    let dst_base = owned.data.as_mut_ptr() as *mut u8;
    // Upper bound on valid physical offset (elements) for bounds checking.
    let buf_len = view.buffer_len();
    for out_off in 0..n {
        let mut rem = out_off;
        for d in (0..ndim).rev() {
            let dim = view.shape[d].max(1) as usize;
            idx[d] = (rem % dim) as i64;
            rem /= dim;
        }
        let mut phys: i64 = 0;
        for d in 0..ndim {
            phys += idx[d] * view.strides[d];
        }
        let phys_usize = phys.max(0) as usize;
        // Bounds check: skip elements beyond the valid buffer range to
        // prevent out-of-bounds reads from malformed DLPack capsules.
        if phys_usize >= buf_len {
            continue;
        }
        let src = view.data.add(phys_usize * elem);
        let dst = dst_base.add(out_off * elem);
        std::ptr::copy_nonoverlapping(src, dst, elem);
    }
    owned
}
