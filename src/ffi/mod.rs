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

/// Copy data from a `BorrowedTensor` into an owned allocation.
/// Handles non-contiguous (strided) views via gather; contiguous fast-path memcpys.
pub(crate) unsafe fn capsule_to_owned(view: &dlpack::BorrowedTensor) -> dlpack::OwnedTensor {
    let n = dlpack::elem_count(&view.shape);
    let mut owned = dlpack::OwnedTensor::new(view.dtype, view.shape.to_vec());
    if n == 0 {
        return owned;
    }
    if view.is_contiguous() {
        let bytes = n * view.dtype.elem_size();
        std::ptr::copy_nonoverlapping(view.data, owned.data.as_mut_ptr() as *mut u8, bytes);
        return owned;
    }
    let ndim = view.shape.len();
    let elem = view.dtype.elem_size();
    let mut idx = vec![0i64; ndim];
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
        let src = view.data.add((phys.max(0) as usize) * elem);
        let dst = (owned.data.as_mut_ptr() as *mut u8).add(out_off * elem);
        std::ptr::copy_nonoverlapping(src, dst, elem);
    }
    owned
}
