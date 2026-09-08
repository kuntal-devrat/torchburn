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

/// Copy data from a `BorrowedTensor` (DLPack capsule view) into an owned
/// Rust allocation.  Used by the autograd and dropout FFI wrappers.
pub(crate) unsafe fn capsule_to_owned(view: &dlpack::BorrowedTensor) -> dlpack::OwnedTensor {
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
        dlpack::DType::F16 | dlpack::DType::BF16 => {
            // F16/BF16 are 2-byte types — copy as raw u16
            let src = std::slice::from_raw_parts(view.data as *const u16, n);
            let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut u16, n);
            dst.copy_from_slice(src);
        }
        dlpack::DType::I8 | dlpack::DType::U8 | dlpack::DType::Bool => {
            let src = std::slice::from_raw_parts(view.data as *const u8, n);
            let dst = std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut u8, n);
            dst.copy_from_slice(src);
        }
    }
    owned
}
