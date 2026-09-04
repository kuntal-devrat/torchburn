//! Zero-copy DLPack FFI layer (REQ-003).
//!
//! PyTorch tensors expose their native buffers through the DLPack open
//! standard: `tensor.__dlpack__()` returns a `PyCapsule` wrapping a
//! `DLManagedTensor*`. We read the raw pointer from the capsule and operate
//! on the *same* memory PyTorch owns — no O(N) data duplication.
//!
//! Ownership rules (mirroring the DLPack spec):
//! * **Inputs** (borrowed): we only read `DLManagedTensor` fields for the
//!   duration of a synchronous call. The Python caller keeps the capsule
//!   (and therefore the underlying buffer) alive.
//! * **Outputs** (owned): we allocate the result buffer, box a
//!   `DLManagedTensor` around it, and hand it to Python inside a capsule.
//!   The capsule destructor invokes the DLPack deleter, so memory is freed
//!   exactly once regardless of who consumes the capsule first
//!   (`torch.from_dlpack` nulls the capsule destructor and takes over).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use std::os::raw::c_void;

/// DLDeviceType: kDLCPU
pub const DL_DEVICE_CPU: i32 = 1;
/// DLDataTypeCode: kDLInt / kDLUInt / kDLFloat / kDLOBool
pub const DL_DTYPE_INT: u8 = 0;
#[allow(dead_code)]
pub const DL_DTYPE_UINT: u8 = 1;
pub const DL_DTYPE_FLOAT: u8 = 2;
/// kDLOBool (torch bool tensors are code 6, 8 bits).
pub const DL_DTYPE_BOOL: u8 = 6;

/// Marker prefix used on errors that Python should treat as "route this node
/// to the eager fallback pipeline" (REQ-002) rather than a hard crash.
pub const UNSUPPORTED_MARKER: &str = "TB_UNSUPPORTED:";

pub fn unsupported(msg: &str) -> PyErr {
    PyRuntimeError::new_err(format!("{UNSUPPORTED_MARKER} {msg}"))
}

// ---------------------------------------------------------------------------
// DLPack C ABI structures (DLPack v0.8)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

pub type Deleter = unsafe extern "C" fn(*mut DLManagedTensor);

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<Deleter>,
}

// ---------------------------------------------------------------------------
// DTypes supported by the native engine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F64,
    /// Signed 64-bit integers (indices, class targets).
    I64,
    /// Signed 32-bit integers (compact indices).
    I32,
    /// Boolean masks (attention masks, `where` conditions).
    Bool,
}

impl DType {
    pub fn elem_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I64 => 8,
            DType::I32 => 4,
            DType::Bool => 1,
        }
    }

    pub fn dl_code(self) -> u8 {
        match self {
            DType::F32 | DType::F64 => DL_DTYPE_FLOAT,
            DType::I64 | DType::I32 => DL_DTYPE_INT,
            DType::Bool => DL_DTYPE_BOOL,
        }
    }

    pub fn dl_bits(self) -> u8 {
        match self {
            DType::F32 => 32,
            DType::F64 => 64,
            DType::I64 => 64,
            DType::I32 => 32,
            DType::Bool => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::Bool => "bool",
        }
    }
}

pub fn dtype_from_spec(spec: &str) -> Option<DType> {
    match spec {
        "f32" => Some(DType::F32),
        "f64" => Some(DType::F64),
        "i64" => Some(DType::I64),
        "i32" => Some(DType::I32),
        "bool" => Some(DType::Bool),
        _ => None,
    }
}

/// Row-contiguous strides for a shape (DLPack stride units are elements).
pub fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i].max(0));
    }
    strides
}

pub fn elem_count(shape: &[i64]) -> usize {
    shape.iter().map(|&d| d.max(0) as usize).product()
}

// ---------------------------------------------------------------------------
// Borrowed (zero-copy) view over a PyTorch-owned buffer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct BorrowedTensor {
    /// Raw pointer to element data (already offset by `byte_offset`).
    pub data: *const u8,
    pub shape: Vec<i64>,
    /// Element strides; always materialized (contiguous if the capsule says so).
    pub strides: Vec<i64>,
    pub dtype: DType,
}

// SAFETY: a `BorrowedTensor` is a read-only view over buffers owned by the
// Python caller (capsules kept alive for the whole synchronous call) or by an
// `OwnedTensor` that outlives it. Kernel threads only read through the view
// while the GIL is released, and the underlying buffers cannot be mutated or
// freed during the call.
unsafe impl Send for BorrowedTensor {}
// SAFETY: same reasoning as `Send` — the view is read-only and its buffers
// are immutably borrowed for the duration of the call, so sharing the view
// across kernel worker threads is safe.
unsafe impl Sync for BorrowedTensor {}

impl BorrowedTensor {
    // Used by the (optional) burn engine; the native path extracts CapsuleRefs.
    #[cfg_attr(not(feature = "burn"), allow(dead_code))]
    pub fn from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let ptr = capsule.pointer();
        if ptr.is_null() {
            return Err(PyValueError::new_err("null DLPack capsule pointer"));
        }
        // SAFETY: `ptr` is a `DLManagedTensor*` produced by `tensor.__dlpack__()`
        // and the capsule is alive for the duration of this synchronous call.
        unsafe { Self::from_managed(ptr as *mut DLManagedTensor) }
    }

    /// Read a view directly from a raw `DLManagedTensor*` (no Python API).
    /// Safe to call with the GIL released: the caller guarantees the capsule
    /// stays alive and unmutated for the duration of the call.
    // SAFETY: `ptr` must reference a valid `DLManagedTensor` whose backing
    // capsule outlives the resulting view.
    pub(crate) unsafe fn from_managed(ptr: *mut DLManagedTensor) -> PyResult<Self> {
        unsafe {
            let dl = &(*(ptr as *const DLManagedTensor)).dl_tensor;
            if dl.device.device_type != DL_DEVICE_CPU {
                return Err(unsupported(&format!(
                    "non-CPU tensor (device_type={})",
                    dl.device.device_type
                )));
            }
            let dtype = match (dl.dtype.code, dl.dtype.bits) {
                (DL_DTYPE_FLOAT, 32) => DType::F32,
                (DL_DTYPE_FLOAT, 64) => DType::F64,
                (DL_DTYPE_INT, 64) => DType::I64,
                (DL_DTYPE_INT, 32) => DType::I32,
                (DL_DTYPE_BOOL, 8) => DType::Bool,
                (code, bits) => {
                    return Err(unsupported(&format!(
                        "unsupported dtype (code={code}, bits={bits})"
                    )))
                }
            };
            if dl.ndim < 0 {
                return Err(PyValueError::new_err("negative ndim in DLPack tensor"));
            }
            if dl.ndim as usize > 32 {
                return Err(PyValueError::new_err(format!(
                    "ndim too large in DLPack tensor: {}",
                    dl.ndim
                )));
            }
            let ndim = dl.ndim as usize;
            if ndim > 0 && dl.shape.is_null() {
                return Err(PyValueError::new_err("null shape pointer in DLPack tensor"));
            }
            let shape: Vec<i64> = std::slice::from_raw_parts(dl.shape, ndim).to_vec();
            if shape.iter().any(|&d| d < 0) {
                return Err(PyValueError::new_err("negative dimension in DLPack shape"));
            }
            let strides: Vec<i64> = if dl.strides.is_null() {
                contiguous_strides(&shape)
            } else {
                std::slice::from_raw_parts(dl.strides, ndim).to_vec()
            };
            let is_empty = elem_count(&shape) == 0;
            let data = if is_empty {
                std::ptr::NonNull::<u64>::dangling().as_ptr() as *const u8
            } else if dl.data.is_null() {
                return Err(PyValueError::new_err("null data pointer in DLPack tensor"));
            } else {
                let d = dl.data.cast::<u8>();
                if dl.byte_offset > 1_000_000_000 {
                    return Err(PyValueError::new_err(format!(
                        "byte_offset too large: {}",
                        dl.byte_offset
                    )));
                }
                match (d as usize).checked_add(dl.byte_offset as usize) {
                    Some(addr) => addr as *const u8,
                    None => return Err(PyValueError::new_err("byte_offset overflow")),
                }
            };
            // Validate that byte_offset is element-aligned for non-empty tensors
            if !is_empty {
                let elem_size = dtype.elem_size();
                if (dl.byte_offset as usize) % elem_size != 0 {
                    return Err(PyValueError::new_err(format!(
                        "byte_offset {} not aligned to dtype {} ({} bytes)",
                        dl.byte_offset,
                        dtype.name(),
                        elem_size
                    )));
                }
            }
            Ok(BorrowedTensor {
                data,
                shape,
                strides,
                dtype,
            })
        }
    }

    /// Borrowed view over a Rust-owned tensor (used for intermediate results).
    pub fn from_owned(t: &OwnedTensor) -> Self {
        let shape = t.shape.clone();
        let strides = contiguous_strides(&shape);
        BorrowedTensor {
            data: t.data.as_ptr() as *const u8,
            shape,
            strides,
            dtype: t.dtype,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected_stride: i64 = 1;
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()).rev() {
            if dim == 0 {
                return true;
            }
            if dim == 1 {
                continue;
            }
            if stride != expected_stride {
                return false;
            }
            expected_stride = expected_stride.saturating_mul(dim);
        }
        true
    }

    #[allow(dead_code)]
    pub fn elem_count(&self) -> usize {
        elem_count(&self.shape)
    }

    pub fn buffer_len(&self) -> usize {
        if self.shape.is_empty() {
            return 1;
        }
        if self.shape.iter().any(|&d| d <= 0) {
            return 0;
        }
        let mut max_idx: i64 = 0;
        for d in 0..self.shape.len() {
            if self.shape[d] > 1 {
                max_idx += (self.shape[d] - 1) * self.strides[d].abs();
            }
        }
        let count = elem_count(&self.shape);
        ((max_idx + 1).max(0) as usize).max(count)
    }
}

// ---------------------------------------------------------------------------
// Owned tensor (Rust-allocated, handed back to Python via a capsule)
// ---------------------------------------------------------------------------

/// Backing storage is `Vec<u64>` to guarantee 8-byte alignment for both
/// f32 (4) and f64 (8) payloads.
#[derive(Clone)]
pub struct OwnedTensor {
    pub data: Vec<u64>,
    pub shape: Vec<i64>,
    pub dtype: DType,
}

impl Default for OwnedTensor {
    fn default() -> Self {
        OwnedTensor {
            data: Vec::new(),
            shape: Vec::new(),
            dtype: DType::F32,
        }
    }
}

impl OwnedTensor {
    pub fn new(dtype: DType, shape: Vec<i64>) -> Self {
        let bytes = elem_count(&shape) * dtype.elem_size();
        let words = bytes.div_ceil(8);
        let data = crate::pool::take_buffer(dtype, words);
        OwnedTensor { data, shape, dtype }
    }

    pub fn elem_count(&self) -> usize {
        elem_count(&self.shape)
    }

    /// Borrowed view over this owned tensor.
    pub fn as_view(&self) -> BorrowedTensor {
        BorrowedTensor::from_owned(self)
    }

    /// Create from a pooled buffer (avoids allocation).
    pub fn from_pool(dtype: DType, shape: Vec<i64>, mut data: Vec<u64>) -> Self {
        let bytes = elem_count(&shape) * dtype.elem_size();
        let words = bytes.div_ceil(8);
        data.resize(words, 0u64);
        OwnedTensor { data, shape, dtype }
    }

    /// Consume self, returning the raw buffer for pool recycling.
    pub fn into_pool_buffer(self) -> Vec<u64> {
        self.data
    }
}

/// The boxed object behind every output capsule: the `DLManagedTensor` first
/// so a `*mut DLManagedTensor` cast from `Box<ManagedBuffer>` is valid.
#[repr(C)]
struct ManagedBuffer {
    dl: DLManagedTensor,
    data: Vec<u64>,
    shape: Vec<i64>,
}

/// DLPack deleter: frees the boxed `ManagedBuffer` (reconstructed from the
/// raw pointer we boxed).
unsafe extern "C" fn managed_buffer_deleter(ptr: *mut DLManagedTensor) {
    let _ = Box::from_raw(ptr as *mut ManagedBuffer);
}

/// PyCapsule destructor for capsules we produce.
///
/// DLPack ownership protocol: a consumer that takes the tensor renames the
/// capsule (e.g. torch's `from_dlpack` renames it to `used_dltensor`) and
/// becomes responsible for calling the deleter. `PyCapsule_IsValid(.., "dltensor")`
/// is false for renamed capsules, so a consumed capsule is a no-op here and
/// the buffer is freed exactly once.
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(capsule, c"dltensor".as_ptr()) != 1 {
            return; // already consumed (renamed) or invalid
        }
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr());
        let managed = ptr as *mut DLManagedTensor;
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}

/// Wrap an owned tensor in a fresh `PyCapsule` named `"dltensor"` that
/// PyTorch's `torch.from_dlpack` can consume.
///
/// Created via the raw C API (not `PyCapsule::new_with_destructor`) so the
/// destructor can implement the DLPack name-check protocol above.
pub fn owned_to_capsule(py: Python<'_>, tensor: &OwnedTensor) -> PyResult<Py<PyCapsule>> {
    let data = tensor.data.clone();
    let shape = tensor.shape.clone();
    let dtype = tensor.dtype;
    // Raw pointers taken before moving the Vecs into the box remain valid:
    // moving a `Vec` never moves its heap allocation.
    let data_ptr = data.as_ptr() as *mut c_void;
    let shape_ptr = shape.as_ptr() as *mut i64;
    let buffer = ManagedBuffer {
        dl: DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: DL_DEVICE_CPU,
                    device_id: 0,
                },
                ndim: shape.len() as i32,
                dtype: DLDataType {
                    code: dtype.dl_code(),
                    bits: dtype.dl_bits(),
                    lanes: 1,
                },
                shape: shape_ptr,
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(managed_buffer_deleter),
        },
        data,
        shape,
    };
    let raw = Box::into_raw(Box::new(buffer)) as *mut DLManagedTensor;
    // SAFETY: PyCapsule_New either returns a valid owned reference or null;
    // the capsule owns `raw` and frees it via the destructor above.
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            raw as *mut c_void,
            c"dltensor".as_ptr(),
            Some(dlpack_capsule_destructor),
        )
    };
    let capsule: Bound<'_, PyCapsule> =
        unsafe { Bound::from_owned_ptr_or_err(py, capsule_ptr)?.downcast_into_unchecked() };
    Ok(capsule.unbind())
}

/// Zero-copy variant that takes ownership of the tensor (avoids clone).
pub fn owned_to_capsule_owned(py: Python<'_>, tensor: OwnedTensor) -> PyResult<Py<PyCapsule>> {
    let dtype = tensor.dtype;
    let data = tensor.data;
    let shape = tensor.shape;
    let data_ptr = data.as_ptr() as *mut c_void;
    let shape_ptr = shape.as_ptr() as *mut i64;
    let buffer = ManagedBuffer {
        dl: DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: DL_DEVICE_CPU,
                    device_id: 0,
                },
                ndim: shape.len() as i32,
                dtype: DLDataType {
                    code: dtype.dl_code(),
                    bits: dtype.dl_bits(),
                    lanes: 1,
                },
                shape: shape_ptr,
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(managed_buffer_deleter),
        },
        data,
        shape,
    };
    let raw = Box::into_raw(Box::new(buffer)) as *mut DLManagedTensor;
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            raw as *mut c_void,
            c"dltensor".as_ptr(),
            Some(dlpack_capsule_destructor),
        )
    };
    let capsule: Bound<'_, PyCapsule> =
        unsafe { Bound::from_owned_ptr_or_err(py, capsule_ptr)?.downcast_into_unchecked() };
    Ok(capsule.unbind())
}

/// A Send/Sync wrapper around a capsule's `DLManagedTensor*`, extracted while
/// the GIL is held so kernels can run with the GIL released.
#[derive(Clone, Copy)]
pub struct CapsuleRef(pub(crate) *mut DLManagedTensor);

// SAFETY: the capsule (and its buffer) is kept alive by the Python caller for
// the whole synchronous `execute` call; views created from it are read-only
// and share nothing mutable, so moving/sharing the pointer is safe.
unsafe impl Send for CapsuleRef {}
unsafe impl Sync for CapsuleRef {}

/// Extract the raw managed-tensor pointer from a capsule (GIL held).
pub fn capsule_ref(capsule: &Bound<'_, PyCapsule>) -> PyResult<CapsuleRef> {
    let ptr = capsule.pointer();
    if ptr.is_null() {
        return Err(PyValueError::new_err("null DLPack capsule pointer"));
    }
    Ok(CapsuleRef(ptr as *mut DLManagedTensor))
}

/// Debug helper used by the zero-copy tests: the absolute address of the
/// buffer behind a capsule, so we can prove Rust reads PyTorch's memory
/// in place.
pub fn capsule_data_ptr(capsule: &Bound<'_, PyCapsule>) -> PyResult<usize> {
    let ptr = capsule.pointer();
    if ptr.is_null() {
        return Err(PyValueError::new_err("null DLPack capsule pointer"));
    }
    // SAFETY: capsule alive for the call; the DLManagedTensor is valid.
    unsafe {
        let dl = &(*(ptr as *const DLManagedTensor)).dl_tensor;
        Ok(dl.data as usize + dl.byte_offset as usize)
    }
}

/// Debug helper: human-readable dump of the raw DLPack fields.
pub fn capsule_debug_dump(capsule: &Bound<'_, PyCapsule>) -> PyResult<String> {
    let ptr = capsule.pointer();
    if ptr.is_null() {
        return Err(PyValueError::new_err("null DLPack capsule pointer"));
    }
    // SAFETY: capsule alive for the call; the DLManagedTensor is valid.
    unsafe {
        let managed = &*(ptr as *const DLManagedTensor);
        let dl = &managed.dl_tensor;
        let shape: Vec<i64> = if dl.ndim > 0 && dl.ndim < 64 {
            std::slice::from_raw_parts(dl.shape, dl.ndim as usize).to_vec()
        } else {
            Vec::new()
        };
        Ok(format!(
            "data={:#x} device=({},{}) ndim={} dtype=(code={},bits={},lanes={}) shape={shape:?} byte_offset={} deleter={:#x}",
            dl.data as usize,
            dl.device.device_type,
            dl.device.device_id,
            dl.ndim,
            dl.dtype.code,
            dl.dtype.bits,
            dl.dtype.lanes,
            dl.byte_offset,
            managed.deleter.map(|d| d as usize).unwrap_or(0),
        ))
    }
}
