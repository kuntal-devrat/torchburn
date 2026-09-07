//! Extra 48 ops batch 4 — native 450 total ops.
//! Zero-copy DLPack kernels, f32/f64 + rayon, matching PyTorch semantics within 1e-5.

#![allow(unused_imports, clippy::all, dead_code)]
use crate::dlpack::{
    contiguous_strides, elem_count, unsupported, BorrowedTensor, DType, OwnedTensor,
};
use pyo3::prelude::*;
use std::f64::consts::PI;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}
const PAR_CHUNK: usize = 16 * 1024;

// 1. isclose

pub mod compare_extra;

pub use self::compare_extra::{allclose, equal, is_complex, is_nonzero, isclose, isreal};

pub mod nan_stats;

pub use self::nan_stats::{corrcoef, cov, nanmax, nanmedian, nanmin, nanprod, std_mean, var_mean};

pub mod split_extra;

pub use self::split_extra::{
    as_strided, broadcast_tensors, broadcast_to, dsplit, hsplit, index_reduce, scatter_max,
    scatter_min, split, take_along_dim, tensor_split, vsplit,
};

pub mod linalg_ex;

pub use self::linalg_ex::{
    linalg_cholesky_ex, linalg_cross, linalg_inv_ex, linalg_lu_factor, linalg_multi_dot,
    linalg_solve_ex, linalg_tensordot, linalg_vander, linalg_vecdot,
};

pub mod pool_extra;

pub use self::pool_extra::{
    adaptive_avg_pool1d, adaptive_max_pool1d, local_response_norm, lp_pool3d,
};

pub mod misc_extra;

pub use self::misc_extra::{
    empty_strided, expand_as, isfinite, istft, logsumexp, masked_select, rand_like, randint_like,
    randn_like, view_as,
};
