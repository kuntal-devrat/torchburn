//! Extra 150 ops batch 3 — truly native for 375 total ops.
//! All kernels are zero-copy DLPack compatible, supporting f32/f64 with rayon parallelism.
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

pub mod float_special;

pub use self::float_special::{
    bessel_j0, bessel_j1, bessel_y0, bessel_y1, deg2rad, digamma, erfcinv, erfinv, expit,
    heaviside, i0, i0e, i1, i1e, lgamma, log_ndtr, logaddexp, logaddexp2, logit, mvlgamma,
    nan_to_num, ndtr, ndtri, nextafter, polygamma, rad2deg, sinc,
};

pub mod int_extra;

pub use self::int_extra::{fmax, fmin, gcd, lcm, maximum, minimum, signbit};

pub mod blas_extra;

pub use self::blas_extra::{
    addbmm, addcdiv, addcmul, addmv, addr, baddbmm, ger, inner, kron, mv, outer, vdot,
};

pub mod numeric_extra;

pub use self::numeric_extra::{cumulative_trapezoid, trapezoid, trapz};

pub mod activations_extra;

pub use self::activations_extra::{
    celu, hardshrink, logsigmoid, rrelu, softshrink, tanhshrink, threshold,
};

pub mod losses_extra;

pub use self::losses_extra::{
    cosine_embedding_loss, ctc_loss, hinge_embedding_loss, kl_div, margin_ranking_loss,
    multilabel_margin_loss, multilabel_soft_margin_loss, poisson_nll_loss, soft_margin_loss,
    triplet_margin_loss,
};

pub mod windows_extra;

pub use self::windows_extra::{
    exponential_window, gaussian_window, hamming_window, kaiser_window, triangular_window,
};

pub mod linalg_decomp;

pub use self::linalg_decomp::{
    cholesky, cholesky_inverse, cholesky_solve, cross, eig, eigh, eigvals, eigvalsh,
    frobenius_norm, linalg_norm, lu, matrix_power, matrix_rank, nuclear_norm, qr, svd, svdvals,
    triangular_solve,
};

pub mod scatter_extra;

pub use self::scatter_extra::{
    diagonal_scatter, index_copy, narrow_copy, select_scatter, slice_scatter,
};

pub mod shape_extra;

pub use self::shape_extra::{
    atleast_1d, atleast_2d, atleast_3d, block_diag, cartesian_prod, column_stack, combinations,
    constant_pad_nd, dstack, hstack, moveaxis, movedim, pad, reflection_pad1d, reflection_pad2d,
    replication_pad1d, replication_pad2d, row_stack, swapaxes, swapdims, vstack, zero_pad2d,
};

pub mod nn3d_extra;

pub use self::nn3d_extra::{
    adaptive_avg_pool3d, adaptive_max_pool3d, avg_pool3d, conv3d, conv_transpose3d,
    fractional_max_pool2d, fractional_max_pool3d, lp_pool1d, lp_pool2d, max_pool3d, max_unpool1d,
    max_unpool2d, max_unpool3d,
};

pub mod random_extra;

pub use self::random_extra::{
    empty, full_like, ones_like, rand, randint, randn, randperm, zeros_like,
};

pub mod rnn_extra;

pub use self::rnn_extra::{
    gru_cell, lstm_cell, multi_head_attention_forward, rnn_relu_cell, rnn_tanh_cell,
    transformer_encoder_layer_fwd,
};

pub mod linalg_solve;

pub use self::linalg_solve::{
    linalg_cond, linalg_det, linalg_inv, linalg_pinv, linalg_slogdet, linalg_solve, lu_solve,
    lu_unpack,
};

pub(crate) use self::float_special::bessel_i0_f64;
