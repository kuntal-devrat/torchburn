//! Solvers: lu_solve/unpack, linalg solve/inv/pinv/det/slogdet/cond. Inherits linalg root imports via super; pure move.

use super::*;

pub fn lu_solve(_b: &BorrowedTensor, _lu_data: &BorrowedTensor) -> PyResult<OwnedTensor> {
    Err(crate::dlpack::unsupported(
        "lu_solve requires eager fallback",
    ))
}

pub fn lu_unpack(lu_data: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    lu(lu_data)
}

pub fn linalg_solve(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.shape.len() == 2 && a.shape[0] == a.shape[1] {
        let inv = crate::kernels::reductions::pinverse(a)?;
        crate::linalg::matmul(&BorrowedTensor::from_owned(&inv), b)
    } else {
        Err(crate::dlpack::unsupported(
            "linalg_solve: general batched or non-square solve requires eager fallback",
        ))
    }
}

pub fn linalg_inv(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::kernels::reductions::pinverse(a)
}

pub fn linalg_pinv(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::kernels::reductions::pinverse(a)
}

pub fn linalg_det(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::kernels::reductions::det(a)
}

pub fn linalg_slogdet(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    crate::kernels::reductions::slogdet(a)
}

pub fn linalg_cond(_a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    Err(crate::dlpack::unsupported(
        "linalg_cond requires eager fallback",
    ))
}
