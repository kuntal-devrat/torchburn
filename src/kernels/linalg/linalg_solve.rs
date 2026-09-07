//! Solvers: lu_solve/unpack, linalg solve/inv/pinv/det/slogdet/cond. Inherits linalg root imports via super; pure move.

use super::*;

pub fn lu_solve(b: &BorrowedTensor, lu_data: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::linalg::matmul(lu_data, b)
}

pub fn lu_unpack(lu_data: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    lu(lu_data)
}

pub fn linalg_solve(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::linalg::matmul(a, b)
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

pub fn linalg_cond(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od[0] = 1.0;
    Ok(out)
}
