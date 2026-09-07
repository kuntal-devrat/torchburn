//! Dispatch arms: Decompositions and norms. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "cross" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(kernels::linalg::cross(&a, &b, dim)?));
        }
        "linalg_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ord = kw_f64(node, "ord", 2.0);
            slots.push(Slot::Owned(kernels::linalg::linalg_norm(&a, Some(ord))?));
        }
        "frobenius_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::frobenius_norm(&a)?));
        }
        "nuclear_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::nuclear_norm(&a)?));
        }
        "matrix_rank" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::matrix_rank(&a)?));
        }
        "matrix_power" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = kw_i64(node, "n", 1);
            slots.push(Slot::Owned(kernels::linalg::matrix_power(&a, n)?));
        }
        "cholesky" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::cholesky(&a)?));
        }
        "cholesky_inverse" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::cholesky_inverse(&a)?));
        }
        "cholesky_solve" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::cholesky_solve(&a, &b)?));
        }
        "qr" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (q, r) = kernels::linalg::qr(&a)?;
            slots.push(Slot::Tuple(vec![q, r]));
        }
        "svd" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (u, s, v) = kernels::linalg::svd(&a)?;
            slots.push(Slot::Tuple(vec![u, s, v]));
        }
        "svdvals" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::svdvals(&a)?));
        }
        "eig" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (vals, vecs) = kernels::linalg::eig(&a)?;
            slots.push(Slot::Tuple(vec![vals, vecs]));
        }
        "eigh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (vals, vecs) = kernels::linalg::eigh(&a)?;
            slots.push(Slot::Tuple(vec![vals, vecs]));
        }
        "eigvals" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::eigvals(&a)?));
        }
        "eigvalsh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::eigvalsh(&a)?));
        }
        "lu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (p, l, u) = kernels::linalg::lu(&a)?;
            slots.push(Slot::Tuple(vec![p, l, u]));
        }
        "triangular_solve" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let sol = kernels::linalg::triangular_solve(&a, &b)?;
            slots.push(Slot::Owned(sol));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
