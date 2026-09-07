//! Dispatch arms: Solvers. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "lu_solve" => {
            let b = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lu_d = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::lu_solve(&b, &lu_d)?));
        }
        "lu_unpack" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (p, l, u) = kernels::linalg::lu_unpack(&a)?;
            slots.push(Slot::Tuple(vec![p, l, u]));
        }
        "linalg_solve" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::linalg_solve(&a, &b)?));
        }
        "linalg_inv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::linalg_inv(&a)?));
        }
        "linalg_pinv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::linalg_pinv(&a)?));
        }
        "linalg_det" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::linalg_det(&a)?));
        }
        "linalg_slogdet" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (s, l) = kernels::linalg::linalg_slogdet(&a)?;
            slots.push(Slot::Tuple(vec![s, l]));
        }
        "linalg_cond" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::linalg_cond(&a)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
