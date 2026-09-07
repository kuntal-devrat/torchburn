//! Dispatch arms: Random and like-creation. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "rand" => {
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(kernels::linalg::rand(&sz)?));
        }
        "randn" => {
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(kernels::linalg::randn(&sz)?));
        }
        "randint" => {
            let low = kw_i64(node, "low", 0);
            let high = kw_i64(node, "high", 100);
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(kernels::linalg::randint(low, high, &sz)?));
        }
        "randperm" => {
            let n = kw_i64(node, "n", 10);
            slots.push(Slot::Owned(kernels::linalg::randperm(n)?));
        }
        "empty" => {
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(kernels::linalg::empty(&sz, DType::F32)?));
        }
        "zeros_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::zeros_like(&a)?));
        }
        "ones_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::ones_like(&a)?));
        }
        "full_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let val = kw_f64(node, "fill_value", 0.0);
            slots.push(Slot::Owned(kernels::linalg::full_like(&a, val)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
