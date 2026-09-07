//! Dispatch arms: BLAS-like extensions and trapezoid. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "addcdiv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let t1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let val = kw_f64(node, "value", 1.0);
            slots.push(Slot::Owned(kernels::linalg::addcdiv(&a, &t1, &t2, val)?));
        }
        "addcmul" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let t1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let val = kw_f64(node, "value", 1.0);
            slots.push(Slot::Owned(kernels::linalg::addcmul(&a, &t1, &t2, val)?));
        }
        "addr" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let v1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(kernels::linalg::addr(
                &a, &v1, &v2, beta, alpha,
            )?));
        }
        "outer" | "ger" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::outer(&a, &b)?));
        }
        "mv" => {
            let mat = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let vec = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::mv(&mat, &vec)?));
        }
        "vdot" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::vdot(&a, &b)?));
        }
        "baddbmm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(kernels::linalg::baddbmm(
                &a, &b1, &b2, beta, alpha,
            )?));
        }
        "addbmm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(kernels::linalg::addbmm(
                &a, &b1, &b2, beta, alpha,
            )?));
        }
        "addmv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mat = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let vec = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(kernels::linalg::addmv(
                &a, &mat, &vec, beta, alpha,
            )?));
        }
        "kron" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::kron(&a, &b)?));
        }
        "inner" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::inner(&a, &b)?));
        }
        "trapz" | "trapezoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dx = kw_f64(node, "dx", 1.0);
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(kernels::linalg::trapezoid(&a, None, dx, dim)?));
        }
        "cumulative_trapezoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dx = kw_f64(node, "dx", 1.0);
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(kernels::linalg::cumulative_trapezoid(
                &a, dx, dim,
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
