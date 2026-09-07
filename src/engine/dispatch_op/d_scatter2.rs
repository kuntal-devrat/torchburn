//! Dispatch arms: Scatter variants. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "select_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = kw_i64(node, "index", 0);
            slots.push(Slot::Owned(kernels::linalg::select_scatter(
                &a, &src, dim, idx,
            )?));
        }
        "slice_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            let start = kw_isize(node, "start", 0);
            let end = kw_isize(node, "end", 0);
            let step = kw_isize(node, "step", 1);
            slots.push(Slot::Owned(kernels::linalg::slice_scatter(
                &a,
                &src,
                dim,
                Some(start as i64),
                Some(end as i64),
                step as i64,
            )?));
        }
        "diagonal_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let offset = kw_i64(node, "offset", 0);
            slots.push(Slot::Owned(kernels::linalg::diagonal_scatter(
                &a, &src, offset,
            )?));
        }
        "index_copy" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::linalg::index_copy(
                &a, dim, &idx, &src,
            )?));
        }
        "narrow_copy" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let start = kw_i64(node, "start", 0);
            let len = kw_i64(node, "length", 1);
            slots.push(Slot::Owned(kernels::linalg::narrow_copy(
                &a,
                dim,
                start as usize,
                len as usize,
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
