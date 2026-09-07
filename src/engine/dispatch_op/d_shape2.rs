//! Dispatch arms: Stacks, moves, pads. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "movedim" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = kw_isize_vec(node, "source");
            let dst = kw_isize_vec(node, "destination");
            slots.push(Slot::Owned(kernels::linalg::movedim(&a, &src, &dst)?));
        }
        "moveaxis" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = kw_isize_vec(node, "source");
            let dst = kw_isize_vec(node, "destination");
            slots.push(Slot::Owned(kernels::linalg::moveaxis(&a, &src, &dst)?));
        }
        "swapdims" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d0 = kw_isize(node, "dim0", 0);
            let d1 = kw_isize(node, "dim1", 1);
            slots.push(Slot::Owned(kernels::linalg::swapdims(&a, d0, d1)?));
        }
        "swapaxes" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d0 = kw_isize(node, "axis0", kw_isize(node, "dim0", 0));
            let d1 = kw_isize(node, "axis1", kw_isize(node, "dim1", 1));
            slots.push(Slot::Owned(kernels::linalg::swapaxes(&a, d0, d1)?));
        }
        "column_stack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::column_stack(&tens)?));
        }
        "row_stack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::row_stack(&tens)?));
        }
        "dstack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::dstack(&tens)?));
        }
        "hstack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::hstack(&tens)?));
        }
        "vstack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::vstack(&tens)?));
        }
        "atleast_1d" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::atleast_1d(&tens)?));
        }
        "atleast_2d" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::atleast_2d(&tens)?));
        }
        "atleast_3d" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::atleast_3d(&tens)?));
        }
        "block_diag" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::block_diag(&tens)?));
        }
        "cartesian_prod" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(kernels::linalg::cartesian_prod(&tens)?));
        }
        "combinations" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let r = kw_usize(node, "r", 2);
            slots.push(Slot::Owned(kernels::linalg::combinations(&a, r)?));
        }
        "pad" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            let mode = kw_str(node, "mode", "constant");
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(kernels::linalg::pad(&a, &pad, mode, val)?));
        }
        "constant_pad_nd" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(kernels::linalg::constant_pad_nd(
                &a, &pad, val,
            )?));
        }
        "reflection_pad1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(kernels::linalg::reflection_pad1d(&a, &pad)?));
        }
        "reflection_pad2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(kernels::linalg::reflection_pad2d(&a, &pad)?));
        }
        "replication_pad1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(kernels::linalg::replication_pad1d(&a, &pad)?));
        }
        "replication_pad2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(kernels::linalg::replication_pad2d(&a, &pad)?));
        }
        "zero_pad2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(kernels::linalg::zero_pad2d(&a, &pad)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
