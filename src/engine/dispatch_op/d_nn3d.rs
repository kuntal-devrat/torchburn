//! Dispatch arms: 3D conv/pool/unpool. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "conv3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b = if node.args.len() > 2 {
                slot_view(slots, capsules, arg_index(node, 2)?).ok()
            } else {
                None
            };
            slots.push(Slot::Owned(kernels::linalg::conv3d(&a, &w, b.as_ref())?));
        }
        "conv_transpose3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b = if node.args.len() > 2 {
                slot_view(slots, capsules, arg_index(node, 2)?).ok()
            } else {
                None
            };
            slots.push(Slot::Owned(kernels::linalg::conv_transpose3d(
                &a,
                &w,
                b.as_ref(),
            )?));
        }
        "max_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_i64_vec(node, "kernel_size");
            let s = kw_i64_vec(node, "stride");
            slots.push(Slot::Owned(kernels::linalg::max_pool3d(&a, &k, &s)?));
        }
        "avg_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_i64_vec(node, "kernel_size");
            let s = kw_i64_vec(node, "stride");
            slots.push(Slot::Owned(kernels::linalg::avg_pool3d(&a, &k, &s)?));
        }
        "adaptive_max_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::linalg::adaptive_max_pool3d(
                &a, &out_sz,
            )?));
        }
        "adaptive_avg_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::linalg::adaptive_avg_pool3d(
                &a, &out_sz,
            )?));
        }
        "fractional_max_pool2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::linalg::fractional_max_pool2d(
                &a, &out_sz,
            )?));
        }
        "fractional_max_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::linalg::fractional_max_pool3d(
                &a, &out_sz,
            )?));
        }
        "lp_pool1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let norm = kw_f64(node, "norm_type", 2.0);
            slots.push(Slot::Owned(kernels::linalg::lp_pool1d(&a, norm)?));
        }
        "lp_pool2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let norm = kw_f64(node, "norm_type", 2.0);
            slots.push(Slot::Owned(kernels::linalg::lp_pool2d(&a, norm)?));
        }
        "max_unpool1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::linalg::max_unpool1d(&a, &out_sz)?));
        }
        "max_unpool2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::linalg::max_unpool2d(&a, &out_sz)?));
        }
        "max_unpool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::linalg::max_unpool3d(&a, &out_sz)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
