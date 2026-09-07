//! Dispatch arms: Normalization layers. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Phase 2: norm
        "layer_norm" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(norm::layer_norm(&input, &weight, &bias, eps)?));
        }
        "batch_norm" => {
            // torch signature: batch_norm(x, running_mean, running_var, weight, bias, ...)
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let running_mean = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let running_var = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let bias = slot_view(slots, capsules, arg_index(node, 4)?)?;
            let eps = kw_f64(node, "eps", 1e-5);
            let training = kw_bool(node, "training", false);
            slots.push(Slot::Owned(norm::batch_norm(
                &input,
                &weight,
                &bias,
                &running_mean,
                &running_var,
                eps,
                training,
            )?));
        }
        "group_norm" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let num_groups = kw_usize(node, "num_groups", 32);
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(norm::group_norm(
                &input, &weight, &bias, num_groups, eps,
            )?));
        }
        "rms_norm" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let eps = kw_f64(node, "eps", 1e-6);
            slots.push(Slot::Owned(norm::rms_norm(&input, &weight, eps)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
