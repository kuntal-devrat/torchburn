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
            let weight = if node.args.len() > 1 {
                arg_index(node, 1)
                    .ok()
                    .and_then(|idx| slot_view(slots, capsules, idx).ok())
            } else {
                None
            };
            let bias = if node.args.len() > 2 {
                arg_index(node, 2)
                    .ok()
                    .and_then(|idx| slot_view(slots, capsules, idx).ok())
            } else {
                None
            };
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(norm::layer_norm(
                &input,
                weight.as_ref(),
                bias.as_ref(),
                eps,
            )?));
        }
        "batch_norm" => {
            // torch signature: batch_norm(x, running_mean, running_var, weight, bias, ...)
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let running_mean = if node.args.len() > 1 {
                arg_index(node, 1)
                    .ok()
                    .and_then(|idx| slot_view(slots, capsules, idx).ok())
            } else {
                None
            };
            let running_var = if node.args.len() > 2 {
                arg_index(node, 2)
                    .ok()
                    .and_then(|idx| slot_view(slots, capsules, idx).ok())
            } else {
                None
            };
            let weight = if node.args.len() > 3 {
                arg_index(node, 3)
                    .ok()
                    .and_then(|idx| slot_view(slots, capsules, idx).ok())
            } else {
                None
            };
            let bias = if node.args.len() > 4 {
                arg_index(node, 4)
                    .ok()
                    .and_then(|idx| slot_view(slots, capsules, idx).ok())
            } else {
                None
            };
            let eps = kw_f64(node, "eps", 1e-5);
            let training = kw_bool(node, "training", false);
            slots.push(Slot::Owned(norm::batch_norm(
                &input,
                weight.as_ref(),
                bias.as_ref(),
                running_mean.as_ref(),
                running_var.as_ref(),
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
