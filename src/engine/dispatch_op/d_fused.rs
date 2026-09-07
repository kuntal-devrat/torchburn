//! Dispatch arms: Fused entry points. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Advanced LLM & FlashAttention
        "flash_attention" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let mask = if let Ok(idx) = arg_index(node, 3) {
                Some(slot_view(slots, capsules, idx)?)
            } else {
                None
            };
            let is_causal = kw_bool(node, "is_causal", false);
            let scale = node.kwargs.get("scale").and_then(|v| v.as_f64());
            slots.push(Slot::Owned(attention::flash_attention_forward(
                &q,
                &k,
                &v,
                mask.as_ref(),
                is_causal,
                scale,
            )?));
        }
        "fused_swiglu" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let gate_w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let up_w = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(attention::fused_swiglu(&x, &gate_w, &up_w)?));
        }
        "fused_geglu" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let gate_w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let up_w = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(attention::fused_geglu(&x, &gate_w, &up_w)?));
        }
        "fused_rmsnorm_residual" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let residual = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(attention::fused_rmsnorm_residual(
                &x, &residual, &weight, eps,
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
