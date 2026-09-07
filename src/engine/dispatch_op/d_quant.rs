//! Dispatch arms: Quantize/dequantize and low-bit linear. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Universal Low-Bit Quantization & GEMM
        "quantize_per_tensor" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scale = kw_f64(node, "scale", 1.0);
            let zero_point = node
                .kwargs
                .get("zero_point")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let dtype = if let Some(s) = node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                dtype_from_spec(s).unwrap_or(DType::I32)
            } else {
                DType::I32
            };
            slots.push(Slot::Owned(quantization::quantize_per_tensor(
                &x, scale, zero_point, dtype,
            )?));
        }
        "dequantize_per_tensor" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scale = kw_f64(node, "scale", 1.0);
            let zero_point = node
                .kwargs
                .get("zero_point")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            slots.push(Slot::Owned(quantization::dequantize_per_tensor(
                &q, scale, zero_point,
            )?));
        }
        "quantize_per_channel" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let zero_points = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let axis = kw_isize(node, "axis", 0) as usize;
            slots.push(Slot::Owned(quantization::quantize_per_channel(
                &x,
                &scales,
                &zero_points,
                axis,
            )?));
        }
        "dequantize_per_channel" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let zero_points = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let axis = kw_isize(node, "axis", 0) as usize;
            slots.push(Slot::Owned(quantization::dequantize_per_channel(
                &q,
                &scales,
                &zero_points,
                axis,
            )?));
        }
        "int8_gemm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let scale_a = kw_f64(node, "scale_a", 1.0);
            let scale_b = kw_f64(node, "scale_b", 1.0);
            slots.push(Slot::Owned(quantization::int8_gemm(
                &a, &b, scale_a, scale_b,
            )?));
        }
        "nf4_dequantize" => {
            let packed = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let absmax = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let group_size = kw_isize(node, "group_size", 64) as usize;
            slots.push(Slot::Owned(quantization::nf4_dequantize(
                &packed, &absmax, group_size,
            )?));
        }
        "int4_unpack_dequantize" => {
            let packed = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let zeros = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let group_size = kw_isize(node, "group_size", 128) as usize;
            slots.push(Slot::Owned(quantization::int4_unpack_dequantize(
                &packed, &scales, &zeros, group_size,
            )?));
        }
        "w8a32_linear" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let bias = if node.args.len() > 3 {
                Some(slot_view(slots, capsules, arg_index(node, 3)?)?)
            } else {
                None
            };
            slots.push(Slot::Owned(quantization::w8a32_linear(
                &x,
                &w,
                &scales,
                bias.as_ref(),
            )?));
        }
        "w4a32_linear" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w_packed = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let bias = if node.args.len() > 3 {
                Some(slot_view(slots, capsules, arg_index(node, 3)?)?)
            } else {
                None
            };
            slots.push(Slot::Owned(quantization::w4a32_linear(
                &x,
                &w_packed,
                &scales,
                bias.as_ref(),
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
