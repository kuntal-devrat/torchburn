//! Dispatch arms: Binary/relu/logical/math/activations. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Phase 1: binary elementwise
        "add" | "sub" | "mul" | "div" => {
            let op =
                kernels::BinaryOp::from_target(target).expect("binary op target already validated");
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(kernels::binary(op, &a, &b)?));
        }
        "relu" => {
            let ai = arg_index(node, 0)?;
            let a = slot_view(slots, capsules, ai)?;
            slots.push(Slot::Owned(kernels::relu(&a)?));
        }

        // Phase 2: logical ops
        "logical_and" | "logical_or" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            let out = if target == "logical_and" {
                math_ops::logical_and(&a, &b)?
            } else {
                math_ops::logical_or(&a, &b)?
            };
            slots.push(Slot::Owned(out));
        }
        "logical_not" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::logical_not(&a)?));
        }
        "to_dtype" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            // dtype comes from the "dtype" kwarg (aten._to_copy) or from a
            // positional const arg (x.to(torch.float64) -> args[1].value).
            let dtype_str: Option<&str> = node.kwargs.get("dtype").and_then(|v| v.as_str());
            let dtype_str = dtype_str.or_else(|| {
                node.args
                    .get(1)
                    .and_then(|arg| arg.value.as_ref())
                    .and_then(|v| v.as_str())
            });
            let dtype_str = dtype_str.ok_or_else(|| unsupported("to_dtype: missing dtype"))?;
            let target = crate::dlpack::dtype_from_spec(dtype_str)
                .ok_or_else(|| unsupported(&format!("to_dtype: unknown dtype '{dtype_str}'")))?;
            slots.push(Slot::Owned(math_ops::to_dtype(&a, target)?));
        }

        // Phase 2: comparison ops
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(math_ops::comparison(target, &a, &b)?));
        }

        // Phase 2: unary math
        "abs" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::abs(&a)?));
        }
        "neg" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::neg(&a)?));
        }
        "sign" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::sign(&a)?));
        }
        "sqrt" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::sqrt(&a)?));
        }
        "rsqrt" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::rsqrt(&a)?));
        }
        "exp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::exp(&a)?));
        }
        "log" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::log(&a)?));
        }
        "reciprocal" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::reciprocal(&a)?));
        }
        "ceil" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::ceil(&a)?));
        }
        "floor" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::floor(&a)?));
        }
        "clamp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let min = kw_f64(node, "min", f64::NEG_INFINITY);
            let max = kw_f64(node, "max", f64::INFINITY);
            slots.push(Slot::Owned(math_ops::clamp(&a, min, max)?));
        }
        "clamp_min" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let min = kw_f64(node, "min", 0.0);
            slots.push(Slot::Owned(math_ops::clamp_min(&a, min)?));
        }
        "clamp_max" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let max = kw_f64(node, "max", 0.0);
            slots.push(Slot::Owned(math_ops::clamp_max(&a, max)?));
        }
        "sin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::sin(&a)?));
        }
        "cos" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::cos(&a)?));
        }
        "round" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::round(&a)?));
        }
        "pow" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let exp = kw_f64(node, "exp", 2.0);
            slots.push(Slot::Owned(math_ops::pow_scalar(&a, exp)?));
        }

        // Phase 2: activations
        "sigmoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(activations::sigmoid(&a)?));
        }
        "tanh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(activations::tanh_act(&a)?));
        }
        "gelu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let approx = kw_str(node, "approximate", "tanh");
            slots.push(Slot::Owned(activations::gelu(&a, approx)?));
        }
        "silu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(activations::silu(&a)?));
        }
        "leaky_relu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ns = kw_f64(node, "negative_slope", 0.01);
            slots.push(Slot::Owned(activations::leaky_relu(&a, ns)?));
        }
        "elu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(activations::elu(&a, alpha)?));
        }
        "selu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(activations::selu(&a)?));
        }
        "softplus" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(activations::softplus(&a)?));
        }
        "hardswish" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(activations::hardswish(&a)?));
        }
        "mish" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(activations::mish(&a)?));
        }
        "softmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(activations::softmax(&a, dim)?));
        }
        "log_softmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(activations::log_softmax(&a, dim)?));
        }
        "threshold_backward" => {
            let grad = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let threshold = kw_f64(node, "threshold", 0.0);
            slots.push(Slot::Owned(activations::threshold_backward(
                &grad, &x, threshold,
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
