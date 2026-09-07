//! Dispatch arms: Reductions and matmul-family. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Phase 2: reductions
        "sum" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dims = kw_opt_dims(node);
            let keepdim = kw_bool(node, "keepdim", false);
            if dims.len() > 1 {
                slots.push(Slot::Owned(reductions::sum_dims(&a, &dims, keepdim)?));
            } else {
                slots.push(Slot::Owned(reductions::sum(
                    &a,
                    dims.first().copied(),
                    keepdim,
                )?));
            }
        }
        "mean" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dims = kw_opt_dims(node);
            let keepdim = kw_bool(node, "keepdim", false);
            if dims.len() > 1 {
                slots.push(Slot::Owned(reductions::mean_dims(&a, &dims, keepdim)?));
            } else {
                slots.push(Slot::Owned(reductions::mean(
                    &a,
                    dims.first().copied(),
                    keepdim,
                )?));
            }
        }
        "max" | "max_reduce" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let (val, idx) = reductions::max_reduce(&a, dim, keepdim)?;
            slots.push(Slot::Tuple(vec![val, idx]));
        }
        "min" | "min_reduce" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let (val, idx) = reductions::min_reduce(&a, dim, keepdim)?;
            slots.push(Slot::Tuple(vec![val, idx]));
        }
        "argmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::argmax(&a, dim, keepdim)?));
        }
        "argmin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::argmin(&a, dim, keepdim)?));
        }
        "std" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let unbiased = kw_bool(node, "unbiased", true);
            slots.push(Slot::Owned(reductions::std_dev(
                &a, dim, keepdim, unbiased,
            )?));
        }
        "var" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let unbiased = kw_bool(node, "unbiased", true);
            slots.push(Slot::Owned(reductions::var(&a, dim, keepdim, unbiased)?));
        }
        "cumsum" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(reductions::cumsum(&a, dim)?));
        }
        "prod" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::prod(&a, dim, keepdim)?));
        }
        "norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64_allow_inf(node, "p", 2.0);
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::p_norm(&a, p, dim, keepdim)?));
        }
        "linalg_vector_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64_allow_inf(node, "ord", 2.0);
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::p_norm(&a, p, dim, keepdim)?));
        }

        // Phase 2: linalg
        "matmul" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(linalg::matmul(&a, &b)?));
        }
        "bmm" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(linalg::bmm(&a, &b)?));
        }
        "linear" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = if node.args.len() > 2 {
                Some(slot_view(slots, capsules, arg_index(node, 2)?)?)
            } else {
                None
            };
            slots.push(Slot::Owned(linalg::linear(
                &input,
                &weight,
                bias.as_ref(),
                None,
            )?));
        }
        "dot" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(linalg::dot(&a, &b)?));
        }
        "addmm" => {
            // aten.addmm(bias, mat1, mat2) — mat2 is NOT transposed.
            let bias = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mat1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let mat2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(linalg::addmm(&bias, &mat1, &mat2, None)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
