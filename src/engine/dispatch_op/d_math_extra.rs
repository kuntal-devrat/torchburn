//! Dispatch arms: Extended math batch 1. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // ── 50 extra super ops (SIMD + tiled) ──
        "atan" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::atan(&a)?));
        }
        "asin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::asin(&a)?));
        }
        "acos" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::acos(&a)?));
        }
        "sinh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::sinh(&a)?));
        }
        "cosh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::cosh(&a)?));
        }
        "asinh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::asinh(&a)?));
        }
        "acosh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::acosh(&a)?));
        }
        "atanh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::atanh(&a)?));
        }
        "erf" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::erf(&a)?));
        }
        "erfc" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::erfc(&a)?));
        }
        "expm1" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::expm1(&a)?));
        }
        "log1p" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::log1p(&a)?));
        }
        "log2" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::log2(&a)?));
        }
        "log10" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::log10(&a)?));
        }
        "trunc" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::trunc(&a)?));
        }
        "frac" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::frac(&a)?));
        }
        "square" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::square(&a)?));
        }
        "exp2" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::exp2(&a)?));
        }
        "atan2" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::atan2(&a, &b)?));
        }
        "hypot" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::hypot(&a, &b)?));
        }
        "fmod" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::fmod(&a, &b)?));
        }
        "remainder" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::remainder(&a, &b)?));
        }
        "copysign" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::copysign(&a, &b)?));
        }
        "ldexp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::ldexp(&a, &b)?));
        }
        "lerp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let w = kw_f64(node, "weight", kw_f64(node, "w", 0.5));
            slots.push(Slot::Owned(kernels::elementwise::lerp(&a, &b, w)?));
        }
        "bitwise_and" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::bitwise_and(&a, &b)?));
        }
        "bitwise_or" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::bitwise_or(&a, &b)?));
        }
        "bitwise_xor" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::bitwise_xor(&a, &b)?));
        }
        "bitwise_not" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::bitwise_not(&a)?));
        }
        "isfinite" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::isfinite(&a)?));
        }
        "isinf" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::isinf(&a)?));
        }
        "isnan" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::isnan(&a)?));
        }
        "all" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::all(&a)?));
        }
        "any" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::any(&a)?));
        }
        "amax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::amax(&a)?));
        }
        "amin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::amin(&a)?));
        }
        "count_nonzero" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::count_nonzero(&a)?));
        }
        "nansum" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::nansum(&a)?));
        }
        "nanmean" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::nanmean(&a)?));
        }
        "tile" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let repeats = kw_i64_vec(node, "repeats");
            let repeats = if repeats.is_empty() {
                kw_i64_vec(node, "dims")
            } else {
                repeats
            };
            slots.push(Slot::Owned(kernels::elementwise::tile(&a, &repeats)?));
        }
        "roll" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let shift = node
                .kwargs
                .get("shifts")
                .or_else(|| node.kwargs.get("shift"))
                .and_then(|v| {
                    v.as_i64().or_else(|| {
                        v.as_array()
                            .and_then(|arr| arr.first().and_then(|x| x.as_i64()))
                    })
                })
                .unwrap_or(1);
            let dim = node
                .kwargs
                .get("dims")
                .or_else(|| node.kwargs.get("dim"))
                .and_then(|v| {
                    v.as_i64().or_else(|| {
                        v.as_array()
                            .and_then(|arr| arr.first().and_then(|x| x.as_i64()))
                    })
                })
                .unwrap_or(0) as isize;
            slots.push(Slot::Owned(kernels::elementwise::roll(&a, shift, dim)?));
        }
        "pixel_shuffle" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let r = kw_isize(node, "upscale_factor", kw_isize(node, "upscale", 2)) as i64;
            slots.push(Slot::Owned(kernels::elementwise::pixel_shuffle(&a, r)?));
        }
        "instance_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(kernels::elementwise::instance_norm(&a, eps)?));
        }
        "cross_entropy" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::cross_entropy(&a, &b)?));
        }
        "huber_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let d = kw_f64(node, "delta", 1.0);
            slots.push(Slot::Owned(kernels::elementwise::huber_loss(&a, &b, d)?));
        }
        "hardtanh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lo = kw_f64(node, "min_val", kw_f64(node, "min", -1.0));
            let hi = kw_f64(node, "max_val", kw_f64(node, "max", 1.0));
            slots.push(Slot::Owned(kernels::elementwise::hardtanh(&a, lo, hi)?));
        }
        "hardsigmoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::elementwise::hardsigmoid(&a)?));
        }
        "glu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(kernels::elementwise::glu(&a, dim)?));
        }
        "bucketize" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::elementwise::bucketize(&a, &b)?));
        }
        "histc" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let bins = kw_usize(node, "bins", 100);
            let min = kw_f64(node, "min", 0.0);
            let max = kw_f64(node, "max", 0.0);
            slots.push(Slot::Owned(kernels::elementwise::histc(
                &a, bins, min, max,
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
