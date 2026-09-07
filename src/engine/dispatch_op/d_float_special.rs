//! Dispatch arms: Float special functions. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // ── Batch 3 operations (149 ops) ──
        "nextafter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::nextafter(&a, &b)?));
        }
        "heaviside" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::heaviside(&a, &b)?));
        }
        "nan_to_num" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let nan = kw_f64(node, "nan", 0.0);
            let posinf = kw_f64(node, "posinf", f64::MAX);
            let neginf = kw_f64(node, "neginf", f64::MIN);
            slots.push(Slot::Owned(kernels::linalg::nan_to_num(
                &a,
                nan,
                Some(posinf),
                Some(neginf),
            )?));
        }
        "logaddexp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::logaddexp(&a, &b)?));
        }
        "logaddexp2" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::logaddexp2(&a, &b)?));
        }
        "sinc" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::sinc(&a)?));
        }
        "i0" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::i0(&a)?));
        }
        "i1" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::i1(&a)?));
        }
        "i0e" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::i0e(&a)?));
        }
        "i1e" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::i1e(&a)?));
        }
        "bessel_j0" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::bessel_j0(&a)?));
        }
        "bessel_j1" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::bessel_j1(&a)?));
        }
        "bessel_y0" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::bessel_y0(&a)?));
        }
        "bessel_y1" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::bessel_y1(&a)?));
        }
        "digamma" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::digamma(&a)?));
        }
        "lgamma" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::lgamma(&a)?));
        }
        "polygamma" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = kw_i64(node, "n", 1);
            slots.push(Slot::Owned(kernels::linalg::polygamma(n, &a)?));
        }
        "mvlgamma" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_i64(node, "p", 1);
            slots.push(Slot::Owned(kernels::linalg::mvlgamma(&a, p)?));
        }
        "erfinv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::erfinv(&a)?));
        }
        "erfcinv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::erfcinv(&a)?));
        }
        "ndtri" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::ndtri(&a)?));
        }
        "ndtr" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::ndtr(&a)?));
        }
        "log_ndtr" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::log_ndtr(&a)?));
        }
        "logit" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let eps = kw_f64(node, "eps", -1.0);
            slots.push(Slot::Owned(kernels::linalg::logit(
                &a,
                if eps < 0.0 { None } else { Some(eps) },
            )?));
        }
        "expit" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::expit(&a)?));
        }
        "rad2deg" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::rad2deg(&a)?));
        }
        "deg2rad" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::deg2rad(&a)?));
        }
        "gcd" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::gcd(&a, &b)?));
        }
        "lcm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::lcm(&a, &b)?));
        }
        "fmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::fmax(&a, &b)?));
        }
        "fmin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::fmin(&a, &b)?));
        }
        "maximum" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::maximum(&a, &b)?));
        }
        "minimum" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::minimum(&a, &b)?));
        }
        "signbit" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::signbit(&a)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
