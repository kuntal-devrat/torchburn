//! Dispatch arms: Extra activations, losses, windows. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "celu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(kernels::linalg::celu(&a, alpha)?));
        }
        "hardshrink" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lambd = kw_f64(node, "lambd", 0.5);
            slots.push(Slot::Owned(kernels::linalg::hardshrink(&a, lambd)?));
        }
        "softshrink" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lambd = kw_f64(node, "lambd", 0.5);
            slots.push(Slot::Owned(kernels::linalg::softshrink(&a, lambd)?));
        }
        "tanhshrink" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::tanhshrink(&a)?));
        }
        "threshold" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let th = kw_f64(node, "threshold", 0.0);
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(kernels::linalg::threshold(&a, th, val)?));
        }
        "logsigmoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::linalg::logsigmoid(&a)?));
        }
        "rrelu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lower = kw_f64(node, "lower", 1.0 / 8.0);
            let upper = kw_f64(node, "upper", 1.0 / 3.0);
            slots.push(Slot::Owned(kernels::linalg::rrelu(&a, lower, upper)?));
        }
        "kl_div" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let log_target = kw_bool(node, "log_target", false);
            slots.push(Slot::Owned(kernels::linalg::kl_div(&a, &b, log_target)?));
        }
        "poisson_nll_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let log_input = kw_bool(node, "log_input", true);
            let full = kw_bool(node, "full", false);
            let eps = kw_f64(node, "eps", 1e-8);
            slots.push(Slot::Owned(kernels::linalg::poisson_nll_loss(
                &a, &b, log_input, full, eps,
            )?));
        }
        "margin_ranking_loss" => {
            let x1 = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x2 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let margin = kw_f64(node, "margin", 0.0);
            slots.push(Slot::Owned(kernels::linalg::margin_ranking_loss(
                &x1, &x2, &t, margin,
            )?));
        }
        "hinge_embedding_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let margin = kw_f64(node, "margin", 1.0);
            slots.push(Slot::Owned(kernels::linalg::hinge_embedding_loss(
                &a, &b, margin,
            )?));
        }
        "multilabel_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::multilabel_margin_loss(
                &a, &b,
            )?));
        }
        "soft_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::soft_margin_loss(&a, &b)?));
        }
        "multilabel_soft_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::multilabel_soft_margin_loss(
                &a, &b,
            )?));
        }
        "cosine_embedding_loss" => {
            let x1 = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x2 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let margin = kw_f64(node, "margin", 0.0);
            slots.push(Slot::Owned(kernels::linalg::cosine_embedding_loss(
                &x1, &x2, &t, margin,
            )?));
        }
        "triplet_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pos = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let neg = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let margin = kw_f64(node, "margin", 1.0);
            slots.push(Slot::Owned(kernels::linalg::triplet_margin_loss(
                &a, &pos, &neg, margin,
            )?));
        }
        "ctc_loss" => {
            let log_p = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let tgt = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::linalg::ctc_loss(&log_p, &tgt)?));
        }
        "hamming_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(kernels::linalg::hamming_window(n, periodic)?));
        }
        "kaiser_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            let beta = kw_f64(node, "beta", 12.0);
            slots.push(Slot::Owned(kernels::linalg::kaiser_window(
                n, beta, periodic,
            )?));
        }
        "gaussian_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            let std = kw_f64(node, "std", 1.0);
            slots.push(Slot::Owned(kernels::linalg::gaussian_window(
                n, std, periodic,
            )?));
        }
        "exponential_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            let tau = kw_f64(node, "tau", 1.0);
            slots.push(Slot::Owned(kernels::linalg::exponential_window(
                n, tau, periodic,
            )?));
        }
        "triangular_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(kernels::linalg::triangular_window(
                n, periodic,
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
