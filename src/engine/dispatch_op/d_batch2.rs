//! Dispatch arms: Extended batch 2. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // ── Batch 2 operations (49 ops) ──
        "embedding_bag" => {
            let w = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let mode = kw_str(node, "mode", "mean");
            slots.push(Slot::Owned(kernels::reductions::embedding_bag(
                &w, &idx, mode,
            )?));
        }
        "unfold" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dimension", kw_isize(node, "dim", 0));
            let size = kw_i64(node, "size", 1);
            let step = kw_i64(node, "step", 1);
            slots.push(Slot::Owned(kernels::reductions::unfold(
                &a, dim, size, step,
            )?));
        }
        "fold" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(kernels::reductions::fold(&a, &out_sz)?));
        }
        "grid_sample" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let g = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::reductions::grid_sample(&a, &g)?));
        }
        "affine_grid" => {
            let theta = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(kernels::reductions::affine_grid(&theta, &sz)?));
        }
        "pixel_unshuffle" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let r = kw_i64(node, "downscale_factor", kw_i64(node, "downscale", 2));
            slots.push(Slot::Owned(kernels::reductions::pixel_unshuffle(&a, r)?));
        }
        "channel_shuffle" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let g = kw_i64(node, "groups", 2);
            slots.push(Slot::Owned(kernels::reductions::channel_shuffle(&a, g)?));
        }
        "cummax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(kernels::reductions::cummax(&a, dim)?));
        }
        "cummin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(kernels::reductions::cummin(&a, dim)?));
        }
        "logcumsumexp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(kernels::reductions::logcumsumexp(&a, dim)?));
        }
        "scatter_reduce" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let reduce = kw_str(node, "reduce", "sum");
            slots.push(Slot::Owned(kernels::reductions::scatter_reduce(
                &a, dim, &idx, &src, reduce,
            )?));
        }
        "index_put" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let val = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::reductions::index_put(&a, &idx, &val)?));
        }
        "index_add" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::reductions::index_add(
                &a, dim, &idx, &src,
            )?));
        }
        "masked_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let m = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::reductions::masked_scatter(
                &a, &m, &src,
            )?));
        }
        "take" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::reductions::take(&a, &idx)?));
        }
        "put" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let acc = kw_bool(node, "accumulate", false);
            slots.push(Slot::Owned(kernels::reductions::put(&a, &idx, &src, acc)?));
        }
        "masked_select" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let m = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::reductions::masked_select(&a, &m)?));
        }
        "index_fill" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = kw_i64(node, "index", 0);
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(kernels::reductions::index_fill(
                &a, dim, idx, val,
            )?));
        }
        "bincount" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = if node.args.len() > 1 {
                slot_view(slots, capsules, arg_index(node, 1)?).ok()
            } else {
                None
            };
            slots.push(Slot::Owned(kernels::reductions::bincount(&a, w.as_ref())?));
        }
        "unique" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::unique(&a)?));
        }
        "kthvalue" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_usize(node, "k", 1);
            slots.push(Slot::Owned(kernels::reductions::kthvalue(&a, k)?));
        }
        "median" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::median(&a)?));
        }
        "quantile" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let q = kw_f64(node, "q", 0.5);
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(kernels::reductions::quantile(
                &a, q, dim, keepdim,
            )?));
        }
        "histogram" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let bins = kw_usize(node, "bins", 100);
            slots.push(Slot::Owned(kernels::reductions::histogram(&a, bins)?));
        }
        "searchsorted" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::reductions::searchsorted(&a, &v)?));
        }
        "meshgrid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let (o1, o2) = kernels::reductions::meshgrid(&a, &b)?;
            slots.push(Slot::Tuple(vec![o1, o2]));
        }
        "cdist" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::reductions::cdist(&a, &b)?));
        }
        "pdist" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::pdist(&a)?));
        }
        "renorm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64(node, "p", 2.0);
            let dim = kw_isize(node, "dim", 0);
            let maxnorm = kw_f64(node, "maxnorm", 1.0);
            slots.push(Slot::Owned(kernels::reductions::renorm(
                &a, p, dim, maxnorm,
            )?));
        }
        "bernoulli" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64(node, "p", 0.5);
            slots.push(Slot::Owned(kernels::reductions::bernoulli(&a, p)?));
        }
        "multinomial" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let num_samples = kw_usize(node, "num_samples", 1);
            slots.push(Slot::Owned(kernels::reductions::multinomial(
                &a,
                num_samples,
            )?));
        }
        "logspace" => {
            let start = kw_f64(node, "start", 0.0);
            let end = kw_f64(node, "end", 1.0);
            let steps = kw_usize(node, "steps", 100);
            slots.push(Slot::Owned(kernels::reductions::logspace(
                start, end, steps,
            )?));
        }
        "eye" => {
            let n = kw_i64(
                node,
                "n",
                kw_i64_vec(node, "shape").first().copied().unwrap_or(3),
            );
            slots.push(Slot::Owned(kernels::reductions::eye(n)?));
        }
        "diag" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::diag(&a)?));
        }
        "diagonal" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let offset = kw_i64(node, "offset", 0);
            let dim1 = kw_isize(node, "dim1", 0);
            let dim2 = kw_isize(node, "dim2", 1);
            slots.push(Slot::Owned(kernels::reductions::diagonal(
                &a, offset, dim1, dim2,
            )?));
        }
        "trace" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::trace(&a)?));
        }
        "matrix_exp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::matrix_exp(&a)?));
        }
        "slogdet" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (s, l) = kernels::reductions::slogdet(&a)?;
            slots.push(Slot::Tuple(vec![s, l]));
        }
        "det" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::det(&a)?));
        }
        "lstsq" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::reductions::lstsq(&a, &b)?));
        }
        "pinverse" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::reductions::pinverse(&a)?));
        }
        "normal" => {
            let mean = kw_f64(node, "mean", 0.0);
            let std = kw_f64(node, "std", 1.0);
            let size = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(kernels::reductions::normal(mean, std, &size)?));
        }
        "uniform" => {
            let from = kw_f64(node, "from", 0.0);
            let to = kw_f64(node, "to", 1.0);
            let size = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(kernels::reductions::uniform(from, to, &size)?));
        }
        "triu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d = kw_i64(node, "diagonal", 0);
            slots.push(Slot::Owned(kernels::reductions::triu(&a, d)?));
        }
        "tril" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d = kw_i64(node, "diagonal", 0);
            slots.push(Slot::Owned(kernels::reductions::tril(&a, d)?));
        }
        "hann_window" => {
            let win_len = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(kernels::reductions::hann_window(
                win_len, periodic,
            )?));
        }
        "bartlett_window" => {
            let win_len = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(kernels::reductions::bartlett_window(
                win_len, periodic,
            )?));
        }
        "blackman_window" => {
            let win_len = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(kernels::reductions::blackman_window(
                win_len, periodic,
            )?));
        }
        "stft" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n_fft = kw_usize(node, "n_fft", 256);
            let hop = kw_usize(node, "hop_length", n_fft / 4);
            let win = kw_usize(node, "win_length", n_fft);
            slots.push(Slot::Owned(kernels::reductions::stft(&a, n_fft, hop, win)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
