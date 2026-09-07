//! Dispatch arms: Batch 4 tail. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Extra ops batch 4 — 48 ops for 450 total
        "isclose" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let rtol = kw_f64(node, "rtol", 1e-05);
            let atol = kw_f64(node, "atol", 1e-08);
            let equal_nan = kw_bool(node, "equal_nan", false);
            slots.push(Slot::Owned(kernels::tensor_ops::isclose(
                &a, &b, rtol, atol, equal_nan,
            )?));
        }
        "allclose" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let rtol = kw_f64(node, "rtol", 1e-05);
            let atol = kw_f64(node, "atol", 1e-08);
            let equal_nan = kw_bool(node, "equal_nan", false);
            slots.push(Slot::Owned(kernels::tensor_ops::allclose(
                &a, &b, rtol, atol, equal_nan,
            )?));
        }
        "equal" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::equal(&a, &b)?));
        }
        "isreal" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::isreal(&a)?));
        }
        "is_complex" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::is_complex(&a)?));
        }
        "is_nonzero" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::is_nonzero(&a)?));
        }
        "nanprod" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(kernels::tensor_ops::nanprod(&a, dim, keepdim)?));
        }
        "nanmin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::nanmin(&a)?));
        }
        "nanmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::nanmax(&a)?));
        }
        "var_mean" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let unbiased = kw_bool(node, "unbiased", true);
            let (v, m) = kernels::tensor_ops::var_mean(&a, dim, keepdim, unbiased)?;
            slots.push(Slot::Tuple(vec![v, m]));
        }
        "std_mean" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let unbiased = kw_bool(node, "unbiased", true);
            let (s, m) = kernels::tensor_ops::std_mean(&a, dim, keepdim, unbiased)?;
            slots.push(Slot::Tuple(vec![s, m]));
        }
        "nanmedian" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::nanmedian(&a)?));
        }
        "cov" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let corr = kw_i64(node, "correction", 1);
            slots.push(Slot::Owned(kernels::tensor_ops::cov(&a, corr)?));
        }
        "corrcoef" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::corrcoef(&a)?));
        }
        "as_strided" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let size = kw_i64_vec(node, "size");
            let stride = kw_i64_vec(node, "stride");
            let offset = kw_usize(node, "storage_offset", 0);
            slots.push(Slot::Owned(kernels::tensor_ops::as_strided(
                &a, size, stride, offset,
            )?));
        }
        "broadcast_to" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let shape = kw_i64_vec(node, "shape");
            slots.push(Slot::Owned(kernels::tensor_ops::broadcast_to(&a, shape)?));
        }
        "broadcast_tensors" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let (ea, eb) = kernels::tensor_ops::broadcast_tensors(&a, &b)?;
            slots.push(Slot::Tuple(vec![ea, eb]));
        }
        "split" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let split_size = kw_usize(
                node,
                "split_size",
                kw_usize(node, "split_size_or_sections", 1),
            );
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Tuple(kernels::tensor_ops::split(
                &a, split_size, dim,
            )?));
        }
        "vsplit" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let sec = kw_usize(node, "sections", kw_usize(node, "split_size", 2));
            slots.push(Slot::Tuple(kernels::tensor_ops::vsplit(&a, sec)?));
        }
        "hsplit" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let sec = kw_usize(node, "sections", kw_usize(node, "split_size", 2));
            slots.push(Slot::Tuple(kernels::tensor_ops::hsplit(&a, sec)?));
        }
        "dsplit" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let sec = kw_usize(node, "sections", kw_usize(node, "split_size", 2));
            slots.push(Slot::Tuple(kernels::tensor_ops::dsplit(&a, sec)?));
        }
        "tensor_split" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let indices = kw_i64_vec(node, "indices");
            let idx_us: Vec<usize> = indices.iter().map(|&v| v as usize).collect();
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Tuple(kernels::tensor_ops::tensor_split(
                &a, idx_us, dim,
            )?));
        }
        "take_along_dim" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(kernels::tensor_ops::take_along_dim(
                &a, &idx, dim,
            )?));
        }
        "index_reduce" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let reduce = kw_str(node, "reduce", "sum");
            slots.push(Slot::Owned(kernels::tensor_ops::index_reduce(
                &a, dim, &idx, &src, reduce,
            )?));
        }
        "scatter_max" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::scatter_max(
                &a, dim, &idx, &src,
            )?));
        }
        "scatter_min" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::scatter_min(
                &a, dim, &idx, &src,
            )?));
        }
        "linalg_multi_dot" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    tens.push(slot_view(slots, capsules, idx)?);
                }
            }
            slots.push(Slot::Owned(kernels::tensor_ops::linalg_multi_dot(tens)?));
        }
        "linalg_vander" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node
                .kwargs
                .get("N")
                .and_then(|v| v.as_i64())
                .map(|v| v as usize);
            slots.push(Slot::Owned(kernels::tensor_ops::linalg_vander(&x, n)?));
        }
        "linalg_vecdot" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(kernels::tensor_ops::linalg_vecdot(
                &a, &b, dim,
            )?));
        }
        "linalg_cross" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(kernels::tensor_ops::linalg_cross(&a, &b, dim)?));
        }
        "linalg_tensordot" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dims = kw_usize(node, "dims", 2);
            slots.push(Slot::Owned(kernels::tensor_ops::linalg_tensordot(
                &a, &b, dims,
            )?));
        }
        "linalg_cholesky_ex" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (l, info) = kernels::tensor_ops::linalg_cholesky_ex(&a)?;
            slots.push(Slot::Tuple(vec![l, info]));
        }
        "linalg_inv_ex" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (inv, info) = kernels::tensor_ops::linalg_inv_ex(&a)?;
            slots.push(Slot::Tuple(vec![inv, info]));
        }
        "linalg_solve_ex" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let (sol, info) = kernels::tensor_ops::linalg_solve_ex(&a, &b)?;
            slots.push(Slot::Tuple(vec![sol, info]));
        }
        "linalg_lu_factor" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (lu, piv) = kernels::tensor_ops::linalg_lu_factor(&a)?;
            slots.push(Slot::Tuple(vec![lu, piv]));
        }
        "local_response_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let size = kw_usize(node, "size", 5);
            let alpha = kw_f64(node, "alpha", 1e-4);
            let beta = kw_f64(node, "beta", 0.75);
            let k = kw_f64(node, "k", 1.0);
            slots.push(Slot::Owned(kernels::tensor_ops::local_response_norm(
                &a, size, alpha, beta, k,
            )?));
        }
        "adaptive_avg_pool1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_usize(node, "output_size", kw_usize(node, "output_size_0", 1));
            slots.push(Slot::Owned(kernels::tensor_ops::adaptive_avg_pool1d(
                &a, out_sz,
            )?));
        }
        "adaptive_max_pool1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_usize(node, "output_size", kw_usize(node, "output_size_0", 1));
            slots.push(Slot::Owned(kernels::tensor_ops::adaptive_max_pool1d(
                &a, out_sz,
            )?));
        }
        "lp_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64(node, "norm_type", 2.0);
            let k = kw_usize(node, "kernel_size", 2);
            let s = kw_usize(node, "stride", 2);
            slots.push(Slot::Owned(kernels::tensor_ops::lp_pool3d(&a, p, k, s)?));
        }
        "logsumexp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(kernels::tensor_ops::logsumexp(
                &a, dim, keepdim,
            )?));
        }
        "randn_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::randn_like(&a)?));
        }
        "rand_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::rand_like(&a)?));
        }
        "randint_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let low = kw_i64(node, "low", 0);
            let high = kw_i64(node, "high", 10);
            slots.push(Slot::Owned(kernels::tensor_ops::randint_like(
                &a, low, high,
            )?));
        }
        "empty_strided" => {
            let size = kw_i64_vec(node, "size");
            let stride = kw_i64_vec(node, "stride");
            slots.push(Slot::Owned(kernels::tensor_ops::empty_strided(
                size, stride,
            )?));
        }
        "view_as" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::view_as(&a, &b)?));
        }
        "expand_as" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::expand_as(&a, &b)?));
        }
        "masked_select_extra" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let m = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::masked_select(&a, &m)?));
        }
        "istft" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(kernels::tensor_ops::istft(&a)?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
