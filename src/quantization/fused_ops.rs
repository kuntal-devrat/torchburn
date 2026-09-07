//! Fused quantized ops: SwiGLU MLP, attention decode steps, RMS-norm,
//! vector add, weight quantizers. Inherits root imports via super.

use super::*;
/// Fused SwiGLU MLP for INT8 (W8A32):
/// intermediate = silu(gate_proj(x)) * up_proj(x)
/// out = down_proj(intermediate)
pub fn fused_swiglu_mlp_w8a32(
    x: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    gate_s: &BorrowedTensor,
    gate_b: Option<&BorrowedTensor>,
    up_w: &BorrowedTensor,
    up_s: &BorrowedTensor,
    up_b: Option<&BorrowedTensor>,
    down_w: &BorrowedTensor,
    down_s: &BorrowedTensor,
    down_b: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "fused_swiglu_mlp requires x with at least 1 dim",
        ));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    let n_inter = gate_w.shape[0] as usize;
    let n_out = down_w.shape[0] as usize;

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n_out as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let gw_slice = unsafe { typed_slice::<i8>(gate_w) };
    let gs_slice = unsafe { typed_slice::<f32>(gate_s) };
    let gb_slice = gate_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let uw_slice = unsafe { typed_slice::<i8>(up_w) };
    let us_slice = unsafe { typed_slice::<f32>(up_s) };
    let ub_slice = up_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let dw_slice = unsafe { typed_slice::<i8>(down_w) };
    let ds_slice = unsafe { typed_slice::<f32>(down_s) };
    let db_slice = down_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    use rayon::prelude::*;

    for token_idx in 0..m {
        let x_tok = unsafe { x_slice.as_ptr().add(token_idx * k) };
        let out_tok = unsafe { out_slice.as_mut_ptr().add(token_idx * n_out) };

        let mut h_buf = vec![0.0f32; n_inter];
        let h_ptr = h_buf.as_mut_ptr() as usize;

        let x_usize = x_tok as usize;
        let gw_usize = gw_slice.as_ptr() as usize;
        let gs_usize = gs_slice.as_ptr() as usize;
        let gs_len = gs_slice.len();
        let gb_usize = gb_slice.map(|b| b.as_ptr() as usize);

        let uw_usize = uw_slice.as_ptr() as usize;
        let us_usize = us_slice.as_ptr() as usize;
        let us_len = us_slice.len();
        let ub_usize = ub_slice.map(|b| b.as_ptr() as usize);

        let n_threads = rayon::current_num_threads();
        let min_chunk = (n_inter / (n_threads * 4)).max(8);

        (0..n_inter)
            .into_par_iter()
            .with_min_len(min_chunk)
            .for_each(|j| {
                let x_p = x_usize as *const f32;
                let gw_p = unsafe { (gw_usize as *const i8).add(j * k) };
                let gs = if gs_len > 1 {
                    unsafe { *((gs_usize as *const f32).add(j)) }
                } else {
                    unsafe { *(gs_usize as *const f32) }
                };
                let gb = if let Some(bp) = gb_usize {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0
                };

                let uw_p = unsafe { (uw_usize as *const i8).add(j * k) };
                let us = if us_len > 1 {
                    unsafe { *((us_usize as *const f32).add(j)) }
                } else {
                    unsafe { *(us_usize as *const f32) }
                };
                let ub = if let Some(bp) = ub_usize {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0
                };

                let g_dot = unsafe { dot_f32_i8(x_p, gw_p, k) };
                let g = g_dot * gs + gb;

                let u_dot = unsafe { dot_f32_i8(x_p, uw_p, k) };
                let u = u_dot * us + ub;

                let silu_g = g / (1.0 + (-g).exp());
                let val = silu_g * u;

                unsafe {
                    let h_p = h_ptr as *mut f32;
                    *h_p.add(j) = val;
                }
            });

        unsafe {
            gemv_w8a32(
                h_buf.as_ptr(),
                dw_slice.as_ptr(),
                ds_slice.as_ptr(),
                ds_slice.len(),
                db_slice.map(|b| b.as_ptr()),
                out_tok,
                n_out,
                n_inter,
            );
        }
    }

    Ok(out)
}

/// Fused SwiGLU MLP for Grouped INT4 (W4A32):
/// intermediate = silu(gate_proj(x)) * up_proj(x)
/// out = down_proj(intermediate)
pub fn fused_swiglu_mlp_w4a32(
    x: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    gate_s: &BorrowedTensor,
    gate_b: Option<&BorrowedTensor>,
    up_w: &BorrowedTensor,
    up_s: &BorrowedTensor,
    up_b: Option<&BorrowedTensor>,
    down_w: &BorrowedTensor,
    down_s: &BorrowedTensor,
    down_b: Option<&BorrowedTensor>,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "fused_swiglu_mlp_w4a32 requires x with at least 1 dim",
        ));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    let n_inter = gate_w.shape[0] as usize;
    let n_out = down_w.shape[0] as usize;

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n_out as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let gw_slice = unsafe { typed_slice::<u8>(gate_w) };
    let gs_slice = unsafe { typed_slice::<f32>(gate_s) };
    let gb_slice = gate_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let uw_slice = unsafe { typed_slice::<u8>(up_w) };
    let us_slice = unsafe { typed_slice::<f32>(up_s) };
    let ub_slice = up_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let dw_slice = unsafe { typed_slice::<u8>(down_w) };
    let ds_slice = unsafe { typed_slice::<f32>(down_s) };
    let db_slice = down_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    let num_groups_k = (k + group_size - 1) / group_size;
    let bytes_per_row_k = (k + 1) / 2;

    use rayon::prelude::*;

    for token_idx in 0..m {
        let x_tok = unsafe { x_slice.as_ptr().add(token_idx * k) };
        let out_tok = unsafe { out_slice.as_mut_ptr().add(token_idx * n_out) };

        let mut h_buf = vec![0.0f32; n_inter];
        let h_ptr = h_buf.as_mut_ptr() as usize;

        let x_usize = x_tok as usize;
        let gw_usize = gw_slice.as_ptr() as usize;
        let gs_usize = gs_slice.as_ptr() as usize;
        let gb_usize = gb_slice.map(|b| b.as_ptr() as usize);

        let uw_usize = uw_slice.as_ptr() as usize;
        let us_usize = us_slice.as_ptr() as usize;
        let ub_usize = ub_slice.map(|b| b.as_ptr() as usize);

        let n_threads = rayon::current_num_threads();
        let min_chunk = (n_inter / (n_threads * 4)).max(8);

        #[cfg(target_arch = "x86_64")]
        let has_vnni = is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw");
        #[cfg(not(target_arch = "x86_64"))]
        let has_vnni = false;

        #[cfg(target_arch = "x86_64")]
        let has_avx512 =
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx512 = false;

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx2 = false;

        let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
            let (u, s) = unsafe { quantize_activation_to_u8(x_tok, k) };
            (Some(u), s)
        } else {
            (None, 1.0f32)
        };
        let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

        (0..n_inter)
            .into_par_iter()
            .with_min_len(min_chunk)
            .for_each(|j| {
                let x_p = x_usize as *const f32;
                let gw_row = unsafe { (gw_usize as *const u8).add(j * bytes_per_row_k) };
                let gs_row = unsafe { (gs_usize as *const f32).add(j * num_groups_k) };
                let gb = if let Some(bp) = gb_usize {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0
                };

                let uw_row = unsafe { (uw_usize as *const u8).add(j * bytes_per_row_k) };
                let us_row = unsafe { (us_usize as *const f32).add(j * num_groups_k) };
                let ub = if let Some(bp) = ub_usize {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0
                };

                let (g_sum, u_sum) = if let Some(x_u8_p) = x_u8_ptr {
                    #[cfg(target_arch = "x86_64")]
                    {
                        unsafe {
                            swiglu_neuron_w4a8_group64_vnni_avx512(
                                x_u8_p as *const u8,
                                s_x,
                                gw_row,
                                gs_row,
                                uw_row,
                                us_row,
                                num_groups_k,
                            )
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let _ = (x_u8_p, s_x);
                        (0.0, 0.0)
                    }
                } else if has_avx512 {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if group_size == 64 {
                            unsafe {
                                swiglu_neuron_w4a32_group64_avx512(
                                    x_p,
                                    gw_row,
                                    gs_row,
                                    uw_row,
                                    us_row,
                                    num_groups_k,
                                )
                            }
                        } else if group_size == 32 {
                            unsafe {
                                swiglu_neuron_w4a32_group32_avx512(
                                    x_p,
                                    gw_row,
                                    gs_row,
                                    uw_row,
                                    us_row,
                                    num_groups_k,
                                )
                            }
                        } else {
                            let mut gs = 0.0f32;
                            let mut us = 0.0f32;
                            for g in 0..num_groups_k {
                                let cur_len = (k - g * group_size).min(group_size);
                                gs += unsafe {
                                    dot_f32_u4_group_scalar(
                                        x_p.add(g * group_size),
                                        gw_row.add(g * (group_size / 2)),
                                        cur_len,
                                    ) * *gs_row.add(g)
                                };
                                us += unsafe {
                                    dot_f32_u4_group_scalar(
                                        x_p.add(g * group_size),
                                        uw_row.add(g * (group_size / 2)),
                                        cur_len,
                                    ) * *us_row.add(g)
                                };
                            }
                            (gs, us)
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        (0.0, 0.0)
                    }
                } else if has_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if group_size == 64 {
                            unsafe {
                                swiglu_neuron_w4a32_group64_avx2(
                                    x_p,
                                    gw_row,
                                    gs_row,
                                    uw_row,
                                    us_row,
                                    num_groups_k,
                                )
                            }
                        } else if group_size == 32 {
                            unsafe {
                                swiglu_neuron_w4a32_group32_avx2(
                                    x_p,
                                    gw_row,
                                    gs_row,
                                    uw_row,
                                    us_row,
                                    num_groups_k,
                                )
                            }
                        } else {
                            let mut gs = 0.0f32;
                            let mut us = 0.0f32;
                            for g in 0..num_groups_k {
                                let cur_len = (k - g * group_size).min(group_size);
                                gs += unsafe {
                                    dot_f32_u4_group_scalar(
                                        x_p.add(g * group_size),
                                        gw_row.add(g * (group_size / 2)),
                                        cur_len,
                                    ) * *gs_row.add(g)
                                };
                                us += unsafe {
                                    dot_f32_u4_group_scalar(
                                        x_p.add(g * group_size),
                                        uw_row.add(g * (group_size / 2)),
                                        cur_len,
                                    ) * *us_row.add(g)
                                };
                            }
                            (gs, us)
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        (0.0, 0.0)
                    }
                } else {
                    let mut gs = 0.0f32;
                    let mut us = 0.0f32;
                    for g in 0..num_groups_k {
                        let cur_len = (k - g * group_size).min(group_size);
                        gs += unsafe {
                            dot_f32_u4_group_scalar(
                                x_p.add(g * group_size),
                                gw_row.add(g * (group_size / 2)),
                                cur_len,
                            ) * *gs_row.add(g)
                        };
                        us += unsafe {
                            dot_f32_u4_group_scalar(
                                x_p.add(g * group_size),
                                uw_row.add(g * (group_size / 2)),
                                cur_len,
                            ) * *us_row.add(g)
                        };
                    }
                    (gs, us)
                };

                let g = g_sum + gb;
                let u = u_sum + ub;
                let silu_g = g / (1.0 + (-g).exp());
                let val = silu_g * u;

                unsafe {
                    let h_p = h_ptr as *mut f32;
                    *h_p.add(j) = val;
                }
            });

        unsafe {
            gemv_w4a32_grouped(
                h_buf.as_ptr(),
                dw_slice.as_ptr(),
                ds_slice.as_ptr(),
                db_slice.map(|b| b.as_ptr()),
                out_tok,
                n_out,
                n_inter,
                group_size,
            );
        }
    }

    Ok(out)
}

/// Portable grouped-int4 SwiGLU gate/up dot-pair dispatch for one neuron.
///
/// Picks the widest kernel the *current* CPU actually supports (checked at
/// runtime, like every other kernel in this crate):
///   AVX-512 VNNI (W4A8) > AVX-512 (W4A32) > AVX2 (W4A32) > portable scalar,
/// and honours the group size: the group64/group32 SIMD kernels assume every
/// group is full (`k % group_size == 0`), otherwise the per-group scalar dot
/// loop (which clamps the tail group to the remaining length) is used.
///
/// `x_u8`/`s_x` are `Some` when the caller pre-quantised the activation to
/// u8 for the VNNI path (only valid for `group_size == 64`).
///
/// # Safety
/// `x`/`gw_row`/`uw_row` must be readable for `k` elements/`(k+1)/2` packed
/// bytes, `gs_row`/`us_row` for `num_groups`, and `x_u8` for `k` bytes when
/// present.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn swiglu_neuron_w4a32_dot_dispatch(
    x: *const f32,
    x_u8: Option<*const u8>,
    s_x: f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
    k: usize,
    group_size: usize,
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    let has_vnni = is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let has_vnni = false;
    #[cfg(target_arch = "x86_64")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx512 = false;
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    let full_groups = k % group_size == 0 && k / group_size == num_groups;

    // Scalar fallback for arbitrary group sizes (tail groups clamped).
    let scalar = |x_p: *const f32| -> (f32, f32) {
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        for g in 0..num_groups {
            let cur_len = (k - g * group_size).min(group_size);
            gs += dot_f32_u4_group_scalar(
                x_p.add(g * group_size),
                gw_row.add(g * (group_size / 2)),
                cur_len,
            ) * *gs_row.add(g);
            us += dot_f32_u4_group_scalar(
                x_p.add(g * group_size),
                uw_row.add(g * (group_size / 2)),
                cur_len,
            ) * *us_row.add(g);
        }
        (gs, us)
    };

    if let Some(x_u8_p) = x_u8 {
        // W4A8 VNNI: only defined for full 64-wide groups.
        if has_vnni && group_size == 64 && full_groups {
            #[cfg(target_arch = "x86_64")]
            {
                return swiglu_neuron_w4a8_group64_vnni_avx512(
                    x_u8_p, s_x, gw_row, gs_row, uw_row, us_row, num_groups,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = (x_u8_p, s_x);
            }
        }
    }
    if full_groups && group_size == 64 && has_avx512 {
        #[cfg(target_arch = "x86_64")]
        {
            return swiglu_neuron_w4a32_group64_avx512(x, gw_row, gs_row, uw_row, us_row, num_groups);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = has_avx512;
        }
    }
    if full_groups && group_size == 32 && has_avx512 {
        #[cfg(target_arch = "x86_64")]
        {
            return swiglu_neuron_w4a32_group32_avx512(x, gw_row, gs_row, uw_row, us_row, num_groups);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = has_avx512;
        }
    }
    if full_groups && group_size == 64 && has_avx2 {
        #[cfg(target_arch = "x86_64")]
        {
            return swiglu_neuron_w4a32_group64_avx2(x, gw_row, gs_row, uw_row, us_row, num_groups);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = has_avx2;
        }
    }
    if full_groups && group_size == 32 && has_avx2 {
        #[cfg(target_arch = "x86_64")]
        {
            return swiglu_neuron_w4a32_group32_avx2(x, gw_row, gs_row, uw_row, us_row, num_groups);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = has_avx2;
        }
    }
    scalar(x)
}

/// Quantize a 2D float weight matrix (N, K) to per-channel INT8 with scales (N,).
pub fn quantize_linear_weights_int8(w: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    if w.shape.len() != 2 {
        return Err(unsupported(
            "quantize_linear_weights_int8 requires 2D matrix",
        ));
    }
    let n = w.shape[0] as usize;
    let k = w.shape[1] as usize;
    let mut out_w = OwnedTensor::new(DType::Bool, w.shape.clone());
    let mut out_s = OwnedTensor::new(DType::F32, vec![n as i64]);

    let w_src = unsafe { typed_slice::<f32>(w) };
    let w_dst = unsafe { typed_mut_slice::<i8>(&mut out_w) };
    let s_dst = unsafe { typed_mut_slice::<f32>(&mut out_s) };

    for j in 0..n {
        let row = &w_src[j * k..(j + 1) * k];
        let mut max_abs = 0.0f32;
        for &val in row {
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1.0 };
        s_dst[j] = scale;
        let inv_scale = 1.0 / scale;
        let out_row = &mut w_dst[j * k..(j + 1) * k];
        for p in 0..k {
            let q = (row[p] * inv_scale).round();
            out_row[p] = q.clamp(-127.0, 127.0) as i8;
        }
    }

    Ok((out_w, out_s))
}

/// Quantize a 2D float weight matrix (N, K) to symmetric 4-bit packed INT4 with scales (N,).
pub fn quantize_linear_weights_int4(w: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    if w.shape.len() != 2 {
        return Err(unsupported(
            "quantize_linear_weights_int4 requires 2D matrix",
        ));
    }
    let n = w.shape[0] as usize;
    let k = w.shape[1] as usize;
    let k_packed = (k + 1) / 2;
    let mut out_w = OwnedTensor::new(DType::Bool, vec![n as i64, k_packed as i64]);
    let mut out_s = OwnedTensor::new(DType::F32, vec![n as i64]);

    let w_src = unsafe { typed_slice::<f32>(w) };
    let w_dst = unsafe { typed_mut_slice::<u8>(&mut out_w) };
    let s_dst = unsafe { typed_mut_slice::<f32>(&mut out_s) };

    for j in 0..n {
        let row = &w_src[j * k..(j + 1) * k];
        let mut max_abs = 0.0f32;
        for &val in row {
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 1e-8 { max_abs / 7.0 } else { 1.0 };
        s_dst[j] = scale;
        let inv_scale = 1.0 / scale;
        let out_row = &mut w_dst[j * k_packed..(j + 1) * k_packed];
        for b in 0..k_packed {
            let p0 = b * 2;
            let p1 = b * 2 + 1;
            let q0 = if p0 < k {
                ((row[p0] * inv_scale).round().clamp(-8.0, 7.0) as i8) + 8
            } else {
                8
            } as u8;
            let q1 = if p1 < k {
                ((row[p1] * inv_scale).round().clamp(-8.0, 7.0) as i8) + 8
            } else {
                8
            } as u8;
            out_row[b] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }

    Ok((out_w, out_s))
}

/// Dot product of two f32 slices.
///
/// # Safety
/// `a` and `b` must each point to at least `len` readable f32 elements.
#[inline(always)]
pub unsafe fn dot_f32_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let feats = crate::dispatch::cpu_features();
        if feats.avx512f && len >= 16 {
            use std::arch::x86_64::*;
            let mut acc = _mm512_setzero_ps();
            let mut i = 0;
            while i + 16 <= len {
                let av = _mm512_loadu_ps(a.add(i));
                let bv = _mm512_loadu_ps(b.add(i));
                acc = _mm512_fmadd_ps(av, bv, acc);
                i += 16;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            while i < len {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            return sum;
        }
        if feats.avx2 && feats.fma && len >= 8 {
            use std::arch::x86_64::*;
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= len {
                let av = _mm256_loadu_ps(a.add(i));
                let bv = _mm256_loadu_ps(b.add(i));
                acc = _mm256_fmadd_ps(av, bv, acc);
                i += 8;
            }
            let mut sum = hsum256_ps_avx(acc);
            while i < len {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            return sum;
        }
    }
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += *a.add(i) * *b.add(i);
    }
    sum
}

/// Fused Attention Decode Step for W8A32 (T=1):
/// Computes:
/// 1. qkv = gemv_w8a32(x, qkv_w, qkv_s, qkv_b)
/// 2. In-place RoPE on q and k with cos and sin
/// 3. In-place update of k_cache and v_cache at offset
/// 4. GQA Attention scores, softmax, and weighted value accumulation
/// 5. o_out = gemv_w8a32(attn_out, o_w, o_s, o_b)
pub fn fused_attention_step_w8a32(
    x: &BorrowedTensor,
    qkv_w: &BorrowedTensor,
    qkv_s: &BorrowedTensor,
    qkv_b: Option<&BorrowedTensor>,
    o_w: &BorrowedTensor,
    o_s: &BorrowedTensor,
    o_b: Option<&BorrowedTensor>,
    k_cache: &BorrowedTensor,
    v_cache: &BorrowedTensor,
    cos: &BorrowedTensor,
    sin: &BorrowedTensor,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "fused_attention_step requires x with at least 1 dim",
        ));
    }
    let hidden_size = x.shape[x_rank - 1] as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let total_qkv_dim = q_dim + 2 * kv_dim;

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let qkv_w_slice = unsafe { typed_slice::<i8>(qkv_w) };
    let qkv_s_slice = unsafe { typed_slice::<f32>(qkv_s) };
    let qkv_b_slice = qkv_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let o_w_slice = unsafe { typed_slice::<i8>(o_w) };
    let o_s_slice = unsafe { typed_slice::<f32>(o_s) };
    let o_b_slice = o_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let cos_slice = unsafe { typed_slice::<f32>(cos) };
    let sin_slice = unsafe { typed_slice::<f32>(sin) };

    // 1. Compute QKV projection
    let mut qkv = vec![0.0f32; total_qkv_dim];
    unsafe {
        gemv_w8a32(
            x_slice.as_ptr(),
            qkv_w_slice.as_ptr(),
            qkv_s_slice.as_ptr(),
            qkv_s_slice.len(),
            qkv_b_slice.map(|b| b.as_ptr()),
            qkv.as_mut_ptr(),
            total_qkv_dim,
            hidden_size,
        );
    }

    let (q, kv_rest) = qkv.split_at_mut(q_dim);
    let (k, v) = kv_rest.split_at_mut(kv_dim);

    // 2. In-place RoPE on q and k
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let q1 = q_head[i];
            let q2 = q_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            q_head[i] = q1 * c1 - q2 * s1;
            q_head[i + half_dim] = q2 * c2 + q1 * s2;
        }
    }

    for h in 0..num_kv_heads {
        let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let k1 = k_head[i];
            let k2 = k_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            k_head[i] = k1 * c1 - k2 * s1;
            k_head[i + half_dim] = k2 * c2 + k1 * s2;
        }
    }

    // 3. Update KV cache
    let max_seq_len = k_cache.shape[2] as usize;
    let head_stride = max_seq_len * head_dim;

    let k_cache_mut =
        unsafe { std::slice::from_raw_parts_mut(k_cache.data as *mut f32, k_cache.buffer_len()) };
    let v_cache_mut =
        unsafe { std::slice::from_raw_parts_mut(v_cache.data as *mut f32, v_cache.buffer_len()) };

    for kv_h in 0..num_kv_heads {
        let dst_offset = kv_h * head_stride + offset * head_dim;
        let k_src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
        let v_src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
        k_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(k_src);
        v_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(v_src);
    }

    // 4. Attention (GQA)
    let seq_len = offset + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    let mut attn_out = vec![0.0f32; q_dim];
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
        let k_base_ptr = unsafe { (k_cache.data as *const f32).add(kv_h * head_stride) };
        let v_base_ptr = unsafe { (v_cache.data as *const f32).add(kv_h * head_stride) };

        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
            let dot = unsafe { dot_f32_f32(q_ptr, k_t_ptr, head_dim) };
            let sc = dot * scale;
            scores[t] = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        let mut exp_sum = 0.0f32;
        for t in 0..seq_len {
            let ex = (scores[t] - max_score).exp();
            scores[t] = ex;
            exp_sum += ex;
        }
        let inv_sum = 1.0f32 / exp_sum;

        let out_h_ptr = unsafe { attn_out.as_mut_ptr().add(h * head_dim) };
        for t in 0..seq_len {
            let w = scores[t] * inv_sum;
            let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
            for d in 0..head_dim {
                unsafe {
                    *out_h_ptr.add(d) += w * *v_t_ptr.add(d);
                }
            }
        }
    }

    // 5. Output projection
    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = hidden_size as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    unsafe {
        gemv_w8a32(
            attn_out.as_ptr(),
            o_w_slice.as_ptr(),
            o_s_slice.as_ptr(),
            o_s_slice.len(),
            o_b_slice.map(|b| b.as_ptr()),
            out_slice.as_mut_ptr(),
            hidden_size,
            q_dim,
        );
    }

    Ok(out)
}

/// Fused Attention Decode Step for Grouped INT4 (T=1):
pub fn fused_attention_step_w4a32(
    x: &BorrowedTensor,
    qkv_w: &BorrowedTensor,
    qkv_s: &BorrowedTensor,
    qkv_b: Option<&BorrowedTensor>,
    o_w: &BorrowedTensor,
    o_s: &BorrowedTensor,
    o_b: Option<&BorrowedTensor>,
    k_cache: &BorrowedTensor,
    v_cache: &BorrowedTensor,
    cos: &BorrowedTensor,
    sin: &BorrowedTensor,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "fused_attention_step_w4a32 requires x with at least 1 dim",
        ));
    }
    let hidden_size = x.shape[x_rank - 1] as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let total_qkv_dim = q_dim + 2 * kv_dim;

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let qkv_w_slice = unsafe { typed_slice::<u8>(qkv_w) };
    let qkv_s_slice = unsafe { typed_slice::<f32>(qkv_s) };
    let qkv_b_slice = qkv_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let o_w_slice = unsafe { typed_slice::<u8>(o_w) };
    let o_s_slice = unsafe { typed_slice::<f32>(o_s) };
    let o_b_slice = o_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let cos_slice = unsafe { typed_slice::<f32>(cos) };
    let sin_slice = unsafe { typed_slice::<f32>(sin) };

    // 1. Compute QKV projection with gemv_w4a32_grouped
    let mut qkv = vec![0.0f32; total_qkv_dim];
    unsafe {
        gemv_w4a32_grouped(
            x_slice.as_ptr(),
            qkv_w_slice.as_ptr(),
            qkv_s_slice.as_ptr(),
            qkv_b_slice.map(|b| b.as_ptr()),
            qkv.as_mut_ptr(),
            total_qkv_dim,
            hidden_size,
            group_size,
        );
    }

    let (q, kv_rest) = qkv.split_at_mut(q_dim);
    let (k, v) = kv_rest.split_at_mut(kv_dim);

    // 2. In-place RoPE on q and k
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let q1 = q_head[i];
            let q2 = q_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            q_head[i] = q1 * c1 - q2 * s1;
            q_head[i + half_dim] = q2 * c2 + q1 * s2;
        }
    }

    for h in 0..num_kv_heads {
        let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let k1 = k_head[i];
            let k2 = k_head[i + half_dim];
            let c1 = cos_slice[i];
            let s1 = sin_slice[i];
            let c2 = cos_slice[i + half_dim];
            let s2 = sin_slice[i + half_dim];
            k_head[i] = k1 * c1 - k2 * s1;
            k_head[i + half_dim] = k2 * c2 + k1 * s2;
        }
    }

    // 3. Update KV cache
    let max_seq_len = k_cache.shape[2] as usize;
    let head_stride = max_seq_len * head_dim;

    let k_cache_mut =
        unsafe { std::slice::from_raw_parts_mut(k_cache.data as *mut f32, k_cache.buffer_len()) };
    let v_cache_mut =
        unsafe { std::slice::from_raw_parts_mut(v_cache.data as *mut f32, v_cache.buffer_len()) };

    for kv_h in 0..num_kv_heads {
        let dst_offset = kv_h * head_stride + offset * head_dim;
        let k_src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
        let v_src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
        k_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(k_src);
        v_cache_mut[dst_offset..dst_offset + head_dim].copy_from_slice(v_src);
    }

    // 4. Attention (GQA)
    let seq_len = offset + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let heads_per_kv = num_heads / num_kv_heads;

    let mut attn_out = vec![0.0f32; q_dim];
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
        let k_base_ptr = unsafe { (k_cache.data as *const f32).add(kv_h * head_stride) };
        let v_base_ptr = unsafe { (v_cache.data as *const f32).add(kv_h * head_stride) };

        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
            let dot = unsafe { dot_f32_f32(q_ptr, k_t_ptr, head_dim) };
            let sc = dot * scale;
            scores[t] = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        let mut exp_sum = 0.0f32;
        for t in 0..seq_len {
            let ex = (scores[t] - max_score).exp();
            scores[t] = ex;
            exp_sum += ex;
        }
        let inv_sum = 1.0f32 / exp_sum;

        let out_h_ptr = unsafe { attn_out.as_mut_ptr().add(h * head_dim) };
        for t in 0..seq_len {
            let w = scores[t] * inv_sum;
            let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
            for d in 0..head_dim {
                unsafe {
                    *out_h_ptr.add(d) += w * *v_t_ptr.add(d);
                }
            }
        }
    }

    // 5. Output projection with gemv_w4a32_grouped
    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = hidden_size as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    unsafe {
        gemv_w4a32_grouped(
            attn_out.as_ptr(),
            o_w_slice.as_ptr(),
            o_s_slice.as_ptr(),
            o_b_slice.map(|b| b.as_ptr()),
            out_slice.as_mut_ptr(),
            hidden_size,
            q_dim,
            group_size,
        );
    }

    Ok(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fast_rms_norm_avx512(x: *const f32, w: *const f32, out: *mut f32, n: usize, eps: f32) {
    use std::arch::x86_64::*;
    let mut sum_sq = _mm512_setzero_ps();
    let num_chunks = n / 16;
    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        sum_sq = _mm512_fmadd_ps(v, v, sum_sq);
    }
    let mut total_sq = _mm512_reduce_add_ps(sum_sq);
    for i in (num_chunks * 16)..n {
        let v = *x.add(i);
        total_sq += v * v;
    }
    let rms = (total_sq / (n as f32) + eps).sqrt();
    let inv_rms = _mm512_set1_ps(1.0 / rms);

    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        let weight = _mm512_loadu_ps(w.add(c * 16));
        let norm = _mm512_mul_ps(_mm512_mul_ps(v, inv_rms), weight);
        _mm512_storeu_ps(out.add(c * 16), norm);
    }
    for i in (num_chunks * 16)..n {
        *out.add(i) = (*x.add(i) / rms) * *w.add(i);
    }
}

/// RMS layer norm: `out[i] = x[i] / rms(x) * w[i]`.
///
/// # Safety
/// `x` and `w` must each point to at least `n` readable f32 elements;
/// `out` must point to at least `n` writable f32 elements.
#[inline(always)]
pub unsafe fn fast_rms_norm(x: *const f32, w: *const f32, out: *mut f32, n: usize, eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            fast_rms_norm_avx512(x, w, out, n, eps);
            return;
        }
    }
    let mut total_sq = 0.0f32;
    for i in 0..n {
        let v = *x.add(i);
        total_sq += v * v;
    }
    let rms = (total_sq / (n as f32) + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        *out.add(i) = *x.add(i) * inv_rms * *w.add(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fast_vector_add_avx512(dst: *mut f32, src: *const f32, n: usize) {
    use std::arch::x86_64::*;
    let num_chunks = n / 16;
    for c in 0..num_chunks {
        let d = _mm512_loadu_ps(dst.add(c * 16));
        let s = _mm512_loadu_ps(src.add(c * 16));
        _mm512_storeu_ps(dst.add(c * 16), _mm512_add_ps(d, s));
    }
    for i in (num_chunks * 16)..n {
        *dst.add(i) += *src.add(i);
    }
}

#[inline(always)]
pub(crate) unsafe fn fast_vector_add(dst: *mut f32, src: *const f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            fast_vector_add_avx512(dst, src, n);
            return;
        }
    }
    for i in 0..n {
        *dst.add(i) += *src.add(i);
    }
}
