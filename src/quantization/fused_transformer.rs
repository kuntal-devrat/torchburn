//! Fused full-transformer decode step (grouped INT4, T=1, in-place).
//! Inherits root imports via super; pure move.

use super::*;
/// Fused Transformer Layer Decode Step for Grouped INT4 (T=1) in-place:
/// 1. RMSNorm(x, input_norm_w) -> normed
/// 2. QKV GEMV -> qkv
/// 3. RoPE on q and k
/// 4. Write k, v to cache
/// 5. Attention (GQA) -> attn_out
/// 6. O GEMV -> o_out
/// 7. x += o_out
/// 8. RMSNorm(x, post_norm_w) -> normed
/// 9. SwiGLU MLP: Gate + Up GEMV -> Silu(Gate) * Up -> Down GEMV -> mlp_out
/// 10. x += mlp_out
pub fn fused_transformer_layer_step_w4a32(
    x: &mut BorrowedTensor,
    input_norm_w: &BorrowedTensor,
    qkv_w: &BorrowedTensor,
    qkv_s: &BorrowedTensor,
    qkv_b: Option<&BorrowedTensor>,
    o_w: &BorrowedTensor,
    o_s: &BorrowedTensor,
    o_b: Option<&BorrowedTensor>,
    post_norm_w: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    gate_s: &BorrowedTensor,
    gate_b: Option<&BorrowedTensor>,
    up_w: &BorrowedTensor,
    up_s: &BorrowedTensor,
    up_b: Option<&BorrowedTensor>,
    down_w: &BorrowedTensor,
    down_s: &BorrowedTensor,
    down_b: Option<&BorrowedTensor>,
    k_cache: &BorrowedTensor,
    v_cache: &BorrowedTensor,
    cos: &BorrowedTensor,
    sin: &BorrowedTensor,
    offset: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
    eps: f64,
) -> PyResult<()> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "fused_transformer_layer_step_w4a32 requires x with at least 1 dim",
        ));
    }
    let hidden_size = x.shape[x_rank - 1] as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let total_qkv_dim = q_dim + 2 * kv_dim;
    let n_inter = gate_w.shape[0] as usize;

    use rayon::prelude::*;
    let x_slice = unsafe { std::slice::from_raw_parts_mut(x.data as *mut f32, x.buffer_len()) };
    let input_norm_w_slice = unsafe { typed_slice::<f32>(input_norm_w) };
    let qkv_w_slice = unsafe { typed_slice::<u8>(qkv_w) };
    let qkv_s_slice = unsafe { typed_slice::<f32>(qkv_s) };
    let qkv_b_slice = qkv_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let o_w_slice = unsafe { typed_slice::<u8>(o_w) };
    let o_s_slice = unsafe { typed_slice::<f32>(o_s) };
    let o_b_slice = o_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let post_norm_w_slice = unsafe { typed_slice::<f32>(post_norm_w) };
    let gw_slice = unsafe { typed_slice::<u8>(gate_w) };
    let gs_slice = unsafe { typed_slice::<f32>(gate_s) };
    let gb_slice = gate_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let uw_slice = unsafe { typed_slice::<u8>(up_w) };
    let us_slice = unsafe { typed_slice::<f32>(up_s) };
    let ub_slice = up_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let dw_slice = unsafe { typed_slice::<u8>(down_w) };
    let ds_slice = unsafe { typed_slice::<f32>(down_s) };
    let db_slice = down_b.map(|b| unsafe { typed_slice::<f32>(b) });

    let cos_slice = unsafe { typed_slice::<f32>(cos) };
    let sin_slice = unsafe { typed_slice::<f32>(sin) };

    let x_ptr = x_slice.as_mut_ptr();
    let mut normed = vec![0.0f32; hidden_size];

    // 1. Pre-attention RMSNorm
    unsafe {
        fast_rms_norm(
            x_ptr,
            input_norm_w_slice.as_ptr(),
            normed.as_mut_ptr(),
            hidden_size,
            eps as f32,
        );
    }

    // 2. QKV projection
    let mut qkv = vec![0.0f32; total_qkv_dim];
    unsafe {
        gemv_w4a32_grouped(
            normed.as_ptr(),
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

    // 3. RoPE on q and k
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

    // 4. Update KV cache
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

    // 5. GQA Attention
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

    // 6. O projection
    let mut o_out = vec![0.0f32; hidden_size];
    unsafe {
        gemv_w4a32_grouped(
            attn_out.as_ptr(),
            o_w_slice.as_ptr(),
            o_s_slice.as_ptr(),
            o_b_slice.map(|b| b.as_ptr()),
            o_out.as_mut_ptr(),
            hidden_size,
            q_dim,
            group_size,
        );
    }

    // 7. Residual add: x += o_out
    unsafe {
        fast_vector_add(x_ptr, o_out.as_ptr(), hidden_size);
    }

    // 8. Post-attention RMSNorm: normed = rms_norm(x, post_norm_w)
    unsafe {
        fast_rms_norm(
            x_ptr,
            post_norm_w_slice.as_ptr(),
            normed.as_mut_ptr(),
            hidden_size,
            eps as f32,
        );
    }

    // 9. SwiGLU MLP
    let mut h_buf = vec![0.0f32; n_inter];
    let h_ptr = h_buf.as_mut_ptr() as usize;

    let num_groups_k = (hidden_size + group_size - 1) / group_size;
    let bytes_per_row_k = (hidden_size + 1) / 2;

    let x_usize = normed.as_ptr() as usize;
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

    // Pre-quantise the activation for W4A8 VNNI only when the CPU actually
    // supports it and the layout is full 64-wide groups.  The neuron helper
    // re-checks every tier, so non-VNNI / non-AVX-512 / non-64-group CPUs
    // always take a kernel they support (never an unconditional AVX-512 call
    // and never a (0,0) placeholder).
    let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
        let (u, s) = unsafe { quantize_activation_to_u8(normed.as_ptr(), hidden_size) };
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

            // CPU-feature + group-size aware dispatch (VNNI > AVX-512 > AVX2 >
            // scalar) — portable across x86 tiers and non-x86 (Apple/ARM).
            let (g_sum, u_sum) = unsafe {
                swiglu_neuron_w4a32_dot_dispatch(
                    x_p,
                    x_u8_ptr.map(|p| p as *const u8),
                    s_x,
                    gw_row,
                    gs_row,
                    uw_row,
                    us_row,
                    num_groups_k,
                    hidden_size,
                    group_size,
                )
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

    // Down projection
    let mut down_out = vec![0.0f32; hidden_size];
    unsafe {
        gemv_w4a32_grouped(
            h_buf.as_ptr(),
            dw_slice.as_ptr(),
            ds_slice.as_ptr(),
            db_slice.map(|b| b.as_ptr()),
            down_out.as_mut_ptr(),
            hidden_size,
            n_inter,
            group_size,
        );
    }

    // 10. Residual add: x += down_out
    unsafe {
        fast_vector_add(x_ptr, down_out.as_ptr(), hidden_size);
    }

    Ok(())
}
