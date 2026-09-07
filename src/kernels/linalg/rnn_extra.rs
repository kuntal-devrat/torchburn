//! Recurrent cells and transformer blocks: rnn/gru/lstm cells, multi-head attention forward, encoder layer. Inherits linalg root imports via super; pure move.

use super::*;

// ── 116-125. Recurrent Cells & Attention & Solvers ──
pub fn rnn_tanh_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let lin1 = crate::linalg::linear(input, w_ih, b_ih, None)?;
    let lin2 = crate::linalg::linear(hx, w_hh, b_hh, None)?;
    let l1_b = BorrowedTensor::from_owned(&lin1);
    let l2_b = BorrowedTensor::from_owned(&lin2);
    let sum = crate::kernels::binary(crate::kernels::BinaryOp::Add, &l1_b, &l2_b)?;
    let sum_b = BorrowedTensor::from_owned(&sum);
    crate::nn::activations::tanh_act(&sum_b)
}

pub fn rnn_relu_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let lin1 = crate::linalg::linear(input, w_ih, b_ih, None)?;
    let lin2 = crate::linalg::linear(hx, w_hh, b_hh, None)?;
    let l1_b = BorrowedTensor::from_owned(&lin1);
    let l2_b = BorrowedTensor::from_owned(&lin2);
    let sum = crate::kernels::binary(crate::kernels::BinaryOp::Add, &l1_b, &l2_b)?;
    let sum_b = BorrowedTensor::from_owned(&sum);
    crate::kernels::relu(&sum_b)
}

pub fn gru_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh)
}

pub fn lstm_cell(
    input: &BorrowedTensor,
    hx: &BorrowedTensor,
    cx: &BorrowedTensor,
    w_ih: &BorrowedTensor,
    w_hh: &BorrowedTensor,
    b_ih: Option<&BorrowedTensor>,
    b_hh: Option<&BorrowedTensor>,
) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let h = rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh)?;
    let mut c = OwnedTensor::new(cx.dtype, cx.shape.clone());
    let cd = unsafe { typed_slice::<f32>(cx) };
    let od = unsafe { typed_mut_slice::<f32>(&mut c) };
    od.copy_from_slice(cd);
    Ok((h, c))
}

pub fn multi_head_attention_forward(
    query: &BorrowedTensor,
    key: &BorrowedTensor,
    value: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    crate::nn::attention::scaled_dot_product_attention(query, key, value, None, false)
}

// ── Fused TransformerEncoderLayer (torch._transformer_encoder_layer_fwd) ──
// Composes native kernels: qkv projection -> multi-head SDPA -> out-proj ->
// residual + layer-norm -> FFN (gelu/relu) -> residual (+ norm).  Matches
// nn.TransformerEncoderLayer semantics for both pre-norm (norm_first) and
// post-norm layouts.
pub fn transformer_encoder_layer_fwd(
    src: &BorrowedTensor,
    num_heads: usize,
    use_gelu: bool,
    norm_first: bool,
    eps: f64,
    qkv_w: &BorrowedTensor,
    qkv_b: &BorrowedTensor,
    proj_w: &BorrowedTensor,
    proj_b: &BorrowedTensor,
    nw1: &BorrowedTensor,
    nb1: &BorrowedTensor,
    nw2: &BorrowedTensor,
    nb2: &BorrowedTensor,
    ffn_w1: &BorrowedTensor,
    ffn_b1: &BorrowedTensor,
    ffn_w2: &BorrowedTensor,
    ffn_b2: &BorrowedTensor,
    mask: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    if mask.is_some() {
        return Err(unsupported(
            "transformer_encoder_layer_fwd: attention mask not supported natively",
        ));
    }
    if src.shape.len() != 3 {
        return Err(unsupported(
            "transformer_encoder_layer_fwd: src must be [B, T, D]",
        ));
    }
    let b = src.shape[0] as usize;
    let t = src.shape[1] as usize;
    let d = src.shape[2] as usize;
    if num_heads == 0 || d % num_heads != 0 {
        return Err(unsupported(
            "transformer_encoder_layer_fwd: embed_dim must be divisible by num_heads",
        ));
    }
    let dh = d / num_heads;

    // Pre-norm: layer-norm before attention.
    let pre = if norm_first {
        crate::nn::norm::layer_norm(src, nw1, nb1, eps)?
    } else {
        crate::shape_ops::to_contiguous(src)?
    };
    let pre_view = pre.as_view();

    // QKV projection -> [B, T, 3D], then split into 4D [B, H, T, DH] heads.
    let qkv = crate::linalg::linear(&pre_view, qkv_w, Some(qkv_b), None)?;
    let (q4, k4, v4) = split_qkv(&qkv, b, t, num_heads, dh)?;

    // Multi-head scaled dot-product attention (4D path).
    let q4v = q4.as_view();
    let k4v = k4.as_view();
    let v4v = v4.as_view();
    let attn4 = crate::nn::attention::scaled_dot_product_attention(&q4v, &k4v, &v4v, None, false)?;
    let attn3 = merge_heads(&attn4, b, t, num_heads, dh, src.dtype)?;

    // Output projection.
    let attn_out = crate::linalg::linear(&attn3.as_view(), proj_w, Some(proj_b), None)?;

    // Residual 1: x + attn_out.
    let r1 = elementwise_add(src, &attn_out.as_view())?;

    // Layer-norm before FFN: norm_first -> norm2, post-norm -> norm1.
    let ffn_in = if norm_first {
        crate::nn::norm::layer_norm(&r1.as_view(), nw2, nb2, eps)?
    } else {
        crate::nn::norm::layer_norm(&r1.as_view(), nw1, nb1, eps)?
    };

    // FFN: linear1 -> gelu/relu -> linear2.
    let mut ffn = crate::linalg::linear(&ffn_in.as_view(), ffn_w1, Some(ffn_b1), None)?;
    ffn = if use_gelu {
        crate::nn::activations::gelu(&ffn.as_view(), "none")?
    } else {
        crate::kernels::relu(&ffn.as_view())?
    };
    let ffn2 = crate::linalg::linear(&ffn.as_view(), ffn_w2, Some(ffn_b2), None)?;

    // Residual 2 and final norm (post-norm only; pre-norm output is residual).
    let out = elementwise_add(&r1.as_view(), &ffn2.as_view())?;
    if norm_first {
        Ok(out)
    } else {
        crate::nn::norm::layer_norm(&out.as_view(), nw2, nb2, eps)
    }
}

fn elementwise_add(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::kernels::binary(crate::kernels::BinaryOp::Add, a, b)
}

/// Split a fused qkv tensor [B, T, 3D] into q/k/v as contiguous 4D [B, H, T, DH].
fn split_qkv(
    qkv: &OwnedTensor,
    b: usize,
    t: usize,
    h: usize,
    dh: usize,
) -> PyResult<(OwnedTensor, OwnedTensor, OwnedTensor)> {
    let d = h * dh;
    let mut q = OwnedTensor::new(qkv.dtype, vec![b as i64, h as i64, t as i64, dh as i64]);
    let mut k = OwnedTensor::new(qkv.dtype, vec![b as i64, h as i64, t as i64, dh as i64]);
    let mut v = OwnedTensor::new(qkv.dtype, vec![b as i64, h as i64, t as i64, dh as i64]);
    let qkv_view = qkv.as_view();
    match qkv.dtype {
        DType::F32 => {
            let src = unsafe { typed_slice::<f32>(&qkv_view) };
            let (qd, kd, vd) = unsafe {
                (
                    typed_mut_slice::<f32>(&mut q),
                    typed_mut_slice::<f32>(&mut k),
                    typed_mut_slice::<f32>(&mut v),
                )
            };
            for bi in 0..b {
                for ti in 0..t {
                    let base = (bi * t + ti) * 3 * d;
                    for hi in 0..h {
                        for di in 0..dh {
                            let off = hi * dh + di;
                            let dst = ((bi * h + hi) * t + ti) * dh + di;
                            qd[dst] = src[base + off];
                            kd[dst] = src[base + d + off];
                            vd[dst] = src[base + 2 * d + off];
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let src = unsafe { typed_slice::<f64>(&qkv_view) };
            let (qd, kd, vd) = unsafe {
                (
                    typed_mut_slice::<f64>(&mut q),
                    typed_mut_slice::<f64>(&mut k),
                    typed_mut_slice::<f64>(&mut v),
                )
            };
            for bi in 0..b {
                for ti in 0..t {
                    let base = (bi * t + ti) * 3 * d;
                    for hi in 0..h {
                        for di in 0..dh {
                            let off = hi * dh + di;
                            let dst = ((bi * h + hi) * t + ti) * dh + di;
                            qd[dst] = src[base + off];
                            kd[dst] = src[base + d + off];
                            vd[dst] = src[base + 2 * d + off];
                        }
                    }
                }
            }
        }
        _ => {
            return Err(unsupported(
                "transformer_encoder_layer_fwd: qkv must be f32/f64",
            ))
        }
    }
    Ok((q, k, v))
}

/// Merge 4D [B, H, T, DH] attention output back to contiguous [B, T, D].
fn merge_heads(
    attn: &OwnedTensor,
    b: usize,
    t: usize,
    h: usize,
    dh: usize,
    dtype: DType,
) -> PyResult<OwnedTensor> {
    let d = h * dh;
    let mut out = OwnedTensor::new(dtype, vec![b as i64, t as i64, d as i64]);
    let attn_view = attn.as_view();
    match dtype {
        DType::F32 => {
            let src = unsafe { typed_slice::<f32>(&attn_view) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for bi in 0..b {
                for ti in 0..t {
                    for hi in 0..h {
                        for di in 0..dh {
                            od[(bi * t + ti) * d + hi * dh + di] =
                                src[((bi * h + hi) * t + ti) * dh + di];
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let src = unsafe { typed_slice::<f64>(&attn_view) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            for bi in 0..b {
                for ti in 0..t {
                    for hi in 0..h {
                        for di in 0..dh {
                            od[(bi * t + ti) * d + hi * dh + di] =
                                src[((bi * h + hi) * t + ti) * dh + di];
                        }
                    }
                }
            }
        }
        _ => {
            return Err(unsupported(
                "transformer_encoder_layer_fwd: attention must be f32/f64",
            ))
        }
    }
    Ok(out)
}
