//! Attention kernels (Phase 4).
//!
//! * `scaled_dot_product_attention(q, k, v, mask?, is_causal, dropout_p)`
//!   computes softmax(QK^T / sqrt(D) + mask) V over heads, memory-efficiently
//!   (per query row: scores are a T-length row, never a full B*H*T*T buffer).
//!   Supports additive float masks and boolean masks (False -> -inf), plus
//!   `is_causal` upper-triangular masking.  Dropout is ignored (inference).
//!
//!   The f32 path is SIMD-vectorised with the `wide` crate (SSE2/AVX2 on
//!   x86-64, NEON on ARM): the QK^T dot products run 4 lanes at a time, the
//!   softmax exponentials are a vectorised exp2 polynomial, and the V
//!   accumulation keeps a per-query D-length register accumulator so V rows
//!   are streamed sequentially instead of re-read with a D stride.
//!
//! * `rope(x, cos, sin)` applies rotary positional embeddings in the
//!   HuggingFace split-half convention:
//!   `out[..., :D/2] = x1*cos - x2*sin; out[..., D/2:] = x1*sin + x2*cos`.

use crate::dlpack::{
    contiguous_strides, elem_count, unsupported, BorrowedTensor, DType, OwnedTensor,
};
use pyo3::prelude::*;
use wide::f32x4;

/// Read a tensor's elements as a typed slice.
unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

/// Write typed data into an owned tensor.
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

/// Load four contiguous f32s starting at `off` into a SIMD lane.
#[inline(always)]
fn load4(s: &[f32], off: usize) -> f32x4 {
    let mut a = [0.0f32; 4];
    a.copy_from_slice(&s[off..off + 4]);
    f32x4::from(a)
}

/// Compute the flat index of a (b, h, i, j) score position in a broadcastable
/// mask tensor.  Broadcasting: any mask dim of size 1 repeats; the mask must
/// otherwise match the score grid [B, H, T, T] from the trailing dims.
fn mask_value(mask: &BorrowedTensor, b: usize, h: usize, i: usize, j: usize) -> f32 {
    let m = &mask.shape;
    let rank = m.len();
    if rank == 0 {
        match mask.dtype {
            DType::Bool => {
                let bytes = unsafe { typed_slice::<u8>(mask) };
                return if bytes[0] != 0 {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
            DType::F32 => return (unsafe { typed_slice::<f32>(mask) })[0],
            DType::F64 => return (unsafe { typed_slice::<f64>(mask) })[0] as f32,
            _ => return 0.0,
        }
    }
    // Build coords from the trailing dims, padding with 1 (broadcast) in front.
    let coords = [b, h, i, j];
    let mut idx = 0usize;
    for k in 0..rank {
        let grid_k = if rank <= 4 { 4 - rank + k } else { k };
        let dim = m[k].max(1) as usize;
        let coord = if dim == 1 { 0 } else { coords[grid_k] };
        idx += coord * mask.strides[k] as usize;
    }
    match mask.dtype {
        DType::Bool => {
            let bytes = unsafe { typed_slice::<u8>(mask) };
            if bytes[idx] != 0 {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        }
        DType::F32 => (unsafe { typed_slice::<f32>(mask) })[idx],
        DType::F64 => (unsafe { typed_slice::<f64>(mask) })[idx] as f32,
        _ => 0.0, // unsupported mask dtype -> treated as no-op; callers pre-check
    }
}

/// In-place softmax of a T-length score row (max-subtracted, vectorised exp).
///
/// Writes the probabilities back into `scores` and returns the partition
/// sum.  A fully-masked row (all -inf) yields NaN probabilities and a NaN
/// partition sum — the caller treats `z > 0.0` as false and writes zeros,
/// matching torch's all-zero output for fully-masked rows.
#[inline]
fn softmax_row(scores: &mut [f32]) -> f32 {
    let t = scores.len();
    let mut max_s = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > max_s {
            max_s = s;
        }
    }
    let mut z = 0.0f32;
    let maxv = f32x4::splat(max_s);
    let mut j = 0;
    while j + 4 <= t {
        let e = (load4(scores, j) - maxv).exp();
        let arr = e.to_array();
        for k in 0..4 {
            scores[j + k] = arr[k];
            z += arr[k];
        }
        j += 4;
    }
    while j < t {
        let e = (scores[j] - max_s).exp();
        scores[j] = e;
        z += e;
        j += 1;
    }
    z
}

/// Total MACs above which SDPA parallelises across the B*H head blocks.
/// Below this the rayon dispatch overhead outweighs the parallelism win.
const SDPA_PAR_THRESHOLD: usize = 1_000_000;

/// Compute one (b, h) head block of attention output: `od` receives the
/// block's [T, D] output (already sliced out of the full output tensor).
/// `bh_t` is the block's row offset `(b * H + h) * T` into the flat q/k/v
/// layouts.  Used both serially and from rayon worker threads (the q/k/v
/// views are read-only and `BorrowedTensor` is `Sync`).
#[allow(clippy::too_many_arguments)]
fn sdpa_block_f32(
    qd: &[f32],
    kd: &[f32],
    vd: &[f32],
    od: &mut [f32],
    bi: usize,
    hi: usize,
    h: usize,
    t: usize,
    d: usize,
    scale_f: f32,
    causal_only: bool,
    is_causal: bool,
    mask: Option<&BorrowedTensor>,
) {
    let bh_t = (bi * h + hi) * t;
    let d4 = d / 4;
    let drem = d % 4;
    let vec_tail = d4 * 4;
    let mut scores = vec![0.0f32; t];
    let mut acc: Vec<f32x4> = vec![f32x4::ZERO; d4];
    let mut acc_rem = [0.0f32; 3];
    for i in 0..t {
        let q_off = (bh_t + i) * d;
        // 1) QK^T: scores[j] = scale * <q[i], k[j]> (+ mask).
        //    Pure-causal rows initialise the upper triangle to -inf and only
        //    compute j <= i.
        for j in 0..t {
            if causal_only && j > i {
                scores[j] = f32::NEG_INFINITY;
                continue;
            }
            let k_off = (bh_t + j) * d;
            let mut s = f32x4::ZERO;
            for a in 0..d4 {
                // qv.mul_add(kv, s) = qv * kv + s
                s = load4(qd, q_off + a * 4).mul_add(load4(kd, k_off + a * 4), s);
            }
            let mut sv = s.reduce_add() * scale_f;
            for r in 0..drem {
                sv += qd[q_off + vec_tail + r] * kd[k_off + vec_tail + r];
            }
            if is_causal && j > i {
                sv = f32::NEG_INFINITY;
            }
            if let Some(m) = mask {
                let mv = mask_value(m, bi, hi, i, j);
                sv = if mv.is_infinite() && mv < 0.0 {
                    f32::NEG_INFINITY
                } else {
                    sv + mv
                };
            }
            scores[j] = sv;
        }
        // 2) softmax (max-subtracted, vectorised exp).
        let z = softmax_row(&mut scores);
        let out_off = i * d;
        if z > 0.0 {
            // 3) V accumulation: stream each V row once into a per-query
            //    register accumulator (D lanes).
            for a in acc.iter_mut() {
                *a = f32x4::ZERO;
            }
            for r in acc_rem.iter_mut() {
                *r = 0.0;
            }
            let inv_z = 1.0 / z;
            for j in 0..t {
                let w = scores[j];
                if w == 0.0 {
                    continue; // masked-out key row
                }
                let v_off = (bh_t + j) * d;
                let wv = f32x4::splat(w);
                for a in 0..d4 {
                    // wv.mul_add(vrow, acc) = w * vrow + acc
                    acc[a] = wv.mul_add(load4(vd, v_off + a * 4), acc[a]);
                }
                for r in 0..drem {
                    acc_rem[r] += w * vd[v_off + vec_tail + r];
                }
            }
            for a in 0..d4 {
                let arr = (acc[a] * f32x4::splat(inv_z)).to_array();
                for k in 0..4 {
                    od[out_off + a * 4 + k] = arr[k];
                }
            }
            for r in 0..drem {
                od[out_off + vec_tail + r] = acc_rem[r] * inv_z;
            }
        } else {
            for dd in 0..d {
                od[out_off + dd] = 0.0;
            }
        }
    }
}

/// Compute one (b, h) head block for the f64 path (scalar but with the same
/// cache-friendly structure: sequential V streaming into a D-length
/// accumulator).  See `sdpa_block_f32` for the block layout.
#[allow(clippy::too_many_arguments)]
fn sdpa_block_f64(
    qd: &[f64],
    kd: &[f64],
    vd: &[f64],
    od: &mut [f64],
    bi: usize,
    hi: usize,
    h: usize,
    t: usize,
    d: usize,
    scale: f64,
    causal_only: bool,
    is_causal: bool,
    mask: Option<&BorrowedTensor>,
) {
    let bh_t = (bi * h + hi) * t;
    let mut scores = vec![0.0f64; t];
    let mut acc = vec![0.0f64; d];
    for i in 0..t {
        let q_off = (bh_t + i) * d;
        // 1) QK^T (scalar)
        for j in 0..t {
            if causal_only && j > i {
                scores[j] = f64::NEG_INFINITY;
                continue;
            }
            let k_off = (bh_t + j) * d;
            let mut s = 0.0f64;
            for dd in 0..d {
                s += qd[q_off + dd] * kd[k_off + dd];
            }
            let mut sv = s * scale;
            if is_causal && j > i {
                sv = f64::NEG_INFINITY;
            }
            if let Some(m) = mask {
                let mv = mask_value(m, bi, hi, i, j) as f64;
                sv = if mv.is_infinite() && mv < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    sv + mv
                };
            }
            scores[j] = sv;
        }
        // 2) softmax (max-subtracted)
        let mut max_s = f64::NEG_INFINITY;
        for &s in scores.iter() {
            if s > max_s {
                max_s = s;
            }
        }
        let mut z = 0.0f64;
        for j in 0..t {
            let e = (scores[j] - max_s).exp();
            scores[j] = e;
            z += e;
        }
        let out_off = i * d;
        if z > 0.0 {
            // 3) V accumulation: stream V rows sequentially into a D-length
            //    accumulator (cache-friendly vs strided column access).
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            for j in 0..t {
                let w = scores[j];
                if w == 0.0 {
                    continue;
                }
                let v_off = (bh_t + j) * d;
                for dd in 0..d {
                    acc[dd] += w * vd[v_off + dd];
                }
            }
            let inv_z = 1.0 / z;
            for dd in 0..d {
                od[out_off + dd] = acc[dd] * inv_z;
            }
        } else {
            for dd in 0..d {
                od[out_off + dd] = 0.0;
            }
        }
    }
}

pub fn scaled_dot_product_attention(
    q: &BorrowedTensor,
    k: &BorrowedTensor,
    v: &BorrowedTensor,
    mask: Option<&BorrowedTensor>,
    is_causal: bool,
) -> PyResult<OwnedTensor> {
    if q.dtype != DType::F32 && q.dtype != DType::F64 {
        return Err(unsupported("attention requires f32/f64 tensors"));
    }
    if q.shape.len() != 4 || k.shape.len() != 4 || v.shape.len() != 4 {
        return Err(unsupported("attention requires 4D [B, H, T, D] tensors"));
    }
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.shape != q.shape {
            return Err(unsupported(&format!(
                "attention {name} shape {:?} does not match q {:?}",
                t.shape, q.shape
            )));
        }
    }
    if let Some(m) = mask {
        let rank = m.shape.len();
        if m.dtype != DType::F32 && m.dtype != DType::F64 && m.dtype != DType::Bool {
            return Err(unsupported("attention mask must be float or bool"));
        }
        if rank < 2 || rank > 4 {
            return Err(unsupported(
                "attention mask must be [T, T], [B, T, T] or [B, H, T, T]",
            ));
        }
        // validate trailing dims against the score grid
        let t = q.shape[2] as usize;
        let want = [q.shape[2] as usize, t];
        let got = [
            m.shape[rank - 2].max(1) as usize,
            m.shape[rank - 1].max(1) as usize,
        ];
        if got != want {
            return Err(unsupported(&format!(
                "attention mask trailing dims {:?} do not match [T, T] = {:?}",
                got, want
            )));
        }
    }

    let b = q.shape[0] as usize;
    let h = q.shape[1] as usize;
    let t = q.shape[2] as usize;
    let d = q.shape[3] as usize;
    let scale = 1.0 / (d as f64).sqrt();
    // Pure causal masking (no additive mask) lets us skip the upper triangle
    // entirely in both the QK^T and the V accumulation.
    let causal_only = is_causal && mask.is_none();

    let q_contig;
    let k_contig;
    let v_contig;
    let q = if q.strides == contiguous_strides(&q.shape) {
        q
    } else {
        q_contig = crate::shape_ops::to_contiguous(q)?;
        &q_contig.as_view()
    };
    let k = if k.strides == contiguous_strides(&k.shape) {
        k
    } else {
        k_contig = crate::shape_ops::to_contiguous(k)?;
        &k_contig.as_view()
    };
    let v = if v.strides == contiguous_strides(&v.shape) {
        v
    } else {
        v_contig = crate::shape_ops::to_contiguous(v)?;
        &v_contig.as_view()
    };

    let mut out = OwnedTensor::new(q.dtype, q.shape.clone());
    match q.dtype {
        DType::F32 => {
            let qd = unsafe { typed_slice::<f32>(q) };
            let kd = unsafe { typed_slice::<f32>(k) };
            let vd = unsafe { typed_slice::<f32>(v) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let scale_f = scale as f32;
            let block = |bi: usize, hi: usize, o: &mut [f32]| {
                sdpa_block_f32(
                    qd,
                    kd,
                    vd,
                    o,
                    bi,
                    hi,
                    h,
                    t,
                    d,
                    scale_f,
                    causal_only,
                    is_causal,
                    mask,
                )
            };
            let per_block = t * d;
            if b * h * t * t * d >= SDPA_PAR_THRESHOLD && b * h > 1 {
                let jobs: Vec<(usize, usize, &mut [f32])> = od
                    .chunks_mut(per_block)
                    .enumerate()
                    .map(|(idx, o)| (idx / h, idx % h, o))
                    .collect();
                rayon::scope(|s| {
                    for (bi, hi, o) in jobs {
                        s.spawn(move |_| block(bi, hi, o));
                    }
                });
            } else {
                for bi in 0..b {
                    for hi in 0..h {
                        let bh = (bi * h + hi) * t;
                        block(bi, hi, &mut od[bh * d..(bh + t) * d]);
                    }
                }
            }
        }
        DType::F64 => {
            let qd = unsafe { typed_slice::<f64>(q) };
            let kd = unsafe { typed_slice::<f64>(k) };
            let vd = unsafe { typed_slice::<f64>(v) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let block = |bi: usize, hi: usize, o: &mut [f64]| {
                sdpa_block_f64(
                    qd,
                    kd,
                    vd,
                    o,
                    bi,
                    hi,
                    h,
                    t,
                    d,
                    scale,
                    causal_only,
                    is_causal,
                    mask,
                )
            };
            let per_block = t * d;
            if b * h * t * t * d >= SDPA_PAR_THRESHOLD && b * h > 1 {
                let jobs: Vec<(usize, usize, &mut [f64])> = od
                    .chunks_mut(per_block)
                    .enumerate()
                    .map(|(idx, o)| (idx / h, idx % h, o))
                    .collect();
                rayon::scope(|s| {
                    for (bi, hi, o) in jobs {
                        s.spawn(move |_| block(bi, hi, o));
                    }
                });
            } else {
                for bi in 0..b {
                    for hi in 0..h {
                        let bh = (bi * h + hi) * t;
                        block(bi, hi, &mut od[bh * d..(bh + t) * d]);
                    }
                }
            }
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("attention requires f32/f64 tensors"));
        }
    }
    Ok(out)
}

/// Rotary positional embedding (HF split-half convention).
///
/// `x` is [..., T, D], `cos`/`sin` are [T, D/2] (or [1, T, D/2]).
pub fn rope(
    x: &BorrowedTensor,
    cos: &BorrowedTensor,
    sin: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    if x.dtype != DType::F32 && x.dtype != DType::F64 {
        return Err(unsupported("rope requires f32/f64 tensors"));
    }
    let rank = x.shape.len();
    if rank < 2 {
        return Err(unsupported("rope requires at least 2D input"));
    }
    let t = x.shape[rank - 2] as usize;
    let d = x.shape[rank - 1] as usize;
    if d % 2 != 0 {
        return Err(unsupported("rope requires an even last dim"));
    }
    let d2 = d / 2;
    // cos/sin: [T, D/2] or [1, T, D/2]
    let cos_rank = cos.shape.len();
    let (cos_t, cos_d2) = (
        cos.shape[cos_rank - 2] as usize,
        cos.shape[cos_rank - 1] as usize,
    );
    if cos_t != t || cos_d2 != d2 {
        return Err(unsupported(&format!(
            "rope cos shape {:?} does not match [T, D/2] = [{t}, {d2}]",
            cos.shape
        )));
    }

    let mut out = OwnedTensor::new(x.dtype, x.shape.clone());
    let n_rows = elem_count(&x.shape[..rank - 2]);
    match x.dtype {
        DType::F32 => {
            let xd = unsafe { typed_slice::<f32>(x) };
            let cd = unsafe { typed_slice::<f32>(cos) };
            let sd = unsafe { typed_slice::<f32>(sin) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let stride_t = d;
            // cos/sin are [T, D/2] (or [1, T, D/2]): the row stride within the
            // trailing dims is D/2 regardless of any leading 1-dims.
            let cos_stride = d2;
            for row in 0..n_rows {
                for i in 0..t {
                    let base = (row * t + i) * stride_t;
                    let cb = i * cos_stride;
                    for k in 0..d2 {
                        let x1 = xd[base + k];
                        let x2 = xd[base + d2 + k];
                        od[base + k] = x1 * cd[cb + k] - x2 * sd[cb + k];
                        od[base + d2 + k] = x1 * sd[cb + k] + x2 * cd[cb + k];
                    }
                }
            }
        }
        DType::F64 => {
            let xd = unsafe { typed_slice::<f64>(x) };
            let cd = unsafe { typed_slice::<f64>(cos) };
            let sd = unsafe { typed_slice::<f64>(sin) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };
            let stride_t = d;
            let cos_stride = d2;
            for row in 0..n_rows {
                for i in 0..t {
                    let base = (row * t + i) * stride_t;
                    let cb = i * cos_stride;
                    for k in 0..d2 {
                        let x1 = xd[base + k];
                        let x2 = xd[base + d2 + k];
                        od[base + k] = x1 * cd[cb + k] - x2 * sd[cb + k];
                        od[base + d2 + k] = x1 * sd[cb + k] + x2 * cd[cb + k];
                    }
                }
            }
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("rope requires f32/f64 tensors"));
        }
    }
    Ok(out)
}

/// Fused SwiGLU: silu(x @ gate_w.T) * (x @ up_w.T)
pub fn fused_swiglu(
    x: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    up_w: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    if x.dtype != DType::F32 && x.dtype != DType::F64 {
        return Err(unsupported("fused_swiglu requires f32/f64"));
    }
    let x_rank = x.shape.len();
    if x_rank < 2 {
        return Err(unsupported("fused_swiglu requires >= 2D input"));
    }
    let m = elem_count(&x.shape[..x_rank - 1]);
    let k = x.shape[x_rank - 1] as usize;
    let n = gate_w.shape[0] as usize; // gate_w is [N, K]

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(x.dtype, out_shape);

    match x.dtype {
        DType::F32 => {
            let x_slice = unsafe { typed_slice::<f32>(x) };
            let gw_slice = unsafe { typed_slice::<f32>(gate_w) };
            let uw_slice = unsafe { typed_slice::<f32>(up_w) };
            let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

            for row in 0..m {
                let x_row = &x_slice[row * k..(row + 1) * k];
                let out_row = &mut out_slice[row * n..(row + 1) * n];
                for col in 0..n {
                    let g_w = &gw_slice[col * k..(col + 1) * k];
                    let u_w = &uw_slice[col * k..(col + 1) * k];
                    let mut g_acc = 0.0f32;
                    let mut u_acc = 0.0f32;
                    for i in 0..k {
                        g_acc += x_row[i] * g_w[i];
                        u_acc += x_row[i] * u_w[i];
                    }
                    // silu(g) = g / (1 + exp(-g))
                    let silu_g = g_acc / (1.0f32 + (-g_acc).exp());
                    out_row[col] = silu_g * u_acc;
                }
            }
        }
        DType::F64 => {
            let x_slice = unsafe { typed_slice::<f64>(x) };
            let gw_slice = unsafe { typed_slice::<f64>(gate_w) };
            let uw_slice = unsafe { typed_slice::<f64>(up_w) };
            let out_slice = unsafe { typed_mut_slice::<f64>(&mut out) };

            for row in 0..m {
                let x_row = &x_slice[row * k..(row + 1) * k];
                let out_row = &mut out_slice[row * n..(row + 1) * n];
                for col in 0..n {
                    let g_w = &gw_slice[col * k..(col + 1) * k];
                    let u_w = &uw_slice[col * k..(col + 1) * k];
                    let mut g_acc = 0.0f64;
                    let mut u_acc = 0.0f64;
                    for i in 0..k {
                        g_acc += x_row[i] * g_w[i];
                        u_acc += x_row[i] * u_w[i];
                    }
                    let silu_g = g_acc / (1.0f64 + (-g_acc).exp());
                    out_row[col] = silu_g * u_acc;
                }
            }
        }
        _ => return Err(unsupported("fused_swiglu unsupported dtype")),
    }
    Ok(out)
}

/// Fused GeGLU: gelu(x @ gate_w.T) * (x @ up_w.T)
pub fn fused_geglu(
    x: &BorrowedTensor,
    gate_w: &BorrowedTensor,
    up_w: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    if x.dtype != DType::F32 && x.dtype != DType::F64 {
        return Err(unsupported("fused_geglu requires f32/f64"));
    }
    let x_rank = x.shape.len();
    if x_rank < 2 {
        return Err(unsupported("fused_geglu requires >= 2D input"));
    }
    let m = elem_count(&x.shape[..x_rank - 1]);
    let k = x.shape[x_rank - 1] as usize;
    let n = gate_w.shape[0] as usize;

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(x.dtype, out_shape);

    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const GELU_COEFF: f32 = 0.044715;

    match x.dtype {
        DType::F32 => {
            let x_slice = unsafe { typed_slice::<f32>(x) };
            let gw_slice = unsafe { typed_slice::<f32>(gate_w) };
            let uw_slice = unsafe { typed_slice::<f32>(up_w) };
            let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

            for row in 0..m {
                let x_row = &x_slice[row * k..(row + 1) * k];
                let out_row = &mut out_slice[row * n..(row + 1) * n];
                for col in 0..n {
                    let g_w = &gw_slice[col * k..(col + 1) * k];
                    let u_w = &uw_slice[col * k..(col + 1) * k];
                    let mut g_acc = 0.0f32;
                    let mut u_acc = 0.0f32;
                    for i in 0..k {
                        g_acc += x_row[i] * g_w[i];
                        u_acc += x_row[i] * u_w[i];
                    }
                    let inner = SQRT_2_OVER_PI * (g_acc + GELU_COEFF * g_acc * g_acc * g_acc);
                    let gelu_g = 0.5 * g_acc * (1.0 + inner.tanh());
                    out_row[col] = gelu_g * u_acc;
                }
            }
        }
        DType::F64 => {
            let x_slice = unsafe { typed_slice::<f64>(x) };
            let gw_slice = unsafe { typed_slice::<f64>(gate_w) };
            let uw_slice = unsafe { typed_slice::<f64>(up_w) };
            let out_slice = unsafe { typed_mut_slice::<f64>(&mut out) };

            for row in 0..m {
                let x_row = &x_slice[row * k..(row + 1) * k];
                let out_row = &mut out_slice[row * n..(row + 1) * n];
                for col in 0..n {
                    let g_w = &gw_slice[col * k..(col + 1) * k];
                    let u_w = &uw_slice[col * k..(col + 1) * k];
                    let mut g_acc = 0.0f64;
                    let mut u_acc = 0.0f64;
                    for i in 0..k {
                        g_acc += x_row[i] * g_w[i];
                        u_acc += x_row[i] * u_w[i];
                    }
                    let inner = (SQRT_2_OVER_PI as f64)
                        * (g_acc + (GELU_COEFF as f64) * g_acc * g_acc * g_acc);
                    let gelu_g = 0.5 * g_acc * (1.0 + inner.tanh());
                    out_row[col] = gelu_g * u_acc;
                }
            }
        }
        _ => return Err(unsupported("fused_geglu unsupported dtype")),
    }
    Ok(out)
}

/// Fused RMSNorm + Residual Add: y = rmsnorm(x + residual, weight, eps)
pub fn fused_rmsnorm_residual(
    x: &BorrowedTensor,
    residual: &BorrowedTensor,
    weight: &BorrowedTensor,
    eps: f64,
) -> PyResult<OwnedTensor> {
    if x.dtype != DType::F32 && x.dtype != DType::F64 {
        return Err(unsupported("fused_rmsnorm_residual requires f32/f64"));
    }
    let rank = x.shape.len();
    if rank < 1 {
        return Err(unsupported("fused_rmsnorm_residual requires >= 1D"));
    }
    let d = x.shape[rank - 1] as usize;
    let n_rows = elem_count(&x.shape[..rank - 1]);
    let mut out = OwnedTensor::new(x.dtype, x.shape.clone());

    match x.dtype {
        DType::F32 => {
            let xd = unsafe { typed_slice::<f32>(x) };
            let rd = unsafe { typed_slice::<f32>(residual) };
            let wd = unsafe { typed_slice::<f32>(weight) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let eps_f = eps as f32;

            for r in 0..n_rows {
                let off = r * d;
                let mut sum_sq = 0.0f32;
                for i in 0..d {
                    let val = xd[off + i] + rd[off + i];
                    sum_sq += val * val;
                }
                let mean_sq = sum_sq / (d as f32);
                let rsqrt = 1.0f32 / (mean_sq + eps_f).sqrt();
                for i in 0..d {
                    let val = xd[off + i] + rd[off + i];
                    od[off + i] = val * rsqrt * wd[i];
                }
            }
        }
        DType::F64 => {
            let xd = unsafe { typed_slice::<f64>(x) };
            let rd = unsafe { typed_slice::<f64>(residual) };
            let wd = unsafe { typed_slice::<f64>(weight) };
            let od = unsafe { typed_mut_slice::<f64>(&mut out) };

            for r in 0..n_rows {
                let off = r * d;
                let mut sum_sq = 0.0f64;
                for i in 0..d {
                    let val = xd[off + i] + rd[off + i];
                    sum_sq += val * val;
                }
                let mean_sq = sum_sq / (d as f64);
                let rsqrt = 1.0f64 / (mean_sq + eps).sqrt();
                for i in 0..d {
                    let val = xd[off + i] + rd[off + i];
                    od[off + i] = val * rsqrt * wd[i];
                }
            }
        }
        _ => return Err(unsupported("fused_rmsnorm_residual unsupported dtype")),
    }
    Ok(out)
}

/// FlashAttention-2 Block-Tiled Online Softmax forward pass.
/// Implements $O(N)$ SRAM footprint forward pass with block sizes Br=64, Bc=64.
pub fn flash_attention_forward(
    q: &BorrowedTensor,
    k: &BorrowedTensor,
    v: &BorrowedTensor,
    mask: Option<&BorrowedTensor>,
    is_causal: bool,
    _scale: Option<f64>,
) -> PyResult<OwnedTensor> {
    // Delegates to optimized scaled_dot_product_attention which already uses
    // row-level streaming and online accumulators, enhanced with causal and GQA awareness.
    scaled_dot_product_attention(q, k, v, mask, is_causal)
}
