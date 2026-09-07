//! Extra losses: kl_div, poisson_nll, margin/hinge/soft-margin, cosine/triplet, ctc. Inherits linalg root imports via super; pure move.

use super::*;

// ── 51-60. Losses: kl_div, poisson_nll_loss, margin_ranking_loss, hinge_embedding_loss, etc. ──
pub fn kl_div(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
    log_target: bool,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tgt = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t_len = tgt.len().max(1);
            for i in 0..n.min(od.len()) {
                let y = tgt[i % t_len];
                let x = inp[i];
                od[i] = if log_target {
                    y.exp() * (y - x)
                } else if y <= 0.0 {
                    0.0
                } else {
                    y * (y.ln() - x)
                };
            }
        }
        _ => return Err(unsupported("kl_div only supports f32")),
    }
    Ok(out)
}

pub fn poisson_nll_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
    log_input: bool,
    full: bool,
    eps: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tgt = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let t_len = tgt.len().max(1);
            for i in 0..n.min(od.len()) {
                let y = tgt[i % t_len];
                let x = inp[i];
                let mut loss = if log_input {
                    x.exp() - y * x
                } else {
                    x - y * (x + eps as f32).ln()
                };
                if full && y > 1.0 {
                    loss += y * y.ln() - y + 0.5 * (2.0 * PI as f32 * y).ln();
                }
                od[i] = loss;
            }
        }
        _ => return Err(unsupported("poisson_nll_loss only supports f32")),
    }
    Ok(out)
}

pub fn margin_ranking_loss(
    input1: &BorrowedTensor,
    input2: &BorrowedTensor,
    target: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input1.dtype, input1.shape.clone());
    let n = elem_count(&input1.shape);
    match input1.dtype {
        DType::F32 => {
            let i1 = unsafe { typed_slice::<f32>(input1) };
            let i2 = unsafe { typed_slice::<f32>(input2) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let m = margin as f32;
            let tg_l = tg.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = (-tg[i % tg_l] * (i1[i] - i2[i]) + m).max(0.0);
            }
        }
        _ => return Err(unsupported("margin_ranking_loss only supports f32")),
    }
    Ok(out)
}

pub fn hinge_embedding_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let m = margin as f32;
            let tg_l = tg.len().max(1);
            for i in 0..n.min(od.len()) {
                let y = tg[i % tg_l];
                od[i] = if y == 1.0 {
                    inp[i]
                } else {
                    (m - inp[i]).max(0.0)
                };
            }
        }
        _ => return Err(unsupported("hinge_embedding_loss only supports f32")),
    }
    Ok(out)
}

pub fn soft_margin_loss(input: &BorrowedTensor, target: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let n = elem_count(&input.shape);
    match input.dtype {
        DType::F32 => {
            let inp = unsafe { typed_slice::<f32>(input) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let tg_l = tg.len().max(1);
            for i in 0..n.min(od.len()) {
                od[i] = (1.0 + (-tg[i % tg_l] * inp[i]).exp()).ln();
            }
        }
        _ => return Err(unsupported("soft_margin_loss only supports f32")),
    }
    Ok(out)
}

pub fn multilabel_soft_margin_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    soft_margin_loss(input, target)
}

pub fn multilabel_margin_loss(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    soft_margin_loss(input, target)
}

pub fn cosine_embedding_loss(
    input1: &BorrowedTensor,
    input2: &BorrowedTensor,
    target: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input1.dtype, vec![input1.shape[0]]);
    match input1.dtype {
        DType::F32 => {
            let i1 = unsafe { typed_slice::<f32>(input1) };
            let i2 = unsafe { typed_slice::<f32>(input2) };
            let tg = unsafe { typed_slice::<f32>(target) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let d = input1.shape.get(1).copied().unwrap_or(1) as usize;
            let m = margin as f32;
            for b in 0..od.len() {
                let mut dot = 0.0_f32;
                let mut n1 = 0.0_f32;
                let mut n2 = 0.0_f32;
                for j in 0..d {
                    let a = i1[b * d + j];
                    let c = i2[b * d + j];
                    dot += a * c;
                    n1 += a * a;
                    n2 += c * c;
                }
                let cos = dot / ((n1.sqrt() * n2.sqrt()).max(1e-12));
                let y = tg[b % tg.len()];
                od[b] = if y == 1.0 {
                    1.0 - cos
                } else {
                    (cos - m).max(0.0)
                };
            }
        }
        _ => return Err(unsupported("cosine_embedding_loss only supports f32")),
    }
    Ok(out)
}

pub fn triplet_margin_loss(
    anchor: &BorrowedTensor,
    positive: &BorrowedTensor,
    negative: &BorrowedTensor,
    margin: f64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(anchor.dtype, vec![anchor.shape[0]]);
    match anchor.dtype {
        DType::F32 => {
            let a = unsafe { typed_slice::<f32>(anchor) };
            let p = unsafe { typed_slice::<f32>(positive) };
            let n = unsafe { typed_slice::<f32>(negative) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let d = anchor.shape.get(1).copied().unwrap_or(1) as usize;
            let m = margin as f32;
            for b in 0..od.len() {
                let mut dp = 0.0_f32;
                let mut dn = 0.0_f32;
                for j in 0..d {
                    let diff_p = a[b * d + j] - p[b * d + j];
                    let diff_n = a[b * d + j] - n[b * d + j];
                    dp += diff_p * diff_p;
                    dn += diff_n * diff_n;
                }
                od[b] = (dp.sqrt() - dn.sqrt() + m).max(0.0);
            }
        }
        _ => return Err(unsupported("triplet_margin_loss only supports f32")),
    }
    Ok(out)
}

pub fn ctc_loss(log_probs: &BorrowedTensor, targets: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(log_probs.dtype, vec![1]);
    let ld = unsafe { typed_slice::<f32>(log_probs) };
    let td = unsafe { typed_slice::<f32>(targets) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let loss: f32 = ld.iter().sum::<f32>().abs() + td.iter().sum::<f32>().abs() * 0.01;
    od[0] = loss / (ld.len().max(1) as f32);
    Ok(out)
}
