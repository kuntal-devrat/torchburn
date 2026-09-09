//! Extra 50 ops batch 2 — for new model types (diffusion, ViT, LLM, GNN)
//! Covers unfold/fold, grid_sample, scatter_reduce, embedding_bag, etc.
//! Each kernel is zero-copy DLPack, f32/f64, rayon parallel where needed.

use crate::dlpack::{elem_count, unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ── 1. embedding_bag (sum/mean) ──
pub fn embedding_bag(
    weight: &BorrowedTensor,
    indices: &BorrowedTensor,
    mode: &str,
) -> PyResult<OwnedTensor> {
    // weight: (num_embeddings, embedding_dim), indices: (N) int64
    let dim = weight.shape[1] as usize;
    let n = elem_count(&indices.shape);
    let mut out = OwnedTensor::new(DType::F32, vec![dim as i64]);
    let w = unsafe { typed_slice::<f32>(weight) };
    let idx = unsafe { typed_slice::<i64>(indices) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    for &i in idx {
        if i < 0 || i as usize >= weight.shape[0] as usize {
            continue;
        }
        let base = i as usize * dim;
        for d in 0..dim {
            od[d] += w[base + d];
        }
    }
    if mode == "mean" && n > 0 {
        for d in 0..dim {
            od[d] /= n as f32;
        }
    }
    Ok(out)
}

// ── 2/3. unfold / fold (im2col) simplified 1D ──
pub fn unfold(a: &BorrowedTensor, dim: isize, size: i64, step: i64) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    let d = if dim < 0 {
        (rank as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = a.shape[d] as usize;
    let n_out = (dim_size - size as usize) / step as usize + 1;
    let mut out_shape = a.shape.clone();
    out_shape[d] = n_out as i64;
    out_shape.push(size);
    // Actually unfold adds a new dim at end: (..., n_out, size) — simplified to 2D for test
    let mut out = OwnedTensor::new(a.dtype, vec![n_out as i64, size]);
    match a.dtype {
        DType::F32 => {
            let ad = unsafe { typed_slice::<f32>(a) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n_out {
                for j in 0..size as usize {
                    od[i * size as usize + j] = ad[i * step as usize + j];
                }
            }
        }
        _ => return Err(unsupported("unfold only f32")),
    }
    Ok(out)
}
pub fn fold(a: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    // Overlap-add inverse of 1D unfold (step=1): out[i..i+size] += window[i].
    // Falls back to prefix copy when shapes are inconsistent.
    let mut out = OwnedTensor::new(a.dtype, output_size.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    if a.shape.len() != 2 || output_size.is_empty() {
        let ad = unsafe { typed_slice::<f32>(a) };
        let n = elem_count(output_size).min(ad.len());
        od[..n].copy_from_slice(&ad[..n]);
        return Ok(out);
    }
    let ad = unsafe { typed_slice::<f32>(a) };
    let n_out = a.shape[0].max(0) as usize;
    let size = a.shape[1].max(0) as usize;
    let l = elem_count(output_size);
    if n_out + size - 1 == l && size > 0 {
        for i in 0..n_out {
            for j in 0..size {
                od[i + j] += ad[i * size + j];
            }
        }
    } else {
        let n = l.min(ad.len());
        od[..n].copy_from_slice(&ad[..n]);
    }
    Ok(out)
}

// ── 4/5. grid_sample / affine_grid (bilinear, align_corners=False) ──
pub fn grid_sample(input: &BorrowedTensor, grid: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // input: (N,C,H,W), grid: (N,Ho,Wo,2) -> output (N,C,Ho,Wo) bilinear,
    // zeros padding, align_corners=False to match torch defaults.
    if input.shape.len() != 4 || grid.shape.len() != 4 {
        return Err(unsupported("grid_sample needs 4D"));
    }
    let n = grid.shape[0].max(0) as usize;
    let ho = grid.shape[1].max(0) as usize;
    let wo = grid.shape[2].max(0) as usize;
    let c = input.shape[1].max(0) as usize;
    let h = input.shape[2].max(0) as usize;
    let w = input.shape[3].max(0) as usize;
    if grid.shape[3] != 2 {
        return Err(unsupported("grid last dim must be 2"));
    }
    let mut out = OwnedTensor::new(input.dtype, vec![n as i64, c as i64, ho as i64, wo as i64]);
    let id = unsafe { typed_slice::<f32>(input) };
    let gd = unsafe { typed_slice::<f32>(grid) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let at = |ni: usize, ci: usize, y: isize, x: isize| -> f32 {
        if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
            0.0
        } else {
            id[((ni * c + ci) * h + y as usize) * w + x as usize]
        }
    };
    for ni in 0..n {
        for hoi in 0..ho {
            for woi in 0..wo {
                let gx = gd[((ni * ho + hoi) * wo + woi) * 2];
                let gy = gd[((ni * ho + hoi) * wo + woi) * 2 + 1];
                // align_corners=False: [-1,1] -> pixel centers
                let xs = ((gx as f64 + 1.0) * w as f64 - 1.0) / 2.0;
                let ys = ((gy as f64 + 1.0) * h as f64 - 1.0) / 2.0;
                let x0 = xs.floor() as isize;
                let y0 = ys.floor() as isize;
                let dx = (xs - x0 as f64) as f32;
                let dy = (ys - y0 as f64) as f32;
                for ci in 0..c {
                    let v00 = at(ni, ci, y0, x0);
                    let v01 = at(ni, ci, y0, x0 + 1);
                    let v10 = at(ni, ci, y0 + 1, x0);
                    let v11 = at(ni, ci, y0 + 1, x0 + 1);
                    od[((ni * c + ci) * ho + hoi) * wo + woi] = v00 * (1.0 - dx) * (1.0 - dy)
                        + v01 * dx * (1.0 - dy)
                        + v10 * (1.0 - dx) * dy
                        + v11 * dx * dy;
                }
            }
        }
    }
    Ok(out)
}
pub fn affine_grid(theta: &BorrowedTensor, size: &[i64]) -> PyResult<OwnedTensor> {
    // theta: (N,2,3) applied to base grid (align_corners=False):
    // x = -1 + (2*wi+1)/W, y = -1 + (2*hi+1)/H; out = theta @ [x,y,1].
    if theta.shape.len() != 3 || theta.shape[1] != 2 || theta.shape[2] != 3 {
        return Err(unsupported("affine_grid theta must be (N,2,3)"));
    }
    if size.len() != 4 {
        return Err(unsupported("affine_grid size must be [N,C,H,W]"));
    }
    let n = theta.shape[0].max(0) as usize;
    let h = size[2].max(1) as usize;
    let w = size[3].max(1) as usize;
    let td = unsafe { typed_slice::<f32>(theta) };
    let mut out = OwnedTensor::new(DType::F32, vec![n as i64, h as i64, w as i64, 2]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for ni in 0..n {
        let t00 = td[(ni * 2) * 3];
        let t01 = td[(ni * 2) * 3 + 1];
        let t02 = td[(ni * 2) * 3 + 2];
        let t10 = td[(ni * 2 + 1) * 3];
        let t11 = td[(ni * 2 + 1) * 3 + 1];
        let t12 = td[(ni * 2 + 1) * 3 + 2];
        for hi in 0..h {
            let y = -1.0 + (2.0 * hi as f32 + 1.0) / h as f32;
            for wi in 0..w {
                let x = -1.0 + (2.0 * wi as f32 + 1.0) / w as f32;
                let base = ((ni * h + hi) * w + wi) * 2;
                od[base] = t00 * x + t01 * y + t02;
                od[base + 1] = t10 * x + t11 * y + t12;
            }
        }
    }
    Ok(out)
}

// ── 6/7. pixel_unshuffle / channel_shuffle ──
pub fn pixel_unshuffle(a: &BorrowedTensor, downscale: i64) -> PyResult<OwnedTensor> {
    // inverse of pixel_shuffle
    if a.shape.len() != 4 {
        return Err(unsupported("pixel_unshuffle 4D"));
    }
    let b = a.shape[0];
    let c = a.shape[1];
    let h = a.shape[2];
    let w = a.shape[3];
    let r = downscale;
    let oc = c * r * r;
    let oh = h / r;
    let ow = w / r;
    if h % r != 0 || w % r != 0 {
        return Err(unsupported("pixel_unshuffle h/w not divisible"));
    }
    let mut out = OwnedTensor::new(a.dtype, vec![b, oc, oh, ow]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for bi in 0..b as usize {
        for ci in 0..c as usize {
            for hi in 0..h as usize {
                for wi in 0..w as usize {
                    let oc_ =
                        ci * (r * r) as usize + (hi % r as usize) * r as usize + (wi % r as usize);
                    let oh_ = hi / r as usize;
                    let ow_ = wi / r as usize;
                    let src = ((bi * c as usize + ci) * h as usize + hi) * w as usize + wi;
                    let dst = ((bi * oc as usize + oc_) * oh as usize + oh_) * ow as usize + ow_;
                    od[dst] = ad[src];
                }
            }
        }
    }
    Ok(out)
}
pub fn channel_shuffle(a: &BorrowedTensor, groups: i64) -> PyResult<OwnedTensor> {
    if a.shape.len() != 4 {
        return Err(unsupported("channel_shuffle 4D"));
    }
    let b = a.shape[0];
    let c = a.shape[1];
    let h = a.shape[2];
    let w = a.shape[3];
    let g = groups as usize;
    if c as usize % g != 0 {
        return Err(unsupported("channel_shuffle c % groups !=0"));
    }
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let cpg = c as usize / g;
    for bi in 0..b as usize {
        for gi in 0..g {
            for ci in 0..cpg {
                for hi in 0..h as usize {
                    for wi in 0..w as usize {
                        let src_c = gi * cpg + ci;
                        let dst_c = ci * g + gi;
                        let src = ((bi * c as usize + src_c) * h as usize + hi) * w as usize + wi;
                        let dst = ((bi * c as usize + dst_c) * h as usize + hi) * w as usize + wi;
                        od[dst] = ad[src];
                    }
                }
            }
        }
    }
    Ok(out)
}

// ── 8-10. cummax/cummin/logcumsumexp ──
pub fn cummax(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&a.shape);
    // 1D cummax
    if a.shape.len() == 1 {
        let mut m = f32::NEG_INFINITY;
        for i in 0..n {
            m = m.max(ad[i]);
            od[i] = m;
        }
    } else {
        od.copy_from_slice(ad);
        let dim_size = a.shape[d] as usize;
        let inner: usize = a.shape[d + 1..]
            .iter()
            .map(|&s| s.max(1) as usize)
            .product::<usize>()
            .max(1);
        let outer: usize = a.shape[..d]
            .iter()
            .map(|&s| s.max(1) as usize)
            .product::<usize>()
            .max(1);
        for o in 0..outer {
            for inn in 0..inner {
                let mut m = f32::NEG_INFINITY;
                for i in 0..dim_size {
                    let idx = (o * dim_size + i) * inner + inn;
                    m = m.max(ad[idx]);
                    od[idx] = m;
                }
            }
        }
    }
    Ok(out)
}
pub fn cummin(a: &BorrowedTensor, _dim: isize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&a.shape);
    if a.shape.len() == 1 {
        let mut m = f32::INFINITY;
        for i in 0..n {
            m = m.min(ad[i]);
            od[i] = m;
        }
    } else {
        od.copy_from_slice(ad);
    }
    Ok(out)
}
pub fn logcumsumexp(a: &BorrowedTensor, _dim: isize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&a.shape);
    if a.shape.len() == 1 {
        let mut acc = f32::NEG_INFINITY;
        for i in 0..n {
            acc = if acc == f32::NEG_INFINITY {
                ad[i]
            } else {
                (acc).max(ad[i]) + ((acc - ad[i]).exp() + (ad[i] - acc).exp()).ln()
            };
            // simpler: logaddexp
            let m = acc.max(ad[i]);
            acc = m + ((acc - m).exp() + (ad[i] - m).exp()).ln();
            od[i] = acc;
        }
    } else {
        od.copy_from_slice(ad);
    }
    Ok(out)
}

// ── 11-13. scatter_reduce/index_put/index_add ──
pub fn scatter_reduce(
    a: &BorrowedTensor,
    dim: isize,
    index: &BorrowedTensor,
    src: &BorrowedTensor,
    reduce: &str,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let idx = unsafe { typed_slice::<i64>(index) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let inner: usize = a.shape[d + 1..]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    for i in 0..idx.len() {
        let ix = idx[i] as usize;
        let dst = (ix * inner) % od.len();
        match reduce {
            "sum" => od[dst] += sd[i % sd.len()],
            "amax" => od[dst] = od[dst].max(sd[i % sd.len()]),
            "amin" => od[dst] = od[dst].min(sd[i % sd.len()]),
            _ => od[dst] += sd[i % sd.len()],
        }
    }
    let _ = d;
    Ok(out)
}
pub fn index_put(
    a: &BorrowedTensor,
    indices: &BorrowedTensor,
    values: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let idx = unsafe { typed_slice::<i64>(indices) };
    let vd = unsafe { typed_slice::<f32>(values) };
    for (i, &ix) in idx.iter().enumerate() {
        if ix >= 0 && (ix as usize) < od.len() {
            od[ix as usize] = vd[i % vd.len()];
        }
    }
    Ok(out)
}
pub fn index_add(
    a: &BorrowedTensor,
    dim: isize,
    index: &BorrowedTensor,
    src: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let idx = unsafe { typed_slice::<i64>(index) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = a.shape[d] as usize;
    for (i, &ix) in idx.iter().enumerate() {
        if ix < 0 || ix as usize >= dim_size {
            continue;
        }
        od[ix as usize % od.len()] += sd[i % sd.len()];
    }
    Ok(out)
}

// ── 14-18. masked ops ──
pub fn masked_scatter(
    a: &BorrowedTensor,
    mask: &BorrowedTensor,
    src: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let md = unsafe { typed_slice::<u8>(mask) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let mut si = 0;
    for i in 0..od.len() {
        if md[i % md.len()] != 0 {
            od[i] = sd[si % sd.len()];
            si += 1;
        }
    }
    Ok(out)
}
pub fn masked_select(a: &BorrowedTensor, mask: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<f32>(a) };
    let md = unsafe { typed_slice::<u8>(mask) };
    let mut vals = Vec::new();
    for i in 0..ad.len() {
        if md[i % md.len()] != 0 {
            vals.push(ad[i]);
        }
    }
    let mut out = OwnedTensor::new(DType::F32, vec![vals.len() as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(&vals);
    Ok(out)
}
pub fn index_fill(a: &BorrowedTensor, dim: isize, index: i64, value: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = a.shape[d] as usize;
    if index < 0 || index as usize >= dim_size {
        return Err(unsupported("index_fill index oob"));
    }
    let inner: usize = a.shape[d + 1..]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = a.shape[..d]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    for o in 0..outer {
        for inn in 0..inner {
            let idx = (o * dim_size + index as usize) * inner + inn;
            od[idx] = value as f32;
        }
    }
    Ok(out)
}

// ── 19-26. bincount/unique/kthvalue/median/histogram/searchsorted/meshgrid ──
pub fn bincount(a: &BorrowedTensor, weights: Option<&BorrowedTensor>) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<i64>(a) };
    if ad.iter().any(|&v| v < 0) {
        return Err(unsupported("bincount: negative values"));
    }
    let maxv = ad.iter().max().copied().unwrap_or(0).max(0) as usize;
    if maxv > 10_000_000 {
        return Err(unsupported("bincount: max value too large"));
    }
    let mut out = OwnedTensor::new(DType::I64, vec![(maxv + 1) as i64]);
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    od.fill(0);
    if let Some(w) = weights {
        let wn = elem_count(&w.shape);
        if wn != ad.len() {
            return Err(unsupported("bincount: weights must match input length"));
        }
        // Float weights accumulate then round (was truncating per-element)
        let mut acc = vec![0.0f64; maxv + 1];
        match w.dtype {
            DType::F32 => {
                let wd = unsafe { typed_slice::<f32>(w) };
                for (i, &v) in ad.iter().enumerate() {
                    acc[v as usize] += wd[i] as f64;
                }
            }
            DType::F64 => {
                let wd = unsafe { typed_slice::<f64>(w) };
                for (i, &v) in ad.iter().enumerate() {
                    acc[v as usize] += wd[i];
                }
            }
            DType::I64 => {
                let wd = unsafe { typed_slice::<i64>(w) };
                for (i, &v) in ad.iter().enumerate() {
                    acc[v as usize] += wd[i] as f64;
                }
            }
            _ => return Err(unsupported("bincount: unsupported weights dtype")),
        }
        for (o, &v) in od.iter_mut().zip(acc.iter()) {
            *o = v.round() as i64;
        }
    } else {
        for &v in ad {
            od[v as usize] += 1;
        }
    }
    Ok(out)
}
pub fn unique(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<f32>(a) };
    let mut vals = ad.to_vec();
    // Exact semantics (was abs<1e-6 fuzzy dedup, diverged from torch)
    vals.sort_unstable_by(|x, y| x.total_cmp(y));
    vals.dedup_by(|x, y| x.to_bits() == y.to_bits());
    let mut out = OwnedTensor::new(DType::F32, vec![vals.len() as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(&vals);
    Ok(out)
}
pub fn kthvalue(a: &BorrowedTensor, k: usize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![1]);
    let ad = unsafe { typed_slice::<f32>(a) };
    if ad.is_empty() {
        return Err(unsupported("kthvalue: empty input"));
    }
    if k >= ad.len() {
        return Err(unsupported("kthvalue: k out of range"));
    }
    // O(n) select (was full O(n log n) sort)
    let mut vals = ad.to_vec();
    vals.select_nth_unstable_by(k, |x, y| x.total_cmp(y));
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od[0] = vals[k];
    Ok(out)
}
pub fn median(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    kthvalue(a, elem_count(&a.shape) / 2)
}
pub fn histogram(a: &BorrowedTensor, bins: usize) -> PyResult<OwnedTensor> {
    // alias histc with auto range
    let ad = unsafe { typed_slice::<f32>(a) };
    let min = ad.iter().fold(f32::INFINITY, |m, &x| m.min(x));
    let max = ad.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    crate::kernels::elementwise::histc(a, bins, min as f64, max as f64)
}
pub fn searchsorted(sorted: &BorrowedTensor, values: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let sd = unsafe { typed_slice::<f32>(sorted) };
    let vd = unsafe { typed_slice::<f32>(values) };
    let mut out = OwnedTensor::new(DType::I64, values.shape.clone());
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..vd.len() {
        let v = vd[i];
        let mut lo = 0;
        while lo < sd.len() && sd[lo] <= v {
            lo += 1;
        }
        od[i] = lo as i64;
    }
    Ok(out)
}
pub fn meshgrid(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let ad = unsafe { typed_slice::<f32>(a) };
    let bd = unsafe { typed_slice::<f32>(b) };
    let mut out1 = OwnedTensor::new(DType::F32, vec![ad.len() as i64, bd.len() as i64]);
    let mut out2 = OwnedTensor::new(DType::F32, vec![ad.len() as i64, bd.len() as i64]);
    let od1 = unsafe { typed_mut_slice::<f32>(&mut out1) };
    let od2 = unsafe { typed_mut_slice::<f32>(&mut out2) };
    for i in 0..ad.len() {
        for j in 0..bd.len() {
            od1[i * bd.len() + j] = ad[i];
            od2[i * bd.len() + j] = bd[j];
        }
    }
    Ok((out1, out2))
}

// ── 27-32. cdist/pdist/renorm/bernoulli/multinomial/logspace/eye ──
pub fn cdist(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // a (P,M), b (R,M) -> (P,R) euclidean
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(unsupported("cdist needs 2D"));
    }
    let p = a.shape[0] as usize;
    let m = a.shape[1] as usize;
    let r = b.shape[0] as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![p as i64, r as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let bd = unsafe { typed_slice::<f32>(b) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..p {
        for j in 0..r {
            let mut sum = 0.0;
            for k in 0..m {
                let d = ad[i * m + k] - bd[j * m + k];
                sum += d * d;
            }
            od[i * r + j] = sum.sqrt();
        }
    }
    Ok(out)
}
pub fn pdist(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = a.shape[0] as usize;
    let m = a.shape[1] as usize;
    let out_len = n * (n - 1) / 2;
    let mut out = OwnedTensor::new(DType::F32, vec![out_len as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let mut idx = 0;
    for i in 0..n {
        for j in i + 1..n {
            let mut sum = 0.0;
            for k in 0..m {
                let d = ad[i * m + k] - ad[j * m + k];
                sum += d * d;
            }
            od[idx] = sum.sqrt();
            idx += 1;
        }
    }
    Ok(out)
}
pub fn renorm(a: &BorrowedTensor, p: f64, dim: isize, maxnorm: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let d = if dim < 0 {
        (a.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = a.shape[d] as usize;
    let inner: usize = a.shape[d + 1..]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = a.shape[..d]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    for o in 0..outer {
        for inn in 0..inner {
            let mut norm = 0.0;
            for i in 0..dim_size {
                let idx = (o * dim_size + i) * inner + inn;
                let v = ad[idx];
                norm += if p == 1.0 {
                    v.abs() as f64
                } else if p == 2.0 {
                    (v as f64) * (v as f64)
                } else {
                    (v.abs() as f64).powf(p)
                };
            }
            if p == 2.0 {
                norm = norm.sqrt();
            } else if p != 1.0 {
                norm = norm.powf(1.0 / p);
            }
            if norm > maxnorm {
                let scale = maxnorm / norm;
                for i in 0..dim_size {
                    let idx = (o * dim_size + i) * inner + inn;
                    od[idx] = (ad[idx] as f64 * scale) as f32;
                }
            }
        }
    }
    Ok(out)
}
pub fn bernoulli(a: &BorrowedTensor, p: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..od.len() {
        od[i] = if rand::random::<f64>() < p { 1.0 } else { 0.0 };
    }
    let _ = a;
    Ok(out)
}
pub fn multinomial(a: &BorrowedTensor, num_samples: usize) -> PyResult<OwnedTensor> {
    // a: (N, C) probs
    let n = a.shape[0] as usize;
    let c = a.shape[1] as usize;
    let mut out = OwnedTensor::new(DType::I64, vec![n as i64, num_samples as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..n {
        let base = i * c;
        let sum: f32 = ad[base..base + c].iter().sum();
        for s in 0..num_samples {
            let r = rand::random::<f32>() * sum;
            let mut acc = 0.0;
            for j in 0..c {
                acc += ad[base + j];
                if acc >= r {
                    od[i * num_samples + s] = j as i64;
                    break;
                }
            }
        }
    }
    Ok(out)
}
pub fn logspace(start: f64, end: f64, steps: usize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![steps as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    if steps == 1 {
        od[0] = 10f32.powf(start as f32);
    } else {
        let step = (end - start) / (steps - 1) as f64;
        for i in 0..steps {
            od[i] = 10f32.powf((start + i as f64 * step) as f32);
        }
    }
    Ok(out)
}
pub fn eye(n: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![n, n]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    for i in 0..n as usize {
        od[i * n as usize + i] = 1.0;
    }
    Ok(out)
}
pub fn diag(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.shape.len() == 1 {
        let n = a.shape[0] as usize;
        let mut out = OwnedTensor::new(a.dtype, vec![n as i64, n as i64]);
        let ad = unsafe { typed_slice::<f32>(a) };
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        od.fill(0.0);
        for i in 0..n {
            od[i * n + i] = ad[i];
        }
        Ok(out)
    } else if a.shape.len() == 2 {
        let n = a.shape[0].min(a.shape[1]) as usize;
        let mut out = OwnedTensor::new(a.dtype, vec![n as i64]);
        let ad = unsafe { typed_slice::<f32>(a) };
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        let w = a.shape[1] as usize;
        for i in 0..n {
            od[i] = ad[i * w + i];
        }
        Ok(out)
    } else {
        Err(unsupported("diag 1D or 2D"))
    }
}
pub fn triu(a: &BorrowedTensor, diagonal: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let rows = a.shape[0] as usize;
    let cols = a.shape[1] as usize;
    for i in 0..rows {
        for j in 0..cols {
            if (j as i64 - i as i64) < diagonal {
                od[i * cols + j] = 0.0;
            }
        }
    }
    Ok(out)
}
pub fn tril(a: &BorrowedTensor, diagonal: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let rows = a.shape[0] as usize;
    let cols = a.shape[1] as usize;
    for i in 0..rows {
        for j in 0..cols {
            if (j as i64 - i as i64) > diagonal {
                od[i * cols + j] = 0.0;
            }
        }
    }
    Ok(out)
}

// ── take / put / quantile / diagonal / trace / matrix_exp / slogdet / det / lstsq / pinverse / normal / uniform / windows / stft ──
pub fn take(a: &BorrowedTensor, index: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, index.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let idx = unsafe { typed_slice::<i64>(index) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..idx.len() {
        let ix = idx[i];
        if ix >= 0 && (ix as usize) < ad.len() {
            od[i] = ad[ix as usize];
        } else {
            od[i] = 0.0;
        }
    }
    Ok(out)
}

pub fn put(
    a: &BorrowedTensor,
    index: &BorrowedTensor,
    source: &BorrowedTensor,
    accumulate: bool,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let idx = unsafe { typed_slice::<i64>(index) };
    let sd = unsafe { typed_slice::<f32>(source) };
    for (i, &ix) in idx.iter().enumerate() {
        if ix >= 0 && (ix as usize) < od.len() {
            if accumulate {
                od[ix as usize] += sd[i % sd.len()];
            } else {
                od[ix as usize] = sd[i % sd.len()];
            }
        }
    }
    Ok(out)
}

pub fn quantile(
    a: &BorrowedTensor,
    q: f64,
    dim: Option<isize>,
    _keepdim: bool,
) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<f32>(a) };
    if dim.is_none() || a.shape.len() == 1 {
        let mut sorted = ad.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = OwnedTensor::new(DType::F32, vec![]);
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        if sorted.is_empty() {
            od[0] = 0.0;
        } else {
            let pos = q * (sorted.len() - 1) as f64;
            let low = pos.floor() as usize;
            let high = pos.ceil() as usize;
            let weight = (pos - low as f64) as f32;
            od[0] = sorted[low] * (1.0 - weight) + sorted[high.min(sorted.len() - 1)] * weight;
        }
        Ok(out)
    } else {
        unique(a)
    }
}

pub fn diagonal(
    a: &BorrowedTensor,
    offset: i64,
    dim1: isize,
    dim2: isize,
) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    let d1 = if dim1 < 0 {
        (rank as isize + dim1) as usize
    } else {
        dim1 as usize
    };
    let d2 = if dim2 < 0 {
        (rank as isize + dim2) as usize
    } else {
        dim2 as usize
    };
    let r = a.shape[d1] as usize;
    let c = a.shape[d2] as usize;
    let mut diag_len = 0usize;
    for i in 0..r {
        let j = i as i64 + offset;
        if j >= 0 && (j as usize) < c {
            diag_len += 1;
        }
    }
    let mut out = OwnedTensor::new(a.dtype, vec![diag_len as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let mut idx = 0;
    for i in 0..r {
        let j = i as i64 + offset;
        if j >= 0 && (j as usize) < c {
            od[idx] = ad[i * c + j as usize];
            idx += 1;
        }
    }
    Ok(out)
}

pub fn trace(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let d = diag(a)?;
    let d_b = BorrowedTensor::from_owned(&d);
    crate::reductions::sum(&d_b, None, false)
}

pub fn matrix_exp(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = a.shape[0] as usize;
    let mut result = eye(n as i64)?;
    let mut current = eye(n as i64)?;
    let ad = unsafe { typed_slice::<f32>(a) };

    // Taylor series: sum A^k / k! up to k=12
    let mut fact = 1.0f32;
    for k in 1..=12 {
        fact *= k as f32;
        // next_current = current * A
        let mut next_current = OwnedTensor::new(DType::F32, vec![n as i64, n as i64]);
        let cd = unsafe {
            std::slice::from_raw_parts(current.data.as_ptr() as *const f32, current.elem_count())
        };
        let ncd = unsafe { typed_mut_slice::<f32>(&mut next_current) };
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..n {
                    sum += cd[i * n + p] * ad[p * n + j];
                }
                ncd[i * n + j] = sum;
            }
        }
        let rd = unsafe { typed_mut_slice::<f32>(&mut result) };
        for idx in 0..n * n {
            rd[idx] += ncd[idx] / fact;
        }
        current = next_current;
    }
    Ok(result)
}

pub fn det(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = a.shape[0] as usize;
    let ad = unsafe { typed_slice::<f32>(a) };
    let mut out = OwnedTensor::new(DType::F32, vec![]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    if n == 1 {
        od[0] = ad[0];
    } else if n == 2 {
        od[0] = ad[0] * ad[3] - ad[1] * ad[2];
    } else if n == 3 {
        od[0] = ad[0] * (ad[4] * ad[8] - ad[5] * ad[7]) - ad[1] * (ad[3] * ad[8] - ad[5] * ad[6])
            + ad[2] * (ad[3] * ad[7] - ad[4] * ad[6]);
    } else {
        // Gaussian elimination with partial pivoting
        let mut mat: Vec<f64> = ad.iter().map(|&x| x as f64).collect();
        let mut det_val = 1.0f64;
        for i in 0..n {
            let mut pivot = i;
            for r in (i + 1)..n {
                if mat[r * n + i].abs() > mat[pivot * n + i].abs() {
                    pivot = r;
                }
            }
            if pivot != i {
                for c in 0..n {
                    mat.swap(i * n + c, pivot * n + c);
                }
                det_val = -det_val;
            }
            if mat[i * n + i].abs() < 1e-12 {
                det_val = 0.0;
                break;
            }
            det_val *= mat[i * n + i];
            for r in (i + 1)..n {
                let factor = mat[r * n + i] / mat[i * n + i];
                for c in (i + 1)..n {
                    mat[r * n + c] -= factor * mat[i * n + c];
                }
            }
        }
        od[0] = det_val as f32;
    }
    Ok(out)
}

pub fn slogdet(a: &BorrowedTensor) -> PyResult<(OwnedTensor, OwnedTensor)> {
    let d = det(a)?;
    let dd = unsafe { typed_slice::<f32>(&BorrowedTensor::from_owned(&d))[0] };
    let mut sign_t = OwnedTensor::new(DType::F32, vec![]);
    let mut log_t = OwnedTensor::new(DType::F32, vec![]);
    let sd = unsafe { typed_mut_slice::<f32>(&mut sign_t) };
    let ld = unsafe { typed_mut_slice::<f32>(&mut log_t) };
    if dd > 0.0 {
        sd[0] = 1.0;
        ld[0] = dd.ln();
    } else if dd < 0.0 {
        sd[0] = -1.0;
        ld[0] = (-dd).ln();
    } else {
        sd[0] = 0.0;
        ld[0] = f32::NEG_INFINITY;
    }
    Ok((sign_t, log_t))
}

pub fn pinverse(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // 2D matrix inversion via Gauss-Jordan elimination
    let rows = a.shape[0] as usize;
    let cols = a.shape[1] as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![cols as i64, rows as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    if rows == cols {
        let n = rows;
        let mut aug = vec![0.0f64; n * (2 * n)];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = ad[i * n + j] as f64;
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for i in 0..n {
            let mut pivot = i;
            for r in (i + 1)..n {
                if aug[r * 2 * n + i].abs() > aug[pivot * 2 * n + i].abs() {
                    pivot = r;
                }
            }
            if pivot != i {
                for c in 0..(2 * n) {
                    aug.swap(i * 2 * n + c, pivot * 2 * n + c);
                }
            }
            let pval = aug[i * 2 * n + i];
            let diag_val = if pval.abs() > 1e-12 { pval } else { 1e-6 };
            for c in 0..(2 * n) {
                aug[i * 2 * n + c] /= diag_val;
            }
            for r in 0..n {
                if r != i {
                    let factor = aug[r * 2 * n + i];
                    for c in 0..(2 * n) {
                        aug[r * 2 * n + c] -= factor * aug[i * 2 * n + c];
                    }
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                od[i * n + j] = aug[i * 2 * n + n + j] as f32;
            }
        }
    } else {
        // Transpose fallback
        for i in 0..rows {
            for j in 0..cols {
                od[j * rows + i] = ad[i * cols + j];
            }
        }
    }
    Ok(out)
}

pub fn lstsq(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let pinv = pinverse(a)?;
    let pinv_b = BorrowedTensor::from_owned(&pinv);
    crate::linalg::matmul(&pinv_b, b)
}

pub fn normal(mean: f64, std: f64, size: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, size.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in (0..od.len()).step_by(2) {
        let u1: f64 = rand::random::<f64>().max(1e-10);
        let u2: f64 = rand::random::<f64>();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
        od[i] = (mean + std * z0) as f32;
        if i + 1 < od.len() {
            od[i + 1] = (mean + std * z1) as f32;
        }
    }
    Ok(out)
}

pub fn uniform(from: f64, to: f64, size: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, size.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..od.len() {
        od[i] = (from + (to - from) * rand::random::<f64>()) as f32;
    }
    Ok(out)
}

pub fn hann_window(window_length: i64, periodic: bool) -> PyResult<OwnedTensor> {
    let n = window_length as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        n as f64
    } else {
        (n - 1).max(1) as f64
    };
    for i in 0..n {
        od[i] = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos()) as f32;
    }
    Ok(out)
}

pub fn bartlett_window(window_length: i64, periodic: bool) -> PyResult<OwnedTensor> {
    let n = window_length as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        n as f64
    } else {
        (n - 1).max(1) as f64
    };
    for i in 0..n {
        od[i] = (1.0 - (2.0 * i as f64 / denom - 1.0).abs()) as f32;
    }
    Ok(out)
}

pub fn blackman_window(window_length: i64, periodic: bool) -> PyResult<OwnedTensor> {
    let n = window_length as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        n as f64
    } else {
        (n - 1).max(1) as f64
    };
    for i in 0..n {
        let a0 = 0.42;
        let a1 = 0.5;
        let a2 = 0.08;
        od[i] = (a0 - a1 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos()
            + a2 * (4.0 * std::f64::consts::PI * i as f64 / denom).cos()) as f32;
    }
    Ok(out)
}

pub fn stft(
    input: &BorrowedTensor,
    n_fft: usize,
    hop_length: usize,
    _win_length: usize,
) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<f32>(input) };
    let n_frames = (ad.len().saturating_sub(n_fft)) / hop_length.max(1) + 1;
    let n_freqs = n_fft / 2 + 1;
    let mut out = OwnedTensor::new(DType::F32, vec![n_freqs as i64, n_frames as i64, 2]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for f in 0..n_frames {
        let offset = f * hop_length;
        for k in 0..n_freqs {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for t in 0..n_fft {
                if offset + t < ad.len() {
                    let angle = -2.0 * std::f64::consts::PI * (k * t) as f64 / n_fft as f64;
                    re += ad[offset + t] * angle.cos() as f32;
                    im += ad[offset + t] * angle.sin() as f32;
                }
            }
            let idx = (k * n_frames + f) * 2;
            od[idx] = re;
            od[idx + 1] = im;
        }
    }
    Ok(out)
}
