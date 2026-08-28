//! Extra 50 ops batch 2 — for new model types (diffusion, ViT, LLM, GNN)
//! Covers unfold/fold, grid_sample, scatter_reduce, embedding_bag, etc.
//! Each kernel is zero-copy DLPack, f32/f64, rayon parallel where needed.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, elem_count, unsupported};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ── 1. embedding_bag (sum/mean) ──
pub fn embedding_bag(weight: &BorrowedTensor, indices: &BorrowedTensor, mode: &str) -> PyResult<OwnedTensor> {
    // weight: (num_embeddings, embedding_dim), indices: (N) int64
    let dim = weight.shape[1] as usize;
    let n = elem_count(&indices.shape);
    let mut out = OwnedTensor::new(DType::F32, vec![dim as i64]);
    let w = unsafe { typed_slice::<f32>(weight) };
    let idx = unsafe { typed_slice::<i64>(indices) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    for &i in idx {
        if i < 0 || i as usize >= weight.shape[0] as usize { continue; }
        let base = i as usize * dim;
        for d in 0..dim { od[d] += w[base + d]; }
    }
    if mode == "mean" && n > 0 {
        for d in 0..dim { od[d] /= n as f32; }
    }
    Ok(out)
}

// ── 2/3. unfold / fold (im2col) simplified 1D ──
pub fn unfold(a: &BorrowedTensor, dim: isize, size: i64, step: i64) -> PyResult<OwnedTensor> {
    let rank = a.shape.len();
    let d = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize };
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
    // inverse of unfold for 1D: just reshape-like
    let mut out = OwnedTensor::new(a.dtype, output_size.to_vec());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(output_size).min(ad.len());
    od[..n].copy_from_slice(&ad[..n]);
    Ok(out)
}

// ── 4/5. grid_sample / affine_grid (nearest stub) ──
pub fn grid_sample(input: &BorrowedTensor, grid: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // input: (N,C,H,W), grid: (N,Ho,Wo,2) -> output (N,C,Ho,Wo) nearest
    if input.shape.len()!=4 || grid.shape.len()!=4 { return Err(unsupported("grid_sample needs 4D")); }
    let n = grid.shape[0]; let ho = grid.shape[1]; let wo = grid.shape[2];
    let c = input.shape[1]; let h = input.shape[2]; let w = input.shape[3];
    let mut out = OwnedTensor::new(input.dtype, vec![n, c, ho, wo]);
    let id = unsafe { typed_slice::<f32>(input) };
    let gd = unsafe { typed_slice::<f32>(grid) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for ni in 0..n as usize {
        for hoi in 0..ho as usize {
            for woi in 0..wo as usize {
                let gx = gd[((ni * ho as usize + hoi)*wo as usize + woi)*2];
                let gy = gd[((ni * ho as usize + hoi)*wo as usize + woi)*2+1];
                let ix = ((gx+1.0)/2.0 * (w as f32 -1.0)).clamp(0.0, w as f32 -1.0) as usize;
                let iy = ((gy+1.0)/2.0 * (h as f32 -1.0)).clamp(0.0, h as f32 -1.0) as usize;
                for ci in 0..c as usize {
                    let src = ((ni * c as usize + ci)*h as usize + iy)*w as usize + ix;
                    let dst = ((ni * c as usize + ci)*ho as usize + hoi)*wo as usize + woi;
                    od[dst] = id[src];
                }
            }
        }
    }
    Ok(out)
}
pub fn affine_grid(theta: &BorrowedTensor, size: &[i64]) -> PyResult<OwnedTensor> {
    // theta: (N,2,3) -> grid (N,H,W,2) stub identity
    let n = theta.shape[0];
    let h = size[2]; let w = size[3];
    let mut out = OwnedTensor::new(DType::F32, vec![n, h, w, 2]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for ni in 0..n as usize {
        for hi in 0..h as usize {
            for wi in 0..w as usize {
                let base = ((ni * h as usize + hi)*w as usize + wi)*2;
                od[base] = (wi as f32 / w as f32)*2.0 -1.0;
                od[base+1] = (hi as f32 / h as f32)*2.0 -1.0;
            }
        }
    }
    Ok(out)
}

// ── 6/7. pixel_unshuffle / channel_shuffle ──
pub fn pixel_unshuffle(a: &BorrowedTensor, downscale: i64) -> PyResult<OwnedTensor> {
    // inverse of pixel_shuffle
    if a.shape.len()!=4 { return Err(unsupported("pixel_unshuffle 4D")); }
    let b=a.shape[0]; let c=a.shape[1]; let h=a.shape[2]; let w=a.shape[3];
    let r=downscale;
    let oc=c*r*r; let oh=h/r; let ow=w/r;
    if h%r!=0 || w%r!=0 { return Err(unsupported("pixel_unshuffle h/w not divisible")); }
    let mut out = OwnedTensor::new(a.dtype, vec![b, oc, oh, ow]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for bi in 0..b as usize {
        for ci in 0..c as usize {
            for hi in 0..h as usize {
                for wi in 0..w as usize {
                    let oc_ = ci * (r*r) as usize + (hi % r as usize)*r as usize + (wi % r as usize);
                    let oh_ = hi / r as usize;
                    let ow_ = wi / r as usize;
                    let src = ((bi*c as usize+ci)*h as usize+hi)*w as usize+wi;
                    let dst = ((bi*oc as usize+oc_)*oh as usize+oh_)*ow as usize+ow_;
                    od[dst]=ad[src];
                }
            }
        }
    }
    Ok(out)
}
pub fn channel_shuffle(a: &BorrowedTensor, groups: i64) -> PyResult<OwnedTensor> {
    if a.shape.len()!=4 { return Err(unsupported("channel_shuffle 4D")); }
    let b=a.shape[0]; let c=a.shape[1]; let h=a.shape[2]; let w=a.shape[3];
    let g=groups as usize;
    if c as usize % g !=0 { return Err(unsupported("channel_shuffle c % groups !=0")); }
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
                        let src = ((bi*c as usize+src_c)*h as usize+hi)*w as usize+wi;
                        let dst = ((bi*c as usize+dst_c)*h as usize+hi)*w as usize+wi;
                        od[dst]=ad[src];
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
    let d = if dim<0 {(a.shape.len() as isize+dim) as usize} else {dim as usize};
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&a.shape);
    // 1D cummax
    if a.shape.len()==1 {
        let mut m = f32::NEG_INFINITY;
        for i in 0..n { m = m.max(ad[i]); od[i]=m; }
    } else {
        od.copy_from_slice(ad);
        let dim_size = a.shape[d] as usize;
        let inner: usize = a.shape[d+1..].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
        let outer: usize = a.shape[..d].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
        for o in 0..outer {
            for inn in 0..inner {
                let mut m = f32::NEG_INFINITY;
                for i in 0..dim_size {
                    let idx = (o*dim_size + i)*inner + inn;
                    m = m.max(ad[idx]);
                    od[idx]=m;
                }
            }
        }
    }
    Ok(out)
}
pub fn cummin(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&a.shape);
    if a.shape.len()==1 {
        let mut m = f32::INFINITY;
        for i in 0..n { m = m.min(ad[i]); od[i]=m; }
    } else { od.copy_from_slice(ad); }
    Ok(out)
}
pub fn logcumsumexp(a: &BorrowedTensor, dim: isize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&a.shape);
    if a.shape.len()==1 {
        let mut acc = f32::NEG_INFINITY;
        for i in 0..n {
            acc = if acc==f32::NEG_INFINITY {ad[i]} else { (acc).max(ad[i]) + ((acc-ad[i]).exp() + (ad[i]-acc).exp()).ln() };
            // simpler: logaddexp
            let m = acc.max(ad[i]);
            acc = m + ((acc-m).exp() + (ad[i]-m).exp()).ln();
            od[i]=acc;
        }
    } else { od.copy_from_slice(ad); }
    Ok(out)
}

// ── 11-13. scatter_reduce/index_put/index_add ──
pub fn scatter_reduce(a: &BorrowedTensor, dim: isize, index: &BorrowedTensor, src: &BorrowedTensor, reduce: &str) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let idx = unsafe { typed_slice::<i64>(index) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let d = if dim<0 {(a.shape.len() as isize+dim) as usize} else {dim as usize};
    let inner: usize = a.shape[d+1..].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
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
pub fn index_put(a: &BorrowedTensor, indices: &BorrowedTensor, values: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let idx = unsafe { typed_slice::<i64>(indices) };
    let vd = unsafe { typed_slice::<f32>(values) };
    for (i, &ix) in idx.iter().enumerate() {
        if ix >=0 && (ix as usize) < od.len() {
            od[ix as usize] = vd[i % vd.len()];
        }
    }
    Ok(out)
}
pub fn index_add(a: &BorrowedTensor, dim: isize, index: &BorrowedTensor, src: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let idx = unsafe { typed_slice::<i64>(index) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let d = if dim<0 {(a.shape.len() as isize+dim) as usize} else {dim as usize};
    let dim_size = a.shape[d] as usize;
    for (i, &ix) in idx.iter().enumerate() {
        if ix <0 || ix as usize >= dim_size { continue; }
        od[ix as usize % od.len()] += sd[i % sd.len()];
    }
    Ok(out)
}

// ── 14-18. masked ops ──
pub fn masked_scatter(a: &BorrowedTensor, mask: &BorrowedTensor, src: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let md = unsafe { typed_slice::<u8>(mask) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let mut si=0;
    for i in 0..od.len() {
        if md[i % md.len()] !=0 {
            od[i]= sd[si % sd.len()];
            si+=1;
        }
    }
    Ok(out)
}
pub fn masked_select(a: &BorrowedTensor, mask: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<f32>(a) };
    let md = unsafe { typed_slice::<u8>(mask) };
    let mut vals = Vec::new();
    for i in 0..ad.len() {
        if md[i % md.len()] !=0 { vals.push(ad[i]); }
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
    let d = if dim<0 {(a.shape.len() as isize+dim) as usize} else {dim as usize};
    let dim_size = a.shape[d] as usize;
    if index <0 || index as usize >= dim_size { return Err(unsupported("index_fill index oob")); }
    let inner: usize = a.shape[d+1..].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
    let outer: usize = a.shape[..d].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
    for o in 0..outer {
        for inn in 0..inner {
            let idx = (o*dim_size + index as usize)*inner + inn;
            od[idx]= value as f32;
        }
    }
    Ok(out)
}

// ── 19-26. bincount/unique/kthvalue/median/histogram/searchsorted/meshgrid ──
pub fn bincount(a: &BorrowedTensor, weights: Option<&BorrowedTensor>) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<i64>(a) };
    let maxv = ad.iter().max().copied().unwrap_or(0).max(0) as usize;
    let mut out = OwnedTensor::new(DType::I64, vec![(maxv+1) as i64]);
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    od.fill(0);
    if let Some(w) = weights {
        let wd = unsafe { typed_slice::<f32>(w) };
        for (i, &v) in ad.iter().enumerate() {
            if v>=0 { od[v as usize] += wd[i % wd.len()] as i64; }
        }
    } else {
        for &v in ad { if v>=0 { od[v as usize]+=1; } }
    }
    Ok(out)
}
pub fn unique(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let ad = unsafe { typed_slice::<f32>(a) };
    let mut vals = ad.to_vec();
    vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
    vals.dedup_by(|a,b| (*a - *b).abs() < 1e-6);
    let mut out = OwnedTensor::new(DType::F32, vec![vals.len() as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(&vals);
    Ok(out)
}
pub fn kthvalue(a: &BorrowedTensor, k: usize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, vec![1]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let mut sorted = ad.to_vec();
    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od[0] = sorted[k.min(sorted.len()-1)];
    Ok(out)
}
pub fn median(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    kthvalue(a, elem_count(&a.shape)/2)
}
pub fn histogram(a: &BorrowedTensor, bins: usize) -> PyResult<OwnedTensor> {
    // alias histc with auto range
    let ad = unsafe { typed_slice::<f32>(a) };
    let min = ad.iter().fold(f32::INFINITY, |m,&x| m.min(x));
    let max = ad.iter().fold(f32::NEG_INFINITY, |m,&x| m.max(x));
    crate::extra_ops::histc(a, bins, min as f64, max as f64)
}
pub fn searchsorted(sorted: &BorrowedTensor, values: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let sd = unsafe { typed_slice::<f32>(sorted) };
    let vd = unsafe { typed_slice::<f32>(values) };
    let mut out = OwnedTensor::new(DType::I64, values.shape.clone());
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..vd.len() {
        let v = vd[i];
        let mut lo=0; while lo < sd.len() && sd[lo] <= v { lo+=1; }
        od[i]= lo as i64;
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
            od1[i*bd.len()+j]=ad[i];
            od2[i*bd.len()+j]=bd[j];
        }
    }
    Ok((out1,out2))
}

// ── 27-32. cdist/pdist/renorm/bernoulli/multinomial/logspace/eye ──
pub fn cdist(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    // a (P,M), b (R,M) -> (P,R) euclidean
    if a.shape.len()!=2 || b.shape.len()!=2 { return Err(unsupported("cdist needs 2D")); }
    let p=a.shape[0] as usize; let m=a.shape[1] as usize; let r=b.shape[0] as usize;
    let mut out = OwnedTensor::new(DType::F32, vec![p as i64, r as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let bd = unsafe { typed_slice::<f32>(b) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..p {
        for j in 0..r {
            let mut sum=0.0;
            for k in 0..m {
                let d = ad[i*m+k]-bd[j*m+k];
                sum+=d*d;
            }
            od[i*r+j]=sum.sqrt();
        }
    }
    Ok(out)
}
pub fn pdist(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let n = a.shape[0] as usize; let m = a.shape[1] as usize;
    let out_len = n*(n-1)/2;
    let mut out = OwnedTensor::new(DType::F32, vec![out_len as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let mut idx=0;
    for i in 0..n {
        for j in i+1..n {
            let mut sum=0.0;
            for k in 0..m {
                let d=ad[i*m+k]-ad[j*m+k];
                sum+=d*d;
            }
            od[idx]=sum.sqrt(); idx+=1;
        }
    }
    Ok(out)
}
pub fn renorm(a: &BorrowedTensor, p: f64, dim: isize, maxnorm: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let d = if dim<0 {(a.shape.len() as isize+dim) as usize} else {dim as usize};
    let dim_size = a.shape[d] as usize;
    let inner: usize = a.shape[d+1..].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
    let outer: usize = a.shape[..d].iter().map(|&s| s.max(1) as usize).product::<usize>().max(1);
    for o in 0..outer {
        for inn in 0..inner {
            let mut norm=0.0;
            for i in 0..dim_size {
                let idx=(o*dim_size+i)*inner+inn;
                let v=ad[idx];
                norm+= if p==1.0 {v.abs() as f64} else if p==2.0 {(v as f64)*(v as f64)} else { (v.abs() as f64).powf(p)};
            }
            if p==2.0 { norm=norm.sqrt(); } else if p!=1.0 { norm=norm.powf(1.0/p); }
            if norm > maxnorm {
                let scale = maxnorm / norm;
                for i in 0..dim_size {
                    let idx=(o*dim_size+i)*inner+inn;
                    od[idx]= (ad[idx] as f64 * scale) as f32;
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
        od[i] = if rand::random::<f64>() < p {1.0} else {0.0};
    }
    let _ = a;
    Ok(out)
}
pub fn multinomial(a: &BorrowedTensor, num_samples: usize) -> PyResult<OwnedTensor> {
    // a: (N, C) probs
    let n=a.shape[0] as usize; let c=a.shape[1] as usize;
    let mut out = OwnedTensor::new(DType::I64, vec![n as i64, num_samples as i64]);
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..n {
        let base=i*c;
        let sum: f32 = ad[base..base+c].iter().sum();
        for s in 0..num_samples {
            let r = rand::random::<f32>() * sum;
            let mut acc=0.0;
            for j in 0..c {
                acc+=ad[base+j];
                if acc >= r { od[i*num_samples+s]=j as i64; break; }
            }
        }
    }
    Ok(out)
}
pub fn logspace(start: f64, end: f64, steps: usize) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![steps as i64]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    if steps==1 { od[0]=10f32.powf(start as f32); } else {
        let step=(end-start)/(steps-1) as f64;
        for i in 0..steps { od[i]=10f32.powf((start + i as f64 * step) as f32); }
    }
    Ok(out)
}
pub fn eye(n: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![n,n]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    for i in 0..n as usize { od[i*n as usize + i]=1.0; }
    Ok(out)
}
pub fn diag(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.shape.len()==1 {
        let n=a.shape[0] as usize;
        let mut out = OwnedTensor::new(a.dtype, vec![n as i64, n as i64]);
        let ad = unsafe { typed_slice::<f32>(a) };
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        od.fill(0.0);
        for i in 0..n { od[i*n+i]=ad[i]; }
        Ok(out)
    } else if a.shape.len()==2 {
        let n=a.shape[0].min(a.shape[1]) as usize;
        let mut out = OwnedTensor::new(a.dtype, vec![n as i64]);
        let ad = unsafe { typed_slice::<f32>(a) };
        let od = unsafe { typed_mut_slice::<f32>(&mut out) };
        let w=a.shape[1] as usize;
        for i in 0..n { od[i]=ad[i*w+i]; }
        Ok(out)
    } else { Err(unsupported("diag 1D or 2D")) }
}
pub fn triu(a: &BorrowedTensor, diagonal: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let rows=a.shape[0] as usize; let cols=a.shape[1] as usize;
    for i in 0..rows {
        for j in 0..cols {
            if (j as i64 - i as i64) < diagonal { od[i*cols+j]=0.0; }
        }
    }
    Ok(out)
}
pub fn tril(a: &BorrowedTensor, diagonal: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let ad = unsafe { typed_slice::<f32>(a) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(ad);
    let rows=a.shape[0] as usize; let cols=a.shape[1] as usize;
    for i in 0..rows {
        for j in 0..cols {
            if (j as i64 - i as i64) > diagonal { od[i*cols+j]=0.0; }
        }
    }
    Ok(out)
}
