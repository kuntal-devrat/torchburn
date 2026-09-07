//! Scatter variants: select/slice/diagonal scatter, index_copy, narrow_copy. Inherits linalg root imports via super; pure move.

use super::*;

// ── 76-85. Scattering & Slicing ──
pub fn select_scatter(
    input: &BorrowedTensor,
    src: &BorrowedTensor,
    dim: isize,
    index: i64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let d = if dim < 0 {
        (input.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = input.shape[d] as usize;
    let inner: usize = input.shape[d + 1..]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = input.shape[..d]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let mut src_idx = 0;
    for o in 0..outer {
        for inn in 0..inner {
            let idx = (o * dim_size + index as usize) * inner + inn;
            if idx < od.len() {
                od[idx] = sd[src_idx % sd.len()];
                src_idx += 1;
            }
        }
    }
    Ok(out)
}

pub fn slice_scatter(
    input: &BorrowedTensor,
    src: &BorrowedTensor,
    dim: isize,
    start: Option<i64>,
    end: Option<i64>,
    step: i64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let d = if dim < 0 {
        (input.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = input.shape[d] as usize;
    let s = start.unwrap_or(0).max(0) as usize;
    let e = end.unwrap_or(dim_size as i64).min(dim_size as i64) as usize;
    let st = step.max(1) as usize;
    let inner: usize = input.shape[d + 1..]
        .iter()
        .map(|&x| x.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = input.shape[..d]
        .iter()
        .map(|&x| x.max(1) as usize)
        .product::<usize>()
        .max(1);
    let mut src_i = 0;
    for o in 0..outer {
        for pos in (s..e).step_by(st) {
            for inn in 0..inner {
                let idx = (o * dim_size + pos) * inner + inn;
                if idx < od.len() {
                    od[idx] = sd[src_i % sd.len()];
                    src_i += 1;
                }
            }
        }
    }
    Ok(out)
}

pub fn diagonal_scatter(
    input: &BorrowedTensor,
    src: &BorrowedTensor,
    offset: i64,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let sd = unsafe { typed_slice::<f32>(src) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let n = input.shape[0].min(input.shape[1]) as usize;
    let w = input.shape[1] as usize;
    for i in 0..n {
        let col = (i as i64 + offset) as usize;
        if col < w {
            od[i * w + col] = sd[i % sd.len()];
        }
    }
    Ok(out)
}

pub fn index_copy(
    input: &BorrowedTensor,
    dim: isize,
    index: &BorrowedTensor,
    source: &BorrowedTensor,
) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, input.shape.clone());
    let id = unsafe { typed_slice::<f32>(input) };
    let idx = unsafe { typed_slice::<i64>(index) };
    let sd = unsafe { typed_slice::<f32>(source) };
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.copy_from_slice(id);
    let d = if dim < 0 {
        (input.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let dim_size = input.shape[d] as usize;
    let inner: usize = input.shape[d + 1..]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    let outer: usize = input.shape[..d]
        .iter()
        .map(|&s| s.max(1) as usize)
        .product::<usize>()
        .max(1);
    for (si, &pos) in idx.iter().enumerate() {
        if pos < 0 || pos as usize >= dim_size {
            continue;
        }
        for o in 0..outer {
            for inn in 0..inner {
                let dst_idx = (o * dim_size + pos as usize) * inner + inn;
                let src_idx = (o * idx.len() + si) * inner + inn;
                od[dst_idx] = sd[src_idx % sd.len()];
            }
        }
    }
    Ok(out)
}

pub fn narrow_copy(
    input: &BorrowedTensor,
    dim: isize,
    start: usize,
    length: usize,
) -> PyResult<OwnedTensor> {
    crate::shape_ops::narrow(input, dim, start, length)
}
