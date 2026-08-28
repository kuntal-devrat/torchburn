//! Math and comparison operations.
//!
//! Comparison ops (eq, ne, lt, le, gt, ge) return boolean-like f32 tensors
//! (1.0 for true, 0.0 for false) to stay within the f32/f64 engine.
//! Unary math ops (abs, neg, sign, sqrt, exp, log, etc.) are elementwise.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, contiguous_strides, unsupported};
use crate::ops::Scalar;
use pyo3::prelude::*;
use std::f64;

/// Read a tensor's elements as a typed slice.
unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

/// Write typed data into an owned tensor.
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

const PAR_CHUNK: usize = 16 * 1024;

// ---------------------------------------------------------------------------
// Comparison ops — return f32 (1.0/0.0) so results stay in the float engine
// ---------------------------------------------------------------------------

fn run_cmp<T: Scalar + PartialOrd>(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    out: &mut OwnedTensor,
    cmp_fn: impl Fn(T, T) -> bool + Sync + Send,
) {
    let a_data = unsafe { typed_slice::<T>(a) };
    let b_data = unsafe { typed_slice::<T>(b) };
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(PAR_CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let start = ci * PAR_CHUNK;
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = if cmp_fn(a_data[start + i], b_data[start + i]) { 1.0 } else { 0.0 };
            }
        });
}

fn run_cmp_broadcast<T: Scalar + PartialOrd>(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    out: &mut OwnedTensor,
    cmp_fn: impl Fn(T, T) -> bool + Sync + Send,
) {
    let a_data = unsafe { typed_slice::<T>(a) };
    let b_data = unsafe { typed_slice::<T>(b) };
    let n = out.elem_count();
    let out_rank = out.shape.len();
    let out_shape = out.shape.clone();
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    let a_pad = out_rank - a_rank;
    let b_pad = out_rank - b_rank;
    let mut coords = vec![0usize; out_rank];
    for oi in 0..n {
        let mut rem = oi;
        for d in (0..out_rank).rev() {
            coords[d] = rem % (out_shape[d].max(1) as usize);
            rem /= out_shape[d].max(1) as usize;
        }
        let mut ai = 0usize;
        let mut bi = 0usize;
        for d in 0..out_rank {
            let off = coords[d];
            if d >= a_pad && a.shape[d - a_pad] > 1 {
                ai += off * a.strides[d - a_pad] as usize;
            }
            if d >= b_pad && b.shape[d - b_pad] > 1 {
                bi += off * b.strides[d - b_pad] as usize;
            }
        }
        out_data[oi] = if cmp_fn(a_data[ai], b_data[bi]) { 1.0 } else { 0.0 };
    }
}

pub fn comparison(op: &str, a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.dtype != b.dtype {
        return Err(unsupported(&format!("dtype mismatch in comparison: {} vs {}", a.dtype.name(), b.dtype.name())));
    }
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let a_contig = a.strides == contiguous_strides(&a.shape);
    let b_contig = b.strides == contiguous_strides(&b.shape);
    let same_shape = a.shape == b.shape && a_contig && b_contig;

    match a.dtype {
        DType::F32 => {
            let cmp_f32 = match op {
                "eq" => |x: f32, y: f32| x == y,
                "ne" => |x: f32, y: f32| x != y,
                "lt" => |x: f32, y: f32| x < y,
                "le" => |x: f32, y: f32| x <= y,
                "gt" => |x: f32, y: f32| x > y,
                "ge" => |x: f32, y: f32| x >= y,
                _ => return Err(unsupported(&format!("unknown comparison op {op}"))),
            };
            if same_shape {
                run_cmp::<f32>(a, b, &mut out, cmp_f32);
            } else {
                run_cmp_broadcast::<f32>(a, b, &mut out, cmp_f32);
            }
        }
        DType::F64 => {
            let cmp_f64 = match op {
                "eq" => |x: f64, y: f64| x == y,
                "ne" => |x: f64, y: f64| x != y,
                "lt" => |x: f64, y: f64| x < y,
                "le" => |x: f64, y: f64| x <= y,
                "gt" => |x: f64, y: f64| x > y,
                "ge" => |x: f64, y: f64| x >= y,
                _ => return Err(unsupported(&format!("unknown comparison op {op}"))),
            };
            if same_shape {
                run_cmp::<f64>(a, b, &mut out, cmp_f64);
            } else {
                run_cmp_broadcast::<f64>(a, b, &mut out, cmp_f64);
            }
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"))
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Logical ops — operate on f32 (nonzero = true) and return f32
// ---------------------------------------------------------------------------

/// Broadcast-aware logical op over f32 tensors (nonzero = true).
fn run_logical_binary(
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    out: &mut OwnedTensor,
    op: impl Fn(bool, bool) -> bool + Sync + Send,
) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let b_data = unsafe { typed_slice::<f32>(b) };
    let n = out.elem_count();
    let out_rank = out.shape.len();
    let out_shape = out.shape.clone();
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    let a_pad = out_rank - a_rank;
    let b_pad = out_rank - b_rank;
    let mut coords = vec![0usize; out_rank];
    for oi in 0..n {
        let mut rem = oi;
        for d in (0..out_rank).rev() {
            coords[d] = rem % (out_shape[d].max(1) as usize);
            rem /= out_shape[d].max(1) as usize;
        }
        let mut ai = 0usize;
        let mut bi = 0usize;
        for d in 0..out_rank {
            let off = coords[d];
            if d >= a_pad && a.shape[d - a_pad] > 1 {
                ai += off * a.strides[d - a_pad] as usize;
            }
            if d >= b_pad && b.shape[d - b_pad] > 1 {
                bi += off * b.strides[d - b_pad] as usize;
            }
        }
        out_data[oi] = if op(a_data[ai] != 0.0, b_data[bi] != 0.0) { 1.0 } else { 0.0 };
    }
}

pub fn logical_and(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let same_shape = a.shape == b.shape && a.strides == contiguous_strides(&a.shape)
        && b.strides == contiguous_strides(&b.shape);
    if same_shape {
        let n = out.elem_count();
        let a_data = unsafe { typed_slice::<f32>(a) };
        let b_data = unsafe { typed_slice::<f32>(b) };
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        for i in 0..n {
            out_data[i] = if a_data[i] != 0.0 && b_data[i] != 0.0 { 1.0 } else { 0.0 };
        }
    } else {
        run_logical_binary(a, b, &mut out, |x, y| x && y);
    }
    Ok(out)
}

pub fn logical_or(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let out_shape = crate::ops::broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let same_shape = a.shape == b.shape && a.strides == contiguous_strides(&a.shape)
        && b.strides == contiguous_strides(&b.shape);
    if same_shape {
        let n = out.elem_count();
        let a_data = unsafe { typed_slice::<f32>(a) };
        let b_data = unsafe { typed_slice::<f32>(b) };
        let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
        for i in 0..n {
            out_data[i] = if a_data[i] != 0.0 || b_data[i] != 0.0 { 1.0 } else { 0.0 };
        }
    } else {
        run_logical_binary(a, b, &mut out, |x, y| x || y);
    }
    Ok(out)
}

pub fn logical_not(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = a.elem_count();
    if a.strides == contiguous_strides(&a.shape) {
        for i in 0..n {
            out_data[i] = if a_data[i] == 0.0 { 1.0 } else { 0.0 };
        }
    } else {
        // Strided input: map output index -> source index via coords.
        let a_rank = a.shape.len();
        let mut coords = vec![0usize; a_rank];
        for i in 0..n {
            let mut rem = i;
            for d in (0..a_rank).rev() {
                coords[d] = rem % (a.shape[d].max(1) as usize);
                rem /= a.shape[d].max(1) as usize;
            }
            let mut ai = 0usize;
            for d in 0..a_rank {
                if a.shape[d] > 1 {
                    ai += coords[d] * a.strides[d] as usize;
                }
            }
            out_data[i] = if a_data[ai] == 0.0 { 1.0 } else { 0.0 };
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Unary math ops — elementwise, contiguous fast path + general stride path
// ---------------------------------------------------------------------------

fn run_unary_contig_f32(a: &BorrowedTensor, out: &mut OwnedTensor, f: impl Fn(f32) -> f32 + Sync + Send) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = f(a_data[start + i]);
        }
    });
}

fn run_unary_contig_f64(a: &BorrowedTensor, out: &mut OwnedTensor, f: impl Fn(f64) -> f64 + Sync + Send) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = f(a_data[start + i]);
        }
    });
}

fn run_unary_general_f32(a: &BorrowedTensor, out: &mut OwnedTensor, f: impl Fn(f32) -> f32) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    let n = a.elem_count();
    let a_rank = a.shape.len();
    let mut coords = vec![0usize; a_rank];
    for i in 0..n {
        let mut rem = i;
        for d in (0..a_rank).rev() {
            coords[d] = rem % (a.shape[d].max(1) as usize);
            rem /= a.shape[d].max(1) as usize;
        }
        let mut ai = 0usize;
        for d in 0..a_rank {
            if a.shape[d] > 1 {
                ai += coords[d] * a.strides[d] as usize;
            }
        }
        out_data[i] = f(a_data[ai]);
    }
}

fn run_unary_general_f64(a: &BorrowedTensor, out: &mut OwnedTensor, f: impl Fn(f64) -> f64) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    let n = a.elem_count();
    let a_rank = a.shape.len();
    let mut coords = vec![0usize; a_rank];
    for i in 0..n {
        let mut rem = i;
        for d in (0..a_rank).rev() {
            coords[d] = rem % (a.shape[d].max(1) as usize);
            rem /= a.shape[d].max(1) as usize;
        }
        let mut ai = 0usize;
        for d in 0..a_rank {
            if a.shape[d] > 1 {
                ai += coords[d] * a.strides[d] as usize;
            }
        }
        out_data[i] = f(a_data[ai]);
    }
}

fn unary_f32(a: &BorrowedTensor, f: impl Fn(f32) -> f32 + Sync + Send) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, a.shape.clone());
    if a.strides == contiguous_strides(&a.shape) {
        run_unary_contig_f32(a, &mut out, f);
    } else {
        run_unary_general_f32(a, &mut out, f);
    }
    Ok(out)
}

fn unary_f64(a: &BorrowedTensor, f: impl Fn(f64) -> f64 + Sync + Send) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F64, a.shape.clone());
    if a.strides == contiguous_strides(&a.shape) {
        run_unary_contig_f64(a, &mut out, f);
    } else {
        run_unary_general_f64(a, &mut out, f);
    }
    Ok(out)
}

pub fn abs(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.abs()),
        DType::F64 => unary_f64(a, |x| x.abs()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn neg(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| -x),
        DType::F64 => unary_f64(a, |x| -x),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn sign(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.signum()),
        DType::F64 => unary_f64(a, |x| x.signum()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn sqrt(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.sqrt()),
        DType::F64 => unary_f64(a, |x| x.sqrt()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn rsqrt(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| 1.0 / x.sqrt()),
        DType::F64 => unary_f64(a, |x| 1.0 / x.sqrt()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn exp(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.exp()),
        DType::F64 => unary_f64(a, |x| x.exp()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn log(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.ln()),
        DType::F64 => unary_f64(a, |x| x.ln()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn reciprocal(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| 1.0 / x),
        DType::F64 => unary_f64(a, |x| 1.0 / x),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn ceil(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.ceil()),
        DType::F64 => unary_f64(a, |x| x.ceil()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

pub fn floor(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.floor()),
        DType::F64 => unary_f64(a, |x| x.floor()),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

// ---------------------------------------------------------------------------
// clamp — elementwise min/max bounds
// ---------------------------------------------------------------------------

fn run_clamp_f32(a: &BorrowedTensor, out: &mut OwnedTensor, min: f32, max: f32) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            let v = a_data[start + i];
            *o = v.max(min).min(max);
        }
    });
}

fn run_clamp_f64(a: &BorrowedTensor, out: &mut OwnedTensor, min: f64, max: f64) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            let v = a_data[start + i];
            *o = v.max(min).min(max);
        }
    });
}

pub fn clamp(a: &BorrowedTensor, min: f64, max: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => run_clamp_f32(a, &mut out, min as f32, max as f32),
        DType::F64 => run_clamp_f64(a, &mut out, min, max),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
    Ok(out)
}

pub fn sin(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.sin()),
        DType::F64 => unary_f64(a, |x| x.sin()),
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

pub fn cos(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.cos()),
        DType::F64 => unary_f64(a, |x| x.cos()),
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

pub fn round(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x: f32| {
            // Banker's rounding: round half to even (matches PyTorch)
            let r = x;
            let floor = r.floor();
            let frac = r - floor;
            if (frac - 0.5).abs() < 1e-6 {
                // Half case: round to even
                let f = floor as i64;
                if f % 2 == 0 { floor } else { floor + 1.0 }
            } else {
                r.round()
            }
        }),
        DType::F64 => unary_f64(a, |x: f64| {
            let r = x;
            let floor = r.floor();
            let frac = r - floor;
            if (frac - 0.5).abs() < 1e-9 {
                let f = floor as i64;
                if f % 2 == 0 { floor } else { floor + 1.0 }
            } else {
                r.round()
            }
        }),
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

// ---------------------------------------------------------------------------
// clamp_min / clamp_max — elementwise bound on one side
// ---------------------------------------------------------------------------

fn run_clamp_min_f32(a: &BorrowedTensor, out: &mut OwnedTensor, min: f32) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = a_data[start + i].max(min);
        }
    });
}

fn run_clamp_min_f64(a: &BorrowedTensor, out: &mut OwnedTensor, min: f64) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = a_data[start + i].max(min);
        }
    });
}

fn run_clamp_max_f32(a: &BorrowedTensor, out: &mut OwnedTensor, max: f32) {
    let a_data = unsafe { typed_slice::<f32>(a) };
    let out_data = unsafe { typed_mut_slice::<f32>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = a_data[start + i].min(max);
        }
    });
}

fn run_clamp_max_f64(a: &BorrowedTensor, out: &mut OwnedTensor, max: f64) {
    let a_data = unsafe { typed_slice::<f64>(a) };
    let out_data = unsafe { typed_mut_slice::<f64>(out) };
    use rayon::prelude::*;
    out_data.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * PAR_CHUNK;
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = a_data[start + i].min(max);
        }
    });
}

pub fn clamp_min(a: &BorrowedTensor, min: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => run_clamp_min_f32(a, &mut out, min as f32),
        DType::F64 => run_clamp_min_f64(a, &mut out, min),
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

pub fn clamp_max(a: &BorrowedTensor, max: f64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => run_clamp_max_f32(a, &mut out, max as f32),
        DType::F64 => run_clamp_max_f64(a, &mut out, max),
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// pow — elementwise x^y (scalar exponent)
// ---------------------------------------------------------------------------

pub fn pow_scalar(a: &BorrowedTensor, exp: f64) -> PyResult<OwnedTensor> {
    match a.dtype {
        DType::F32 => unary_f32(a, |x| x.powf(exp as f32)),
        DType::F64 => unary_f64(a, |x| x.powf(exp)),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

// ---------------------------------------------------------------------------
// Type cast — convert between f32/f64
// ---------------------------------------------------------------------------

pub fn to_dtype(a: &BorrowedTensor, target: DType) -> PyResult<OwnedTensor> {
    if a.dtype == target {
        // Same dtype: just copy
        let n = a.elem_count();
        let mut out = OwnedTensor::new(target, a.shape.clone());
        let bytes = n * a.dtype.elem_size();
        unsafe {
            std::ptr::copy_nonoverlapping(a.data, out.data.as_mut_ptr() as *mut u8, bytes);
        }
        return Ok(out);
    }
    let n = a.elem_count();
    let mut out = OwnedTensor::new(target, a.shape.clone());
    match (a.dtype, target) {
        (DType::F32, DType::F64) => {
            let src = unsafe { typed_slice::<f32>(a) };
            let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as f64;
            }
        }
        (DType::F64, DType::F32) => {
            let src = unsafe { typed_slice::<f64>(a) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as f32;
            }
        }
        (DType::I64, DType::F32) => {
            let src = unsafe { typed_slice::<i64>(a) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as f32;
            }
        }
        (DType::I64, DType::F64) => {
            let src = unsafe { typed_slice::<i64>(a) };
            let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as f64;
            }
        }
        (DType::I32, DType::F32) => {
            let src = unsafe { typed_slice::<i32>(a) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as f32;
            }
        }
        (DType::I32, DType::F64) => {
            let src = unsafe { typed_slice::<i32>(a) };
            let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as f64;
            }
        }
        (DType::Bool, DType::F32) => {
            let src = unsafe { typed_slice::<u8>(a) };
            let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
            for i in 0..n {
                dst[i] = if src[i] != 0 { 1.0 } else { 0.0 };
            }
        }
        (DType::Bool, DType::F64) => {
            let src = unsafe { typed_slice::<u8>(a) };
            let dst = unsafe { typed_mut_slice::<f64>(&mut out) };
            for i in 0..n {
                dst[i] = if src[i] != 0 { 1.0 } else { 0.0 };
            }
        }
        (DType::F32, DType::I64) => {
            let src = unsafe { typed_slice::<f32>(a) };
            let dst = unsafe { typed_mut_slice::<i64>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as i64;
            }
        }
        (DType::F32, DType::I32) => {
            let src = unsafe { typed_slice::<f32>(a) };
            let dst = unsafe { typed_mut_slice::<i32>(&mut out) };
            for i in 0..n {
                dst[i] = src[i] as i32;
            }
        }
        _ => return Err(unsupported(&format!("unsupported cast from {:?} to {:?}", a.dtype, target))),
    }
    Ok(out)
}
