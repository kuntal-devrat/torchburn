//! Elementary tensor kernels (Phase 1: add/mul; plus sub/div/relu).
//!
//! Inputs are read directly from PyTorch-owned DLPack buffers (zero-copy),
//! respecting arbitrary stride layouts. Outputs are fresh Rust allocations.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, contiguous_strides, elem_count, unsupported};
use pyo3::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub fn from_target(target: &str) -> Option<BinaryOp> {
        match target {
            "add" => Some(BinaryOp::Add),
            "sub" => Some(BinaryOp::Sub),
            "mul" => Some(BinaryOp::Mul),
            "div" => Some(BinaryOp::Div),
            _ => None,
        }
    }
}

/// Minimal numeric trait so the kernels can be written once for f32/f64.
pub trait Scalar:
    Copy
        + PartialOrd
        + Send
        + Sync
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
{
    fn zero() -> Self;
}

impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
}
impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
}

pub fn broadcast_shape(a: &[i64], b: &[i64]) -> PyResult<Vec<i64>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0i64; rank];
    for i in 0..rank {
        let ai = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let bi = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        if ai == bi {
            out[i] = ai;
        } else if ai == 1 {
            out[i] = bi;
        } else if bi == 1 {
            out[i] = ai;
        } else {
            return Err(unsupported(&format!(
                "incompatible broadcast shapes {a:?} vs {b:?}"
            )));
        }
    }
    Ok(out)
}

fn apply<T: Scalar>(op: BinaryOp, x: T, y: T) -> T {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
    }
}

/// Read a tensor's elements as a typed slice.
///
/// Strided views can reach element indices beyond `elem_count` (their shape
/// product), so the slice length is the maximum linear index addressable via
/// shape/strides + 1 — exactly what the producer guarantees is allocated.
unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

/// Elements per parallel work chunk; keeps per-thread scheduling overhead low
/// while still splitting large tensors across cores.
const PAR_CHUNK: usize = 16 * 1024;
/// Tensors smaller than this run serially: rayon pool dispatch (~50-100us)
/// costs more than the elementwise work itself on the sizes typical of
/// transformer graphs (residual adds, activations on 256x128 tensors).
const PAR_THRESHOLD: usize = 64 * 1024;

/// Fill `out[i] = f(i)` — serially below the threshold, chunked-parallel above
/// it (avoids the rayon dispatch tax on small tensors).
fn map_in_place<T>(n: usize, out: &mut [T], f: impl Fn(usize) -> T + Sync)
where
    T: Send + Sync,
{
    if n >= PAR_THRESHOLD {
        use rayon::prelude::*;
        out.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
            let start = ci * PAR_CHUNK;
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = f(start + i);
            }
        });
    } else {
        for (i, o) in out.iter_mut().enumerate().take(n) {
            *o = f(i);
        }
    }
}

fn run_binary<T: Scalar>(op: BinaryOp, a: &BorrowedTensor, b: &BorrowedTensor, out: &mut OwnedTensor) {
    // SAFETY: buffers sized by elem_count and dtype.
    let a_data = unsafe { typed_slice::<T>(a) };
    let b_data = unsafe { typed_slice::<T>(b) };
    let out_data = unsafe {
        std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut T, out.elem_count())
    };
    let n = out.elem_count();

    let a_contig = a.strides == contiguous_strides(&a.shape);
    let b_contig = b.strides == contiguous_strides(&b.shape);

    // Fast path 1: identical shapes, both contiguous (the common compiled-graph
    // case; linear indexing is only valid for contiguous layouts).
    if a.shape == b.shape && a_contig && b_contig {
        map_in_place(n, out_data, |i| apply(op, a_data[i], b_data[i]));
        return;
    }

    // Fast path 2: one operand is an effective scalar (0-d or all-1 dims).
    // The other must be contiguous so plain linear indexing reads it.
    if a.elem_count() == 1 && b_contig {
        let scalar = a_data[0];
        map_in_place(n, out_data, |i| apply(op, scalar, b_data[i]));
        return;
    }
    if b.elem_count() == 1 && a_contig {
        let scalar = b_data[0];
        map_in_place(n, out_data, |i| apply(op, a_data[i], scalar));
        return;
    }

    // General numpy-style broadcasting. Dimensions are right-aligned:
    // out dim `d` maps to source dim `d - (out_rank - src_rank)` when
    // within the source's rank, otherwise the source is broadcast (size 1).
    let out_rank = out.shape.len();
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    let a_pad = out_rank - a_rank;
    let b_pad = out_rank - b_rank;
    let mut coords = vec![0usize; out_rank];
    for oi in 0..n {
        let mut rem = oi;
        for d in (0..out_rank).rev() {
            coords[d] = rem % (out.shape[d].max(1) as usize);
            rem /= out.shape[d].max(1) as usize;
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
        out_data[oi] = apply(op, a_data[ai], b_data[bi]);
    }
}

/// Elementwise binary op with broadcasting. Both operands must share dtype.
/// When dtypes differ (e.g. i64 scalar + f32 tensor), the integer operand
/// is promoted to f32 so operations like batch_norm's num_batches_tracked
/// (i64) can flow through arithmetic with f32 tensors without cascading.
pub fn binary(op: BinaryOp, a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let (a_dtype, b_dtype) = (a.dtype, b.dtype);
    if a_dtype != b_dtype {
        // Promotion: integer scalar + float tensor -> float
        let is_a_int = matches!(a_dtype, DType::I64 | DType::I32 | DType::Bool);
        let is_b_int = matches!(b_dtype, DType::I64 | DType::I32 | DType::Bool);
        let target = if is_a_int && !is_b_int {
            b_dtype
        } else if !is_a_int && is_b_int {
            a_dtype
        } else {
            return Err(unsupported(&format!(
                "dtype mismatch in binary op: {} vs {}",
                a_dtype.name(),
                b_dtype.name()
            )));
        };
        // Cast the integer side to the float side's dtype.
        let a_owned = if is_a_int {
            crate::math_ops::to_dtype(a, target)?
        } else {
            // Need to copy the data since we only have a BorrowedTensor
            let n = crate::dlpack::elem_count(&a.shape);
            let mut owned = OwnedTensor::new(a.dtype, a.shape.clone());
            match a.dtype {
                DType::F32 => {
                    let src = unsafe { std::slice::from_raw_parts(a.data as *const f32, n) };
                    let dst = unsafe { std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n) };
                    dst.copy_from_slice(src);
                }
                DType::F64 => {
                    let src = unsafe { std::slice::from_raw_parts(a.data as *const f64, n) };
                    let dst = unsafe { std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n) };
                    dst.copy_from_slice(src);
                }
                _ => return Err(unsupported("binary promotion: cannot copy source dtype")),
            }
            owned
        };
        let b_owned = if is_b_int {
            crate::math_ops::to_dtype(b, target)?
        } else {
            let n = crate::dlpack::elem_count(&b.shape);
            let mut owned = OwnedTensor::new(b.dtype, b.shape.clone());
            match b.dtype {
                DType::F32 => {
                    let src = unsafe { std::slice::from_raw_parts(b.data as *const f32, n) };
                    let dst = unsafe { std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n) };
                    dst.copy_from_slice(src);
                }
                DType::F64 => {
                    let src = unsafe { std::slice::from_raw_parts(b.data as *const f64, n) };
                    let dst = unsafe { std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n) };
                    dst.copy_from_slice(src);
                }
                _ => return Err(unsupported("binary promotion: cannot copy source dtype")),
            }
            owned
        };
        let ap = BorrowedTensor::from_owned(&a_owned);
        let bp = BorrowedTensor::from_owned(&b_owned);
        let out_shape = broadcast_shape(&ap.shape, &bp.shape)?;
        let mut out = OwnedTensor::new(target, out_shape);
        match target {
            DType::F32 => run_binary::<f32>(op, &ap, &bp, &mut out),
            DType::F64 => run_binary::<f64>(op, &ap, &bp, &mut out),
            _ => return Err(unsupported("binary promotion: unsupported target dtype")),
        }
        return Ok(out);
    }
    let out_shape = broadcast_shape(&a.shape, &b.shape)?;
    let mut out = OwnedTensor::new(a.dtype, out_shape);
    match a.dtype {
        DType::F32 => run_binary::<f32>(op, a, b, &mut out),
        DType::F64 => run_binary::<f64>(op, a, b, &mut out),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
    Ok(out)
}

fn run_relu<T: Scalar>(a: &BorrowedTensor, out: &mut OwnedTensor) {
    // SAFETY: buffers sized by elem_count and dtype.
    let a_data = unsafe { typed_slice::<T>(a) };
    let out_data = unsafe {
        std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut T, out.elem_count())
    };
    let n = out.elem_count();
    let zero = T::zero();

    // Fast path: contiguous input -> linear loop (serial below the threshold,
    // parallel above; autovectorizes in both cases).
    if a.strides == contiguous_strides(&a.shape) {
        map_in_place(n, out_data, |i| if a_data[i] > zero { a_data[i] } else { zero });
        return;
    }

    // General path: honor strides via coordinate decomposition.
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
        let x = a_data[ai];
        out_data[i] = if x > zero { x } else { zero };
    }
}

/// `ReLU(x) = max(x, 0)` — elementwise, preserves shape/dtype.
pub fn relu(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => run_relu::<f32>(a, &mut out),
        DType::F64 => run_relu::<f64>(a, &mut out),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
    Ok(out)
}

/// Sanity helper for unit tests: total elements covered by a shape.
pub fn _elem_count_debug(shape: &[i64]) -> usize {
    elem_count(shape)
}
