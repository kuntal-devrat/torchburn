//! Native tensor kernels: elementwise binary/relu plus the extended
//! math, index, scatter/sort and batch op families in submodules.
//!
//! Inputs are read directly from PyTorch-owned DLPack buffers (zero-copy),
//! respecting arbitrary stride layouts. Outputs are fresh Rust allocations.
//!
//! The `raw` re-exports below are the pointer/slice-level entry points for
//! criterion benches and Rust parity tests (no pyo3 types, linkable without
//! a Python interpreter).
pub mod elementwise;
pub mod linalg;
pub mod reductions;
pub mod special;
pub mod tensor_ops;

/// Raw kernel surface for criterion benches and Rust parity tests.
#[cfg(not(feature = "openblas"))]
pub use crate::linalg::gemm_f32_trans_b_into_accum;
#[cfg(not(feature = "openblas"))]
pub use crate::linalg::gemm_f64_trans_b_into_accum;
pub use crate::llm::sample_logits;
pub use crate::quantization::{
    dot_f32_f32, dot_f32_i8, f16_to_f32, f32_to_f16, fast_rms_norm, gemv_w4a32_grouped,
    gemv_w4a32_grouped_v2, pack_rows_w4a32_group64_v1_to_v2,
};

use crate::dlpack::{elem_count, unsupported, BorrowedTensor, DType, OwnedTensor};
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
        let ai = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let bi = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
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
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
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

// ── SIMD super-path: 8-wide f32, 4-wide f64 with AVX2/NEON ──
//
// Implemented as straight-line zipped slice loops: with `-C target-cpu=native`
// (and at baseline SSE2 for portable wheels) LLVM auto-vectorises these into
// unaligned vector loads/stores, which beats hand-rolling `wide` vector
// construction + `to_array` round-trips. Results are bit-identical to the
// scalar path (every output element is a function of only the inputs at that
// index), so parity tests are unaffected.

/// Binary elementwise engine over two contiguous equal-length inputs.
/// Splits across rayon chunks above `PAR_THRESHOLD`; the per-element closure
/// is monomorphised per op so the inner loop is branch-free.
#[inline(always)]
fn binary_zip_f32(a: &[f32], b: &[f32], out: &mut [f32], f: impl Fn(f32, f32) -> f32 + Sync) {
    let n = out.len();
    let run = |a: &[f32], b: &[f32], o: &mut [f32]| {
        for ((&av, &bv), ov) in a.iter().zip(b.iter()).zip(o.iter_mut()) {
            *ov = f(av, bv);
        }
    };
    if n >= PAR_THRESHOLD {
        use rayon::prelude::*;
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let s = ci * PAR_CHUNK;
                run(&a[s..s + chunk.len()], &b[s..s + chunk.len()], chunk);
            });
    } else {
        run(a, b, out);
    }
}

#[inline(always)]
fn binary_zip_f64(a: &[f64], b: &[f64], out: &mut [f64], f: impl Fn(f64, f64) -> f64 + Sync) {
    let n = out.len();
    let run = |a: &[f64], b: &[f64], o: &mut [f64]| {
        for ((&av, &bv), ov) in a.iter().zip(b.iter()).zip(o.iter_mut()) {
            *ov = f(av, bv);
        }
    };
    if n >= PAR_THRESHOLD {
        use rayon::prelude::*;
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let s = ci * PAR_CHUNK;
                run(&a[s..s + chunk.len()], &b[s..s + chunk.len()], chunk);
            });
    } else {
        run(a, b, out);
    }
}

/// Scalar-splat engine: `out[i] = f(a[i])` over a contiguous tensor, where the
/// closure captures the scalar operand (`f(x) = apply(op, x, s)` or
/// `f(x) = apply(op, s, x)`).
#[inline(always)]
fn binary_splat_f32(a: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32 + Sync) {
    let n = out.len();
    let run = |a: &[f32], o: &mut [f32]| {
        for (&av, ov) in a.iter().zip(o.iter_mut()) {
            *ov = f(av);
        }
    };
    if n >= PAR_THRESHOLD {
        use rayon::prelude::*;
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let s = ci * PAR_CHUNK;
                run(&a[s..s + chunk.len()], chunk);
            });
    } else {
        run(a, out);
    }
}

#[inline(always)]
fn binary_splat_f64(a: &[f64], out: &mut [f64], f: impl Fn(f64) -> f64 + Sync) {
    let n = out.len();
    let run = |a: &[f64], o: &mut [f64]| {
        for (&av, ov) in a.iter().zip(o.iter_mut()) {
            *ov = f(av);
        }
    };
    if n >= PAR_THRESHOLD {
        use rayon::prelude::*;
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let s = ci * PAR_CHUNK;
                run(&a[s..s + chunk.len()], chunk);
            });
    } else {
        run(a, out);
    }
}

#[inline(always)]
fn simd_binary_f32_contig(op: BinaryOp, a: &[f32], b: &[f32], out: &mut [f32]) {
    match op {
        BinaryOp::Add => binary_zip_f32(a, b, out, |x, y| x + y),
        BinaryOp::Sub => binary_zip_f32(a, b, out, |x, y| x - y),
        BinaryOp::Mul => binary_zip_f32(a, b, out, |x, y| x * y),
        BinaryOp::Div => binary_zip_f32(a, b, out, |x, y| x / y),
    }
}
#[inline(always)]
fn simd_binary_f64_contig(op: BinaryOp, a: &[f64], b: &[f64], out: &mut [f64]) {
    match op {
        BinaryOp::Add => binary_zip_f64(a, b, out, |x, y| x + y),
        BinaryOp::Sub => binary_zip_f64(a, b, out, |x, y| x - y),
        BinaryOp::Mul => binary_zip_f64(a, b, out, |x, y| x * y),
        BinaryOp::Div => binary_zip_f64(a, b, out, |x, y| x / y),
    }
}

/// ReLU `max(0, x)` zip engine (f32).
#[inline(always)]
fn relu_zip_f32(a: &[f32], out: &mut [f32]) {
    let n = out.len();
    let run = |a: &[f32], o: &mut [f32]| {
        for (&av, ov) in a.iter().zip(o.iter_mut()) {
            *ov = if av > 0.0 { av } else { 0.0 };
        }
    };
    if n >= PAR_THRESHOLD {
        use rayon::prelude::*;
        out.par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let s = ci * PAR_CHUNK;
                run(&a[s..s + chunk.len()], chunk);
            });
    } else {
        run(a, out);
    }
}

#[inline(always)]
fn simd_relu_f32_contig(a: &[f32], out: &mut [f32]) {
    relu_zip_f32(a, out);
}

fn run_binary<T: Scalar>(
    op: BinaryOp,
    a: &BorrowedTensor,
    b: &BorrowedTensor,
    out: &mut OwnedTensor,
) {
    // SAFETY: buffers sized by elem_count and dtype.
    let a_data = unsafe { typed_slice::<T>(a) };
    let b_data = unsafe { typed_slice::<T>(b) };
    let out_data = unsafe {
        std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut T, out.elem_count())
    };
    let n = out.elem_count();

    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();

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
        let is_a_int = matches!(
            a_dtype,
            DType::I64
                | DType::I32
                | DType::I8
                | DType::U8
                | DType::Bool
                | DType::F16
                | DType::BF16
        );
        let is_b_int = matches!(
            b_dtype,
            DType::I64
                | DType::I32
                | DType::I8
                | DType::U8
                | DType::Bool
                | DType::F16
                | DType::BF16
        );
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
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n)
                    };
                    dst.copy_from_slice(src);
                }
                DType::F64 => {
                    let src = unsafe { std::slice::from_raw_parts(a.data as *const f64, n) };
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n)
                    };
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
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f32, n)
                    };
                    dst.copy_from_slice(src);
                }
                DType::F64 => {
                    let src = unsafe { std::slice::from_raw_parts(b.data as *const f64, n) };
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(owned.data.as_mut_ptr() as *mut f64, n)
                    };
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
    let mut out = OwnedTensor::new(a.dtype, out_shape.clone());
    // ── Super-fast SIMD fast-path for identical contiguous shapes ──
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    if a.shape == b.shape && a_contig && b_contig && a.shape == out_shape {
        match a.dtype {
            DType::F32 => {
                let a_data = unsafe { typed_slice::<f32>(a) };
                let b_data = unsafe { typed_slice::<f32>(b) };
                let out_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.data.as_mut_ptr() as *mut f32,
                        out.elem_count(),
                    )
                };
                simd_binary_f32_contig(op, a_data, b_data, out_data);
                return Ok(out);
            }
            DType::F64 => {
                let a_data = unsafe { typed_slice::<f64>(a) };
                let b_data = unsafe { typed_slice::<f64>(b) };
                let out_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.data.as_mut_ptr() as *mut f64,
                        out.elem_count(),
                    )
                };
                simd_binary_f64_contig(op, a_data, b_data, out_data);
                return Ok(out);
            }
            _ => {}
        }
    }
    // ── Scalar + tensor (splat) ──
    if a.elem_count() == 1 && b_contig && b.shape == out_shape {
        match a.dtype {
            DType::F32 => {
                let scalar = unsafe { *typed_slice::<f32>(a).as_ptr() };
                let b_data = unsafe { typed_slice::<f32>(b) };
                let out_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.data.as_mut_ptr() as *mut f32,
                        out.elem_count(),
                    )
                };
                binary_splat_f32(b_data, out_data, |x| apply(op, scalar, x));
                return Ok(out);
            }
            DType::F64 => {
                let scalar = unsafe { *typed_slice::<f64>(a).as_ptr() };
                let b_data = unsafe { typed_slice::<f64>(b) };
                let out_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.data.as_mut_ptr() as *mut f64,
                        out.elem_count(),
                    )
                };
                binary_splat_f64(b_data, out_data, |x| apply(op, scalar, x));
                return Ok(out);
            }
            _ => {}
        }
    }
    if b.elem_count() == 1 && a_contig && a.shape == out_shape {
        match a.dtype {
            DType::F32 => {
                let scalar = unsafe { *typed_slice::<f32>(b).as_ptr() };
                let a_data = unsafe { typed_slice::<f32>(a) };
                let out_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.data.as_mut_ptr() as *mut f32,
                        out.elem_count(),
                    )
                };
                binary_splat_f32(a_data, out_data, |x| apply(op, x, scalar));
                return Ok(out);
            }
            DType::F64 => {
                let scalar = unsafe { *typed_slice::<f64>(b).as_ptr() };
                let a_data = unsafe { typed_slice::<f64>(a) };
                let out_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.data.as_mut_ptr() as *mut f64,
                        out.elem_count(),
                    )
                };
                binary_splat_f64(a_data, out_data, |x| apply(op, x, scalar));
                return Ok(out);
            }
            _ => {}
        }
    }
    match a.dtype {
        DType::F32 => run_binary::<f32>(op, a, b, &mut out),
        DType::F64 => run_binary::<f64>(op, a, b, &mut out),

        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
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
    if a.is_contiguous() {
        map_in_place(
            n,
            out_data,
            |i| if a_data[i] > zero { a_data[i] } else { zero },
        );
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
        DType::F32 => {
            if a.is_contiguous() {
                let a_data = unsafe { typed_slice::<f32>(a) };
                let out_data = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.data.as_mut_ptr() as *mut f32,
                        out.elem_count(),
                    )
                };
                simd_relu_f32_contig(a_data, out_data);
            } else {
                run_relu::<f32>(a, &mut out);
            }
        }
        DType::F64 => run_relu::<f64>(a, &mut out),

        DType::I64
        | DType::I32
        | DType::I8
        | DType::U8
        | DType::Bool
        | DType::F16
        | DType::BF16 => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

/// Sanity helper for unit tests: total elements covered by a shape.
pub fn _elem_count_debug(shape: &[i64]) -> usize {
    elem_count(shape)
}
