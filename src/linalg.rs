//! Linear algebra operations: matmul, bmm, linear, dot.
//!
//! Matmul uses a tiled, rayon-parallel, explicitly SIMD implementation for
//! CPU f32 (the `wide` crate: SSE2/AVX2 on x86, NEON on ARM).  `linear` and
//! `addmm` write their output directly (bias in the zero-init, optional fused
//! activation in the epilogue) — no intermediate matmul buffer or copy.

use crate::dlpack::{contiguous_strides, unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;
use wide::f32x8;

#[cfg(not(feature = "openblas"))]
use matrixmultiply::{dgemm, sgemm};

#[cfg(feature = "openblas")]
use matrixmultiply::{dgemm as dgemm_mm, sgemm as sgemm_mm};

fn use_openblas() -> bool {
    #[cfg(not(feature = "openblas"))]
    {
        false
    }
    #[cfg(feature = "openblas")]
    {
        match std::env::var("TORCHBURN_MATMUL").as_deref() {
            Ok("matrixmultiply") | Ok("matmul") | Ok("sgemm") => false,
            _ => true,
        }
    }
}

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

// ---------------------------------------------------------------------------
// 2D Matmul: (M, K) x (K, N) -> (M, N)
// ---------------------------------------------------------------------------

/// GEMM wrapper: C = A @ B where A is (M,K), B is (K,N), C is (M,N).
/// alpha=1.0, beta=0.0 → C is zeroed and filled with A@B.
/// alpha=1.0, beta=1.0 → C += A@B (used by addmm/linear with bias pre-filled).
fn gemm_f32_into(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "openblas")]
    unsafe {
        crate::blas::sgemm_f32(
            m as i32,
            k as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(feature = "openblas"))]
    unsafe {
        if m == 1 && n >= 32 {
            use rayon::prelude::*;
            let a_p = a.as_ptr() as usize;
            let b_p = b.as_ptr() as usize;
            let out_p = out.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|j| {
                let a_ptr = a_p as *const f32;
                let b_ptr = b_p as *const f32;
                let out_ptr = out_p as *mut f32;
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += *a_ptr.add(p) * *b_ptr.add(p * n + j);
                }
                *out_ptr.add(j) = sum;
            });
            return;
        }
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn gemm_f64_into(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
    #[cfg(feature = "openblas")]
    unsafe {
        crate::blas::dgemm_f64(
            m as i32,
            k as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(feature = "openblas"))]
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// GEMM with beta=1.0: out += A @ B (out must be pre-initialized with bias).
fn gemm_f32_into_accum(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "openblas")]
    unsafe {
        crate::blas::sgemm_f32(
            m as i32,
            k as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            1.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(feature = "openblas"))]
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            1.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn gemm_f64_into_accum(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
    #[cfg(feature = "openblas")]
    unsafe {
        crate::blas::dgemm_f64(
            m as i32,
            k as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            1.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(feature = "openblas"))]
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            1.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Transposed-B GEMM: C = A @ B^T
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum0 = f32x8::ZERO;
    let mut sum1 = f32x8::ZERO;
    let mut sum2 = f32x8::ZERO;
    let mut sum3 = f32x8::ZERO;
    let chunks32 = len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let a0 = f32x8::from(std::ptr::read_unaligned(a.add(offset) as *const [f32; 8]));
        let b0 = f32x8::from(std::ptr::read_unaligned(b.add(offset) as *const [f32; 8]));
        sum0 = a0.mul_add(b0, sum0);

        let a1 = f32x8::from(std::ptr::read_unaligned(a.add(offset + 8) as *const [f32; 8]));
        let b1 = f32x8::from(std::ptr::read_unaligned(b.add(offset + 8) as *const [f32; 8]));
        sum1 = a1.mul_add(b1, sum1);

        let a2 = f32x8::from(std::ptr::read_unaligned(a.add(offset + 16) as *const [f32; 8]));
        let b2 = f32x8::from(std::ptr::read_unaligned(b.add(offset + 16) as *const [f32; 8]));
        sum2 = a2.mul_add(b2, sum2);

        let a3 = f32x8::from(std::ptr::read_unaligned(a.add(offset + 24) as *const [f32; 8]));
        let b3 = f32x8::from(std::ptr::read_unaligned(b.add(offset + 24) as *const [f32; 8]));
        sum3 = a3.mul_add(b3, sum3);

        offset += 32;
    }

    let mut sum = (sum0 + sum1) + (sum2 + sum3);

    let chunks8 = (len - offset) / 8;
    for _ in 0..chunks8 {
        let a0 = f32x8::from(std::ptr::read_unaligned(a.add(offset) as *const [f32; 8]));
        let b0 = f32x8::from(std::ptr::read_unaligned(b.add(offset) as *const [f32; 8]));
        sum = a0.mul_add(b0, sum);
        offset += 8;
    }

    let mut total = sum.reduce_add();
    while offset < len {
        total += *a.add(offset) * *b.add(offset);
        offset += 1;
    }
    total
}

/// C = A @ B^T where A is (M, K), B is (N, K) in memory (i.e. B^T is (K, N)).
/// B is stored row-major (N, K): row_stride = K, col_stride = 1.
/// To transpose, we pass row_stride = 1, col_stride = N.
#[cfg(not(feature = "openblas"))]
fn gemm_f32_trans_b_into(
    a: *const f32,
    m: usize,
    k: usize,
    b: *const f32,
    b_n: usize,
    out: *mut f32,
    n: usize,
) {
    if m <= 8 && n >= 32 {
        use rayon::prelude::*;
        let a_p = a as usize;
        let b_p = b as usize;
        let out_p = out as usize;
        (0..n).into_par_iter().for_each(|j| {
            let a_ptr = a_p as *const f32;
            let b_ptr = (b_p as *const f32).wrapping_add(j * k);
            let out_ptr = out_p as *mut f32;
            for r in 0..m {
                let a_row = unsafe { a_ptr.add(r * k) };
                let val = unsafe { dot_f32(a_row, b_ptr, k) };
                unsafe {
                    *out_ptr.add(r * n + j) = val;
                }
            }
        });
        return;
    }
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a,
            k as isize,
            1,
            b,
            1,
            b_n as isize, // transposed: row_stride=1, col_stride=N
            0.0,
            out,
            n as isize,
            1,
        );
    }
}

#[cfg(not(feature = "openblas"))]
fn gemm_f32_trans_b_into_accum(
    a: *const f32,
    m: usize,
    k: usize,
    b: *const f32,
    b_n: usize,
    out: *mut f32,
    n: usize,
) {
    if m <= 8 && n >= 32 {
        use rayon::prelude::*;
        let a_p = a as usize;
        let b_p = b as usize;
        let out_p = out as usize;
        (0..n).into_par_iter().for_each(|j| {
            let a_ptr = a_p as *const f32;
            let b_ptr = (b_p as *const f32).wrapping_add(j * k);
            let out_ptr = out_p as *mut f32;
            for r in 0..m {
                let a_row = unsafe { a_ptr.add(r * k) };
                let val = unsafe { dot_f32(a_row, b_ptr, k) };
                unsafe {
                    *out_ptr.add(r * n + j) += val;
                }
            }
        });
        return;
    }
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a,
            k as isize,
            1,
            b,
            1,
            b_n as isize,
            1.0,
            out,
            n as isize,
            1,
        );
    }
}

#[cfg(not(feature = "openblas"))]
fn gemm_f64_trans_b_into(
    a: *const f64,
    m: usize,
    k: usize,
    b: *const f64,
    b_n: usize,
    out: *mut f64,
    n: usize,
) {
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a,
            k as isize,
            1,
            b,
            1,
            b_n as isize,
            0.0,
            out,
            n as isize,
            1,
        );
    }
}

#[cfg(not(feature = "openblas"))]
fn gemm_f64_trans_b_into_accum(
    a: *const f64,
    m: usize,
    k: usize,
    b: *const f64,
    b_n: usize,
    out: *mut f64,
    n: usize,
) {
    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a,
            k as isize,
            1,
            b,
            1,
            b_n as isize,
            1.0,
            out,
            n as isize,
            1,
        );
    }
}

#[cfg(feature = "openblas")]
fn gemm_f32_trans_b_into(
    a: *const f32,
    m: usize,
    k: usize,
    b: *const f32,
    b_n: usize,
    out: *mut f32,
    n: usize,
) {
    unsafe {
        crate::blas::cblas_sgemm(
            crate::blas::CBLAS_ROW_MAJOR,
            crate::blas::CBLAS_NO_TRANS, // A: no trans
            crate::blas::CBLAS_TRANS,    // B: transpose
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            b_n as i32,
            0.0,
            out,
            n as i32,
        );
    }
}

#[cfg(feature = "openblas")]
fn gemm_f32_trans_b_into_accum(
    a: *const f32,
    m: usize,
    k: usize,
    b: *const f32,
    b_n: usize,
    out: *mut f32,
    n: usize,
) {
    unsafe {
        crate::blas::cblas_sgemm(
            crate::blas::CBLAS_ROW_MAJOR,
            crate::blas::CBLAS_NO_TRANS,
            crate::blas::CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            b_n as i32,
            1.0,
            out,
            n as i32,
        );
    }
}

#[cfg(feature = "openblas")]
fn gemm_f64_trans_b_into(
    a: *const f64,
    m: usize,
    k: usize,
    b: *const f64,
    b_n: usize,
    out: *mut f64,
    n: usize,
) {
    unsafe {
        crate::blas::cblas_dgemm(
            crate::blas::CBLAS_ROW_MAJOR,
            crate::blas::CBLAS_NO_TRANS,
            crate::blas::CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            b_n as i32,
            0.0,
            out,
            n as i32,
        );
    }
}

#[cfg(feature = "openblas")]
fn gemm_f64_trans_b_into_accum(
    a: *const f64,
    m: usize,
    k: usize,
    b: *const f64,
    b_n: usize,
    out: *mut f64,
    n: usize,
) {
    unsafe {
        crate::blas::cblas_dgemm(
            crate::blas::CBLAS_ROW_MAJOR,
            crate::blas::CBLAS_NO_TRANS,
            crate::blas::CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            b_n as i32,
            1.0,
            out,
            n as i32,
        );
    }
}

fn matmul_2d_f32(a: &BorrowedTensor, b: &BorrowedTensor, out: &mut OwnedTensor) {
    let m = a.shape[0] as usize;
    let k = a.shape[1] as usize;
    let n = b.shape[1] as usize;
    let out_data = unsafe { typed_mut_slice::<f32>(out) };

    let a_contig = a.strides == contiguous_strides(&a.shape);
    let b_contig = b.strides == contiguous_strides(&b.shape);

    if a_contig && b_contig {
        let a_data = unsafe { typed_slice::<f32>(a) };
        let b_data = unsafe { typed_slice::<f32>(b) };
        gemm_f32_into(a_data, b_data, out_data, m, k, n);
    } else {
        // Strided inputs: materialize contiguous copies, then the tiled GEMM.
        let mut a_c = vec![0.0f32; m * k];
        let mut b_c = vec![0.0f32; k * n];
        let a_data = unsafe { typed_slice::<f32>(a) };
        let b_data = unsafe { typed_slice::<f32>(b) };
        for i in 0..m {
            for j in 0..k {
                let ai = i * a.strides[0] as usize + j * a.strides[1] as usize;
                a_c[i * k + j] = a_data[ai];
            }
        }
        for i in 0..k {
            for j in 0..n {
                let bi = i * b.strides[0] as usize + j * b.strides[1] as usize;
                b_c[i * n + j] = b_data[bi];
            }
        }
        gemm_f32_into(&a_c, &b_c, out_data, m, k, n);
    }
}

fn matmul_2d_f64(a: &BorrowedTensor, b: &BorrowedTensor, out: &mut OwnedTensor) {
    let m = a.shape[0] as usize;
    let k = a.shape[1] as usize;
    let n = b.shape[1] as usize;
    let out_data = unsafe { typed_mut_slice::<f64>(out) };

    let a_contig = a.strides == contiguous_strides(&a.shape);
    let b_contig = b.strides == contiguous_strides(&b.shape);

    if a_contig && b_contig {
        let a_data = unsafe { typed_slice::<f64>(a) };
        let b_data = unsafe { typed_slice::<f64>(b) };
        gemm_f64_into(a_data, b_data, out_data, m, k, n);
    } else {
        let mut a_c = vec![0.0f64; m * k];
        let mut b_c = vec![0.0f64; k * n];
        let a_data = unsafe { typed_slice::<f64>(a) };
        let b_data = unsafe { typed_slice::<f64>(b) };
        for i in 0..m {
            for j in 0..k {
                let ai = i * a.strides[0] as usize + j * a.strides[1] as usize;
                a_c[i * k + j] = a_data[ai];
            }
        }
        for i in 0..k {
            for j in 0..n {
                let bi = i * b.strides[0] as usize + j * b.strides[1] as usize;
                b_c[i * n + j] = b_data[bi];
            }
        }
        gemm_f64_into(&a_c, &b_c, out_data, m, k, n);
    }
}

/// 2D matrix multiply: (M, K) x (K, N) -> (M, N).
pub fn matmul_2d(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.dtype != b.dtype {
        return Err(unsupported(&format!(
            "dtype mismatch in matmul: {} vs {}",
            a.dtype.name(),
            b.dtype.name()
        )));
    }
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(unsupported("matmul_2d requires 2D tensors"));
    }
    if a.shape[1] != b.shape[0] {
        return Err(unsupported(&format!(
            "matmul shape mismatch: ({}, {}) x ({}, {})",
            a.shape[0], a.shape[1], b.shape[0], b.shape[1]
        )));
    }
    let m = a.shape[0];
    let n = b.shape[1];
    let mut out = OwnedTensor::new(a.dtype, vec![m, n]);
    match a.dtype {
        DType::F32 => matmul_2d_f32(a, b, &mut out),
        DType::F64 => matmul_2d_f64(a, b, &mut out),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Batched matmul (bmm): (B, M, K) x (B, K, N) -> (B, M, N)
// ---------------------------------------------------------------------------

pub fn bmm(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.dtype != b.dtype {
        return Err(unsupported(&format!(
            "dtype mismatch in bmm: {} vs {}",
            a.dtype.name(),
            b.dtype.name()
        )));
    }
    if a.shape.len() != 3 || b.shape.len() != 3 {
        return Err(unsupported("bmm requires 3D tensors"));
    }
    if a.shape[0] != b.shape[0] {
        return Err(unsupported(&format!(
            "bmm batch mismatch: {} vs {}",
            a.shape[0], b.shape[0]
        )));
    }
    if a.shape[2] != b.shape[1] {
        return Err(unsupported(&format!(
            "bmm inner dimension mismatch: {} vs {}",
            a.shape[2], b.shape[1]
        )));
    }

    let _a_contig;
    let a = if a.strides != contiguous_strides(&a.shape) {
        _a_contig = crate::shape_ops::to_contiguous(a)?;
        BorrowedTensor::from_owned(&_a_contig)
    } else {
        a.clone()
    };
    let a = &a;

    let _b_contig;
    let b = if b.strides != contiguous_strides(&b.shape) {
        _b_contig = crate::shape_ops::to_contiguous(b)?;
        BorrowedTensor::from_owned(&_b_contig)
    } else {
        b.clone()
    };
    let b = &b;

    let batch = a.shape[0] as usize;
    let m = a.shape[1] as usize;
    let k = a.shape[2] as usize;
    let n = b.shape[2] as usize;

    let mut out = OwnedTensor::new(a.dtype, vec![a.shape[0], a.shape[1], b.shape[2]]);

    // Rayon-parallel batch GEMM: each batch element is independent.
    let a_ptr = a.data;
    let b_ptr = b.data;
    let out_ptr = out.data.as_mut_ptr();
    let elem_size = a.dtype.elem_size();
    let chunk_a = m * k;
    let chunk_b = k * n;
    let chunk_o = m * n;

    // Parallelize when batch > 1 and enough work per batch.
    let parallel = batch > 1 && (chunk_a + chunk_b + chunk_o) > 4096;
    if parallel {
        match a.dtype {
            DType::F32 => {
                // SAFETY: rayon::scope guarantees all threads join before return.
                // Each batch writes to a distinct slice of out.
                unsafe {
                    let a_p0 = a_ptr as *const f32 as usize;
                    let b_p0 = b_ptr as *const f32 as usize;
                    let o_p0 = out_ptr as *mut f32 as usize;
                    rayon::scope(|s| {
                        for bi in 0..batch {
                            let a_p = a_p0;
                            let b_p = b_p0;
                            let o_p = o_p0;
                            s.spawn(move |_| {
                                let a_off = bi * chunk_a;
                                let b_off = bi * chunk_b;
                                let o_off = bi * chunk_o;
                                let ap = (a_p as *const f32).add(a_off);
                                let bp = (b_p as *const f32).add(b_off);
                                let op = (o_p as *mut f32).add(o_off);
                                gemm_f32_into(
                                    std::slice::from_raw_parts(ap, chunk_a),
                                    std::slice::from_raw_parts(bp, chunk_b),
                                    std::slice::from_raw_parts_mut(op, chunk_o),
                                    m,
                                    k,
                                    n,
                                );
                            });
                        }
                    });
                }
            }
            DType::F64 => unsafe {
                let a_p0 = a_ptr as *const f64 as usize;
                let b_p0 = b_ptr as *const f64 as usize;
                let o_p0 = out_ptr as *mut f64 as usize;
                rayon::scope(|s| {
                    for bi in 0..batch {
                        let a_p = a_p0;
                        let b_p = b_p0;
                        let o_p = o_p0;
                        s.spawn(move |_| {
                            let a_off = bi * chunk_a;
                            let b_off = bi * chunk_b;
                            let o_off = bi * chunk_o;
                            let ap = (a_p as *const f64).add(a_off);
                            let bp = (b_p as *const f64).add(b_off);
                            let op = (o_p as *mut f64).add(o_off);
                            gemm_f64_into(
                                std::slice::from_raw_parts(ap, chunk_a),
                                std::slice::from_raw_parts(bp, chunk_b),
                                std::slice::from_raw_parts_mut(op, chunk_o),
                                m,
                                k,
                                n,
                            );
                        });
                    }
                });
            },
            _ => return Err(unsupported("this kernel only supports f32/f64 tensors")),
        }
    } else {
        // Small batch or single element: sequential.
        for bi in 0..batch {
            let a_offset = bi * chunk_a;
            let b_offset = bi * chunk_b;
            let o_offset = bi * chunk_o;
            let a_view = BorrowedTensor {
                data: unsafe { (a_ptr as *const u8).add(a_offset * elem_size) },
                shape: vec![m as i64, k as i64],
                strides: contiguous_strides(&[m as i64, k as i64]),
                dtype: a.dtype,
            };
            let b_view = BorrowedTensor {
                data: unsafe { (b_ptr as *const u8).add(b_offset * elem_size) },
                shape: vec![k as i64, n as i64],
                strides: contiguous_strides(&[k as i64, n as i64]),
                dtype: b.dtype,
            };
            match a.dtype {
                DType::F32 => {
                    let a_slice = unsafe { typed_slice::<f32>(&a_view) };
                    let b_slice = unsafe { typed_slice::<f32>(&b_view) };
                    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
                    gemm_f32_into(
                        a_slice,
                        b_slice,
                        &mut out_data[o_offset..o_offset + chunk_o],
                        m,
                        k,
                        n,
                    );
                }
                DType::F64 => {
                    let a_slice = unsafe { typed_slice::<f64>(&a_view) };
                    let b_slice = unsafe { typed_slice::<f64>(&b_view) };
                    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
                    gemm_f64_into(
                        a_slice,
                        b_slice,
                        &mut out_data[o_offset..o_offset + chunk_o],
                        m,
                        k,
                        n,
                    );
                }
                _ => return Err(unsupported("this kernel only supports f32/f64 tensors")),
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Linear: (B, I) x (I, O) + (O,) -> (B, O) or (I,) x (I, O) + (O,) -> (O,)
// ---------------------------------------------------------------------------

#[inline(always)]
fn apply_epilogue_f32(out_data: &mut [f32], ep: &crate::fusion::ActSpec) {
    if out_data.len() >= 16 * 1024 {
        use rayon::prelude::*;
        out_data
            .par_chunks_mut(16 * 1024)
            .for_each(|chunk| match ep.kind {
                crate::fusion::UnaryKind::Relu => {
                    for v in chunk.iter_mut() {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    }
                }
                _ => {
                    for v in chunk.iter_mut() {
                        *v = crate::fusion::apply_unary_f32(ep.kind, *v, ep.params);
                    }
                }
            });
    } else {
        match ep.kind {
            crate::fusion::UnaryKind::Relu => {
                for v in out_data.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            _ => {
                for v in out_data.iter_mut() {
                    *v = crate::fusion::apply_unary_f32(ep.kind, *v, ep.params);
                }
            }
        }
    }
}

#[inline(always)]
fn apply_epilogue_f64(out_data: &mut [f64], ep: &crate::fusion::ActSpec) {
    if out_data.len() >= 16 * 1024 {
        use rayon::prelude::*;
        out_data
            .par_chunks_mut(16 * 1024)
            .for_each(|chunk| match ep.kind {
                crate::fusion::UnaryKind::Relu => {
                    for v in chunk.iter_mut() {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    }
                }
                _ => {
                    for v in chunk.iter_mut() {
                        *v = crate::fusion::apply_unary_f64(ep.kind, *v, ep.params);
                    }
                }
            });
    } else {
        match ep.kind {
            crate::fusion::UnaryKind::Relu => {
                for v in out_data.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            _ => {
                for v in out_data.iter_mut() {
                    *v = crate::fusion::apply_unary_f64(ep.kind, *v, ep.params);
                }
            }
        }
    }
}

// aten.addmm(bias, mat1, mat2) = mat1 @ mat2 + bias, where mat2 is (I, O)
// (NOT transposed — unlike `linear`, whose weight is (O, I)).
pub fn addmm(
    bias: &BorrowedTensor,
    mat1: &BorrowedTensor,
    mat2: &BorrowedTensor,
    epilogue: Option<&crate::fusion::ActSpec>,
) -> PyResult<OwnedTensor> {
    if mat1.dtype != mat2.dtype || mat1.dtype != bias.dtype {
        return Err(unsupported(&format!(
            "dtype mismatch in addmm: {} vs {} vs {}",
            mat1.dtype.name(),
            mat2.dtype.name(),
            bias.dtype.name()
        )));
    }
    if mat1.shape.len() != 2 || mat2.shape.len() != 2 {
        return Err(unsupported("addmm requires 2D mat1 and mat2"));
    }
    if mat1.shape[1] != mat2.shape[0] {
        return Err(unsupported(&format!(
            "addmm shape mismatch: ({}, {}) x ({}, {})",
            mat1.shape[0], mat1.shape[1], mat2.shape[0], mat2.shape[1]
        )));
    }
    let m = mat1.shape[0] as usize;
    let n = mat2.shape[1] as usize;
    if bias.shape != [n as i64] {
        return Err(unsupported(&format!(
            "addmm bias shape {:?} != [{n}]",
            bias.shape
        )));
    }
    let mut out = OwnedTensor::new(mat1.dtype, vec![mat1.shape[0], mat2.shape[1]]);
    let k = mat1.shape[1] as usize;
    match mat1.dtype {
        DType::F32 => {
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            let bias_data = unsafe { typed_slice::<f32>(bias) };
            for row in 0..m {
                out_data[row * n..(row + 1) * n].copy_from_slice(bias_data);
            }
            let a_contig = mat1.strides == contiguous_strides(&mat1.shape);
            let b_contig = mat2.strides == contiguous_strides(&mat2.shape);
            if a_contig && b_contig {
                let a_data = unsafe { typed_slice::<f32>(mat1) };
                let b_data = unsafe { typed_slice::<f32>(mat2) };
                gemm_f32_into_accum(a_data, b_data, out_data, m, k, n);
            } else {
                let mut a_c = vec![0.0f32; m * k];
                let mut b_c = vec![0.0f32; k * n];
                let a_data = unsafe { typed_slice::<f32>(mat1) };
                let b_data = unsafe { typed_slice::<f32>(mat2) };
                for i in 0..m {
                    for j in 0..k {
                        let ai = i * mat1.strides[0] as usize + j * mat1.strides[1] as usize;
                        a_c[i * k + j] = a_data[ai];
                    }
                }
                for i in 0..k {
                    for j in 0..n {
                        let bi = i * mat2.strides[0] as usize + j * mat2.strides[1] as usize;
                        b_c[i * n + j] = b_data[bi];
                    }
                }
                gemm_f32_into_accum(&a_c, &b_c, out_data, m, k, n);
            }
            if let Some(ep) = epilogue {
                apply_epilogue_f32(out_data, ep);
            }
        }
        DType::F64 => {
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            let bias_data = unsafe { typed_slice::<f64>(bias) };
            for row in 0..m {
                out_data[row * n..(row + 1) * n].copy_from_slice(bias_data);
            }
            let a_contig = mat1.strides == contiguous_strides(&mat1.shape);
            let b_contig = mat2.strides == contiguous_strides(&mat2.shape);
            if a_contig && b_contig {
                let a_data = unsafe { typed_slice::<f64>(mat1) };
                let b_data = unsafe { typed_slice::<f64>(mat2) };
                gemm_f64_into_accum(a_data, b_data, out_data, m, k, n);
            } else {
                let mut a_c = vec![0.0f64; m * k];
                let mut b_c = vec![0.0f64; k * n];
                let a_data = unsafe { typed_slice::<f64>(mat1) };
                let b_data = unsafe { typed_slice::<f64>(mat2) };
                for i in 0..m {
                    for j in 0..k {
                        let ai = i * mat1.strides[0] as usize + j * mat1.strides[1] as usize;
                        a_c[i * k + j] = a_data[ai];
                    }
                }

                for i in 0..k {
                    for j in 0..n {
                        let bi = i * mat2.strides[0] as usize + j * mat2.strides[1] as usize;
                        b_c[i * n + j] = b_data[bi];
                    }
                }
                gemm_f64_into_accum(&a_c, &b_c, out_data, m, k, n);
            }
            if let Some(ep) = epilogue {
                apply_epilogue_f64(out_data, ep);
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

pub fn linear(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    epilogue: Option<&crate::fusion::ActSpec>,
) -> PyResult<OwnedTensor> {
    if input.dtype != weight.dtype {
        return Err(unsupported(&format!(
            "dtype mismatch in linear: {} vs {}",
            input.dtype.name(),
            weight.dtype.name()
        )));
    }

    // weight is (O, I), input is (..., I)
    let weight_shape = &weight.shape;
    if weight_shape.len() != 2 {
        return Err(unsupported("linear weight must be 2D"));
    }

    let o = weight_shape[0] as usize;
    let i = weight_shape[1] as usize;

    let input_shape = &input.shape;
    let input_rank = input_shape.len();
    if input_shape[input_rank - 1] != i as i64 {
        return Err(unsupported(&format!(
            "linear input dim mismatch: last dim {} vs weight dim {}",
            input_shape[input_rank - 1],
            i
        )));
    }

    // Compute output shape: replace last dim with O
    let mut out_shape = input_shape.to_vec();
    out_shape[input_rank - 1] = o as i64;

    let mut out = OwnedTensor::new(input.dtype, out_shape);
    let n_batch: usize = input_shape[..input_rank - 1]
        .iter()
        .map(|&d| d.max(0) as usize)
        .product();
    let input_contig = input.strides == contiguous_strides(&input.shape);

    // linear(x, w, b) == x @ w^T + b.
    // Instead of transposing weight (O,I) -> (I,O), use transposed-B GEMM
    // which reads weight in-place with stride tricks (zero extra allocation).
    match input.dtype {
        DType::F32 => {
            // Contiguous (n_batch, I) view of the input (copy if strided).
            let a_buf: Option<Vec<f32>> = if input_contig {
                None
            } else {
                let mut buf = vec![0.0f32; n_batch * i];
                for batch in 0..n_batch {
                    let base = batch_offset(input, batch);
                    let row = &mut buf[batch * i..(batch + 1) * i];
                    for k in 0..i {
                        row[k] = unsafe {
                            *((input.data as *const f32)
                                .add(base + k * input.strides[input_rank - 1] as usize))
                        };
                    }
                }
                Some(buf)
            };
            let a_ptr = match &a_buf {
                Some(buf) => buf.as_ptr() as *const f32,
                None => input.data as *const f32,
            };
            let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
            match bias {
                Some(b) => {
                    let b_data = unsafe { typed_slice::<f32>(b) };
                    for row in 0..n_batch {
                        out_data[row * o..(row + 1) * o].copy_from_slice(b_data);
                    }
                }
                None => out_data.fill(0.0),
            }
            // weight is (O, I) in memory.  trans_b reads it as (K=N, M=O)
            // with row_stride=1, col_stride=O — zero-copy transpose.
            let w_ptr = weight.data as *const f32;
            gemm_f32_trans_b_into_accum(a_ptr, n_batch, i, w_ptr, i, out_data.as_mut_ptr(), o);
            if let Some(ep) = epilogue {
                apply_epilogue_f32(out_data, ep);
            }
        }
        DType::F64 => {
            let a_buf: Option<Vec<f64>> = if input_contig {
                None
            } else {
                let mut buf = vec![0.0f64; n_batch * i];
                for batch in 0..n_batch {
                    let base = batch_offset(input, batch);
                    let row = &mut buf[batch * i..(batch + 1) * i];
                    for k in 0..i {
                        row[k] = unsafe {
                            *((input.data as *const f64)
                                .add(base + k * input.strides[input_rank - 1] as usize))
                        };
                    }
                }
                Some(buf)
            };
            let a_ptr = match &a_buf {
                Some(buf) => buf.as_ptr() as *const f64,
                None => input.data as *const f64,
            };
            let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
            match bias {
                Some(b) => {
                    let b_data = unsafe { typed_slice::<f64>(b) };
                    for row in 0..n_batch {
                        out_data[row * o..(row + 1) * o].copy_from_slice(b_data);
                    }
                }
                None => out_data.fill(0.0),
            }
            let w_ptr = weight.data as *const f64;
            gemm_f64_trans_b_into_accum(a_ptr, n_batch, i, w_ptr, i, out_data.as_mut_ptr(), o);
            if let Some(ep) = epilogue {
                apply_epilogue_f64(out_data, ep);
            }
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

/// Flat element offset of row `batch` (leading dims flattened) in a strided
/// rank-R input, assuming the leading dims are traversed row-major.
fn batch_offset(input: &BorrowedTensor, batch: usize) -> usize {
    let input_rank = input.shape.len();
    let mut idx = 0usize;
    let mut rem = batch;
    for d in (0..input_rank - 1).rev() {
        let dim_size = input.shape[d].max(0) as usize;
        idx += (rem % dim_size) * input.strides[d] as usize;
        rem /= dim_size;
    }
    idx
}

// ---------------------------------------------------------------------------
// Dot product: (N,) . (N,) -> scalar
// ---------------------------------------------------------------------------

pub fn dot(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.dtype != b.dtype {
        return Err(unsupported(&format!(
            "dtype mismatch in dot: {} vs {}",
            a.dtype.name(),
            b.dtype.name()
        )));
    }
    if a.shape.len() != 1 || b.shape.len() != 1 {
        return Err(unsupported("dot requires 1D tensors"));
    }
    if a.shape[0] != b.shape[0] {
        return Err(unsupported(&format!(
            "dot length mismatch: {} vs {}",
            a.shape[0], b.shape[0]
        )));
    }

    let n = a.shape[0] as usize;
    let mut out = OwnedTensor::new(a.dtype, vec![]);

    match a.dtype {
        DType::F32 => {
            let a_data = unsafe { typed_slice::<f32>(a) };
            let b_data = unsafe { typed_slice::<f32>(b) };
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += a_data[i] * b_data[i];
            }
            let d = unsafe { typed_mut_slice::<f32>(&mut out) };
            d[0] = sum;
        }
        DType::F64 => {
            let a_data = unsafe { typed_slice::<f64>(a) };
            let b_data = unsafe { typed_slice::<f64>(b) };
            let mut sum = 0.0f64;
            for i in 0..n {
                sum += a_data[i] * b_data[i];
            }
            let d = unsafe { typed_mut_slice::<f64>(&mut out) };
            d[0] = sum;
        }

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Batched matmul with broadcasting: handles (..., M, K) x (..., K, N)
// where leading dims can broadcast
// ---------------------------------------------------------------------------

pub fn matmul(a: &BorrowedTensor, b: &BorrowedTensor) -> PyResult<OwnedTensor> {
    if a.dtype != b.dtype {
        return Err(unsupported(&format!(
            "dtype mismatch in matmul: {} vs {}",
            a.dtype.name(),
            b.dtype.name()
        )));
    }
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    if a_rank < 2 || b_rank < 2 {
        return Err(unsupported("matmul requires at least 2D tensors"));
    }

    // Last two dims are the matrix dims
    let m = a.shape[a_rank - 2];
    let k_a = a.shape[a_rank - 1];
    let k_b = b.shape[b_rank - 2];
    let n = b.shape[b_rank - 1];

    if k_a != k_b {
        return Err(unsupported(&format!(
            "matmul inner dimension mismatch: {} vs {}",
            k_a, k_b
        )));
    }

    if a_rank == 2 && b_rank == 2 {
        return matmul_2d(a, b);
    }

    let _a_contig;
    let a = if a.strides != contiguous_strides(&a.shape) {
        _a_contig = crate::shape_ops::to_contiguous(a)?;
        BorrowedTensor::from_owned(&_a_contig)
    } else {
        a.clone()
    };
    let a = &a;

    let _b_contig;
    let b = if b.strides != contiguous_strides(&b.shape) {
        _b_contig = crate::shape_ops::to_contiguous(b)?;
        BorrowedTensor::from_owned(&_b_contig)
    } else {
        b.clone()
    };
    let b = &b;

    // For batched: fall back to naive batch loop
    // Compute batch shape (broadcast leading dims)
    let a_batch = &a.shape[..a_rank - 2];
    let b_batch = &b.shape[..b_rank - 2];
    let batch_rank = a_batch.len().max(b_batch.len());
    let mut batch_shape = vec![0i64; batch_rank];
    for i in 0..batch_rank {
        let ai = if i < batch_rank - a_batch.len() {
            1
        } else {
            a_batch[i - (batch_rank - a_batch.len())]
        };
        let bi = if i < batch_rank - b_batch.len() {
            1
        } else {
            b_batch[i - (batch_rank - b_batch.len())]
        };
        if ai == bi {
            batch_shape[i] = ai;
        } else if ai == 1 {
            batch_shape[i] = bi;
        } else if bi == 1 {
            batch_shape[i] = ai;
        } else {
            return Err(unsupported("matmul batch shape broadcast failed"));
        }
    }

    let batch_size: usize = batch_shape.iter().map(|&d| d.max(0) as usize).product();
    let mut out = OwnedTensor::new(a.dtype, [batch_shape.clone(), vec![m, n]].concat());

    // Rayon-parallel batch GEMM for >2D matmul.
    let m_usize = m as usize;
    let k_usize = k_a as usize;
    let n_usize = n as usize;
    let chunk_a = m_usize * k_usize;
    let chunk_b = k_usize * n_usize;
    let chunk_o = m_usize * n_usize;
    let a_ptr = a.data;
    let b_ptr = b.data;
    let out_ptr = out.data.as_mut_ptr();
    let elem_size = a.dtype.elem_size();
    let parallel = batch_size > 1 && (chunk_a + chunk_b + chunk_o) > 4096;
    // Precompute batch strides for broadcast-aware indexing
    let batch_rank = batch_shape.len();
    let mut batch_strides = vec![1usize; batch_rank];
    for i in (0..batch_rank.saturating_sub(1)).rev() {
        batch_strides[i] = batch_strides[i + 1] * batch_shape[i + 1] as usize;
    }
    let a_batch_vec = a_batch.to_vec();
    let b_batch_vec = b_batch.to_vec();
    let batch_shape_clone = batch_shape.clone();
    let batch_strides_clone = batch_strides.clone();

    // Helper to compute flat batch index for a given operand
    let flat_for = |batch: &[i64], bi: usize, strides: &[usize], shape: &[i64]| -> usize {
        if batch.is_empty() {
            return 0;
        }
        let rank = shape.len();
        let blen = batch.len();
        let mut flat = 0usize;
        let mut stride = 1usize;
        for j in (0..blen).rev() {
            let pos = rank - blen + j;
            let dim = batch[j] as usize;
            let coord = if dim == 1 {
                0
            } else {
                (bi / strides[pos]) % shape[pos] as usize
            };
            flat += coord * stride;
            stride *= dim;
        }
        flat
    };

    if parallel {
        match a.dtype {
            DType::F32 => unsafe {
                let a_p0 = a_ptr as *const f32 as usize;
                let b_p0 = b_ptr as *const f32 as usize;
                let o_p0 = out_ptr as *mut f32 as usize;
                let a_batch_c = a_batch_vec.clone();
                let b_batch_c = b_batch_vec.clone();
                let bs = batch_strides_clone.clone();
                let bshape = batch_shape_clone.clone();
                rayon::scope(|s| {
                    for bi in 0..batch_size {
                        let a_p = a_p0;
                        let b_p = b_p0;
                        let o_p = o_p0;
                        let a_batch_l = a_batch_c.clone();
                        let b_batch_l = b_batch_c.clone();
                        let bs_l = bs.clone();
                        let bshape_l = bshape.clone();
                        s.spawn(move |_| {
                            let a_flat = flat_for(&a_batch_l, bi, &bs_l, &bshape_l);
                            let b_flat = flat_for(&b_batch_l, bi, &bs_l, &bshape_l);
                            let a_off = a_flat * chunk_a;
                            let b_off = b_flat * chunk_b;
                            let o_off = bi * chunk_o;
                            let ap = (a_p as *const f32).add(a_off);
                            let bp = (b_p as *const f32).add(b_off);
                            let op = (o_p as *mut f32).add(o_off);
                            gemm_f32_into(
                                std::slice::from_raw_parts(ap, chunk_a),
                                std::slice::from_raw_parts(bp, chunk_b),
                                std::slice::from_raw_parts_mut(op, chunk_o),
                                m_usize,
                                k_usize,
                                n_usize,
                            );
                        });
                    }
                });
            },
            DType::F64 => unsafe {
                let a_p0 = a_ptr as *const f64 as usize;
                let b_p0 = b_ptr as *const f64 as usize;
                let o_p0 = out_ptr as *mut f64 as usize;
                let a_batch_c = a_batch_vec.clone();
                let b_batch_c = b_batch_vec.clone();
                let bs = batch_strides_clone.clone();
                let bshape = batch_shape_clone.clone();
                rayon::scope(|s| {
                    for bi in 0..batch_size {
                        let a_p = a_p0;
                        let b_p = b_p0;
                        let o_p = o_p0;
                        let a_batch_l = a_batch_c.clone();
                        let b_batch_l = b_batch_c.clone();
                        let bs_l = bs.clone();
                        let bshape_l = bshape.clone();
                        s.spawn(move |_| {
                            let a_flat = flat_for(&a_batch_l, bi, &bs_l, &bshape_l);
                            let b_flat = flat_for(&b_batch_l, bi, &bs_l, &bshape_l);
                            let a_off = a_flat * chunk_a;
                            let b_off = b_flat * chunk_b;
                            let o_off = bi * chunk_o;
                            let ap = (a_p as *const f64).add(a_off);
                            let bp = (b_p as *const f64).add(b_off);
                            let op = (o_p as *mut f64).add(o_off);
                            gemm_f64_into(
                                std::slice::from_raw_parts(ap, chunk_a),
                                std::slice::from_raw_parts(bp, chunk_b),
                                std::slice::from_raw_parts_mut(op, chunk_o),
                                m_usize,
                                k_usize,
                                n_usize,
                            );
                        });
                    }
                });
            },
            _ => return Err(unsupported("this kernel only supports f32/f64 tensors")),
        }
    } else {
        // Small batch: sequential.
        for bi in 0..batch_size {
            let a_flat = flat_for(&a_batch_vec, bi, &batch_strides, &batch_shape);
            let b_flat = flat_for(&b_batch_vec, bi, &batch_strides, &batch_shape);
            let a_offset = a_flat * chunk_a;
            let b_offset = b_flat * chunk_b;
            let o_offset = bi * chunk_o;
            let a_view = BorrowedTensor {
                data: unsafe { (a_ptr as *const u8).add(a_offset * elem_size) },
                shape: vec![m, k_a],
                strides: contiguous_strides(&[m, k_a]),
                dtype: a.dtype,
            };
            let b_view = BorrowedTensor {
                data: unsafe { (b_ptr as *const u8).add(b_offset * elem_size) },
                shape: vec![k_b, n],
                strides: contiguous_strides(&[k_b, n]),
                dtype: b.dtype,
            };
            match a.dtype {
                DType::F32 => {
                    let a_slice = unsafe { typed_slice::<f32>(&a_view) };
                    let b_slice = unsafe { typed_slice::<f32>(&b_view) };
                    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
                    gemm_f32_into(
                        a_slice,
                        b_slice,
                        &mut out_data[o_offset..o_offset + chunk_o],
                        m_usize,
                        k_usize,
                        n_usize,
                    );
                }
                DType::F64 => {
                    let a_slice = unsafe { typed_slice::<f64>(&a_view) };
                    let b_slice = unsafe { typed_slice::<f64>(&b_view) };
                    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
                    gemm_f64_into(
                        a_slice,
                        b_slice,
                        &mut out_data[o_offset..o_offset + chunk_o],
                        m_usize,
                        k_usize,
                        n_usize,
                    );
                }
                _ => return Err(unsupported("this kernel only supports f32/f64 tensors")),
            }
        }
    }
    Ok(out)
}
