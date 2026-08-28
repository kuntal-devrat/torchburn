//! CBLAS FFI bindings for OpenBLAS.
//!
//! Provides raw C function declarations for `cblas_sgemm` and `cblas_dgemm`
//! so the Rust engine can call the vendor-tuned BLAS library directly.

// CBLAS constants (row-major ordering).
pub const CBLAS_ROW_MAJOR: i32 = 101;
pub const CBLAS_NO_TRANS: i32 = 111;
pub const CBLAS_TRANS: i32 = 112;

extern "C" {
    /// Single-precision GEMM: C = alpha * op(A) * op(B) + beta * C.
    ///
    /// `lda`, `ldb`, `ldc` are leading dimensions (row-major).
    pub fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

    /// Double-precision GEMM.
    pub fn cblas_dgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

/// Set the number of threads OpenBLAS uses for parallel execution.
pub fn set_num_threads(n: i32) {
    unsafe {
        openblas_set_num_threads(n);
    }
}

extern "C" {
    fn openblas_set_num_threads(n: i32);
}

/// Row-major GEMM wrapper: C = alpha * A @ B + beta * C.
///
/// A is (M, K), B is (K, N), C is (M, N), all row-major.
/// Parameters are (M, K, N) to match matrixmultiply's calling convention.
///
/// # Safety
/// Pointers `a`, `b`, and `c` must be valid, properly aligned, and point to
/// memory buffers of sufficient length for the specified matrix dimensions and strides.
#[inline]
pub unsafe fn sgemm_f32(
    m: i32,
    k: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    cblas_sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc,
    );
}

/// Row-major GEMM wrapper for f64.
///
/// # Safety
/// Pointers `a`, `b`, and `c` must be valid, properly aligned, and point to
/// memory buffers of sufficient length for the specified matrix dimensions and strides.
#[inline]
pub unsafe fn dgemm_f64(
    m: i32,
    k: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    cblas_dgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc,
    );
}
