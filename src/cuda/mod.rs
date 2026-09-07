//! CUDA backend for NVIDIA GPUs via cudarc.
//!
//! Provides GEMV, GEMM, and fused INT4/INT8 quantized kernels using cuBLAS
//! and custom CUDA compute kernels. This module is feature-gated behind `cuda`.
//!
//! # Architecture
//! - Lazy CUDA device initialization via `OnceLock<CudaDevice>`
//! - cuBLAS GEMM for dense f16/f32 matrix operations
//! - Custom INT4 GEMV kernel via raw CUDA kernel launches
//! - Persistent GPU weight buffers with content-addressed caching
//! - Async stream-based execution for overlapped data transfer

#[cfg(feature = "cuda")]
mod backend;

#[cfg(feature = "cuda")]
pub use backend::*;

#[cfg(not(feature = "cuda"))]
mod fallback {
    use std::sync::OnceLock;

    /// Fallback when CUDA feature is not enabled.
    #[derive(Debug)]
    pub struct CudaBackend;

    static FALLBACK: OnceLock<CudaBackend> = OnceLock::new();

    impl CudaBackend {
        pub fn is_available() -> bool {
            false
        }

        pub fn instance() -> &'static CudaBackend {
            FALLBACK.get_or_init(CudaBackend)
        }

        pub fn name(&self) -> &'static str {
            "cuda (disabled)"
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use fallback::CudaBackend;
