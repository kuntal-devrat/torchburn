//! Metal native backend for Apple Silicon (bypasses WGPU).
//!
//! Direct Metal compute pipelines for unified memory architectures.
//! Uses metal-rs for low-level Metal API access, providing:
//! - Zero-copy weight access (unified memory)
//! - Optimized INT4/INT8 GEMV kernels
//! - Fused attention + RoPE kernels
//! - Direct MSL (Metal Shading Language) compute pipelines
//!
//! Feature-gated behind `metal-native`.

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod backend;

#[cfg(all(target_os = "macos", feature = "metal-native"))]
pub use backend::*;

#[cfg(not(all(target_os = "macos", feature = "metal-native")))]
mod fallback {
    /// Fallback when Metal native is not available.
    #[derive(Debug)]
    pub struct MetalBackend;

    impl MetalBackend {
        pub fn is_available() -> bool {
            false
        }

        pub fn name(&self) -> &'static str {
            "metal-native (disabled)"
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal-native")))]
pub use fallback::MetalBackend;
