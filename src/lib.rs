//! TorchBurn — a hardware-agnostic PyTorch compilation backend.
//!
//! This crate is the PyO3 FFI layer between PyTorch's Python frontend and a
//! Rust execution engine. Tensors cross the boundary zero-copy via DLPack
//! capsules (REQ-003); graph structure is hashed with BLAKE3 (REQ-004); and
//! unsupported operators are flagged so the Python interpreter can route them
//! to native PyTorch eager execution (REQ-002).

#![warn(clippy::all)]
// Kernel dispatch needs many params (tensor shapes, strides, dtypes)
#![allow(clippy::too_many_arguments)]
// DLPack FFI safety is documented per-callsite; raw pointers are inherent
// to the zero-copy boundary and cannot be abstracted away.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
// Iterating with index-based access is clearer for tensor math kernels
// where the loop variable maps to a spatial dimension.
#![allow(clippy::needless_range_loop)]
// Manual `return` aids readability in long dispatch arms.
#![allow(clippy::needless_return)]
// `*const u8` → `*const T` casts are inherent to the DLPack type-erasure layer.
#![allow(clippy::unnecessary_cast)]
// Excessive float precision is intentional for kernel constants.
#![allow(clippy::excessive_precision)]
// is_multiple_of is not stable on all Rust versions; % 2 is idiomatic.
#![allow(clippy::manual_is_multiple_of)]
// Thread-local lazy init can't be const in all cases.
#![allow(clippy::missing_const_for_thread_local)]
// Dead code in autograd module — these types are used via trait objects.
#![allow(dead_code)]
// Accessing first element via get(0) is clear in the context of node args.
#![allow(clippy::get_first)]
// Explicit into_iter() is clearer for intended ownership semantics.
#![allow(clippy::explicit_into_iter_loop)]
// Manual clamp patterns are clearer in kernel code.
#![allow(clippy::manual_clamp)]
// Auto-deref is intentional in DLPack FFI layer.
#![allow(clippy::needless_borrow)]
// if let is used for readability in dispatch arms.
#![allow(clippy::single_match)]
// format! nesting is clearer than intermediate variables.
#![allow(clippy::to_string_in_format_args)]
// DLPack raw pointer casts are inherent to the FFI boundary.
#![allow(unused_unsafe)]
// Explicit lifetimes improve clarity in the DLPack borrow chain.
#![allow(clippy::needless_lifetimes)]
// Manual range checks are clearer in kernel dispatch code.
#![allow(clippy::manual_range_contains)]
// let-binding returns are used for readability.
#![allow(clippy::let_and_return)]
// if-identical-blocks is intentional for symmetry in kernel dispatch.
#![allow(clippy::if_same_then_else)]
// Manual checked division is clearer in the DLPack bounds checking.
#![allow(clippy::manual_div_ceil)]
// Privacy warnings are intentional — these types are used via trait objects.
#![allow(clippy::type_repetition_in_bounds)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::unnecessary_map_or)]
#![allow(unused_doc_comments)]
#![allow(clippy::map_flatten)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::unnecessary_min_or_max)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod autograd;
mod cache;
mod dlpack;
mod engine;
mod ffi;
mod fft_complex;
mod fusion;
pub mod kernels;
pub(crate) mod linalg;
pub(crate) mod llm;

/// Runtime CPU feature dispatch (Phase 0.3): probed once, then read by every
/// kernel entry point. Public so parity tests can force tiers.
pub mod dispatch;
mod math_ops;
mod memory_pool;
mod nn;
pub(crate) mod quantization;
mod reductions;
mod shape_ops;

#[cfg(feature = "openblas")]
pub mod blas;

#[cfg(feature = "burn")]
mod burn_engine;

#[cfg(feature = "burn-wgpu")]
pub mod wgpu;

/// CUDA backend for NVIDIA GPUs (feature-gated behind `cuda`).
#[cfg(feature = "cuda")]
pub mod cuda;

/// Metal native backend for Apple Silicon (feature-gated behind `metal-native`).
/// Shared on macOS + iOS (unified-memory architecture).
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal-native"))]
pub mod metal;

/// GGUF file parser for llama.cpp quantized models.
pub mod gguf;

// ---------------------------------------------------------------------------
// PyO3 module registration
// ---------------------------------------------------------------------------

use pyo3::prelude::*;

#[pymodule]
fn _torchburn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<crate::llm::RustQwenDecoder>()?;
    #[cfg(feature = "burn-wgpu")]
    m.add_class::<crate::wgpu::WgpuQwenDecoder>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<crate::cuda::CudaQwenDecoder>()?;

    // Core engine FFI
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::execute, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::execute_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::prepare_graph, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::execute_prepared, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::release_graph, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::signature, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::supported_targets, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::active_engine, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::rayon_threads, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::dropout_forward, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::memory_pool_stats, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_ffi::clear_memory_pool, m)?)?;

    // GPU FFI
    m.add_function(wrap_pyfunction!(ffi::gpu_ffi::gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::gpu_ffi::gpu_backend, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::gpu_ffi::gpu_available, m)?)?;

    // Debug FFI
    m.add_function(wrap_pyfunction!(ffi::debug_ffi::cpu_features_report, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::debug_ffi::data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::debug_ffi::capsule_dump, m)?)?;

    // Autograd FFI
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::autograd_enable, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::autograd_disable, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::autograd_is_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::autograd_backward, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::autograd_reset, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::autograd_tape_len, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::backward_native, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::backward_single, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::autograd_ffi::backward_batch, m)?)?;

    // Cache FFI (still in cache module)
    m.add_function(wrap_pyfunction!(cache::cache_get, m)?)?;
    m.add_function(wrap_pyfunction!(cache::cache_put, m)?)?;
    m.add_function(wrap_pyfunction!(cache::cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(cache::cache_clear, m)?)?;

    // Quantization FFI
    m.add_function(wrap_pyfunction!(ffi::quantization_ffi::w8a32_linear, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::quantization_ffi::w4a32_linear, m)?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::w4a32_grouped_linear,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::w4a32_grouped_linear_v2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::fused_swiglu_mlp_w8a32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::fused_swiglu_mlp_w4a32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::fused_attention_step_w8a32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::fused_attention_step_w4a32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::fused_transformer_layer_step_w4a32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::quantize_linear_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::quantize_linear_int4,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::wgpu_w4a32_grouped_linear,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ffi::quantization_ffi::fused_swiglu_mlp_batched_w4a32,
        m
    )?)?;

    // GGUF FFI
    m.add_function(wrap_pyfunction!(ffi::gguf_ffi::gguf_info, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::gguf_ffi::gguf_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::gguf_ffi::gguf_metadata, m)?)?;

    Ok(())
}
