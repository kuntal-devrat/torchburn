//! Native (non-Burn) LLM inference decoders.
//!
//! [`decoder`] hosts the pure-Rust CPU decoder, which doubles as the numeric
//! reference the GPU path is regression-tested against
//! (`tests/test_wgpu_decoder_parity.py`).  GPU decoders live under
//! [`crate::wgpu`].

mod decoder;

/// Checked capsule→slice reinterpretation (validates element size against
/// the requested type). Shared with the WGPU/CUDA decoders, which ingest the
/// same user-supplied weight capsules.
#[allow(unused_imports)]
pub(crate) use decoder::typed_slice;
pub(crate) use decoder::RustQwenDecoder;
/// Sampling entry point, exposed publicly for benches/parity tests and WGPU backend.
#[allow(unused_imports)]
pub use decoder::{apply_repetition_penalty, push_recent_window, sample_logits, RECENT_WINDOW};
