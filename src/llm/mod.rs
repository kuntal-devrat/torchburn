//! Native (non-Burn) LLM inference decoders.
//!
//! [`decoder`] hosts the pure-Rust CPU decoder, which doubles as the numeric
//! reference the GPU path is regression-tested against
//! (`tests/test_wgpu_decoder_parity.py`).  GPU decoders live under
//! [`crate::wgpu`].

mod decoder;

/// Sampling entry point, exposed publicly for benches/parity tests.
pub use decoder::sample_logits;
pub(crate) use decoder::RustQwenDecoder;
