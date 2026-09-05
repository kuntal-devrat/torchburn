//! Native (non-Burn) LLM inference decoders.
//!
//! [`decoder`] hosts the pure-Rust CPU decoder, which doubles as the numeric
//! reference the GPU path is regression-tested against
//! (`tests/test_wgpu_decoder_parity.py`).  GPU decoders live under
//! [`crate::wgpu`].

mod decoder;

#[cfg(feature = "burn-wgpu")]
pub(crate) use decoder::sample_logits;
pub(crate) use decoder::RustQwenDecoder;
