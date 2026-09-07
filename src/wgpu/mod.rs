//! WGPU / Vulkan LLM decode engine (`feature = "burn-wgpu"`).
//!
//! - [`pipelines`] — compute pipelines + layouts built from `src/shaders/`.
//! - [`bind_groups`] — weight uploads and pre-baked per-layer bind groups.
//! - [`decode`] — the [`WgpuQwenDecoder`] pyclass and single-pass runtime.
//! - [`profiler`] — per-step timing and stage-by-stage debug readbacks.

pub mod backend;
pub(crate) mod bind_groups;
pub(crate) mod decode;
pub(crate) mod pipelines;
pub(crate) mod profiler;

pub(crate) use decode::WgpuQwenDecoder;
