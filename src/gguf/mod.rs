//! GGUF file parser and loader.
//!
//! Reads GGUF v3 files (llama.cpp's quantized format) and maps quantization
//! types onto TorchBurn's kernel set. Supports loading models directly from
//! community-quantized GGUF files without conversion.
//!
//! # Supported GGUF Quant Types
//! - `GGML_Q8_0`: INT8 per-block symmetric quantization → maps to W8A32 kernels
//! - `GGML_Q4_0`: INT4 per-block symmetric quantization → maps to W4A32 kernels
//! - `GGML_Q4_1`: INT4 per-block asymmetric quantization → maps to W4A32 kernels
//! - `GGML_F32`: FP32 full precision
//! - `GGML_F16`: FP16 half precision → converted to f32 for processing
//!
//! # GGUF v3 Format Reference
//! Magic: `GGUF` (4 bytes) | Version: u32 | Tensor count: u64 | Metadata KV: ...
//! | Tensor info: name + n_dims + dims + type + offset | Padding | Tensor data

mod parser;
mod types;

pub use parser::*;
pub use types::*;

/// GGUF magic bytes
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
/// Current supported GGUF version
pub const GGUF_VERSION: u32 = 3;
