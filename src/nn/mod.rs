//! Neural-network layers: activations, attention, convolution, embedding,
//! losses, normalization, pooling, upsampling.
//!
//! Grouped from crate-root modules; paths changed `crate::X` → `crate::nn::X`
//! with no logic changes.

pub mod activations;
pub mod attention;
pub mod convolution;
pub mod embedding;
pub mod losses;
pub mod norm;
pub mod pooling;
pub mod upsample;
