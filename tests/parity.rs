//! Rust parity / golden-value tests (Phase 0.4).
//!
//! Every kernel is checked against a naive reference implementation:
//!   - `f32`: GEMM + dot vs triple-loop references
//!   - `int4_g64`: grouped-int4 GEMV vs exact dequant-dot reference
//!   - `int8`: int8 dot vs reference
//!   - `decoder`: rms_norm + sampling vs references
//!   - `dispatch_tiers` (feature `dispatch-test` only): same inputs through
//!     every forced CPU tier must produce allclose outputs, and tier
//!     performance must be monotonic on the host.
//!
//! Run:
//!   cargo test --no-default-features --features matrixmultiply,dispatch-test --test parity

#[path = "parity/decoder.rs"]
mod decoder;
#[path = "parity/dispatch_tiers.rs"]
mod dispatch_tiers;
#[path = "parity/f32.rs"]
mod f32;
#[path = "parity/int4.rs"]
mod int4;
#[path = "parity/int8.rs"]
mod int8;
