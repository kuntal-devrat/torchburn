//! Per-op backward rules, split by op family.
pub mod binary;

#[deprecated(
    since = "1.0.1",
    note = "Legacy tape recording API; use direct batch backward instead"
)]
pub use self::binary::{record_add, record_div, record_mul, record_sub};

pub mod unary;

pub mod matmul_linear;

#[deprecated(
    since = "1.0.1",
    note = "Legacy tape recording API; use direct batch backward instead"
)]
pub use self::matmul_linear::{record_linear, record_matmul};

pub mod norm_act;

#[deprecated(
    since = "1.0.1",
    note = "Legacy tape recording API; use direct batch backward instead"
)]
pub use self::norm_act::{record_dropout, record_layer_norm, record_softmax};

pub mod shape;

#[deprecated(
    since = "1.0.1",
    note = "Legacy tape recording API; use direct batch backward instead"
)]
pub use self::shape::{record_cat, record_permute, record_reshape, record_sum};

pub mod losses;

#[deprecated(
    since = "1.0.1",
    note = "Legacy tape recording API; use direct batch backward instead"
)]
pub use self::losses::{record_mse_loss, record_nll_loss};
