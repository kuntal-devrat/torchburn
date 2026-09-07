//! Per-op backward rules, split by op family.
pub mod binary;

pub use self::binary::{record_add, record_div, record_mul, record_sub};

pub mod unary;

pub mod matmul_linear;

pub use self::matmul_linear::{record_linear, record_matmul};

pub mod norm_act;

pub use self::norm_act::{record_dropout, record_layer_norm, record_softmax};

pub mod shape;

pub use self::shape::{record_cat, record_permute, record_reshape, record_sum};

pub mod losses;

pub use self::losses::{record_mse_loss, record_nll_loss};
