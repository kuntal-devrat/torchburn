//! Per-op dispatch: route a plan Node to its native kernel.
//!
//! Domain arms live in submodules; this router tries each in turn.
//! Inherits the engine root imports via super.

use super::*;

pub mod d_act_loss;
pub mod d_batch2;
pub mod d_batch4;
pub mod d_blas_extra;
pub mod d_decomp;
pub mod d_elementwise;
pub mod d_fft;
pub mod d_float_special;
pub mod d_fused;
pub mod d_math_extra;
pub mod d_nn;
pub mod d_nn3d;
pub mod d_norm;
pub mod d_quant;
pub mod d_random;
pub mod d_reducers;
pub mod d_rnn;
pub mod d_scatter;
pub mod d_scatter2;
pub mod d_shape;
pub mod d_shape2;
pub mod d_solve;

/// Execute a node by dispatching to the appropriate kernel.
pub(crate) fn dispatch_node(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<()> {
    if d_elementwise::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_reducers::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_norm::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_shape::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_nn::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_scatter::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_math_extra::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_batch2::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_float_special::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_blas_extra::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_act_loss::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_decomp::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_scatter2::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_shape2::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_nn3d::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_random::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_rnn::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_solve::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_fused::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_quant::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_fft::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    if d_batch4::try_dispatch(node, slots, capsules)? {
        return Ok(());
    }
    Err(unsupported(&format!("unknown target {:?}", node.target)))
}
