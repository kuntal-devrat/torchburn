//! Dispatch arms: RNN cells and transformer blocks. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        "rnn_tanh_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let b_ih = if node.args.len() > 4 {
                slot_view(slots, capsules, arg_index(node, 4)?).ok()
            } else {
                None
            };
            let b_hh = if node.args.len() > 5 {
                slot_view(slots, capsules, arg_index(node, 5)?).ok()
            } else {
                None
            };
            slots.push(Slot::Owned(kernels::linalg::rnn_tanh_cell(
                &x,
                &hx,
                &w_ih,
                &w_hh,
                b_ih.as_ref(),
                b_hh.as_ref(),
            )?));
        }
        "rnn_relu_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let b_ih = if node.args.len() > 4 {
                slot_view(slots, capsules, arg_index(node, 4)?).ok()
            } else {
                None
            };
            let b_hh = if node.args.len() > 5 {
                slot_view(slots, capsules, arg_index(node, 5)?).ok()
            } else {
                None
            };
            slots.push(Slot::Owned(kernels::linalg::rnn_relu_cell(
                &x,
                &hx,
                &w_ih,
                &w_hh,
                b_ih.as_ref(),
                b_hh.as_ref(),
            )?));
        }
        "gru_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let b_ih = if node.args.len() > 4 {
                slot_view(slots, capsules, arg_index(node, 4)?).ok()
            } else {
                None
            };
            let b_hh = if node.args.len() > 5 {
                slot_view(slots, capsules, arg_index(node, 5)?).ok()
            } else {
                None
            };
            slots.push(Slot::Owned(kernels::linalg::gru_cell(
                &x,
                &hx,
                &w_ih,
                &w_hh,
                b_ih.as_ref(),
                b_hh.as_ref(),
            )?));
        }
        "lstm_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let cx = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 4)?)?;
            let b_ih = if node.args.len() > 5 {
                slot_view(slots, capsules, arg_index(node, 5)?).ok()
            } else {
                None
            };
            let b_hh = if node.args.len() > 6 {
                slot_view(slots, capsules, arg_index(node, 6)?).ok()
            } else {
                None
            };
            let (h, c) = kernels::linalg::lstm_cell(
                &x,
                &hx,
                &cx,
                &w_ih,
                &w_hh,
                b_ih.as_ref(),
                b_hh.as_ref(),
            )?;
            slots.push(Slot::Tuple(vec![h, c]));
        }
        "multi_head_attention_forward" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(kernels::linalg::multi_head_attention_forward(
                &q, &k, &v,
            )?));
        }
        "transformer_encoder_layer_fwd" => {
            let src = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let num_heads = kw_usize(node, "num_heads", 1);
            let use_gelu = kw_bool(node, "use_gelu", true);
            let norm_first = kw_bool(node, "norm_first", false);
            let eps = kw_f64(node, "eps", 1e-5);
            let qkv_w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let qkv_b = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let proj_w = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let proj_b = slot_view(slots, capsules, arg_index(node, 4)?)?;
            let nw1 = slot_view(slots, capsules, arg_index(node, 5)?)?;
            let nb1 = slot_view(slots, capsules, arg_index(node, 6)?)?;
            let nw2 = slot_view(slots, capsules, arg_index(node, 7)?)?;
            let nb2 = slot_view(slots, capsules, arg_index(node, 8)?)?;
            let ffn_w1 = slot_view(slots, capsules, arg_index(node, 9)?)?;
            let ffn_b1 = slot_view(slots, capsules, arg_index(node, 10)?)?;
            let ffn_w2 = slot_view(slots, capsules, arg_index(node, 11)?)?;
            let ffn_b2 = slot_view(slots, capsules, arg_index(node, 12)?)?;
            let mask = if node.args.len() > 13 {
                Some(slot_view(slots, capsules, arg_index(node, 13)?)?)
            } else {
                None
            };
            slots.push(Slot::Owned(kernels::linalg::transformer_encoder_layer_fwd(
                &src,
                num_heads,
                use_gelu,
                norm_first,
                eps,
                &qkv_w,
                &qkv_b,
                &proj_w,
                &proj_b,
                &nw1,
                &nb1,
                &nw2,
                &nb2,
                &ffn_w1,
                &ffn_b1,
                &ffn_w2,
                &ffn_b2,
                mask.as_ref(),
            )?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
