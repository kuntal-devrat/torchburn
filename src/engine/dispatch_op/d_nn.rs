//! Dispatch arms: Creation, embedding, attention, losses, conv/pool/upsample. Inherits engine root via super; pure move.

use super::*;

pub(crate) fn try_dispatch(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<bool> {
    let target = node.target.as_str();

    match target {
        // Phase 4: scalar constants used in mask graphs (aten.scalar_tensor)
        "scalar_tensor" => {
            let v = kw_f64_allow_inf(node, "value", 0.0);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                _ => DType::F32,
            };
            let mut out = OwnedTensor::new(dtype, vec![]);
            match dtype {
                DType::F32 => {
                    let d = unsafe {
                        std::slice::from_raw_parts_mut(
                            out.data.as_mut_ptr() as *mut f32,
                            out.elem_count(),
                        )
                    };
                    d[0] = v as f32;
                }
                _ => {
                    let d = unsafe {
                        std::slice::from_raw_parts_mut(
                            out.data.as_mut_ptr() as *mut f64,
                            out.elem_count(),
                        )
                    };
                    d[0] = v;
                }
            }
            slots.push(Slot::Owned(out));
        }

        // Phase 10: tensor creation ops
        "full" => {
            let shape = kw_i64_vec(node, "shape");
            let value = kw_f64(node, "value", 0.0);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                Some("i32") => DType::I32,
                Some("bool") => DType::Bool,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::full(&shape, value, dtype)?));
        }
        "zeros" => {
            let shape = kw_i64_vec(node, "shape");
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::zeros(&shape, dtype)?));
        }
        "ones" => {
            let shape = kw_i64_vec(node, "shape");
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::ones(&shape, dtype)?));
        }
        "arange" => {
            let start = kw_f64(node, "start", 0.0);
            let end = kw_f64(node, "end", 0.0);
            let step = kw_f64(node, "step", 1.0);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::arange(start, end, step, dtype)?));
        }
        "linspace" => {
            let start = kw_f64(node, "start", 0.0);
            let end = kw_f64(node, "end", 0.0);
            let steps = kw_usize(node, "steps", 100);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::linspace(start, end, steps, dtype)?));
        }

        // Phase 4: embedding (weight, indices) — int64/int32 indices
        "embedding" => {
            let weight = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let indices = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(embedding::embedding(&weight, &indices)?));
        }

        // Phase 4: attention
        "scaled_dot_product_attention" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let mask = if node.args.len() > 3 {
                Some(slot_view(slots, capsules, arg_index(node, 3)?)?)
            } else {
                None
            };
            let is_causal = kw_bool(node, "is_causal", false);
            slots.push(Slot::Owned(attention::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                mask.as_ref(),
                is_causal,
            )?));
        }
        "rope" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let cos = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let sin = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(attention::rope(&x, &cos, &sin)?));
        }

        // Phase 4: losses
        "nll_loss_forward" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let target = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let reduction = kw_isize(node, "reduction", 1);
            let ignore_index = kw_isize(node, "ignore_index", -100);
            slots.push(Slot::Owned(losses::nll_loss_forward(
                &input,
                &target,
                reduction as i64,
                ignore_index as i64,
            )?));
        }
        "mse_loss" | "smooth_l1_loss" | "binary_cross_entropy" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let reduction = kw_isize(node, "reduction", 1);
            let beta = kw_f64(node, "beta", 1.0);
            match target {
                "mse_loss" => {
                    slots.push(Slot::Owned(losses::mse_loss(&a, &b, reduction as i64)?));
                }
                "smooth_l1_loss" => {
                    slots.push(Slot::Owned(losses::smooth_l1_loss(
                        &a,
                        &b,
                        reduction as i64,
                        beta,
                    )?));
                }
                _ => {
                    slots.push(Slot::Owned(losses::binary_cross_entropy(
                        &a,
                        &b,
                        reduction as i64,
                    )?));
                }
            }
        }

        // Phase 3: convolution
        "conv1d" | "conv2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = if node.args.len() > 2 {
                Some(slot_view(slots, capsules, arg_index(node, 2)?)?)
            } else {
                None
            };
            let groups = kw_usize(node, "groups", 1) as i64;
            if target == "conv2d" {
                slots.push(Slot::Owned(convolution::conv2d(
                    &input,
                    &weight,
                    bias.as_ref(),
                    node.kwargs.get("stride"),
                    node.kwargs.get("padding"),
                    node.kwargs.get("dilation"),
                    groups,
                )?));
            } else {
                slots.push(Slot::Owned(convolution::conv1d(
                    &input,
                    &weight,
                    bias.as_ref(),
                    node.kwargs.get("stride"),
                    node.kwargs.get("padding"),
                    node.kwargs.get("dilation"),
                    groups,
                )?));
            }
        }
        "conv_transpose1d" | "conv_transpose2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = if node.args.len() > 2 {
                Some(slot_view(slots, capsules, arg_index(node, 2)?)?)
            } else {
                None
            };
            let groups = kw_usize(node, "groups", 1) as i64;
            let (stride, padding, output_padding, dilation) = (
                node.kwargs.get("stride"),
                node.kwargs.get("padding"),
                node.kwargs.get("output_padding"),
                node.kwargs.get("dilation"),
            );
            slots.push(Slot::Owned(if target == "conv_transpose1d" {
                convolution::conv_transpose1d(
                    &input,
                    &weight,
                    bias.as_ref(),
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    groups,
                )?
            } else {
                convolution::conv_transpose2d(
                    &input,
                    &weight,
                    bias.as_ref(),
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    groups,
                )?
            }));
        }

        // Phase 3: pooling
        "max_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            let kernel = node
                .kwargs
                .get("kernel")
                .or_else(|| node.kwargs.get("kernel_size"));
            slots.push(Slot::Owned(pooling::max_pool2d(
                &input,
                kernel,
                node.kwargs.get("stride"),
                node.kwargs.get("padding"),
                node.kwargs.get("dilation"),
                ceil_mode,
            )?));
        }
        "avg_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            let count_include_pad = kw_bool(node, "count_include_pad", true);
            let kernel = node
                .kwargs
                .get("kernel")
                .or_else(|| node.kwargs.get("kernel_size"));
            slots.push(Slot::Owned(pooling::avg_pool2d(
                &input,
                kernel,
                node.kwargs.get("stride"),
                node.kwargs.get("padding"),
                ceil_mode,
                count_include_pad,
            )?));
        }
        "adaptive_avg_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(pooling::adaptive_avg_pool2d(
                &input,
                node.kwargs.get("output_size"),
            )?));
        }
        "adaptive_max_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(pooling::adaptive_max_pool2d(
                &input,
                node.kwargs.get("output_size"),
            )?));
        }
        "max_pool1d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            slots.push(Slot::Owned(pooling::max_pool1d(
                &input,
                node.kwargs.get("kernel"),
                node.kwargs.get("stride"),
                node.kwargs.get("padding"),
                ceil_mode,
            )?));
        }
        "avg_pool1d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            let count_include_pad = kw_bool(node, "count_include_pad", true);
            slots.push(Slot::Owned(pooling::avg_pool1d(
                &input,
                node.kwargs.get("kernel"),
                node.kwargs.get("stride"),
                node.kwargs.get("padding"),
                ceil_mode,
                count_include_pad,
            )?));
        }

        // Phase 3: upsampling
        "upsample_nearest2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(upsample::upsample_nearest2d(
                &input,
                node.kwargs.get("size"),
            )?));
        }
        "upsample_bilinear2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(upsample::upsample_bilinear2d(
                &input,
                node.kwargs.get("size"),
            )?));
        }
        "interpolate" => {
            // F.interpolate(x, size=..., mode=...) — route by mode. Only the
            // 4-D (N,C,H,W) case with an explicit size is supported here;
            // scale_factor and other modes fall back to eager (REQ-002).
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mode = node
                .kwargs
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("nearest");
            match mode {
                "nearest" | "nearest-exact" => slots.push(Slot::Owned(
                    upsample::upsample_nearest2d(&input, node.kwargs.get("size"))?,
                )),
                "bilinear" => {
                    // Only align_corners=false is implemented; reject the
                    // true variant rather than silently returning wrong values.
                    if node.kwargs.get("align_corners").and_then(|v| v.as_bool()) == Some(true) {
                        return Err(unsupported("interpolate: align_corners=true not supported"));
                    }
                    slots.push(Slot::Owned(upsample::upsample_bilinear2d(
                        &input,
                        node.kwargs.get("size"),
                    )?))
                }
                _ => {
                    return Err(unsupported(&format!(
                        "interpolate: unsupported mode '{mode}'"
                    )))
                }
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
