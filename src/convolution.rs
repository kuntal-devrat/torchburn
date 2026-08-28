//! Convolution operations: conv1d, conv2d, conv_transpose2d (im2col-style
//! direct convolution, Phase 3).
//!
//! Implements the direct (output-stationary) convolution algorithm with
//! strided, padded, dilated, grouped kernels and optional bias. The kernel
//! is the same shape as PyTorch's weight layout `(C_out, C_in/groups, K...)`,
//! so `torch.compile` graphs (aten.convolution.default / torch.conv2d) map
//! straight onto this module.
//!
//! Requirements: contiguous f32/f64 inputs. The interpreter guarantees
//! contiguity before dispatch; strided inputs raise `TB_UNSUPPORTED` and fall
//! back to eager PyTorch (REQ-002).

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, contiguous_strides, unsupported};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

fn require_contiguous(t: &BorrowedTensor, what: &str) -> PyResult<()> {
    if !t.is_contiguous() {
        return Err(unsupported(&format!("{what} must be contiguous for convolution")));
    }
    Ok(())
}

/// Output spatial size for a convolution dimension.
fn conv_out_size(input: i64, kernel: i64, stride: i64, padding: i64, dilation: i64) -> PyResult<i64> {
    let effective = dilation * (kernel - 1) + 1;
    let numerator = input + 2 * padding - effective;
    if numerator < 0 {
        return Err(unsupported(&format!(
            "convolution output size is negative (input={input}, kernel={kernel}, \
             stride={stride}, padding={padding}, dilation={dilation})"
        )));
    }
    Ok(numerator / stride + 1)
}

/// Validate scalar-or-pair params (int or [int, int]) coming from kwargs.
fn pair(v: Option<&serde_json::Value>, name: &str, default: i64) -> PyResult<(i64, i64)> {
    let Some(v) = v else { return Ok((default, default)) };
    if let Some(s) = v.as_i64() {
        return Ok((s, s));
    }
    if let Some(arr) = v.as_array() {
        let vals: Vec<i64> = arr.iter().filter_map(|x| x.as_i64()).collect();
        if vals.len() == 1 {
            return Ok((vals[0], vals[0]));
        }
        if vals.len() == 2 {
            return Ok((vals[0], vals[1]));
        }
    }
    Err(unsupported(&format!("{name} must be an int or 2-int list")))
}

// ---------------------------------------------------------------------------
// conv2d
// ---------------------------------------------------------------------------

fn conv2d_f32(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: (i64, i64),
    padding: (i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let cin = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let cout = weight.shape[0] as usize;
    let cin_g = weight.shape[1] as usize;
    let kh = weight.shape[2] as usize;
    let kw = weight.shape[3] as usize;

    if groups <= 0 || cin % groups as usize != 0 || cout % groups as usize != 0 {
        return Err(unsupported(&format!(
            "conv2d: groups={groups} must divide channels (C_in={cin}, C_out={cout})"
        )));
    }
    let g = groups as usize;
    let cout_g = cout / g;
    if cin_g * g != cin {
        return Err(unsupported("conv2d: weight channel dim does not match input/groups"));
    }
    if let Some(bias) = bias {
        if bias.shape != vec![cout as i64] {
            return Err(unsupported("conv2d: bias must have shape (C_out,)"));
        }
    }

    let out_h = conv_out_size(h as i64, kh as i64, stride.0, padding.0, dilation.0)? as usize;
    let out_w = conv_out_size(w as i64, kw as i64, stride.1, padding.1, dilation.1)? as usize;

    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, cout as i64, out_h as i64, out_w as i64]);
    let input_data = unsafe { typed_slice::<f32>(input) };
    let weight_data = unsafe { typed_slice::<f32>(weight) };
    let bias_data: Option<&[f32]> = bias.map(|x| unsafe { typed_slice::<f32>(x) });
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let (sh, sw) = (stride.0 as usize, stride.1 as usize);
    let (ph, pw) = (padding.0, padding.1);
    let (dh, dw) = (dilation.0 as usize, dilation.1 as usize);
    let channel_plane_elems = out_h * out_w;

    use rayon::prelude::*;
    // Parallelize over (batch, channel): each worker owns a disjoint
    // output slice of `channel_plane_elems` elements for one output channel.
    out_data.par_chunks_mut(channel_plane_elems).enumerate().for_each(|(plane_idx, out_channel)| {
        let bi = plane_idx / cout;
        let co = plane_idx % cout;
        let group = co / cout_g;
        let cin_start = group * cin_g;
        let bias_val = bias_data.map(|bd| bd[co]).unwrap_or(0.0);
        for oh in 0..out_h {
            let ih_start = oh as i64 * sh as i64 - ph;
            for ow in 0..out_w {
                let iw_start = ow as i64 * sw as i64 - pw;
                let mut acc = bias_val;
                for ci in 0..cin_g {
                    let in_plane = cin_start + ci;
                    let w_row = ((co * cin_g) + ci) * kh;
                    for khh in 0..kh {
                        let ih = ih_start + (khh as i64 * dh as i64);
                        if ih < 0 || ih >= h as i64 {
                            continue;
                        }
                        for kww in 0..kw {
                            let iw = iw_start + (kww as i64 * dw as i64);
                            if iw < 0 || iw >= w as i64 {
                                continue;
                            }
                            let in_idx = ((bi * cin + in_plane) * h + ih as usize) * w + iw as usize;
                            let w_idx = (w_row + khh) * kw + kww;
                            acc += input_data[in_idx] * weight_data[w_idx];
                        }
                    }
                }
                out_channel[oh * out_w + ow] = acc;
            }
        }
    });
    Ok(out)
}

fn conv2d_f64(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: (i64, i64),
    padding: (i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let cin = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let cout = weight.shape[0] as usize;
    let cin_g = weight.shape[1] as usize;
    let kh = weight.shape[2] as usize;
    let kw = weight.shape[3] as usize;

    if groups <= 0 || cin % groups as usize != 0 || cout % groups as usize != 0 {
        return Err(unsupported(&format!(
            "conv2d: groups={groups} must divide channels (C_in={cin}, C_out={cout})"
        )));
    }
    let g = groups as usize;
    let cout_g = cout / g;
    if cin_g * g != cin {
        return Err(unsupported("conv2d: weight channel dim does not match input/groups"));
    }
    if let Some(bias) = bias {
        if bias.shape != vec![cout as i64] {
            return Err(unsupported("conv2d: bias must have shape (C_out,)"));
        }
    }

    let out_h = conv_out_size(h as i64, kh as i64, stride.0, padding.0, dilation.0)? as usize;
    let out_w = conv_out_size(w as i64, kw as i64, stride.1, padding.1, dilation.1)? as usize;

    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, cout as i64, out_h as i64, out_w as i64]);
    let input_data = unsafe { typed_slice::<f64>(input) };
    let weight_data = unsafe { typed_slice::<f64>(weight) };
    let bias_data: Option<&[f64]> = bias.map(|x| unsafe { typed_slice::<f64>(x) });
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let (sh, sw) = (stride.0 as usize, stride.1 as usize);
    let (ph, pw) = (padding.0, padding.1);
    let (dh, dw) = (dilation.0 as usize, dilation.1 as usize);
    let channel_plane_elems = out_h * out_w;

    use rayon::prelude::*;
    out_data.par_chunks_mut(channel_plane_elems).enumerate().for_each(|(plane_idx, out_channel)| {
        let bi = plane_idx / cout;
        let co = plane_idx % cout;
        let group = co / cout_g;
        let cin_start = group * cin_g;
        let bias_val = bias_data.map(|bd| bd[co]).unwrap_or(0.0);
        for oh in 0..out_h {
            let ih_start = oh as i64 * sh as i64 - ph;
            for ow in 0..out_w {
                let iw_start = ow as i64 * sw as i64 - pw;
                let mut acc = bias_val;
                for ci in 0..cin_g {
                    let in_plane = cin_start + ci;
                    let w_row = ((co * cin_g) + ci) * kh;
                    for khh in 0..kh {
                        let ih = ih_start + (khh as i64 * dh as i64);
                        if ih < 0 || ih >= h as i64 {
                            continue;
                        }
                        for kww in 0..kw {
                            let iw = iw_start + (kww as i64 * dw as i64);
                            if iw < 0 || iw >= w as i64 {
                                continue;
                            }
                            let in_idx = ((bi * cin + in_plane) * h + ih as usize) * w + iw as usize;
                            let w_idx = (w_row + khh) * kw + kww;
                            acc += input_data[in_idx] * weight_data[w_idx];
                        }
                    }
                }
                out_channel[oh * out_w + ow] = acc;
            }
        }
    });
    Ok(out)
}

/// 2D convolution. `weight` is (C_out, C_in/groups, KH, KW).
pub fn conv2d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    dilation: Option<&serde_json::Value>,
    groups: i64,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "conv2d input")?;
    require_contiguous(weight, "conv2d weight")?;
    if input.dtype != weight.dtype {
        return Err(unsupported("conv2d: dtype mismatch between input and weight"));
    }
    if input.shape.len() != 4 || weight.shape.len() != 4 {
        return Err(unsupported("conv2d: input must be 4-D (B,C,H,W) and weight 4-D"));
    }
    if let Some(bias) = bias {
        if bias.dtype != input.dtype {
            return Err(unsupported("conv2d: dtype mismatch with bias"));
        }
        require_contiguous(bias, "conv2d bias")?;
    }
    let stride = pair(stride, "stride", 1)?;
    let padding = pair(padding, "padding", 0)?;
    let dilation = pair(dilation, "dilation", 1)?;
    if groups <= 0 {
        return Err(unsupported("conv2d: groups must be positive"));
    }
    match input.dtype {
        DType::F32 => conv2d_f32(input, weight, bias, stride, padding, dilation, groups),
        DType::F64 => conv2d_f64(input, weight, bias, stride, padding, dilation, groups),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

// ---------------------------------------------------------------------------
// conv1d
// ---------------------------------------------------------------------------

fn conv1d_f32(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let cin = input.shape[1] as usize;
    let l = input.shape[2] as usize;
    let cout = weight.shape[0] as usize;
    let cin_g = weight.shape[1] as usize;
    let k = weight.shape[2] as usize;

    if groups <= 0 || cin % groups as usize != 0 || cout % groups as usize != 0 {
        return Err(unsupported(&format!(
            "conv1d: groups={groups} must divide channels (C_in={cin}, C_out={cout})"
        )));
    }
    let g = groups as usize;
    let cout_g = cout / g;
    if cin_g * g != cin {
        return Err(unsupported("conv1d: weight channel dim does not match input/groups"));
    }
    if let Some(bias) = bias {
        if bias.shape != vec![cout as i64] {
            return Err(unsupported("conv1d: bias must have shape (C_out,)"));
        }
    }

    let out_l = conv_out_size(l as i64, k as i64, stride, padding, dilation)? as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, cout as i64, out_l as i64]);
    let input_data = unsafe { typed_slice::<f32>(input) };
    let weight_data = unsafe { typed_slice::<f32>(weight) };
    let bias_data: Option<&[f32]> = bias.map(|x| unsafe { typed_slice::<f32>(x) });
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let (s, p, d) = (stride as usize, padding, dilation as usize);
    let plane_elems = cout * out_l;
    use rayon::prelude::*;
    out_data.par_chunks_mut(plane_elems).enumerate().for_each(|(bi, out_plane)| {
        for co in 0..cout {
            let group = co / cout_g;
            let cin_start = group * cin_g;
            let base_out = co * out_l;
            let bias_val = bias_data.map(|bd| bd[co]).unwrap_or(0.0);
            for ol in 0..out_l {
                let il_start = ol as i64 * s as i64 - p;
                let mut acc = bias_val;
                for ci in 0..cin_g {
                    let in_plane = cin_start + ci;
                    let w_row = (co * cin_g + ci) * k;
                    for kk in 0..k {
                        let il = il_start + (kk as i64 * d as i64);
                        if il < 0 || il >= l as i64 {
                            continue;
                        }
                        acc += input_data[(bi * cin + in_plane) * l + il as usize]
                            * weight_data[w_row + kk];
                    }
                }
                out_plane[base_out + ol] = acc;
            }
        }
    });
    Ok(out)
}

fn conv1d_f64(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let cin = input.shape[1] as usize;
    let l = input.shape[2] as usize;
    let cout = weight.shape[0] as usize;
    let cin_g = weight.shape[1] as usize;
    let k = weight.shape[2] as usize;

    if groups <= 0 || cin % groups as usize != 0 || cout % groups as usize != 0 {
        return Err(unsupported(&format!(
            "conv1d: groups={groups} must divide channels (C_in={cin}, C_out={cout})"
        )));
    }
    let g = groups as usize;
    let cout_g = cout / g;
    if cin_g * g != cin {
        return Err(unsupported("conv1d: weight channel dim does not match input/groups"));
    }
    if let Some(bias) = bias {
        if bias.shape != vec![cout as i64] {
            return Err(unsupported("conv1d: bias must have shape (C_out,)"));
        }
    }

    let out_l = conv_out_size(l as i64, k as i64, stride, padding, dilation)? as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, cout as i64, out_l as i64]);
    let input_data = unsafe { typed_slice::<f64>(input) };
    let weight_data = unsafe { typed_slice::<f64>(weight) };
    let bias_data: Option<&[f64]> = bias.map(|x| unsafe { typed_slice::<f64>(x) });
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let (s, p, d) = (stride as usize, padding, dilation as usize);
    let plane_elems = cout * out_l;
    use rayon::prelude::*;
    out_data.par_chunks_mut(plane_elems).enumerate().for_each(|(bi, out_plane)| {
        for co in 0..cout {
            let group = co / cout_g;
            let cin_start = group * cin_g;
            let base_out = co * out_l;
            let bias_val = bias_data.map(|bd| bd[co]).unwrap_or(0.0);
            for ol in 0..out_l {
                let il_start = ol as i64 * s as i64 - p;
                let mut acc = bias_val;
                for ci in 0..cin_g {
                    let in_plane = cin_start + ci;
                    let w_row = (co * cin_g + ci) * k;
                    for kk in 0..k {
                        let il = il_start + (kk as i64 * d as i64);
                        if il < 0 || il >= l as i64 {
                            continue;
                        }
                        acc += input_data[(bi * cin + in_plane) * l + il as usize]
                            * weight_data[w_row + kk];
                    }
                }
                out_plane[base_out + ol] = acc;
            }
        }
    });
    Ok(out)
}

/// 1D convolution. `weight` is (C_out, C_in/groups, K).
pub fn conv1d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    dilation: Option<&serde_json::Value>,
    groups: i64,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "conv1d input")?;
    require_contiguous(weight, "conv1d weight")?;
    if input.dtype != weight.dtype {
        return Err(unsupported("conv1d: dtype mismatch between input and weight"));
    }
    if input.shape.len() != 3 || weight.shape.len() != 3 {
        return Err(unsupported("conv1d: input must be 3-D (B,C,L) and weight 3-D"));
    }
    if let Some(bias) = bias {
        if bias.dtype != input.dtype {
            return Err(unsupported("conv1d: dtype mismatch with bias"));
        }
        require_contiguous(bias, "conv1d bias")?;
    }
    let stride = pair(stride, "stride", 1)?.0;
    let padding = pair(padding, "padding", 0)?.0;
    let dilation = pair(dilation, "dilation", 1)?.0;
    if groups <= 0 {
        return Err(unsupported("conv1d: groups must be positive"));
    }
    match input.dtype {
        DType::F32 => conv1d_f32(input, weight, bias, stride, padding, dilation, groups),
        DType::F64 => conv1d_f64(input, weight, bias, stride, padding, dilation, groups),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}

// ---------------------------------------------------------------------------
// conv_transpose1d (maps to the 2-D kernel with a unit H dimension)
// ---------------------------------------------------------------------------

/// Insert a unit H dimension into a contiguous tensor view without copying.
/// The 2-D kernels index flat contiguous buffers, so a (B,C,1,L) view over
/// (B,C,L) data is layout-identical.
fn view_with_unit_h(t: &BorrowedTensor) -> BorrowedTensor {
    let mut shape = t.shape.clone();
    shape.insert(2, 1); // (B, C, 1, L)
    BorrowedTensor {
        data: t.data,
        strides: contiguous_strides(&shape),
        shape,
        dtype: t.dtype,
    }
}

pub fn conv_transpose1d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    output_padding: Option<&serde_json::Value>,
    dilation: Option<&serde_json::Value>,
    groups: i64,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "conv_transpose1d input")?;
    require_contiguous(weight, "conv_transpose1d weight")?;
    if input.dtype != weight.dtype {
        return Err(unsupported("conv_transpose1d: dtype mismatch between input and weight"));
    }
    if input.shape.len() != 3 || weight.shape.len() != 3 {
        return Err(unsupported("conv_transpose1d: input must be 3-D (B,C,L) and weight 3-D"));
    }
    if let Some(bias) = bias {
        if bias.dtype != input.dtype {
            return Err(unsupported("conv_transpose1d: dtype mismatch with bias"));
        }
        require_contiguous(bias, "conv_transpose1d bias")?;
    }
    // Promote to 4-D with H=1: (B,C,1,L) x (C_in,C_out/g,1,K)
    let input2 = view_with_unit_h(input);
    let weight2 = view_with_unit_h(weight);
    let s = pair(stride, "stride", 1)?;
    let p = pair(padding, "padding", 0)?;
    let op = pair(output_padding, "output_padding", 0)?;
    let d = pair(dilation, "dilation", 1)?;
    let mut out = conv_transpose2d(
        &input2, &weight2, bias,
        Some(&serde_json::json!([1, s.0])),
        Some(&serde_json::json!([0, p.0])),
        Some(&serde_json::json!([0, op.0])),
        Some(&serde_json::json!([1, d.0])),
        groups,
    )?;
    // Strip the unit H dim: (B, C_out, 1, L') -> (B, C_out, L')
    out.shape.remove(2);
    Ok(out)
}

// ---------------------------------------------------------------------------
// conv_transpose2d
// ---------------------------------------------------------------------------

/// Output size for a transposed convolution dimension.
fn conv_transpose_out_size(
    input: i64,
    kernel: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
    output_padding: i64,
) -> PyResult<i64> {
    let effective = dilation * (kernel - 1) + 1;
    let out = (input - 1) * stride - 2 * padding + effective + output_padding;
    if out < 1 {
        return Err(unsupported("conv_transpose2d: output size must be positive"));
    }
    Ok(out)
}

fn conv_transpose2d_f32(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: (i64, i64),
    padding: (i64, i64),
    output_padding: (i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let cin = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    // Weight layout for transposed conv: (C_in, C_out/groups, KH, KW)
    let cin_w = weight.shape[0] as usize;
    let cout_g = weight.shape[1] as usize;
    let kh = weight.shape[2] as usize;
    let kw = weight.shape[3] as usize;

    if groups <= 0 || cin % groups as usize != 0 {
        return Err(unsupported("conv_transpose2d: groups must divide C_in"));
    }
    let g = groups as usize;
    let cout = cout_g * g;
    if cin_w != cin {
        return Err(unsupported("conv_transpose2d: weight C_in does not match input"));
    }
    if let Some(bias) = bias {
        if bias.shape != vec![cout as i64] {
            return Err(unsupported("conv_transpose2d: bias must have shape (C_out,)"));
        }
    }

    let out_h = conv_transpose_out_size(h as i64, kh as i64, stride.0, padding.0, dilation.0, output_padding.0)? as usize;
    let out_w = conv_transpose_out_size(w as i64, kw as i64, stride.1, padding.1, dilation.1, output_padding.1)? as usize;

    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, cout as i64, out_h as i64, out_w as i64]);
    let input_data = unsafe { typed_slice::<f32>(input) };
    let weight_data = unsafe { typed_slice::<f32>(weight) };
    let bias_data: Option<&[f32]> = bias.map(|x| unsafe { typed_slice::<f32>(x) });
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let (sh, sw) = (stride.0 as usize, stride.1 as usize);
    let (ph, pw) = (padding.0, padding.1);
    let (dh, dw) = (dilation.0 as usize, dilation.1 as usize);
    let plane_elems = cout * out_h * out_w;

    use rayon::prelude::*;
    // Scatter approach: for each input position and kernel tap, add the
    // contribution to the corresponding output position.
    out_data.par_chunks_mut(plane_elems).enumerate().for_each(|(bi, out_plane)| {
        // initialize output with bias
        for co in 0..cout {
            let bias_val = bias_data.map(|bd| bd[co]).unwrap_or(0.0);
            let base = co * out_h * out_w;
            for i in 0..out_h * out_w {
                out_plane[base + i] = bias_val;
            }
        }
        for ci in 0..cin {
            let group = ci / (cin / g);
            let co_start = group * cout_g;
            // weight layout is (C_in, C_out/g, KH, KW)
            let w_ci_base = ci * cout_g * kh * kw;
            for ih in 0..h {
                for iw in 0..w {
                    let in_val = input_data[(bi * cin + ci) * h * w + ih * w + iw];
                    for khh in 0..kh {
                        for kww in 0..kw {
                            let oh = ih as i64 * sh as i64 - ph as i64 + (khh as i64 * dh as i64);
                            let ow = iw as i64 * sw as i64 - pw as i64 + (kww as i64 * dw as i64);
                            if oh < 0 || oh >= out_h as i64 || ow < 0 || ow >= out_w as i64 {
                                continue;
                            }
                            for cog in 0..cout_g {
                                let co = co_start + cog;
                                let w_idx = w_ci_base + cog * kh * kw + khh * kw + kww;
                                out_plane[(co * out_h + oh as usize) * out_w + ow as usize]
                                    += in_val * weight_data[w_idx];
                            }
                        }
                    }
                }
            }
        }
    });
    Ok(out)
}

fn conv_transpose2d_f64(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: (i64, i64),
    padding: (i64, i64),
    output_padding: (i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let cin = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let cin_w = weight.shape[0] as usize;
    let cout_g = weight.shape[1] as usize;
    let kh = weight.shape[2] as usize;
    let kw = weight.shape[3] as usize;

    if groups <= 0 || cin % groups as usize != 0 {
        return Err(unsupported("conv_transpose2d: groups must divide C_in"));
    }
    let g = groups as usize;
    let cout = cout_g * g;
    if cin_w != cin {
        return Err(unsupported("conv_transpose2d: weight C_in does not match input"));
    }
    if let Some(bias) = bias {
        if bias.shape != vec![cout as i64] {
            return Err(unsupported("conv_transpose2d: bias must have shape (C_out,)"));
        }
    }

    let out_h = conv_transpose_out_size(h as i64, kh as i64, stride.0, padding.0, dilation.0, output_padding.0)? as usize;
    let out_w = conv_transpose_out_size(w as i64, kw as i64, stride.1, padding.1, dilation.1, output_padding.1)? as usize;

    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, cout as i64, out_h as i64, out_w as i64]);
    let input_data = unsafe { typed_slice::<f64>(input) };
    let weight_data = unsafe { typed_slice::<f64>(weight) };
    let bias_data: Option<&[f64]> = bias.map(|x| unsafe { typed_slice::<f64>(x) });
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let (sh, sw) = (stride.0 as usize, stride.1 as usize);
    let (ph, pw) = (padding.0, padding.1);
    let (dh, dw) = (dilation.0 as usize, dilation.1 as usize);
    let plane_elems = cout * out_h * out_w;

    use rayon::prelude::*;
    out_data.par_chunks_mut(plane_elems).enumerate().for_each(|(bi, out_plane)| {
        for co in 0..cout {
            let bias_val = bias_data.map(|bd| bd[co]).unwrap_or(0.0);
            let base = co * out_h * out_w;
            for i in 0..out_h * out_w {
                out_plane[base + i] = bias_val;
            }
        }
        for ci in 0..cin {
            let group = ci / (cin / g);
            let co_start = group * cout_g;
            // weight layout is (C_in, C_out/g, KH, KW)
            let w_ci_base = ci * cout_g * kh * kw;
            for ih in 0..h {
                for iw in 0..w {
                    let in_val = input_data[(bi * cin + ci) * h * w + ih * w + iw];
                    for khh in 0..kh {
                        for kww in 0..kw {
                            let oh = ih as i64 * sh as i64 - ph as i64 + (khh as i64 * dh as i64);
                            let ow = iw as i64 * sw as i64 - pw as i64 + (kww as i64 * dw as i64);
                            if oh < 0 || oh >= out_h as i64 || ow < 0 || ow >= out_w as i64 {
                                continue;
                            }
                            for cog in 0..cout_g {
                                let co = co_start + cog;
                                let w_idx = w_ci_base + cog * kh * kw + khh * kw + kww;
                                out_plane[(co * out_h + oh as usize) * out_w + ow as usize]
                                    += in_val * weight_data[w_idx];
                            }
                        }
                    }
                }
            }
        }
    });
    Ok(out)
}

/// 2D transposed convolution. `weight` is (C_in, C_out/groups, KH, KW).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    output_padding: Option<&serde_json::Value>,
    dilation: Option<&serde_json::Value>,
    groups: i64,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "conv_transpose2d input")?;
    require_contiguous(weight, "conv_transpose2d weight")?;
    if input.dtype != weight.dtype {
        return Err(unsupported("conv_transpose2d: dtype mismatch"));
    }
    if input.shape.len() != 4 || weight.shape.len() != 4 {
        return Err(unsupported("conv_transpose2d: input and weight must be 4-D"));
    }
    if let Some(bias) = bias {
        if bias.dtype != input.dtype {
            return Err(unsupported("conv_transpose2d: dtype mismatch with bias"));
        }
        require_contiguous(bias, "conv_transpose2d bias")?;
    }
    let stride = pair(stride, "stride", 1)?;
    let padding = pair(padding, "padding", 0)?;
    let output_padding = pair(output_padding, "output_padding", 0)?;
    let dilation = pair(dilation, "dilation", 1)?;
    if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
        return Err(unsupported("conv_transpose2d: output_padding must be < stride"));
    }
    if groups <= 0 {
        return Err(unsupported("conv_transpose2d: groups must be positive"));
    }
    match input.dtype {
        DType::F32 => conv_transpose2d_f32(input, weight, bias, stride, padding, output_padding, dilation, groups),
        DType::F64 => conv_transpose2d_f64(input, weight, bias, stride, padding, output_padding, dilation, groups),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }

    }
}
