//! Pooling operations (Phase 3): max_pool2d, avg_pool2d, adaptive_avg_pool2d,
//! adaptive_max_pool2d, plus 1-D variants. All kernels support f32/f64 and
//! require contiguous inputs (the interpreter guarantees this before dispatch;
//! strided inputs raise `TB_UNSUPPORTED` and fall back to eager).

use crate::dlpack::{unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    if t.buffer_len() == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
    }
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    if t.elem_count() == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
    }
}

fn require_contiguous(t: &BorrowedTensor, what: &str) -> PyResult<()> {
    if !t.is_contiguous() {
        return Err(unsupported(&format!(
            "{what} must be contiguous for pooling"
        )));
    }
    Ok(())
}

/// Normalize a scalar-or-pair param from kwargs into `(h, w)`.
fn pair(v: Option<&serde_json::Value>, name: &str, default: i64) -> PyResult<(i64, i64)> {
    let Some(v) = v else {
        return Ok((default, default));
    };
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

fn scalar(v: Option<&serde_json::Value>, name: &str, default: i64) -> PyResult<i64> {
    let Some(v) = v else { return Ok(default) };
    if let Some(s) = v.as_i64() {
        return Ok(s);
    }
    if let Some(arr) = v.as_array() {
        if let Some(s) = arr.first().and_then(|x| x.as_i64()) {
            return Ok(s);
        }
    }
    Err(unsupported(&format!("{name} must be an int")))
}

/// Floor-based pooling output size (torch's default, ceil_mode=False).
/// Uses floor division so a kernel larger than the (padded) input yields 0,
/// matching torch, which then rejects the configuration.
fn pool_out_size(input: i64, kernel: i64, stride: i64, padding: i64, dilation: i64) -> i64 {
    let effective = dilation * (kernel - 1) + 1;
    let numer = input + 2 * padding - effective;
    if numer >= 0 {
        numer / stride + 1
    } else {
        // floor division (Rust's `/` truncates toward zero)
        -((-numer + stride - 1) / stride) + 1
    }
}

// ---------------------------------------------------------------------------
// max_pool2d
// ---------------------------------------------------------------------------

fn max_pool2d_f32(
    input: &BorrowedTensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (i64, i64),
    dilation: (usize, usize),
    ceil_mode: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = (padding.0, padding.1);
    let (dh, dw) = (dilation.0, dilation.1);

    let out_h = if ceil_mode {
        (((h as i64 + 2 * ph - (dh as i64) * (kh as i64 - 1) - 1) as f64 / sh as f64).ceil() as i64
            + 1)
        .max(1)
    } else {
        pool_out_size(h as i64, kh as i64, sh as i64, ph, dh as i64)
    };
    let out_w = if ceil_mode {
        (((w as i64 + 2 * pw - (dw as i64) * (kw as i64 - 1) - 1) as f64 / sw as f64).ceil() as i64
            + 1)
        .max(1)
    } else {
        pool_out_size(w as i64, kw as i64, sw as i64, pw, dw as i64)
    };

    if out_h < 1 || out_w < 1 {
        return Err(unsupported("max_pool2d: output size must be positive"));
    }
    let (out_h, out_w) = (out_h as usize, out_w as usize);

    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f32>(input) };
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let plane_elems = c * out_h * out_w;
    if plane_elems == 0 || out_data.is_empty() {
        return Ok(out);
    }
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let h_start = oh as i64 * sh as i64 - ph;
                    for ow in 0..out_w {
                        let w_start = ow as i64 * sw as i64 - pw;
                        let mut best = f32::NEG_INFINITY;
                        for khh in 0..kh {
                            let ih = h_start + (khh as i64 * dh as i64);
                            if ih < 0 || ih >= h as i64 {
                                continue;
                            }
                            for kww in 0..kw {
                                let iw = w_start + (kww as i64 * dw as i64);
                                if iw < 0 || iw >= w as i64 {
                                    continue;
                                }
                                let v = input_data[plane + ih as usize * w + iw as usize];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        out_plane[oh * out_w + ow] = best;
                    }
                }
            }
        });
    Ok(out)
}

fn max_pool2d_f64(
    input: &BorrowedTensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (i64, i64),
    dilation: (usize, usize),
    ceil_mode: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = (padding.0, padding.1);
    let (dh, dw) = (dilation.0, dilation.1);

    let out_h = if ceil_mode {
        (((h as i64 + 2 * ph - (dh as i64) * (kh as i64 - 1) - 1) as f64 / sh as f64).ceil() as i64
            + 1)
        .max(1)
    } else {
        pool_out_size(h as i64, kh as i64, sh as i64, ph, dh as i64)
    };
    let out_w = if ceil_mode {
        (((w as i64 + 2 * pw - (dw as i64) * (kw as i64 - 1) - 1) as f64 / sw as f64).ceil() as i64
            + 1)
        .max(1)
    } else {
        pool_out_size(w as i64, kw as i64, sw as i64, pw, dw as i64)
    };

    if out_h < 1 || out_w < 1 {
        return Err(unsupported("max_pool2d: output size must be positive"));
    }
    let (out_h, out_w) = (out_h as usize, out_w as usize);

    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f64>(input) };
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let plane_elems = c * out_h * out_w;
    if plane_elems == 0 || out_data.is_empty() {
        return Ok(out);
    }
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let h_start = oh as i64 * sh as i64 - ph;
                    for ow in 0..out_w {
                        let w_start = ow as i64 * sw as i64 - pw;
                        let mut best = f64::NEG_INFINITY;
                        for khh in 0..kh {
                            let ih = h_start + (khh as i64 * dh as i64);
                            if ih < 0 || ih >= h as i64 {
                                continue;
                            }
                            for kww in 0..kw {
                                let iw = w_start + (kww as i64 * dw as i64);
                                if iw < 0 || iw >= w as i64 {
                                    continue;
                                }
                                let v = input_data[plane + ih as usize * w + iw as usize];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        out_plane[oh * out_w + ow] = best;
                    }
                }
            }
        });
    Ok(out)
}

/// 2-D max pooling. Kernel/stride/padding/dilation are scalar-or-pair.
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d(
    input: &BorrowedTensor,
    kernel: Option<&serde_json::Value>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    dilation: Option<&serde_json::Value>,
    ceil_mode: bool,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "max_pool2d input")?;
    if input.shape.len() != 4 {
        return Err(unsupported("max_pool2d: input must be 4-D (B,C,H,W)"));
    }
    let kernel = pair(kernel, "kernel", 0)?;
    if kernel.0 <= 0 || kernel.1 <= 0 {
        return Err(unsupported("max_pool2d: kernel must be positive"));
    }
    let stride = pair(stride, "stride", 0)?;
    // torch: stride defaults to kernel
    let stride = if stride.0 == 0 && stride.1 == 0 {
        kernel
    } else {
        stride
    };
    let padding = pair(padding, "padding", 0)?;
    let dilation = pair(dilation, "dilation", 1)?;
    let (kh, kw) = (kernel.0 as usize, kernel.1 as usize);
    let (sh, sw) = (stride.0 as usize, stride.1 as usize);
    let (dh, dw) = (dilation.0 as usize, dilation.1 as usize);
    match input.dtype {
        DType::F32 => max_pool2d_f32(input, (kh, kw), (sh, sw), padding, (dh, dw), ceil_mode),
        DType::F64 => max_pool2d_f64(input, (kh, kw), (sh, sw), padding, (dh, dw), ceil_mode),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

// ---------------------------------------------------------------------------
// avg_pool2d
// ---------------------------------------------------------------------------

fn avg_pool2d_f32(
    input: &BorrowedTensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (i64, i64),
    ceil_mode: bool,
    count_include_pad: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = (padding.0, padding.1);

    let out_h = if ceil_mode {
        (((h as i64 + 2 * ph - kh as i64 - 1) as f64 / sh as f64).ceil() as i64 + 1).max(1)
    } else {
        pool_out_size(h as i64, kh as i64, sh as i64, ph, 1)
    };
    let out_w = if ceil_mode {
        (((w as i64 + 2 * pw - kw as i64 - 1) as f64 / sw as f64).ceil() as i64 + 1).max(1)
    } else {
        pool_out_size(w as i64, kw as i64, sw as i64, pw, 1)
    };

    if out_h < 1 || out_w < 1 {
        return Err(unsupported("avg_pool2d: output size must be positive"));
    }
    let (out_h, out_w) = (out_h as usize, out_w as usize);
    let window = kh * kw;

    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f32>(input) };
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let plane_elems = c * out_h * out_w;
    if plane_elems == 0 || out_data.is_empty() {
        return Ok(out);
    }
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let h_start = oh as i64 * sh as i64 - ph;
                    for ow in 0..out_w {
                        let w_start = ow as i64 * sw as i64 - pw;
                        let mut sum = 0.0f32;
                        let mut count = 0usize;
                        for khh in 0..kh {
                            let ih = h_start + khh as i64;
                            if ih < 0 || ih >= h as i64 {
                                continue;
                            }
                            for kww in 0..kw {
                                let iw = w_start + kww as i64;
                                if iw < 0 || iw >= w as i64 {
                                    continue;
                                }
                                sum += input_data[plane + ih as usize * w + iw as usize];
                                count += 1;
                            }
                        }
                        let divisor = if count_include_pad {
                            window
                        } else {
                            count.max(1)
                        };
                        out_plane[oh * out_w + ow] = sum / divisor as f32;
                    }
                }
            }
        });
    Ok(out)
}

fn avg_pool2d_f64(
    input: &BorrowedTensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (i64, i64),
    ceil_mode: bool,
    count_include_pad: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = (padding.0, padding.1);

    let out_h = if ceil_mode {
        (((h as i64 + 2 * ph - kh as i64 - 1) as f64 / sh as f64).ceil() as i64 + 1).max(1)
    } else {
        pool_out_size(h as i64, kh as i64, sh as i64, ph, 1)
    };
    let out_w = if ceil_mode {
        (((w as i64 + 2 * pw - kw as i64 - 1) as f64 / sw as f64).ceil() as i64 + 1).max(1)
    } else {
        pool_out_size(w as i64, kw as i64, sw as i64, pw, 1)
    };

    if out_h < 1 || out_w < 1 {
        return Err(unsupported("avg_pool2d: output size must be positive"));
    }
    let (out_h, out_w) = (out_h as usize, out_w as usize);
    let window = kh * kw;

    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f64>(input) };
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let plane_elems = c * out_h * out_w;
    if plane_elems == 0 || out_data.is_empty() {
        return Ok(out);
    }
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let h_start = oh as i64 * sh as i64 - ph;
                    for ow in 0..out_w {
                        let w_start = ow as i64 * sw as i64 - pw;
                        let mut sum = 0.0f64;
                        let mut count = 0usize;
                        for khh in 0..kh {
                            let ih = h_start + khh as i64;
                            if ih < 0 || ih >= h as i64 {
                                continue;
                            }
                            for kww in 0..kw {
                                let iw = w_start + kww as i64;
                                if iw < 0 || iw >= w as i64 {
                                    continue;
                                }
                                sum += input_data[plane + ih as usize * w + iw as usize];
                                count += 1;
                            }
                        }
                        let divisor = if count_include_pad {
                            window
                        } else {
                            count.max(1)
                        };
                        out_plane[oh * out_w + ow] = sum / divisor as f64;
                    }
                }
            }
        });
    Ok(out)
}

/// 2-D average pooling.
#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d(
    input: &BorrowedTensor,
    kernel: Option<&serde_json::Value>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    ceil_mode: bool,
    count_include_pad: bool,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "avg_pool2d input")?;
    if input.shape.len() != 4 {
        return Err(unsupported("avg_pool2d: input must be 4-D (B,C,H,W)"));
    }
    let kernel = pair(kernel, "kernel", 0)?;
    if kernel.0 <= 0 || kernel.1 <= 0 {
        return Err(unsupported("avg_pool2d: kernel must be positive"));
    }
    let stride = pair(stride, "stride", 0)?;
    let stride = if stride.0 == 0 && stride.1 == 0 {
        kernel
    } else {
        stride
    };
    let padding = pair(padding, "padding", 0)?;
    let (kh, kw) = (kernel.0 as usize, kernel.1 as usize);
    let (sh, sw) = (stride.0 as usize, stride.1 as usize);
    match input.dtype {
        DType::F32 => avg_pool2d_f32(
            input,
            (kh, kw),
            (sh, sw),
            padding,
            ceil_mode,
            count_include_pad,
        ),
        DType::F64 => avg_pool2d_f64(
            input,
            (kh, kw),
            (sh, sw),
            padding,
            ceil_mode,
            count_include_pad,
        ),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d — output is (oh, ow); windows split the input evenly
// ---------------------------------------------------------------------------

fn adaptive_start(idx: i64, out: i64, input: i64) -> i64 {
    // floor(idx * input / out)
    (idx * input) / out
}

fn adaptive_end(idx: i64, out: i64, input: i64) -> i64 {
    // ceil((idx+1) * input / out)
    ((idx + 1) * input + out - 1) / out
}

fn adaptive_avg_pool2d_f32(
    input: &BorrowedTensor,
    out_h: usize,
    out_w: usize,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f32>(input) };
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let plane_elems = c * out_h * out_w;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let hs = adaptive_start(oh as i64, out_h as i64, h as i64) as usize;
                    let he = adaptive_end(oh as i64, out_h as i64, h as i64) as usize;
                    for ow in 0..out_w {
                        let ws = adaptive_start(ow as i64, out_w as i64, w as i64) as usize;
                        let we = adaptive_end(ow as i64, out_w as i64, w as i64) as usize;
                        let mut sum = 0.0f32;
                        for ih in hs..he {
                            for iw in ws..we {
                                sum += input_data[plane + ih * w + iw];
                            }
                        }
                        let count = ((he - hs) * (we - ws)).max(1);
                        out_plane[oh * out_w + ow] = sum / count as f32;
                    }
                }
            }
        });
    Ok(out)
}

fn adaptive_avg_pool2d_f64(
    input: &BorrowedTensor,
    out_h: usize,
    out_w: usize,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f64>(input) };
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let plane_elems = c * out_h * out_w;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let hs = adaptive_start(oh as i64, out_h as i64, h as i64) as usize;
                    let he = adaptive_end(oh as i64, out_h as i64, h as i64) as usize;
                    for ow in 0..out_w {
                        let ws = adaptive_start(ow as i64, out_w as i64, w as i64) as usize;
                        let we = adaptive_end(ow as i64, out_w as i64, w as i64) as usize;
                        let mut sum = 0.0f64;
                        for ih in hs..he {
                            for iw in ws..we {
                                sum += input_data[plane + ih * w + iw];
                            }
                        }
                        let count = ((he - hs) * (we - ws)).max(1);
                        out_plane[oh * out_w + ow] = sum / count as f64;
                    }
                }
            }
        });
    Ok(out)
}

/// Adaptive average pooling to a fixed (oh, ow) output (global pooling: (1,1)).
pub fn adaptive_avg_pool2d(
    input: &BorrowedTensor,
    output_size: Option<&serde_json::Value>,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "adaptive_avg_pool2d input")?;
    if input.shape.len() != 4 {
        return Err(unsupported(
            "adaptive_avg_pool2d: input must be 4-D (B,C,H,W)",
        ));
    }
    // output_size may be an int, a 2-list, or a single 1-item list.
    let (out_h, out_w) = match output_size {
        None => (1, 1),
        Some(v) => {
            if let Some(s) = v.as_i64() {
                (s, s)
            } else if let Some(arr) = v.as_array() {
                let vals: Vec<i64> = arr.iter().filter_map(|x| x.as_i64()).collect();
                match vals.len() {
                    1 => (vals[0], vals[0]),
                    2 => (vals[0], vals[1]),
                    _ => {
                        return Err(unsupported(
                            "adaptive_avg_pool2d: output_size must be an int or pair",
                        ))
                    }
                }
            } else {
                return Err(unsupported(
                    "adaptive_avg_pool2d: output_size must be an int or pair",
                ));
            }
        }
    };
    if out_h <= 0 || out_w <= 0 {
        return Err(unsupported(
            "adaptive_avg_pool2d: output_size must be positive",
        ));
    }
    match input.dtype {
        DType::F32 => adaptive_avg_pool2d_f32(input, out_h as usize, out_w as usize),
        DType::F64 => adaptive_avg_pool2d_f64(input, out_h as usize, out_w as usize),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

// ---------------------------------------------------------------------------
// adaptive_max_pool2d
// ---------------------------------------------------------------------------

fn adaptive_max_pool2d_f32(
    input: &BorrowedTensor,
    out_h: usize,
    out_w: usize,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f32>(input) };
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };

    let plane_elems = c * out_h * out_w;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let hs = adaptive_start(oh as i64, out_h as i64, h as i64) as usize;
                    let he = adaptive_end(oh as i64, out_h as i64, h as i64) as usize;
                    for ow in 0..out_w {
                        let ws = adaptive_start(ow as i64, out_w as i64, w as i64) as usize;
                        let we = adaptive_end(ow as i64, out_w as i64, w as i64) as usize;
                        let mut best = f32::NEG_INFINITY;
                        for ih in hs..he {
                            for iw in ws..we {
                                let v = input_data[plane + ih * w + iw];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        out_plane[oh * out_w + ow] = best;
                    }
                }
            }
        });
    Ok(out)
}

fn adaptive_max_pool2d_f64(
    input: &BorrowedTensor,
    out_h: usize,
    out_w: usize,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let h = input.shape[2] as usize;
    let w = input.shape[3] as usize;
    let mut out = OwnedTensor::new(
        input.dtype,
        vec![b as i64, c as i64, out_h as i64, out_w as i64],
    );
    let input_data = unsafe { typed_slice::<f64>(input) };
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };

    let plane_elems = c * out_h * out_w;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * h * w;
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                for oh in 0..out_h {
                    let hs = adaptive_start(oh as i64, out_h as i64, h as i64) as usize;
                    let he = adaptive_end(oh as i64, out_h as i64, h as i64) as usize;
                    for ow in 0..out_w {
                        let ws = adaptive_start(ow as i64, out_w as i64, w as i64) as usize;
                        let we = adaptive_end(ow as i64, out_w as i64, w as i64) as usize;
                        let mut best = f64::NEG_INFINITY;
                        for ih in hs..he {
                            for iw in ws..we {
                                let v = input_data[plane + ih * w + iw];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        out_plane[oh * out_w + ow] = best;
                    }
                }
            }
        });
    Ok(out)
}

/// Adaptive max pooling to a fixed (oh, ow) output.
pub fn adaptive_max_pool2d(
    input: &BorrowedTensor,
    output_size: Option<&serde_json::Value>,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "adaptive_max_pool2d input")?;
    if input.shape.len() != 4 {
        return Err(unsupported(
            "adaptive_max_pool2d: input must be 4-D (B,C,H,W)",
        ));
    }
    let (out_h, out_w) = match output_size {
        None => (1, 1),
        Some(v) => {
            if let Some(s) = v.as_i64() {
                (s, s)
            } else if let Some(arr) = v.as_array() {
                let vals: Vec<i64> = arr.iter().filter_map(|x| x.as_i64()).collect();
                match vals.len() {
                    1 => (vals[0], vals[0]),
                    2 => (vals[0], vals[1]),
                    _ => {
                        return Err(unsupported(
                            "adaptive_max_pool2d: output_size must be an int or pair",
                        ))
                    }
                }
            } else {
                return Err(unsupported(
                    "adaptive_max_pool2d: output_size must be an int or pair",
                ));
            }
        }
    };
    if out_h <= 0 || out_w <= 0 {
        return Err(unsupported(
            "adaptive_max_pool2d: output_size must be positive",
        ));
    }
    match input.dtype {
        DType::F32 => adaptive_max_pool2d_f32(input, out_h as usize, out_w as usize),
        DType::F64 => adaptive_max_pool2d_f64(input, out_h as usize, out_w as usize),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

// ---------------------------------------------------------------------------
// 1-D variants (B, C, L)
// ---------------------------------------------------------------------------

fn max_pool1d_f32(
    input: &BorrowedTensor,
    kernel: usize,
    stride: usize,
    padding: i64,
    ceil_mode: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let l = input.shape[2] as usize;
    let out_l = if ceil_mode {
        (((l as i64 + 2 * padding - kernel as i64 - 1) as f64 / stride as f64).ceil() as i64 + 1)
            .max(1)
    } else {
        pool_out_size(l as i64, kernel as i64, stride as i64, padding, 1)
    };
    if out_l < 1 {
        return Err(unsupported("max_pool1d: output size must be positive"));
    }
    let out_l = out_l as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, c as i64, out_l as i64]);
    let input_data = unsafe { typed_slice::<f32>(input) };
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
    let plane_elems = c * out_l;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * l;
                let out_plane = &mut out_plane[ci * out_l..(ci + 1) * out_l];
                for ol in 0..out_l {
                    let l_start = ol as i64 * stride as i64 - padding;
                    let mut best = f32::NEG_INFINITY;
                    for k in 0..kernel {
                        let il = l_start + k as i64;
                        if il < 0 || il >= l as i64 {
                            continue;
                        }
                        let v = input_data[plane + il as usize];
                        if v > best {
                            best = v;
                        }
                    }
                    out_plane[ol] = best;
                }
            }
        });
    Ok(out)
}

fn max_pool1d_f64(
    input: &BorrowedTensor,
    kernel: usize,
    stride: usize,
    padding: i64,
    ceil_mode: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let l = input.shape[2] as usize;
    let out_l = if ceil_mode {
        (((l as i64 + 2 * padding - kernel as i64 - 1) as f64 / stride as f64).ceil() as i64 + 1)
            .max(1)
    } else {
        pool_out_size(l as i64, kernel as i64, stride as i64, padding, 1)
    };
    if out_l < 1 {
        return Err(unsupported("max_pool1d: output size must be positive"));
    }
    let out_l = out_l as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, c as i64, out_l as i64]);
    let input_data = unsafe { typed_slice::<f64>(input) };
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
    let plane_elems = c * out_l;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * l;
                let out_plane = &mut out_plane[ci * out_l..(ci + 1) * out_l];
                for ol in 0..out_l {
                    let l_start = ol as i64 * stride as i64 - padding;
                    let mut best = f64::NEG_INFINITY;
                    for k in 0..kernel {
                        let il = l_start + k as i64;
                        if il < 0 || il >= l as i64 {
                            continue;
                        }
                        let v = input_data[plane + il as usize];
                        if v > best {
                            best = v;
                        }
                    }
                    out_plane[ol] = best;
                }
            }
        });
    Ok(out)
}

/// 1-D max pooling.
pub fn max_pool1d(
    input: &BorrowedTensor,
    kernel: Option<&serde_json::Value>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    ceil_mode: bool,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "max_pool1d input")?;
    if input.shape.len() != 3 {
        return Err(unsupported("max_pool1d: input must be 3-D (B,C,L)"));
    }
    let kernel = scalar(kernel, "kernel", 0)?;
    if kernel <= 0 {
        return Err(unsupported("max_pool1d: kernel must be positive"));
    }
    let stride = scalar(stride, "stride", 0)?;
    let stride = if stride == 0 { kernel } else { stride };
    let padding = scalar(padding, "padding", 0)?;
    match input.dtype {
        DType::F32 => max_pool1d_f32(input, kernel as usize, stride as usize, padding, ceil_mode),
        DType::F64 => max_pool1d_f64(input, kernel as usize, stride as usize, padding, ceil_mode),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

fn avg_pool1d_f32(
    input: &BorrowedTensor,
    kernel: usize,
    stride: usize,
    padding: i64,
    ceil_mode: bool,
    count_include_pad: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let l = input.shape[2] as usize;
    let out_l = if ceil_mode {
        (((l as i64 + 2 * padding - kernel as i64 - 1) as f64 / stride as f64).ceil() as i64 + 1)
            .max(1)
    } else {
        pool_out_size(l as i64, kernel as i64, stride as i64, padding, 1)
    };
    if out_l < 1 {
        return Err(unsupported("avg_pool1d: output size must be positive"));
    }
    let out_l = out_l as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, c as i64, out_l as i64]);
    let input_data = unsafe { typed_slice::<f32>(input) };
    let out_data = unsafe { typed_mut_slice::<f32>(&mut out) };
    let plane_elems = c * out_l;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * l;
                let out_plane = &mut out_plane[ci * out_l..(ci + 1) * out_l];
                for ol in 0..out_l {
                    let l_start = ol as i64 * stride as i64 - padding;
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for k in 0..kernel {
                        let il = l_start + k as i64;
                        if il < 0 || il >= l as i64 {
                            continue;
                        }
                        sum += input_data[plane + il as usize];
                        count += 1;
                    }
                    let divisor = if count_include_pad {
                        kernel
                    } else {
                        count.max(1)
                    };
                    out_plane[ol] = sum / divisor as f32;
                }
            }
        });
    Ok(out)
}

fn avg_pool1d_f64(
    input: &BorrowedTensor,
    kernel: usize,
    stride: usize,
    padding: i64,
    ceil_mode: bool,
    count_include_pad: bool,
) -> PyResult<OwnedTensor> {
    let b = input.shape[0] as usize;
    let c = input.shape[1] as usize;
    let l = input.shape[2] as usize;
    let out_l = if ceil_mode {
        (((l as i64 + 2 * padding - kernel as i64 - 1) as f64 / stride as f64).ceil() as i64 + 1)
            .max(1)
    } else {
        pool_out_size(l as i64, kernel as i64, stride as i64, padding, 1)
    };
    if out_l < 1 {
        return Err(unsupported("avg_pool1d: output size must be positive"));
    }
    let out_l = out_l as usize;
    let mut out = OwnedTensor::new(input.dtype, vec![b as i64, c as i64, out_l as i64]);
    let input_data = unsafe { typed_slice::<f64>(input) };
    let out_data = unsafe { typed_mut_slice::<f64>(&mut out) };
    let plane_elems = c * out_l;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let plane = (bi * c + ci) * l;
                let out_plane = &mut out_plane[ci * out_l..(ci + 1) * out_l];
                for ol in 0..out_l {
                    let l_start = ol as i64 * stride as i64 - padding;
                    let mut sum = 0.0f64;
                    let mut count = 0usize;
                    for k in 0..kernel {
                        let il = l_start + k as i64;
                        if il < 0 || il >= l as i64 {
                            continue;
                        }
                        sum += input_data[plane + il as usize];
                        count += 1;
                    }
                    let divisor = if count_include_pad {
                        kernel
                    } else {
                        count.max(1)
                    };
                    out_plane[ol] = sum / divisor as f64;
                }
            }
        });
    Ok(out)
}

/// 1-D average pooling.
pub fn avg_pool1d(
    input: &BorrowedTensor,
    kernel: Option<&serde_json::Value>,
    stride: Option<&serde_json::Value>,
    padding: Option<&serde_json::Value>,
    ceil_mode: bool,
    count_include_pad: bool,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "avg_pool1d input")?;
    if input.shape.len() != 3 {
        return Err(unsupported("avg_pool1d: input must be 3-D (B,C,L)"));
    }
    let kernel = scalar(kernel, "kernel", 0)?;
    if kernel <= 0 {
        return Err(unsupported("avg_pool1d: kernel must be positive"));
    }
    let stride = scalar(stride, "stride", 0)?;
    let stride = if stride == 0 { kernel } else { stride };
    let padding = scalar(padding, "padding", 0)?;
    match input.dtype {
        DType::F32 => avg_pool1d_f32(
            input,
            kernel as usize,
            stride as usize,
            padding,
            ceil_mode,
            count_include_pad,
        ),
        DType::F64 => avg_pool1d_f64(
            input,
            kernel as usize,
            stride as usize,
            padding,
            ceil_mode,
            count_include_pad,
        ),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}
