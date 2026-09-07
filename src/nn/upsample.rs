//! Upsampling (Phase 3): nearest-neighbor and bilinear 2-D upsampling.
//!
//! Input is (B, C, H, W); output size is given explicitly (torch's
//! `F.interpolate(x, size=...)` lowers to these ops in compiled graphs).
//! All kernels support f32/f64 and require contiguous inputs.

use crate::dlpack::{unsupported, BorrowedTensor, DType, OwnedTensor};
use pyo3::prelude::*;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

fn require_contiguous(t: &BorrowedTensor, what: &str) -> PyResult<()> {
    if !t.is_contiguous() {
        return Err(unsupported(&format!(
            "{what} must be contiguous for upsampling"
        )));
    }
    Ok(())
}

/// Parse the `size` kwarg: an int, a 2-list, or a single 1-item list.
fn size_pair(v: Option<&serde_json::Value>) -> PyResult<(usize, usize)> {
    let Some(v) = v else {
        return Err(unsupported("upsample: missing size"));
    };
    if let Some(s) = v.as_i64() {
        if s <= 0 {
            return Err(unsupported("upsample: size must be positive"));
        }
        return Ok((s as usize, s as usize));
    }
    if let Some(arr) = v.as_array() {
        let vals: Vec<i64> = arr.iter().filter_map(|x| x.as_i64()).collect();
        let (h, w) = match vals.len() {
            1 => (vals[0], vals[0]),
            2 => (vals[0], vals[1]),
            _ => return Err(unsupported("upsample: size must be an int or 2-int list")),
        };
        if h <= 0 || w <= 0 {
            return Err(unsupported("upsample: size must be positive"));
        }
        return Ok((h as usize, w as usize));
    }
    Err(unsupported("upsample: size must be an int or 2-int list"))
}

// ---------------------------------------------------------------------------
// Nearest neighbor
// ---------------------------------------------------------------------------

fn upsample_nearest2d_f32(
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
    // torch's nearest (align_corners ignored): src = floor(dst * in/out), clamped.
    let (sh, sw) = (h as f64 / out_h as f64, w as f64 / out_w as f64);
    let (max_h, max_w) = ((h - 1) as f64, (w - 1) as f64);

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
                    let ih = ((oh as f64) * sh).floor().min(max_h) as usize;
                    for ow in 0..out_w {
                        let iw = ((ow as f64) * sw).floor().min(max_w) as usize;
                        out_plane[oh * out_w + ow] = input_data[plane + ih * w + iw];
                    }
                }
            }
        });
    Ok(out)
}

fn upsample_nearest2d_f64(
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
    let (sh, sw) = (h as f64 / out_h as f64, w as f64 / out_w as f64);
    let (max_h, max_w) = ((h - 1) as f64, (w - 1) as f64);

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
                    let ih = ((oh as f64) * sh).floor().min(max_h) as usize;
                    for ow in 0..out_w {
                        let iw = ((ow as f64) * sw).floor().min(max_w) as usize;
                        out_plane[oh * out_w + ow] = input_data[plane + ih * w + iw];
                    }
                }
            }
        });
    Ok(out)
}

/// Nearest-neighbor 2-D upsample to an explicit output size.
pub fn upsample_nearest2d(
    input: &BorrowedTensor,
    size: Option<&serde_json::Value>,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "upsample_nearest2d input")?;
    if input.shape.len() != 4 {
        return Err(unsupported(
            "upsample_nearest2d: input must be 4-D (B,C,H,W)",
        ));
    }
    let (out_h, out_w) = size_pair(size)?;
    match input.dtype {
        DType::F32 => upsample_nearest2d_f32(input, out_h, out_w),
        DType::F64 => upsample_nearest2d_f64(input, out_h, out_w),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}

// ---------------------------------------------------------------------------
// Bilinear (align_corners=false, the torch default for F.interpolate)
// ---------------------------------------------------------------------------

fn bilinear_sample(data: &[f32], plane: usize, h: usize, w: usize, y: f64, x: f64) -> f32 {
    let y = y.max(0.0).min((h - 1) as f64);
    let x = x.max(0.0).min((w - 1) as f64);
    let y0 = y.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let wy = (y - y0 as f64) as f32;
    let wx = (x - x0 as f64) as f32;
    let v00 = data[plane + y0 * w + x0];
    let v01 = data[plane + y0 * w + x1];
    let v10 = data[plane + y1 * w + x0];
    let v11 = data[plane + y1 * w + x1];
    let top = v00 * (1.0 - wx) + v01 * wx;
    let bottom = v10 * (1.0 - wx) + v11 * wx;
    top * (1.0 - wy) + bottom * wy
}

fn upsample_bilinear2d_f32(
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
    let (sy, sx) = (h as f64 / out_h as f64, w as f64 / out_w as f64);

    let plane_elems = c * out_h * out_w;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                let plane = (bi * c + ci) * h * w;
                for oh in 0..out_h {
                    // align_corners=false: src = (dst + 0.5) * scale - 0.5
                    let y = (oh as f64 + 0.5) * sy - 0.5;
                    for ow in 0..out_w {
                        let x = (ow as f64 + 0.5) * sx - 0.5;
                        out_plane[oh * out_w + ow] = bilinear_sample(input_data, plane, h, w, y, x);
                    }
                }
            }
        });
    Ok(out)
}

fn bilinear_sample_f64(data: &[f64], plane: usize, h: usize, w: usize, y: f64, x: f64) -> f64 {
    let y = y.max(0.0).min((h - 1) as f64);
    let x = x.max(0.0).min((w - 1) as f64);
    let y0 = y.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let wy = y - y0 as f64;
    let wx = x - x0 as f64;
    let v00 = data[plane + y0 * w + x0];
    let v01 = data[plane + y0 * w + x1];
    let v10 = data[plane + y1 * w + x0];
    let v11 = data[plane + y1 * w + x1];
    let top = v00 * (1.0 - wx) + v01 * wx;
    let bottom = v10 * (1.0 - wx) + v11 * wx;
    top * (1.0 - wy) + bottom * wy
}

fn upsample_bilinear2d_f64(
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
    let (sy, sx) = (h as f64 / out_h as f64, w as f64 / out_w as f64);

    let plane_elems = c * out_h * out_w;
    use rayon::prelude::*;
    out_data
        .par_chunks_mut(plane_elems)
        .enumerate()
        .for_each(|(bi, out_plane)| {
            for ci in 0..c {
                let out_plane = &mut out_plane[ci * out_h * out_w..(ci + 1) * out_h * out_w];
                let plane = (bi * c + ci) * h * w;
                for oh in 0..out_h {
                    let y = (oh as f64 + 0.5) * sy - 0.5;
                    for ow in 0..out_w {
                        let x = (ow as f64 + 0.5) * sx - 0.5;
                        out_plane[oh * out_w + ow] =
                            bilinear_sample_f64(input_data, plane, h, w, y, x);
                    }
                }
            }
        });
    Ok(out)
}

/// Bilinear 2-D upsample (align_corners=false) to an explicit output size.
pub fn upsample_bilinear2d(
    input: &BorrowedTensor,
    size: Option<&serde_json::Value>,
) -> PyResult<OwnedTensor> {
    require_contiguous(input, "upsample_bilinear2d input")?;
    if input.shape.len() != 4 {
        return Err(unsupported(
            "upsample_bilinear2d: input must be 4-D (B,C,H,W)",
        ));
    }
    let (out_h, out_w) = size_pair(size)?;
    match input.dtype {
        DType::F32 => upsample_bilinear2d_f32(input, out_h, out_w),
        DType::F64 => upsample_bilinear2d_f64(input, out_h, out_w),

        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("this kernel only supports f32/f64 tensors"));
        }
    }
}
