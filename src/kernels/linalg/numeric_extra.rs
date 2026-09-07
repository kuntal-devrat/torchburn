//! Numerical integration: trapezoid, trapz, cumulative_trapezoid. Inherits linalg root imports via super; pure move.

use super::*;

// ── 41-43. trapz, trapezoid, cumulative_trapezoid ──
pub fn trapezoid(
    y: &BorrowedTensor,
    x: Option<&BorrowedTensor>,
    dx: f64,
    dim: isize,
) -> PyResult<OwnedTensor> {
    let d = if dim < 0 {
        (y.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let n = y.shape[d] as usize;
    let mut out_shape = y.shape.clone();
    out_shape.remove(d);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let mut out = OwnedTensor::new(y.dtype, out_shape);
    match y.dtype {
        DType::F32 => {
            let yd = unsafe { typed_slice::<f32>(y) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut sum = 0.0_f32;
            let step = dx as f32;
            for i in 0..n.saturating_sub(1) {
                sum += 0.5 * (yd[i] + yd[i + 1]) * step;
            }
            od[0] = sum;
        }
        _ => return Err(unsupported("trapezoid only supports f32")),
    }
    let _ = x;
    Ok(out)
}

pub fn trapz(
    y: &BorrowedTensor,
    x: Option<&BorrowedTensor>,
    dx: f64,
    dim: isize,
) -> PyResult<OwnedTensor> {
    trapezoid(y, x, dx, dim)
}

pub fn cumulative_trapezoid(y: &BorrowedTensor, dx: f64, dim: isize) -> PyResult<OwnedTensor> {
    let d = if dim < 0 {
        (y.shape.len() as isize + dim) as usize
    } else {
        dim as usize
    };
    let mut out_shape = y.shape.clone();
    out_shape[d] = (out_shape[d] - 1).max(1);
    let mut out = OwnedTensor::new(y.dtype, out_shape);
    match y.dtype {
        DType::F32 => {
            let yd = unsafe { typed_slice::<f32>(y) };
            let od = unsafe { typed_mut_slice::<f32>(&mut out) };
            let mut acc = 0.0_f32;
            let step = dx as f32;
            for i in 0..od.len().min(yd.len().saturating_sub(1)) {
                acc += 0.5 * (yd[i] + yd[i + 1]) * step;
                od[i] = acc;
            }
        }
        _ => return Err(unsupported("cumulative_trapezoid only supports f32")),
    }
    Ok(out)
}
