//! Loss kernels (Phase 4).
//!
//! All losses support torch's reduction convention: 0 = sum, 1 = mean,
//! 2 = none (elementwise).  The `nll_loss_forward` op mirrors aten's tuple
//! contract (loss, total_weight): the engine produces the loss; getitem(0)
//! aliases it and dead getitem(1) nodes are dropped by the parser.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, elem_count, unsupported};
use pyo3::prelude::*;

/// Read a tensor's elements as a typed slice.
unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

/// Write typed data into an owned tensor.
unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

/// Wrap a scalar f32 into a 0-dim owned tensor.
fn scalar_f32(v: f32) -> OwnedTensor {
    let mut out = OwnedTensor::new(DType::F32, vec![]);
    let d = unsafe { typed_mut_slice::<f32>(&mut out) };
    d[0] = v;
    out
}

fn scalar_f64(v: f64) -> OwnedTensor {
    let mut out = OwnedTensor::new(DType::F64, vec![]);
    let d = unsafe { typed_mut_slice::<f64>(&mut out) };
    d[0] = v;
    out
}

/// Numeric helpers for loss kernels (f32/f64 without From<f64> friction).
trait LossScalar: Copy + Send + Sync + std::ops::Add<Output = Self> {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}
impl LossScalar for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl LossScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Apply a reduction (0=sum, 1=mean, 2=none) to an elementwise loss buffer.
fn reduce_loss<T: LossScalar>(data: &[T], n: usize, reduction: i64, elem_out: &mut OwnedTensor) -> PyResult<()> {
    match reduction {
        2 => {
            // none: copy elements through (keep the elementwise shape)
            let out_data = unsafe { typed_mut_slice::<T>(elem_out) };
            out_data.copy_from_slice(data);
        }
        0 | 1 => {
            // sum/mean: scalar output (0-dim tensor, like torch)
            elem_out.shape = vec![];
            let mut total = T::from_f64(0.0);
            for &v in data {
                total = total + v;
            }
            let value: f64 = if reduction == 1 {
                total.to_f64() / n.max(1) as f64
            } else {
                total.to_f64()
            };
            let d = unsafe { typed_mut_slice::<T>(elem_out) };
            d[0] = T::from_f64(value);
        }
        other => return Err(unsupported(&format!("unsupported reduction {other}"))),
    }
    Ok(())
}

/// aten.nll_loss_forward(input, target, weight, reduction, ignore_index)
///
/// `input` is log-probabilities [N, C] (or [N, C, ...]), `target` is int64
/// class indices with the leading dims of `input`.  Returns only the loss
/// (the total_weight half of aten's tuple is produced by eager only).
pub fn nll_loss_forward(
    input: &BorrowedTensor,
    target: &BorrowedTensor,
    reduction: i64,
    ignore_index: i64,
) -> PyResult<OwnedTensor> {
    if input.shape.len() < 2 {
        return Err(unsupported("nll_loss input must be at least 2D"));
    }
    if target.dtype != DType::I64 && target.dtype != DType::I32 {
        return Err(unsupported("nll_loss target must be int64/int32"));
    }
    let n = elem_count(&input.shape[..input.shape.len() - 1]);
    let c = input.shape[input.shape.len() - 1] as usize;
    if elem_count(&target.shape) != n {
        return Err(unsupported(&format!(
            "nll_loss target size {} does not match input batch {}",
            elem_count(&target.shape),
            n
        )));
    }

    match input.dtype {
        DType::F32 => {
            let x = unsafe { typed_slice::<f32>(input) };
            let losses: Vec<f32> = match target.dtype {
                DType::I64 => {
                    let t = unsafe { typed_slice::<i64>(target) };
                    (0..n)
                        .map(|i| {
                            let cls = t[i] as isize;
                            if cls == ignore_index as isize {
                                Ok(0.0)
                            } else {
                                if cls < 0 || cls as usize >= c {
                                    return Err(unsupported(&format!(
                                        "nll_loss target {cls} out of range [0, {c})"
                                    )));
                                }
                                Ok(-x[i * c + cls as usize])
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?
                }
                DType::I32 => {
                    let t = unsafe { typed_slice::<i32>(target) };
                    (0..n)
                        .map(|i| {
                            let cls = t[i] as isize;
                            if cls == ignore_index as isize {
                                Ok(0.0)
                            } else {
                                if cls < 0 || cls as usize >= c {
                                    return Err(unsupported(&format!(
                                        "nll_loss target {cls} out of range [0, {c})"
                                    )));
                                }
                                Ok(-x[i * c + cls as usize])
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?
                }
                _ => unreachable!(),
            };
            // mean over non-ignored samples
            let ignored: usize = match target.dtype {
                DType::I64 => unsafe { typed_slice::<i64>(target) }.iter().filter(|&&v| v as isize == ignore_index as isize).count(),
                _ => unsafe { typed_slice::<i32>(target) }.iter().filter(|&&v| v as isize == ignore_index as isize).count(),
            };
            let denom = if reduction == 1 { (n - ignored).max(1) as f32 } else { 1.0 };
            let total: f32 = losses.iter().sum();
            let out = match reduction {
                0 => total,
                1 => total / denom,
                2 => {
                    // none: elementwise output [N]
                    let mut o = OwnedTensor::new(DType::F32, target.shape.clone());
                    unsafe { typed_mut_slice::<f32>(&mut o) }.copy_from_slice(&losses);
                    return Ok(o);
                }
                other => return Err(unsupported(&format!("unsupported reduction {other}"))),
            };
            Ok(scalar_f32(out))
        }
        DType::F64 => {
            let x = unsafe { typed_slice::<f64>(input) };
            let losses: Vec<f64> = match target.dtype {
                DType::I64 => {
                    let t = unsafe { typed_slice::<i64>(target) };
                    (0..n)
                        .map(|i| {
                            let cls = t[i] as isize;
                            if cls == ignore_index as isize {
                                Ok(0.0)
                            } else {
                                if cls < 0 || cls as usize >= c {
                                    return Err(unsupported(&format!(
                                        "nll_loss target {cls} out of range [0, {c})"
                                    )));
                                }
                                Ok(-x[i * c + cls as usize])
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?
                }
                DType::I32 => {
                    let t = unsafe { typed_slice::<i32>(target) };
                    (0..n)
                        .map(|i| {
                            let cls = t[i] as isize;
                            if cls == ignore_index as isize {
                                Ok(0.0)
                            } else {
                                if cls < 0 || cls as usize >= c {
                                    return Err(unsupported(&format!(
                                        "nll_loss target {cls} out of range [0, {c})"
                                    )));
                                }
                                Ok(-x[i * c + cls as usize])
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?
                }
                _ => unreachable!(),
            };
            let ignored: usize = match target.dtype {
                DType::I64 => unsafe { typed_slice::<i64>(target) }.iter().filter(|&&v| v as isize == ignore_index as isize).count(),
                _ => unsafe { typed_slice::<i32>(target) }.iter().filter(|&&v| v as isize == ignore_index as isize).count(),
            };
            let denom = if reduction == 1 { (n - ignored).max(1) as f64 } else { 1.0 };
            let total: f64 = losses.iter().sum();
            let out = match reduction {
                0 => total,
                1 => total / denom,
                2 => {
                    let mut o = OwnedTensor::new(DType::F64, target.shape.clone());
                    unsafe { typed_mut_slice::<f64>(&mut o) }.copy_from_slice(&losses);
                    return Ok(o);
                }
                other => return Err(unsupported(&format!("unsupported reduction {other}"))),
            };
            Ok(scalar_f64(out))
        }
        DType::I64 | DType::I32 | DType::Bool => {
            Err(unsupported("nll_loss input must be f32/f64"))
        }
    }
}

/// aten.mse_loss(input, target, reduction)
pub fn mse_loss(a: &BorrowedTensor, b: &BorrowedTensor, reduction: i64) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    match a.dtype {
        DType::F32 => {
            let x = unsafe { typed_slice::<f32>(a) };
            let y = unsafe { typed_slice::<f32>(b) };
            let buf: Vec<f32> = x.iter().zip(y.iter()).map(|(x, y)| (x - y) * (x - y)).collect();
            reduce_loss(&buf, n, reduction, &mut out)?;
        }
        DType::F64 => {
            let x = unsafe { typed_slice::<f64>(a) };
            let y = unsafe { typed_slice::<f64>(b) };
            let buf: Vec<f64> = x.iter().zip(y.iter()).map(|(x, y)| (x - y) * (x - y)).collect();
            reduce_loss(&buf, n, reduction, &mut out)?;
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("mse_loss requires f32/f64 tensors"));
        }
    }
    Ok(out)
}

/// aten.smooth_l1_loss(input, target, reduction, beta)
pub fn smooth_l1_loss(a: &BorrowedTensor, b: &BorrowedTensor, reduction: i64, beta: f64) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let beta = beta as f32;
    match a.dtype {
        DType::F32 => {
            let x = unsafe { typed_slice::<f32>(a) };
            let y = unsafe { typed_slice::<f32>(b) };
            let buf: Vec<f32> = x
                .iter()
                .zip(y.iter())
                .map(|(x, y)| {
                    let d = (x - y).abs();
                    if d < beta { 0.5 * d * d / beta } else { d - 0.5 * beta }
                })
                .collect();
            reduce_loss(&buf, n, reduction, &mut out)?;
        }
        DType::F64 => {
            let x = unsafe { typed_slice::<f64>(a) };
            let y = unsafe { typed_slice::<f64>(b) };
            let beta64 = beta as f64;
            let buf: Vec<f64> = x
                .iter()
                .zip(y.iter())
                .map(|(x, y)| {
                    let d = (x - y).abs();
                    if d < beta64 { 0.5 * d * d / beta64 } else { d - 0.5 * beta64 }
                })
                .collect();
            reduce_loss(&buf, n, reduction, &mut out)?;
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("smooth_l1_loss requires f32/f64 tensors"));
        }
    }
    Ok(out)
}

/// aten.binary_cross_entropy(input, target, weight, reduction)
pub fn binary_cross_entropy(a: &BorrowedTensor, b: &BorrowedTensor, reduction: i64) -> PyResult<OwnedTensor> {
    let n = elem_count(&a.shape);
    let mut out = OwnedTensor::new(a.dtype, a.shape.clone());
    let clamp = |x: f32| x.clamp(1e-12, 1.0 - 1e-12);
    match a.dtype {
        DType::F32 => {
            let x = unsafe { typed_slice::<f32>(a) };
            let y = unsafe { typed_slice::<f32>(b) };
            let buf: Vec<f32> = x
                .iter()
                .zip(y.iter())
                .map(|(&x, &y)| -(y * clamp(x).ln() + (1.0 - y) * (1.0 - clamp(x)).ln()))
                .collect();
            reduce_loss(&buf, n, reduction, &mut out)?;
        }
        DType::F64 => {
            let x = unsafe { typed_slice::<f64>(a) };
            let y = unsafe { typed_slice::<f64>(b) };
            let buf: Vec<f64> = x
                .iter()
                .zip(y.iter())
                .map(|(&x, &y)| {
                    let cx = x.clamp(1e-12, 1.0 - 1e-12);
                    -(y * cx.ln() + (1.0 - y) * (1.0 - cx).ln())
                })
                .collect();
            reduce_loss(&buf, n, reduction, &mut out)?;
        }
        DType::I64 | DType::I32 | DType::Bool => {
            return Err(unsupported("binary_cross_entropy requires f32/f64 tensors"));
        }
    }
    Ok(out)
}
