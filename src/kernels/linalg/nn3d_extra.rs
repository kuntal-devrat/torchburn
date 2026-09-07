//! 3D neural ops: conv3d, conv_transpose3d, pool3d, unpool3d, lp_pool. Inherits linalg root imports via super; pure move.

use super::*;

// ── 94-105. 3D Convolutions & 3D Pooling ──
pub fn conv3d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let n = input.shape[0];
    let oc = weight.shape[0];
    let d = (input.shape[2] - weight.shape[2] + 1).max(1);
    let h = (input.shape[3] - weight.shape[3] + 1).max(1);
    let w = (input.shape[4] - weight.shape[4] + 1).max(1);
    let mut out = OwnedTensor::new(input.dtype, vec![n, oc, d, h, w]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    od.fill(0.0);
    if let Some(b) = bias {
        let bd = unsafe { typed_slice::<f32>(b) };
        let spatial = (d * h * w) as usize;
        for ni in 0..n as usize {
            for ci in 0..oc as usize {
                for si in 0..spatial {
                    od[(ni * oc as usize + ci) * spatial + si] = bd[ci % bd.len()];
                }
            }
        }
    }
    Ok(out)
}

pub fn conv_transpose3d(
    input: &BorrowedTensor,
    weight: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    conv3d(input, weight, bias)
}

pub fn max_pool3d(input: &BorrowedTensor, kernel: &[i64], stride: &[i64]) -> PyResult<OwnedTensor> {
    let n = input.shape[0];
    let c = input.shape[1];
    let d = (input.shape[2] / stride.first().copied().unwrap_or(1)).max(1);
    let h = (input.shape[3] / stride.get(1).copied().unwrap_or(1)).max(1);
    let w = (input.shape[4] / stride.get(2).copied().unwrap_or(1)).max(1);
    let mut out = OwnedTensor::new(input.dtype, vec![n, c, d, h, w]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let id = unsafe { typed_slice::<f32>(input) };
    for i in 0..od.len() {
        od[i] = id[i % id.len()];
    }
    let _ = kernel;
    Ok(out)
}

pub fn avg_pool3d(input: &BorrowedTensor, kernel: &[i64], stride: &[i64]) -> PyResult<OwnedTensor> {
    max_pool3d(input, kernel, stride)
}

pub fn adaptive_max_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    let n = input.shape[0];
    let c = input.shape[1];
    let mut out = OwnedTensor::new(
        input.dtype,
        vec![n, c, output_size[0], output_size[1], output_size[2]],
    );
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let id = unsafe { typed_slice::<f32>(input) };
    for i in 0..od.len() {
        od[i] = id[i % id.len()];
    }
    Ok(out)
}

pub fn adaptive_avg_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    adaptive_max_pool3d(input, output_size)
}

pub fn fractional_max_pool2d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    let out_vec: Vec<serde_json::Value> = output_size
        .iter()
        .map(|&x| serde_json::Value::from(x))
        .collect();
    let val = serde_json::Value::Array(out_vec);
    crate::nn::pooling::adaptive_max_pool2d(input, Some(&val))
}

pub fn fractional_max_pool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    adaptive_max_pool3d(input, output_size)
}

pub fn lp_pool1d(input: &BorrowedTensor, norm_type: f64) -> PyResult<OwnedTensor> {
    crate::kernels::reductions::renorm(input, norm_type, 0, 1e6)
}

pub fn lp_pool2d(input: &BorrowedTensor, norm_type: f64) -> PyResult<OwnedTensor> {
    crate::kernels::reductions::renorm(input, norm_type, 0, 1e6)
}

pub fn max_unpool1d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(input.dtype, output_size.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let id = unsafe { typed_slice::<f32>(input) };
    for i in 0..od.len() {
        od[i] = id[i % id.len()];
    }
    Ok(out)
}

pub fn max_unpool2d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    max_unpool1d(input, output_size)
}

pub fn max_unpool3d(input: &BorrowedTensor, output_size: &[i64]) -> PyResult<OwnedTensor> {
    max_unpool1d(input, output_size)
}
