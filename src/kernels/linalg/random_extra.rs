//! Random and creation: rand/randn/randint/randperm, empty/zeros_like/ones_like/full_like. Inherits linalg root imports via super; pure move.

use super::*;

// ── 106-115. Random number generation & creation ──
pub fn rand(shape: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, shape.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in 0..od.len() {
        od[i] = rand::random::<f32>();
    }
    Ok(out)
}

pub fn randn(shape: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, shape.to_vec());
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    for i in (0..od.len()).step_by(2) {
        let u1: f32 = rand::random::<f32>().max(1e-7);
        let u2: f32 = rand::random::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI as f32 * u2;
        od[i] = r * theta.cos();
        if i + 1 < od.len() {
            od[i + 1] = r * theta.sin();
        }
    }
    Ok(out)
}

pub fn randint(low: i64, high: i64, shape: &[i64]) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, shape.to_vec());
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    let span = (high - low).max(1) as u64;
    for i in 0..od.len() {
        od[i] = low + (rand::random::<u64>() % span) as i64;
    }
    Ok(out)
}

pub fn randperm(n: i64) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::I64, vec![n]);
    let od = unsafe { typed_mut_slice::<i64>(&mut out) };
    for i in 0..n as usize {
        od[i] = i as i64;
    }
    for i in (1..n as usize).rev() {
        let j = rand::random::<usize>() % (i + 1);
        od.swap(i, j);
    }
    Ok(out)
}

pub fn empty(shape: &[i64], dtype: DType) -> PyResult<OwnedTensor> {
    crate::shape_ops::zeros(shape, dtype)
}

pub fn zeros_like(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::shape_ops::zeros(&a.shape, a.dtype)
}

pub fn ones_like(a: &BorrowedTensor) -> PyResult<OwnedTensor> {
    crate::shape_ops::ones(&a.shape, a.dtype)
}

pub fn full_like(a: &BorrowedTensor, value: f64) -> PyResult<OwnedTensor> {
    crate::shape_ops::full(&a.shape, value, a.dtype)
}
