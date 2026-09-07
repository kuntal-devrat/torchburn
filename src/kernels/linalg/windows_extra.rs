//! Window functions: hamming, kaiser, gaussian, exponential, triangular. Inherits linalg root imports via super; pure move.

use super::*;

// ── 61-65. Windows: hamming, kaiser, gaussian, exponential, triangular ──
pub fn hamming_window(window_length: i64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    for n in 0..window_length as usize {
        od[n] = (0.54 - 0.46 * (2.0 * PI * n as f64 / denom).cos()) as f32;
    }
    Ok(out)
}

pub fn kaiser_window(window_length: i64, beta: f64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let i0_beta = bessel_i0_f64(beta);
    for n in 0..window_length as usize {
        let x = 2.0 * n as f64 / denom - 1.0;
        let val = if x.abs() <= 1.0 {
            bessel_i0_f64(beta * (1.0 - x * x).sqrt()) / i0_beta
        } else {
            0.0
        };
        od[n] = val as f32;
    }
    Ok(out)
}

pub fn gaussian_window(window_length: i64, std: f64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let center = denom / 2.0;
    for n in 0..window_length as usize {
        let diff = (n as f64 - center) / std;
        od[n] = (-0.5 * diff * diff).exp() as f32;
    }
    Ok(out)
}

pub fn exponential_window(window_length: i64, tau: f64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        window_length as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let center = denom / 2.0;
    for n in 0..window_length as usize {
        let diff = (n as f64 - center).abs() / tau;
        od[n] = (-diff).exp() as f32;
    }
    Ok(out)
}

pub fn triangular_window(window_length: i64, periodic: bool) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(DType::F32, vec![window_length]);
    let od = unsafe { typed_mut_slice::<f32>(&mut out) };
    let denom = if periodic {
        (window_length + 1) as f64
    } else {
        (window_length - 1).max(1) as f64
    };
    let center = (window_length - 1) as f64 / 2.0;
    for n in 0..window_length as usize {
        od[n] = (1.0 - (n as f64 - center).abs() / (denom / 2.0)) as f32;
    }
    Ok(out)
}
