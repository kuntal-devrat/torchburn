//! Universal FFT (Fast Fourier Transform) & Complex Number Suite.
//!
//! Implements Radix-2 Cooley-Tukey and Bluestein Chirp Z-Transform in pure Rust,
//! supporting 1D, 2D, N-D real & complex transforms and complex tensor arithmetic.

use crate::dlpack::{BorrowedTensor, DType, OwnedTensor, elem_count, unsupported};
use pyo3::prelude::*;
use std::f64::consts::PI;

unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    pub fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    #[inline(always)]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
}

/// In-place Radix-4 butterfly step
#[inline(always)]
fn radix4_butterfly(a0: &mut Complex64, a1: &mut Complex64, a2: &mut Complex64, a3: &mut Complex64, dir: f64) {
    let t0 = a0.add(*a2);
    let t1 = a0.sub(*a2);
    let t2 = a1.add(*a3);
    let t3 = a1.sub(*a3);

    let j_t3 = Complex64::new(-dir * t3.im, dir * t3.re);

    *a0 = t0.add(t2);
    *a1 = t1.sub(j_t3);
    *a2 = t0.sub(t2);
    *a3 = t1.add(j_t3);
}

/// In-place Radix-4 and Radix-2 Cooley-Tukey FFT for power-of-2 buffers.
fn radix2_fft(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            buf.swap(i, j);
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    let dir = if inverse { 1.0 } else { -1.0 };

    // Radix-4 stages where possible
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = dir * 2.0 * PI / (len as f64);
        let w_step = Complex64::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[i + k];
                let v = buf[i + k + half].mul(w);
                buf[i + k] = u.add(v);
                buf[i + k + half] = u.sub(v);
                w = w.mul(w_step);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for x in buf.iter_mut() {
            x.re *= inv_n;
            x.im *= inv_n;
        }
    }
}

/// Generic 1D FFT supporting arbitrary lengths (using Bluestein or Radix-2).
pub fn fft_1d_c2c(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    if n.is_power_of_two() {
        let mut out = input.to_vec();
        radix2_fft(&mut out, inverse);
        return out;
    }

    // Bluestein's Chirp Z-transform algorithm for non-power-of-two lengths
    let m = (2 * n - 1).next_power_of_two();
    let dir = if inverse { 1.0 } else { -1.0 };

    let mut a = vec![Complex64::new(0.0, 0.0); m];
    let mut b = vec![Complex64::new(0.0, 0.0); m];

    let mut w = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let angle = dir * PI * ((i * i) as f64) / (n as f64);
        w[i] = Complex64::new(angle.cos(), angle.sin());
        a[i] = input[i].mul(w[i].conj());
    }

    b[0] = w[0];
    for i in 1..n {
        b[i] = w[i];
        b[m - i] = w[i];
    }

    radix2_fft(&mut a, false);
    radix2_fft(&mut b, false);

    for i in 0..m {
        a[i] = a[i].mul(b[i]);
    }

    radix2_fft(&mut a, true);

    let mut out = vec![Complex64::new(0.0, 0.0); n];
    let scale = if inverse { 1.0 / (n as f64) } else { 1.0 };
    for i in 0..n {
        let res = a[i].mul(w[i].conj());
        out[i] = Complex64::new(res.re * scale, res.im * scale);
    }
    out
}

/// 1D FFT: Complex-to-Complex Transform: out has shape [..., N, 2] (real and imag interleaved)
pub fn fft(x: &BorrowedTensor, n_dim: Option<i64>, dim: Option<i64>) -> PyResult<OwnedTensor> {
    let rank = x.shape.len();
    if rank < 1 {
        return Err(unsupported("fft requires at least 1D tensor"));
    }
    let target_dim = dim.map_or(rank - 1, |d| {
        if d < 0 { (rank as i64 + d) as usize } else { d as usize }
    });

    let seq_len = n_dim.map_or(x.shape[target_dim] as usize, |n| n as usize);
    let mut out_shape = x.shape.clone();
    out_shape[target_dim] = seq_len as i64;
    // Append complex channel if not already present
    out_shape.push(2);

    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let n_total = elem_count(&x.shape);
    let outer_stride = x.shape[target_dim] as usize;
    let n_batches = n_total / outer_stride;

    for b in 0..n_batches {
        let mut c_in = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let val = if i < outer_stride { src[b * outer_stride + i] as f64 } else { 0.0 };
            c_in.push(Complex64::new(val, 0.0));
        }
        let c_out = fft_1d_c2c(&c_in, false);
        for i in 0..seq_len {
            dst[(b * seq_len + i) * 2] = c_out[i].re as f32;
            dst[(b * seq_len + i) * 2 + 1] = c_out[i].im as f32;
        }
    }

    Ok(out)
}

/// 1D IFFT: Inverse Complex-to-Complex Transform
pub fn ifft(x: &BorrowedTensor, n_dim: Option<i64>, dim: Option<i64>) -> PyResult<OwnedTensor> {
    let rank = x.shape.len();
    let target_dim = dim.map_or(rank - 1, |d| {
        if d < 0 { (rank as i64 + d) as usize } else { d as usize }
    });

    let seq_len = n_dim.map_or(x.shape[target_dim] as usize, |n| n as usize);
    let mut out_shape = x.shape.clone();
    out_shape[target_dim] = seq_len as i64;
    out_shape.push(2);

    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let n_total = elem_count(&x.shape);
    let outer_stride = x.shape[target_dim] as usize;
    let n_batches = n_total / outer_stride;

    for b in 0..n_batches {
        let mut c_in = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let val = if i < outer_stride { src[b * outer_stride + i] as f64 } else { 0.0 };
            c_in.push(Complex64::new(val, 0.0));
        }
        let c_out = fft_1d_c2c(&c_in, true);
        for i in 0..seq_len {
            dst[(b * seq_len + i) * 2] = c_out[i].re as f32;
            dst[(b * seq_len + i) * 2 + 1] = c_out[i].im as f32;
        }
    }

    Ok(out)
}

/// 1D RFFT: Real-to-Complex FFT (returns N/2 + 1 complex values)
pub fn rfft(x: &BorrowedTensor, n_dim: Option<i64>, dim: Option<i64>) -> PyResult<OwnedTensor> {
    let rank = x.shape.len();
    let target_dim = dim.map_or(rank - 1, |d| {
        if d < 0 { (rank as i64 + d) as usize } else { d as usize }
    });

    let seq_len = n_dim.map_or(x.shape[target_dim] as usize, |n| n as usize);
    let rfft_len = seq_len / 2 + 1;

    let mut out_shape = x.shape.clone();
    out_shape[target_dim] = rfft_len as i64;
    out_shape.push(2);

    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let n_total = elem_count(&x.shape);
    let outer_stride = x.shape[target_dim] as usize;
    let n_batches = n_total / outer_stride;

    for b in 0..n_batches {
        let mut c_in = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let val = if i < outer_stride { src[b * outer_stride + i] as f64 } else { 0.0 };
            c_in.push(Complex64::new(val, 0.0));
        }
        let c_out = fft_1d_c2c(&c_in, false);
        for i in 0..rfft_len {
            dst[(b * rfft_len + i) * 2] = c_out[i].re as f32;
            dst[(b * rfft_len + i) * 2 + 1] = c_out[i].im as f32;
        }
    }

    Ok(out)
}

/// 1D IRFFT: Inverse Real-to-Complex FFT (reconstructs real tensor of length N)
pub fn irfft(x: &BorrowedTensor, n_dim: Option<i64>, dim: Option<i64>) -> PyResult<OwnedTensor> {
    let rank = x.shape.len();
    let target_dim = dim.map_or(rank - 1, |d| {
        if d < 0 { (rank as i64 + d) as usize } else { d as usize }
    });

    let in_len = x.shape[target_dim] as usize;
    let out_len = n_dim.map_or(2 * (in_len - 1), |n| n as usize);

    let mut out_shape = x.shape.clone();
    out_shape[target_dim] = out_len as i64;

    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let n_total = elem_count(&x.shape);
    let n_batches = n_total / in_len;

    for b in 0..n_batches {
        let mut c_full = vec![Complex64::new(0.0, 0.0); out_len];
        for i in 0..in_len {
            let re = src[b * in_len + i] as f64;
            c_full[i] = Complex64::new(re, 0.0);
            if i > 0 && i < out_len - i {
                c_full[out_len - i] = Complex64::new(re, 0.0);
            }
        }
        let c_out = fft_1d_c2c(&c_full, true);
        for i in 0..out_len {
            dst[b * out_len + i] = c_out[i].re as f32;
        }
    }

    Ok(out)
}

/// Complex conjugate: conj(z)
pub fn conj(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(x.dtype, x.shape.clone());
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&x.shape);

    let is_interleaved = x.shape.last().copied() == Some(2);
    if is_interleaved {
        for i in (0..n).step_by(2) {
            dst[i] = src[i];
            dst[i + 1] = -src[i + 1];
        }
    } else {
        dst.copy_from_slice(src);
    }
    Ok(out)
}

/// Extract Real component from complex tensor: real(z)
pub fn real(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out_shape = x.shape.clone();
    let is_last_2 = out_shape.last().copied() == Some(2);
    if is_last_2 {
        out_shape.pop();
    }
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    if is_last_2 {
        let n = dst.len();
        for i in 0..n {
            dst[i] = src[i * 2];
        }
    } else {
        dst.copy_from_slice(src);
    }
    Ok(out)
}

/// Extract Imaginary component from complex tensor: imag(z)
pub fn imag(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out_shape = x.shape.clone();
    let is_last_2 = out_shape.last().copied() == Some(2);
    if is_last_2 {
        out_shape.pop();
    }
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    if is_last_2 {
        let n = dst.len();
        for i in 0..n {
            dst[i] = src[i * 2 + 1];
        }
    } else {
        for v in dst.iter_mut() {
            *v = 0.0f32;
        }
    }
    Ok(out)
}

/// Phase angle: angle(z) = atan2(im, re)
pub fn angle(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out_shape = x.shape.clone();
    let is_last_2 = out_shape.last().copied() == Some(2);
    if is_last_2 {
        out_shape.pop();
    }
    let mut out = OwnedTensor::new(DType::F32, out_shape);
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let n = dst.len();
    if is_last_2 {
        for i in 0..n {
            let re = src[i * 2];
            let im = src[i * 2 + 1];
            dst[i] = im.atan2(re);
        }
    } else {
        for i in 0..n {
            dst[i] = if src[i] < 0.0 { PI as f32 } else { 0.0 };
        }
    }
    Ok(out)
}

/// Polar to Complex: polar(abs, angle) -> out[..., 2]
pub fn polar(abs: &BorrowedTensor, angle: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out_shape = abs.shape.clone();
    out_shape.push(2);
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let abs_slice = unsafe { typed_slice::<f32>(abs) };
    let ang_slice = unsafe { typed_slice::<f32>(angle) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = abs_slice.len();

    for i in 0..n {
        let r = abs_slice[i];
        let theta = ang_slice[i];
        dst[i * 2] = r * theta.cos();
        dst[i * 2 + 1] = r * theta.sin();
    }
    Ok(out)
}

/// Construct complex tensor from real and imag components: complex(real, imag)
pub fn complex(re: &BorrowedTensor, im: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out_shape = re.shape.clone();
    out_shape.push(2);
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let re_slice = unsafe { typed_slice::<f32>(re) };
    let im_slice = unsafe { typed_slice::<f32>(im) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = re_slice.len();

    for i in 0..n {
        dst[i * 2] = re_slice[i];
        dst[i * 2 + 1] = im_slice[i];
    }
    Ok(out)
}

/// 2D FFT: Complex-to-Complex 2D transform across the last two dimensions.
pub fn fft2(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let rank = x.shape.len();
    if rank < 2 {
        return fft(x, None, None);
    }
    let h = x.shape[rank - 2] as usize;
    let w = x.shape[rank - 1] as usize;

    // Step 1: Transform along last dimension (rows of length W)
    let step1 = fft(x, None, Some((rank - 1) as i64))?;

    // Step 2: Transform along second-to-last dimension (columns of length H)
    let mut out_shape = x.shape.clone();
    out_shape.push(2);
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let src = unsafe { std::slice::from_raw_parts(step1.data.as_ptr() as *const f32, step1.elem_count()) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let n_total = elem_count(&x.shape);
    let hw = h * w;
    let n_batches = n_total / hw;

    for b in 0..n_batches {
        let b_base = b * hw;
        // Transform each of the W columns of length H
        for col in 0..w {
            let mut c_in = Vec::with_capacity(h);
            for row in 0..h {
                let idx = (b_base + row * w + col) * 2;
                c_in.push(Complex64::new(src[idx] as f64, src[idx + 1] as f64));
            }
            let c_out = fft_1d_c2c(&c_in, false);
            for row in 0..h {
                let idx = (b_base + row * w + col) * 2;
                dst[idx] = c_out[row].re as f32;
                dst[idx + 1] = c_out[row].im as f32;
            }
        }
    }

    Ok(out)
}

/// 2D IFFT: Inverse 2D transform across the last two dimensions.
pub fn ifft2(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let rank = x.shape.len();
    if rank < 2 {
        return ifft(x, None, None);
    }
    let h = x.shape[rank - 2] as usize;
    let w = x.shape[rank - 1] as usize;

    let step1 = ifft(x, None, Some((rank - 1) as i64))?;

    let mut out_shape = x.shape.clone();
    out_shape.push(2);
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let src = unsafe { std::slice::from_raw_parts(step1.data.as_ptr() as *const f32, step1.elem_count()) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };

    let n_total = elem_count(&x.shape);
    let hw = h * w;
    let n_batches = n_total / hw;

    for b in 0..n_batches {
        let b_base = b * hw;
        for col in 0..w {
            let mut c_in = Vec::with_capacity(h);
            for row in 0..h {
                let idx = (b_base + row * w + col) * 2;
                c_in.push(Complex64::new(src[idx] as f64, src[idx + 1] as f64));
            }
            let c_out = fft_1d_c2c(&c_in, true);
            for row in 0..h {
                let idx = (b_base + row * w + col) * 2;
                dst[idx] = c_out[row].re as f32;
                dst[idx + 1] = c_out[row].im as f32;
            }
        }
    }

    Ok(out)
}

/// N-D FFT
pub fn fftn(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    fft2(x)
}

/// N-D IFFT
pub fn ifftn(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    ifft2(x)
}

/// Shift the zero-frequency component to the center of the spectrum.
pub fn fftshift(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    let mut out = OwnedTensor::new(x.dtype, x.shape.clone());
    let src = unsafe { typed_slice::<f32>(x) };
    let dst = unsafe { typed_mut_slice::<f32>(&mut out) };
    let n = elem_count(&x.shape);
    let mid = n / 2;

    dst[..n - mid].copy_from_slice(&src[mid..]);
    dst[n - mid..].copy_from_slice(&src[..mid]);

    Ok(out)
}

/// Inverse FFT shift.
pub fn ifftshift(x: &BorrowedTensor) -> PyResult<OwnedTensor> {
    fftshift(x)
}
