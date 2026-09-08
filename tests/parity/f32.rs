//! f32 kernel parity: GEMM (trans-B accumulate) and dot vs naive references.

use _torchburn::kernels::{dot_f32_f32, gemm_f32_trans_b_into_accum, gemm_f64_trans_b_into_accum};

fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn naive_gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // a: m×k row-major, b: n×k row-major (trans_b), out: m×n row-major
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn naive_gemm_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn max_rel_err(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| {
            let denom = e.abs().max(1e-6);
            (a - e).abs() / denom
        })
        .fold(0.0f32, f32::max)
}

#[test]
fn dot_f32_matches_reference() {
    let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.03).cos()).collect();
    let got = unsafe { dot_f32_f32(a.as_ptr(), b.as_ptr(), a.len()) };
    let expected = naive_dot(&a, &b);
    let rel = (got - expected).abs() / expected.abs().max(1e-6);
    assert!(
        rel < 1e-3,
        "dot_f32_f32 rel err {rel:.2e} (got {got}, want {expected})"
    );
}

#[test]
fn gemm_f32_matches_reference() {
    let (m, k, n) = (8usize, 64, 16); // skinny-M: also exercises the m<=8 fast path
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).sin() * 2.0).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos() * 2.0).collect();
    let mut out = vec![0.0f32; m * n];
    gemm_f32_trans_b_into_accum(a.as_ptr(), m, k, b.as_ptr(), k, out.as_mut_ptr(), n);
    let expected = naive_gemm_f32(&a, &b, m, k, n);
    let err = max_rel_err(&out, &expected);
    assert!(err < 1e-3, "gemm_f32 rel err {err:.2e}");
}

#[test]
fn gemm_f32_square_matches_reference() {
    let (m, k, n) = (64usize, 64, 64);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.019).cos()).collect();
    let mut out = vec![0.0f32; m * n];
    gemm_f32_trans_b_into_accum(a.as_ptr(), m, k, b.as_ptr(), k, out.as_mut_ptr(), n);
    let expected = naive_gemm_f32(&a, &b, m, k, n);
    let err = max_rel_err(&out, &expected);
    assert!(err < 1e-3, "gemm_f32 square rel err {err:.2e}");
}

#[test]
fn gemm_f64_matches_reference() {
    let (m, k, n) = (8usize, 64, 16);
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.07).sin() * 2.0).collect();
    let b: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.11).cos() * 2.0).collect();
    let mut out = vec![0.0f64; m * n];
    gemm_f64_trans_b_into_accum(a.as_ptr(), m, k, b.as_ptr(), k, out.as_mut_ptr(), n);
    let expected = naive_gemm_f64(&a, &b, m, k, n);
    let err = expected
        .iter()
        .zip(&out)
        .map(|(e, a)| (a - e).abs() / e.abs().max(1e-9))
        .fold(0.0f64, f64::max);
    assert!(err < 1e-9, "gemm_f64 rel err {err:.2e}");
}
