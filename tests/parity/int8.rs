//! int8 parity: `dot_f32_i8` (the W8A32 GEMV inner kernel) vs naive
//! reference, plus the recorded quantize→dequant quality budget.

use _torchburn::kernels::dot_f32_i8;

fn naive_dot_i8(x: &[f32], w: &[i8]) -> f32 {
    x.iter().zip(w).map(|(xv, wv)| xv * (*wv as f32)).sum()
}

/// Robust relative error: max-abs diff / max-abs expected.
fn max_rel_err(actual: &[f32], expected: &[f32]) -> f32 {
    let denom = expected
        .iter()
        .fold(0.0f32, |a, &v| a.max(v.abs()))
        .max(1e-6);
    actual
        .iter()
        .zip(expected)
        .fold(0.0f32, |a, (x, e)| a.max((x - e).abs()))
        / denom
}

#[test]
fn dot_f32_i8_matches_reference() {
    let x: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.041).cos() * 2.0).collect();
    let w: Vec<i8> = (0..2048)
        .map(|i| (((i * 31) % 251) as i16 - 125) as i8)
        .collect();
    let got = unsafe { dot_f32_i8(x.as_ptr(), w.as_ptr(), x.len()) };
    let expected = naive_dot_i8(&x, &w);
    let rel = (got - expected).abs() / expected.abs().max(1e-6);
    assert!(
        rel < 1e-3,
        "dot_f32_i8 rel err {rel:.2e} (got {got}, want {expected})"
    );
}

/// Records the int8-vs-f32 quantization budget on synthetic weights (the
/// kernel itself is exact; the budget is the quantizer's rounding error).
#[test]
fn quantized_int8_quality_budget_recorded() {
    let (n, k) = (32usize, 512usize);
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.051).cos() * 1.1).collect();
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.19).sin() * 1.8).collect();

    // Per-row int8 GEMV: out[j] = dot_f32_i8(x, q_j) * scale_j
    let mut out = vec![0.0f32; n];
    let mut f32_out = vec![0.0f32; n];
    for row in 0..n {
        let wrow = &w[row * k..row * k + k];
        let max_abs = wrow.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1.0 };
        let mut q: Vec<i8> = wrow
            .iter()
            .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        out[row] = unsafe { dot_f32_i8(x.as_ptr(), q.as_mut_ptr(), k) } * scale;
        f32_out[row] = x.iter().zip(wrow).map(|(xv, wv)| xv * wv).sum();
    }
    let err = max_rel_err(&out, &f32_out);
    println!("int8 vs f32 max-abs rel err on uniform weights: {err:.2e}");
    // int8 rounding is ~16x finer than int4; document, don't over-constrain.
    assert!(
        err < 0.05,
        "int8 vs f32 synthetic rel err {err:.2e} (budget 0.05)"
    );
}
