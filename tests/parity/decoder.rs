//! Decoder component parity: `fast_rms_norm` and `sample_logits` vs naive
//! references. Full end-to-end decode parity lives in the Python suite
//! (`tests/test_parity.py`) and `tests/test_wgpu_decoder_parity.py`.

use _torchburn::kernels::{fast_rms_norm, sample_logits};

fn naive_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(xv, wv)| xv / rms * wv).collect()
}

#[test]
fn rms_norm_matches_reference() {
    let n = 896usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin() * 3.0).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut out = vec![0.0f32; n];
    unsafe {
        fast_rms_norm(x.as_ptr(), w.as_ptr(), out.as_mut_ptr(), n, 1e-6);
    }
    let expected = naive_rms_norm(&x, &w, 1e-6);
    let max_err = out
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1e-6))
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "rms_norm rel err {max_err:.2e}");
}

#[test]
fn greedy_sample_is_argmax() {
    let mut logits: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.013).sin()).collect();
    logits[1234] = 100.0; // clear winner
    let got = sample_logits(&logits, 0.0, 1, 1.0); // temperature <= 0 -> greedy
    assert_eq!(got, 1234);
    // top_k == 1 forces greedy too
    let got2 = sample_logits(&logits, 0.7, 1, 1.0);
    assert_eq!(got2, 1234);
}

#[test]
fn sample_within_vocab_and_nonzero_mass() {
    // Sampling with real params must stay in-bounds and (for top_p=1) pick
    // from the top-k with positive probability.
    let mut logits: Vec<f32> = (0..151936)
        .map(|i| (i as f32 * 0.007).sin() * 5.0)
        .collect();
    logits[7] = 40.0;
    logits[88] = 39.0;
    let mut seen: Vec<usize> = Vec::new();
    for _ in 0..64 {
        let tok = sample_logits(&logits, 0.7, 40, 1.0);
        assert!(tok < logits.len());
        seen.push(tok);
    }
    assert!(
        seen.contains(&7) || seen.contains(&88),
        "top tokens never sampled"
    );
}
