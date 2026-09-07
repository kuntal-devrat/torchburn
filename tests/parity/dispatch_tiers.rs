//! Dispatch-tier parity (Phase 0.3, feature `dispatch-test`).
//!
//! Forces each CPU feature tier and asserts:
//!   1. every exact-dequant tier (scalar / AVX2 / AVX-512) produces allclose
//!      outputs on identical inputs,
//!   2. the AVX-512 VNNI tier — which internally runs W4A8 (activations
//!      quantized to u8, see `quantize_activation_to_u8`) — stays within its
//!      documented approximation budget vs the exact tiers,
//!   3. on x86-64, GEMV throughput is monotonic non-decreasing across tiers.
//!
//! Run: cargo test --no-default-features --features matrixmultiply,dispatch-test

#![cfg(feature = "dispatch-test")]

use std::sync::Mutex;
use _torchburn::dispatch::{clear_override, cpu_features, force_tier, CpuTier};
#[cfg(all(target_arch = "x86_64", not(debug_assertions)))]
use std::time::Instant;

/// Both dispatch tests mutate the process-global `OVERRIDE` atomic that
/// `force_tier`/`cpu_features` share with the kernels. The Rust test harness
/// runs tests in parallel, so without a lock the two tests clobber each
/// other's forced tier mid-measurement (observed: forced-Scalar timing
/// suddenly matching VNNI). Serialize them.
static TIER_LOCK: Mutex<()> = Mutex::new(());
use _torchburn::kernels::gemv_w4a32_grouped;

const GROUP_SIZE: usize = 64;

fn run_gemv(x: &[f32], w: &[u8], scales: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        gemv_w4a32_grouped(
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            None,
            out.as_mut_ptr(),
            n,
            k,
            GROUP_SIZE,
        );
    }
    out
}

fn host_supports(tier: CpuTier) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match tier {
            CpuTier::Scalar => true,
            CpuTier::Avx2 => std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"),
            CpuTier::Avx512 => std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw"),
            CpuTier::Avx512Vnni => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = tier;
        false
    }
}

fn all_tiers() -> [CpuTier; 4] {
    [
        CpuTier::Scalar,
        CpuTier::Avx2,
        CpuTier::Avx512,
        CpuTier::Avx512Vnni,
    ]
}

/// Robust relative error: max-abs diff / max-abs expected. Immune to
/// near-zero rows that a per-row relative metric would amplify.
fn rel_err(actual: &[f32], expected: &[f32]) -> f32 {
    let denom = expected.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
    actual
        .iter()
        .zip(expected)
        .fold(0.0f32, |a, (x, e)| a.max((x - e).abs()))
        / denom
}

#[test]
fn exact_tiers_are_allclose_vnni_within_w4a8_budget() {
    let _guard = TIER_LOCK.lock().unwrap();
    let (n, k) = (256usize, 896usize);
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.13).sin() * 2.0).collect();
    let mut w = vec![0u8; n * ((k + 1) / 2)];
    let num_groups = k / GROUP_SIZE;
    let mut scales = vec![0.0f32; n * num_groups];
    for (i, v) in w.iter_mut().enumerate() {
        *v = ((i * 2654435761) % 256) as u8;
    }
    for (i, v) in scales.iter_mut().enumerate() {
        *v = 0.5 + ((i * 31) % 97) as f32 * 0.01;
    }

    // Scalar tier output is the exact reference (dequant-dot in f32).
    force_tier(CpuTier::Scalar);
    let scalar = run_gemv(&x, &w, &scales, n, k);

    let mut exact_tiers_ok = true;
    let mut vnni_err = 0.0f32;
    for tier in all_tiers() {
        if !host_supports(tier) {
            continue;
        }
        force_tier(tier);
        let out = run_gemv(&x, &w, &scales, n, k);
        let err = rel_err(&out, &scalar);
        println!("tier {tier:?} vs scalar: rel err {err:.2e}");
        if tier == CpuTier::Avx512Vnni {
            vnni_err = err;
        } else if err > 1e-3 {
            exact_tiers_ok = false;
        }
    }

    assert!(exact_tiers_ok, "exact-dequant tiers diverged from scalar (budget 1e-3)");
    // The VNNI tier runs W4A8 (activation quantization), a documented
    // approximation — assert it stays within a sane bound relative to W4A32.
    assert!(
        vnni_err < 0.1,
        "VNNI/W4A8 tier rel err {vnni_err:.2e} exceeded 0.1 budget"
    );
    clear_override();
}

/// Performance must be asserted on a release build (`cargo test --release`):
/// in debug builds `#[target_feature]` code is not representative and the
/// measurement is noise. The parity test above always runs; this one is the
/// monotonicity gate for the release profile.
#[test]
fn forced_tiers_performance_is_monotonic() {
    let _guard = TIER_LOCK.lock().unwrap();
    #[cfg(all(target_arch = "x86_64", not(debug_assertions)))]
    {
        // n=4096 (4x Qwen's 896 rows): at n=1024 per-call time is ~30µs and
        // AVX-512 frequency downclocking + scheduler noise swamp the signal
        // (measured tiers shuffled between runs). n=4096 separates the tiers
        // cleanly (~10x scalar→VNNI) and is still Qwen-shaped (k=896).
        let (n, k) = (4096usize, 896usize);
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.11).sin() * 2.0).collect();
        let mut w = vec![0u8; n * ((k + 1) / 2)];
        let num_groups = k / GROUP_SIZE;
        let mut scales = vec![0.0f32; n * num_groups];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 40503) % 256) as u8;
        }
        for (i, v) in scales.iter_mut().enumerate() {
            *v = 0.5 + ((i * 17) % 91) as f32 * 0.01;
        }

        // Best-of-3 rounds of `iters` calls per tier: min-of-rounds is robust
        // against CPU frequency scaling and thread scheduling on laptops, and
        // represents the achievable per-call throughput.
        let rounds = 3;
        let iters = 100;
        let mut times: Vec<(CpuTier, f64)> = Vec::new();
        for tier in all_tiers() {
            if !host_supports(tier) {
                continue;
            }
            force_tier(tier);
            assert_eq!(
                cpu_features().tier(),
                tier,
                "forced tier {tier:?} did not take effect"
            );
            // warmup
            let _ = run_gemv(&x, &w, &scales, n, k);
            let mut best = f64::INFINITY;
            for _ in 0..rounds {
                let t0 = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(run_gemv(&x, &w, &scales, n, k));
                }
                best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            times.push((tier, best));
        }
        clear_override();

        assert!(times.len() >= 2, "need at least two runnable tiers");
        println!("tier best-of-{rounds} timings: {times:?}");
        for pair in times.windows(2) {
            // Monotonic: a higher tier must not be meaningfully slower than a
            // lower one. Neighboring tiers (e.g. AVX2 vs AVX-512 at 30µs) can
            // land within noise, so allow 10% jitter.
            assert!(
                pair[1].1 <= pair[0].1 * 1.10,
                "tier {:?} ({:.2e}s) slower than {:?} ({:.2e}s)",
                pair[1].0,
                pair[1].1,
                pair[0].0,
                pair[0].1
            );
        }
    }
    #[cfg(all(not(target_arch = "x86_64"), not(debug_assertions)))]
    {
        let _ = cpu_features();
    }
    #[cfg(debug_assertions)]
    {
        // No-op in debug builds: run `cargo test --release` for the perf gate.
        let _ = cpu_features();
    }
}