//! `bench_wgpu_submit` — GPU command-stream submit + readback latency,
//! no decode work. Measures the fixed per-token overhead the iGPU path pays
//! (command encoding → submit → poll(Wait) → staging map_async readback).
//!
//! Uses the real wgpu backend's tiny GEMV path (`wgpu_gemv_w4a32` at a
//! 1×64 shape) so the measured cost is dominated by submit/readback, not
//! shader execution.
//!
//! Requires the `burn-wgpu` feature and a GPU:
//!   cargo bench --features burn-wgpu --bench bench_wgpu_submit
//! On GPU-less machines the bench reports a placeholder instead of failing.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "burn-wgpu")]
fn bench_wgpu_submit(c: &mut Criterion) {
    let mut group = c.benchmark_group("wgpu_submit");
    group.sample_size(50);

    if !_torchburn::wgpu::backend::gpu_available() {
        group.bench_function("no_gpu_placeholder", |bb| bb.iter(|| 1u32));
        group.finish();
        return;
    }

    let x = vec![0.5f32; 64];
    let w = vec![0u8; 32]; // k=64 -> 32 packed bytes
    let scales = vec![1.0f32; 1];
    let mut out = vec![0.0f32; 1];

    group.bench_function("submit_readback_1x64", |bb| {
        bb.iter(|| {
            let _ = black_box(_torchburn::wgpu::backend::wgpu_gemv_w4a32(
                &x, &w, &scales, &mut out, 1, 64, 64,
            ));
        });
    });
    group.finish();
}

#[cfg(not(feature = "burn-wgpu"))]
fn bench_wgpu_submit(c: &mut Criterion) {
    let mut group = c.benchmark_group("wgpu_submit");
    group.bench_function("burn_wgpu_feature_off", |bb| bb.iter(|| 1u32));
    group.finish();
}

criterion_group!(benches, bench_wgpu_submit);
criterion_main!(benches);