//! `bench_gemm` — sgemm/dgemm throughput at (64, 256, 1024, 4096) square and
//! skinny-M shapes.
//!
//! Measures the crate's real GEMM kernels (`gemm_f32_trans_b_into_accum` /
//! `gemm_f64_trans_b_into_accum`): A is m×k row-major, B is n×k row-major
//! ("trans_b" layout), C is m×n row-major with beta=1 accumulation.
//!
//! Run:
//!   cargo bench --no-default-features --features matrixmultiply --bench bench_gemm

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use _torchburn::kernels::{gemm_f32_trans_b_into_accum, gemm_f64_trans_b_into_accum};

fn shapes() -> Vec<(usize, usize, usize)> {
    // (m, k, n): squares plus skinny-M (the decode GEMV regime)
    let mut out = Vec::new();
    for s in [64usize, 256, 1024, 4096] {
        out.push((s, s, s));
    }
    for m in [1usize, 8, 32] {
        out.push((m, 4096, 4096)); // skinny-M f32 decode regime
        out.push((m, 896, 896)); // Qwen hidden
    }
    out
}

fn bench_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f32");
    for (m, k, n) in shapes() {
        let mut a = vec![0.5f32; m * k];
        let mut b = vec![0.25f32; n * k];
        let mut out = vec![0.0f32; m * n];
        group.bench_function(format!("m{m}_k{k}_n{n}"), |bb| {
            bb.iter(|| unsafe {
                gemm_f32_trans_b_into_accum(
                    a.as_mut_ptr(),
                    m,
                    k,
                    b.as_mut_ptr(),
                    k,
                    out.as_mut_ptr(),
                    n,
                );
                black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f64");
    for (m, k, n) in shapes() {
        let mut a = vec![0.5f64; m * k];
        let mut b = vec![0.25f64; n * k];
        let mut out = vec![0.0f64; m * n];
        group.bench_function(format!("m{m}_k{k}_n{n}"), |bb| {
            bb.iter(|| unsafe {
                gemm_f64_trans_b_into_accum(
                    a.as_mut_ptr(),
                    m,
                    k,
                    b.as_mut_ptr(),
                    k,
                    out.as_mut_ptr(),
                    n,
                );
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gemm_f32, bench_gemm_f64);
criterion_main!(benches);