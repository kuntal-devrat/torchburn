//! `bench_int4_gemv` — grouped int4 decode GEMV at Qwen2.5-0.5B shapes
//! (hidden 896, heads 14, head_dim 64, intermediate 4864, vocab 151936).
//!
//! This is the kernel that dominates per-token decode. Weights are synthetic
//! (deterministic packed bytes + scales) so the bench is pure kernel cost,
//! isolated from the model loader and Python.
//!
//! Run:
//!   cargo bench --no-default-features --features matrixmultiply --bench bench_int4_gemv

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use _torchburn::kernels::gemv_w4a32_grouped;

const GROUP_SIZE: usize = 64;
const HIDDEN: usize = 896;
const INTERMEDIATE: usize = 4864;
const VOCAB: usize = 151936;

/// Deterministic pseudo-random fill (xorshift) — reproducible across runs.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn bench_qkv(c: &mut Criterion) {
    // qkv_proj: hidden -> 3*hidden
    let n = 3 * HIDDEN;
    let k = HIDDEN;
    let mut group = c.benchmark_group("int4_gemv");
    group.bench_function("qkv_896_to_2688", |bb| {
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let mut x = vec![0.0f32; k];
        let mut w = vec![0u8; n * ((k + 1) / 2)];
        let num_groups = k / GROUP_SIZE;
        let mut scales = vec![0.0f32; n * num_groups];
        let mut out = vec![0.0f32; n];
        for v in x.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        for v in w.iter_mut() {
            *v = (rng.next() % 256) as u8;
        }
        for v in scales.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 + 0.1;
        }
        bb.iter(|| unsafe {
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
            black_box(&out);
        });
    });
    group.finish();
}

fn bench_mlp(c: &mut Criterion) {
    // gate/up: hidden -> intermediate (2x), down: intermediate -> hidden
    let mut group = c.benchmark_group("int4_gemv");
    group.bench_function("gate_896_to_4864", |bb| {
        let n = INTERMEDIATE;
        let k = HIDDEN;
        let mut rng = Lcg(0x123456789ABCDEF);
        let mut x = vec![0.0f32; k];
        let mut w = vec![0u8; n * ((k + 1) / 2)];
        let num_groups = k / GROUP_SIZE;
        let mut scales = vec![0.0f32; n * num_groups];
        let mut out = vec![0.0f32; n];
        for v in x.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        for v in w.iter_mut() {
            *v = (rng.next() % 256) as u8;
        }
        for v in scales.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 + 0.1;
        }
        bb.iter(|| unsafe {
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
            black_box(&out);
        });
    });
    group.bench_function("down_4864_to_896", |bb| {
        let n = HIDDEN;
        let k = INTERMEDIATE;
        let mut rng = Lcg(0xDEADBEEFCAFEF00D);
        let mut x = vec![0.0f32; k];
        let mut w = vec![0u8; n * ((k + 1) / 2)];
        let num_groups = k / GROUP_SIZE;
        let mut scales = vec![0.0f32; n * num_groups];
        let mut out = vec![0.0f32; n];
        for v in x.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        for v in w.iter_mut() {
            *v = (rng.next() % 256) as u8;
        }
        for v in scales.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 + 0.1;
        }
        bb.iter(|| unsafe {
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
            black_box(&out);
        });
    });
    group.finish();
}

fn bench_lm_head(c: &mut Criterion) {
    // lm_head: hidden -> vocab (the memory-bound tail of every token)
    let n = VOCAB;
    let k = HIDDEN;
    let mut group = c.benchmark_group("int4_gemv");
    group.bench_function("lm_head_896_to_151936", |bb| {
        let mut rng = Lcg(0xFEEDFACECAFEBEEF);
        let mut x = vec![0.0f32; k];
        let mut w = vec![0u8; n * ((k + 1) / 2)];
        let num_groups = k / GROUP_SIZE;
        let mut scales = vec![0.0f32; n * num_groups];
        let mut out = vec![0.0f32; n];
        for v in x.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        for v in w.iter_mut() {
            *v = (rng.next() % 256) as u8;
        }
        for v in scales.iter_mut() {
            *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 + 0.1;
        }
        bb.iter(|| unsafe {
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
            black_box(&out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_qkv, bench_mlp, bench_lm_head);
criterion_main!(benches);