//! `bench_decoder_step` — full per-token decode latency (the kernel chain that
//! `decode_and_sample` runs), isolated from Python.
//!
//! Replicates the Qwen2.5-0.5B per-token work at native shapes with synthetic
//! weights: rms_norm → qkv GEMV → o GEMV → gate/up GEMV → down GEMV →
//! lm_head GEMV → top-k sampling. This is the same kernel sequence the Rust
//! decoder executes every token; the bench removes model loading and FFI so
//! kernel cost is measured directly.
//!
//! Run:
//!   cargo bench --no-default-features --features matrixmultiply --bench bench_decoder_step

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use _torchburn::kernels::{fast_rms_norm, gemv_w4a32_grouped, sample_logits};

const GROUP_SIZE: usize = 64;
const HIDDEN: usize = 896;
const INTERMEDIATE: usize = 4864;
const VOCAB: usize = 151936;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn fill_x(rng: &mut Lcg, x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 - 1.0;
    }
}

fn bench_decode_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_step");
    group.sample_size(50); // full step is heavier than a single GEMV

    group.bench_function("qwen_0_5b_per_token", |bb| {
        let mut rng = Lcg(0xA5A5A5A5DEADBEEF);

        // Layer 1 (representative): qkv 896→2688, o 896→896, gate/up 896→4864, down 4864→896
        let mut x = vec![0.0f32; HIDDEN];
        let mut norm_w = vec![1.0f32; HIDDEN];
        let mut norm_out = vec![0.0f32; HIDDEN];

        let make_w = |n: usize, k: usize, rng: &mut Lcg| -> (Vec<u8>, Vec<f32>) {
            let mut w = vec![0u8; n * ((k + 1) / 2)];
            let num_groups = k / GROUP_SIZE;
            let mut scales = vec![0.0f32; n * num_groups];
            for v in w.iter_mut() {
                *v = (rng.next() % 256) as u8;
            }
            for v in scales.iter_mut() {
                *v = (rng.next() as f32 / u64::MAX as f32) * 2.0 + 0.1;
            }
            (w, scales)
        };

        let (qkv_w, qkv_s) = make_w(3 * HIDDEN, HIDDEN, &mut rng);
        let (o_w, o_s) = make_w(HIDDEN, HIDDEN, &mut rng);
        let (gate_w, gate_s) = make_w(INTERMEDIATE, HIDDEN, &mut rng);
        let (up_w, up_s) = make_w(INTERMEDIATE, HIDDEN, &mut rng);
        let (down_w, down_s) = make_w(HIDDEN, INTERMEDIATE, &mut rng);
        let (lm_head_w, lm_head_s) = make_w(VOCAB, HIDDEN, &mut rng);

        let mut qkv_out = vec![0.0f32; 3 * HIDDEN];
        let mut attn_out = vec![0.0f32; HIDDEN];
        let mut o_out = vec![0.0f32; HIDDEN];
        let mut gate_out = vec![0.0f32; INTERMEDIATE];
        let mut up_out = vec![0.0f32; INTERMEDIATE];
        let mut down_out = vec![0.0f32; HIDDEN];
        let mut logits = vec![0.0f32; VOCAB];

        fill_x(&mut rng, &mut x);

        bb.iter(|| unsafe {
            // 1. input rms_norm
            fast_rms_norm(x.as_ptr(), norm_w.as_ptr(), norm_out.as_mut_ptr(), HIDDEN, 1e-6);
            // 2. qkv GEMV
            gemv_w4a32_grouped(
                norm_out.as_ptr(),
                qkv_w.as_ptr(),
                qkv_s.as_ptr(),
                None,
                qkv_out.as_mut_ptr(),
                3 * HIDDEN,
                HIDDEN,
                GROUP_SIZE,
            );
            // 3. o GEMV (attention output projection; attention itself is T=1
            //    and cheap relative to the GEMVs at this size)
            gemv_w4a32_grouped(
                attn_out.as_ptr(),
                o_w.as_ptr(),
                o_s.as_ptr(),
                None,
                o_out.as_mut_ptr(),
                HIDDEN,
                HIDDEN,
                GROUP_SIZE,
            );
            // 4. gate/up GEMVs
            gemv_w4a32_grouped(
                x.as_ptr(),
                gate_w.as_ptr(),
                gate_s.as_ptr(),
                None,
                gate_out.as_mut_ptr(),
                INTERMEDIATE,
                HIDDEN,
                GROUP_SIZE,
            );
            gemv_w4a32_grouped(
                x.as_ptr(),
                up_w.as_ptr(),
                up_s.as_ptr(),
                None,
                up_out.as_mut_ptr(),
                INTERMEDIATE,
                HIDDEN,
                GROUP_SIZE,
            );
            // 5. down GEMV (reads the up projection output, 4864 wide)
            gemv_w4a32_grouped(
                up_out.as_ptr(),
                down_w.as_ptr(),
                down_s.as_ptr(),
                None,
                down_out.as_mut_ptr(),
                HIDDEN,
                INTERMEDIATE,
                GROUP_SIZE,
            );
            // 6. lm_head GEMV → logits
            gemv_w4a32_grouped(
                norm_out.as_ptr(),
                lm_head_w.as_ptr(),
                lm_head_s.as_ptr(),
                None,
                logits.as_mut_ptr(),
                VOCAB,
                HIDDEN,
                GROUP_SIZE,
            );
            // 7. sample
            let tok = sample_logits(&logits, 0.7, 40, 0.9);
            black_box(tok);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_decode_step);
criterion_main!(benches);