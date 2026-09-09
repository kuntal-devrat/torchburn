//! Native Rust (CPU) LLM decoder (`RustQwenDecoder`) — the AVX-512 / rayon
//! baseline that the wgpu decoder is validated against.
//!
//! Moved out of `quantization.rs` so that file owns one concern (low-bit GEMM
//! kernels + their Python-facing quantized ops).  The decoder-specific kernels
//! (`gemv_w4a32_grouped`, `fast_rms_norm`, SwiGLU neurons, ...) still live in
//! [`crate::quantization`] and are imported here.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use rayon::prelude::*;

use crate::dlpack::{self, unsupported, BorrowedTensor, DType, OwnedTensor};
use crate::quantization::{
    dot_f32_f32, fast_rms_norm, fast_vector_add, fast_vector_fma, gemv_w4a32_grouped,
    gemv_w4a32_grouped_v2, pack_rows_w4a32_group64_v1_to_v2, quantize_activation_to_u8,
    swiglu_neuron_w4a32_dot_dispatch,
};

thread_local! {
    static THREAD_SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[inline(always)]
fn rope_apply_head(head: &mut [f32], cos_p: &[f32], sin_p: &[f32], half_dim: usize) {
    let num_vec = half_dim / 4;
    for c in 0..num_vec {
        let i = c * 4;
        let mut x1_arr = [0.0f32; 4];
        let mut x2_arr = [0.0f32; 4];
        let mut c1_arr = [0.0f32; 4];
        let mut s1_arr = [0.0f32; 4];
        let mut c2_arr = [0.0f32; 4];
        let mut s2_arr = [0.0f32; 4];
        x1_arr.copy_from_slice(&head[i..i + 4]);
        x2_arr.copy_from_slice(&head[i + half_dim..i + half_dim + 4]);
        c1_arr.copy_from_slice(&cos_p[i..i + 4]);
        s1_arr.copy_from_slice(&sin_p[i..i + 4]);
        c2_arr.copy_from_slice(&cos_p[i + half_dim..i + half_dim + 4]);
        s2_arr.copy_from_slice(&sin_p[i + half_dim..i + half_dim + 4]);

        let x1 = wide::f32x4::from(x1_arr);
        let x2 = wide::f32x4::from(x2_arr);
        let c1 = wide::f32x4::from(c1_arr);
        let s1 = wide::f32x4::from(s1_arr);
        let c2 = wide::f32x4::from(c2_arr);
        let s2 = wide::f32x4::from(s2_arr);

        let out1 = (x1 * c1) - (x2 * s1);
        let out2 = (x2 * c2) + (x1 * s2);

        head[i..i + 4].copy_from_slice(&out1.to_array());
        head[i + half_dim..i + half_dim + 4].copy_from_slice(&out2.to_array());
    }
    for i in (num_vec * 4)..half_dim {
        let q1 = head[i];
        let q2 = head[i + half_dim];
        let c1 = cos_p[i];
        let s1 = sin_p[i];
        let c2 = cos_p[i + half_dim];
        let s2 = sin_p[i + half_dim];
        head[i] = q1 * c1 - q2 * s1;
        head[i + half_dim] = q2 * c2 + q1 * s2;
    }
}

pub(crate) unsafe fn typed_slice<T>(t: &BorrowedTensor) -> &[T] {
    std::slice::from_raw_parts(t.data as *const T, t.buffer_len())
}

unsafe fn typed_mut_slice<T>(t: &mut OwnedTensor) -> &mut [T] {
    std::slice::from_raw_parts_mut(t.data.as_mut_ptr() as *mut T, t.elem_count())
}

pub struct RustLayerData {
    pub input_norm_w: Vec<f32>,
    pub qkv_w: Vec<u8>,
    pub qkv_s: Vec<f32>,
    /// v2 interleaved layout (Phase 1.2), empty when v2 packing is disabled.
    pub qkv_w2: Vec<u8>,
    pub qkv_b: Option<Vec<f32>>,
    pub o_w: Vec<u8>,
    pub o_s: Vec<f32>,
    pub o_w2: Vec<u8>,
    pub o_b: Option<Vec<f32>>,
    pub post_norm_w: Vec<f32>,
    pub gate_w: Vec<u8>,
    pub gate_s: Vec<f32>,
    pub gate_b: Option<Vec<f32>>,
    pub up_w: Vec<u8>,
    pub up_s: Vec<f32>,
    pub up_b: Option<Vec<f32>>,
    pub down_w: Vec<u8>,
    pub down_s: Vec<f32>,
    pub down_w2: Vec<u8>,
    pub down_b: Option<Vec<f32>>,
}

#[pyclass]
pub struct RustQwenDecoder {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub group_size: usize,
    /// True when every projection was v2-packed at construction (group=64
    /// and all K dims divisible by 64); routes GEMVs to the fused kernel.
    pub use_v2: bool,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<RustLayerData>,
    pub final_norm_w: Vec<f32>,
    pub lm_head_w: Vec<u8>,
    pub lm_head_s: Vec<f32>,
    pub lm_head_w2: Vec<u8>,
    pub k_caches: Vec<Vec<f32>>,
    pub v_caches: Vec<Vec<f32>>,
    /// Highest KV position written since last reset (for fast prefix-only clear).
    pub kv_used: usize,
    pub cos_table: Vec<f32>,
    pub sin_table: Vec<f32>,
    // Preallocated scratch buffers
    x_buf: Vec<f32>,
    normed_buf: Vec<f32>,
    qkv_buf: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    h_buf: Vec<f32>,
    down_out: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
}

#[pymethods]
impl RustQwenDecoder {
    #[new]
    #[pyo3(signature = (
        embed_tokens,
        layers_data,
        final_norm_w,
        lm_head_w,
        lm_head_s,
        num_layers,
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        group_size=64,
        rms_norm_eps=1e-6,
        max_seq_len=2048,
        rope_theta=1000000.0
    ))]
    pub fn new(
        py: Python<'_>,
        embed_tokens: &Bound<'_, PyCapsule>,
        layers_data: Vec<Vec<Bound<'_, PyCapsule>>>,
        final_norm_w: &Bound<'_, PyCapsule>,
        lm_head_w: &Bound<'_, PyCapsule>,
        lm_head_s: &Bound<'_, PyCapsule>,
        num_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        group_size: usize,
        rms_norm_eps: f64,
        max_seq_len: usize,
        rope_theta: f64,
    ) -> PyResult<Self> {
        let _ = py;
        let emb_view = unsafe { dlpack::BorrowedTensor::from_capsule(embed_tokens)? };
        let vocab_size = emb_view.shape[0] as usize;
        let emb_slice = unsafe { typed_slice::<f32>(&emb_view) };
        let embed_tokens_vec = emb_slice.to_vec();

        let fnorm_view = unsafe { dlpack::BorrowedTensor::from_capsule(final_norm_w)? };
        let final_norm_vec = unsafe { typed_slice::<f32>(&fnorm_view) }.to_vec();

        let lm_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(lm_head_w)? };
        let lm_head_w_vec = unsafe { typed_slice::<u8>(&lm_w_view) }.to_vec();

        let lm_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(lm_head_s)? };
        let lm_head_s_vec = unsafe { typed_slice::<f32>(&lm_s_view) }.to_vec();

        let mut rust_layers = Vec::with_capacity(num_layers);
        for l_caps in layers_data {
            if l_caps.len() < 12 {
                return Err(unsupported("Each layer requires 12 capsules: input_norm, qkv_w, qkv_s, o_w, o_s, post_norm, gate_w, gate_s, up_w, up_s, down_w, down_s"));
            }
            let in_norm =
                unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[0])?) }
                    .to_vec();
            let qkv_w =
                unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[1])?) }
                    .to_vec();
            let qkv_s =
                unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[2])?) }
                    .to_vec();
            let o_w =
                unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[3])?) }
                    .to_vec();
            let o_s =
                unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[4])?) }
                    .to_vec();
            let post_norm =
                unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[5])?) }
                    .to_vec();
            let gate_w =
                unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[6])?) }
                    .to_vec();
            let gate_s =
                unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[7])?) }
                    .to_vec();
            let up_w =
                unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[8])?) }
                    .to_vec();
            let up_s =
                unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[9])?) }
                    .to_vec();
            let down_w =
                unsafe { typed_slice::<u8>(&dlpack::BorrowedTensor::from_capsule(&l_caps[10])?) }
                    .to_vec();
            let down_s =
                unsafe { typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[11])?) }
                    .to_vec();
            let qkv_b = if l_caps.len() >= 13 {
                Some(
                    unsafe {
                        typed_slice::<f32>(&dlpack::BorrowedTensor::from_capsule(&l_caps[12])?)
                    }
                    .to_vec(),
                )
            } else {
                None
            };

            rust_layers.push(RustLayerData {
                input_norm_w: in_norm,
                qkv_w,
                qkv_s,
                qkv_w2: Vec::new(),
                qkv_b,
                o_w,
                o_s,
                o_w2: Vec::new(),
                o_b: None,
                post_norm_w: post_norm,
                gate_w,
                gate_s,
                gate_b: None,
                up_w,
                up_s,
                up_b: None,
                down_w,
                down_s,
                down_w2: Vec::new(),
                down_b: None,
            });
        }

        // Precompute RoPE tables
        let half_dim = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * head_dim];
        let mut sin_table = vec![0.0f32; max_seq_len * head_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0f32 / (rope_theta as f32).powf((2.0 * i as f32) / (head_dim as f32));
                let val = (pos as f32) * freq;
                let c = val.cos();
                let s = val.sin();
                cos_table[pos * head_dim + i] = c;
                cos_table[pos * head_dim + i + half_dim] = c;
                sin_table[pos * head_dim + i] = s;
                sin_table[pos * head_dim + i + half_dim] = s;
            }
        }

        // Allocate static KV caches for all layers
        let kv_cache_len = num_kv_heads * max_seq_len * head_dim;
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(vec![0.0f32; kv_cache_len]);
            v_caches.push(vec![0.0f32; kv_cache_len]);
        }

        let total_qkv = (num_heads + 2 * num_kv_heads) * head_dim;

        // Phase 1.2: repack the v1 weights into the v2 interleaved layout so
        // every projection GEMV (QKV, O, down, lm_head) runs the fused blocked
        // kernel off one sequential 34-byte block stream per group. One-time
        // transform at load; v1 buffers are kept for the fallback path.
        let use_v2 = group_size == 64
            && hidden_size % 64 == 0
            && total_qkv % 64 == 0
            && num_heads * head_dim % 64 == 0
            && intermediate_size % 64 == 0
            && vocab_size % 64 == 0;
        let lm_head_w2_vec = if use_v2 {
            pack_rows_w4a32_group64_v1_to_v2(&lm_head_w_vec, &lm_head_s_vec, vocab_size)
        } else {
            Vec::new()
        };
        if use_v2 {
            for layer in rust_layers.iter_mut() {
                layer.qkv_w2 =
                    pack_rows_w4a32_group64_v1_to_v2(&layer.qkv_w, &layer.qkv_s, total_qkv);
                layer.o_w2 =
                    pack_rows_w4a32_group64_v1_to_v2(&layer.o_w, &layer.o_s, num_heads * head_dim);
                layer.down_w2 = pack_rows_w4a32_group64_v1_to_v2(
                    &layer.down_w,
                    &layer.down_s,
                    intermediate_size,
                );
            }
        }

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            group_size,
            use_v2,
            rms_norm_eps: rms_norm_eps as f32,
            max_seq_len,
            embed_tokens: embed_tokens_vec,
            layers: rust_layers,
            final_norm_w: final_norm_vec,
            lm_head_w: lm_head_w_vec,
            lm_head_s: lm_head_s_vec,
            lm_head_w2: lm_head_w2_vec,
            k_caches,
            v_caches,
            kv_used: 0,
            cos_table,
            sin_table,
            x_buf: vec![0.0f32; hidden_size],
            normed_buf: vec![0.0f32; hidden_size],
            qkv_buf: vec![0.0f32; total_qkv],
            attn_out: vec![0.0f32; num_heads * head_dim],
            o_out: vec![0.0f32; hidden_size],
            h_buf: vec![0.0f32; intermediate_size],
            down_out: vec![0.0f32; hidden_size],
            scores: vec![0.0f32; max_seq_len],
            logits: vec![0.0f32; vocab_size],
        })
    }

    /// Batched prompt prefill in a single FFI call: runs `step_internal`
    /// for each prompt token at successive offsets. Replaces per-token Python
    /// round-trips (each paying DLPack + GIL + dispatch). Still GEMV-bound
    /// (one GEMV per token); a future GEMM prefill can reuse the batched
    /// `fused_swiglu_mlp_batched` pattern for QKV/O.
    pub fn prefill_tokens(&mut self, tokens: Vec<usize>, start_offset: usize) -> PyResult<usize> {
        if tokens.is_empty() {
            return Ok(start_offset);
        }
        if start_offset + tokens.len() > self.max_seq_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "prefill {} tokens at offset {} exceeds max_seq_len {}",
                tokens.len(),
                start_offset,
                self.max_seq_len
            )));
        }
        for (i, &tok) in tokens.iter().enumerate() {
            self.step_internal(tok, start_offset + i);
        }
        Ok(start_offset + tokens.len())
    }

    pub fn reset_kv_cache(&mut self) {
        // Fast prefix-only clear: only zero positions actually written
        // (kv_used tracks max offset+1). Full 200MB fill only on first/long runs.
        let used = self.kv_used.min(self.max_seq_len);
        if used == 0 {
            return;
        }
        if used >= self.max_seq_len {
            use rayon::prelude::*;
            self.k_caches.par_iter_mut().for_each(|kc| kc.fill(0.0));
            self.v_caches.par_iter_mut().for_each(|vc| vc.fill(0.0));
        } else {
            let hd = self.head_dim;
            let stride = self.max_seq_len * hd;
            for kc in &mut self.k_caches {
                for kv_h in 0..self.num_kv_heads {
                    let base = kv_h * stride;
                    kc[base..base + used * hd].fill(0.0);
                }
            }
            for vc in &mut self.v_caches {
                for kv_h in 0..self.num_kv_heads {
                    let base = kv_h * stride;
                    vc[base..base + used * hd].fill(0.0);
                }
            }
        }
        self.kv_used = 0;
    }

    pub fn copy_kv_cache_from_tensors(
        &mut self,
        k_tensors: Vec<Bound<'_, PyCapsule>>,
        v_tensors: Vec<Bound<'_, PyCapsule>>,
        seq_len: usize,
    ) -> PyResult<()> {
        let dst_head_stride = self.max_seq_len * self.head_dim;
        for l in 0..self.num_layers.min(k_tensors.len()) {
            let k_view = unsafe { dlpack::BorrowedTensor::from_capsule(&k_tensors[l])? };
            let v_view = unsafe { dlpack::BorrowedTensor::from_capsule(&v_tensors[l])? };
            let k_slice = unsafe { typed_slice::<f32>(&k_view) };
            let v_slice = unsafe { typed_slice::<f32>(&v_view) };

            let k_shape = &k_view.shape;
            let src_max_len = if k_shape.len() >= 2 {
                k_shape[k_shape.len() - 2] as usize
            } else {
                seq_len
            };
            let src_head_stride = src_max_len * self.head_dim;

            let dst_k = &mut self.k_caches[l];
            let dst_v = &mut self.v_caches[l];

            for kv_h in 0..self.num_kv_heads {
                for t in 0..seq_len {
                    let src_offset = kv_h * src_head_stride + t * self.head_dim;
                    let dst_offset = kv_h * dst_head_stride + t * self.head_dim;
                    if src_offset + self.head_dim > k_slice.len()
                        || dst_offset + self.head_dim > dst_k.len()
                    {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "KV cache copy out of bounds",
                        ));
                    }
                    dst_k[dst_offset..dst_offset + self.head_dim]
                        .copy_from_slice(&k_slice[src_offset..src_offset + self.head_dim]);
                    dst_v[dst_offset..dst_offset + self.head_dim]
                        .copy_from_slice(&v_slice[src_offset..src_offset + self.head_dim]);
                }
            }
        }
        self.kv_used = self.kv_used.max(seq_len);
        Ok(())
    }

    /// Current KV length (max offset written + 1). Used by Python prefix-reuse
    /// to skip re-prefilling resident prefixes across multi-turn chat.
    pub fn kv_len(&self) -> usize {
        self.kv_used
    }

    /// `SpeculativeDecoder` adapter: single-token step returning logits vec.
    /// Matches the `step(token_id, offset) -> list[float]` protocol.
    pub fn step(&mut self, token_id: usize, offset: usize) -> Vec<f32> {
        self.step_internal(token_id, offset);
        self.logits.clone()
    }

    /// `SpeculativeDecoder` adapter: prefill a prompt, return last-token logits.
    /// Resets nothing; caller must `reset_kv_cache()` first for a new sequence.
    pub fn prefill(&mut self, tokens: Vec<usize>) -> PyResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty prefill"));
        }
        if tokens.len() > self.max_seq_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "prefill exceeds max_seq_len",
            ));
        }
        for (i, &tok) in tokens.iter().enumerate() {
            self.step_internal(tok, i);
        }
        Ok(self.logits.clone())
    }

    /// Native speculative batched verify: run `tokens` at successive offsets
    /// in ONE Rust call, returning per-position logits (K+1 rows).
    /// Replaces K+1 per-token Python `step()` FFI round-trips with a single
    /// crossing. Still GEMV-bound per position (true batched GEMM is future),
    /// but eliminates DLPack/GIL/JSON overhead per token (~30-50% verify win).
    pub fn verify_tokens(
        &mut self,
        start_offset: usize,
        tokens: Vec<usize>,
    ) -> PyResult<Vec<Vec<f32>>> {
        if start_offset + tokens.len() > self.max_seq_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "verify exceeds max_seq_len",
            ));
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        for (i, &tok) in tokens.iter().enumerate() {
            self.step_internal(tok, start_offset + i);
            out.push(self.logits.clone());
        }
        Ok(out)
    }

    /// Native end-to-end speculative step: verify `draft_tokens` at `offset`
    /// with the target model and apply Leviathan acceptance using `draft_probs`
    /// (flattened K×V row-major). Returns (accepted_count, bonus_token).
    /// Keeps RNG on the Rust side for determinism with `generate_loop`.
    #[pyo3(signature = (offset, draft_tokens, draft_probs, temperature=1.0, top_p=1.0))]
    pub fn speculative_accept(
        &mut self,
        offset: usize,
        draft_tokens: Vec<usize>,
        draft_probs: Vec<f32>,
        temperature: f32,
        top_p: f32,
    ) -> PyResult<(usize, usize)> {
        let k = draft_tokens.len();
        if k == 0 {
            return Ok((0, self.logits.len().saturating_sub(1)));
        }
        let v = self.logits.len();
        if draft_probs.len() != k * v {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "draft_probs must be K×V row-major",
            ));
        }
        // Verify all K positions, capturing target logits per position
        let mut target_logits: Vec<Vec<f32>> = Vec::with_capacity(k + 1);
        for (i, &tok) in draft_tokens.iter().enumerate() {
            self.step_internal(tok, offset + i);
            target_logits.push(self.logits.clone());
        }
        // Final position after last draft token (bonus candidate slot)
        // Note: step_internal for bonus is deferred to caller to avoid
        // double-advancing KV on reject paths; sample bonus from last logits here.
        let mut accepted = 0usize;
        let mut bonus: usize;
        // Softmax helper (temperature-scaled, NaN-safe)
        fn softmax_row(logits: &[f32], temp: f32) -> Vec<f32> {
            let t = if temp.is_finite() && temp > 0.0 {
                temp
            } else {
                1.0
            };
            let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| {
                a.max(if b.is_finite() { b } else { f32::NEG_INFINITY })
            });
            let mut exps: Vec<f32> = logits
                .iter()
                .map(|&x| ((if x.is_finite() { x } else { f32::NEG_INFINITY } - m) / t).exp())
                .collect();
            let s: f32 = exps.iter().sum();
            let inv = 1.0 / s.max(1e-30);
            for e in &mut exps {
                *e *= inv;
            }
            exps
        }
        // Simple deterministic RNG (splitmix64) seeded by time+offset
        let mut rng_state: u64 = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0x12345678)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(offset as u64))
        .max(1);
        let mut next_u01 = || {
            rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            ((z >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for i in 0..k {
            let tp = softmax_row(&target_logits[i], temperature);
            let dp_off = i * v;
            // Top-p filter on target for acceptance prob (match Python _softmax path
            // when top_p==1: full distribution)
            let p_target = tp[draft_tokens[i].min(v - 1)];
            let p_draft = draft_probs[dp_off + draft_tokens[i].min(v - 1)].max(0.0);
            let accept = (p_target / (p_draft + 1e-12)).min(1.0);
            if next_u01() <= accept as f64 {
                accepted += 1;
            } else {
                // Resample from max(0, p_target - p_draft)
                let mut corrected = vec![0.0f32; v];
                let mut s = 0.0f32;
                for j in 0..v {
                    let c = (tp[j] - draft_probs[dp_off + j]).max(0.0);
                    corrected[j] = c;
                    s += c;
                }
                if s > 1e-12 {
                    let r = next_u01() as f32 * s;
                    let mut acc = 0.0f32;
                    bonus = v - 1;
                    for (j, &c) in corrected.iter().enumerate() {
                        acc += c;
                        if acc >= r {
                            bonus = j;
                            break;
                        }
                    }
                } else {
                    bonus = tp
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(j, _)| j)
                        .unwrap_or(0);
                }
                return Ok((accepted, bonus));
            }
        }
        // All accepted: sample bonus from last target row (+ top-p)
        let last = softmax_row(&target_logits[k - 1], temperature);
        // Apply top-p nucleus on bonus sample to match sampler
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_unstable_by(|&a, &b| last[b].total_cmp(&last[a]));
        let mut kept = v;
        if top_p > 0.0 && top_p < 1.0 {
            let mut cum = 0.0f32;
            for (i, &j) in idx.iter().enumerate() {
                if cum > top_p {
                    kept = i;
                    break;
                }
                cum += last[j];
            }
            kept = kept.max(1);
        }
        let mut mass = 0.0f32;
        for &j in &idx[..kept] {
            mass += last[j];
        }
        let r = next_u01() as f32 * mass.max(1e-30);
        let mut acc = 0.0f32;
        bonus = idx[kept - 1];
        for &j in &idx[..kept] {
            acc += last[j];
            if acc >= r {
                bonus = j;
                break;
            }
        }
        Ok((accepted, bonus))
    }

    pub fn decode_step(
        &mut self,
        py: Python<'_>,
        token_id: usize,
        offset: usize,
    ) -> PyResult<PyObject> {
        self.step_internal(token_id, offset);
        let mut out = OwnedTensor::new(DType::F32, vec![1, self.vocab_size as i64]);
        let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };
        out_slice.copy_from_slice(&self.logits);
        dlpack::owned_to_capsule_owned(py, out).map(|c| c.into_any())
    }

    #[pyo3(signature = (token_id, offset, temperature=0.7, top_k=40, repetition_penalty=1.0, recent_tokens=None, top_p=1.0))]
    pub fn decode_and_sample(
        &mut self,
        token_id: usize,
        offset: usize,
        temperature: f32,
        top_k: usize,
        repetition_penalty: f32,
        recent_tokens: Option<Vec<usize>>,
        top_p: f32,
    ) -> PyResult<usize> {
        let rec = recent_tokens.unwrap_or_default();
        Ok(self.decode_and_sample_inner(
            token_id,
            offset,
            temperature,
            top_k,
            repetition_penalty,
            &rec,
            top_p,
        ))
    }

    /// Phase 2.3: run the entire decode loop in Rust. One FFI crossing per
    /// generation instead of one per token — the Python engine passes the
    /// first token and gets back every generated token id plus the final KV
    /// position, so multi-turn state stays synchronizable.
    #[pyo3(signature = (first_token, seq_len, max_new_tokens, temperature=0.7,
                        top_k=40, repetition_penalty=1.0, top_p=1.0, eos_token_id=None))]
    pub fn generate_loop(
        &mut self,
        first_token: usize,
        seq_len: usize,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        repetition_penalty: f32,
        top_p: f32,
        eos_token_id: Option<usize>,
    ) -> PyResult<(Vec<usize>, usize)> {
        let mut tokens: Vec<usize> = Vec::with_capacity(max_new_tokens);
        let mut next_token = first_token;
        let mut generated = 0usize;
        while generated < max_new_tokens {
            if Some(next_token) == eos_token_id {
                break;
            }
            tokens.push(next_token);
            generated += 1;
            let offset = seq_len + generated - 1;
            next_token = self.decode_and_sample_inner(
                next_token,
                offset,
                temperature,
                top_k,
                repetition_penalty,
                &tokens,
                top_p,
            );
        }
        Ok((tokens, seq_len + generated))
    }
}

impl RustQwenDecoder {
    /// Sampling + penalty logic shared by `decode_and_sample` (pyo3) and
    /// `generate_loop` (no per-token Python round-trip). Free of the pyo3
    /// impl block so it can take `&[usize]` without argument conversion.
    fn decode_and_sample_inner(
        &mut self,
        token_id: usize,
        offset: usize,
        temperature: f32,
        top_k: usize,
        repetition_penalty: f32,
        recent_tokens: &[usize],
        top_p: f32,
    ) -> usize {
        self.step_internal(token_id, offset);
        if repetition_penalty > 1.0 {
            let mut seen = std::collections::HashSet::new();
            for &t in recent_tokens {
                if t < self.logits.len() && seen.insert(t) {
                    let l = self.logits[t];
                    if l > 0.0 {
                        self.logits[t] = l / repetition_penalty;
                    } else {
                        self.logits[t] = l * repetition_penalty;
                    }
                }
            }
        }
        sample_logits(&self.logits, temperature, top_k, top_p)
    }
}

/// Temperature-scaled, top-k (+ optional nucleus top-p) sampling.
///
/// Matches the Python reference in `UniversalEngine._sample`: candidate logits
/// are the top-`k`, temperature-scaled with softmax; with `0 < top_p < 1` only
/// the smallest nucleus whose cumulative mass exceeds `top_p` is kept (the
/// same cutoff torch uses: drop items whose *preceding* cumulative mass is
/// already above the threshold) and the result is re-normalized.
pub fn sample_logits(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return 0;
    }
    // NaN/inf temperature -> greedy (avoids exp(inf/NaN) poisoning distribution)
    if !temperature.is_finite() || temperature <= 0.0 || top_k == 1 {
        // Greedy argmax
        let mut best_idx = 0;
        let mut best_val = logits[0];
        for i in 1..vocab_size {
            if logits[i] > best_val {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Top-k: NaN-safe, no huge alloc. Small-k linear scan; large-k partial select.
    let k = top_k.min(vocab_size).max(1);
    // Filter non-finite logits to -inf (NaN would poison max/softmax)
    // Fast path: k >= vocab -> keep all (no selection)
    let mut top_items: Vec<(usize, f32)> = Vec::with_capacity(k.min(4096) + 1);
    if k >= vocab_size {
        top_items.extend(
            logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, if v.is_finite() { v } else { f32::NEG_INFINITY })),
        );
    } else if k > 1024 {
        // Large k: partial select via nth_element (O(V) avg, no O(V*k) rescan)
        let mut indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, if v.is_finite() { v } else { f32::NEG_INFINITY }))
            .collect();
        let nth = k.min(indexed.len() - 1);
        indexed.select_nth_unstable_by(nth, |a, b| b.1.total_cmp(&a.1));
        top_items.extend_from_slice(&indexed[..k]);
    } else {
        let mut min_val = f32::NEG_INFINITY;
        let mut min_pos = 0;
        for (i, &raw) in logits.iter().enumerate() {
            let val = if raw.is_finite() {
                raw
            } else {
                f32::NEG_INFINITY
            };
            if top_items.len() < k {
                top_items.push((i, val));
                if val < min_val || top_items.len() == 1 {
                    min_val = val;
                    min_pos = top_items.len() - 1;
                }
            } else if val > min_val {
                top_items[min_pos] = (i, val);
                let mut new_min = top_items[0].1;
                let mut new_pos = 0;
                for (idx, &(_, v)) in top_items.iter().enumerate() {
                    if v < new_min {
                        new_min = v;
                        new_pos = idx;
                    }
                }
                min_val = new_min;
                min_pos = new_pos;
            }
        }
    }

    let max_logit = top_items
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let inv_temp = 1.0 / temperature;
    for item in &mut top_items {
        let p = ((item.1 - max_logit) * inv_temp).exp();
        item.1 = p;
    }

    // Optional nucleus (top-p) filtering on the temperature-scaled distribution.
    let mut kept_end = top_items.len();
    if top_p > 0.0 && top_p < 1.0 && kept_end > 1 {
        top_items.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut prev = 0.0f32;
        for (i, &(_, p)) in top_items.iter().enumerate() {
            if prev > top_p {
                kept_end = i;
                break;
            }
            prev += p;
        }
    }

    let kept = &top_items[..kept_end];
    let kept_sum: f32 = kept.iter().map(|&(_, p)| p).sum();
    let r = rand::random::<f32>() * kept_sum;
    let mut accum = 0.0f32;
    for &(idx, p) in kept {
        accum += p;
        if accum >= r {
            return idx;
        }
    }
    kept[0].0
}

impl RustQwenDecoder {
    /// Phase 1.2 dispatch: run a projection GEMV through the fused v2
    /// interleaved kernel when the decoder was constructed with v2-packed
    /// weights, else the v1 per-row path. Bias, when present, is added inside
    /// both kernels. Associated fn (no `&self` receiver) so callers can borrow
    /// weight slices and output buffers from `self` disjointly.
    #[inline]
    fn projection_gemv(
        use_v2: bool,
        group_size: usize,
        x: *const f32,
        w1: &[u8],
        s1: &[f32],
        w2: &[u8],
        bias: Option<&[f32]>,
        out: *mut f32,
        n: usize,
        k: usize,
    ) {
        let b_ptr = bias.map(|b| b.as_ptr());
        unsafe {
            if use_v2 {
                gemv_w4a32_grouped_v2(x, w2.as_ptr(), b_ptr, out, n, k, 64);
            } else {
                gemv_w4a32_grouped(x, w1.as_ptr(), s1.as_ptr(), b_ptr, out, n, k, group_size);
            }
        }
    }

    fn step_internal(&mut self, token_id: usize, offset: usize) {
        let hidden_size = self.hidden_size;
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let total_qkv = q_dim + 2 * kv_dim;
        let intermediate_size = self.intermediate_size;
        let group_size = self.group_size;
        let max_seq_len = self.max_seq_len;
        let head_stride = max_seq_len * head_dim;
        let half_dim = head_dim / 2;
        let eps = self.rms_norm_eps;

        // 1. Embedding lookup (bounds-checked; OOB -> zeroed step, no panic)
        let vocab = self.embed_tokens.len() / hidden_size.max(1);
        if token_id >= vocab || offset >= max_seq_len {
            self.x_buf.fill(0.0);
            self.logits.fill(0.0);
            return;
        }
        let emb_start = token_id * hidden_size;
        if emb_start + hidden_size > self.embed_tokens.len() {
            self.x_buf.fill(0.0);
            self.logits.fill(0.0);
            return;
        }
        self.x_buf
            .copy_from_slice(&self.embed_tokens[emb_start..emb_start + hidden_size]);

        let cos_offset = offset * head_dim;
        if cos_offset + head_dim > self.cos_table.len()
            || cos_offset + head_dim > self.sin_table.len()
        {
            self.x_buf.fill(0.0);
            self.logits.fill(0.0);
            return;
        }
        let cos_p = &self.cos_table[cos_offset..cos_offset + head_dim];
        let sin_p = &self.sin_table[cos_offset..cos_offset + head_dim];
        self.kv_used = self.kv_used.max(offset + 1);

        // 2. Iterate all layers
        for l in 0..self.num_layers {
            let layer = &self.layers[l];

            // A. Pre-attention RMSNorm
            unsafe {
                fast_rms_norm(
                    self.x_buf.as_ptr(),
                    layer.input_norm_w.as_ptr(),
                    self.normed_buf.as_mut_ptr(),
                    hidden_size,
                    eps,
                );
            }

            // B. QKV projection
            Self::projection_gemv(
                self.use_v2,
                self.group_size,
                self.normed_buf.as_ptr(),
                &layer.qkv_w,
                &layer.qkv_s,
                &layer.qkv_w2,
                layer.qkv_b.as_deref(),
                self.qkv_buf.as_mut_ptr(),
                total_qkv,
                hidden_size,
            );

            let (q, kv_rest) = self.qkv_buf.split_at_mut(q_dim);
            let (k, v) = kv_rest.split_at_mut(kv_dim);

            // C. RoPE (SIMD vectorised)
            for h in 0..num_heads {
                let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
                rope_apply_head(q_head, cos_p, sin_p, half_dim);
            }

            for h in 0..num_kv_heads {
                let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
                rope_apply_head(k_head, cos_p, sin_p, half_dim);
            }

            // D. Update KV caches
            let k_cache = &mut self.k_caches[l];
            let v_cache = &mut self.v_caches[l];
            for kv_h in 0..num_kv_heads {
                let dst_offset = kv_h * head_stride + offset * head_dim;
                let k_src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
                let v_src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
                k_cache[dst_offset..dst_offset + head_dim].copy_from_slice(k_src);
                v_cache[dst_offset..dst_offset + head_dim].copy_from_slice(v_src);
            }

            // E. GQA Attention (Rayon parallel across heads + SIMD FMA V-accumulation)
            let seq_len = offset + 1;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let heads_per_kv = (num_heads / num_kv_heads.max(1)).max(1);

            let q_ptr_val = q.as_ptr() as usize;
            let k_cache_ptr_val = k_cache.as_ptr() as usize;
            let v_cache_ptr_val = v_cache.as_ptr() as usize;

            self.attn_out
                .par_chunks_exact_mut(head_dim)
                .enumerate()
                .for_each(|(h, out_h)| {
                    out_h.fill(0.0);
                    let kv_h = h / heads_per_kv;
                    let q_head_ptr = unsafe { (q_ptr_val as *const f32).add(h * head_dim) };
                    let k_base_ptr =
                        unsafe { (k_cache_ptr_val as *const f32).add(kv_h * head_stride) };
                    let v_base_ptr =
                        unsafe { (v_cache_ptr_val as *const f32).add(kv_h * head_stride) };

                    // Fast path: seq_len==1 (steady-state decode) -> weight=1, copy V
                    if seq_len == 1 {
                        let v_t_ptr = unsafe { v_base_ptr as *const f32 };
                        unsafe {
                            std::ptr::copy_nonoverlapping(v_t_ptr, out_h.as_mut_ptr(), head_dim);
                        }
                        return;
                    }
                    THREAD_SCORES.with(|cell| {
                        let mut scores_buf = cell.borrow_mut();
                        // Cap at max_seq_len to avoid unbounded growth on 32k contexts
                        let capped = seq_len.min(32768);
                        if scores_buf.len() < capped {
                            scores_buf.resize(capped.max(512), 0.0);
                        }
                        let scores = &mut scores_buf[..capped];

                        let mut max_score = f32::NEG_INFINITY;
                        for t in 0..capped {
                            let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
                            let dot = unsafe { dot_f32_f32(q_head_ptr, k_t_ptr, head_dim) };
                            let sc = dot * scale;
                            scores[t] = sc;
                            if sc > max_score {
                                max_score = sc;
                            }
                        }

                        let mut exp_sum = 0.0f32;
                        for t in 0..capped {
                            let ex = (scores[t] - max_score).exp();
                            scores[t] = ex;
                            exp_sum += ex;
                        }
                        let inv_sum = 1.0f32 / exp_sum.max(1e-30);

                        for t in 0..capped {
                            let w = scores[t] * inv_sum;
                            let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
                            unsafe {
                                fast_vector_fma(out_h.as_mut_ptr(), v_t_ptr, w, head_dim);
                            }
                        }
                    });
                });

            // F. Output projection
            Self::projection_gemv(
                self.use_v2,
                self.group_size,
                self.attn_out.as_ptr(),
                &layer.o_w,
                &layer.o_s,
                &layer.o_w2,
                layer.o_b.as_deref(),
                self.o_out.as_mut_ptr(),
                hidden_size,
                q_dim,
            );

            // G. Residual add: x += o_out
            unsafe {
                fast_vector_add(self.x_buf.as_mut_ptr(), self.o_out.as_ptr(), hidden_size);
            }

            // H. Post-attention RMSNorm
            unsafe {
                fast_rms_norm(
                    self.x_buf.as_ptr(),
                    layer.post_norm_w.as_ptr(),
                    self.normed_buf.as_mut_ptr(),
                    hidden_size,
                    eps,
                );
            }

            // I. SwiGLU MLP
            let h_ptr = self.h_buf.as_mut_ptr() as usize;
            let num_groups_k = (hidden_size + group_size - 1) / group_size;
            let bytes_per_row_k = (hidden_size + 1) / 2;

            let x_usize = self.normed_buf.as_ptr() as usize;
            let gw_usize = layer.gate_w.as_ptr() as usize;
            let gs_usize = layer.gate_s.as_ptr() as usize;
            let gb_usize = layer.gate_b.as_ref().map(|b| b.as_ptr() as usize);

            let uw_usize = layer.up_w.as_ptr() as usize;
            let us_usize = layer.up_s.as_ptr() as usize;
            let ub_usize = layer.up_b.as_ref().map(|b| b.as_ptr() as usize);

            let n_threads = rayon::current_num_threads();
            let min_chunk = (intermediate_size / (n_threads * 4)).max(8);

            #[cfg(target_arch = "x86_64")]
            let has_vnni = crate::dispatch::cpu_features().avx512vnni;
            #[cfg(not(target_arch = "x86_64"))]
            let has_vnni = false;

            // Pre-quantise the activation for the W4A8 VNNI path only when this
            // CPU actually has VNNI and the layout is 64-wide groups (the only
            // layout the packed-VNNI kernel understands).  The neuron helper
            // re-checks and falls back to W4A32 AVX-512 / AVX2 / scalar.
            let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
                let (u, s) =
                    unsafe { quantize_activation_to_u8(self.normed_buf.as_ptr(), hidden_size) };
                (Some(u), s)
            } else {
                (None, 1.0f32)
            };
            let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

            (0..intermediate_size)
                .into_par_iter()
                .with_min_len(min_chunk)
                .for_each(|j| {
                    let x_p = x_usize as *const f32;
                    let gw_row = unsafe { (gw_usize as *const u8).add(j * bytes_per_row_k) };
                    let gs_row = unsafe { (gs_usize as *const f32).add(j * num_groups_k) };
                    let gb = if let Some(bp) = gb_usize {
                        unsafe { *((bp as *const f32).add(j)) }
                    } else {
                        0.0
                    };

                    let uw_row = unsafe { (uw_usize as *const u8).add(j * bytes_per_row_k) };
                    let us_row = unsafe { (us_usize as *const f32).add(j * num_groups_k) };
                    let ub = if let Some(bp) = ub_usize {
                        unsafe { *((bp as *const f32).add(j)) }
                    } else {
                        0.0
                    };

                    // CPU-feature + group-size aware dispatch (AVX-512 VNNI >
                    // AVX-512 > AVX2 > portable scalar).  Portable and safe on
                    // every tier — no unconditional AVX-512 calls.
                    let (g_sum, u_sum) = unsafe {
                        swiglu_neuron_w4a32_dot_dispatch(
                            x_p,
                            x_u8_ptr.map(|p| p as *const u8),
                            s_x,
                            gw_row,
                            gs_row,
                            uw_row,
                            us_row,
                            num_groups_k,
                            hidden_size,
                            group_size,
                        )
                    };

                    let g = g_sum + gb;
                    let u = u_sum + ub;
                    let silu_g = g / (1.0 + (-g).exp());
                    let val = silu_g * u;

                    unsafe {
                        let h_p = h_ptr as *mut f32;
                        *h_p.add(j) = val;
                    }
                });

            // Down GEMV
            Self::projection_gemv(
                self.use_v2,
                self.group_size,
                self.h_buf.as_ptr(),
                &layer.down_w,
                &layer.down_s,
                &layer.down_w2,
                layer.down_b.as_deref(),
                self.down_out.as_mut_ptr(),
                hidden_size,
                intermediate_size,
            );

            // J. Residual add: x += down_out
            unsafe {
                fast_vector_add(self.x_buf.as_mut_ptr(), self.down_out.as_ptr(), hidden_size);
            }
        }

        // 3. Final RMSNorm
        unsafe {
            fast_rms_norm(
                self.x_buf.as_ptr(),
                self.final_norm_w.as_ptr(),
                self.normed_buf.as_mut_ptr(),
                hidden_size,
                eps,
            );
        }

        // 4. LM Head GEMV
        Self::projection_gemv(
            self.use_v2,
            self.group_size,
            self.normed_buf.as_ptr(),
            &self.lm_head_w,
            &self.lm_head_s,
            &self.lm_head_w2,
            None,
            self.logits.as_mut_ptr(),
            self.vocab_size,
            hidden_size,
        );
    }
}
