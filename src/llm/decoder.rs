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
    dot_f32_f32, fast_rms_norm, fast_vector_add, gemv_w4a32_grouped, gemv_w4a32_grouped_v2,
    pack_rows_w4a32_group64_v1_to_v2, quantize_activation_to_u8,
    swiglu_neuron_w4a32_group64_avx512, swiglu_neuron_w4a8_group64_vnni_avx512,
};

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

    pub fn reset_kv_cache(&mut self) {
        for kc in &mut self.k_caches {
            kc.fill(0.0);
        }
        for vc in &mut self.v_caches {
            vc.fill(0.0);
        }
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
                    dst_k[dst_offset..dst_offset + self.head_dim]
                        .copy_from_slice(&k_slice[src_offset..src_offset + self.head_dim]);
                    dst_v[dst_offset..dst_offset + self.head_dim]
                        .copy_from_slice(&v_slice[src_offset..src_offset + self.head_dim]);
                }
            }
        }
        Ok(())
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
    if temperature <= 0.0 || top_k == 1 {
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

    // Top-k sampling: bounded vector tracking top-k (no 2.4 MB heap allocation per token)
    let k = top_k.min(vocab_size).max(1);
    let mut top_items: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    let mut min_val = f32::NEG_INFINITY;
    let mut min_pos = 0;

    for (i, &val) in logits.iter().enumerate() {
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
        top_items
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

        // 1. Embedding lookup
        let emb_start = token_id * hidden_size;
        self.x_buf
            .copy_from_slice(&self.embed_tokens[emb_start..emb_start + hidden_size]);

        let cos_offset = offset * head_dim;
        let cos_p = &self.cos_table[cos_offset..cos_offset + head_dim];
        let sin_p = &self.sin_table[cos_offset..cos_offset + head_dim];

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

            // C. RoPE
            for h in 0..num_heads {
                let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
                for i in 0..half_dim {
                    let q1 = q_head[i];
                    let q2 = q_head[i + half_dim];
                    let c1 = cos_p[i];
                    let s1 = sin_p[i];
                    let c2 = cos_p[i + half_dim];
                    let s2 = sin_p[i + half_dim];
                    q_head[i] = q1 * c1 - q2 * s1;
                    q_head[i + half_dim] = q2 * c2 + q1 * s2;
                }
            }

            for h in 0..num_kv_heads {
                let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
                for i in 0..half_dim {
                    let k1 = k_head[i];
                    let k2 = k_head[i + half_dim];
                    let c1 = cos_p[i];
                    let s1 = sin_p[i];
                    let c2 = cos_p[i + half_dim];
                    let s2 = sin_p[i + half_dim];
                    k_head[i] = k1 * c1 - k2 * s1;
                    k_head[i + half_dim] = k2 * c2 + k1 * s2;
                }
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

            // E. GQA Attention
            let seq_len = offset + 1;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let heads_per_kv = num_heads / num_kv_heads;

            self.attn_out.fill(0.0);

            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;
                let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
                let k_base_ptr = unsafe { k_cache.as_ptr().add(kv_h * head_stride) };
                let v_base_ptr = unsafe { v_cache.as_ptr().add(kv_h * head_stride) };

                let mut max_score = f32::NEG_INFINITY;
                for t in 0..seq_len {
                    let k_t_ptr = unsafe { k_base_ptr.add(t * head_dim) };
                    let dot = unsafe { dot_f32_f32(q_ptr, k_t_ptr, head_dim) };
                    let sc = dot * scale;
                    self.scores[t] = sc;
                    if sc > max_score {
                        max_score = sc;
                    }
                }

                let mut exp_sum = 0.0f32;
                for t in 0..seq_len {
                    let ex = (self.scores[t] - max_score).exp();
                    self.scores[t] = ex;
                    exp_sum += ex;
                }
                let inv_sum = 1.0f32 / exp_sum;

                let out_h_ptr = unsafe { self.attn_out.as_mut_ptr().add(h * head_dim) };
                for t in 0..seq_len {
                    let w = self.scores[t] * inv_sum;
                    let v_t_ptr = unsafe { v_base_ptr.add(t * head_dim) };
                    for d in 0..head_dim {
                        unsafe {
                            *out_h_ptr.add(d) += w * *v_t_ptr.add(d);
                        }
                    }
                }
            }

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
            let has_vnni = is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw");
            #[cfg(not(target_arch = "x86_64"))]
            let has_vnni = false;

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

                    let (g_sum, u_sum) = if let Some(x_u8_p) = x_u8_ptr {
                        #[cfg(target_arch = "x86_64")]
                        {
                            unsafe {
                                swiglu_neuron_w4a8_group64_vnni_avx512(
                                    x_u8_p as *const u8,
                                    s_x,
                                    gw_row,
                                    gs_row,
                                    uw_row,
                                    us_row,
                                    num_groups_k,
                                )
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            let _ = (x_u8_p, s_x);
                            unsafe {
                                swiglu_neuron_w4a32_group64_avx512(
                                    x_p,
                                    gw_row,
                                    gs_row,
                                    uw_row,
                                    us_row,
                                    num_groups_k,
                                )
                            }
                        }
                    } else {
                        unsafe {
                            swiglu_neuron_w4a32_group64_avx512(
                                x_p,
                                gw_row,
                                gs_row,
                                uw_row,
                                us_row,
                                num_groups_k,
                            )
                        }
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
