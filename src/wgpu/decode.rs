//! `WgpuQwenDecoder` — end-to-end GPU (Vulkan / Metal / DX12 / WebGPU) LLM
//! decoder driven by the wgpu compute stack.
//!
//! Fuses all transformer layers, RMSNorm, RoPE, attention, SwiGLU and GEMVs
//! into one GPU command buffer per token with exactly one hardware sync,
//! eliminating per-dispatch driver fence latency (measured ~12% faster decode
//! on Intel Iris Xe / Vulkan).

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::dlpack;
use crate::quantization::typed_slice;
use crate::wgpu::backend::get_wgpu_int4_context;
use crate::wgpu::bind_groups::{
    bake_model_bind_groups, create_and_upload_storage_buffer, create_storage_buffer,
    create_uniform_buffer, WgpuLayerBindGroups,
};
use crate::wgpu::pipelines::WgpuPipelines;

#[pyclass]
pub struct WgpuQwenDecoder {
    #[pyo3(get)]
    pub vocab_size: usize,
    #[pyo3(get)]
    pub hidden_size: usize,
    #[pyo3(get)]
    pub intermediate_size: usize,
    #[pyo3(get)]
    pub num_heads: usize,
    #[pyo3(get)]
    pub num_kv_heads: usize,
    #[pyo3(get)]
    pub head_dim: usize,
    #[pyo3(get)]
    pub num_layers: usize,
    #[pyo3(get)]
    pub group_size: usize,
    #[pyo3(get)]
    pub max_seq_len: usize,
    #[pyo3(get)]
    pub rows_per_wg: u32,
    // The full token-embedding table is kept host-side (fp32) and the current
    // row is uploaded per token. On Intel iGPUs the shared-memory VRAM heap is
    // the scarce resource, not system RAM: a 1.5B model's 933 MB embed table
    // does not fit comfortably in the GPU heap alongside the weights.
    pub(crate) embed_tokens: Vec<f32>,
    pub logits_cpu: Vec<f32>,

    // WGPU Device & Queue
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<WgpuPipelines>,

    // Intermediate persistent GPU buffers
    pub(crate) x_buf: wgpu::Buffer,
    pub(crate) normed_buf: wgpu::Buffer,
    pub(crate) qkv_buf: wgpu::Buffer,
    pub(crate) q_buf: wgpu::Buffer,
    pub(crate) attn_out: wgpu::Buffer,
    pub(crate) o_out: wgpu::Buffer,
    pub(crate) gate_buf: wgpu::Buffer,
    pub(crate) up_buf: wgpu::Buffer,
    pub(crate) mlp_act: wgpu::Buffer,
    pub(crate) down_out: wgpu::Buffer,
    pub(crate) logits_buf: wgpu::Buffer,
    /// Ring of staging buffers (Phase 2.2): each token reads back through a
    /// different slot so a map/unmap cycle never stalls the next submission.
    pub(crate) staging_bufs: Vec<wgpu::Buffer>,
    pub(crate) staging_ring_idx: usize,

    // Uniform buffers
    pub(crate) rope_params_buf: wgpu::Buffer,
    pub(crate) attn_params_buf: wgpu::Buffer,

    // Pre-baked bind groups per layer
    pub(crate) layer_bgs: Vec<WgpuLayerBindGroups>,
    pub(crate) layer_k_caches: Vec<wgpu::Buffer>,
    pub(crate) layer_v_caches: Vec<wgpu::Buffer>,
    pub(crate) bg_rmsnorm_final: wgpu::BindGroup,
    pub(crate) bg_gemv_lm_head: wgpu::BindGroup,
}

#[inline]
pub(crate) fn dispatch_gemv_tiled(
    cpass: &mut wgpu::ComputePass,
    num_rows: usize,
    rows_per_wg: u32,
) {
    let wgs = (num_rows as u32 + rows_per_wg - 1) / rows_per_wg;
    cpass.dispatch_workgroups(wgs.min(65535), (wgs + 65534) / 65535, 1);
}

impl WgpuQwenDecoder {
    pub(crate) fn read_buffer(&self, buf: &wgpu::Buffer, num_floats: usize) -> PyResult<Vec<f32>> {
        let size_bytes = (num_floats * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_staging"),
            size: size_bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..size_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut data = vec![0.0f32; num_floats];
        {
            let view = slice.get_mapped_range();
            let ptr = view.as_ptr() as *const f32;
            let s = unsafe { std::slice::from_raw_parts(ptr, num_floats) };
            data.copy_from_slice(s);
        }
        staging.unmap();
        Ok(data)
    }

    /// Writes the per-token dynamic inputs (token embedding row + RoPE/
    /// attention uniform blocks) to their GPU buffers.
    pub(crate) fn write_token_inputs(&self, token_id: usize, offset: usize) {
        let hidden_size = self.hidden_size;
        let emb_start = token_id * hidden_size;
        let emb_slice = &self.embed_tokens[emb_start..emb_start + hidden_size];
        let emb_bytes =
            unsafe { std::slice::from_raw_parts(emb_slice.as_ptr() as *const u8, hidden_size * 4) };
        self.queue.write_buffer(&self.x_buf, 0, emb_bytes);

        let rope_data: [u32; 8] = [
            offset as u32,
            self.num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            1u32, // has_bias = 1
            0,
            0,
        ];
        self.queue.write_buffer(&self.rope_params_buf, 0, unsafe {
            std::slice::from_raw_parts(rope_data.as_ptr() as *const u8, 32)
        });

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_data: [u32; 8] = [
            offset as u32,
            self.num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            scale.to_bits(),
            0,
            0,
        ];
        self.queue.write_buffer(&self.attn_params_buf, 0, unsafe {
            std::slice::from_raw_parts(attn_data.as_ptr() as *const u8, 32)
        });
    }

    /// Records the full model compute graph into a *single* compute pass.
    ///
    /// All ~200 dispatches (layer-0 RMSNorm, per-layer QKV/out/down GEMVs,
    /// RoPE + KV append, decode attention, fused residual + norm, fused
    /// gate + up + SwiGLU, and the LM head) are recorded inside one pass per
    /// token. wgpu inserts automatic buffer barriers between dispatches
    /// whenever a buffer's usage transitions (storage read_write -> read,
    /// etc.), so the sequential inter-op dependencies stay correct. Collapsing
    /// ~200 passes into one removes per-pass driver overhead (measured ~12%
    /// faster decode on Intel Iris Xe / Vulkan).
    pub(crate) fn record_model_dispatches(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        include_lm_head: bool,
    ) {
        let total_qkv = (self.num_heads * self.head_dim) + 2 * (self.num_kv_heads * self.head_dim);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_qwen_token_pass"),
                timestamp_writes: None,
            });

            // Layer 0 starts with RMSNorm on x_buf (which contains embed_tokens)
            cpass.set_pipeline(&self.pipelines.rmsnorm_pipeline);
            cpass.set_bind_group(0, &self.layer_bgs[0].bg_rmsnorm_in, &[]);
            cpass.dispatch_workgroups(1, 1, 1);

            for l in 0..self.num_layers {
                let bgs = &self.layer_bgs[l];

                // A. QKV GEMV
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_qkv, &[]);
                dispatch_gemv_tiled(&mut cpass, total_qkv, self.rows_per_wg);

                // B. RoPE & KV-Cache Append
                cpass.set_pipeline(&self.pipelines.rope_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rope, &[]);
                cpass.dispatch_workgroups((self.num_heads + self.num_kv_heads) as u32, 1, 1);

                // C. Decode Attention
                cpass.set_pipeline(&self.pipelines.attn_pipeline);
                cpass.set_bind_group(0, &bgs.bg_attn, &[]);
                cpass.dispatch_workgroups(self.num_heads as u32, 1, 1);

                // D. Out GEMV
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_o, &[]);
                dispatch_gemv_tiled(&mut cpass, self.hidden_size, self.rows_per_wg);

                // E. Fused Attn Residual Add + Post-RMSNorm
                cpass.set_pipeline(&self.pipelines.fused_add_rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_fused_add_rmsnorm_attn, &[]);
                cpass.dispatch_workgroups(1, 1, 1);

                // F. Fused Gate + Up GEMV + SwiGLU
                cpass.set_pipeline(&self.pipelines.gemv_swiglu_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_swiglu, &[]);
                dispatch_gemv_tiled(&mut cpass, self.intermediate_size, self.rows_per_wg);

                // G. Down GEMV
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_down, &[]);
                dispatch_gemv_tiled(&mut cpass, self.hidden_size, self.rows_per_wg);

                // H. Fused MLP Residual Add + Next-Layer Pre-RMSNorm
                //    (layer N-1's also applies the final RMSNorm)
                cpass.set_pipeline(&self.pipelines.fused_add_rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_fused_add_rmsnorm_mlp, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            if include_lm_head {
                // LM Head GEMV
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &self.bg_gemv_lm_head, &[]);
                dispatch_gemv_tiled(&mut cpass, self.vocab_size, self.rows_per_wg);
            }
        } // compute pass ends here
    }
}

#[pymethods]
impl WgpuQwenDecoder {
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
        let ctx = get_wgpu_int4_context().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("WGPU device is not available")
        })?;
        let rows_per_wg = ctx.rows_per_wg;
        let device = Arc::new(ctx.device.clone());
        let queue = Arc::new(ctx.queue.clone());
        let pipelines = Arc::new(WgpuPipelines::new(&device, rows_per_wg));

        let emb_view = unsafe { dlpack::BorrowedTensor::from_capsule(embed_tokens)? };
        let vocab_size = emb_view.shape[0] as usize;
        let emb_slice = unsafe { typed_slice::<f32>(&emb_view) };
        let embed_tokens_vec = emb_slice.to_vec();

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let total_qkv = q_dim + 2 * kv_dim;

        // 1. Allocate intermediate buffers on GPU
        let x_buf = create_storage_buffer(&device, hidden_size * 4, false);
        let normed_buf = create_storage_buffer(&device, hidden_size * 4, false);
        let qkv_buf = create_storage_buffer(&device, total_qkv * 4, false);
        let q_buf = create_storage_buffer(&device, q_dim * 4, false);
        let attn_out = create_storage_buffer(&device, hidden_size * 4, false);
        let o_out = create_storage_buffer(&device, hidden_size * 4, false);
        let gate_buf = create_storage_buffer(&device, intermediate_size * 4, false);
        let up_buf = create_storage_buffer(&device, intermediate_size * 4, false);
        let mlp_act = create_storage_buffer(&device, intermediate_size * 4, false);
        let down_out = create_storage_buffer(&device, hidden_size * 4, false);
        let logits_buf = create_storage_buffer(&device, vocab_size * 4, false);
        // Phase 2.2: N>=3 pinned staging buffers in a ring. A buffer must be
        // unmapped before it can be written again; with one buffer that forces
        // the CPU to finish reading before the next token's submit. With three,
        // the next submit always targets a fresh slot.
        const STAGING_RING: usize = 3;
        let staging_bufs: Vec<wgpu::Buffer> = (0..STAGING_RING)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("wgpu_staging_buf_{i}")),
                    size: ((vocab_size * 4).max(16)) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // 2. Uniform buffers
        let rmsnorm_params_data: [u32; 4] =
            [hidden_size as u32, (rms_norm_eps as f32).to_bits(), 0, 0];
        let rmsnorm_params_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(rmsnorm_params_data.as_ptr() as *const u8, 16)
        });

        let swiglu_params_data: [u32; 4] = [intermediate_size as u32, 0, 0, 0];
        let swiglu_params_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(swiglu_params_data.as_ptr() as *const u8, 16)
        });

        let residual_params_data: [u32; 4] = [hidden_size as u32, 0, 0, 0];
        let residual_params_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(residual_params_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_qkv_data: [u32; 4] = [
            total_qkv as u32,
            hidden_size as u32,
            group_size as u32,
            (hidden_size / group_size) as u32,
        ];
        let gemv_params_qkv_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_qkv_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_o_data: [u32; 4] = [
            hidden_size as u32,
            hidden_size as u32,
            group_size as u32,
            (hidden_size / group_size) as u32,
        ];
        let gemv_params_o_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_o_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_gate_up_data: [u32; 4] = [
            intermediate_size as u32,
            hidden_size as u32,
            group_size as u32,
            (hidden_size / group_size) as u32,
        ];
        let gemv_params_gate_up_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_gate_up_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_down_data: [u32; 4] = [
            hidden_size as u32,
            intermediate_size as u32,
            group_size as u32,
            (intermediate_size / group_size) as u32,
        ];
        let gemv_params_down_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_down_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_lm_head_data: [u32; 4] = [
            vocab_size as u32,
            hidden_size as u32,
            group_size as u32,
            (hidden_size / group_size) as u32,
        ];
        let gemv_params_lm_head_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_lm_head_data.as_ptr() as *const u8, 16)
        });

        // Dynamic uniform buffers updated per step
        let rope_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rope_params_buf"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let attn_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_params_buf"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Precompute RoPE tables and upload to GPU
        let half_dim = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * head_dim];
        let mut sin_table = vec![0.0f32; max_seq_len * head_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32);
                let val = (pos as f32) * freq;
                let c = val.cos();
                let s = val.sin();
                let idx1 = pos * head_dim + i;
                let idx2 = pos * head_dim + i + half_dim;
                cos_table[idx1] = c;
                cos_table[idx2] = c;
                sin_table[idx1] = s;
                sin_table[idx2] = s;
            }
        }
        let cos_table_buf = create_and_upload_storage_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(cos_table.as_ptr() as *const u8, cos_table.len() * 4)
        });
        let sin_table_buf = create_and_upload_storage_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(sin_table.as_ptr() as *const u8, sin_table.len() * 4)
        });

        let baked = bake_model_bind_groups(
            &device,
            &queue,
            &pipelines,
            &layers_data,
            final_norm_w,
            lm_head_w,
            lm_head_s,
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            total_qkv,
            &x_buf,
            &normed_buf,
            &qkv_buf,
            &q_buf,
            &attn_out,
            &o_out,
            &gate_buf,
            &up_buf,
            &mlp_act,
            &down_out,
            &logits_buf,
            &cos_table_buf,
            &sin_table_buf,
            &rope_params_buf,
            &attn_params_buf,
            &rmsnorm_params_buf,
            &swiglu_params_buf,
            &residual_params_buf,
            &gemv_params_qkv_buf,
            &gemv_params_o_buf,
            &gemv_params_gate_up_buf,
            &gemv_params_down_buf,
            &gemv_params_lm_head_buf,
        )?;

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            group_size,
            max_seq_len,
            rows_per_wg,
            embed_tokens: embed_tokens_vec,
            logits_cpu: vec![0.0f32; vocab_size],
            device,
            queue,
            pipelines,
            x_buf,
            normed_buf,
            qkv_buf,
            q_buf,
            attn_out,
            o_out,
            gate_buf,
            up_buf,
            mlp_act,
            down_out,
            logits_buf,
            staging_bufs,
            staging_ring_idx: 0,
            rope_params_buf,
            attn_params_buf,
            layer_bgs: baked.layer_bgs,
            layer_k_caches: baked.layer_k_caches,
            layer_v_caches: baked.layer_v_caches,
            bg_rmsnorm_final: baked.bg_rmsnorm_final,
            bg_gemv_lm_head: baked.bg_gemv_lm_head,
        })
    }

    /// Encodes and submits all layers of the model in one GPU command stream
    /// (1 compute pass, 1 queue submission, 1 readback sync per token).
    /// Returns the ring slot the logits were copied into.
    fn record_and_submit_step(&mut self, token_id: usize, offset: usize) -> PyResult<usize> {
        self.write_token_inputs(token_id, offset);

        let vocab_size = self.vocab_size;
        let slot = self.staging_ring_idx % self.staging_bufs.len();
        self.staging_ring_idx += 1;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_qwen_token_encoder"),
            });
        self.record_model_dispatches(&mut encoder, true);

        // Copy logits to the ring slot for CPU readback
        encoder.copy_buffer_to_buffer(
            &self.logits_buf,
            0,
            &self.staging_bufs[slot],
            0,
            (vocab_size * 4) as u64,
        );

        // Submit once to Vulkan / WGPU
        self.queue.submit(Some(encoder.finish()));

        Ok(slot)
    }

    /// Phase 2.2: non-blocking readback. Instead of `poll(PollType::Wait)` +
    /// blocking `rx.recv()` (which parks the OS thread until the whole GPU
    /// pipeline drains), this issues the map request immediately and then runs
    /// a poll loop that *yields* to other threads (Rayon, Python) while the
    /// GPU finishes. Short waits spin; long waits (full-model decode on an
    /// iGPU) fall back to a 100µs sleep so we never busy-burn a core.
    fn readback_logits(&self, slot: usize) -> PyResult<Vec<f32>> {
        let vocab_size = self.vocab_size;
        let buffer_slice = self.staging_bufs[slot].slice(..(vocab_size * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut spins: u32 = 0;
        let mapped = loop {
            // Drive wgpu maintenance + fire completed callbacks without blocking.
            let _ = self.device.poll(wgpu::PollType::Poll);
            match rx.try_recv() {
                Ok(result) => break result,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if std::time::Instant::now() > deadline {
                        return Err(pyo3::exceptions::PyTimeoutError::new_err(
                            "wgpu readback timed out after 10s",
                        ));
                    }
                    spins += 1;
                    if spins > 2000 {
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    } else {
                        std::thread::yield_now();
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "wgpu map_async callback channel disconnected",
                    ));
                }
            }
        };
        mapped.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut logits = vec![0.0f32; vocab_size];
        {
            let view = buffer_slice.get_mapped_range();
            let f32_data: &[f32] =
                unsafe { std::slice::from_raw_parts(view.as_ptr() as *const f32, vocab_size) };
            logits.copy_from_slice(f32_data);
        }
        self.staging_bufs[slot].unmap();
        Ok(logits)
    }

    /// Executes all 24 layers of the transformer model in a single GPU command stream.
    /// Exactly 1 hardware sync / poll per token.
    pub fn step(&mut self, token_id: usize, offset: usize) -> PyResult<Vec<f32>> {
        let slot = self.record_and_submit_step(token_id, offset)?;
        let mut logits_cpu = std::mem::take(&mut self.logits_cpu);
        readback_into(self, slot, &mut logits_cpu)?;
        self.logits_cpu = logits_cpu;
        Ok(self.logits_cpu.clone())
    }

    pub fn reset_kv_cache(&mut self) {
        let zeroes = vec![0u8; self.num_kv_heads * self.max_seq_len * self.head_dim * 4];
        for kc in &self.layer_k_caches {
            self.queue.write_buffer(kc, 0, &zeroes);
        }
        for vc in &self.layer_v_caches {
            self.queue.write_buffer(vc, 0, &zeroes);
        }
    }

    pub fn copy_kv_cache_from_tensors(
        &mut self,
        k_tensors: Vec<Bound<'_, PyCapsule>>,
        v_tensors: Vec<Bound<'_, PyCapsule>>,
        seq_len: usize,
    ) -> PyResult<()> {
        let dst_head_stride = self.max_seq_len * self.head_dim;
        let mut k_upload = vec![0.0f32; self.num_kv_heads * dst_head_stride];
        let mut v_upload = vec![0.0f32; self.num_kv_heads * dst_head_stride];

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

            for kv_h in 0..self.num_kv_heads {
                for t in 0..seq_len {
                    let src_offset = kv_h * src_head_stride + t * self.head_dim;
                    let dst_offset = kv_h * dst_head_stride + t * self.head_dim;
                    k_upload[dst_offset..dst_offset + self.head_dim]
                        .copy_from_slice(&k_slice[src_offset..src_offset + self.head_dim]);
                    v_upload[dst_offset..dst_offset + self.head_dim]
                        .copy_from_slice(&v_slice[src_offset..src_offset + self.head_dim]);
                }
            }

            let k_bytes = unsafe {
                std::slice::from_raw_parts(k_upload.as_ptr() as *const u8, k_upload.len() * 4)
            };
            let v_bytes = unsafe {
                std::slice::from_raw_parts(v_upload.as_ptr() as *const u8, v_upload.len() * 4)
            };
            self.queue.write_buffer(&self.layer_k_caches[l], 0, k_bytes);
            self.queue.write_buffer(&self.layer_v_caches[l], 0, v_bytes);
        }
        Ok(())
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
        let slot = self.record_and_submit_step(token_id, offset)?;
        let logits = self.readback_logits(slot)?;

        let token = {
            if let (true, Some(tokens)) = (repetition_penalty > 1.0, recent_tokens) {
                let mut logits_copy = logits;
                let mut seen = std::collections::HashSet::new();
                for t in tokens {
                    if t < self.vocab_size && seen.insert(t) {
                        let l = logits_copy[t];
                        if l > 0.0 {
                            logits_copy[t] = l / repetition_penalty;
                        } else {
                            logits_copy[t] = l * repetition_penalty;
                        }
                    }
                }
                crate::llm::sample_logits(&logits_copy, temperature, top_k, top_p)
            } else {
                crate::llm::sample_logits(&logits, temperature, top_k, top_p)
            }
        };

        Ok(token)
    }

    /// Phase 2.3: run the entire decode loop in Rust for the GPU decoder —
    /// one FFI crossing per generation. Submits, readbacks and sampling all
    /// stay on the Rust side; Python receives the token ids at the end.
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
        let mut logits_scratch: Vec<f32> = vec![0.0; self.vocab_size];
        let mut next_token = first_token;
        let mut generated = 0usize;
        while generated < max_new_tokens {
            if Some(next_token) == eos_token_id {
                break;
            }
            tokens.push(next_token);
            generated += 1;
            let offset = seq_len + generated - 1;
            let slot = self.record_and_submit_step(next_token, offset)?;
            readback_into(self, slot, &mut logits_scratch)?;
            if repetition_penalty > 1.0 {
                let mut seen = std::collections::HashSet::new();
                for &t in &tokens {
                    if t < self.vocab_size && seen.insert(t) {
                        let l = logits_scratch[t];
                        if l > 0.0 {
                            logits_scratch[t] = l / repetition_penalty;
                        } else {
                            logits_scratch[t] = l * repetition_penalty;
                        }
                    }
                }
            }
            next_token = crate::llm::sample_logits(&logits_scratch, temperature, top_k, top_p);
        }
        Ok((tokens, seq_len + generated))
    }
}

/// Phase 2.3: read logits from a ring slot directly into `dst`, avoiding the
/// intermediate `Vec<f32>` allocation + copy per token. Free function (outside
/// `#[pymethods]`) so the `&mut [f32]` parameter needs no pyo3 conversion.
fn readback_into(decoder: &WgpuQwenDecoder, slot: usize, dst: &mut [f32]) -> PyResult<()> {
    let vocab_size = decoder.vocab_size;
    let buffer_slice = decoder.staging_bufs[slot].slice(..(vocab_size * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut spins: u32 = 0;
    let mapped = loop {
        let _ = decoder.device.poll(wgpu::PollType::Poll);
        match rx.try_recv() {
            Ok(result) => break result,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if std::time::Instant::now() > deadline {
                    return Err(pyo3::exceptions::PyTimeoutError::new_err(
                        "wgpu readback timed out after 10s",
                    ));
                }
                spins += 1;
                if spins > 2000 {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                } else {
                    std::thread::yield_now();
                }
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "wgpu map_async callback channel disconnected",
                ));
            }
        }
    };
    mapped.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    {
        let view = buffer_slice.get_mapped_range();
        let f32_data: &[f32] =
            unsafe { std::slice::from_raw_parts(view.as_ptr() as *const f32, vocab_size) };
        dst.copy_from_slice(f32_data);
    }
    decoder.staging_bufs[slot].unmap();
    Ok(())
}
