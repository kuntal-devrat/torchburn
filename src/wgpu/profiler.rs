//! Per-step profiling / debug instrumentation for `WgpuQwenDecoder`.
//!
//! `profile_step_breakdown` splits one decode step into layer compute, LM-head
//! compute and full-logits readback phases (used by the benchmarks);
//! `debug_step_stages` runs layer 0 op-by-op with a CPU readback after every
//! stage so intermediate activations can be compared against the CPU decoder.

use pyo3::prelude::*;

use crate::wgpu::decode::{dispatch_gemv_tiled, WgpuQwenDecoder};

#[pymethods]
impl WgpuQwenDecoder {
    pub fn profile_step_breakdown(
        &mut self,
        token_id: usize,
        offset: usize,
    ) -> PyResult<(f64, f64, f64)> {
        let vocab_size = self.vocab_size;

        // Phase 1: token inputs + all 24 layers in a single merged compute pass
        self.write_token_inputs(token_id, offset);
        let t0 = std::time::Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.record_model_dispatches(&mut encoder, false);
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
        let t_layers = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: LM head GEMV + logits copy
        let t1 = std::time::Instant::now();
        let mut encoder2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.gemv_pipeline);
            cpass.set_bind_group(0, &self.bg_gemv_lm_head, &[]);
            dispatch_gemv_tiled(&mut cpass, vocab_size, self.rows_per_wg);
        }
        encoder2.copy_buffer_to_buffer(
            &self.logits_buf,
            0,
            &self.staging_bufs[0],
            0,
            (vocab_size * 4) as u64,
        );
        self.queue.submit(Some(encoder2.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
        let t_lm_head = t1.elapsed().as_secs_f64() * 1000.0;

        // Phase 3: full-logits readback
        let t2 = std::time::Instant::now();
        let buffer_slice = self.staging_bufs[0].slice(..(vocab_size * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        let _ = rx.recv();
        {
            let _view = buffer_slice.get_mapped_range();
        }
        self.staging_bufs[0].unmap();
        let t_readback = t2.elapsed().as_secs_f64() * 1000.0;

        Ok((t_layers, t_lm_head, t_readback))
    }

    pub fn debug_step_stages(
        &mut self,
        token_id: usize,
        offset: usize,
    ) -> PyResult<std::collections::HashMap<String, Vec<f32>>> {
        let hidden_size = self.hidden_size;
        let total_qkv = (self.num_heads * self.head_dim) + 2 * (self.num_kv_heads * self.head_dim);
        let intermediate_size = self.intermediate_size;
        let mut map = std::collections::HashMap::new();

        // 1. GPU-side embedding lookup (token_id_buf → embed_lookup shader → x_buf).
        // This mirrors the normal decode path: upload only the 4-byte token id,
        // dispatch the embed_lookup shader to fill x_buf, then readback for the
        // debug snapshot.
        let tid_bytes = (token_id as u32).to_le_bytes();
        self.queue.write_buffer(&self.token_id_buf, 0, &tid_bytes);
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.embed_lookup_pipeline);
                cpass.set_bind_group(0, &self.bg_embed_lookup, &[]);
                let embed_wgs = (hidden_size as u32 + 255) / 256;
                cpass.dispatch_workgroups(embed_wgs, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait);
        }
        map.insert(
            "embed".to_string(),
            self.read_buffer(&self.x_buf, hidden_size)?,
        );

        // 2. Uniforms
        let rope_data: [u32; 8] = [
            offset as u32,
            self.num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            1u32,
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

        let bgs = &self.layer_bgs[0];

        // A. Pre-RMSNorm
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rmsnorm_in, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "normed".to_string(),
                self.read_buffer(&self.normed_buf, hidden_size)?,
            );
        }

        // B. QKV GEMV
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_qkv, &[]);
                dispatch_gemv_tiled(&mut cpass, total_qkv, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "qkv".to_string(),
                self.read_buffer(&self.qkv_buf, total_qkv)?,
            );
        }

        // C. RoPE & KV-Cache Append
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.rope_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rope, &[]);
                cpass.dispatch_workgroups((self.num_heads + self.num_kv_heads) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "q".to_string(),
                self.read_buffer(&self.q_buf, self.num_heads * self.head_dim)?,
            );
        }

        // D. Decode Attention
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.attn_pipeline);
                cpass.set_bind_group(0, &bgs.bg_attn, &[]);
                cpass.dispatch_workgroups(self.num_heads as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "attn_out".to_string(),
                self.read_buffer(&self.attn_out, hidden_size)?,
            );
        }

        // E. Out GEMV
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_o, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "o_out".to_string(),
                self.read_buffer(&self.o_out, hidden_size)?,
            );
        }

        // F. Residual Add (Attn)
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.residual_pipeline);
                cpass.set_bind_group(0, &bgs.bg_residual_attn, &[]);
                cpass.dispatch_workgroups(((hidden_size + 63) / 64) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "x_attn".to_string(),
                self.read_buffer(&self.x_buf, hidden_size)?,
            );
        }

        // G. Post-RMSNorm
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rmsnorm_post, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "normed_post".to_string(),
                self.read_buffer(&self.normed_buf, hidden_size)?,
            );
        }

        // H & I. Gate & Up GEMV
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_gate, &[]);
                dispatch_gemv_tiled(&mut cpass, intermediate_size, self.rows_per_wg);

                cpass.set_bind_group(0, &bgs.bg_gemv_up, &[]);
                dispatch_gemv_tiled(&mut cpass, intermediate_size, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "gate".to_string(),
                self.read_buffer(&self.gate_buf, intermediate_size)?,
            );
            map.insert(
                "up".to_string(),
                self.read_buffer(&self.up_buf, intermediate_size)?,
            );
        }

        // J. SwiGLU
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.swiglu_pipeline);
                cpass.set_bind_group(0, &bgs.bg_swiglu, &[]);
                cpass.dispatch_workgroups(((intermediate_size + 63) / 64) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "mlp_act".to_string(),
                self.read_buffer(&self.mlp_act, intermediate_size)?,
            );
        }

        // K. Down GEMV
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_down, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "down_out".to_string(),
                self.read_buffer(&self.down_out, hidden_size)?,
            );
        }

        // L. Residual Add (MLP)
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.residual_pipeline);
                cpass.set_bind_group(0, &bgs.bg_residual_mlp, &[]);
                cpass.dispatch_workgroups(((hidden_size + 63) / 64) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert(
                "x_final_layer0".to_string(),
                self.read_buffer(&self.x_buf, hidden_size)?,
            );
        }

        Ok(map)
    }
}
