//! Bind-group + weight-buffer construction for the WGPU LLM decoder.
//!
//! Every per-layer BindGroup is pre-baked once at construction time (bind
//! groups are immutable in WebGPU), so token decode only ever *sets* pre-built
//! bind groups and never allocates.  The weight upload + bind-group bake
//! formerly buried in `WgpuQwenDecoder::new` lives here.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::dlpack;
use crate::wgpu::pipelines::WgpuPipelines;

pub(crate) struct WgpuLayerBindGroups {
    pub(crate) bg_rmsnorm_in: wgpu::BindGroup,
    pub(crate) bg_gemv_qkv: wgpu::BindGroup,
    pub(crate) bg_rope: wgpu::BindGroup,
    pub(crate) bg_attn: wgpu::BindGroup,
    pub(crate) bg_gemv_o: wgpu::BindGroup,
    pub(crate) bg_residual_attn: wgpu::BindGroup,
    pub(crate) bg_rmsnorm_post: wgpu::BindGroup,
    pub(crate) bg_gemv_swiglu: wgpu::BindGroup,
    pub(crate) bg_gemv_gate: wgpu::BindGroup,
    pub(crate) bg_gemv_up: wgpu::BindGroup,
    pub(crate) bg_swiglu: wgpu::BindGroup,
    pub(crate) bg_gemv_down: wgpu::BindGroup,
    pub(crate) bg_residual_mlp: wgpu::BindGroup,
    pub(crate) bg_fused_add_rmsnorm_attn: wgpu::BindGroup,
    pub(crate) bg_fused_add_rmsnorm_mlp: wgpu::BindGroup,
}

pub(crate) fn create_storage_buffer(
    device: &wgpu::Device,
    size: usize,
    read_only: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if !read_only {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    let aligned_size = ((size + 15) & !15).max(16);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: aligned_size as u64,
        usage,
        mapped_at_creation: false,
    })
}

pub(crate) fn create_and_upload_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
) -> wgpu::Buffer {
    let aligned_size = ((data.len() + 3) & !3).max(16);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: aligned_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, data);
    buf
}

pub(crate) fn create_uniform_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
) -> wgpu::Buffer {
    let aligned_size = ((data.len() + 15) & !15).max(16);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: aligned_size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, data);
    buf
}

unsafe fn tensor_to_bytes(t: &dlpack::BorrowedTensor) -> &[u8] {
    let total_bytes = t.buffer_len() * t.dtype.elem_size();
    std::slice::from_raw_parts(t.data as *const u8, total_bytes)
}

/// Everything `bake_model_bind_groups` produced, handed back to the decoder.
pub(crate) struct BakedBindGroups {
    pub(crate) layer_bgs: Vec<WgpuLayerBindGroups>,
    pub(crate) layer_k_caches: Vec<wgpu::Buffer>,
    pub(crate) layer_v_caches: Vec<wgpu::Buffer>,
    pub(crate) bg_rmsnorm_final: wgpu::BindGroup,
    pub(crate) bg_gemv_lm_head: wgpu::BindGroup,
}

/// Uploads every layer's INT4 weights and bakes all per-layer + final
/// (final-RMSNorm / LM-head) bind groups against the caller's shared scratch
/// and uniform buffers.  Returns the bind groups plus the KV-cache buffers —
/// the caches must outlive their bind-group references so they can later be
/// reset and copied into.
pub(crate) fn bake_model_bind_groups(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &WgpuPipelines,
    layers_data: &[Vec<Bound<'_, PyCapsule>>],
    final_norm_w: &Bound<'_, PyCapsule>,
    lm_head_w: &Bound<'_, PyCapsule>,
    lm_head_s: &Bound<'_, PyCapsule>,
    num_layers: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
    total_qkv: usize,
    x_buf: &wgpu::Buffer,
    normed_buf: &wgpu::Buffer,
    qkv_buf: &wgpu::Buffer,
    q_buf: &wgpu::Buffer,
    attn_out: &wgpu::Buffer,
    o_out: &wgpu::Buffer,
    gate_buf: &wgpu::Buffer,
    up_buf: &wgpu::Buffer,
    mlp_act: &wgpu::Buffer,
    down_out: &wgpu::Buffer,
    logits_buf: &wgpu::Buffer,
    cos_table_buf: &wgpu::Buffer,
    sin_table_buf: &wgpu::Buffer,
    rope_params_buf: &wgpu::Buffer,
    attn_params_buf: &wgpu::Buffer,
    rmsnorm_params_buf: &wgpu::Buffer,
    swiglu_params_buf: &wgpu::Buffer,
    residual_params_buf: &wgpu::Buffer,
    gemv_params_qkv_buf: &wgpu::Buffer,
    gemv_params_o_buf: &wgpu::Buffer,
    gemv_params_gate_up_buf: &wgpu::Buffer,
    gemv_params_down_buf: &wgpu::Buffer,
    gemv_params_lm_head_buf: &wgpu::Buffer,
) -> PyResult<BakedBindGroups> {
    let fnorm_view = unsafe { dlpack::BorrowedTensor::from_capsule(final_norm_w)? };
    let final_norm_buf =
        create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&fnorm_view) });
    let in_norm_bufs: Vec<wgpu::Buffer> = (0..num_layers)
        .map(|l| {
            let in_norm_v = unsafe { dlpack::BorrowedTensor::from_capsule(&layers_data[l][0])? };
            Ok(create_and_upload_storage_buffer(&device, &queue, unsafe {
                tensor_to_bytes(&in_norm_v)
            }))
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Upload layer weights and pre-bake bind groups
    let mut layer_bgs = Vec::with_capacity(num_layers);
    let mut layer_k_caches = Vec::with_capacity(num_layers);
    let mut layer_v_caches = Vec::with_capacity(num_layers);
    let kv_cache_size = num_kv_heads * max_seq_len * head_dim * 4;

    for l in 0..num_layers {
        let caps = &layers_data[l];
        let in_norm_buf = &in_norm_bufs[l];
        let qkv_w_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[1])? };
        let qkv_s_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[2])? };
        let o_w_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[3])? };
        let o_s_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[4])? };
        let post_norm_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[5])? };
        let gate_w_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[6])? };
        let gate_s_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[7])? };
        let up_w_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[8])? };
        let up_s_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[9])? };
        let down_w_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[10])? };
        let down_s_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[11])? };
        let qkv_w_buf =
            create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&qkv_w_v) });
        let qkv_s_buf =
            create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&qkv_s_v) });

        let qkv_b_buf = if caps.len() > 12 {
            let qkv_b_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[12])? };
            create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&qkv_b_v) })
        } else {
            create_storage_buffer(&device, total_qkv * 4, true)
        };

        let o_w_buf =
            create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&o_w_v) });
        let o_s_buf =
            create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&o_s_v) });
        let post_norm_buf = create_and_upload_storage_buffer(&device, &queue, unsafe {
            tensor_to_bytes(&post_norm_v)
        });
        let gate_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe {
            tensor_to_bytes(&gate_w_v)
        });
        let gate_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe {
            tensor_to_bytes(&gate_s_v)
        });
        let up_w_buf =
            create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&up_w_v) });
        let up_s_buf =
            create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&up_s_v) });
        let down_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe {
            tensor_to_bytes(&down_w_v)
        });
        let down_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe {
            tensor_to_bytes(&down_s_v)
        });

        let k_cache_buf = create_storage_buffer(&device, kv_cache_size, false);
        let v_cache_buf = create_storage_buffer(&device, kv_cache_size, false);

        // Pre-bake all BindGroups for this layer
        let bg_rmsnorm_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_rmsnorm_in"),
            layout: &pipelines.rmsnorm_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_norm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rmsnorm_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_gemv_qkv = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_gemv_qkv"),
            layout: &pipelines.gemv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qkv_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qkv_s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: qkv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gemv_params_qkv_buf.as_entire_binding(),
                },
            ],
        });

        let bg_rope = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_rope"),
            layout: &pipelines.rope_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: qkv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: qkv_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cos_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sin_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: k_cache_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: v_cache_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: rope_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_attn = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_attn"),
            layout: &pipelines.attn_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_cache_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_cache_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: attn_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: attn_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_gemv_o = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_gemv_o"),
            layout: &pipelines.gemv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: attn_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: o_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: o_s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: o_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gemv_params_o_buf.as_entire_binding(),
                },
            ],
        });

        let bg_residual_attn = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_residual_attn"),
            layout: &pipelines.residual_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: o_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: residual_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_rmsnorm_post = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_rmsnorm_post"),
            layout: &pipelines.rmsnorm_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: post_norm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rmsnorm_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_gemv_gate = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_gemv_gate"),
            layout: &pipelines.gemv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gate_s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gemv_params_gate_up_buf.as_entire_binding(),
                },
            ],
        });

        let bg_gemv_up = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_gemv_up"),
            layout: &pipelines.gemv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: up_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: up_s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: up_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gemv_params_gate_up_buf.as_entire_binding(),
                },
            ],
        });

        let bg_gemv_swiglu = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_gemv_swiglu"),
            layout: &pipelines.gemv_swiglu_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gate_s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: up_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: up_s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mlp_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: gemv_params_gate_up_buf.as_entire_binding(),
                },
            ],
        });

        let bg_swiglu = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_swiglu"),
            layout: &pipelines.swiglu_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: up_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mlp_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: swiglu_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_gemv_down = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_gemv_down"),
            layout: &pipelines.gemv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mlp_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: down_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: down_s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: down_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gemv_params_down_buf.as_entire_binding(),
                },
            ],
        });

        let bg_residual_mlp = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_residual_mlp"),
            layout: &pipelines.residual_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: down_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: residual_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg_fused_add_rmsnorm_attn = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_fused_add_rmsnorm_attn"),
            layout: &pipelines.fused_add_rmsnorm_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: o_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_norm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: rmsnorm_params_buf.as_entire_binding(),
                },
            ],
        });

        let next_norm_buf = if l + 1 < num_layers {
            &in_norm_bufs[l + 1]
        } else {
            &final_norm_buf
        };
        let bg_fused_add_rmsnorm_mlp = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_fused_add_rmsnorm_mlp"),
            layout: &pipelines.fused_add_rmsnorm_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: down_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: next_norm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: normed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: rmsnorm_params_buf.as_entire_binding(),
                },
            ],
        });

        layer_bgs.push(WgpuLayerBindGroups {
            bg_rmsnorm_in,
            bg_gemv_qkv,
            bg_rope,
            bg_attn,
            bg_gemv_o,
            bg_residual_attn,
            bg_rmsnorm_post,
            bg_gemv_swiglu,
            bg_gemv_gate,
            bg_gemv_up,
            bg_swiglu,
            bg_gemv_down,
            bg_residual_mlp,
            bg_fused_add_rmsnorm_attn,
            bg_fused_add_rmsnorm_mlp,
        });
        layer_k_caches.push(k_cache_buf);
        layer_v_caches.push(v_cache_buf);
    }

    // 5. Final norm & LM head
    let lm_w_view = unsafe { dlpack::BorrowedTensor::from_capsule(lm_head_w)? };
    let lm_w_buf =
        create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&lm_w_view) });
    let lm_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(lm_head_s)? };
    let lm_s_buf =
        create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&lm_s_view) });

    let bg_rmsnorm_final = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_rmsnorm_final"),
        layout: &pipelines.rmsnorm_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: final_norm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: normed_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: rmsnorm_params_buf.as_entire_binding(),
            },
        ],
    });

    let bg_gemv_lm_head = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_gemv_lm_head"),
        layout: &pipelines.gemv_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: normed_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lm_w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lm_s_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: logits_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: gemv_params_lm_head_buf.as_entire_binding(),
            },
        ],
    });

    Ok(BakedBindGroups {
        layer_bgs,
        layer_k_caches,
        layer_v_caches,
        bg_rmsnorm_final,
        bg_gemv_lm_head,
    })
}
