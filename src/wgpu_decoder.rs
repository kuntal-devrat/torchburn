//! End-to-End GPU Compute Graph Decoder for Qwen / LLaMA architectures using WGPU.
//!
//! Fuses all 24 layers, RMSNorm, RoPE, Attention, SwiGLU, and GEMVs into a single
//! GPU command buffer per token with exactly 1 hardware sync, eliminating driver fence latency.

#[cfg(feature = "burn-wgpu")]
use std::sync::Arc;
#[cfg(feature = "burn-wgpu")]
use pyo3::prelude::*;
#[cfg(feature = "burn-wgpu")]
use pyo3::types::PyCapsule;
#[cfg(feature = "burn-wgpu")]
use crate::dlpack;
#[cfg(feature = "burn-wgpu")]
use crate::quantization::typed_slice;
#[cfg(feature = "burn-wgpu")]
use crate::wgpu_backend::get_wgpu_int4_context;

#[cfg(feature = "burn-wgpu")]
struct WgpuPipelines {
    gemv_pipeline: wgpu::ComputePipeline,
    gemv_bgl: wgpu::BindGroupLayout,
    gemv_swiglu_pipeline: wgpu::ComputePipeline,
    gemv_swiglu_bgl: wgpu::BindGroupLayout,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    rmsnorm_bgl: wgpu::BindGroupLayout,
    rope_pipeline: wgpu::ComputePipeline,
    rope_bgl: wgpu::BindGroupLayout,
    attn_pipeline: wgpu::ComputePipeline,
    attn_bgl: wgpu::BindGroupLayout,
    swiglu_pipeline: wgpu::ComputePipeline,
    swiglu_bgl: wgpu::BindGroupLayout,
    residual_pipeline: wgpu::ComputePipeline,
    residual_bgl: wgpu::BindGroupLayout,
    fused_add_rmsnorm_pipeline: wgpu::ComputePipeline,
    fused_add_rmsnorm_bgl: wgpu::BindGroupLayout,
    rows_per_wg: u32,
}

#[cfg(feature = "burn-wgpu")]
impl WgpuPipelines {
    fn new(device: &wgpu::Device, rows_per_wg: u32) -> Self {
        let wg_size = rows_per_wg * 16;
        let gemv_shader_raw = include_str!("shaders/gemv_w4a32.wgsl");
        let gemv_shader_src = gemv_shader_raw
            .replace("const ROWS_PER_WG: u32 = 4u;", &format!("const ROWS_PER_WG: u32 = {}u;", rows_per_wg))
            .replace("const WG_SIZE: u32 = 64u;", &format!("const WG_SIZE: u32 = {}u;", wg_size));

        // 1. GEMV pipeline & layout
        let gemv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv_w4a32.wgsl"),
            source: wgpu::ShaderSource::Wgsl(gemv_shader_src.into()),
        });
        let gemv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemv_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let gemv_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemv_layout"),
            bind_group_layouts: &[&gemv_bgl],
            push_constant_ranges: &[],
        });
        let gemv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemv_pipeline"),
            layout: Some(&gemv_layout),
            module: &gemv_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 1b. Fused Gate + Up GEMV + SwiGLU pipeline & layout
        let gemv_swiglu_raw = include_str!("shaders/gemv_swiglu_w4a32.wgsl");
        let gemv_swiglu_src = gemv_swiglu_raw
            .replace("const ROWS_PER_WG: u32 = 4u;", &format!("const ROWS_PER_WG: u32 = {}u;", rows_per_wg))
            .replace("const WG_SIZE: u32 = 64u;", &format!("const WG_SIZE: u32 = {}u;", wg_size));
        let gemv_swiglu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv_swiglu_w4a32.wgsl"),
            source: wgpu::ShaderSource::Wgsl(gemv_swiglu_src.into()),
        });
        let gemv_swiglu_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemv_swiglu_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let gemv_swiglu_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemv_swiglu_layout"),
            bind_group_layouts: &[&gemv_swiglu_bgl],
            push_constant_ranges: &[],
        });
        let gemv_swiglu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemv_swiglu_pipeline"),
            layout: Some(&gemv_swiglu_layout),
            module: &gemv_swiglu_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 2. RMSNorm pipeline & layout
        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rmsnorm.wgsl").into()),
        });
        let rmsnorm_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rmsnorm_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let rmsnorm_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rmsnorm_layout"),
            bind_group_layouts: &[&rmsnorm_bgl],
            push_constant_ranges: &[],
        });
        let rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rmsnorm_pipeline"),
            layout: Some(&rmsnorm_layout),
            module: &rmsnorm_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 3. RoPE + KV Cache Append pipeline & layout
        let rope_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope_append.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rope_append.wgsl").into()),
        });
        let rope_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rope_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let rope_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rope_layout"),
            bind_group_layouts: &[&rope_bgl],
            push_constant_ranges: &[],
        });
        let rope_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rope_pipeline"),
            layout: Some(&rope_layout),
            module: &rope_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 4. Attn Decode pipeline & layout
        let attn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attn_decode.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/attn_decode.wgsl").into()),
        });
        let attn_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let attn_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attn_layout"),
            bind_group_layouts: &[&attn_bgl],
            push_constant_ranges: &[],
        });
        let attn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attn_pipeline"),
            layout: Some(&attn_layout),
            module: &attn_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 5. SwiGLU pipeline & layout
        let swiglu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("swiglu.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/swiglu.wgsl").into()),
        });
        let swiglu_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("swiglu_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let swiglu_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("swiglu_layout"),
            bind_group_layouts: &[&swiglu_bgl],
            push_constant_ranges: &[],
        });
        let swiglu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("swiglu_pipeline"),
            layout: Some(&swiglu_layout),
            module: &swiglu_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 6. Residual Add pipeline & layout
        let residual_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("residual_add.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/residual_add.wgsl").into()),
        });
        let residual_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("residual_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let residual_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("residual_layout"),
            bind_group_layouts: &[&residual_bgl],
            push_constant_ranges: &[],
        });
        let residual_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("residual_pipeline"),
            layout: Some(&residual_layout),
            module: &residual_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 7. Fused Residual Add + RMSNorm pipeline & layout
        let fused_add_rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fused_add_rmsnorm.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fused_add_rmsnorm.wgsl").into()),
        });
        let fused_add_rmsnorm_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fused_add_rmsnorm_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let fused_add_rmsnorm_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fused_add_rmsnorm_layout"),
            bind_group_layouts: &[&fused_add_rmsnorm_bgl],
            push_constant_ranges: &[],
        });
        let fused_add_rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused_add_rmsnorm_pipeline"),
            layout: Some(&fused_add_rmsnorm_layout),
            module: &fused_add_rmsnorm_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            gemv_pipeline,
            gemv_bgl,
            gemv_swiglu_pipeline,
            gemv_swiglu_bgl,
            rmsnorm_pipeline,
            rmsnorm_bgl,
            rope_pipeline,
            rope_bgl,
            attn_pipeline,
            attn_bgl,
            swiglu_pipeline,
            swiglu_bgl,
            residual_pipeline,
            residual_bgl,
            fused_add_rmsnorm_pipeline,
            fused_add_rmsnorm_bgl,
            rows_per_wg,
        }
    }
}

#[cfg(feature = "burn-wgpu")]
struct WgpuLayerBindGroups {
    bg_rmsnorm_in: wgpu::BindGroup,
    bg_gemv_qkv: wgpu::BindGroup,
    bg_rope: wgpu::BindGroup,
    bg_attn: wgpu::BindGroup,
    bg_gemv_o: wgpu::BindGroup,
    bg_residual_attn: wgpu::BindGroup,
    bg_rmsnorm_post: wgpu::BindGroup,
    bg_gemv_swiglu: wgpu::BindGroup,
    bg_gemv_gate: wgpu::BindGroup,
    bg_gemv_up: wgpu::BindGroup,
    bg_swiglu: wgpu::BindGroup,
    bg_gemv_down: wgpu::BindGroup,
    bg_residual_mlp: wgpu::BindGroup,
    bg_fused_add_rmsnorm_attn: wgpu::BindGroup,
    bg_fused_add_rmsnorm_mlp: wgpu::BindGroup,
}

#[cfg(feature = "burn-wgpu")]
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
    embed_tokens: Vec<f32>,
    pub logits_cpu: Vec<f32>,

    // WGPU Device & Queue
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<WgpuPipelines>,

    // Intermediate persistent GPU buffers
    x_buf: wgpu::Buffer,
    normed_buf: wgpu::Buffer,
    qkv_buf: wgpu::Buffer,
    q_buf: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    o_out: wgpu::Buffer,
    gate_buf: wgpu::Buffer,
    up_buf: wgpu::Buffer,
    mlp_act: wgpu::Buffer,
    down_out: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,

    // Uniform buffers
    rope_params_buf: wgpu::Buffer,
    attn_params_buf: wgpu::Buffer,

    // Pre-baked bind groups per layer
    layer_bgs: Vec<WgpuLayerBindGroups>,
    layer_k_caches: Vec<wgpu::Buffer>,
    layer_v_caches: Vec<wgpu::Buffer>,
    bg_rmsnorm_final: wgpu::BindGroup,
    bg_gemv_lm_head: wgpu::BindGroup,
}

#[cfg(feature = "burn-wgpu")]
#[inline]
fn dispatch_gemv_tiled(cpass: &mut wgpu::ComputePass, num_rows: usize, rows_per_wg: u32) {
    let wgs = (num_rows as u32 + rows_per_wg - 1) / rows_per_wg;
    cpass.dispatch_workgroups(wgs.min(65535), (wgs + 65534) / 65535, 1);
}

#[cfg(feature = "burn-wgpu")]
fn create_storage_buffer(device: &wgpu::Device, size: usize, read_only: bool) -> wgpu::Buffer {
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

#[cfg(feature = "burn-wgpu")]
fn create_and_upload_storage_buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> wgpu::Buffer {
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

#[cfg(feature = "burn-wgpu")]
fn create_uniform_buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> wgpu::Buffer {
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

#[cfg(feature = "burn-wgpu")]
unsafe fn tensor_to_bytes(t: &dlpack::BorrowedTensor) -> &[u8] {
    let total_bytes = t.buffer_len() * t.dtype.elem_size();
    std::slice::from_raw_parts(t.data as *const u8, total_bytes)
}

#[cfg(feature = "burn-wgpu")]
impl WgpuQwenDecoder {
    fn read_buffer(&self, buf: &wgpu::Buffer, num_floats: usize) -> PyResult<Vec<f32>> {
        let size_bytes = (num_floats * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_staging"),
            size: size_bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..size_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
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
}

#[cfg(feature = "burn-wgpu")]
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
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_staging_buf"),
            size: ((vocab_size * 4).max(16)) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. Uniform buffers
        let rmsnorm_params_data: [u32; 4] = [hidden_size as u32, (rms_norm_eps as f32).to_bits(), 0, 0];
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

        let gemv_params_qkv_data: [u32; 4] = [total_qkv as u32, hidden_size as u32, group_size as u32, (hidden_size / group_size) as u32];
        let gemv_params_qkv_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_qkv_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_o_data: [u32; 4] = [hidden_size as u32, hidden_size as u32, group_size as u32, (hidden_size / group_size) as u32];
        let gemv_params_o_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_o_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_gate_up_data: [u32; 4] = [intermediate_size as u32, hidden_size as u32, group_size as u32, (hidden_size / group_size) as u32];
        let gemv_params_gate_up_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_gate_up_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_down_data: [u32; 4] = [hidden_size as u32, intermediate_size as u32, group_size as u32, (intermediate_size / group_size) as u32];
        let gemv_params_down_buf = create_uniform_buffer(&device, &queue, unsafe {
            std::slice::from_raw_parts(gemv_params_down_data.as_ptr() as *const u8, 16)
        });

        let gemv_params_lm_head_data: [u32; 4] = [vocab_size as u32, hidden_size as u32, group_size as u32, (hidden_size / group_size) as u32];
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

        // 4. Upload final norm and in_norm buffers first, so layers can link forward
        let fnorm_view = unsafe { dlpack::BorrowedTensor::from_capsule(final_norm_w)? };
        let final_norm_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&fnorm_view) });
        let in_norm_bufs: Vec<wgpu::Buffer> = (0..num_layers)
            .map(|l| {
                let in_norm_v = unsafe { dlpack::BorrowedTensor::from_capsule(&layers_data[l][0])? };
                Ok(create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&in_norm_v) }))
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
            let qkv_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&qkv_w_v) });
            let qkv_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&qkv_s_v) });

            let qkv_b_buf = if caps.len() > 12 {
                let qkv_b_v = unsafe { dlpack::BorrowedTensor::from_capsule(&caps[12])? };
                create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&qkv_b_v) })
            } else {
                create_storage_buffer(&device, total_qkv * 4, true)
            };

            let o_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&o_w_v) });
            let o_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&o_s_v) });
            let post_norm_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&post_norm_v) });
            let gate_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&gate_w_v) });
            let gate_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&gate_s_v) });
            let up_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&up_w_v) });
            let up_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&up_s_v) });
            let down_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&down_w_v) });
            let down_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&down_s_v) });

            let k_cache_buf = create_storage_buffer(&device, kv_cache_size, false);
            let v_cache_buf = create_storage_buffer(&device, kv_cache_size, false);

            // Pre-bake all BindGroups for this layer
            let bg_rmsnorm_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_rmsnorm_in"),
                layout: &pipelines.rmsnorm_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: in_norm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: rmsnorm_params_buf.as_entire_binding() },
                ],
            });

            let bg_gemv_qkv = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_gemv_qkv"),
                layout: &pipelines.gemv_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: qkv_w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: qkv_s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: qkv_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gemv_params_qkv_buf.as_entire_binding() },
                ],
            });

            let bg_rope = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_rope"),
                layout: &pipelines.rope_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: qkv_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: qkv_b_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cos_table_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: sin_table_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: q_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: k_cache_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: v_cache_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: rope_params_buf.as_entire_binding() },
                ],
            });

            let bg_attn = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_attn"),
                layout: &pipelines.attn_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: k_cache_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: v_cache_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: attn_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: attn_params_buf.as_entire_binding() },
                ],
            });

            let bg_gemv_o = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_gemv_o"),
                layout: &pipelines.gemv_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: attn_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: o_w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: o_s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: o_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gemv_params_o_buf.as_entire_binding() },
                ],
            });

            let bg_residual_attn = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_residual_attn"),
                layout: &pipelines.residual_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: o_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: residual_params_buf.as_entire_binding() },
                ],
            });

            let bg_rmsnorm_post = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_rmsnorm_post"),
                layout: &pipelines.rmsnorm_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: post_norm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: rmsnorm_params_buf.as_entire_binding() },
                ],
            });

            let bg_gemv_gate = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_gemv_gate"),
                layout: &pipelines.gemv_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gate_w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gate_s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: gate_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gemv_params_gate_up_buf.as_entire_binding() },
                ],
            });

            let bg_gemv_up = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_gemv_up"),
                layout: &pipelines.gemv_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: up_w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: up_s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: up_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gemv_params_gate_up_buf.as_entire_binding() },
                ],
            });

            let bg_gemv_swiglu = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_gemv_swiglu"),
                layout: &pipelines.gemv_swiglu_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gate_w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gate_s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: up_w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: up_s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: mlp_act.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: gemv_params_gate_up_buf.as_entire_binding() },
                ],
            });

            let bg_swiglu = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_swiglu"),
                layout: &pipelines.swiglu_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gate_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: up_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: mlp_act.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: swiglu_params_buf.as_entire_binding() },
                ],
            });

            let bg_gemv_down = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_gemv_down"),
                layout: &pipelines.gemv_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: mlp_act.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: down_w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: down_s_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: down_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gemv_params_down_buf.as_entire_binding() },
                ],
            });

            let bg_residual_mlp = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_residual_mlp"),
                layout: &pipelines.residual_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: down_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: residual_params_buf.as_entire_binding() },
                ],
            });

            let bg_fused_add_rmsnorm_attn = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_fused_add_rmsnorm_attn"),
                layout: &pipelines.fused_add_rmsnorm_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: o_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: post_norm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: rmsnorm_params_buf.as_entire_binding() },
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
                    wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: down_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: next_norm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: normed_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: rmsnorm_params_buf.as_entire_binding() },
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
        let lm_w_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&lm_w_view) });
        let lm_s_view = unsafe { dlpack::BorrowedTensor::from_capsule(lm_head_s)? };
        let lm_s_buf = create_and_upload_storage_buffer(&device, &queue, unsafe { tensor_to_bytes(&lm_s_view) });

        let bg_rmsnorm_final = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_rmsnorm_final"),
            layout: &pipelines.rmsnorm_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: final_norm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: normed_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: rmsnorm_params_buf.as_entire_binding() },
            ],
        });

        let bg_gemv_lm_head = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_gemv_lm_head"),
            layout: &pipelines.gemv_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: normed_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lm_w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lm_s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: logits_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gemv_params_lm_head_buf.as_entire_binding() },
            ],
        });

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
            staging_buf,
            rope_params_buf,
            attn_params_buf,
            layer_bgs,
            layer_k_caches,
            layer_v_caches,
            bg_rmsnorm_final,
            bg_gemv_lm_head,
        })
    }

    /// Encodes and submits all 24 layers of the transformer model in a single GPU command stream.
    /// Uses fused Gate + Up GEMV + SwiGLU kernel to eliminate intermediate buffer roundtrips and passes.
    fn record_and_submit_step(&mut self, token_id: usize, offset: usize) -> PyResult<()> {
        let hidden_size = self.hidden_size;
        let total_qkv = (self.num_heads * self.head_dim) + 2 * (self.num_kv_heads * self.head_dim);
        let intermediate_size = self.intermediate_size;
        let vocab_size = self.vocab_size;

        // 1. Upload token embedding to x_buf
        let emb_start = token_id * hidden_size;
        let emb_slice = &self.embed_tokens[emb_start..emb_start + hidden_size];
        let emb_bytes = unsafe { std::slice::from_raw_parts(emb_slice.as_ptr() as *const u8, hidden_size * 4) };
        self.queue.write_buffer(&self.x_buf, 0, emb_bytes);

        // 2. Update dynamic RoPE and Attention uniforms
        let rope_data: [u32; 8] = [
            offset as u32,
            self.num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            1u32, // has_bias = 1
            0, 0,
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
            0, 0,
        ];
        self.queue.write_buffer(&self.attn_params_buf, 0, unsafe {
            std::slice::from_raw_parts(attn_data.as_ptr() as *const u8, 32)
        });

        // 3. Record all 24 layers into a single CommandEncoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_qwen_token_encoder"),
        });

        // Layer 0 starts with RMSNorm on x_buf (which contains embed_tokens)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.rmsnorm_pipeline);
            cpass.set_bind_group(0, &self.layer_bgs[0].bg_rmsnorm_in, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        for l in 0..self.num_layers {
            let bgs = &self.layer_bgs[l];

            // A. QKV GEMV
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_qkv, &[]);
                dispatch_gemv_tiled(&mut cpass, total_qkv, self.rows_per_wg);
            }

            // B. RoPE & KV-Cache Append
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.rope_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rope, &[]);
                cpass.dispatch_workgroups((self.num_heads + self.num_kv_heads) as u32, 1, 1);
            }

            // C. Decode Attention
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.attn_pipeline);
                cpass.set_bind_group(0, &bgs.bg_attn, &[]);
                cpass.dispatch_workgroups(self.num_heads as u32, 1, 1);
            }

            // D. Out GEMV
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_o, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }

            // E. Fused Attn Residual Add + Post-RMSNorm
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.fused_add_rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_fused_add_rmsnorm_attn, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // F. Fused Gate + Up GEMV + SwiGLU
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_swiglu_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_swiglu, &[]);
                dispatch_gemv_tiled(&mut cpass, intermediate_size, self.rows_per_wg);
            }

            // G. Down GEMV
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_down, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }

            // H. Fused MLP Residual Add + Next Layer Pre-RMSNorm (or Final RMSNorm for l == 23)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipelines.fused_add_rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_fused_add_rmsnorm_mlp, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
        }

        // Final RMSNorm is already fused into layer 23's MLP step!

        // LM Head GEMV
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.gemv_pipeline);
            cpass.set_bind_group(0, &self.bg_gemv_lm_head, &[]);
            dispatch_gemv_tiled(&mut cpass, vocab_size, self.rows_per_wg);
        }

        // Copy logits to staging buffer for CPU readback
        encoder.copy_buffer_to_buffer(&self.logits_buf, 0, &self.staging_buf, 0, (vocab_size * 4) as u64);

        // 4. Submit once to Vulkan / WGPU
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Executes all 24 layers of the transformer model in a single GPU command stream.
    /// Exactly 1 hardware sync / poll per token.
    pub fn step(&mut self, token_id: usize, offset: usize) -> PyResult<Vec<f32>> {
        self.record_and_submit_step(token_id, offset)?;

        let vocab_size = self.vocab_size;
        let buffer_slice = self.staging_buf.slice(..(vocab_size * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        {
            let view = buffer_slice.get_mapped_range();
            let f32_data: &[f32] = unsafe {
                std::slice::from_raw_parts(view.as_ptr() as *const f32, vocab_size)
            };
            self.logits_cpu.copy_from_slice(f32_data);
        }
        self.staging_buf.unmap();

        Ok(self.logits_cpu.clone())
    }

    pub fn profile_step_breakdown(&mut self, token_id: usize, offset: usize) -> PyResult<(f64, f64, f64)> {
        let hidden_size = self.hidden_size;
        let total_qkv = (self.num_heads * self.head_dim) + 2 * (self.num_kv_heads * self.head_dim);
        let intermediate_size = self.intermediate_size;
        let vocab_size = self.vocab_size;

        // 1. Upload token embedding to x_buf
        let emb_start = token_id * hidden_size;
        let emb_slice = &self.embed_tokens[emb_start..emb_start + hidden_size];
        let emb_bytes = unsafe { std::slice::from_raw_parts(emb_slice.as_ptr() as *const u8, hidden_size * 4) };
        self.queue.write_buffer(&self.x_buf, 0, emb_bytes);

        // 2. Uniforms
        let rope_data: [u32; 8] = [
            offset as u32,
            self.num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            1u32,
            0, 0,
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
            0, 0,
        ];
        self.queue.write_buffer(&self.attn_params_buf, 0, unsafe {
            std::slice::from_raw_parts(attn_data.as_ptr() as *const u8, 32)
        });

        // 3. Measure 24 layers
        let t0 = std::time::Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Layer 0 starts with RMSNorm on x_buf (which contains embed_tokens)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.pipelines.rmsnorm_pipeline);
            cpass.set_bind_group(0, &self.layer_bgs[0].bg_rmsnorm_in, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        for l in 0..self.num_layers {
            let bgs = &self.layer_bgs[l];
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_qkv, &[]);
                dispatch_gemv_tiled(&mut cpass, total_qkv, self.rows_per_wg);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.rope_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rope, &[]);
                cpass.dispatch_workgroups((self.num_heads + self.num_kv_heads) as u32, 1, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.attn_pipeline);
                cpass.set_bind_group(0, &bgs.bg_attn, &[]);
                cpass.dispatch_workgroups(self.num_heads as u32, 1, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_o, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.fused_add_rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_fused_add_rmsnorm_attn, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_swiglu_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_swiglu, &[]);
                dispatch_gemv_tiled(&mut cpass, intermediate_size, self.rows_per_wg);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_down, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.fused_add_rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_fused_add_rmsnorm_mlp, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
        }
        // Final RMSNorm is already fused into layer 23's MLP step!
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
        let t_layers = t0.elapsed().as_secs_f64() * 1000.0;

        // 4. Measure LM Head
        let t1 = std::time::Instant::now();
        let mut encoder2 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.pipelines.gemv_pipeline);
            cpass.set_bind_group(0, &self.bg_gemv_lm_head, &[]);
            dispatch_gemv_tiled(&mut cpass, vocab_size, self.rows_per_wg);
        }
        encoder2.copy_buffer_to_buffer(&self.logits_buf, 0, &self.staging_buf, 0, (vocab_size * 4) as u64);
        self.queue.submit(Some(encoder2.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);
        let t_lm_head = t1.elapsed().as_secs_f64() * 1000.0;

        // 5. Measure readback
        let t2 = std::time::Instant::now();
        let buffer_slice = self.staging_buf.slice(..(vocab_size * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result); });
        let _ = self.device.poll(wgpu::PollType::Wait);
        let _ = rx.recv();
        {
            let _view = buffer_slice.get_mapped_range();
        }
        self.staging_buf.unmap();
        let t_readback = t2.elapsed().as_secs_f64() * 1000.0;

        Ok((t_layers, t_lm_head, t_readback))
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
                    k_upload[dst_offset..dst_offset + self.head_dim].copy_from_slice(&k_slice[src_offset..src_offset + self.head_dim]);
                    v_upload[dst_offset..dst_offset + self.head_dim].copy_from_slice(&v_slice[src_offset..src_offset + self.head_dim]);
                }
            }

            let k_bytes = unsafe { std::slice::from_raw_parts(k_upload.as_ptr() as *const u8, k_upload.len() * 4) };
            let v_bytes = unsafe { std::slice::from_raw_parts(v_upload.as_ptr() as *const u8, v_upload.len() * 4) };
            self.queue.write_buffer(&self.layer_k_caches[l], 0, k_bytes);
            self.queue.write_buffer(&self.layer_v_caches[l], 0, v_bytes);
        }
        Ok(())
    }

    #[pyo3(signature = (token_id, offset, temperature=0.7, top_k=40, repetition_penalty=1.0, recent_tokens=None))]
    pub fn decode_and_sample(
        &mut self,
        token_id: usize,
        offset: usize,
        temperature: f32,
        top_k: usize,
        repetition_penalty: f32,
        recent_tokens: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.record_and_submit_step(token_id, offset)?;

        let vocab_size = self.vocab_size;
        let buffer_slice = self.staging_buf.slice(..(vocab_size * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let token = {
            let view = buffer_slice.get_mapped_range();
            let f32_data: &[f32] = unsafe {
                std::slice::from_raw_parts(view.as_ptr() as *const f32, vocab_size)
            };
            if repetition_penalty > 1.0 && recent_tokens.is_some() {
                let mut logits_copy = f32_data.to_vec();
                let mut seen = std::collections::HashSet::new();
                for t in recent_tokens.unwrap() {
                    if t < vocab_size && seen.insert(t) {
                        let l = logits_copy[t];
                        if l > 0.0 {
                            logits_copy[t] = l / repetition_penalty;
                        } else {
                            logits_copy[t] = l * repetition_penalty;
                        }
                    }
                }
                crate::quantization::sample_logits(&logits_copy, temperature, top_k)
            } else {
                crate::quantization::sample_logits(f32_data, temperature, top_k)
            }
        };
        self.staging_buf.unmap();

        Ok(token)
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

        // 1. Upload token embedding to x_buf
        let emb_start = token_id * hidden_size;
        let emb_slice = &self.embed_tokens[emb_start..emb_start + hidden_size];
        let emb_bytes = unsafe { std::slice::from_raw_parts(emb_slice.as_ptr() as *const u8, hidden_size * 4) };
        self.queue.write_buffer(&self.x_buf, 0, emb_bytes);
        map.insert("embed".to_string(), self.read_buffer(&self.x_buf, hidden_size)?);

        // 2. Uniforms
        let rope_data: [u32; 8] = [
            offset as u32,
            self.num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            1u32,
            0, 0,
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
            0, 0,
        ];
        self.queue.write_buffer(&self.attn_params_buf, 0, unsafe {
            std::slice::from_raw_parts(attn_data.as_ptr() as *const u8, 32)
        });

        let bgs = &self.layer_bgs[0];

        // A. Pre-RMSNorm
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rmsnorm_in, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("normed".to_string(), self.read_buffer(&self.normed_buf, hidden_size)?);
        }

        // B. QKV GEMV
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_qkv, &[]);
                dispatch_gemv_tiled(&mut cpass, total_qkv, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("qkv".to_string(), self.read_buffer(&self.qkv_buf, total_qkv)?);
        }

        // C. RoPE & KV-Cache Append
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.rope_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rope, &[]);
                cpass.dispatch_workgroups((self.num_heads + self.num_kv_heads) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("q".to_string(), self.read_buffer(&self.q_buf, self.num_heads * self.head_dim)?);
        }

        // D. Decode Attention
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.attn_pipeline);
                cpass.set_bind_group(0, &bgs.bg_attn, &[]);
                cpass.dispatch_workgroups(self.num_heads as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("attn_out".to_string(), self.read_buffer(&self.attn_out, hidden_size)?);
        }

        // E. Out GEMV
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_o, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("o_out".to_string(), self.read_buffer(&self.o_out, hidden_size)?);
        }

        // F. Residual Add (Attn)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.residual_pipeline);
                cpass.set_bind_group(0, &bgs.bg_residual_attn, &[]);
                cpass.dispatch_workgroups(((hidden_size + 63) / 64) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("x_attn".to_string(), self.read_buffer(&self.x_buf, hidden_size)?);
        }

        // G. Post-RMSNorm
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.rmsnorm_pipeline);
                cpass.set_bind_group(0, &bgs.bg_rmsnorm_post, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("normed_post".to_string(), self.read_buffer(&self.normed_buf, hidden_size)?);
        }

        // H & I. Gate & Up GEMV
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_gate, &[]);
                dispatch_gemv_tiled(&mut cpass, intermediate_size, self.rows_per_wg);

                cpass.set_bind_group(0, &bgs.bg_gemv_up, &[]);
                dispatch_gemv_tiled(&mut cpass, intermediate_size, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("gate".to_string(), self.read_buffer(&self.gate_buf, intermediate_size)?);
            map.insert("up".to_string(), self.read_buffer(&self.up_buf, intermediate_size)?);
        }

        // J. SwiGLU
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.swiglu_pipeline);
                cpass.set_bind_group(0, &bgs.bg_swiglu, &[]);
                cpass.dispatch_workgroups(((intermediate_size + 63) / 64) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("mlp_act".to_string(), self.read_buffer(&self.mlp_act, intermediate_size)?);
        }

        // K. Down GEMV
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.gemv_pipeline);
                cpass.set_bind_group(0, &bgs.bg_gemv_down, &[]);
                dispatch_gemv_tiled(&mut cpass, hidden_size, self.rows_per_wg);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("down_out".to_string(), self.read_buffer(&self.down_out, hidden_size)?);
        }

        // L. Residual Add (MLP)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&self.pipelines.residual_pipeline);
                cpass.set_bind_group(0, &bgs.bg_residual_mlp, &[]);
                cpass.dispatch_workgroups(((hidden_size + 63) / 64) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            map.insert("x_final_layer0".to_string(), self.read_buffer(&self.x_buf, hidden_size)?);
        }

        Ok(map)
    }
}
