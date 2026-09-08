//! Compute pipeline & bind-group-layout factory for the WGPU LLM decoder.
//!
//! Builds the seven compute pipelines (GEMV, fused gate+up GEMV + SwiGLU,
//! RMSNorm, RoPE + KV append, decode attention, SwiGLU, residual add, fused
//! residual + RMSNorm) from the WGSL shaders in `src/shaders/`, templating
//! the workgroup-layout constants (ROWS_PER_WG / WG_SIZE) at runtime.

pub(crate) struct WgpuPipelines {
    pub(crate) gemv_pipeline: wgpu::ComputePipeline,
    pub(crate) gemv_bgl: wgpu::BindGroupLayout,
    pub(crate) gemv_swiglu_pipeline: wgpu::ComputePipeline,
    pub(crate) gemv_swiglu_bgl: wgpu::BindGroupLayout,
    pub(crate) rmsnorm_pipeline: wgpu::ComputePipeline,
    pub(crate) rmsnorm_bgl: wgpu::BindGroupLayout,
    pub(crate) rope_pipeline: wgpu::ComputePipeline,
    pub(crate) rope_bgl: wgpu::BindGroupLayout,
    pub(crate) attn_pipeline: wgpu::ComputePipeline,
    pub(crate) attn_bgl: wgpu::BindGroupLayout,
    pub(crate) swiglu_pipeline: wgpu::ComputePipeline,
    pub(crate) swiglu_bgl: wgpu::BindGroupLayout,
    pub(crate) residual_pipeline: wgpu::ComputePipeline,
    pub(crate) residual_bgl: wgpu::BindGroupLayout,
    pub(crate) fused_add_rmsnorm_pipeline: wgpu::ComputePipeline,
    pub(crate) fused_add_rmsnorm_bgl: wgpu::BindGroupLayout,
    /// Persistent embed-table lookup: avoids 3.5 KB host write_buffer per token.
    pub(crate) embed_lookup_pipeline: wgpu::ComputePipeline,
    pub(crate) embed_lookup_bgl: wgpu::BindGroupLayout,
    pub(crate) rows_per_wg: u32,
}

impl WgpuPipelines {
    pub(crate) fn new(device: &wgpu::Device, rows_per_wg: u32, has_subgroups: bool) -> Self {
        let wg_size = rows_per_wg * 16;
        let gemv_shader_raw = include_str!("../shaders/gemv_w4a32.wgsl");
        let gemv_shader_src = gemv_shader_raw
            .replace(
                "const ROWS_PER_WG: u32 = 4u;",
                &format!("const ROWS_PER_WG: u32 = {}u;", rows_per_wg),
            )
            .replace(
                "const WG_SIZE: u32 = 64u;",
                &format!("const WG_SIZE: u32 = {}u;", wg_size),
            );

        // 1. GEMV pipeline & layout
        let gemv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv_w4a32.wgsl"),
            source: wgpu::ShaderSource::Wgsl(gemv_shader_src.into()),
        });
        let gemv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemv_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
        let gemv_swiglu_raw = include_str!("../shaders/gemv_swiglu_w4a32.wgsl");
        let gemv_swiglu_src = gemv_swiglu_raw
            .replace(
                "const ROWS_PER_WG: u32 = 4u;",
                &format!("const ROWS_PER_WG: u32 = {}u;", rows_per_wg),
            )
            .replace(
                "const WG_SIZE: u32 = 64u;",
                &format!("const WG_SIZE: u32 = {}u;", wg_size),
            );
        let gemv_swiglu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv_swiglu_w4a32.wgsl"),
            source: wgpu::ShaderSource::Wgsl(gemv_swiglu_src.into()),
        });
        let gemv_swiglu_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemv_swiglu_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let gemv_swiglu_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemv_swiglu_layout"),
            bind_group_layouts: &[&gemv_swiglu_bgl],
            push_constant_ranges: &[],
        });
        let gemv_swiglu_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gemv_swiglu_pipeline"),
                layout: Some(&gemv_swiglu_layout),
                module: &gemv_swiglu_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // 2. RMSNorm pipeline & layout
        // Select the subgroup-optimised variant when the adapter supports it;
        // fall back to the barrier-tree variant otherwise.
        let rmsnorm_src = if has_subgroups {
            include_str!("../shaders/rmsnorm.wgsl")
        } else {
            include_str!("../shaders/rmsnorm_compat.wgsl")
        };
        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm.wgsl"),
            source: wgpu::ShaderSource::Wgsl(rmsnorm_src.into()),
        });
        let rmsnorm_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rmsnorm_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rope_append.wgsl").into()),
        });
        let rope_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rope_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/attn_decode.wgsl").into()),
        });
        let attn_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/swiglu.wgsl").into()),
        });
        let swiglu_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("swiglu_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/residual_add.wgsl").into()),
        });
        let residual_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("residual_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fused_add_rmsnorm.wgsl").into(),
            ),
        });
        let fused_add_rmsnorm_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fused_add_rmsnorm_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let fused_add_rmsnorm_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fused_add_rmsnorm_layout"),
                bind_group_layouts: &[&fused_add_rmsnorm_bgl],
                push_constant_ranges: &[],
            });
        let fused_add_rmsnorm_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fused_add_rmsnorm_pipeline"),
                layout: Some(&fused_add_rmsnorm_layout),
                module: &fused_add_rmsnorm_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // ── Embed lookup pipeline ───────────────────────────────────────────────
        // Binding layout: 0=embed_table(storage,read), 1=token_id(uniform),
        //                 2=x_buf(storage,rw),          3=params(uniform)
        let embed_lookup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed_lookup.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/embed_lookup.wgsl").into()),
        });
        let embed_lookup_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("embed_lookup_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let embed_lookup_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("embed_lookup_layout"),
            bind_group_layouts: &[&embed_lookup_bgl],
            push_constant_ranges: &[],
        });
        let embed_lookup_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("embed_lookup_pipeline"),
                layout: Some(&embed_lookup_layout),
                module: &embed_lookup_shader,
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
            embed_lookup_pipeline,
            embed_lookup_bgl,
            rows_per_wg,
        }
    }
}
