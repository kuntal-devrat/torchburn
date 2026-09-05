//! Wgpu GPU backend for the Burn engine (feature `burn-wgpu`).
//!
//! `Wgpu` is Burn's cross-platform GPU backend built on top of wgpu.  It
//! executes compute shaders through the native graphics API of the host:
//!
//! * **Metal** on macOS / iOS (Apple Silicon and Intel Macs),
//! * **Vulkan** on Windows, Linux and Android (NVIDIA / AMD / Intel GPUs),
//! * **DirectX 12** on Windows via the same wgpu stack,
//! * **WebGPU** in wasm environments.
//!
//! The `WgpuDevice::default()` (best available adapter) is used, so no device
//! selection is needed — wgpu picks the highest-power GPU it can drive.
//!
//! CPU fallback: when no adapter can be created (headless CI, VMs, machines
//! without a GPU driver), the probe here reports unavailability and the burn
//! engine falls back to the pure-CPU `NdArray` backend.

use burn::backend::Wgpu;
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Tensor, TensorData};

/// The wgpu-backed Burn backend used by the engine.
pub type Backend = Wgpu;

use std::sync::OnceLock;

/// Cached GPU info: (available, adapter_name, backend_name, vram_bytes).
static GPU_INFO: OnceLock<GPUInfo> = OnceLock::new();

struct GPUInfo {
    available: bool,
    adapter_name: String,
    backend_name: String,
    vram_bytes: u64,
}

/// Backend type string for the active wgpu graphics API (fallback).
fn wgpu_backend_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "Metal"
    }
    #[cfg(target_os = "windows")]
    {
        "Vulkan"
    }
    #[cfg(target_os = "linux")]
    {
        "Vulkan"
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        "WebGPU"
    }
}

/// Map wgpu Backend enum to string (when wgpu crate is available).
#[cfg(feature = "burn-wgpu")]
fn backend_to_str(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DirectX 12",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        _ => "Unknown",
    }
}

/// Check if user forced a specific wgpu backend via TORCHBURN_WGPU_BACKEND.
#[cfg(feature = "burn-wgpu")]
fn forced_wgpu_backend() -> Option<wgpu::Backend> {
    let s = std::env::var("TORCHBURN_WGPU_BACKEND").ok()?.to_lowercase();
    match s.as_str() {
        "vulkan" => Some(wgpu::Backend::Vulkan),
        "metal" => Some(wgpu::Backend::Metal),
        "dx12" | "dx12-12" | "d3d12" => Some(wgpu::Backend::Dx12),
        "gl" | "opengl" => Some(wgpu::Backend::Gl),
        "webgpu" => Some(wgpu::Backend::BrowserWebGpu),
        _ => None,
    }
}

/// Lazily probe whether a GPU adapter is available and collect its info.
///
/// This creates a 1-element tensor on the default device and reads it back,
/// which forces wgpu's adapter/device initialization.  If no adapter exists,
/// wgpu panics; we catch that and report unavailability so the caller can
/// fall back to the CPU backend instead of crashing.
pub fn gpu_available() -> bool {
    GPU_INFO.get_or_init(probe_gpu).available
}

/// Get detailed GPU information as a tuple:
/// (available, adapter_name, backend_name, vram_bytes).
pub fn gpu_info() -> (bool, String, String, u64) {
    let info = GPU_INFO.get_or_init(probe_gpu);
    (
        info.available,
        info.adapter_name.clone(),
        info.backend_name.clone(),
        info.vram_bytes,
    )
}

/// Check if the user has requested a specific device via env var.
pub fn device_override() -> Option<String> {
    std::env::var("TORCHBURN_DEVICE")
        .ok()
        .map(|s| s.to_lowercase())
}

/// Should we force CPU execution?
pub fn force_cpu() -> bool {
    matches!(device_override().as_deref(), Some("cpu"))
}

/// Should we force GPU execution (fail if unavailable)?
pub fn force_gpu() -> bool {
    matches!(
        device_override().as_deref(),
        Some("gpu") | Some("auto") | Some("cuda") | Some("metal") | Some("vulkan")
    )
}

fn ensure_shader_cache() {
    if std::env::var("CUBECL_CACHE_DIR").is_err() {
        let cache_dir = std::env::temp_dir().join("torchburn_shader_cache");
        let _ = std::fs::create_dir_all(&cache_dir);
        std::env::set_var("CUBECL_CACHE_DIR", cache_dir);
    }
}

#[cfg(feature = "burn-wgpu")]
pub fn init_wgpu_runtime() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        ensure_shader_cache();
        let device = <Backend as BurnBackend>::Device::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
            &device,
            burn::backend::wgpu::RuntimeOptions {
                tasks_max: 32,
                memory_config: burn::backend::wgpu::MemoryConfiguration::ExclusivePages,
            },
        );
    });
}

fn probe_gpu() -> GPUInfo {
    // First, respect explicit CPU override.
    if force_cpu() {
        return GPUInfo {
            available: false,
            adapter_name: "CPU forced via TORCHBURN_DEVICE=cpu".to_string(),
            backend_name: "none".to_string(),
            vram_bytes: 0,
        };
    }

    // Try to enumerate adapters via wgpu crate for real info (when available).
    #[cfg(feature = "burn-wgpu")]
    {
        init_wgpu_runtime();
        if let Some(info) = probe_via_wgpu() {
            // Validate that Burn can actually create a device (adapter not just enumerated).
            let burn_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let device = <Backend as BurnBackend>::Device::default();
                let t: Tensor<Backend, 1> =
                    Tensor::from_data(TensorData::new(vec![1.0f32], vec![1]), &device);
                let _ = t.into_data();
            }))
            .is_ok();
            if burn_ok {
                return info;
            }
        }
    }

    // Fallback: try Burn tensor creation (works without direct wgpu dep).
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = <Backend as BurnBackend>::Device::default();
        let t: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(vec![1.0f32], vec![1]), &device);
        let _ = t.into_data();
    }));

    match result {
        Ok(()) => GPUInfo {
            available: true,
            adapter_name: "GPU adapter detected (Burn)".to_string(),
            backend_name: wgpu_backend_name().to_string(),
            vram_bytes: 0,
        },
        Err(_) => GPUInfo {
            available: false,
            adapter_name: "No GPU adapter found".to_string(),
            backend_name: "none".to_string(),
            vram_bytes: 0,
        },
    }
}

#[cfg(feature = "burn-wgpu")]
fn probe_via_wgpu() -> Option<GPUInfo> {
    // Create an instance that can enumerate all backends.
    // In wgpu 25, Instance::new takes &InstanceDescriptor.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
    });

    // Optionally filter by forced backend.
    let forced = forced_wgpu_backend();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    if adapters.is_empty() {
        return None;
    }

    // Prefer discrete GPU, then integrated, then others.
    // If forced backend is set, filter to that backend.
    let mut best: Option<(wgpu::Adapter, wgpu::AdapterInfo)> = None;
    let mut best_score = -1i32;
    for adapter in adapters {
        let info = adapter.get_info();
        if let Some(fb) = forced {
            if info.backend != fb {
                continue;
            }
        }
        // Score: DiscreteGpu=3, IntegratedGpu=2, Cpu=1, Other=0
        let score = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 3,
            wgpu::DeviceType::IntegratedGpu => 2,
            wgpu::DeviceType::Cpu => 1,
            _ => 0,
        };
        if score > best_score {
            best_score = score;
            best = Some((adapter, info));
        }
    }

    best.map(|(_, info)| {
        // Try to estimate VRAM: wgpu doesn't expose, so 0. Could query via limits in future.
        GPUInfo {
            available: true,
            adapter_name: info.name.clone(),
            backend_name: backend_to_str(info.backend).to_string(),
            vram_bytes: 0,
        }
    })
}

#[cfg(feature = "burn-wgpu")]
fn block_on<F: std::future::Future>(mut future: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop_clone(_: *const ()) -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) }
    fn noop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);
    let raw_waker = RawWaker::new(std::ptr::null(), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw_waker) };
    let mut cx = Context::from_waker(&waker);
    let mut pinned = unsafe { Pin::new_unchecked(&mut future) };
    loop {
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[cfg(feature = "burn-wgpu")]
pub struct WgpuInt4Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub rows_per_wg: u32,
}

#[cfg(feature = "burn-wgpu")]
static WGPU_INT4_CTX: OnceLock<Option<WgpuInt4Context>> = OnceLock::new();

#[cfg(feature = "burn-wgpu")]
pub fn get_wgpu_int4_context() -> Option<&'static WgpuInt4Context> {
    WGPU_INT4_CTX.get_or_init(|| {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
        });

        let pref_device = std::env::var("TORCHBURN_DEVICE").ok().map(|s| s.trim().to_lowercase());
        let adapter = if let Some(ref pref) = pref_device {
            let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
            let target_type = match pref.as_str() {
                "dgpu" => Some(wgpu::DeviceType::DiscreteGpu),
                "igpu" => Some(wgpu::DeviceType::IntegratedGpu),
                _ => None,
            };
            if let Some(dev_type) = target_type {
                adapters.into_iter().find(|a| a.get_info().device_type == dev_type)
            } else {
                None
            }
        } else {
            None
        };

        let adapter = match adapter {
            Some(a) => a,
            None => block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if pref_device.as_deref() == Some("igpu") {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                compatible_surface: None,
                force_fallback_adapter: false,
            })).ok()?,
        };

        let info = adapter.get_info();
        let default_rows = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 8u32,
            wgpu::DeviceType::IntegratedGpu => 2u32,
            _ => 2u32,
        };
        let rows_per_wg = std::env::var("TORCHBURN_ROWS_PER_WG")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(default_rows);
        let wg_size = rows_per_wg * 16;
        eprintln!(
            "[TorchBurn WGPU] Adapter: '{}' ({:?}, backend: {:?}) | Selected {} rows/workgroup ({} threads)",
            info.name, info.device_type, info.backend, rows_per_wg, wg_size
        );

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("TorchBurn INT4 Vulkan Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        )).ok()?;

        let shader_raw = include_str!("shaders/gemv_w4a32.wgsl");
        let shader_src = shader_raw
            .replace("const ROWS_PER_WG: u32 = 4u;", &format!("const ROWS_PER_WG: u32 = {}u;", rows_per_wg))
            .replace("const WG_SIZE: u32 = 64u;", &format!("const WG_SIZE: u32 = {}u;", wg_size));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv_w4a32.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemv_w4a32_bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemv_w4a32_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemv_w4a32_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Some(WgpuInt4Context {
            device,
            queue,
            pipeline,
            bind_group_layout,
            rows_per_wg,
        })
    }).as_ref()
}

#[cfg(feature = "burn-wgpu")]
struct PersistentWeightBuffers {
    w_buf: wgpu::Buffer,
    s_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    num_rows: usize,
    num_cols: usize,
}

#[cfg(feature = "burn-wgpu")]
static PERSISTENT_WEIGHTS: OnceLock<std::sync::Mutex<std::collections::HashMap<usize, PersistentWeightBuffers>>> = OnceLock::new();

#[cfg(feature = "burn-wgpu")]
fn get_persistent_weights() -> &'static std::sync::Mutex<std::collections::HashMap<usize, PersistentWeightBuffers>> {
    PERSISTENT_WEIGHTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[cfg(feature = "burn-wgpu")]
pub fn wgpu_clear_weight_cache() {
    if let Some(mutex) = PERSISTENT_WEIGHTS.get() {
        if let Ok(mut map) = mutex.lock() {
            map.clear();
        }
    }
}

#[cfg(feature = "burn-wgpu")]
pub fn wgpu_gemv_w4a32(
    x: &[f32],
    w_bytes: &[u8],
    scales: &[f32],
    out: &mut [f32],
    num_rows: usize,
    num_cols: usize,
    group_size: usize,
) -> Result<(), String> {
    let ctx = get_wgpu_int4_context().ok_or_else(|| "WGPU device unavailable".to_string())?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    let num_groups = (num_cols + group_size - 1) / group_size;
    let key = w_bytes.as_ptr() as usize;

    let mut weight_map = get_persistent_weights().lock().map_err(|e| e.to_string())?;
    if !weight_map.contains_key(&key) || weight_map.get(&key).map(|e| e.num_rows != num_rows || e.num_cols != num_cols).unwrap_or(false) {
        // Allocate and populate persistent weight buffers ONCE on GPU
        let w_size = ((w_bytes.len() + 3) & !3).max(16) as u64;
        let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_persistent_w_buf"),
            size: w_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&w_buf, 0, w_bytes);

        let s_size = ((scales.len() * 4).max(16)) as u64;
        let s_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_persistent_s_buf"),
            size: s_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let s_bytes = unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4) };
        queue.write_buffer(&s_buf, 0, s_bytes);

        let y_size = ((num_rows * 4).max(16)) as u64;
        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_persistent_y_buf"),
            size: y_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params: [u32; 4] = [num_rows as u32, num_cols as u32, group_size as u32, num_groups as u32];
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_persistent_params_buf"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bytes = unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, 16) };
        queue.write_buffer(&params_buf, 0, params_bytes);

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_persistent_staging_buf"),
            size: y_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        weight_map.insert(key, PersistentWeightBuffers {
            w_buf,
            s_buf,
            params_buf,
            y_buf,
            staging_buf,
            num_rows,
            num_cols,
        });
    }

    let p = weight_map.get(&key).unwrap();

    // 1. x buffer: only 3.5 KB upload per projection instead of 260 MB!
    let x_size = (((num_cols * 4 + 15) & !15).max(16)) as u64;
    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_x_buf"),
        size: x_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let x_bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, num_cols * 4) };
    queue.write_buffer(&x_buf, 0, x_bytes);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wgpu_gemv_bg"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: p.w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: p.s_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: p.y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: p.params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_gemv_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_gemv_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wgs = (num_rows as u32 + ctx.rows_per_wg - 1) / ctx.rows_per_wg;
        let dispatch_x = wgs.min(65535);
        let dispatch_y = (wgs + 65534) / 65535;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    encoder.copy_buffer_to_buffer(&p.y_buf, 0, &p.staging_buf, 0, (num_rows * 4) as u64);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = p.staging_buf.slice(..(num_rows * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;

    {
        let view = buffer_slice.get_mapped_range();
        let f32_data: &[f32] = unsafe {
            std::slice::from_raw_parts(view.as_ptr() as *const f32, num_rows)
        };
        out.copy_from_slice(f32_data);
    }
    p.staging_buf.unmap();

    Ok(())
}

