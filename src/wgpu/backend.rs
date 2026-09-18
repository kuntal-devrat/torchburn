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
        "DirectX 12"
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
    let s = std::env::var("TORCHBURN_WGPU_BACKEND")
        .or_else(|_| std::env::var("TORCHBURN_DEVICE"))
        .ok()?
        .to_lowercase();
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
    if force_cpu() {
        return false;
    }
    GPU_INFO.get_or_init(probe_gpu).available
}

/// Get detailed GPU information as a tuple:
/// (available, adapter_name, backend_name, vram_bytes).
pub fn gpu_info() -> (bool, String, String, u64) {
    if force_cpu() {
        return (
            false,
            "CPU forced via TORCHBURN_DEVICE=cpu".to_string(),
            "none".to_string(),
            0,
        );
    }
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
    matches!(
        device_override().as_deref(),
        Some("cpu") | Some("native_cpu")
    )
}

/// Should we force GPU execution (fail if unavailable)?
pub fn force_gpu() -> bool {
    matches!(
        device_override().as_deref(),
        Some("gpu")
            | Some("wgpu")
            | Some("burn-wgpu")
            | Some("burn_gpu")
            | Some("auto")
            | Some("igpu")
            | Some("dgpu")
            | Some("metal")
            | Some("vulkan")
            | Some("dx12")
    )
}

/// Ensure the CubeCL shader cache directory exists.  Runs exactly once
/// per process via `OnceLock`; the `set_var` is wrapped in `unsafe` because
/// Rust ≥1.83 treats environment mutation as unsafe (other threads may read).
/// Safety: this is called inside `init_wgpu_runtime`'s own `OnceLock::get_or_init`,
/// so it executes at most once before any wgpu work begins.
fn ensure_shader_cache() {
    static SHADER_CACHE_INIT: OnceLock<()> = OnceLock::new();
    SHADER_CACHE_INIT.get_or_init(|| {
        if std::env::var("CUBECL_CACHE_DIR").is_err() {
            let cache_dir = std::env::temp_dir().join("torchburn_shader_cache");
            if let Err(e) = std::fs::create_dir_all(&cache_dir) {
                eprintln!("torchburn: failed to create shader cache dir: {e}");
            }
            // SAFETY: executed exactly once via OnceLock before any wgpu work;
            // no concurrent readers of CUBECL_CACHE_DIR at this point.
            unsafe { std::env::set_var("CUBECL_CACHE_DIR", cache_dir) };
        }
    });
}

#[cfg(feature = "burn-wgpu")]
pub fn init_wgpu_runtime() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ensure_shader_cache();
            let device = <Backend as BurnBackend>::Device::default();
            burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                &device,
                burn::backend::wgpu::RuntimeOptions {
                    tasks_max: 32,
                    memory_config: burn::backend::wgpu::MemoryConfiguration::ExclusivePages,
                },
            );
        }));
    });
}

fn probe_gpu() -> GPUInfo {
    // Try to enumerate adapters via wgpu crate for real info (when available).
    #[cfg(feature = "burn-wgpu")]
    {
        if let Some(info) = probe_via_wgpu() {
            init_wgpu_runtime();
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

/// Minimal async executor for wgpu adapter/device requests.
/// Uses a noop waker + yield loop — wgpu's adapter/device futures resolve
/// after the driver completes its internal work, which `yield_now` allows.
/// A bounded iteration count prevents infinite loops on broken drivers.
/// Uses progressive backoff (0→1→10→100μs) to avoid wasting time on fast ops.
#[cfg(feature = "burn-wgpu")]
fn block_on<F: std::future::Future>(mut future: F) -> Option<F::Output> {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop_clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    fn noop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);
    let raw_waker = RawWaker::new(std::ptr::null(), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw_waker) };
    let mut cx = Context::from_waker(&waker);
    let mut pinned = unsafe { Pin::new_unchecked(&mut future) };
    // Progressive backoff: yield → 1µs → 10µs → 100µs → 1000µs (1ms).
    // B4 fix: use threshold-based tier selection instead of `iter.min(4)`
    // which jumped to 1ms at iteration 5 instead of 1000.
    const BACKOFF_US: [u64; 5] = [0, 1, 10, 100, 1000];
    for iter in 0..100_000u32 {
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return Some(val),
            Poll::Pending => {
                let tier = if iter < 1 {
                    0
                } else if iter < 10 {
                    1
                } else if iter < 100 {
                    2
                } else if iter < 1000 {
                    3
                } else {
                    4
                };
                let delay_us = BACKOFF_US[tier];
                if delay_us > 0 {
                    std::thread::sleep(std::time::Duration::from_micros(delay_us));
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
    // If we get here, the future never resolved — this is a driver bug.
    // Poll one last time and return None if still pending (prevents process crash).
    match pinned.as_mut().poll(&mut cx) {
        Poll::Ready(val) => Some(val),
        Poll::Pending => {
            eprintln!("torchburn: wgpu adapter/device request timed out after 10s");
            None
        }
    }
}

#[cfg(feature = "burn-wgpu")]
pub struct WgpuInt4Context {
    /// Shared device handle — consumers `Arc::clone()` this instead of
    /// `Device::clone()` to guarantee command ordering on the same queue.
    pub device: std::sync::Arc<wgpu::Device>,
    pub queue: std::sync::Arc<wgpu::Queue>,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub rows_per_wg: u32,
    /// Whether this device supports WGSL subgroup operations.
    /// True on Vulkan/Metal/DX12 adapters with `SUBGROUP` feature; false on
    /// software/CPU adapters. Shaders that use `subgroupAdd` etc. are only
    /// compiled and dispatched when this is true.
    pub has_subgroups: bool,
    /// Whether PIPELINE_CACHE was granted at device creation. Pipeline cache
    /// creation panics without it, so decoder pipelines must check this flag.
    pub has_pipeline_cache: bool,
    /// Flag indicating if the GPU device has entered an unrecoverable error
    /// or device-lost state.
    pub device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "burn-wgpu")]
static WGPU_INT4_CTX: OnceLock<Option<WgpuInt4Context>> = OnceLock::new();

#[cfg(feature = "burn-wgpu")]
pub fn get_wgpu_int4_context() -> Option<&'static WgpuInt4Context> {
    if force_cpu() {
        return None;
    }
    WGPU_INT4_CTX.get_or_init(|| {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
        });

        let pref_device = std::env::var("TORCHBURN_DEVICE").ok().map(|s| s.trim().to_lowercase());
        let forced_backend = forced_wgpu_backend();
        let adapter = if pref_device.is_some() || forced_backend.is_some() {
            let mut candidates: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
            if let Some(fb) = forced_backend {
                candidates.retain(|a| a.get_info().backend == fb);
            }
            let target_type = pref_device.as_deref().and_then(|pref| match pref {
                "dgpu" => Some(wgpu::DeviceType::DiscreteGpu),
                "igpu" => Some(wgpu::DeviceType::IntegratedGpu),
                _ => None,
            });
            let type_rank = |t: wgpu::DeviceType| match t {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Cpu => 3,
                wgpu::DeviceType::Other => 4,
            };
            candidates.sort_by_key(|a| type_rank(a.get_info().device_type));
            if let Some(dev_type) = target_type {
                candidates.into_iter().find(|a| a.get_info().device_type == dev_type)
            } else {
                candidates.into_iter().next()
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
            }))?
            .ok()?,
        };

        let info = adapter.get_info();
        let default_rows = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 8u32,
            wgpu::DeviceType::IntegratedGpu => 4u32,
            _ => 4u32,
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

        // Several GPU buffers can exceed wgpu's conservative defaults: storage
        // buffers (weight matrices on large models, up to 128 MB default
        // binding size) and any single buffer above 256 MB. Raise the limits
        // we depend on to what this adapter actually supports (never beyond).
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_compute_workgroup_storage_size: adapter_limits.max_compute_workgroup_storage_size,
            ..Default::default()
        };

        // Probe for subgroup + pipeline-cache support — available on
        // Vulkan/Metal/DX12 GPU adapters but not on software/CPU fallbacks.
        // PIPELINE_CACHE must be requested at device creation or every
        // create_pipeline_cache call panics with Validation Error.
        let has_subgroups = adapter
            .features()
            .contains(wgpu::Features::SUBGROUP);
        let has_pipeline_cache = adapter.features().contains(
            wgpu::Features::from_bits_retain(
                wgpu::Features::PIPELINE_CACHE.bits(),
            ),
        );
        let mut optional_features = wgpu::Features::empty();
        if has_subgroups {
            optional_features |= wgpu::Features::SUBGROUP;
        }
        if has_pipeline_cache {
            optional_features |= wgpu::Features::from_bits_retain(
                wgpu::Features::PIPELINE_CACHE.bits(),
            );
        }

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("TorchBurn INT4 Vulkan Device"),
                required_features: optional_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ))?
        .ok()?;

        let shader_raw = include_str!("../shaders/gemv_w4a32.wgsl");
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

        let device_lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dl = device_lost.clone();
        device.on_uncaptured_error(Box::new(move |error| {
            eprintln!("torchburn: uncaptured wgpu error (device loss / OOM): {error}");
            dl.store(true, std::sync::atomic::Ordering::SeqCst);
        }));

        Some(WgpuInt4Context {
            device: std::sync::Arc::new(device),
            queue: std::sync::Arc::new(queue),
            pipeline,
            bind_group_layout,
            rows_per_wg,
            has_subgroups,
            has_pipeline_cache,
            device_lost,
        })
    }).as_ref()
}

#[cfg(feature = "burn-wgpu")]
struct PersistentWeightBuffers {
    x_buf: wgpu::Buffer,
    w_buf: wgpu::Buffer,
    s_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    num_rows: usize,
    num_cols: usize,
    group_size: usize,
    /// Cheap strided content fingerprint of the packed weights. If a tensor is
    /// freed and a *different* tensor is later allocated at the same address
    /// (the hazard of pointer-keyed caches), the fingerprint changes and the
    /// buffers are transparently re-uploaded instead of silently reusing stale
    /// weights.
    tag: u64,
    /// Monotonic counter used for LRU eviction.  Updated on every access
    /// (not just insertion) so frequently-used weights survive eviction.
    /// Uses AtomicU64 so we can update through an Arc without reconstructing.
    seq: std::sync::atomic::AtomicU64,
    /// Execution mutex to ensure serialized dispatch and readback when multiple
    /// threads invoke GEMV on the same weight matrix with GIL released.
    exec_lock: std::sync::Mutex<()>,
}

#[cfg(feature = "burn-wgpu")]
static PERSISTENT_WEIGHTS: OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<PersistentWeightBuffers>>>,
> = OnceLock::new();

#[cfg(feature = "burn-wgpu")]
static WEIGHT_CACHE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Maximum number of distinct weight matrices cached on the GPU. Configurable
/// via `TORCHBURN_WEIGHT_CACHE_CAP` env var. Default 256 (large models like
/// Qwen-7B have ~200 unique weight matrices; the previous cap of 64 caused
/// thrashing on every decode step).
#[cfg(feature = "burn-wgpu")]
fn weight_cache_cap() -> usize {
    std::env::var("TORCHBURN_WEIGHT_CACHE_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(256)
}

/// Cheap, deterministic fingerprint over `data`: FNV-1a over a strided sample
/// (>=64 words) plus the first and last bytes. Deliberately not a cryptographic
/// hash — the cost stays ~O(1) per call — while still catching the realistic
/// "buffer reused with different weights" collisions pointer keys can cause.
#[cfg(feature = "burn-wgpu")]
fn content_tag(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    let n = data.len();
    if n == 0 {
        return h;
    }
    // Sample 256 strided positions (up from 64) for stronger collision resistance
    // when multiple weight matrices are hot-swapped at the same memory address.
    let step = (n / 256).max(1);
    let mut i = 0usize;
    while i < n {
        h ^= data[i] as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += step;
    }
    // Fold the tail too, so equal-length tensors whose only differences are at
    // the very end of the buffer are still distinguished.
    let mut j = 0usize;
    while j < 16 && j < n {
        h ^= data[n - 1 - j] as u64;
        h = h.wrapping_mul(0x100000001b3);
        j += 1;
    }
    h
}

/// Thread-safe GPU buffer pool that caches and reuses wgpu buffers across
/// dispatches to reduce driver allocation overhead.
#[cfg(feature = "burn-wgpu")]
pub struct WgpuBufferPool {
    pool: std::sync::Mutex<std::collections::HashMap<(u64, wgpu::BufferUsages), Vec<wgpu::Buffer>>>,
    total_cached: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "burn-wgpu")]
impl Default for WgpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "burn-wgpu")]
impl WgpuBufferPool {
    pub fn new() -> Self {
        Self {
            pool: std::sync::Mutex::new(std::collections::HashMap::new()),
            total_cached: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Round buffer size up to reduce churn across nearby sizes.
    fn round_size(size: u64) -> u64 {
        if size <= 256 {
            256
        } else if size <= 65536 {
            size.next_power_of_two()
        } else {
            (size + 65535) & !65535
        }
    }

    /// Acquire a buffer of at least `size` bytes with the specified `usage`.
    /// Reuses a previously recycled buffer if available; otherwise allocates a new one.
    pub fn acquire(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> wgpu::Buffer {
        let rounded = Self::round_size(size);
        let key = (rounded, usage);
        if let Ok(mut map) = self.pool.lock() {
            if let Some(bufs) = map.get_mut(&key) {
                if let Some(buf) = bufs.pop() {
                    self.total_cached
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    return buf;
                }
            }
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: rounded,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Recycle a buffer back into the pool for future reuse.
    pub fn recycle(&self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        // Limit total pool capacity to 128 buffers to avoid excessive VRAM retention.
        if self.total_cached.load(std::sync::atomic::Ordering::Relaxed) >= 128 {
            return;
        }
        let rounded = Self::round_size(size);
        let key = (rounded, usage);
        if let Ok(mut map) = self.pool.lock() {
            let entry = map.entry(key).or_default();
            if entry.len() < 16 {
                entry.push(buffer);
                self.total_cached
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Clear all pooled buffers, releasing GPU driver memory.
    pub fn clear(&self) {
        if let Ok(mut map) = self.pool.lock() {
            map.clear();
            self.total_cached
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(feature = "burn-wgpu")]
static WGPU_BUFFER_POOL: OnceLock<WgpuBufferPool> = OnceLock::new();

#[cfg(feature = "burn-wgpu")]
pub fn get_wgpu_buffer_pool() -> &'static WgpuBufferPool {
    WGPU_BUFFER_POOL.get_or_init(WgpuBufferPool::new)
}

#[cfg(feature = "burn-wgpu")]
pub fn wgpu_clear_buffer_pool() {
    if let Some(pool) = WGPU_BUFFER_POOL.get() {
        pool.clear();
    }
}

#[cfg(feature = "burn-wgpu")]
fn get_persistent_weights() -> &'static std::sync::Mutex<
    std::collections::HashMap<usize, std::sync::Arc<PersistentWeightBuffers>>,
> {
    PERSISTENT_WEIGHTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[cfg(feature = "burn-wgpu")]
pub fn wgpu_clear_weight_cache() {
    if let Some(mutex) = PERSISTENT_WEIGHTS.get() {
        if let Ok(mut map) = mutex.lock() {
            map.clear();
        }
    }
    wgpu_clear_buffer_pool();
}

/// Dispatches an INT4 quantized GEMV operation to the WGPU compute queue.
///
/// Wrapped in `catch_unwind` so GPU device loss, validation errors, or OOM
/// conditions fail cleanly with an `Err` instead of terminating the process.
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
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        wgpu_gemv_w4a32_inner(x, w_bytes, scales, out, num_rows, num_cols, group_size)
    }));
    match result {
        Ok(res) => res,
        Err(payload) => {
            if let Some(ctx) = get_wgpu_int4_context() {
                ctx.device_lost.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "wgpu_gemv_w4a32 panicked (GPU device loss, validation error, or OOM)".to_string()
            };
            eprintln!("torchburn: wgpu GEMV panic caught ({msg}), falling back to CPU");
            Err(format!("wgpu panic: {msg}"))
        }
    }
}

#[cfg(feature = "burn-wgpu")]
fn wgpu_gemv_w4a32_inner(
    x: &[f32],
    w_bytes: &[u8],
    scales: &[f32],
    out: &mut [f32],
    num_rows: usize,
    num_cols: usize,
    group_size: usize,
) -> Result<(), String> {
    let ctx = get_wgpu_int4_context().ok_or_else(|| "WGPU device unavailable".to_string())?;
    if ctx.device_lost.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("WGPU device is in a lost/error state; falling back to CPU".to_string());
    }
    let device = &ctx.device;
    let queue = &ctx.queue;

    let num_groups = (num_cols + group_size - 1) / group_size;
    let key = w_bytes.as_ptr() as usize;
    let tag = content_tag(w_bytes);

    // ── Phase 1: Acquire weight buffers (lock held briefly) ────────────────
    // Clone the Arc to release the Mutex *before* any GPU work, so concurrent
    // GEMV calls on other weights are not blocked during submission/readback.
    let p: std::sync::Arc<PersistentWeightBuffers> =
        {
            let mut weight_map = get_persistent_weights().lock().map_err(|e| e.to_string())?;
            let stale = weight_map.get(&key).map_or(true, |e| {
                e.num_rows != num_rows
                    || e.num_cols != num_cols
                    || e.group_size != group_size
                    || e.tag != tag
            });
            if stale {
                // Allocate and populate persistent weight buffers ONCE on GPU
                // (re-uploaded when dims or content change at the same address).
                let pool = get_wgpu_buffer_pool();
                let w_size = ((w_bytes.len() + 3) & !3).max(16) as u64;
                let w_buf = pool.acquire(
                    device,
                    w_size,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    Some("wgpu_persistent_w_buf"),
                );
                queue.write_buffer(&w_buf, 0, w_bytes);

                let s_size = ((scales.len() * 4).max(16)) as u64;
                let s_buf = pool.acquire(
                    device,
                    s_size,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    Some("wgpu_persistent_s_buf"),
                );
                let s_bytes = unsafe {
                    std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
                };
                queue.write_buffer(&s_buf, 0, s_bytes);

                let y_size = ((num_rows * 4).max(16)) as u64;
                let y_buf = pool.acquire(
                    device,
                    y_size,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    Some("wgpu_persistent_y_buf"),
                );

                let params: [u32; 4] = [
                    num_rows as u32,
                    num_cols as u32,
                    group_size as u32,
                    num_groups as u32,
                ];
                let params_buf = pool.acquire(
                    device,
                    16,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    Some("wgpu_persistent_params_buf"),
                );
                let params_bytes =
                    unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, 16) };
                queue.write_buffer(&params_buf, 0, params_bytes);

                let staging_buf = pool.acquire(
                    device,
                    y_size,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    Some("wgpu_persistent_staging_buf"),
                );

                let x_size = (((num_cols * 4 + 15) & !15).max(16)) as u64;
                let x_buf = pool.acquire(
                    device,
                    x_size,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    Some("wgpu_persistent_x_buf"),
                );

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wgpu_gemv_bg"),
                    layout: &ctx.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: w_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: s_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: y_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let seq = WEIGHT_CACHE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let recycle_entry = |pool: &WgpuBufferPool, entry: PersistentWeightBuffers| {
                    let y_size = ((entry.num_rows * 4).max(16)) as u64;
                    let x_size = (((entry.num_cols * 4 + 15) & !15).max(16)) as u64;
                    let w_size = ((entry.num_rows * (entry.num_cols / 2) + 3) & !3).max(16) as u64;
                    let s_size = (((entry.num_rows * entry.num_cols / entry.group_size) * 4).max(16)) as u64;
                    pool.recycle(
                        entry.staging_buf,
                        y_size,
                        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    );
                    pool.recycle(
                        entry.y_buf,
                        y_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    );
                    pool.recycle(
                        entry.x_buf,
                        x_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    pool.recycle(
                        entry.w_buf,
                        w_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    pool.recycle(
                        entry.s_buf,
                        s_size,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    pool.recycle(
                        entry.params_buf,
                        16,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    );
                };

                let old_entry = weight_map.insert(
                    key,
                    std::sync::Arc::new(PersistentWeightBuffers {
                        x_buf,
                        w_buf,
                        s_buf,
                        params_buf,
                        y_buf,
                        staging_buf,
                        bind_group,
                        num_rows,
                        num_cols,
                        group_size,
                        tag,
                        seq: std::sync::atomic::AtomicU64::new(seq),
                        exec_lock: std::sync::Mutex::new(()),
                    }),
                );
                if let Some(evicted) = old_entry {
                    if let Ok(entry) = std::sync::Arc::try_unwrap(evicted) {
                        recycle_entry(pool, entry);
                    }
                }

                // Evict the least-recently-used entries once the cache exceeds its bound.
                while weight_map.len() > weight_cache_cap() {
                    if let Some((oldest_key, oldest_seq)) = weight_map
                        .iter()
                        .min_by_key(|(_, e)| e.seq.load(std::sync::atomic::Ordering::Relaxed))
                        .map(|(k, e)| (*k, e.seq.load(std::sync::atomic::Ordering::Relaxed)))
                    {
                        if oldest_seq == seq {
                            break; // never evict the entry we just inserted
                        }
                        if let Some(evicted) = weight_map.remove(&oldest_key) {
                            if let Ok(entry) = std::sync::Arc::try_unwrap(evicted) {
                                recycle_entry(pool, entry);
                            }
                        }
                    } else {
                        break;
                    }
                }
            } else {
                // Cache hit — update the access counter for LRU eviction.
                // AtomicU64 allows in-place update through the Arc.
                if let Some(entry) = weight_map.get(&key) {
                    let new_seq =
                        WEIGHT_CACHE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    entry
                        .seq
                        .store(new_seq, std::sync::atomic::Ordering::Relaxed);
                }
            }

            weight_map.get(&key).ok_or_else(|| {
            "wgpu: weight buffer was evicted from cache during the same call (this is a bug)"
                .to_string()
        })?.clone() // Arc clone — cheap reference count bump
        }; // ← Mutex dropped here, BEFORE any GPU work

    // ── Phase 2: GPU dispatch (lock-free across distinct weights, serialized per-weight) ──
    let _exec_guard = p.exec_lock.lock().unwrap_or_else(|e| e.into_inner());
    let x_bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, num_cols * 4) };
    queue.write_buffer(&p.x_buf, 0, x_bytes);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_gemv_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_gemv_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline);
        cpass.set_bind_group(0, &p.bind_group, &[]);
        let wgs = (num_rows as u32 + ctx.rows_per_wg - 1) / ctx.rows_per_wg;
        let dispatch_x = wgs.min(65535);
        let dispatch_y = (wgs + 65534) / 65535;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    encoder.copy_buffer_to_buffer(&p.y_buf, 0, &p.staging_buf, 0, (num_rows * 4) as u64);
    queue.submit(Some(encoder.finish()));

    // ── Phase 3: Readback (lock-free) ──────────────────────────────────────
    let buffer_slice = p.staging_buf.slice(..(num_rows * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;

    {
        let view = buffer_slice.get_mapped_range();
        let f32_data: &[f32] =
            unsafe { std::slice::from_raw_parts(view.as_ptr() as *const f32, num_rows) };
        out.copy_from_slice(f32_data);
    }
    p.staging_buf.unmap();

    Ok(())
}
