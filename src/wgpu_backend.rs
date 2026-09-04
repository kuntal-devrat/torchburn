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
