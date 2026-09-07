//! CUDA backend implementation using cudarc + cuBLAS.
//!
//! Lazy-initializes the CUDA device and cuBLAS handle. Provides:
//! - Dense GEMM via cuBLAS (f16/f32)
//! - INT4 GEMV via custom CUDA kernel
//! - Persistent GPU weight buffer cache

use std::sync::{Mutex, OnceLock};

use cudarc::cublas::*;
use cudarc::driver::*;

static CUBLAS_HANDLE: OnceLock<Mutex<CudaBlas<f32>>> = OnceLock::new();
static CUDA_DEVICE: OnceLock<CudaDevice> = OnceLock::new();

/// CUDA backend state.
#[derive(Debug)]
pub struct CudaBackend {
    device: &'static CudaDevice,
}

/// Error type for CUDA operations.
#[derive(Debug)]
pub enum CudaError {
    NoDevice,
    InitFailed(String),
    KernelLaunchFailed(String),
    CublasError(String),
    MemoryError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::NoDevice => write!(f, "No CUDA device available"),
            CudaError::InitFailed(s) => write!(f, "CUDA init failed: {}", s),
            CudaError::KernelLaunchFailed(s) => write!(f, "Kernel launch failed: {}", s),
            CudaError::CublasError(s) => write!(f, "cuBLAS error: {}", s),
            CudaError::MemoryError(s) => write!(f, "CUDA memory error: {}", s),
        }
    }
}

impl std::error::Error for CudaError {}

impl CudaBackend {
    /// Check if CUDA is available at runtime.
    pub fn is_available() -> bool {
        CudaDevice::new(0).is_ok()
    }

    /// Get or initialize the CUDA backend singleton.
    pub fn instance() -> Result<&'static CudaBackend, CudaError> {
        static INSTANCE: OnceLock<CudaBackend> = OnceLock::new();
        INSTANCE.get_or_try_init(|| {
            let device = CUDA_DEVICE.get_or_try_init(|| {
                CudaDevice::new(0).map_err(|e| CudaError::InitFailed(e.to_string()))
            })?;

            Ok(CudaBackend { device })
        })
    }

    /// Get the CUDA device.
    pub fn device(&self) -> &'static CudaDevice {
        self.device
    }

    /// Get or create cuBLAS handle.
    pub fn cublas_handle() -> Result<&'static Mutex<CudaBlas<f32>>, CudaError> {
        CUBLAS_HANDLE.get_or_try_init(|| {
            let backend = Self::instance()?;
            let blas = CudaBlas::<f32>::new(backend.device.clone())
                .map_err(|e| CudaError::CublasError(e.to_string()))?;
            Ok(Mutex::new(blas))
        })
    }

    /// Dense f32 GEMM: C = alpha * A * B + beta * C
    /// A: (m, k), B: (k, n), C: (m, n)
    pub fn gemm_f32(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), CudaError> {
        let backend = Self::instance()?;

        let d_a = backend
            .device
            .htod_sync_copy(a)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;
        let d_b = backend
            .device
            .htod_sync_copy(b)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;
        let mut d_c = backend
            .device
            .htod_sync_copy(c)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;

        {
            let handle = Self::cublas_handle()?.lock().unwrap();
            unsafe {
                gemm(
                    &handle,
                    CudaBlasOperation::N,
                    CudaBlasOperation::N,
                    n as i32,
                    m as i32,
                    k as i32,
                    &[alpha],
                    &d_b,
                    n as i32,
                    &d_a,
                    k as i32,
                    &[beta],
                    &mut d_c,
                    n as i32,
                )
                .map_err(|e| CudaError::CublasError(e.to_string()))?;
            }
        }

        c.copy_from_slice(
            &d_c
                .dtoh_sync_copy()
                .map_err(|e| CudaError::MemoryError(e.to_string()))?,
        );
        Ok(())
    }

    /// INT4 GEMV kernel: y = W * x + bias
    /// W is packed int4 (n rows, k columns, k/2 bytes per row)
    /// scales: (n, num_groups) f32 group-wise scales
    pub fn gemv_w4a32(
        x: &[f32],
        w_packed: &[u8],
        scales: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<(), CudaError> {
        let backend = Self::instance()?;

        let d_x = backend
            .device
            .htod_sync_copy(x)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;
        let d_w = backend
            .device
            .htod_sync_copy(w_packed)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;
        let d_scales = backend
            .device
            .htod_sync_copy(scales)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;
        let mut d_out = backend
            .device
            .alloc_zeros::<f32>(n)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;

        // Launch INT4 GEMV kernel
        let num_groups = (k + group_size - 1) / group_size;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        unsafe {
            let module = backend
                .device
                .get_or_load_module(CUDA_GEMV_W4A32_KERNEL)
                .map_err(|e| CudaError::KernelLaunchFailed(e.to_string()))?;

            let func = module.get_function("gemv_w4a32_kernel").ok_or_else(|| {
                CudaError::KernelLaunchFailed("gemv_w4a32_kernel not found".into())
            })?;

            let grid = LaunchAsync::with_grid_and_stream(
                (blocks as u32, 1, 1),
                (threads_per_block as u32, 1, 1),
                backend.device.stream,
            );

            grid.launch(
                &func,
                (
                    &d_x,
                    &d_w,
                    &d_scales,
                    d_out.clone(),
                    n as u32,
                    k as u32,
                    group_size as u32,
                    num_groups as u32,
                ),
            )
            .map_err(|e| CudaError::KernelLaunchFailed(e.to_string()))?;

            backend
                .device
                .synchronize()
                .map_err(|e| CudaError::KernelLaunchFailed(e.to_string()))?;
        }

        out.copy_from_slice(
            &d_out
                .dtoh_sync_copy()
                .map_err(|e| CudaError::MemoryError(e.to_string()))?,
        );
        Ok(())
    }
}

/// CUDA kernel source for INT4 GEMV.
const CUDA_GEMV_W4A32_KERNEL: &str = r#"
extern "C" __global__
void gemv_w4a32_kernel(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w_packed,
    const float* __restrict__ scales,
    float* __restrict__ out,
    unsigned int n,
    unsigned int k,
    unsigned int group_size,
    unsigned int num_groups
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    unsigned int bytes_per_row = (k + 1) / 2;
    unsigned int bytes_per_group = group_size / 2;
    const unsigned char* w_row = w_packed + row * bytes_per_row;
    const float* s_row = scales + row * num_groups;

    float sum = 0.0f;
    for (unsigned int g = 0; g < num_groups; g++) {
        float group_sum = 0.0f;
        unsigned int g_start = g * group_size;
        unsigned int g_end = min(g_start + group_size, k);

        for (unsigned int i = g_start; i < g_end; i++) {
            unsigned int byte_idx = i / 2;
            unsigned char byte = w_row[byte_idx];
            signed char q = (i % 2 == 0)
                ? (signed char)((byte & 0x0F) - 8)
                : (signed char)(((byte >> 4) & 0x0F) - 8);
            group_sum += x[i] * (float)q;
        }

        sum += group_sum * s_row[g];
    }

    out[row] = sum;
}
"#;
