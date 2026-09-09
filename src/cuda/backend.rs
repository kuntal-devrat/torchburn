//! CUDA backend implementation using cudarc + cuBLAS.
//!
//! Lazy-initializes the CUDA device and cuBLAS handle. Provides:
//! - Dense GEMM via cuBLAS (f16/f32)
//! - INT4 GEMV via custom CUDA kernel with shared memory tiling
//! - Persistent GPU weight buffer cache (upload once, reuse forever)

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::*;
use cudarc::driver::*;
use cudarc::nvrtc::compile_ptx;

// cudarc 0.13 API notes (see docs.rs/cudarc/0.13):
// - `htod_sync_copy` / `dtoh_sync_copy` / `alloc_zeros` / `synchronize` are
//   `CudaDevice` methods (receiver `&Arc<Self>`), not `CudaSlice` methods.
// - Kernels: `compile_ptx` (nvrtc) -> `dev.load_ptx(ptx, name, &[fns])` ->
//   `dev.get_func(name, func)` -> `func.launch(LaunchConfig{..}, params)` via
//   the `LaunchAsync` trait. Slices are passed by reference (`&CudaSlice`);
//   `CudaSlice::clone()` is a device-to-device deep copy, never an alias.
// - GEMM: `Gemm` trait method `handle.gemm(GemmConfig{..}, a, b, c)` with
//   `cublasOperation_t::CUBLAS_OP_N/P`.

/// Module name under which the INT4 kernels are registered on the device.
const GEMV_MODULE: &str = "tb_gemv_w4a32";

static CUDA_DEVICE: OnceLock<Result<Arc<CudaDevice>, String>> = OnceLock::new();
static CUBLAS_HANDLE: OnceLock<Result<Mutex<CudaBlas>, String>> = OnceLock::new();

/// Persistent GPU buffer cache: weights are uploaded once and cached by a
/// content hash. Repeated inference calls skip the H2D transfer entirely.
static BUFFER_CACHE: OnceLock<Mutex<GpuBufferCache>> = OnceLock::new();

struct GpuBufferCache {
    /// Map from content hash -> GPU buffer, reference-counted so callers
    /// hold an owned `Arc` instead of a borrow into the pool (which would
    /// conflict with later `&mut` pool calls in the same function).
    f32_buffers: HashMap<u64, Arc<CudaSlice<f32>>>,
    u8_buffers: HashMap<u64, Arc<CudaSlice<u8>>>,
    /// Reusable scratch buffers keyed by size (in elements).
    scratch_f32: Vec<(usize, CudaSlice<f32>)>,
}

impl GpuBufferCache {
    fn new() -> Self {
        Self {
            f32_buffers: HashMap::new(),
            u8_buffers: HashMap::new(),
            scratch_f32: Vec::new(),
        }
    }

    /// Get or upload a f32 buffer. Uses a simple FNV-1a hash of the data.
    /// Returns an owned `Arc` clone so no borrow into `self` escapes.
    fn get_or_upload_f32(
        &mut self,
        data: &[f32],
        dev: &Arc<CudaDevice>,
    ) -> Result<Arc<CudaSlice<f32>>, CudaError> {
        let hash = fnv_hash(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        });
        if !self.f32_buffers.contains_key(&hash) {
            let buf = dev
                .htod_sync_copy(data)
                .map_err(|e| CudaError::MemoryError(e.to_string()))?;
            self.f32_buffers.insert(hash, Arc::new(buf));
        }
        Ok(self.f32_buffers[&hash].clone())
    }

    /// Get or upload a u8 buffer (for packed weights).
    /// Returns an owned `Arc` clone so no borrow into `self` escapes.
    fn get_or_upload_u8(
        &mut self,
        data: &[u8],
        dev: &Arc<CudaDevice>,
    ) -> Result<Arc<CudaSlice<u8>>, CudaError> {
        let hash = fnv_hash(data);
        if !self.u8_buffers.contains_key(&hash) {
            let buf = dev
                .htod_sync_copy(data)
                .map_err(|e| CudaError::MemoryError(e.to_string()))?;
            self.u8_buffers.insert(hash, Arc::new(buf));
        }
        Ok(self.u8_buffers[&hash].clone())
    }

    /// Get a scratch buffer of at least `n` f32 elements (reused across calls).
    fn scratch_f32(
        &mut self,
        n: usize,
        dev: &Arc<CudaDevice>,
    ) -> Result<CudaSlice<f32>, CudaError> {
        // Find best-fit existing scratch buffer
        let mut best = None;
        let mut best_cap = usize::MAX;
        for (i, (cap, _)) in self.scratch_f32.iter().enumerate() {
            if *cap >= n && *cap < best_cap {
                best = Some(i);
                best_cap = *cap;
            }
        }
        if let Some(idx) = best {
            return Ok(self.scratch_f32.remove(idx).1);
        }
        // Allocate new
        dev.alloc_zeros::<f32>(n)
            .map_err(|e| CudaError::MemoryError(e.to_string()))
    }

    /// Return a scratch buffer for reuse.
    fn return_scratch_f32(&mut self, buf: CudaSlice<f32>, cap: usize) {
        if self.scratch_f32.len() < 16 {
            self.scratch_f32.push((cap, buf));
        }
    }
}

/// FNV-1a hash for content-addressable GPU buffer cache.
fn fnv_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// CUDA backend state.
#[derive(Debug)]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
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
    /// Check if CUDA is available at runtime (cached; no fresh probe leak).
    pub fn is_available() -> bool {
        Self::get_device().is_ok()
    }

    /// Get or initialize the CUDA device singleton.
    ///
    /// `OnceLock::get_or_try_init` is avoided (unstable on some toolchains);
    /// the `Result` is stored in the cell so init failure is sticky and
    /// reported on every call without re-probing.
    fn device_id() -> usize {
        // Single-device today; TORCHBURN_CUDA_DEVICE selects id for future
        // multi-GPU (NCCL tensor-parallel). Falls back to 0 on parse failure.
        std::env::var("TORCHBURN_CUDA_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    }

    fn get_device() -> Result<&'static Arc<CudaDevice>, CudaError> {
        CUDA_DEVICE
            .get_or_init(|| CudaDevice::new(Self::device_id()).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| CudaError::InitFailed(e.clone()))
    }

    /// Get or create cuBLAS handle.
    fn cublas() -> Result<&'static Mutex<CudaBlas>, CudaError> {
        CUBLAS_HANDLE
            .get_or_init(|| {
                (|| {
                    let dev = Self::get_device().map_err(|e| e.to_string())?;
                    let blas = CudaBlas::new(dev.clone()).map_err(|e| e.to_string())?;
                    Ok::<_, String>(Mutex::new(blas))
                })()
            })
            .as_ref()
            .map_err(|e| CudaError::CublasError(e.clone()))
    }

    /// Compile (once) and register the INT4 kernels on the device.
    fn ensure_kernels(dev: &Arc<CudaDevice>) -> Result<(), CudaError> {
        if dev.has_func(GEMV_MODULE, "gemv_w4a32_tiled") {
            return Ok(());
        }
        let ptx = compile_ptx(CUDA_GEMV_W4A32_OPTIMIZED)
            .map_err(|e| CudaError::KernelLaunchFailed(e.to_string()))?;
        dev.load_ptx(ptx, GEMV_MODULE, &["gemv_w4a32_tiled"])
            .map_err(|e| CudaError::KernelLaunchFailed(e.to_string()))
    }

    /// Get or create the buffer cache.
    fn cache() -> &'static Mutex<GpuBufferCache> {
        BUFFER_CACHE.get_or_init(|| Mutex::new(GpuBufferCache::new()))
    }

    /// Dense f32 GEMM: C = alpha * A * B + beta * C
    /// A: (m, k), B: (k, n), C: (m, n)
    ///
    /// Uses persistent GPU buffer cache for weights that don't change between
    /// calls. Input activations are uploaded fresh each call (they change).
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
        let dev = Self::get_device()?;

        // Upload A (activations — changes each call)
        let d_a = dev
            .htod_sync_copy(a)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;

        // The pool guard stays alive across the launch: cached buffers are
        // borrowed from it, so it must outlive the kernel + readback.
        let mut cache = Self::cache().lock().unwrap_or_else(|e| e.into_inner());
        // Upload B (weights — cache for reuse)
        let d_b = cache.get_or_upload_f32(b, dev)?;

        // Get scratch buffer for output
        let mut d_c = cache.scratch_f32(m * n, dev)?;

        {
            let handle = Self::cublas()?.lock().unwrap_or_else(|e| e.into_inner());
            // Row-major C = A @ B via column-major C^T = B^T @ A^T:
            // swap A/B and m/n, keep leading dims of the row-major layouts.
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha,
                lda: n as i32,
                ldb: k as i32,
                beta,
                ldc: n as i32,
            };
            unsafe {
                // `Arc` does not implement the device traits: deref to the
                // `&CudaSlice` the generic params require.
                handle
                    .gemm(cfg, &*d_b, &d_a, &mut d_c)
                    .map_err(|e| CudaError::CublasError(e.to_string()))?;
            }
        }

        // Read back result
        let result: Vec<f32> = dev
            .dtoh_sync_copy(&d_c)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;
        c.copy_from_slice(&result);

        // Return scratch buffer to cache
        cache.return_scratch_f32(d_c, m * n);
        Ok(())
    }

    /// INT4 GEMV kernel: y = W * x + bias
    /// W is packed int4 (n rows, k columns, k/2 bytes per row)
    /// scales: (n, num_groups) f32 group-wise scales
    ///
    /// Uses a shared-memory-tiled CUDA kernel: each thread block handles
    /// one output row, with threads cooperating to reduce across the K
    /// dimension using shared memory and warp-level shuffles.
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
        let dev = Self::get_device()?;

        // Upload x (changes each call)
        let d_x = dev
            .htod_sync_copy(x)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;

        // Cache weights and scales (persistent). The pool guard stays alive
        // across the launch: cached buffers are borrowed from it.
        let mut cache = Self::cache().lock().unwrap_or_else(|e| e.into_inner());
        let d_w = cache.get_or_upload_u8(w_packed, dev)?;
        let d_scales = cache.get_or_upload_f32(scales, dev)?;
        let mut d_out = cache.scratch_f32(n, dev)?;

        Self::ensure_kernels(dev)?;

        let num_groups = (k + group_size - 1) / group_size;

        // Launch optimized kernel: 1 block per row, 256 threads per block,
        // on the device default stream.
        let threads_per_block = 256u32;
        let blocks = n as u32;

        let func = dev
            .get_func(GEMV_MODULE, "gemv_w4a32_tiled")
            .ok_or_else(|| CudaError::KernelLaunchFailed("gemv_w4a32_tiled not found".into()))?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            // NOTE: slices pass by reference (`&CudaSlice` is `DeviceRepr`;
            // `&&CudaSlice` is not, so already-borrowed cache buffers pass
            // through unchanged). `CudaSlice::clone()` is a device-to-device
            // deep copy and must NOT be used for the out param.
            func.launch(
                cfg,
                (
                    &d_x,
                    &*d_w,
                    &*d_scales,
                    &mut d_out,
                    n as u32,
                    k as u32,
                    group_size as u32,
                    num_groups as u32,
                ),
            )
            .map_err(|e| CudaError::KernelLaunchFailed(e.to_string()))?;

            dev.synchronize()
                .map_err(|e| CudaError::KernelLaunchFailed(e.to_string()))?;
        }

        let result: Vec<f32> = dev
            .dtoh_sync_copy(&d_out)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?;
        out.copy_from_slice(&result);

        // Apply bias on CPU (tiny cost vs. kernel launch overhead)
        if let Some(b) = bias {
            for (o, &bv) in out.iter_mut().zip(b.iter()) {
                *o += bv;
            }
        }

        // Return scratch buffer (the existing pool guard is still alive).
        cache.return_scratch_f32(d_out, n);
        Ok(())
    }

    /// Clear the persistent GPU buffer cache (useful for memory pressure).
    pub fn clear_cache() {
        if let Some(cache) = BUFFER_CACHE.get() {
            let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
            c.f32_buffers.clear();
            c.u8_buffers.clear();
            c.scratch_f32.clear();
        }
    }
}

/// Optimized CUDA kernel for INT4 GEMV with shared memory tiling and
/// warp-level reductions.
///
/// Each thread block handles one output row. The K dimension is partitioned
/// across 256 threads, each computing a partial dot product. Partial sums
/// are reduced via shared memory + warp shuffle.
///
/// Performance vs. naive kernel:
/// - Shared memory: warp-level reduction avoids global memory atomics
/// - Warp shuffle: __shfl_down_sync for final reduction (no bank conflicts)
/// - Strided access: each thread handles k/256 elements with stride 256
/// - Scale deferred: group scale applied per-element (fused multiply)
const CUDA_GEMV_W4A32_OPTIMIZED: &str = r#"
extern "C" __global__
void gemv_w4a32_tiled(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w_packed,
    const float* __restrict__ scales,
    float* __restrict__ out,
    unsigned int n,
    unsigned int k,
    unsigned int group_size,
    unsigned int num_groups
) {
    // One block per output row
    unsigned int row = blockIdx.x;
    if (row >= n) return;

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;

    // Shared memory for inter-warp reduction
    __shared__ float smem[8]; // max 256/32 = 8 warps

    unsigned int bytes_per_row = (k + 1) / 2;
    const unsigned char* w_row = w_packed + row * bytes_per_row;
    const float* s_row = scales + row * num_groups;

    float thread_sum = 0.0f;

    // Each thread handles strided elements across K
    for (unsigned int i = tid; i < k; i += blockSize) {
        // Load x from global (L1/L2 cached across threads)
        float xv = x[i];

        // Decode INT4 weight
        unsigned int byte_idx = i / 2;
        unsigned char byte_val = w_row[byte_idx];
        int q = (i % 2 == 0)
            ? (int)((byte_val & 0x0F)) - 8
            : (int)(((byte_val >> 4) & 0x0F)) - 8;

        // Get group scale and fuse into accumulation
        unsigned int g = i / group_size;
        float scale = s_row[g];

        thread_sum += xv * (float)q * scale;
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Inter-warp reduction via shared memory
    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;
    unsigned int num_warps = blockSize / 32;

    if (lane == 0) {
        smem[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            out[row] = val;
        }
    }
}
"#;
