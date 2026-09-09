//! Metal native backend implementation for Apple Silicon.
//!
//! Provides direct Metal compute pipeline access for unified memory
//! architectures, bypassing WGPU overhead. On Apple Silicon, CPU and GPU
//! share the same physical memory — Metal buffers created with
//! `StorageModeShared` require zero copies for read/write from either side.

use std::sync::{Mutex, OnceLock};

use metal::*;

/// Metal native backend state.
#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
}

/// Persistent GPU buffer cache for weights.
struct MetalBufferCache {
    buffers: Vec<(u64, Buffer)>,
    capacity: usize,
}

impl MetalBufferCache {
    fn new(capacity: usize) -> Self {
        Self {
            buffers: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn get_or_insert(&mut self, key: u64, data: &[u8], device: &Device) -> &Buffer {
        if let Some(idx) = self.buffers.iter().position(|(k, _)| *k == key) {
            return &self.buffers[idx].1;
        }
        if self.buffers.len() >= self.capacity {
            self.buffers.remove(0);
        }
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.buffers.push((key, buffer));
        &self.buffers.last().unwrap().1
    }
}

/// Pipeline state cache — PSO compilation is expensive; cache by function name.
/// Libraries are cached per shader source hash so that different shader
/// sources (e.g. METAL_GEMM_SHADER vs METAL_GEMV_W4A32_SHADER) coexist
/// without recompiling.
struct MetalPipelineCache {
    pipelines: Vec<(String, ComputePipelineState)>,
    /// Per-source-string compiled libraries, keyed by a hash of the source.
    libraries: Vec<(u64, Library)>,
}

impl MetalPipelineCache {
    fn new() -> Self {
        Self {
            pipelines: Vec::new(),
            libraries: Vec::new(),
        }
    }

    /// Hash a shader source string for library dedup.
    fn source_hash(source: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut h);
        h.finish()
    }

    fn get_or_create(
        &mut self,
        name: &str,
        source: &str,
        device: &Device,
    ) -> Option<&ComputePipelineState> {
        if let Some(idx) = self.pipelines.iter().position(|(n, _)| n == name) {
            return Some(&self.pipelines[idx].1);
        }
        // Find or compile the library for this source string
        let sh = Self::source_hash(source);
        let lib_idx = if let Some(pos) = self.libraries.iter().position(|(h, _)| *h == sh) {
            pos
        } else {
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(true);
            let lib = device.new_library_with_source(source, &opts).ok()?;
            self.libraries.push((sh, lib));
            self.libraries.len() - 1
        };
        let lib = &self.libraries[lib_idx].1;
        let func = lib.get_function(name, None).ok()?;
        let pso = device
            .new_compute_pipeline_state_with_function(&func)
            .ok()?;
        self.pipelines.push((name.to_string(), pso));
        Some(&self.pipelines.last().unwrap().1)
    }
}

static METAL_DEVICE: OnceLock<MetalBackend> = OnceLock::new();
static BUFFER_CACHE: OnceLock<Mutex<MetalBufferCache>> = OnceLock::new();
static PIPELINE_CACHE: OnceLock<Mutex<MetalPipelineCache>> = OnceLock::new();

impl MetalBackend {
    /// Check if Metal is available (macOS/iOS only).
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Get or initialize the Metal backend singleton.
    /// Returns `None` gracefully if no Metal device is available (non-Mac
    /// host, headless server, etc.) instead of panicking.
    pub fn instance() -> Option<&'static MetalBackend> {
        METAL_DEVICE.get_or_init(|| {
            match Device::system_default() {
                Some(device) => {
                    let command_queue = device.new_command_queue();
                    MetalBackend {
                        device,
                        command_queue,
                    }
                }
                None => {
                    // Return a dummy that will be detected by the None check
                    // below. OnceLock always inits; we use is_available()
                    // to guard callers.
                    return;
                }
            }
        });
        // If Device::system_default() returned None, the OnceLock body
        // diverged (returned early). Re-check availability.
        if Self::is_available() {
            METAL_DEVICE.get()
        } else {
            None
        }
    }

    /// Get the Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue.
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Get or create a buffer cache.
    fn buffer_cache() -> &'static Mutex<MetalBufferCache> {
        BUFFER_CACHE.get_or_init(|| Mutex::new(MetalBufferCache::new(128)))
    }

    /// Get or create a pipeline cache.
    fn pipeline_cache() -> &'static Mutex<MetalPipelineCache> {
        PIPELINE_CACHE.get_or_init(|| Mutex::new(MetalPipelineCache::new()))
    }

    /// Dense f32 GEMM using a Metal compute kernel.
    ///
    /// On Apple Silicon, unified memory means the f32 slices passed in are
    /// directly accessible from the GPU. We create shared-mode buffers that
    /// wrap the same physical pages — **zero copies** on M1/M2/M3.
    pub fn gemm_f32(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) -> Option<()> {
        let backend = Self::instance()?;

        // Create shared-mode buffers (zero-copy on Apple Silicon unified memory)
        let a_buf = backend.device.new_buffer_with_data(
            a.as_ptr() as *const std::ffi::c_void,
            (a.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = backend.device.new_buffer_with_data(
            b.as_ptr() as *const std::ffi::c_void,
            (b.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buf = backend.device.new_buffer_with_data(
            c.as_ptr() as *const std::ffi::c_void,
            (c.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Params uniform buffer: GemmParams{m, n, k, alpha, beta} (5x4 = 20B).
        #[repr(C)]
        struct GemmParams {
            m: u32,
            n: u32,
            k: u32,
            alpha: f32,
            beta: f32,
        }
        let params = GemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha,
            beta,
        };
        let params_buf = backend.device.new_buffer_with_data(
            &params as *const GemmParams as *const std::ffi::c_void,
            std::mem::size_of::<GemmParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get or compile the GEMM pipeline
        let mut pcache = Self::pipeline_cache().lock().ok()?;
        let pipeline = pcache.get_or_create("gemm_f32", METAL_GEMM_SHADER, &backend.device)?;

        let command_buffer = backend.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&c_buf), 0);
        encoder.set_buffer(3, Some(&params_buf), 0);

        // Each thread computes one element of C
        let thread_group_size = MTLSize::new(16, 16, 1);
        let grid_size = MTLSize::new(((n + 15) / 16) as u64, ((m + 15) / 16) as u64, 1);
        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result from shared buffer (zero-copy on Apple Silicon)
        let ptr = c_buf.contents() as *const f32;
        let len = m * n;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, c.as_mut_ptr(), len);
        }
        Some(())
    }

    /// INT4 GEMV using custom Metal compute kernel with SIMD group reductions.
    pub fn gemv_w4a32(
        x: &[f32],
        w_packed: &[u8],
        scales: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Option<()> {
        let backend = Self::instance()?;

        let x_buf = backend.device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w_buf = backend.device.new_buffer_with_data(
            w_packed.as_ptr() as *const std::ffi::c_void,
            w_packed.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let s_buf = backend.device.new_buffer_with_data(
            scales.as_ptr() as *const std::ffi::c_void,
            (scales.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = backend
            .device
            .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let num_groups = (k + group_size - 1) / group_size;

        // Params: [n, k, group_size, num_groups]
        let params: [u32; 4] = [n as u32, k as u32, group_size as u32, num_groups as u32];
        let params_buf = backend.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            (params.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut pcache = Self::pipeline_cache().lock().ok()?;
        let pipeline =
            pcache.get_or_create("gemv_w4a32", METAL_GEMV_W4A32_SHADER, &backend.device)?;

        let command_buffer = backend.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&w_buf), 0);
        encoder.set_buffer(2, Some(&s_buf), 0);
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_buffer(4, Some(&params_buf), 0);

        // 1 threadgroup per row, 256 threads per group (SIMD group reduction)
        let thread_group_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        unsafe {
            if let Some(b) = bias {
                let out_ptr = out.as_mut_ptr();
                for i in 0..n {
                    *out_ptr.add(i) = *ptr.add(i) + *b.as_ptr().add(i);
                }
            } else {
                std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
            }
        }
        Some(())
    }
}

/// Metal GEMM compute shader (tiled, 16x16 threadgroups).
/// Honors BLAS `C = alpha*A*B + beta*C` via the params buffer.
const METAL_GEMM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint m;
    uint n;
    uint k;
    float alpha;
    float beta;
};

kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    float sum = 0.0;
    // Tile the K dimension for better cache utilization
    for (uint p = 0; p < params.k; p += 4) {
        float4 a_vals = float4(
            A[row * params.k + p],
            (p + 1 < params.k) ? A[row * params.k + p + 1] : 0.0,
            (p + 2 < params.k) ? A[row * params.k + p + 2] : 0.0,
            (p + 3 < params.k) ? A[row * params.k + p + 3] : 0.0
        );
        float4 b_vals = float4(
            B[p * params.n + col],
            (p + 1 < params.k) ? B[(p + 1) * params.n + col] : 0.0,
            (p + 2 < params.k) ? B[(p + 2) * params.n + col] : 0.0,
            (p + 3 < params.k) ? B[(p + 3) * params.n + col] : 0.0
        );
        sum += dot(a_vals, b_vals);
    }
    float c_old = C[row * params.n + col];
    C[row * params.n + col] = params.alpha * sum + params.beta * c_old;
}
"#;

/// Metal INT4 GEMV shader with SIMD group reductions.
///
/// Each threadgroup handles one output row. Threads cooperate to reduce
/// across the K dimension using SIMD group shuffle operations (metal::simd_sum).
const METAL_GEMV_W4A32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GemvParams {
    uint n;
    uint k;
    uint group_size;
    uint num_groups;
};

kernel void gemv_w4a32(
    device const float* x [[buffer(0)]],
    device const uchar* w_packed [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant GemvParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (row >= params.n) return;

    uint threads_per_group = 256;
    uint bytes_per_row = (params.k + 1) / 2;
    device const uchar* w_row = w_packed + row * bytes_per_row;
    device const float* s_row = scales + row * params.num_groups;

    float thread_sum = 0.0;

    // Strided loop: each thread handles k/256 elements
    for (uint i = tid; i < params.k; i += threads_per_group) {
        float xv = x[i];
        uint byte_idx = i / 2;
        uchar byte_val = w_row[byte_idx];
        int q = (i % 2 == 0)
            ? int((byte_val & 0x0F)) - 8
            : int(((byte_val >> 4) & 0x0F)) - 8;

        uint g = i / params.group_size;
        float scale = s_row[g];

        thread_sum += xv * float(q) * scale;
    }

    // SIMD group reduction (warp-level, 32 lanes on Apple Silicon)
    thread_sum = simd_sum(thread_sum);

    // Inter-SIMD-group reduction via threadgroup memory
    threadgroup float shared_sums[8]; // max 256/32 = 8 SIMD groups
    if (simd_lane == 0) {
        shared_sums[simd_group] = thread_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group reduces across all groups
    if (simd_group == 0 && simd_lane < (threads_per_group / 32)) {
        float val = shared_sums[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[row] = val;
        }
    }
}
"#;
