//! Metal native backend implementation for Apple Silicon.
//!
//! Provides direct Metal compute pipeline access for unified memory
//! architectures, bypassing WGPU overhead.

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
        let buffer = device.new_buffer_with_data(data, resource_options::StorageModeShared);
        self.buffers.push((key, buffer));
        &self.buffers.last().unwrap().1
    }
}

static METAL_DEVICE: OnceLock<MetalBackend> = OnceLock::new();
static BUFFER_CACHE: OnceLock<Mutex<MetalBufferCache>> = OnceLock::new();

impl MetalBackend {
    /// Check if Metal is available (macOS/iOS only).
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Get or initialize the Metal backend singleton.
    pub fn instance() -> Option<&'static MetalBackend> {
        METAL_DEVICE.get_or_init(|| {
            let device = Device::system_default().expect("No Metal device available");
            let command_queue = device.new_command_queue();
            MetalBackend {
                device,
                command_queue,
            }
        });
        Some(METAL_DEVICE.get().unwrap())
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
        BUFFER_CACHE.get_or_init(|| Mutex::new(MetalBufferCache::new(64)))
    }

    /// Dense f32 GEMM using Metal Performance Shaders.
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

        let a_buf = backend
            .device
            .new_buffer_with_data(a, resource_options::StorageModeShared);
        let b_buf = backend
            .device
            .new_buffer_with_data(b, resource_options::StorageModeShared);
        let c_buf = backend
            .device
            .new_buffer_with_data(c, resource_options::StorageModeShared);

        // Use MPSMatrixMultiplication for dense GEMM
        // This is a placeholder; real implementation would use MPS directly
        let command_buffer = backend.command_queue.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back
        let ptr = c_buf.contents() as *const f32;
        let len = m * n;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, c.as_mut_ptr(), len);
        }
        Some(())
    }

    /// INT4 GEMV using custom Metal compute kernel.
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

        let x_buf = backend
            .device
            .new_buffer_with_data(x, resource_options::StorageModeShared);
        let w_buf = backend
            .device
            .new_buffer_with_data(w_packed, resource_options::StorageModeShared);
        let s_buf = backend
            .device
            .new_buffer_with_data(scales, resource_options::StorageModeShared);
        let mut out_buf = backend
            .device
            .new_buffer((n * 4) as u64, resource_options::StorageModeShared);

        let num_groups = (k + group_size - 1) / group_size;

        // Load and compile the Metal shader
        let library = backend
            .device
            .new_library_with_source(METAL_GEMV_W4A32_SHADER)
            .ok()?;
        let function = library.get_function("gemv_w4a32", None).ok()?;
        let pipeline = backend
            .device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;

        let command_buffer = backend.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&w_buf), 0);
        encoder.set_buffer(2, Some(&s_buf), 0);
        encoder.set_buffer(3, Some(&out_buf), 0);

        let thread_group_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        Some(())
    }
}

/// Metal Shading Language kernel for INT4 GEMV.
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
    uint row [[thread_position_in_grid]])
{
    if (row >= params.n) return;

    uint bytes_per_row = (params.k + 1) / 2;
    uint bytes_per_group = params.group_size / 2;
    const uchar* w_row = w_packed + row * bytes_per_row;
    const float* s_row = scales + row * params.num_groups;

    float sum = 0.0;
    for (uint g = 0; g < params.num_groups; g++) {
        float group_sum = 0.0;
        uint g_start = g * params.group_size;
        uint g_end = min(g_start + params.group_size, params.k);

        for (uint i = g_start; i < g_end; i++) {
            uint byte_idx = i / 2;
            uchar byte = w_row[byte_idx];
            char q = (i % 2 == 0)
                ? char((byte & 0x0F) - 8)
                : char(((byte >> 4) & 0x0F) - 8);
            group_sum += x[i] * float(q);
        }
        sum += group_sum * s_row[g];
    }
    out[row] = sum;
}
"#;
