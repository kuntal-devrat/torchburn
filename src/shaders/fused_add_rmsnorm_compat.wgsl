// Fused residual-add + RMSNorm compat fallback — no subgroup builtins.
//
// Same interface as fused_add_rmsnorm.wgsl (bindings 0..4, AddRMSNormParams{n, eps},
// @workgroup_size(64)) but uses a classic workgroup-memory barrier-tree
// reduction so it compiles on adapters without WGSL subgroup support
// (software/CPU fallbacks, older Vulkan drivers, WebGPU targets).
// Peak-path: vec4 loads/stores (4x fewer instructions than scalar).

struct AddRMSNormParams {
    n:   u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read_write> x:        array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       residual: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       weight:   array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y:        array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             params:   AddRMSNormParams;

var<workgroup> partial:   array<f32, 64>;
var<workgroup> s_inv_rms: f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n   = params.n;
    let n4  = n / 4u;

    // ── Phase 1: fused residual add + sum-of-squares (vec4) ───────────────
    var sum_sq = 0.0f;
    var i = tid;
    while (i < n4) {
        let val = x[i] + residual[i];
        x[i]   = val;              // in-place residual add
        sum_sq = sum_sq + dot(val, val);
        i      = i + 64u;
    }
    let tail_start = n4 * 4u;
    var j = tail_start + tid;
    while (j < n) {
        let vi  = j / 4u;
        let ei  = j % 4u;
        let val = x[vi][ei] + residual[vi][ei];
        x[vi][ei] = val;
        sum_sq    = sum_sq + val * val;
        j         = j + 64u;
    }
    partial[tid] = sum_sq;
    workgroupBarrier();

    // ── Phase 2: barrier-tree reduction 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 ──
    if (tid < 32u) { partial[tid] = partial[tid] + partial[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { partial[tid] = partial[tid] + partial[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { partial[tid] = partial[tid] + partial[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { partial[tid] = partial[tid] + partial[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { partial[tid] = partial[tid] + partial[tid + 2u]; }
    workgroupBarrier();
    if (tid < 1u) { partial[tid] = partial[tid] + partial[tid + 1u]; }
    workgroupBarrier();

    if (tid == 0u) {
        let mean  = partial[0] / f32(n);
        s_inv_rms = inverseSqrt(mean + params.eps);
    }
    workgroupBarrier();
    storageBarrier();   // ensure x writes from phase 1 are visible before phase 3

    // ── Phase 3: normalize + scale (vec4 stores) ──────────────────────────
    let inv_rms = s_inv_rms;
    i = tid;
    while (i < n4) {
        y[i] = x[i] * inv_rms * weight[i];
        i    = i + 64u;
    }
    j = tail_start + tid;
    while (j < n) {
        let vi = j / 4u;
        let ei = j % 4u;
        y[vi][ei] = x[vi][ei] * inv_rms * weight[vi][ei];
        j = j + 64u;
    }
}
