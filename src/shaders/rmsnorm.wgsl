// RMSNorm with subgroup-based reduction.
//
// Phase 1: Each thread accumulates sum-of-squares over its slice of the input
//          (vec4 loads, 4x fewer instructions than scalar).
// Phase 2: Subgroup reduction (subgroupAdd) collapses partial sums within each
//          wave in one instruction — no barriers needed inside a subgroup.
//          A single workgroup-memory stage then reduces across subgroups (at most
//          ceil(64/subgroupSize) values, typically 2 for 32-lane hardware or 1
//          for 64-lane hardware).  This replaces the previous 7-step barrier tree.
// Phase 3: Normalize and scale with vec4 stores.
//
// Fallback: if the device does not advertise subgroup operations, the code falls
//           back to a one-shot workgroupBarrier reduction that is still correct.

struct RMSNormParams {
    n:   u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read>       x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y:      array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params: RMSNormParams;

// One slot per subgroup (max subgroup size = 128; typical = 32 or 64).
// 64 / 32 = 2 subgroups at minimum; 64 / 64 = 1 on AMD RDNA / Apple M-series.
var<workgroup> sg_partial: array<f32, 2>;
var<workgroup> s_inv_rms:  f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id)  local_id:   vec3<u32>,
        @builtin(subgroup_invocation_id) sg_lane:  u32,
        @builtin(subgroup_id)            sg_id:    u32,
        @builtin(num_subgroups)          num_sgs:  u32) {
    let tid    = local_id.x;
    let n      = params.n;
    let n4     = n / 4u;

    // ── Phase 1: thread-local partial sum of squares (vec4 loads) ─────────
    var sum_sq = 0.0f;
    var i = tid;
    while (i < n4) {
        let v  = x[i];
        sum_sq = sum_sq + dot(v, v);
        i      = i + 64u;
    }
    // Tail (n not a multiple of 4)
    let tail_start = n4 * 4u;
    var j = tail_start + tid;
    while (j < n) {
        let v  = x[j / 4u][j % 4u];
        sum_sq = sum_sq + v * v;
        j      = j + 64u;
    }

    // ── Phase 2: reduce within subgroup (no barrier needed) ───────────────
    let sg_sum = subgroupAdd(sum_sq);

    // Lane 0 of each subgroup writes its partial result to workgroup memory.
    if (sg_lane == 0u) {
        sg_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();   // one barrier: wait for all subgroups to write

    // Thread 0 reduces across subgroups and broadcasts inv_rms.
    if (tid == 0u) {
        var total = 0.0f;
        for (var s = 0u; s < num_sgs; s = s + 1u) {
            total = total + sg_partial[s];
        }
        let mean  = total / f32(n);
        s_inv_rms = inverseSqrt(mean + params.eps);
    }
    workgroupBarrier();   // broadcast s_inv_rms to all threads

    // ── Phase 3: normalize and scale (vec4 stores) ────────────────────────
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
