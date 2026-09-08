// Fused residual-add + RMSNorm with subgroup-based reduction.
//
// Replaces the previous 7-step barrier tree with subgroupAdd (one instruction
// per subgroup) + a single inter-subgroup barrier.  Typical savings on Intel
// Iris Xe (32-lane subgroups, 2 subgroups per WG): 5 barriers → 1 barrier.
// On AMD RDNA (64-lane, 1 subgroup per WG): 7 barriers → 0 barriers.
//
// Bindings:
//   0: x        (read_write) — residual stream, updated in-place
//   1: residual (read)       — incoming residual to add
//   2: weight   (read)       — RMSNorm gain
//   3: y        (read_write) — normalized output
//   4: params   (uniform)    — {n, eps}

struct AddRMSNormParams {
    n:   u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read_write> x:        array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       residual: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       weight:   array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y:        array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             params:   AddRMSNormParams;

var<workgroup> sg_partial: array<f32, 2>;
var<workgroup> s_inv_rms:  f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id)  local_id: vec3<u32>,
        @builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(subgroup_id)            sg_id:   u32,
        @builtin(num_subgroups)          num_sgs: u32) {
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

    // ── Phase 2: subgroup reduction + one cross-subgroup barrier ──────────
    let sg_sum = subgroupAdd(sum_sq);
    if (sg_lane == 0u) {
        sg_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var total = 0.0f;
        for (var s = 0u; s < num_sgs; s = s + 1u) {
            total = total + sg_partial[s];
        }
        let mean  = total / f32(n);
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
