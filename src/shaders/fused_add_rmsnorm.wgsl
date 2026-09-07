struct AddRMSNormParams {
    n: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> residual: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: AddRMSNormParams;

var<workgroup> sdata: array<f32, 64>;
var<workgroup> s_inv_rms: f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = params.n;
    let n_vec4 = n / 4u;

    // Phase 1: Fused residual add + sum of squares using vec4 loads
    var sum_sq = 0.0;
    var i = tid;
    while (i < n_vec4) {
        let val = x[i] + residual[i];
        x[i] = val; // fused in-place residual add
        sum_sq = sum_sq + dot(val, val);
        i = i + 64u;
    }
    // Handle tail elements
    let vec4_end = n_vec4 * 4u;
    var j = vec4_end + tid;
    while (j < n) {
        let vi = j / 4u;
        let ei = j % 4u;
        let val = x[vi][ei] + residual[vi][ei];
        x[vi][ei] = val;
        sum_sq = sum_sq + val * val;
        j = j + 64u;
    }

    // Phase 2: Workgroup tree reduction
    sdata[tid] = sum_sq;
    workgroupBarrier();
    if (tid < 32u) { sdata[tid] = sdata[tid] + sdata[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { sdata[tid] = sdata[tid] + sdata[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { sdata[tid] = sdata[tid] + sdata[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { sdata[tid] = sdata[tid] + sdata[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { sdata[tid] = sdata[tid] + sdata[tid + 2u]; }
    workgroupBarrier();
    if (tid < 1u) { sdata[tid] = sdata[tid] + sdata[tid + 1u]; }
    workgroupBarrier();

    if (tid == 0u) {
        let mean = sdata[0] / f32(n);
        s_inv_rms = inverseSqrt(mean + params.eps);
    }
    workgroupBarrier();
    storageBarrier();

    // Phase 3: Normalize and scale using vec4 loads/stores
    let inv_rms = s_inv_rms;
    i = tid;
    while (i < n_vec4) {
        let v = x[i];
        let w = weight[i];
        y[i] = v * inv_rms * w;
        i = i + 64u;
    }
    // Handle tail elements
    j = vec4_end + tid;
    while (j < n) {
        let vi = j / 4u;
        let ei = j % 4u;
        let val = x[vi][ei];
        let wt = weight[vi][ei];
        y[j] = val * inv_rms * wt;
        j = j + 64u;
    }
}
