struct RMSNormParams {
    n: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: RMSNormParams;

var<workgroup> sdata: array<f32, 64>;
var<workgroup> s_inv_rms: f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = params.n;
    let n_vec4 = n / 4u;

    // Phase 1: Compute sum of squares using vec4 loads (4x fewer load instructions)
    var sum_sq = 0.0;
    var i = tid;
    while (i < n_vec4) {
        let v = x[i];
        // Horizontal sum of vec4: dot(v, v) = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w
        sum_sq = sum_sq + dot(v, v);
        i = i + 64u;
    }
    // Handle tail elements (if n is not a multiple of 4)
    let vec4_end = n_vec4 * 4u;
    var j = vec4_end + tid;
    while (j < n) {
        let val = x[j / 4u][j % 4u];
        sum_sq = sum_sq + val * val;
        j = j + 64u;
    }

    // Phase 2: Workgroup tree reduction (5 barriers)
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

    // Phase 3: Normalize and scale using vec4 loads/stores (4x fewer instructions)
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
        let val = x[j / 4u][j % 4u];
        let wt = weight[j / 4u][j % 4u];
        y[j] = val * inv_rms * wt;
        j = j + 64u;
    }
}
