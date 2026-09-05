struct AddRMSNormParams {
    n: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: AddRMSNormParams;

var<workgroup> sdata: array<f32, 64>;
var<workgroup> s_inv_rms: f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = params.n;

    var sum_sq = 0.0;
    var i = tid;
    while (i < n) {
        let val = x[i] + residual[i];
        x[i] = val;
        sum_sq = sum_sq + val * val;
        i = i + 64u;
    }

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

    let inv_rms = s_inv_rms;
    i = tid;
    while (i < n) {
        y[i] = x[i] * inv_rms * weight[i];
        i = i + 64u;
    }
}
