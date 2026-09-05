struct SwiGLUParams {
    n: u32,
};

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: SwiGLUParams;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = group_id.x * 64u + local_id.x;
    if (idx >= params.n) {
        return;
    }

    let g = gate[idx];
    let u = up[idx];
    let silu_g = g / (1.0 + exp(-g));
    out[idx] = silu_g * u;
}
