struct SwiGLUParams {
    n: u32,
};

@group(0) @binding(0) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> up: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: SwiGLUParams;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let n_vec4 = params.n / 4u;
    let idx = group_id.x * 64u + local_id.x;
    if (idx >= n_vec4) {
        return;
    }

    let g = gate[idx];
    let u = up[idx];
    // SiLU: g / (1 + exp(-g)) — applied component-wise to vec4
    let silu_g = g / (1.0 + exp(-g));
    out[idx] = silu_g * u;
}
