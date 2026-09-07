struct ResidualParams {
    n: u32,
};

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> residual: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: ResidualParams;

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
    x[idx] = x[idx] + residual[idx];
}
