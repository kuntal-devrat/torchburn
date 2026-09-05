struct Params {
    num_rows: u32,
    num_cols: u32,
    group_size: u32,
    num_groups: u32,
};

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gate_w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> gate_s: array<f32>;
@group(0) @binding(3) var<storage, read> up_w: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> up_s: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

// Multi-row tiling: 16 threads per row.
// 128-bit coalesced memory loads for weights and activations.
// Dynamic workgroup sizing replaces ROWS_PER_WG and WG_SIZE.
const ROWS_PER_WG: u32 = 4u;
const WG_SIZE: u32 = 64u;

var<workgroup> sdata_gate: array<f32, WG_SIZE>;
var<workgroup> sdata_up: array<f32, WG_SIZE>;

fn unpack_and_dot8(u: u32, x4_base: u32) -> f32 {
    let w0 = vec4<f32>(
        f32((u >> 0u) & 0xFu) - 8.0,
        f32((u >> 4u) & 0xFu) - 8.0,
        f32((u >> 8u) & 0xFu) - 8.0,
        f32((u >> 12u) & 0xFu) - 8.0,
    );
    let w1 = vec4<f32>(
        f32((u >> 16u) & 0xFu) - 8.0,
        f32((u >> 20u) & 0xFu) - 8.0,
        f32((u >> 24u) & 0xFu) - 8.0,
        f32((u >> 28u) & 0xFu) - 8.0,
    );

    return dot(w0, x[x4_base + 0u]) + dot(w1, x[x4_base + 1u]);
}

fn unpack_and_dot32(v4: vec4<u32>, x4_base: u32) -> f32 {
    return unpack_and_dot8(v4.x, x4_base + 0u)
         + unpack_and_dot8(v4.y, x4_base + 2u)
         + unpack_and_dot8(v4.z, x4_base + 4u)
         + unpack_and_dot8(v4.w, x4_base + 6u);
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let row_in_wg = tid / 16u;  // 0 .. (ROWS_PER_WG - 1)
    let lane = tid % 16u;       // 0 .. 15 (16 threads per row)

    let wg_id = group_id.y * 65535u + group_id.x;
    let row = wg_id * ROWS_PER_WG + row_in_wg;

    let K = params.num_cols;
    let num_groups = params.num_groups;
    let vec4_per_row = K / 32u;
    let row_vec4_base = row * vec4_per_row;
    let row_scale_base = row * num_groups;

    var acc_gate = 0.0;
    var acc_up = 0.0;

    if (row < params.num_rows) {
        for (var step_g = 0u; step_g < num_groups; step_g = step_g + 8u) {
            let g = step_g + lane / 2u;
            if (g < num_groups) {
                let s_gate = gate_s[row_scale_base + g];
                let s_up = up_s[row_scale_base + g];
                let half_group = lane % 2u;
                let x4_base = g * 16u + half_group * 8u;
                let vec4_idx = row_vec4_base + step_g * 2u + lane;

                let v4_gate = gate_w[vec4_idx];
                let dot_gate = unpack_and_dot32(v4_gate, x4_base);
                acc_gate = acc_gate + dot_gate * s_gate;

                let v4_up = up_w[vec4_idx];
                let dot_up = unpack_and_dot32(v4_up, x4_base);
                acc_up = acc_up + dot_up * s_up;
            }
        }
    }

    sdata_gate[tid] = acc_gate;
    sdata_up[tid] = acc_up;
    workgroupBarrier();

    // Lane 0 reduces partial dots and computes SwiGLU: silu(g) * u
    if (lane == 0u && row < params.num_rows) {
        let row_base = row_in_wg * 16u;
        let g0 = (sdata_gate[row_base + 0u] + sdata_gate[row_base + 8u]) + (sdata_gate[row_base + 4u] + sdata_gate[row_base + 12u]);
        let g1 = (sdata_gate[row_base + 2u] + sdata_gate[row_base + 10u]) + (sdata_gate[row_base + 6u] + sdata_gate[row_base + 14u]);
        let g2 = (sdata_gate[row_base + 1u] + sdata_gate[row_base + 9u]) + (sdata_gate[row_base + 5u] + sdata_gate[row_base + 13u]);
        let g3 = (sdata_gate[row_base + 3u] + sdata_gate[row_base + 11u]) + (sdata_gate[row_base + 7u] + sdata_gate[row_base + 15u]);
        let g = (g0 + g1) + (g2 + g3);

        let u0 = (sdata_up[row_base + 0u] + sdata_up[row_base + 8u]) + (sdata_up[row_base + 4u] + sdata_up[row_base + 12u]);
        let u1 = (sdata_up[row_base + 2u] + sdata_up[row_base + 10u]) + (sdata_up[row_base + 6u] + sdata_up[row_base + 14u]);
        let u2 = (sdata_up[row_base + 1u] + sdata_up[row_base + 9u]) + (sdata_up[row_base + 5u] + sdata_up[row_base + 13u]);
        let u3 = (sdata_up[row_base + 3u] + sdata_up[row_base + 11u]) + (sdata_up[row_base + 7u] + sdata_up[row_base + 15u]);
        let u = (u0 + u1) + (u2 + u3);

        let silu_g = g / (1.0 + exp(-g));
        y[row] = silu_g * u;
    }
}
