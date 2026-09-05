struct Params {
    num_rows: u32,
    num_cols: u32,
    group_size: u32,
    num_groups: u32,
};

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w_bytes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Multi-row tiling: 16 threads per row.
// 128-bit coalesced memory loads for both weights (vec4<u32>) and activations (vec4<f32>).
// Native 4-wide dot product dot() hardware vector intrinsics.
const ROWS_PER_WG: u32 = 4u;
const WG_SIZE: u32 = 64u;

var<workgroup> sdata: array<f32, WG_SIZE>;

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

    var acc = 0.0;

    if (row < params.num_rows) {
        // Each step processes 8 groups (512 INT4 weights) across the 16 lanes.
        // For K = 896: only 2 steps (groups 0..7, then 8..13)!
        for (var step_g = 0u; step_g < num_groups; step_g = step_g + 8u) {
            let g = step_g + lane / 2u;
            if (g < num_groups) {
                let scale = scales[row_scale_base + g];
                let half_group = lane % 2u;
                // col_base is g * 64u + half_group * 32u.
                // In units of vec4<f32> (4 floats), x4_base = g * 16u + half_group * 8u.
                let x4_base = g * 16u + half_group * 8u;
                let vec4_idx = row_vec4_base + step_g * 2u + lane;
                let v4 = w_bytes[vec4_idx];
                let dot32 = unpack_and_dot32(v4, x4_base);
                acc = acc + dot32 * scale;
            }
        }
    }

    sdata[tid] = acc;
    workgroupBarrier();

    // Lane 0 directly reduces the 16 lane partial dots for this row and writes output y[row]
    if (lane == 0u && row < params.num_rows) {
        let row_base = row_in_wg * 16u;
        let sum0 = (sdata[row_base + 0u] + sdata[row_base + 8u]) + (sdata[row_base + 4u] + sdata[row_base + 12u]);
        let sum1 = (sdata[row_base + 2u] + sdata[row_base + 10u]) + (sdata[row_base + 6u] + sdata[row_base + 14u]);
        let sum2 = (sdata[row_base + 1u] + sdata[row_base + 9u]) + (sdata[row_base + 5u] + sdata[row_base + 13u]);
        let sum3 = (sdata[row_base + 3u] + sdata[row_base + 11u]) + (sdata[row_base + 7u] + sdata[row_base + 15u]);
        y[row] = (sum0 + sum1) + (sum2 + sum3);
    }
}


