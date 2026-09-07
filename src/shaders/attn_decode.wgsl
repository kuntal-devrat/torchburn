struct AttnParams {
    offset: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k_cache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v_cache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: AttnParams;

// 64 threads per workgroup for full wave utilization (AMD=64, NVIDIA=32 with 2 waves).
var<workgroup> sdata: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let q_h = group_id.x;
    let tid = local_id.x; // 0 .. 63
    if (q_h >= params.num_heads) {
        return;
    }

    let gqa_group = params.num_heads / params.num_kv_heads;
    let kv_h = q_h / gqa_group;

    let head_dim_vec4 = params.head_dim / 4u;
    // Each thread handles a slice of vec4s across the head dimension.
    // For head_dim=128: 32 vec4s, each thread handles ~0-1 vec4s (strided).
    // For head_dim=64: 16 vec4s, each thread handles a fraction.
    let total_vec4 = head_dim_vec4;
    let vec4s_per_pass = (total_vec4 + 63u) / 64u; // ceil division

    let kv_head_base4 = kv_h * params.max_seq_len * head_dim_vec4;
    let q_base4 = q_h * head_dim_vec4;

    // Load q values for this thread's slice(s)
    var q_val: array<vec4<f32>, 2>;
    var q_count: u32 = 0u;
    for (var i = 0u; i < vec4s_per_pass; i = i + 1u) {
        let idx = tid + i * 64u;
        if (idx < total_vec4) {
            q_val[q_count] = q[q_base4 + idx];
            q_count = q_count + 1u;
        }
    }

    var m = -1e30;
    var l = 0.0;

    // Accumulators for output (one per vec4 this thread owns)
    var acc_v: array<vec4<f32>, 2>;
    var acc_count: u32 = 0u;
    for (var i = 0u; i < vec4s_per_pass; i = i + 1u) {
        let idx = tid + i * 64u;
        if (idx < total_vec4) {
            acc_v[acc_count] = vec4<f32>(0.0);
            acc_count = acc_count + 1u;
        }
    }

    let seq_len = params.offset + 1u;
    for (var t = 0u; t < seq_len; t = t + 1u) {
        // Each thread computes partial dot product over its q slice
        var p = 0.0;
        for (var i = 0u; i < q_count; i = i + 1u) {
            let vec_idx = tid + i * 64u;
            let t_offset4 = kv_head_base4 + t * head_dim_vec4 + vec_idx;
            let k_val = k_cache[t_offset4];
            p = p + dot(q_val[i], k_val);
        }

        // Workgroup reduction of partial dot products
        sdata[tid] = p;
        workgroupBarrier();

        // Tree reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
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

        let s_val = sdata[0] * params.scale;
        let m_new = max(m, s_val);
        let beta = exp(m - m_new);
        let alpha = exp(s_val - m_new);

        l = l * beta + alpha;

        // Accumulate weighted values for this thread's output slices
        for (var i = 0u; i < q_count; i = i + 1u) {
            let vec_idx = tid + i * 64u;
            let t_offset4 = kv_head_base4 + t * head_dim_vec4 + vec_idx;
            let v_val = v_cache[t_offset4];
            acc_v[i] = acc_v[i] * beta + alpha * v_val;
        }
        m = m_new;
    }

    // Write output
    if (l > 0.0) {
        let inv_l = 1.0 / l;
        for (var i = 0u; i < q_count; i = i + 1u) {
            let vec_idx = tid + i * 64u;
            attn_out[q_base4 + vec_idx] = acc_v[i] * inv_l;
        }
    } else {
        for (var i = 0u; i < q_count; i = i + 1u) {
            let vec_idx = tid + i * 64u;
            attn_out[q_base4 + vec_idx] = vec4<f32>(0.0);
        }
    }
}
