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

// 16 threads per workgroup. Dynamically handles head_dim = 64, 128, 256.
var<workgroup> sdata: array<f32, 16>;

@compute @workgroup_size(16)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let q_h = group_id.x;
    let tid = local_id.x; // 0 .. 15
    if (q_h >= params.num_heads) {
        return;
    }

    let gqa_group = params.num_heads / params.num_kv_heads;
    let kv_h = q_h / gqa_group;

    let head_dim_vec4 = params.head_dim / 4u;
    let vec4_per_thread = head_dim_vec4 / 16u; // 1 for head_dim=64, 2 for head_dim=128

    let kv_head_base4 = kv_h * params.max_seq_len * head_dim_vec4;
    let q_base4 = q_h * head_dim_vec4;
    let thread_base4 = q_base4 + tid * vec4_per_thread;

    var q_val: array<vec4<f32>, 4>;
    var acc_v: array<vec4<f32>, 4>;
    for (var i = 0u; i < vec4_per_thread; i = i + 1u) {
        q_val[i] = q[thread_base4 + i];
        acc_v[i] = vec4<f32>(0.0);
    }

    var m = -1e30;
    var l = 0.0;

    let seq_len = params.offset + 1u;
    for (var t = 0u; t < seq_len; t = t + 1u) {
        let t_offset4 = kv_head_base4 + t * head_dim_vec4 + tid * vec4_per_thread;

        var p = 0.0;
        for (var i = 0u; i < vec4_per_thread; i = i + 1u) {
            let k_val = k_cache[t_offset4 + i];
            p = p + dot(q_val[i], k_val);
        }

        sdata[tid] = p;
        workgroupBarrier();

        if (tid == 0u) {
            let sum0 = (sdata[0u] + sdata[8u]) + (sdata[4u] + sdata[12u]);
            let sum1 = (sdata[2u] + sdata[10u]) + (sdata[6u] + sdata[14u]);
            let sum2 = (sdata[1u] + sdata[9u]) + (sdata[5u] + sdata[13u]);
            let sum3 = (sdata[3u] + sdata[11u]) + (sdata[7u] + sdata[15u]);
            sdata[0] = ((sum0 + sum1) + (sum2 + sum3)) * params.scale;
        }
        workgroupBarrier();

        let s_val = sdata[0];
        let m_new = max(m, s_val);
        let beta = exp(m - m_new);
        let alpha = exp(s_val - m_new);

        l = l * beta + alpha;
        for (var i = 0u; i < vec4_per_thread; i = i + 1u) {
            let v_val = v_cache[t_offset4 + i];
            acc_v[i] = acc_v[i] * beta + alpha * v_val;
        }
        m = m_new;
    }

    if (l > 0.0) {
        let inv_l = 1.0 / l;
        for (var i = 0u; i < vec4_per_thread; i = i + 1u) {
            attn_out[thread_base4 + i] = acc_v[i] * inv_l;
        }
    } else {
        for (var i = 0u; i < vec4_per_thread; i = i + 1u) {
            attn_out[thread_base4 + i] = vec4<f32>(0.0);
        }
    }
}
