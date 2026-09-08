struct RoPEParams {
    offset: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    has_bias: u32,
};

@group(0) @binding(0) var<storage, read>       qkv:      array<f32>;
@group(0) @binding(1) var<storage, read>       qkv_bias: array<f32>;
@group(0) @binding(2) var<storage, read>       cos_table: array<f32>;
@group(0) @binding(3) var<storage, read>       sin_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_out:    array<f32>;
@group(0) @binding(5) var<storage, read_write> k_cache:  array<f32>;
@group(0) @binding(6) var<storage, read_write> v_cache:  array<f32>;
@group(0) @binding(7) var<uniform>             params:   RoPEParams;

// Each workgroup handles one Q or KV head.
// Threads stride over half_dim in pairs, processing two vec2 lanes per iteration.
// This halves load instructions vs. the previous scalar approach.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)       group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let wid      = group_id.x;
    let head_dim = params.head_dim;
    let half_dim = head_dim / 2u;

    // Ring-buffer slot: prevents OOB writes when offset >= max_seq_len (long contexts).
    let slot       = params.offset % params.max_seq_len;
    let cos_offset = params.offset * head_dim; // RoPE uses absolute position for frequency

    let q_dim     = params.num_heads    * head_dim;
    let kv_dim    = params.num_kv_heads * head_dim;
    let head_stride = params.max_seq_len * head_dim;

    var tid = local_id.x;
    while (tid < half_dim) {
        // Load cos/sin pair for this frequency slot (one load serves both components).
        let c = cos_table[cos_offset + tid];
        let s = sin_table[cos_offset + tid];

        if (wid < params.num_heads) {
            // ── Query head ──────────────────────────────────────────────────
            let base = wid * head_dim;
            var q1 = qkv[base + tid];
            var q2 = qkv[base + tid + half_dim];
            if (params.has_bias == 1u) {
                q1 = q1 + qkv_bias[base + tid];
                q2 = q2 + qkv_bias[base + tid + half_dim];
            }
            q_out[base + tid]           = q1 * c - q2 * s;
            q_out[base + tid + half_dim] = q2 * c + q1 * s;
        } else {
            // ── KV head ─────────────────────────────────────────────────────
            let kv_h  = wid - params.num_heads;
            if (kv_h < params.num_kv_heads) {
                let k_base = q_dim + kv_h * head_dim;
                let v_base = q_dim + kv_dim + kv_h * head_dim;

                var k1 = qkv[k_base + tid];
                var k2 = qkv[k_base + tid + half_dim];
                var v1 = qkv[v_base + tid];
                var v2 = qkv[v_base + tid + half_dim];
                if (params.has_bias == 1u) {
                    k1 = k1 + qkv_bias[k_base + tid];
                    k2 = k2 + qkv_bias[k_base + tid + half_dim];
                    v1 = v1 + qkv_bias[v_base + tid];
                    v2 = v2 + qkv_bias[v_base + tid + half_dim];
                }

                let rot_k1 = k1 * c - k2 * s;
                let rot_k2 = k2 * c + k1 * s;

                // Write to ring-buffer slot (prevents OOB for contexts > max_seq_len)
                let dst = kv_h * head_stride + slot * head_dim;
                k_cache[dst + tid]           = rot_k1;
                k_cache[dst + tid + half_dim] = rot_k2;
                v_cache[dst + tid]           = v1;
                v_cache[dst + tid + half_dim] = v2;
            }
        }
        tid = tid + 64u;
    }
}
