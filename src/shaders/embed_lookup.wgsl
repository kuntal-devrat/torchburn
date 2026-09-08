// embed_lookup.wgsl — GPU-side embedding table lookup.
//
// Instead of uploading hidden_size*4 bytes (e.g. 3584 bytes for Qwen-0.5B) from
// the CPU every decode step, the full embedding table lives permanently in GPU
// storage. This shader reads a single token-id from a 4-byte uniform buffer and
// copies the corresponding row into x_buf in one compute dispatch.
//
// Dispatch: ceil(hidden_size / WG_SIZE) workgroups, WG_SIZE threads each.
// Each thread writes 4 f32 values (vec4 stride), so effective write is 4*WG_SIZE
// floats per workgroup. For hidden_size=896 we need ceil(896/256)=4 workgroups.

struct Params {
    vocab_size  : u32,
    hidden_size : u32,
};

@group(0) @binding(0) var<storage, read>       embed_table : array<f32>;
@group(0) @binding(1) var<uniform>             token_id    : u32;
@group(0) @binding(2) var<storage, read_write> x_buf       : array<f32>;
@group(0) @binding(3) var<uniform>             params      : Params;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= params.hidden_size) {
        return;
    }
    let src = token_id * params.hidden_size + idx;
    x_buf[idx] = embed_table[src];
}
