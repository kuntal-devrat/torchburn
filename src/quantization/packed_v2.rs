//! V2 interleaved INT4 layout: f16 scales, pack, v2 GEMV row kernels,
//! grouped-linear entry points (incl. wgpu). Inherits root via super.

use super::*;
// ════════════════════════════════════════════════════════════════════════════
// Phase 1.1 — v2 interleaved weight layout (`torchburn_int4_g64_v2`).
//
// Each 64-element quantization group is stored as one 34-byte block:
//   [0..32)   packed int4 nibbles (same bit layout as v1: byte j holds
//             element 2j in the low nibble, element 2j+1 in the high nibble)
//   [32..34)  f16 little-endian group scale
//
// The scale rides inside the weight stream, so the GEMV reads one sequential
// 34-byte block per group (one stream instead of two), and the total footprint
// is 34·G vs v1's 32·G + 4·G (f32 scales) — ~5.5% smaller. The quantization
// math is byte-for-byte identical to v1; only the scale precision changes
// (f16 instead of f32, ~1e-3 relative — within every parity budget).
//
// Note: the roadmap sketch mentioned 64-byte cache-line blocks with 4-value
// groups; that finer scale granularity (16 sub-blocks per cache line) is the
// Q4_K direction deferred to Phase 5. This layout keeps group=64 so v1/v2
// share identical quantization quality while landing the interleave + f16
// scales that Phase 1.2's fused kernel builds on.
// ════════════════════════════════════════════════════════════════════════════

pub const V2_BLOCK_BYTES: usize = 34;
pub const V2_PACKED_BYTES: usize = 32;

/// IEEE-754 binary16 -> binary32 (no dependency on the `half` crate).
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // subnormal: mantissa / 1024 * 2^-14
        let v = (mant as f32 / 1024.0) * 2f32.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    if exp == 0x1F {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    return f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13));
}

/// f16 scale at the tail of a 34-byte v2 group block.
#[inline(always)]
unsafe fn v2_scale_f32(block: *const u8) -> f32 {
    f16_to_f32(u16::from_le_bytes([*block.add(32), *block.add(33)]))
}

/// f32 -> IEEE-754 binary16, round-to-nearest-even (no dependency on `half`).
/// Overflow saturates to infinity; subnormals keep RNE precision.
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp_abs = ((x >> 23) & 0xFF) as i32;
    let mant = x & 0x007F_FFFF;
    if exp_abs == 0xFF {
        return if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00
        };
    }
    let e = exp_abs - 127;
    if e > 15 {
        return sign | 0x7C00;
    }
    if e >= -14 {
        let h_exp = (e + 15) as u32;
        let h_mant = mant >> 13;
        let rem = mant & 0x1FFF;
        let mut h = (h_exp << 10) | h_mant;
        if rem > 0x1000 || (rem == 0x1000 && (h_mant & 1) == 1) {
            h += 1; // carry may propagate into the exponent: correct RNE
        }
        return sign | (h as u16);
    }
    if e < -25 {
        return sign; // underflow to signed zero
    }
    let full = mant | 0x0080_0000;
    let shift = (-e - 1) as u32;
    let h_mant = full >> shift;
    let rem = full & ((1 << shift) - 1);
    let round = 1 << (shift - 1);
    let mut m = h_mant;
    if rem > round || (rem == round && (h_mant & 1) == 1) {
        m += 1;
    }
    sign | (m as u16)
}

/// Pack v1 W4 group-64 weights (row-major packed nibbles + separate f32
/// scales) into the v2 interleaved 34-byte-block layout. One-time transform:
/// the output is consumed by `gemv_w4a32_grouped_v2` and the v2 neuron
/// kernels, which read each group as one sequential 34-byte stream.
///
/// `k` must be a multiple of 64 (the v2 layout has no remainder handling);
/// callers gate on that and keep the v1 path otherwise.
pub fn pack_rows_w4a32_group64_v1_to_v2(w_packed: &[u8], scales: &[f32], k: usize) -> Vec<u8> {
    debug_assert_eq!(
        k % 64,
        0,
        "v2 packing requires k divisible by the group size"
    );
    let num_groups = k / 64;
    let src_row_bytes = (k + 1) / 2;
    debug_assert_eq!(
        w_packed.len() % src_row_bytes,
        0,
        "v1 buffer is not row-aligned"
    );
    let n = w_packed.len() / src_row_bytes;
    debug_assert!(scales.len() >= n * num_groups, "scales buffer too small");

    let mut out = vec![0u8; n * num_groups * V2_BLOCK_BYTES];
    for row in 0..n {
        let src = &w_packed[row * src_row_bytes..(row + 1) * src_row_bytes];
        let dst =
            &mut out[row * num_groups * V2_BLOCK_BYTES..(row + 1) * num_groups * V2_BLOCK_BYTES];
        for g in 0..num_groups {
            dst[g * V2_BLOCK_BYTES..g * V2_BLOCK_BYTES + 32]
                .copy_from_slice(&src[g * 32..g * 32 + 32]);
            let h = f32_to_f16(scales[row * num_groups + g]);
            dst[g * V2_BLOCK_BYTES + 32..g * V2_BLOCK_BYTES + 34].copy_from_slice(&h.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod v2_pack_tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn f16_roundtrip_accuracy() {
        for &v in &[
            0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, 65504.0, -3.75, 1234.5, 1e-4, 6.0e-5, 6.1e-8,
        ] {
            let back = f16_to_f32(f32_to_f16(v));
            if v == 0.0 {
                assert_eq!(back, v, "{v}");
                assert!(back.is_sign_negative() == v.is_sign_negative(), "{v}");
            } else if v.abs() < 6.1e-5 {
                // f16 subnormal range: only absolute precision is guaranteed
                // (step = 2^-24 = 5.96e-8).
                assert!(
                    (back - v).abs() < 6e-8,
                    "f16 roundtrip subnormal {v} -> {back}"
                );
            } else {
                let rel = ((back - v) / v).abs();
                assert!(rel < 6e-4, "f16 roundtrip {v} -> {back} (rel {rel})");
            }
        }
        // Overflow saturates to infinity, preserving sign.
        assert_eq!(f16_to_f32(f32_to_f16(1e6)), f32::INFINITY);
        assert_eq!(f16_to_f32(f32_to_f16(-1e6)), f32::NEG_INFINITY);
    }

    #[test]
    fn pack_produces_block_layout() {
        let n = 3;
        let k = 128;
        let mut seed = 42u64;
        let src: Vec<u8> = (0..n * (k / 2)).map(|_| lcg(&mut seed) as u8).collect();
        let scales: Vec<f32> = (0..n * (k / 64)).map(|i| 0.01 + i as f32 * 0.001).collect();

        let packed = pack_rows_w4a32_group64_v1_to_v2(&src, &scales, k);
        assert_eq!(packed.len(), n * (k / 64) * V2_BLOCK_BYTES);

        for row in 0..n {
            for g in 0..k / 64 {
                let blk = &packed[row * (k / 64) * V2_BLOCK_BYTES + g * V2_BLOCK_BYTES..];
                assert_eq!(
                    &blk[..32],
                    &src[row * (k / 2) + g * 32..row * (k / 2) + g * 32 + 32],
                    "nibbles row {row} group {g}"
                );
                let got = f16_to_f32(u16::from_le_bytes([blk[32], blk[33]]));
                let want = scales[row * (k / 64) + g];
                assert!(
                    (got - want).abs() / want < 6e-4,
                    "scale row {row} group {g}"
                );
            }
        }
    }

    #[test]
    fn v2_gemv_matches_v1_within_f16_tolerance() {
        let n = 8;
        let k = 128;
        let group_size = 64;
        let num_groups = k / group_size;
        let mut seed = 7u64;
        let w: Vec<u8> = (0..n * (k / 2))
            .map(|_| (lcg(&mut seed) & 0xFF) as u8)
            .collect();
        let s: Vec<f32> = (0..n * num_groups)
            .map(|_| 0.002 + (lcg(&mut seed) % 1000) as f32 * 1e-5)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|_| ((lcg(&mut seed) % 2000) as f32 - 1000.0) / 500.0)
            .collect();

        let mut out_v1 = vec![0.0f32; n];
        unsafe {
            gemv_w4a32_grouped(
                x.as_ptr(),
                w.as_ptr(),
                s.as_ptr(),
                None,
                out_v1.as_mut_ptr(),
                n,
                k,
                group_size,
            );
        }

        let w2 = pack_rows_w4a32_group64_v1_to_v2(&w, &s, k);
        let mut out_v2 = vec![0.0f32; n];
        unsafe {
            gemv_w4a32_grouped_v2(
                x.as_ptr(),
                w2.as_ptr(),
                None,
                out_v2.as_mut_ptr(),
                n,
                k,
                group_size,
            );
        }

        for (a, b) in out_v1.iter().zip(&out_v2) {
            let tol = 5e-3 * a.abs().max(1.0);
            assert!((a - b).abs() < tol, "v1 {a} vs v2 {b}");
        }
    }
}

unsafe fn gemv_row_w4a32_group64_v2_scalar(
    x: *const f32,
    w_blocked: *const u8,
    num_groups: usize,
) -> f32 {
    let mut s = 0.0f32;
    for g in 0..num_groups {
        let block = w_blocked.add(g * V2_BLOCK_BYTES);
        let scale = v2_scale_f32(block);
        s += dot_f32_u4_group_scalar(x.add(g * 64), block, 64) * scale;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemv_row_w4a32_group64_v2_avx512(
    x: *const f32,
    w_blocked: *const u8,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm512_setzero_ps();
    for g in 0..num_groups {
        let block = w_blocked.add(g * V2_BLOCK_BYTES);
        let scale = _mm512_set1_ps(v2_scale_f32(block));
        let sum0 = unpack_and_fma_32_avx512(x.add(g * 64), block, mask_low, sub8);
        let sum1 = unpack_and_fma_32_avx512(x.add(g * 64 + 32), block.add(16), mask_low, sub8);
        let grp_sum = _mm512_add_ps(sum0, sum1);
        acc = _mm512_fmadd_ps(grp_sum, scale, acc);
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_w4a32_group64_v2_avx2(
    x: *const f32,
    w_blocked: *const u8,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();
    for g in 0..num_groups {
        let block = w_blocked.add(g * V2_BLOCK_BYTES);
        let scale = _mm256_set1_ps(v2_scale_f32(block));
        let sum0 = unpack_and_fma_32_avx2(x.add(g * 64), block, mask_low, sub8);
        let sum1 = unpack_and_fma_32_avx2(x.add(g * 64 + 32), block.add(16), mask_low, sub8);
        let grp_sum = _mm256_add_ps(sum0, sum1);
        acc = _mm256_fmadd_ps(grp_sum, scale, acc);
    }
    hsum256_ps_avx(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512f,avx512bw")]
unsafe fn gemv_row_w4a8_group64_v2_vnni_avx512(
    x_u8: *const u8,
    s_x: f32,
    w_blocked: *const u8,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let ones = _mm512_set1_epi8(1);
    let mut acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let block = w_blocked.add(g * V2_BLOCK_BYTES);
        let x64 = _mm512_loadu_si512(x_u8.add(g * 64) as *const __m512i);
        let scale_vec = _mm512_set1_ps(s_x * v2_scale_f32(block));

        let raw0 = _mm_loadu_si128(block as *const __m128i);
        let lo0 = _mm_and_si128(raw0, mask_low);
        let hi0 = _mm_and_si128(_mm_srli_epi16::<4>(raw0), mask_low);
        let inter_lo0 = _mm_unpacklo_epi8(lo0, hi0);
        let inter_hi0 = _mm_unpackhi_epi8(lo0, hi0);
        let s_lo0 = _mm_sub_epi8(inter_lo0, sub8);
        let s_hi0 = _mm_sub_epi8(inter_hi0, sub8);
        let w32_0 = _mm256_set_m128i(s_hi0, s_lo0);

        let raw1 = _mm_loadu_si128(block.add(16) as *const __m128i);
        let lo1 = _mm_and_si128(raw1, mask_low);
        let hi1 = _mm_and_si128(_mm_srli_epi16::<4>(raw1), mask_low);
        let inter_lo1 = _mm_unpacklo_epi8(lo1, hi1);
        let inter_hi1 = _mm_unpackhi_epi8(lo1, hi1);
        let s_lo1 = _mm_sub_epi8(inter_lo1, sub8);
        let s_hi1 = _mm_sub_epi8(inter_hi1, sub8);
        let w32_1 = _mm256_set_m128i(s_hi1, s_lo1);

        let w64 = _mm512_inserti64x4(_mm512_castsi256_si512(w32_0), w32_1, 1);

        let dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, w64);
        let w_sum_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, w64);
        let true_i32 = _mm512_sub_epi32(dot_i32, _mm512_slli_epi32::<7>(w_sum_i32));
        let true_f32 = _mm512_cvtepi32_ps(true_i32);

        acc = _mm512_fmadd_ps(true_f32, scale_vec, acc);
    }
    _mm512_reduce_add_ps(acc)
}

/// Unpack a v2 34-byte block's 64 nibbles into signed i8 lanes (low nibble of
/// byte j -> element 2j, high nibble -> element 2j+1, minus 8), as a __m512i
/// with 64 x i8.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn unpack_w64_v2_avx512(block: *const u8) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let raw0 = _mm_loadu_si128(block as *const __m128i);
    let lo0 = _mm_and_si128(raw0, mask_low);
    let hi0 = _mm_and_si128(_mm_srli_epi16::<4>(raw0), mask_low);
    let inter_lo0 = _mm_unpacklo_epi8(lo0, hi0);
    let inter_hi0 = _mm_unpackhi_epi8(lo0, hi0);
    let s_lo0 = _mm_sub_epi8(inter_lo0, sub8);
    let s_hi0 = _mm_sub_epi8(inter_hi0, sub8);
    let w32_0 = _mm256_set_m128i(s_hi0, s_lo0);
    let raw1 = _mm_loadu_si128(block.add(16) as *const __m128i);
    let lo1 = _mm_and_si128(raw1, mask_low);
    let hi1 = _mm_and_si128(_mm_srli_epi16::<4>(raw1), mask_low);
    let inter_lo1 = _mm_unpacklo_epi8(lo1, hi1);
    let inter_hi1 = _mm_unpackhi_epi8(lo1, hi1);
    let s_lo1 = _mm_sub_epi8(inter_lo1, sub8);
    let s_hi1 = _mm_sub_epi8(inter_hi1, sub8);
    let w32_1 = _mm256_set_m128i(s_hi1, s_lo1);
    _mm512_inserti64x4(_mm512_castsi256_si512(w32_0), w32_1, 1)
}

/// Fused dequant-dot kernel (Phase 1.2): computes 8 output rows at once for
/// the W4A8 VNNI tier. The quantized activation block `x_u8[g*64..g*64+64]`
/// is loaded once per group and shared across all 8 rows (one vpdpbusd each),
/// which reuses the x load and gives 8 independent accumulation chains for
/// instruction-level parallelism. The 34-byte v2 block brings the packed
/// nibbles and f16 scale together, so a single sequential block read per
/// (row, group) supplies everything the row needs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512f,avx512bw")]
unsafe fn gemv_row8_w4a8_group64_v2_vnni_avx512(
    x_u8: *const u8,
    s_x: f32,
    w_blocked8: *const u8,
    bytes_per_row: usize,
    num_groups: usize,
    out8: *mut f32,
) {
    use std::arch::x86_64::*;
    let ones = _mm512_set1_epi8(1);
    let mut acc = [_mm512_setzero_ps(); 8];
    for g in 0..num_groups {
        let x64 = _mm512_loadu_si512(x_u8.add(g * 64) as *const __m512i);
        for r in 0..8 {
            let block = w_blocked8.add(r * bytes_per_row + g * V2_BLOCK_BYTES);
            let w64 = unpack_w64_v2_avx512(block);
            let dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, w64);
            let w_sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, w64);
            let true_i32 = _mm512_sub_epi32(dot_i32, _mm512_slli_epi32::<7>(w_sum));
            let true_f32 = _mm512_cvtepi32_ps(true_i32);
            let scale = _mm512_set1_ps(s_x * v2_scale_f32(block));
            acc[r] = _mm512_fmadd_ps(true_f32, scale, acc[r]);
        }
    }
    for r in 0..8 {
        *out8.add(r) = _mm512_reduce_add_ps(acc[r]);
    }
}

/// Fast single-token GEMV for W4A32 v2 (interleaved 34-byte blocks, f16
/// scales embedded per 64-group). Same tier dispatch as v1.
///
/// # Safety
/// `x` must hold `k` readable f32s; `w_blocked` must hold `n` rows of
/// `(k / 64) * V2_BLOCK_BYTES` bytes; `bias` (if `Some`) must hold `n`
/// f32s; `out` must hold `n` writable f32s.
pub unsafe fn gemv_w4a32_grouped_v2(
    x: *const f32,
    w_blocked: *const u8,
    bias: Option<*const f32>,
    out: *mut f32,
    n: usize,
    k: usize,
    group_size: usize,
) {
    debug_assert_eq!(group_size, 64, "v2 layout is defined for 64-element groups");
    use rayon::prelude::*;
    let num_groups = k / group_size;
    let bytes_per_row = num_groups * V2_BLOCK_BYTES;

    let x_usize = x as usize;
    let w_usize = w_blocked as usize;
    let b_usize = bias.map(|bp| bp as usize);
    let out_usize = out as usize;

    let n_threads = rayon::current_num_threads();

    let feats = crate::dispatch::cpu_features();

    #[cfg(target_arch = "x86_64")]
    let has_vnni = feats.avx512vnni && feats.avx512f && feats.avx512bw;
    #[cfg(not(target_arch = "x86_64"))]
    let has_vnni = false;

    #[cfg(target_arch = "x86_64")]
    let has_avx512 = feats.avx512f && feats.avx512bw;
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx512 = false;

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = feats.avx2 && feats.fma;
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
        let (u, s) = unsafe { quantize_activation_to_u8(x, k) };
        (Some(u), s)
    } else {
        (None, 1.0f32)
    };
    let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

    // Phase 1.2: distribute over 8-row blocks. On the VNNI tier each block runs
    // the fused kernel (8 outputs share each x_u8 group load); other tiers fall
    // back to per-row kernels so every tier stays correct.
    //
    // Register blocking helps ALU-bound GEMVs (x reused in registers, 8 ILP
    // chains) but hurts memory-bound ones: interleaving 8 rows destroys the
    // sequential weight stream, which is fatal for huge matrices like lm_head
    // (n=151936, ~72 MB of weights — measured 1.2x slower than per-row).
    // So the fused kernel is used for moderate n; large n streams per-row.
    const BLOCK_ROWS: usize = 8;
    // n below which register-blocking wins over weight streaming (measured on
    // this Tiger Lake host: fused is 0.75x at n=4864, ~1.0x at n=2688/896, but
    // 1.2x at n=151936). 16384 keeps every layer-sized GEMV fused and routes
    // lm_head to per-row.
    const ADAPTIVE_FUSED_MAX_N: usize = 16384;
    let use_fused = has_vnni && n <= ADAPTIVE_FUSED_MAX_N;
    let n_blocks = (n + BLOCK_ROWS - 1) / BLOCK_ROWS;
    let min_blocks = (n_blocks / (n_threads * 2)).max(1);

    (0..n_blocks)
        .into_par_iter()
        .with_min_len(min_blocks)
        .for_each(|bi| {
            let row0 = bi * BLOCK_ROWS;
            let rows = (n - row0).min(BLOCK_ROWS);
            let x_p = x_usize as *const f32;
            let w_block0 = (w_usize as *const u8).add(row0 * bytes_per_row);
            let b_p = b_usize.map(|bp| bp as *const f32);
            let out_p = out_usize as *mut f32;

            if rows == BLOCK_ROWS && use_fused {
                #[cfg(target_arch = "x86_64")]
                {
                    if let Some(x_u8_p) = x_u8_ptr {
                        unsafe {
                            gemv_row8_w4a8_group64_v2_vnni_avx512(
                                x_u8_p as *const u8,
                                s_x,
                                w_block0,
                                bytes_per_row,
                                num_groups,
                                out_p.add(row0),
                            );
                        }
                    }
                }
                for r in 0..rows {
                    let j = row0 + r;
                    let b = if let Some(bp) = b_p {
                        unsafe { *bp.add(j) }
                    } else {
                        0.0
                    };
                    unsafe {
                        *out_p.add(j) += b;
                    }
                }
            } else {
                for r in 0..rows {
                    let j = row0 + r;
                    let w_row = unsafe { w_block0.add(r * bytes_per_row) };
                    let row_sum = if let Some(x_u8_p) = x_u8_ptr {
                        #[cfg(target_arch = "x86_64")]
                        {
                            unsafe {
                                gemv_row_w4a8_group64_v2_vnni_avx512(
                                    x_u8_p as *const u8,
                                    s_x,
                                    w_row,
                                    num_groups,
                                )
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            0.0f32
                        }
                    } else if has_avx512 {
                        #[cfg(target_arch = "x86_64")]
                        {
                            unsafe { gemv_row_w4a32_group64_v2_avx512(x_p, w_row, num_groups) }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            0.0f32
                        }
                    } else if has_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        {
                            unsafe { gemv_row_w4a32_group64_v2_avx2(x_p, w_row, num_groups) }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            0.0f32
                        }
                    } else {
                        unsafe { gemv_row_w4a32_group64_v2_scalar(x_p, w_row, num_groups) }
                    };
                    let b = if let Some(bp) = b_p {
                        unsafe { *bp.add(j) }
                    } else {
                        0.0
                    };
                    unsafe {
                        *out_p.add(j) = row_sum + b;
                    }
                }
            }
        });
}

/// v2 W4A32 Linear projection: x is (..., K), w_blocked is (N, G*34) uint8
/// in the Phase 1.1 interleaved layout (34-byte blocks, embedded f16 scales).
pub fn w4a32_grouped_linear_v2(
    x: &BorrowedTensor,
    w_blocked: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "w4a32_grouped_linear_v2 requires x with at least 1 dimension",
        ));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w_blocked.shape.len() != 2 {
        return Err(unsupported(
            "w4a32_grouped_linear_v2 requires 2D blocked weight matrix",
        ));
    }
    let n = w_blocked.shape[0] as usize;
    let num_groups = k / group_size;
    let expected_bytes = num_groups * V2_BLOCK_BYTES;
    if w_blocked.shape[1] as usize != expected_bytes {
        return Err(unsupported(&format!(
            "w4a32_grouped_linear_v2 dimension mismatch: K={k} group={group_size} needs {} blocked bytes per row, got {}",
            expected_bytes,
            w_blocked.shape[1]
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<u8>(w_blocked) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };
    let bias_slice = bias.map(|b| unsafe { typed_slice::<f32>(b) });
    let b_ptr = bias_slice.map(|b| b.as_ptr());

    if m == 1 {
        unsafe {
            gemv_w4a32_grouped_v2(
                x_slice.as_ptr(),
                w_slice.as_ptr(),
                b_ptr,
                out_slice.as_mut_ptr(),
                n,
                k,
                group_size,
            );
        }
    } else {
        for i in 0..m {
            let x_tok = unsafe { x_slice.as_ptr().add(i * k) };
            let out_tok = unsafe { out_slice.as_mut_ptr().add(i * n) };
            unsafe {
                gemv_w4a32_grouped_v2(x_tok, w_slice.as_ptr(), b_ptr, out_tok, n, k, group_size);
            }
        }
    }

    Ok(out)
}

/// Compute W4A32 Linear projection with grouped scaling factors:
/// out = sum_g( (x_g @ w_g.T) * scale_g ) + bias
pub fn w4a32_grouped_linear(
    x: &BorrowedTensor,
    w_packed: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "w4a32_grouped_linear requires x with at least 1 dimension",
        ));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w_packed.shape.len() != 2 {
        return Err(unsupported(
            "w4a32_grouped_linear requires 2D packed weight matrix",
        ));
    }
    let n = w_packed.shape[0] as usize;
    let w_packed_k = w_packed.shape[1] as usize;
    if (k + 1) / 2 != w_packed_k {
        return Err(unsupported(&format!(
            "w4a32_grouped_linear dimension mismatch: x K={k}, w packed K={w_packed_k} (expected {})",
            (k + 1) / 2
        )));
    }

    let num_groups = (k + group_size - 1) / group_size;
    if scales.shape.len() != 2
        || scales.shape[0] as usize != n
        || scales.shape[1] as usize != num_groups
    {
        return Err(unsupported(&format!(
            "w4a32_grouped_linear scales mismatch: expected [{n}, {num_groups}], got {:?}",
            scales.shape
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<u8>(w_packed) };
    let s_slice = unsafe { typed_slice::<f32>(scales) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };
    let bias_slice = bias.map(|b| unsafe { typed_slice::<f32>(b) });
    let b_ptr = bias_slice.map(|b| b.as_ptr());

    unsafe {
        gemm_w4a32_grouped(
            x_slice.as_ptr(),
            w_slice.as_ptr(),
            s_slice.as_ptr(),
            b_ptr,
            out_slice.as_mut_ptr(),
            m,
            n,
            k,
            group_size,
        );
    }

    Ok(out)
}

#[cfg(feature = "burn-wgpu")]
pub fn wgpu_w4a32_grouped_linear(
    x: &BorrowedTensor,
    w_packed: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
    group_size: usize,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "wgpu_w4a32_grouped_linear requires x with at least 1 dimension",
        ));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w_packed.shape.len() != 2 {
        return Err(unsupported(
            "wgpu_w4a32_grouped_linear requires 2D packed weight matrix",
        ));
    }
    let n = w_packed.shape[0] as usize;
    let w_packed_k = w_packed.shape[1] as usize;
    if (k + 1) / 2 != w_packed_k {
        return Err(unsupported(&format!(
            "wgpu_w4a32_grouped_linear dimension mismatch: x K={k}, w packed K={w_packed_k} (expected {})",
            (k + 1) / 2
        )));
    }

    let num_groups = (k + group_size - 1) / group_size;
    if scales.shape.len() != 2
        || scales.shape[0] as usize != n
        || scales.shape[1] as usize != num_groups
    {
        return Err(unsupported(&format!(
            "wgpu_w4a32_grouped_linear scales mismatch: expected [{n}, {num_groups}], got {:?}",
            scales.shape
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<u8>(w_packed) };
    let s_slice = unsafe { typed_slice::<f32>(scales) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    if m == 1 {
        crate::wgpu::backend::wgpu_gemv_w4a32(
            x_slice, w_slice, s_slice, out_slice, n, k, group_size,
        )
        .map_err(|e| unsupported(&e))?;
    } else {
        for i in 0..m {
            let x_tok = &x_slice[i * k..(i + 1) * k];
            let out_tok = &mut out_slice[i * n..(i + 1) * n];
            crate::wgpu::backend::wgpu_gemv_w4a32(
                x_tok, w_slice, s_slice, out_tok, n, k, group_size,
            )
            .map_err(|e| unsupported(&e))?;
        }
    }

    if let Some(b) = bias {
        let b_slice = unsafe { typed_slice::<f32>(b) };
        for i in 0..m {
            for j in 0..n {
                out_slice[i * n + j] += b_slice[j];
            }
        }
    }

    Ok(out)
}
