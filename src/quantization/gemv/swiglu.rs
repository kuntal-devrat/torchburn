//! SwiGLU neuron kernels for fused MLP (W4A8/W4A32, group32/64).
//! Inherits quantization root via super-super; pure move.

use super::super::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512f,avx512bw")]
pub(crate) unsafe fn swiglu_neuron_w4a8_group64_vnni_avx512(
    x_u8: *const u8,
    s_x: f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let ones = _mm512_set1_epi8(1);
    let mut g_acc = _mm512_setzero_ps();
    let mut u_acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x64 = _mm512_loadu_si512(x_u8.add(g * 64) as *const __m512i);
        let gw_grp = gw_row.add(g * 32);
        let uw_grp = uw_row.add(g * 32);
        let g_scale = _mm512_set1_ps(s_x * *gs_row.add(g));
        let u_scale = _mm512_set1_ps(s_x * *us_row.add(g));

        // Unpack Gate weights
        let g_raw0 = _mm_loadu_si128(gw_grp as *const __m128i);
        let g_lo0 = _mm_and_si128(g_raw0, mask_low);
        let g_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw0), mask_low);
        let g_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo0, g_hi0), sub8);
        let g_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo0, g_hi0), sub8);
        let gw32_0 = _mm256_set_m128i(g_s_hi0, g_s_lo0);

        let g_raw1 = _mm_loadu_si128(gw_grp.add(16) as *const __m128i);
        let g_lo1 = _mm_and_si128(g_raw1, mask_low);
        let g_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw1), mask_low);
        let g_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo1, g_hi1), sub8);
        let g_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo1, g_hi1), sub8);
        let gw32_1 = _mm256_set_m128i(g_s_hi1, g_s_lo1);

        let gw64 = _mm512_inserti64x4(_mm512_castsi256_si512(gw32_0), gw32_1, 1);

        // Gate VNNI
        let g_dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, gw64);
        let g_w_sum_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, gw64);
        let g_true_i32 = _mm512_sub_epi32(g_dot_i32, _mm512_slli_epi32::<7>(g_w_sum_i32));
        let g_true_f32 = _mm512_cvtepi32_ps(g_true_i32);
        g_acc = _mm512_fmadd_ps(g_true_f32, g_scale, g_acc);

        // Unpack Up weights
        let u_raw0 = _mm_loadu_si128(uw_grp as *const __m128i);
        let u_lo0 = _mm_and_si128(u_raw0, mask_low);
        let u_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw0), mask_low);
        let u_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo0, u_hi0), sub8);
        let u_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo0, u_hi0), sub8);
        let uw32_0 = _mm256_set_m128i(u_s_hi0, u_s_lo0);

        let u_raw1 = _mm_loadu_si128(uw_grp.add(16) as *const __m128i);
        let u_lo1 = _mm_and_si128(u_raw1, mask_low);
        let u_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw1), mask_low);
        let u_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo1, u_hi1), sub8);
        let u_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo1, u_hi1), sub8);
        let uw32_1 = _mm256_set_m128i(u_s_hi1, u_s_lo1);

        let uw64 = _mm512_inserti64x4(_mm512_castsi256_si512(uw32_0), uw32_1, 1);

        // Up VNNI
        let u_dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, uw64);
        let u_w_sum_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, uw64);
        let u_true_i32 = _mm512_sub_epi32(u_dot_i32, _mm512_slli_epi32::<7>(u_w_sum_i32));
        let u_true_f32 = _mm512_cvtepi32_ps(u_true_i32);
        u_acc = _mm512_fmadd_ps(u_true_f32, u_scale, u_acc);
    }

    (_mm512_reduce_add_ps(g_acc), _mm512_reduce_add_ps(u_acc))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn swiglu_neuron_w4a32_group64_avx512(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm512_setzero_ps();
    let mut u_acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let gw_grp = gw_row.add(g * 32);
        let uw_grp = uw_row.add(g * 32);
        let g_scale = _mm512_set1_ps(*gs_row.add(g));
        let u_scale = _mm512_set1_ps(*us_row.add(g));

        // Chunk 0 (elements 0..31)
        let xf0 = _mm512_loadu_ps(x_grp);
        let xf1 = _mm512_loadu_ps(x_grp.add(16));

        // Gate Chunk 0
        let g_raw0 = _mm_loadu_si128(gw_grp as *const __m128i);
        let g_lo0 = _mm_and_si128(g_raw0, mask_low);
        let g_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw0), mask_low);
        let g_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo0, g_hi0), sub8);
        let g_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo0, g_hi0), sub8);
        let gwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_lo0));
        let gwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_hi0));
        let g_sum0 = _mm512_fmadd_ps(gwf1, xf1, _mm512_mul_ps(gwf0, xf0));

        // Up Chunk 0
        let u_raw0 = _mm_loadu_si128(uw_grp as *const __m128i);
        let u_lo0 = _mm_and_si128(u_raw0, mask_low);
        let u_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw0), mask_low);
        let u_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo0, u_hi0), sub8);
        let u_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo0, u_hi0), sub8);
        let uwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_lo0));
        let uwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_hi0));
        let u_sum0 = _mm512_fmadd_ps(uwf1, xf1, _mm512_mul_ps(uwf0, xf0));

        // Chunk 1 (elements 32..63)
        let xf2 = _mm512_loadu_ps(x_grp.add(32));
        let xf3 = _mm512_loadu_ps(x_grp.add(48));

        // Gate Chunk 1
        let g_raw1 = _mm_loadu_si128(gw_grp.add(16) as *const __m128i);
        let g_lo1 = _mm_and_si128(g_raw1, mask_low);
        let g_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw1), mask_low);
        let g_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo1, g_hi1), sub8);
        let g_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo1, g_hi1), sub8);
        let gwf2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_lo1));
        let gwf3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_hi1));
        let g_sum1 = _mm512_fmadd_ps(gwf3, xf3, _mm512_mul_ps(gwf2, xf2));

        // Up Chunk 1
        let u_raw1 = _mm_loadu_si128(uw_grp.add(16) as *const __m128i);
        let u_lo1 = _mm_and_si128(u_raw1, mask_low);
        let u_hi1 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw1), mask_low);
        let u_s_lo1 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo1, u_hi1), sub8);
        let u_s_hi1 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo1, u_hi1), sub8);
        let uwf2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_lo1));
        let uwf3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_hi1));
        let u_sum1 = _mm512_fmadd_ps(uwf3, xf3, _mm512_mul_ps(uwf2, xf2));

        let g_grp_sum = _mm512_add_ps(g_sum0, g_sum1);
        let u_grp_sum = _mm512_add_ps(u_sum0, u_sum1);

        g_acc = _mm512_fmadd_ps(g_grp_sum, g_scale, g_acc);
        u_acc = _mm512_fmadd_ps(u_grp_sum, u_scale, u_acc);
    }

    (_mm512_reduce_add_ps(g_acc), _mm512_reduce_add_ps(u_acc))
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn swiglu_neuron_w4a8_group64_vnni_avx512(
    _x_u8: *const u8,
    _s_x: f32,
    _gw_row: *const u8,
    _gs_row: *const f32,
    _uw_row: *const u8,
    _us_row: *const f32,
    _num_groups: usize,
) -> (f32, f32) {
    (0.0, 0.0)
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn swiglu_neuron_w4a32_group64_avx512(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    let mut g_acc = 0.0f32;
    let mut u_acc = 0.0f32;
    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let gw_grp = gw_row.add(g * 32);
        let uw_grp = uw_row.add(g * 32);
        let g_scale = *gs_row.add(g);
        let u_scale = *us_row.add(g);
        let g_dot = dot_f32_u4_group_scalar(x_grp, gw_grp, 64);
        let u_dot = dot_f32_u4_group_scalar(x_grp, uw_grp, 64);
        g_acc += g_dot * g_scale;
        u_acc += u_dot * u_scale;
    }
    (g_acc, u_acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn swiglu_neuron_w4a32_group32_avx512(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm512_setzero_ps();
    let mut u_acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let gw_grp = gw_row.add(g * 16);
        let uw_grp = uw_row.add(g * 16);
        let g_scale = _mm512_set1_ps(*gs_row.add(g));
        let u_scale = _mm512_set1_ps(*us_row.add(g));

        let xf0 = _mm512_loadu_ps(x_grp);
        let xf1 = _mm512_loadu_ps(x_grp.add(16));

        // Gate
        let g_raw0 = _mm_loadu_si128(gw_grp as *const __m128i);
        let g_lo0 = _mm_and_si128(g_raw0, mask_low);
        let g_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(g_raw0), mask_low);
        let g_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(g_lo0, g_hi0), sub8);
        let g_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(g_lo0, g_hi0), sub8);
        let gwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_lo0));
        let gwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(g_s_hi0));
        let g_sum0 = _mm512_fmadd_ps(gwf1, xf1, _mm512_mul_ps(gwf0, xf0));

        // Up
        let u_raw0 = _mm_loadu_si128(uw_grp as *const __m128i);
        let u_lo0 = _mm_and_si128(u_raw0, mask_low);
        let u_hi0 = _mm_and_si128(_mm_srli_epi16::<4>(u_raw0), mask_low);
        let u_s_lo0 = _mm_sub_epi8(_mm_unpacklo_epi8(u_lo0, u_hi0), sub8);
        let u_s_hi0 = _mm_sub_epi8(_mm_unpackhi_epi8(u_lo0, u_hi0), sub8);
        let uwf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_lo0));
        let uwf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(u_s_hi0));
        let u_sum0 = _mm512_fmadd_ps(uwf1, xf1, _mm512_mul_ps(uwf0, xf0));

        g_acc = _mm512_fmadd_ps(g_sum0, g_scale, g_acc);
        u_acc = _mm512_fmadd_ps(u_sum0, u_scale, u_acc);
    }

    (_mm512_reduce_add_ps(g_acc), _mm512_reduce_add_ps(u_acc))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn swiglu_neuron_w4a32_group32_avx2(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm256_setzero_ps();
    let mut u_acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let gw_grp = gw_row.add(g * 16);
        let uw_grp = uw_row.add(g * 16);
        let g_scale = _mm256_set1_ps(*gs_row.add(g));
        let u_scale = _mm256_set1_ps(*us_row.add(g));

        let g_sum0 = unpack_and_fma_32_avx2(x_grp, gw_grp, mask_low, sub8);
        let u_sum0 = unpack_and_fma_32_avx2(x_grp, uw_grp, mask_low, sub8);

        g_acc = _mm256_fmadd_ps(g_sum0, g_scale, g_acc);
        u_acc = _mm256_fmadd_ps(u_sum0, u_scale, u_acc);
    }

    (hsum256_ps_avx(g_acc), hsum256_ps_avx(u_acc))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn swiglu_neuron_w4a32_group64_avx2(
    x: *const f32,
    gw_row: *const u8,
    gs_row: *const f32,
    uw_row: *const u8,
    us_row: *const f32,
    num_groups: usize,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    let mut g_acc = _mm256_setzero_ps();
    let mut u_acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let gw_grp = gw_row.add(g * 32);
        let uw_grp = uw_row.add(g * 32);
        let g_scale = _mm256_set1_ps(*gs_row.add(g));
        let u_scale = _mm256_set1_ps(*us_row.add(g));

        let g_sum0 = unpack_and_fma_32_avx2(x_grp, gw_grp, mask_low, sub8);
        let g_sum1 = unpack_and_fma_32_avx2(x_grp.add(32), gw_grp.add(16), mask_low, sub8);
        let g_grp_sum = _mm256_add_ps(g_sum0, g_sum1);

        let u_sum0 = unpack_and_fma_32_avx2(x_grp, uw_grp, mask_low, sub8);
        let u_sum1 = unpack_and_fma_32_avx2(x_grp.add(32), uw_grp.add(16), mask_low, sub8);
        let u_grp_sum = _mm256_add_ps(u_sum0, u_sum1);

        g_acc = _mm256_fmadd_ps(g_grp_sum, g_scale, g_acc);
        u_acc = _mm256_fmadd_ps(u_grp_sum, u_scale, u_acc);
    }

    (hsum256_ps_avx(g_acc), hsum256_ps_avx(u_acc))
}
