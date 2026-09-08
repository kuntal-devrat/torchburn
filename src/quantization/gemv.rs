//! W8A32/W4A32 GEMV kernels (v1 layout): AVX2/AVX-512/VNNI dots,
//! grouped rows, swiglu neurons. Inherits root imports via super.

use super::*;
// ---------------------------------------------------------------------------
// Native AVX2 / SIMD Quantized Linear Kernels (W8A32 & W4A32)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn hsum256_ps_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let v_hi = _mm256_extractf128_ps::<1>(v);
    let v_lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(v_hi, v_lo);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<0x55>(sum64, sum64));
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_4rows_w8a32_avx2(
    x: *const f32,
    w0: *const i8,
    w1: *const i8,
    w2: *const i8,
    w3: *const i8,
    len: usize,
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;

    let mut acc0_0 = _mm256_setzero_ps();
    let mut acc0_1 = _mm256_setzero_ps();
    let mut acc1_0 = _mm256_setzero_ps();
    let mut acc1_1 = _mm256_setzero_ps();
    let mut acc2_0 = _mm256_setzero_ps();
    let mut acc2_1 = _mm256_setzero_ps();
    let mut acc3_0 = _mm256_setzero_ps();
    let mut acc3_1 = _mm256_setzero_ps();

    let chunks16 = len / 16;
    let mut offset = 0;

    for _ in 0..chunks16 {
        let x0 = _mm256_loadu_ps(x.add(offset));
        let x1 = _mm256_loadu_ps(x.add(offset + 8));

        // Row 0
        let wf0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w0.add(offset) as *const __m128i
        )));
        acc0_0 = _mm256_fmadd_ps(wf0_0, x0, acc0_0);

        let wf0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w0.add(offset + 8) as *const __m128i
        )));
        acc0_1 = _mm256_fmadd_ps(wf0_1, x1, acc0_1);

        // Row 1
        let wf1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w1.add(offset) as *const __m128i
        )));
        acc1_0 = _mm256_fmadd_ps(wf1_0, x0, acc1_0);

        let wf1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w1.add(offset + 8) as *const __m128i
        )));
        acc1_1 = _mm256_fmadd_ps(wf1_1, x1, acc1_1);

        // Row 2
        let wf2_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w2.add(offset) as *const __m128i
        )));
        acc2_0 = _mm256_fmadd_ps(wf2_0, x0, acc2_0);

        let wf2_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w2.add(offset + 8) as *const __m128i
        )));
        acc2_1 = _mm256_fmadd_ps(wf2_1, x1, acc2_1);

        // Row 3
        let wf3_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w3.add(offset) as *const __m128i
        )));
        acc3_0 = _mm256_fmadd_ps(wf3_0, x0, acc3_0);

        let wf3_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w3.add(offset + 8) as *const __m128i
        )));
        acc3_1 = _mm256_fmadd_ps(wf3_1, x1, acc3_1);

        offset += 16;
    }

    let mut sum0 = _mm256_add_ps(acc0_0, acc0_1);
    let mut sum1 = _mm256_add_ps(acc1_0, acc1_1);
    let mut sum2 = _mm256_add_ps(acc2_0, acc2_1);
    let mut sum3 = _mm256_add_ps(acc3_0, acc3_1);

    let chunks8 = (len - offset) / 8;
    for _ in 0..chunks8 {
        let x_vec = _mm256_loadu_ps(x.add(offset));

        let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w0.add(offset) as *const __m128i
        )));
        sum0 = _mm256_fmadd_ps(wf0, x_vec, sum0);

        let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w1.add(offset) as *const __m128i
        )));
        sum1 = _mm256_fmadd_ps(wf1, x_vec, sum1);

        let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w2.add(offset) as *const __m128i
        )));
        sum2 = _mm256_fmadd_ps(wf2, x_vec, sum2);

        let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w3.add(offset) as *const __m128i
        )));
        sum3 = _mm256_fmadd_ps(wf3, x_vec, sum3);

        offset += 8;
    }

    let mut tot0 = hsum256_ps_avx(sum0);
    let mut tot1 = hsum256_ps_avx(sum1);
    let mut tot2 = hsum256_ps_avx(sum2);
    let mut tot3 = hsum256_ps_avx(sum3);

    while offset < len {
        let xv = *x.add(offset);
        tot0 += xv * (*w0.add(offset) as f32);
        tot1 += xv * (*w1.add(offset) as f32);
        tot2 += xv * (*w2.add(offset) as f32);
        tot3 += xv * (*w3.add(offset) as f32);
        offset += 1;
    }

    (tot0, tot1, tot2, tot3)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemv_8rows_w8a32_avx512(
    x: *const f32,
    w_base: *const i8,
    stride: usize,
    len: usize,
) -> [f32; 8] {
    use std::arch::x86_64::*;

    let w0 = w_base;
    let w1 = w_base.add(stride);
    let w2 = w_base.add(stride * 2);
    let w3 = w_base.add(stride * 3);
    let w4 = w_base.add(stride * 4);
    let w5 = w_base.add(stride * 5);
    let w6 = w_base.add(stride * 6);
    let w7 = w_base.add(stride * 7);

    let mut acc0_0 = _mm512_setzero_ps();
    let mut acc0_1 = _mm512_setzero_ps();
    let mut acc1_0 = _mm512_setzero_ps();
    let mut acc1_1 = _mm512_setzero_ps();
    let mut acc2_0 = _mm512_setzero_ps();
    let mut acc2_1 = _mm512_setzero_ps();
    let mut acc3_0 = _mm512_setzero_ps();
    let mut acc3_1 = _mm512_setzero_ps();
    let mut acc4_0 = _mm512_setzero_ps();
    let mut acc4_1 = _mm512_setzero_ps();
    let mut acc5_0 = _mm512_setzero_ps();
    let mut acc5_1 = _mm512_setzero_ps();
    let mut acc6_0 = _mm512_setzero_ps();
    let mut acc6_1 = _mm512_setzero_ps();
    let mut acc7_0 = _mm512_setzero_ps();
    let mut acc7_1 = _mm512_setzero_ps();

    let chunks32 = len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let x0 = _mm512_loadu_ps(x.add(offset));
        let x1 = _mm512_loadu_ps(x.add(offset + 16));

        let load_wf = |w_ptr: *const i8, off: usize| -> __m512 {
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                w_ptr.add(off) as *const __m128i
            )))
        };

        acc0_0 = _mm512_fmadd_ps(load_wf(w0, offset), x0, acc0_0);
        acc0_1 = _mm512_fmadd_ps(load_wf(w0, offset + 16), x1, acc0_1);

        acc1_0 = _mm512_fmadd_ps(load_wf(w1, offset), x0, acc1_0);
        acc1_1 = _mm512_fmadd_ps(load_wf(w1, offset + 16), x1, acc1_1);

        acc2_0 = _mm512_fmadd_ps(load_wf(w2, offset), x0, acc2_0);
        acc2_1 = _mm512_fmadd_ps(load_wf(w2, offset + 16), x1, acc2_1);

        acc3_0 = _mm512_fmadd_ps(load_wf(w3, offset), x0, acc3_0);
        acc3_1 = _mm512_fmadd_ps(load_wf(w3, offset + 16), x1, acc3_1);

        acc4_0 = _mm512_fmadd_ps(load_wf(w4, offset), x0, acc4_0);
        acc4_1 = _mm512_fmadd_ps(load_wf(w4, offset + 16), x1, acc4_1);

        acc5_0 = _mm512_fmadd_ps(load_wf(w5, offset), x0, acc5_0);
        acc5_1 = _mm512_fmadd_ps(load_wf(w5, offset + 16), x1, acc5_1);

        acc6_0 = _mm512_fmadd_ps(load_wf(w6, offset), x0, acc6_0);
        acc6_1 = _mm512_fmadd_ps(load_wf(w6, offset + 16), x1, acc6_1);

        acc7_0 = _mm512_fmadd_ps(load_wf(w7, offset), x0, acc7_0);
        acc7_1 = _mm512_fmadd_ps(load_wf(w7, offset + 16), x1, acc7_1);

        offset += 32;
    }

    let mut sum0 = _mm512_add_ps(acc0_0, acc0_1);
    let mut sum1 = _mm512_add_ps(acc1_0, acc1_1);
    let mut sum2 = _mm512_add_ps(acc2_0, acc2_1);
    let mut sum3 = _mm512_add_ps(acc3_0, acc3_1);
    let mut sum4 = _mm512_add_ps(acc4_0, acc4_1);
    let mut sum5 = _mm512_add_ps(acc5_0, acc5_1);
    let mut sum6 = _mm512_add_ps(acc6_0, acc6_1);
    let mut sum7 = _mm512_add_ps(acc7_0, acc7_1);

    let chunks16 = (len - offset) / 16;
    for _ in 0..chunks16 {
        let x0 = _mm512_loadu_ps(x.add(offset));
        let load_wf = |w_ptr: *const i8, off: usize| -> __m512 {
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
                w_ptr.add(off) as *const __m128i
            )))
        };

        sum0 = _mm512_fmadd_ps(load_wf(w0, offset), x0, sum0);
        sum1 = _mm512_fmadd_ps(load_wf(w1, offset), x0, sum1);
        sum2 = _mm512_fmadd_ps(load_wf(w2, offset), x0, sum2);
        sum3 = _mm512_fmadd_ps(load_wf(w3, offset), x0, sum3);
        sum4 = _mm512_fmadd_ps(load_wf(w4, offset), x0, sum4);
        sum5 = _mm512_fmadd_ps(load_wf(w5, offset), x0, sum5);
        sum6 = _mm512_fmadd_ps(load_wf(w6, offset), x0, sum6);
        sum7 = _mm512_fmadd_ps(load_wf(w7, offset), x0, sum7);

        offset += 16;
    }

    let mut tot = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
        _mm512_reduce_add_ps(sum4),
        _mm512_reduce_add_ps(sum5),
        _mm512_reduce_add_ps(sum6),
        _mm512_reduce_add_ps(sum7),
    ];

    while offset < len {
        let xv = *x.add(offset);
        tot[0] += xv * (*w0.add(offset) as f32);
        tot[1] += xv * (*w1.add(offset) as f32);
        tot[2] += xv * (*w2.add(offset) as f32);
        tot[3] += xv * (*w3.add(offset) as f32);
        tot[4] += xv * (*w4.add(offset) as f32);
        tot[5] += xv * (*w5.add(offset) as f32);
        tot[6] += xv * (*w6.add(offset) as f32);
        tot[7] += xv * (*w7.add(offset) as f32);
        offset += 1;
    }

    tot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_f32_i8_avx512(x: *const f32, w: *const i8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();

    let chunks32 = len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
            w.add(offset) as *const __m128i
        )));
        let xf0 = _mm512_loadu_ps(x.add(offset));
        sum0 = _mm512_fmadd_ps(wf0, xf0, sum0);

        let wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
            w.add(offset + 16) as *const __m128i
        )));
        let xf1 = _mm512_loadu_ps(x.add(offset + 16));
        sum1 = _mm512_fmadd_ps(wf1, xf1, sum1);

        offset += 32;
    }

    let mut sum = _mm512_add_ps(sum0, sum1);

    let chunks16 = (len - offset) / 16;
    for _ in 0..chunks16 {
        let wf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(
            w.add(offset) as *const __m128i
        )));
        let xf = _mm512_loadu_ps(x.add(offset));
        sum = _mm512_fmadd_ps(wf, xf, sum);
        offset += 16;
    }

    let mut total = _mm512_reduce_add_ps(sum);

    while offset < len {
        total += (*x.add(offset)) * ((*w.add(offset)) as f32);
        offset += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_avx2(x: *const f32, w: *const i8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks32 = len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w.add(offset) as *const __m128i
        )));
        let xf0 = _mm256_loadu_ps(x.add(offset));
        sum0 = _mm256_fmadd_ps(wf0, xf0, sum0);

        let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w.add(offset + 8) as *const __m128i
        )));
        let xf1 = _mm256_loadu_ps(x.add(offset + 8));
        sum1 = _mm256_fmadd_ps(wf1, xf1, sum1);

        let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w.add(offset + 16) as *const __m128i
        )));
        let xf2 = _mm256_loadu_ps(x.add(offset + 16));
        sum2 = _mm256_fmadd_ps(wf2, xf2, sum2);

        let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w.add(offset + 24) as *const __m128i
        )));
        let xf3 = _mm256_loadu_ps(x.add(offset + 24));
        sum3 = _mm256_fmadd_ps(wf3, xf3, sum3);

        offset += 32;
    }

    let mut sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    let chunks8 = (len - offset) / 8;
    for _ in 0..chunks8 {
        let wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            w.add(offset) as *const __m128i
        )));
        let xf = _mm256_loadu_ps(x.add(offset));
        sum = _mm256_fmadd_ps(wf, xf, sum);
        offset += 8;
    }

    let mut total = hsum256_ps_avx(sum);

    while offset < len {
        total += (*x.add(offset)) * ((*w.add(offset)) as f32);
        offset += 1;
    }

    total
}

/// Dot product of an f32 activation row with an int8 weight row.
///
/// # Safety
/// `x` and `w` must point to at least `len` readable elements.
#[inline(always)]
pub unsafe fn dot_f32_i8(x: *const f32, w: *const i8, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx512f && crate::dispatch::cpu_features().avx512bw {
            return dot_f32_i8_avx512(x, w, len);
        }
        if crate::dispatch::cpu_features().avx2 && crate::dispatch::cpu_features().fma {
            return dot_f32_i8_avx2(x, w, len);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::dispatch::cpu_features().neon {
            return dot_f32_i8_neon(x, w, len);
        }
    }
    let mut total = 0.0f32;
    for i in 0..len {
        total += *x.add(i) * (*w.add(i) as f32);
    }
    total
}

// ---------------------------------------------------------------------------
// ARM NEON SIMD Kernels
// ---------------------------------------------------------------------------

/// ARM NEON dot product: f32 activation × int8 weight → f32.
/// Processes 16 elements per iteration using `vmull_s8` + `vmlal_s16`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_i8_neon(x: *const f32, w: *const i8, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut sum_f32x4 = vdupq_n_f32(0.0);
    let mut offset = 0;
    let chunks16 = len / 16;

    for _ in 0..chunks16 {
        // Load 16 i8 weights
        let w_i8 = vld1q_s8(w.add(offset));
        // Load 16 f32 activations (4x vld1q_f32)
        let x0 = vld1q_f32(x.add(offset));
        let x1 = vld1q_f32(x.add(offset + 4));
        let x2 = vld1q_f32(x.add(offset + 8));
        let x3 = vld1q_f32(x.add(offset + 12));

        // Unpack i8 → i16 pairs
        let w_lo_i16 = vmovl_s8(vget_low_s8(w_i8));
        let w_hi_i16 = vmovl_s8(vget_high_s8(w_i8));

        // Widen i16 → i32, then FMA with f32 (via conversion)
        // Process low 8 elements
        let w_i32_0 = vmovl_s16(vget_low_s16(w_lo_i16));
        let w_i32_1 = vmovl_s16(vget_high_s16(w_lo_i16));
        let w_i32_2 = vmovl_s16(vget_low_s16(w_hi_i16));
        let w_i32_3 = vmovl_s16(vget_high_s16(w_hi_i16));

        // Multiply-accumulate: f32 × i32 → f32 via vcvtq_f32_s32
        sum_f32x4 = vmlaq_f32(sum_f32x4, x0, vcvtq_f32_s32(w_i32_0));
        sum_f32x4 = vmlaq_f32(sum_f32x4, x1, vcvtq_f32_s32(w_i32_1));
        sum_f32x4 = vmlaq_f32(sum_f32x4, x2, vcvtq_f32_s32(w_i32_2));
        sum_f32x4 = vmlaq_f32(sum_f32x4, x3, vcvtq_f32_s32(w_i32_3));

        offset += 16;
    }

    // Horizontal sum
    let mut total = vaddvq_f32(sum_f32x4);

    while offset < len {
        total += *x.add(offset) * (*w.add(offset) as f32);
        offset += 1;
    }
    total
}

/// ARM NEON dot product: f32 activation × packed int4 weight → f32.
/// Processes 16 elements (8 bytes) per iteration.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_u4_neon(x: *const f32, w_packed: *const u8, len: usize) -> f32 {
    use std::arch::aarch64::*;
    // 64-bit (uint8x8_t) nibble pipeline: 8 packed bytes = 16 int4 values.
    // (`vld1_u8` loads 8 lanes, so all ops below are the non-`q` 64-bit
    // variants: `vand_u8`, `vshr_n_u8`, `vzip_u8`, `vdup_n_*`.)
    let mask_low = vdup_n_u8(0x0F);
    let sub8 = vdup_n_s8(8);
    let zero_f32 = vdupq_n_f32(0.0);
    let mut sum_f32x4 = zero_f32;
    let mut offset = 0;
    let chunks16 = len / 16;

    for _ in 0..chunks16 {
        let byte_offset = offset / 2;
        // Load 8 packed bytes (16 int4 values)
        let raw_u8 = vld1_u8(w_packed.add(byte_offset));

        // Extract low and high nibbles
        let lo = vand_u8(raw_u8, mask_low);
        let hi = vshr_n_u8::<4>(raw_u8);

        // Unpack to i8: interleaving lo,hi gives element order
        // [l0,h0,l1,h1,...]: zip.0 = elements 0..8, zip.1 = elements 8..16.
        let zipped = vzip_u8(lo, hi);
        let inter_lo = vreinterpret_s8_u8(zipped.0);
        let inter_hi = vreinterpret_s8_u8(zipped.1);

        // Subtract 8 to get signed int4
        let s_lo = vsub_s8(inter_lo, sub8);
        let s_hi = vsub_s8(inter_hi, sub8);

        // Widen i8 → i16 → i32 → f32 (vmovl_s16 takes int16x4_t halves)
        let w16_lo = vmovl_s8(s_lo);
        let w16_hi = vmovl_s8(s_hi);
        let w_i32_0 = vmovl_s16(vget_low_s16(w16_lo));
        let w_i32_1 = vmovl_s16(vget_high_s16(w16_lo));
        let w_i32_2 = vmovl_s16(vget_low_s16(w16_hi));
        let w_i32_3 = vmovl_s16(vget_high_s16(w16_hi));

        let x0 = vld1q_f32(x.add(offset));
        let x1 = vld1q_f32(x.add(offset + 4));
        let x2 = vld1q_f32(x.add(offset + 8));
        let x3 = vld1q_f32(x.add(offset + 12));

        sum_f32x4 = vmlaq_f32(sum_f32x4, x0, vcvtq_f32_s32(w_i32_0));
        sum_f32x4 = vmlaq_f32(sum_f32x4, x1, vcvtq_f32_s32(w_i32_1));
        sum_f32x4 = vmlaq_f32(sum_f32x4, x2, vcvtq_f32_s32(w_i32_2));
        sum_f32x4 = vmlaq_f32(sum_f32x4, x3, vcvtq_f32_s32(w_i32_3));

        offset += 16;
    }

    let mut total = vaddvq_f32(sum_f32x4);

    while offset < len {
        let byte = *w_packed.add(offset / 2);
        let q = if (offset % 2) == 0 {
            ((byte & 0x0F) as i8) - 8
        } else {
            (((byte >> 4) & 0x0F) as i8) - 8
        };
        total += *x.add(offset) * (q as f32);
        offset += 1;
    }
    total
}

/// ARM NEON: 64-element grouped INT4 dot product with group-wise scale.
/// Equivalent to `dot_f32_u4_group64_avx512` / `dot_f32_u4_group64_avx2`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_u4_group64_neon(x: *const f32, w_packed: *const u8) -> f32 {
    use std::arch::aarch64::*;
    // 64-bit (uint8x8_t) nibble pipeline — see `dot_f32_u4_neon`.
    let mask_low = vdup_n_u8(0x0F);
    let sub8 = vdup_n_s8(8);
    let mut sum = vdupq_n_f32(0.0);

    // Process 64 elements = 32 bytes = 2 chunks of 16 elements (8 bytes each).
    for chunk in 0..2 {
        let w_off = chunk * 8;
        let x_off = chunk * 16;

        let raw_u8 = vld1_u8(w_packed.add(w_off));
        let lo = vand_u8(raw_u8, mask_low);
        let hi = vshr_n_u8::<4>(raw_u8);
        let zipped = vzip_u8(lo, hi);
        let s_lo = vsub_s8(vreinterpret_s8_u8(zipped.0), sub8);
        let s_hi = vsub_s8(vreinterpret_s8_u8(zipped.1), sub8);

        let w16_lo = vmovl_s8(s_lo);
        let w16_hi = vmovl_s8(s_hi);
        let w_i32_0 = vmovl_s16(vget_low_s16(w16_lo));
        let w_i32_1 = vmovl_s16(vget_high_s16(w16_lo));
        let w_i32_2 = vmovl_s16(vget_low_s16(w16_hi));
        let w_i32_3 = vmovl_s16(vget_high_s16(w16_hi));

        let x0 = vld1q_f32(x.add(x_off));
        let x1 = vld1q_f32(x.add(x_off + 4));
        let x2 = vld1q_f32(x.add(x_off + 8));
        let x3 = vld1q_f32(x.add(x_off + 12));

        sum = vmlaq_f32(sum, x0, vcvtq_f32_s32(w_i32_0));
        sum = vmlaq_f32(sum, x1, vcvtq_f32_s32(w_i32_1));
        sum = vmlaq_f32(sum, x2, vcvtq_f32_s32(w_i32_2));
        sum = vmlaq_f32(sum, x3, vcvtq_f32_s32(w_i32_3));
    }

    vaddvq_f32(sum)
}

/// ARM NEON: full row GEMV with group-wise scales (group_size=64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemv_row_w4a32_group64_neon(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    let bytes_per_group = 64 / 2; // 32 bytes per group
    let mut row_sum = 0.0f32;
    for g in 0..num_groups {
        let dot = dot_f32_u4_group64_neon(x.add(g * 64), w_row.add(g * bytes_per_group));
        row_sum += dot * *s_row.add(g);
    }
    row_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u4_avx2(x: *const f32, w_packed: *const u8, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let chunks32 = len / 32;
    let mut offset = 0;

    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    for _ in 0..chunks32 {
        let byte_offset = offset / 2;
        let raw = _mm_loadu_si128(w_packed.add(byte_offset) as *const __m128i);

        let lo = _mm_and_si128(raw, mask_low);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask_low);

        let inter_lo = _mm_unpacklo_epi8(lo, hi);
        let inter_hi = _mm_unpackhi_epi8(lo, hi);

        let s_lo = _mm_sub_epi8(inter_lo, sub8);
        let s_hi = _mm_sub_epi8(inter_hi, sub8);

        let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo));
        let xf0 = _mm256_loadu_ps(x.add(offset));
        sum0 = _mm256_fmadd_ps(wf0, xf0, sum0);

        let s_lo_hi = _mm_unpackhi_epi64(s_lo, s_lo);
        let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo_hi));
        let xf1 = _mm256_loadu_ps(x.add(offset + 8));
        sum1 = _mm256_fmadd_ps(wf1, xf1, sum1);

        let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi));
        let xf2 = _mm256_loadu_ps(x.add(offset + 16));
        sum0 = _mm256_fmadd_ps(wf2, xf2, sum0);

        let s_hi_hi = _mm_unpackhi_epi64(s_hi, s_hi);
        let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi_hi));
        let xf3 = _mm256_loadu_ps(x.add(offset + 24));
        sum1 = _mm256_fmadd_ps(wf3, xf3, sum1);

        offset += 32;
    }

    let sum = _mm256_add_ps(sum0, sum1);
    let mut total = hsum256_ps_avx(sum);

    while offset < len {
        let byte_idx = offset / 2;
        let byte = *w_packed.add(byte_idx);
        let q = if (offset % 2) == 0 {
            ((byte & 0x0F) as i8) - 8
        } else {
            (((byte >> 4) & 0x0F) as i8) - 8
        };
        total += (*x.add(offset)) * (q as f32);
        offset += 1;
    }

    total
}

#[inline(always)]
unsafe fn dot_f32_u4(x: *const f32, w_packed: *const u8, len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 && crate::dispatch::cpu_features().fma {
            return dot_f32_u4_avx2(x, w_packed, len);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::dispatch::cpu_features().neon {
            return dot_f32_u4_neon(x, w_packed, len);
        }
    }
    let mut total = 0.0f32;
    for i in 0..len {
        let byte = *w_packed.add(i / 2);
        let q = if (i % 2) == 0 {
            ((byte & 0x0F) as i8) - 8
        } else {
            (((byte >> 4) & 0x0F) as i8) - 8
        };
        total += *x.add(i) * (q as f32);
    }
    total
}

/// Fast single-token GEMV (M=1) for W8A32 with 8-row AVX-512 or 4-row AVX2 unrolling and chunked Rayon scheduling.
pub(crate) unsafe fn gemv_w8a32(
    x: *const f32,
    w: *const i8,
    scales: *const f32,
    s_len: usize,
    bias: Option<*const f32>,
    out: *mut f32,
    n: usize,
    k: usize,
) {
    use rayon::prelude::*;

    let has_avx512 = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::dispatch::cpu_features().avx512f && crate::dispatch::cpu_features().avx512bw
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    if has_avx512 {
        let n_octs = n / 8;

        #[inline(always)]
        unsafe fn process_oct(
            oct: usize,
            x: *const f32,
            w: *const i8,
            scales: *const f32,
            s_len: usize,
            bias: Option<*const f32>,
            out: *mut f32,
            k: usize,
        ) {
            let j = oct * 8;
            let w_base = w.add(j * k);
            #[cfg(target_arch = "x86_64")]
            let dots = gemv_8rows_w8a32_avx512(x, w_base, k, k);
            #[cfg(not(target_arch = "x86_64"))]
            let mut dots = [0.0f32; 8];
            #[cfg(not(target_arch = "x86_64"))]
            for r in 0..8 {
                dots[r] = dot_f32_i8(x, w.add((j + r) * k), k);
            }

            for r in 0..8 {
                let idx = j + r;
                let s = if s_len > 1 { *scales.add(idx) } else { *scales };
                let b = if let Some(bp) = bias {
                    *bp.add(idx)
                } else {
                    0.0
                };
                *out.add(idx) = dots[r] * s + b;
            }
        }

        if n <= 256 {
            for oct in 0..n_octs {
                process_oct(oct, x, w, scales, s_len, bias, out, k);
            }
        } else {
            let x_usize = x as usize;
            let w_usize = w as usize;
            let s_usize = scales as usize;
            let b_usize = bias.map(|bp| bp as usize);
            let out_usize = out as usize;

            let n_threads = rayon::current_num_threads();
            let min_chunk = (n_octs / (n_threads * 2)).max(8);

            (0..n_octs)
                .into_par_iter()
                .with_min_len(min_chunk)
                .for_each(|oct| {
                    let x_p = x_usize as *const f32;
                    let w_p = w_usize as *const i8;
                    let s_p = s_usize as *const f32;
                    let b_p = b_usize.map(|bp| bp as *const f32);
                    let out_p = out_usize as *mut f32;
                    unsafe {
                        process_oct(oct, x_p, w_p, s_p, s_len, b_p, out_p, k);
                    }
                });
        }

        // Remainder rows
        let rem_start = n_octs * 8;
        for j in rem_start..n {
            let w_row = w.add(j * k);
            let dot = dot_f32_i8(x, w_row, k);
            let scale = if s_len > 1 { *scales.add(j) } else { *scales };
            let b = if let Some(bp) = bias { *bp.add(j) } else { 0.0 };
            *out.add(j) = dot * scale + b;
        }
        return;
    }

    // AVX2 / fallback path
    let n_quads = n / 4;

    #[inline(always)]
    unsafe fn process_quad(
        q: usize,
        x: *const f32,
        w: *const i8,
        scales: *const f32,
        s_len: usize,
        bias: Option<*const f32>,
        out: *mut f32,
        k: usize,
        has_avx2: bool,
    ) {
        let j = q * 4;
        let w0 = w.add(j * k);
        let w1 = w.add((j + 1) * k);
        let w2 = w.add((j + 2) * k);
        let w3 = w.add((j + 3) * k);

        let (d0, d1, d2, d3) = if has_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                gemv_4rows_w8a32_avx2(x, w0, w1, w2, w3, k)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                (
                    dot_f32_i8(x, w0, k),
                    dot_f32_i8(x, w1, k),
                    dot_f32_i8(x, w2, k),
                    dot_f32_i8(x, w3, k),
                )
            }
        } else {
            (
                dot_f32_i8(x, w0, k),
                dot_f32_i8(x, w1, k),
                dot_f32_i8(x, w2, k),
                dot_f32_i8(x, w3, k),
            )
        };

        let s0 = if s_len > 1 { *scales.add(j) } else { *scales };
        let s1 = if s_len > 1 {
            *scales.add(j + 1)
        } else {
            *scales
        };
        let s2 = if s_len > 1 {
            *scales.add(j + 2)
        } else {
            *scales
        };
        let s3 = if s_len > 1 {
            *scales.add(j + 3)
        } else {
            *scales
        };

        let b0 = if let Some(bp) = bias { *bp.add(j) } else { 0.0 };
        let b1 = if let Some(bp) = bias {
            *bp.add(j + 1)
        } else {
            0.0
        };
        let b2 = if let Some(bp) = bias {
            *bp.add(j + 2)
        } else {
            0.0
        };
        let b3 = if let Some(bp) = bias {
            *bp.add(j + 3)
        } else {
            0.0
        };

        *out.add(j) = d0 * s0 + b0;
        *out.add(j + 1) = d1 * s1 + b1;
        *out.add(j + 2) = d2 * s2 + b2;
        *out.add(j + 3) = d3 * s3 + b3;
    }

    let has_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::dispatch::cpu_features().avx2 && crate::dispatch::cpu_features().fma
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    // If N <= 256, execute sequentially to bypass Rayon thread dispatch overhead completely!
    if n <= 256 {
        for q in 0..n_quads {
            process_quad(q, x, w, scales, s_len, bias, out, k, has_avx2);
        }
    } else {
        let x_usize = x as usize;
        let w_usize = w as usize;
        let s_usize = scales as usize;
        let b_usize = bias.map(|bp| bp as usize);
        let out_usize = out as usize;

        let n_threads = rayon::current_num_threads();
        let min_chunk = (n_quads / (n_threads * 2)).max(16);

        (0..n_quads)
            .into_par_iter()
            .with_min_len(min_chunk)
            .for_each(|q| {
                let x_p = x_usize as *const f32;
                let w_p = w_usize as *const i8;
                let s_p = s_usize as *const f32;
                let b_p = b_usize.map(|bp| bp as *const f32);
                let out_p = out_usize as *mut f32;
                unsafe {
                    process_quad(q, x_p, w_p, s_p, s_len, b_p, out_p, k, has_avx2);
                }
            });
    }

    // Handle remainder rows (n % 4)
    let rem_start = n_quads * 4;
    for j in rem_start..n {
        let w_row = w.add(j * k);
        let dot = dot_f32_i8(x, w_row, k);
        let scale = if s_len > 1 { *scales.add(j) } else { *scales };
        let b = if let Some(bp) = bias { *bp.add(j) } else { 0.0 };
        *out.add(j) = dot * scale + b;
    }
}

/// Compute W8A32 Linear projection: out = (x @ w.T) * scales + bias
///
/// Multi-threaded Rayon execution with AVX2 SIMD dot-products.
pub fn w8a32_linear(
    x: &BorrowedTensor,
    w: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "w8a32_linear requires x with at least 1 dimension",
        ));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w.shape.len() != 2 {
        return Err(unsupported("w8a32_linear requires 2D weight matrix"));
    }
    let n = w.shape[0] as usize;
    let w_k = w.shape[1] as usize;
    if k != w_k {
        return Err(unsupported(&format!(
            "w8a32_linear dimension mismatch: x K={k}, w K={w_k}"
        )));
    }

    let mut out_shape = x.shape.clone();
    out_shape[x_rank - 1] = n as i64;
    let mut out = OwnedTensor::new(DType::F32, out_shape);

    let x_slice = unsafe { typed_slice::<f32>(x) };
    let w_slice = unsafe { typed_slice::<i8>(w) };
    let s_slice = unsafe { typed_slice::<f32>(scales) };
    let out_slice = unsafe { typed_mut_slice::<f32>(&mut out) };

    let bias_slice = bias.map(|b| unsafe { typed_slice::<f32>(b) });

    // Single-token fast path: GEMV with 4-row AVX2 unrolling
    if m == 1 {
        unsafe {
            gemv_w8a32(
                x_slice.as_ptr(),
                w_slice.as_ptr(),
                s_slice.as_ptr(),
                s_slice.len(),
                bias_slice.map(|b| b.as_ptr()),
                out_slice.as_mut_ptr(),
                n,
                k,
            );
        }
        return Ok(out);
    }

    let x_ptr = x_slice.as_ptr() as usize;
    let w_ptr = w_slice.as_ptr() as usize;
    let s_ptr = s_slice.as_ptr() as usize;
    let s_len = s_slice.len();
    let out_ptr = out_slice.as_mut_ptr() as usize;
    let b_ptr = bias_slice.map(|b| b.as_ptr() as usize);

    use rayon::prelude::*;

    (0..n).into_par_iter().with_min_len(8).for_each(|j| {
        let w_row = (w_ptr as *const i8).wrapping_add(j * k);
        let scale = if s_len > 1 {
            unsafe { *((s_ptr as *const f32).add(j)) }
        } else {
            unsafe { *(s_ptr as *const f32) }
        };
        let b = if let Some(bp) = b_ptr {
            unsafe { *((bp as *const f32).add(j)) }
        } else {
            0.0f32
        };

        for r in 0..m {
            let x_row = (x_ptr as *const f32).wrapping_add(r * k);
            let dot = unsafe { dot_f32_i8(x_row, w_row, k) };
            unsafe {
                let out_p = out_ptr as *mut f32;
                *out_p.add(r * n + j) = dot * scale + b;
            }
        }
    });

    Ok(out)
}

/// Compute W4A32 Linear projection: out = (x @ w_unpacked.T) * scales + bias
///
/// Multi-threaded Rayon execution with AVX2 SIMD nibble unpacking.
pub fn w4a32_linear(
    x: &BorrowedTensor,
    w_packed: &BorrowedTensor,
    scales: &BorrowedTensor,
    bias: Option<&BorrowedTensor>,
) -> PyResult<OwnedTensor> {
    let x_rank = x.shape.len();
    if x_rank < 1 {
        return Err(unsupported(
            "w4a32_linear requires x with at least 1 dimension",
        ));
    }
    let k = x.shape[x_rank - 1] as usize;
    let m = elem_count(&x.shape[..x_rank - 1]);

    if w_packed.shape.len() != 2 {
        return Err(unsupported("w4a32_linear requires 2D packed weight matrix"));
    }
    let n = w_packed.shape[0] as usize;
    let w_packed_k = w_packed.shape[1] as usize;
    if (k + 1) / 2 != w_packed_k {
        return Err(unsupported(&format!(
            "w4a32_linear dimension mismatch: x K={k}, w packed K={w_packed_k} (expected {})",
            (k + 1) / 2
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

    let x_ptr = x_slice.as_ptr() as usize;
    let w_ptr = w_slice.as_ptr() as usize;
    let s_ptr = s_slice.as_ptr() as usize;
    let s_len = s_slice.len();
    let out_ptr = out_slice.as_mut_ptr() as usize;
    let b_ptr = bias_slice.map(|b| b.as_ptr() as usize);
    let bytes_per_row = w_packed_k;

    use rayon::prelude::*;

    // Single-token fast path: sequential if N <= 256, chunked Rayon if N > 256
    if m == 1 {
        if n <= 256 {
            for j in 0..n {
                let w_row = unsafe { (w_ptr as *const u8).add(j * bytes_per_row) };
                let scale = if s_len > 1 {
                    unsafe { *((s_ptr as *const f32).add(j)) }
                } else {
                    unsafe { *(s_ptr as *const f32) }
                };
                let b = if let Some(bp) = b_ptr {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0f32
                };
                let dot = unsafe { dot_f32_u4(x_slice.as_ptr(), w_row, k) };
                unsafe {
                    *out_slice.as_mut_ptr().add(j) = dot * scale + b;
                }
            }
            return Ok(out);
        } else {
            (0..n).into_par_iter().with_min_len(8).for_each(|j| {
                let w_row = (w_ptr as *const u8).wrapping_add(j * bytes_per_row);
                let scale = if s_len > 1 {
                    unsafe { *((s_ptr as *const f32).add(j)) }
                } else {
                    unsafe { *(s_ptr as *const f32) }
                };
                let b = if let Some(bp) = b_ptr {
                    unsafe { *((bp as *const f32).add(j)) }
                } else {
                    0.0f32
                };
                let dot = unsafe { dot_f32_u4(x_ptr as *const f32, w_row, k) };
                unsafe {
                    let out_p = out_ptr as *mut f32;
                    *out_p.add(j) = dot * scale + b;
                }
            });
            return Ok(out);
        }
    }

    (0..n).into_par_iter().with_min_len(8).for_each(|j| {
        let w_row = (w_ptr as *const u8).wrapping_add(j * bytes_per_row);
        let scale = if s_len > 1 {
            unsafe { *((s_ptr as *const f32).add(j)) }
        } else {
            unsafe { *(s_ptr as *const f32) }
        };
        let b = if let Some(bp) = b_ptr {
            unsafe { *((bp as *const f32).add(j)) }
        } else {
            0.0f32
        };

        for r in 0..m {
            let x_row = (x_ptr as *const f32).wrapping_add(r * k);
            let dot = unsafe { dot_f32_u4(x_row, w_row, k) };
            unsafe {
                let out_p = out_ptr as *mut f32;
                *out_p.add(r * n + j) = dot * scale + b;
            }
        }
    });

    Ok(out)
}

// ---------------------------------------------------------------------------
// Grouped INT4 (W4A32) SIMD Kernels & Dispatchers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_f32_u4_group64_avx512(x: *const f32, w_packed: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);

    // 64 weights = 32 packed bytes
    // Chunk 0: elements 0..31 (16 bytes)
    let raw0 = _mm_loadu_si128(w_packed as *const __m128i);
    let lo0 = _mm_and_si128(raw0, mask_low);
    let hi0 = _mm_and_si128(_mm_srli_epi16::<4>(raw0), mask_low);
    let inter_lo0 = _mm_unpacklo_epi8(lo0, hi0);
    let inter_hi0 = _mm_unpackhi_epi8(lo0, hi0);
    let s_lo0 = _mm_sub_epi8(inter_lo0, sub8);
    let s_hi0 = _mm_sub_epi8(inter_hi0, sub8);

    let wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_lo0));
    let xf0 = _mm512_loadu_ps(x);
    let mut sum0 = _mm512_mul_ps(wf0, xf0);

    let wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_hi0));
    let xf1 = _mm512_loadu_ps(x.add(16));
    sum0 = _mm512_fmadd_ps(wf1, xf1, sum0);

    // Chunk 1: elements 32..63 (16 bytes)
    let raw1 = _mm_loadu_si128(w_packed.add(16) as *const __m128i);
    let lo1 = _mm_and_si128(raw1, mask_low);
    let hi1 = _mm_and_si128(_mm_srli_epi16::<4>(raw1), mask_low);
    let inter_lo1 = _mm_unpacklo_epi8(lo1, hi1);
    let inter_hi1 = _mm_unpackhi_epi8(lo1, hi1);
    let s_lo1 = _mm_sub_epi8(inter_lo1, sub8);
    let s_hi1 = _mm_sub_epi8(inter_hi1, sub8);

    let wf2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_lo1));
    let xf2 = _mm512_loadu_ps(x.add(32));
    let mut sum1 = _mm512_mul_ps(wf2, xf2);

    let wf3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_hi1));
    let xf3 = _mm512_loadu_ps(x.add(48));
    sum1 = _mm512_fmadd_ps(wf3, xf3, sum1);

    let sum = _mm512_add_ps(sum0, sum1);
    _mm512_reduce_add_ps(sum)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn unpack_and_fma_32_avx512(
    x_ptr: *const f32,
    w_ptr: *const u8,
    mask_low: std::arch::x86_64::__m128i,
    sub8: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let raw = _mm_loadu_si128(w_ptr as *const __m128i);
    let lo = _mm_and_si128(raw, mask_low);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask_low);
    let inter_lo = _mm_unpacklo_epi8(lo, hi);
    let inter_hi = _mm_unpackhi_epi8(lo, hi);
    let s_lo = _mm_sub_epi8(inter_lo, sub8);
    let s_hi = _mm_sub_epi8(inter_hi, sub8);

    let wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_lo));
    let xf0 = _mm512_loadu_ps(x_ptr);
    let wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(s_hi));
    let xf1 = _mm512_loadu_ps(x_ptr.add(16));

    _mm512_fmadd_ps(wf1, xf1, _mm512_mul_ps(wf0, xf0))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn unpack_and_fma_32_avx2(
    x_ptr: *const f32,
    w_ptr: *const u8,
    mask_low: std::arch::x86_64::__m128i,
    sub8: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let raw = _mm_loadu_si128(w_ptr as *const __m128i);
    let lo = _mm_and_si128(raw, mask_low);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask_low);
    let inter_lo = _mm_unpacklo_epi8(lo, hi);
    let inter_hi = _mm_unpackhi_epi8(lo, hi);
    let s_lo = _mm_sub_epi8(inter_lo, sub8);
    let s_hi = _mm_sub_epi8(inter_hi, sub8);

    let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo));
    let xf0 = _mm256_loadu_ps(x_ptr);
    let mut sum0 = _mm256_mul_ps(wf0, xf0);

    let s_lo_hi = _mm_unpackhi_epi64(s_lo, s_lo);
    let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_lo_hi));
    let xf1 = _mm256_loadu_ps(x_ptr.add(8));
    sum0 = _mm256_fmadd_ps(wf1, xf1, sum0);

    let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi));
    let xf2 = _mm256_loadu_ps(x_ptr.add(16));
    let mut sum1 = _mm256_mul_ps(wf2, xf2);

    let s_hi_hi = _mm_unpackhi_epi64(s_hi, s_hi);
    let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s_hi_hi));
    let xf3 = _mm256_loadu_ps(x_ptr.add(24));
    sum1 = _mm256_fmadd_ps(wf3, xf3, sum1);

    _mm256_add_ps(sum0, sum1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_f32_u4_group32_avx512(x: *const f32, w_packed: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let sum0 = unpack_and_fma_32_avx512(x, w_packed, mask_low, sub8);
    _mm512_reduce_add_ps(sum0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u4_group32_avx2(x: *const f32, w_packed: *const u8) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let sum = unpack_and_fma_32_avx2(x, w_packed, mask_low, sub8);
    hsum256_ps_avx(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u4_group64_avx2(x: *const f32, w_packed: *const u8) -> f32 {
    let d0 = dot_f32_u4_group32_avx2(x, w_packed);
    let d1 = dot_f32_u4_group32_avx2(x.add(32), w_packed.add(16));
    d0 + d1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemv_row_w4a32_group64_avx512(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let w_grp = w_row.add(g * 32);
        let scale = _mm512_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx512(x_grp, w_grp, mask_low, sub8);
        let sum1 = unpack_and_fma_32_avx512(x_grp.add(32), w_grp.add(16), mask_low, sub8);
        let grp_sum = _mm512_add_ps(sum0, sum1);
        acc = _mm512_fmadd_ps(grp_sum, scale, acc);
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemv_row_w4a32_group32_avx512(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let w_grp = w_row.add(g * 16);
        let scale = _mm512_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx512(x_grp, w_grp, mask_low, sub8);
        acc = _mm512_fmadd_ps(sum0, scale, acc);
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_w4a32_group64_avx2(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 64);
        let w_grp = w_row.add(g * 32);
        let scale = _mm256_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx2(x_grp, w_grp, mask_low, sub8);
        let sum1 = unpack_and_fma_32_avx2(x_grp.add(32), w_grp.add(16), mask_low, sub8);
        let grp_sum = _mm256_add_ps(sum0, sum1);
        acc = _mm256_fmadd_ps(grp_sum, scale, acc);
    }
    hsum256_ps_avx(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_w4a32_group32_avx2(
    x: *const f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();

    for g in 0..num_groups {
        let x_grp = x.add(g * 32);
        let w_grp = w_row.add(g * 16);
        let scale = _mm256_set1_ps(*s_row.add(g));

        let sum0 = unpack_and_fma_32_avx2(x_grp, w_grp, mask_low, sub8);
        acc = _mm256_fmadd_ps(sum0, scale, acc);
    }
    hsum256_ps_avx(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn quantize_activation_to_u8_avx512(x: *const f32, k: usize, out: *mut u8) -> f32 {
    use std::arch::x86_64::*;
    let mut max_vec = _mm512_setzero_ps();
    let num_chunks = k / 16;
    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        let abs_v = _mm512_abs_ps(v);
        max_vec = _mm512_max_ps(max_vec, abs_v);
    }
    let mut max_val = _mm512_reduce_max_ps(max_vec);
    for i in (num_chunks * 16)..k {
        let a = (*x.add(i)).abs();
        if a > max_val {
            max_val = a;
        }
    }
    if max_val < 1e-10 {
        max_val = 1e-10;
    }
    let s_x = max_val / 127.0;
    let inv_s = _mm512_set1_ps(1.0 / s_x);
    let offset128 = _mm512_set1_epi32(128);

    for c in 0..num_chunks {
        let v = _mm512_loadu_ps(x.add(c * 16));
        let scaled = _mm512_mul_ps(v, inv_s);
        let rounded = _mm512_cvtps_epi32(scaled);
        let shifted = _mm512_add_epi32(rounded, offset128);
        let bytes16 = _mm512_cvtepi32_epi8(shifted);
        _mm_storeu_si128(out.add(c * 16) as *mut __m128i, bytes16);
    }
    for i in (num_chunks * 16)..k {
        let q = ((*x.add(i)) / s_x).round().clamp(-127.0, 127.0) as i16;
        *out.add(i) = (q + 128) as u8;
    }
    s_x
}

#[inline(always)]
pub(crate) unsafe fn quantize_activation_to_u8(x: *const f32, k: usize) -> (Vec<u8>, f32) {
    let mut x_u8 = vec![0u8; k];
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx512f && crate::dispatch::cpu_features().avx512bw {
            let s = quantize_activation_to_u8_avx512(x, k, x_u8.as_mut_ptr());
            return (x_u8, s);
        }
    }
    let mut max_abs = 0.0f32;
    for i in 0..k {
        let v = (*x.add(i)).abs();
        if v > max_abs {
            max_abs = v;
        }
    }
    if max_abs < 1e-10 {
        max_abs = 1e-10;
    }
    let s_x = max_abs / 127.0;
    let inv_s = 1.0 / s_x;

    for i in 0..k {
        let q = ((*x.add(i)) * inv_s).round().clamp(-127.0, 127.0) as i16;
        x_u8[i] = (q + 128) as u8;
    }
    (x_u8, s_x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512f,avx512bw")]
unsafe fn gemv_row_w4a8_group64_vnni_avx512(
    x_u8: *const u8,
    s_x: f32,
    w_row: *const u8,
    s_row: *const f32,
    num_groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let mask_low = _mm_set1_epi8(0x0F);
    let sub8 = _mm_set1_epi8(8);
    let ones = _mm512_set1_epi8(1);
    let mut acc = _mm512_setzero_ps();

    for g in 0..num_groups {
        let x64 = _mm512_loadu_si512(x_u8.add(g * 64) as *const __m512i);
        let w_grp = w_row.add(g * 32);
        let s_w = *s_row.add(g);
        let scale_vec = _mm512_set1_ps(s_x * s_w);

        // Chunk 0 (0..31)
        let raw0 = _mm_loadu_si128(w_grp as *const __m128i);
        let lo0 = _mm_and_si128(raw0, mask_low);
        let hi0 = _mm_and_si128(_mm_srli_epi16::<4>(raw0), mask_low);
        let inter_lo0 = _mm_unpacklo_epi8(lo0, hi0);
        let inter_hi0 = _mm_unpackhi_epi8(lo0, hi0);
        let s_lo0 = _mm_sub_epi8(inter_lo0, sub8);
        let s_hi0 = _mm_sub_epi8(inter_hi0, sub8);
        let w32_0 = _mm256_set_m128i(s_hi0, s_lo0);

        // Chunk 1 (32..63)
        let raw1 = _mm_loadu_si128(w_grp.add(16) as *const __m128i);
        let lo1 = _mm_and_si128(raw1, mask_low);
        let hi1 = _mm_and_si128(_mm_srli_epi16::<4>(raw1), mask_low);
        let inter_lo1 = _mm_unpacklo_epi8(lo1, hi1);
        let inter_hi1 = _mm_unpackhi_epi8(lo1, hi1);
        let s_lo1 = _mm_sub_epi8(inter_lo1, sub8);
        let s_hi1 = _mm_sub_epi8(inter_hi1, sub8);
        let w32_1 = _mm256_set_m128i(s_hi1, s_lo1);

        let w64 = _mm512_inserti64x4(_mm512_castsi256_si512(w32_0), w32_1, 1);

        // Hardware VNNI integer dot product (64 MACs in 1 cycle)
        let dot_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x64, w64);
        let w_sum_i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, w64);
        let true_i32 = _mm512_sub_epi32(dot_i32, _mm512_slli_epi32::<7>(w_sum_i32));
        let true_f32 = _mm512_cvtepi32_ps(true_i32);

        acc = _mm512_fmadd_ps(true_f32, scale_vec, acc);
    }
    _mm512_reduce_add_ps(acc)
}
pub mod swiglu;

// x86-only kernels (AVX2/AVX-512/VNNI intrinsics; scalar ARM fallbacks for
// the group64 entry points live in `swiglu.rs` but have no in-tree callers on
// non-x86, so the whole re-export is x86-gated to keep aarch64 warning-free).
#[cfg(target_arch = "x86_64")]
pub(crate) use self::swiglu::{
    swiglu_neuron_w4a32_group32_avx2, swiglu_neuron_w4a32_group32_avx512,
    swiglu_neuron_w4a32_group64_avx2, swiglu_neuron_w4a32_group64_avx512,
    swiglu_neuron_w4a8_group64_vnni_avx512,
};

pub(crate) unsafe fn dot_f32_u4_group_scalar(
    x: *const f32,
    w_packed: *const u8,
    group_size: usize,
) -> f32 {
    let mut total = 0.0f32;
    for i in 0..group_size {
        let byte = *w_packed.add(i / 2);
        let q = if (i % 2) == 0 {
            ((byte & 0x0F) as i8) - 8
        } else {
            (((byte >> 4) & 0x0F) as i8) - 8
        };
        total += *x.add(i) * (q as f32);
    }
    total
}

#[inline(always)]
unsafe fn dot_f32_u4_group64_fast(
    x: *const f32,
    w_packed: *const u8,
    has_avx512: bool,
    has_avx2: bool,
    has_neon: bool,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = has_neon;
        if has_avx512 {
            return dot_f32_u4_group64_avx512(x, w_packed);
        }
        if has_avx2 {
            return dot_f32_u4_group64_avx2(x, w_packed);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon {
            return dot_f32_u4_group64_neon(x, w_packed);
        }
    }
    #[cfg(target_arch = "aarch64")]
    let _ = (has_avx512, has_avx2);
    dot_f32_u4_group_scalar(x, w_packed, 64)
}

#[inline(always)]
unsafe fn dot_f32_u4_group32_fast(
    x: *const f32,
    w_packed: *const u8,
    has_avx512: bool,
    has_avx2: bool,
    has_neon: bool,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = has_neon;
        if has_avx512 {
            return dot_f32_u4_group32_avx512(x, w_packed);
        }
        if has_avx2 {
            return dot_f32_u4_group32_avx2(x, w_packed);
        }
    }
    // NEON path uses 64-element kernel for 32-element groups (processes 32 + tail)
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon {
            return dot_f32_u4_group64_neon(x, w_packed);
        }
    }
    #[cfg(target_arch = "aarch64")]
    let _ = (has_avx512, has_avx2);
    dot_f32_u4_group_scalar(x, w_packed, 32)
}

#[inline(always)]
unsafe fn dot_f32_u4_group64(x: *const f32, w_packed: *const u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx512f && crate::dispatch::cpu_features().avx512bw {
            return dot_f32_u4_group64_avx512(x, w_packed);
        }
        if crate::dispatch::cpu_features().avx2 && crate::dispatch::cpu_features().fma {
            return dot_f32_u4_group64_avx2(x, w_packed);
        }
    }
    dot_f32_u4_group_scalar(x, w_packed, 64)
}

#[inline(always)]
unsafe fn dot_f32_u4_group32(x: *const f32, w_packed: *const u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx512f && crate::dispatch::cpu_features().avx512bw {
            return dot_f32_u4_group32_avx512(x, w_packed);
        }
        if crate::dispatch::cpu_features().avx2 && crate::dispatch::cpu_features().fma {
            return dot_f32_u4_group32_avx2(x, w_packed);
        }
    }
    dot_f32_u4_group_scalar(x, w_packed, 32)
}

/// Fast single-token GEMV for W4A32 with group-wise scales.
///
/// SIMD tier (scalar / AVX2 / AVX-512 / AVX-512 VNNI) is resolved once via
/// [`crate::dispatch::cpu_features`] instead of re-probing CPUID per call.
///
/// # Safety
/// `x` must hold `k` readable f32s; `w_packed` must hold `n` rows of
/// `k / 2` packed int4 bytes; `scales` must hold `n * num_groups` f32s;
/// `bias` (if `Some`) must hold `n` f32s; `out` must hold `n` writable f32s.
pub unsafe fn gemv_w4a32_grouped(
    x: *const f32,
    w_packed: *const u8,
    scales: *const f32,
    bias: Option<*const f32>,
    out: *mut f32,
    n: usize,
    k: usize,
    group_size: usize,
) {
    use rayon::prelude::*;
    let num_groups = (k + group_size - 1) / group_size;
    let bytes_per_row = (k + 1) / 2;
    let bytes_per_group = group_size / 2;

    let x_usize = x as usize;
    let w_usize = w_packed as usize;
    let s_usize = scales as usize;
    let b_usize = bias.map(|bp| bp as usize);
    let out_usize = out as usize;

    let n_threads = rayon::current_num_threads();
    let min_chunk = (n / (n_threads * 4)).max(8);

    // Resolve the SIMD tier once per call (Phase 0.3 dispatch).
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

    let has_neon = feats.neon;

    let (x_u8_opt, s_x) = if has_vnni && group_size == 64 {
        let (u, s) = unsafe { quantize_activation_to_u8(x, k) };
        (Some(u), s)
    } else {
        (None, 1.0f32)
    };
    let x_u8_ptr = x_u8_opt.as_ref().map(|v| v.as_ptr() as usize);

    (0..n)
        .into_par_iter()
        .with_min_len(min_chunk)
        .for_each(|j| {
            let x_p = x_usize as *const f32;
            let w_row = (w_usize as *const u8).add(j * bytes_per_row);
            let s_row = (s_usize as *const f32).add(j * num_groups);
            let b_p = b_usize.map(|bp| bp as *const f32);
            let out_p = out_usize as *mut f32;

            let row_sum = if let Some(x_u8_p) = x_u8_ptr {
                #[cfg(target_arch = "x86_64")]
                {
                    unsafe {
                        gemv_row_w4a8_group64_vnni_avx512(
                            x_u8_p as *const u8,
                            s_x,
                            w_row,
                            s_row,
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
                    if group_size == 64 {
                        gemv_row_w4a32_group64_avx512(x_p, w_row, s_row, num_groups)
                    } else if group_size == 32 {
                        gemv_row_w4a32_group32_avx512(x_p, w_row, s_row, num_groups)
                    } else {
                        let mut s = 0.0f32;
                        for g in 0..num_groups {
                            let cur_len = (k - g * group_size).min(group_size);
                            s += dot_f32_u4_group_scalar(
                                x_p.add(g * group_size),
                                w_row.add(g * bytes_per_group),
                                cur_len,
                            ) * *s_row.add(g);
                        }
                        s
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    0.0f32
                }
            } else if has_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    if group_size == 64 {
                        gemv_row_w4a32_group64_avx2(x_p, w_row, s_row, num_groups)
                    } else if group_size == 32 {
                        gemv_row_w4a32_group32_avx2(x_p, w_row, s_row, num_groups)
                    } else {
                        let mut s = 0.0f32;
                        for g in 0..num_groups {
                            let cur_len = (k - g * group_size).min(group_size);
                            s += dot_f32_u4_group_scalar(
                                x_p.add(g * group_size),
                                w_row.add(g * bytes_per_group),
                                cur_len,
                            ) * *s_row.add(g);
                        }
                        s
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    0.0f32
                }
            } else if has_neon && group_size == 64 {
                #[cfg(target_arch = "aarch64")]
                {
                    unsafe { gemv_row_w4a32_group64_neon(x_p, w_row, s_row, num_groups) }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    0.0f32
                }
            } else {
                let mut s = 0.0f32;
                for g in 0..num_groups {
                    let cur_len = (k - g * group_size).min(group_size);
                    s += dot_f32_u4_group_scalar(
                        x_p.add(g * group_size),
                        w_row.add(g * bytes_per_group),
                        cur_len,
                    ) * *s_row.add(g);
                }
                s
            };

            let b = if let Some(bp) = b_p { *bp.add(j) } else { 0.0 };
            *out_p.add(j) = row_sum + b;
        });
}

/// Batched W4A32 GEMM for prompt prefill: computes Y = X * W^T + B
/// where X is [M, K] contiguous activations, W is [N, K/2] INT4 packed weights,
/// B is optional [N] bias, and Y is [M, N] output activations.
/// Uses 2D parallel tiling across rows M and output features N to saturate CPU caches.
pub unsafe fn gemm_w4a32_grouped(
    x: *const f32,
    w: *const u8,
    scales: *const f32,
    bias: Option<*const f32>,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    use rayon::prelude::*;

    if m == 1 {
        gemv_w4a32_grouped(x, w, scales, bias, out, n, k, group_size);
        return;
    }

    // Cast raw pointers to usize so the Rayon closure captures Send+Sync values.
    // Safety: caller guarantees the ranges [0, m*k), [0, n*k/2), [0, n*num_groups),
    // and [0, m*n) are all valid and non-overlapping across rows.
    let x_usize = x as usize;
    let w_usize = w as usize;
    let s_usize = scales as usize;
    let b_usize = bias.map(|bp| bp as usize);
    let out_usize = out as usize;

    (0..m).into_par_iter().for_each(|i| {
        let row_x = (x_usize as *const f32).add(i * k);
        let row_out = (out_usize as *mut f32).add(i * n);
        let w_p = w_usize as *const u8;
        let s_p = s_usize as *const f32;
        let b_p = b_usize.map(|bp| bp as *const f32);
        gemv_w4a32_grouped(row_x, w_p, s_p, b_p, row_out, n, k, group_size);
    });
}
