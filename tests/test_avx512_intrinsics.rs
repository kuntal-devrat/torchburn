use std::time::Instant;

#[test]
fn test_avx512_gemv_w8a32() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            println!("AVX512F or AVX512BW not detected");
            return;
        }

        let k = 896;
        let x_f32: Vec<f32> = (0..k).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
        let w8: Vec<Vec<i8>> = (0..8).map(|row| {
            (0..k).map(|col| (((row * 17 + col * 31) % 251) as i16 - 125) as i8).collect()
        }).collect();

        // Reference float dot product for all 8 rows
        let mut ref_dots = [0.0f32; 8];
        for r in 0..8 {
            for i in 0..k {
                ref_dots[r] += x_f32[i] * (w8[r][i] as f32);
            }
        }

        // AVX2 4-row implementation (called twice for 8 rows)
        #[target_feature(enable = "avx,avx2,fma")]
        unsafe fn gemv_4rows_avx2(
            x: *const f32,
            w0: *const i8,
            w1: *const i8,
            w2: *const i8,
            w3: *const i8,
            len: usize,
        ) -> (f32, f32, f32, f32) {
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

                let wf0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w0.add(offset) as *const __m128i)));
                acc0_0 = _mm256_fmadd_ps(wf0_0, x0, acc0_0);
                let wf0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w0.add(offset + 8) as *const __m128i)));
                acc0_1 = _mm256_fmadd_ps(wf0_1, x1, acc0_1);

                let wf1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w1.add(offset) as *const __m128i)));
                acc1_0 = _mm256_fmadd_ps(wf1_0, x0, acc1_0);
                let wf1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w1.add(offset + 8) as *const __m128i)));
                acc1_1 = _mm256_fmadd_ps(wf1_1, x1, acc1_1);

                let wf2_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w2.add(offset) as *const __m128i)));
                acc2_0 = _mm256_fmadd_ps(wf2_0, x0, acc2_0);
                let wf2_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w2.add(offset + 8) as *const __m128i)));
                acc2_1 = _mm256_fmadd_ps(wf2_1, x1, acc2_1);

                let wf3_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w3.add(offset) as *const __m128i)));
                acc3_0 = _mm256_fmadd_ps(wf3_0, x0, acc3_0);
                let wf3_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w3.add(offset + 8) as *const __m128i)));
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
                let wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w0.add(offset) as *const __m128i)));
                sum0 = _mm256_fmadd_ps(wf0, x_vec, sum0);
                let wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w1.add(offset) as *const __m128i)));
                sum1 = _mm256_fmadd_ps(wf1, x_vec, sum1);
                let wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w2.add(offset) as *const __m128i)));
                sum2 = _mm256_fmadd_ps(wf2, x_vec, sum2);
                let wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(w3.add(offset) as *const __m128i)));
                sum3 = _mm256_fmadd_ps(wf3, x_vec, sum3);
                offset += 8;
            }

            unsafe fn hsum256(v: __m256) -> f32 {
                let v_hi = _mm256_extractf128_ps::<1>(v);
                let v_lo = _mm256_castps256_ps128(v);
                let sum128 = _mm_add_ps(v_hi, v_lo);
                let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<0x55>(sum64, sum64));
                _mm_cvtss_f32(sum32)
            }

            let mut tot0 = hsum256(sum0);
            let mut tot1 = hsum256(sum1);
            let mut tot2 = hsum256(sum2);
            let mut tot3 = hsum256(sum3);

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

        // AVX-512 8-row implementation
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn gemv_8rows_avx512(
            x: *const f32,
            w0: *const i8,
            w1: *const i8,
            w2: *const i8,
            w3: *const i8,
            w4: *const i8,
            w5: *const i8,
            w6: *const i8,
            w7: *const i8,
            len: usize,
        ) -> [f32; 8] {
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
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(w_ptr.add(off) as *const __m128i)))
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
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(w_ptr.add(off) as *const __m128i)))
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

        // Check correctness of both
        let (a0, a1, a2, a3) = gemv_4rows_avx2(x_f32.as_ptr(), w8[0].as_ptr(), w8[1].as_ptr(), w8[2].as_ptr(), w8[3].as_ptr(), k);
        let (a4, a5, a6, a7) = gemv_4rows_avx2(x_f32.as_ptr(), w8[4].as_ptr(), w8[5].as_ptr(), w8[6].as_ptr(), w8[7].as_ptr(), k);
        let avx2_res = [a0, a1, a2, a3, a4, a5, a6, a7];

        let avx512_res = gemv_8rows_avx512(
            x_f32.as_ptr(),
            w8[0].as_ptr(), w8[1].as_ptr(), w8[2].as_ptr(), w8[3].as_ptr(),
            w8[4].as_ptr(), w8[5].as_ptr(), w8[6].as_ptr(), w8[7].as_ptr(),
            k,
        );

        println!("Comparing Ref vs AVX2 vs AVX512:");
        for r in 0..8 {
            let ref_v = ref_dots[r];
            let avx2_v = avx2_res[r];
            let avx512_v = avx512_res[r];
            let diff_avx2 = (ref_v - avx2_v).abs();
            let diff_avx512 = (ref_v - avx512_v).abs();
            println!("Row {r}: Ref={ref_v:.4}, AVX2={avx2_v:.4} (diff={diff_avx2:.2e}), AVX512={avx512_v:.4} (diff={diff_avx512:.2e})");
            assert!(diff_avx512 < 0.05, "AVX512 mismatch on row {r}: ref={ref_v}, avx512={avx512_v}");
        }

        // Benchmark speed
        let iters = 100_000;
        let t0 = Instant::now();
        for _ in 0..iters {
            let (b0, b1, b2, b3) = gemv_4rows_avx2(x_f32.as_ptr(), w8[0].as_ptr(), w8[1].as_ptr(), w8[2].as_ptr(), w8[3].as_ptr(), k);
            let (b4, b5, b6, b7) = gemv_4rows_avx2(x_f32.as_ptr(), w8[4].as_ptr(), w8[5].as_ptr(), w8[6].as_ptr(), w8[7].as_ptr(), k);
            std::hint::black_box([b0, b1, b2, b3, b4, b5, b6, b7]);
        }
        let dur_avx2 = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            let res = gemv_8rows_avx512(
                x_f32.as_ptr(),
                w8[0].as_ptr(), w8[1].as_ptr(), w8[2].as_ptr(), w8[3].as_ptr(),
                w8[4].as_ptr(), w8[5].as_ptr(), w8[6].as_ptr(), w8[7].as_ptr(),
                k,
            );
            std::hint::black_box(res);
        }
        let dur_avx512 = t1.elapsed();

        println!("AVX2   8 rows x {k} ({iters} iters): {:?}", dur_avx2);
        println!("AVX512 8 rows x {k} ({iters} iters): {:?}", dur_avx512);
        println!("Speedup: {:.2}x", dur_avx2.as_secs_f64() / dur_avx512.as_secs_f64());
    }
}
