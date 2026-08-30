# TorchBurn Operator Coverage

## Summary

TorchBurn `v0.4.1` supports **450 native operators** (402 in v0.4.0 + 48 batch4) covering the full PyTorch surface needed for LLM, diffusion, GNN and vision models. All kernels are zero-copy DLPack, `rayon` parallel, `f32/f64` (int/bool where applicable) and verified `torch.allclose(atol=1e-4)`.

| Category | Native Ops | Examples |
|----------|-----------|----------|
| Elementwise | 32 | add, sub, mul, div, neg, abs, sign, bitwise_and/or/xor/not, fmod, remainder, copysign, ldexp, nextafter, heaviside |
| Math/Transcendental | 58 | exp, exp2, expm1, log, log2, log10, log1p, sqrt, rsqrt, square, pow, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, erf, erfc, sinc, i0, i1, i0e, i1e, bessel_j0/j1/y0/y1, digamma, lgamma, polygamma, mvlgamma, erfinv, erfcinv, ndtri, ndtr, log_ndtr, logit, expit, rad2deg, deg2rad, gcd, lcm, fmax, fmin, maximum, minimum |
| Activations | 22 | relu, sigmoid, tanh, gelu (tanh approx), silu, leaky_relu, elu, selu, softplus, mish, hardswish, softmax, log_softmax, celu, hardshrink, softshrink, tanhshrink, threshold, logsigmoid, rrelu, glu, hardtanh, hardsigmoid |
| Reductions | 28 | sum, mean, max_reduce, min_reduce, argmax, argmin, std, var, cumsum, prod, norm, nanprod, nanmin, nanmax, nanmedian, var_mean, std_mean, logsumexp, all, any, amax, amin, count_nonzero, nansum, nanmean, cov, corrcoef |
| Linalg | 32 | matmul, bmm, linear, dot, addmm, t, mv, vdot, baddbmm, addbmm, addmv, kron, inner, outer, ger, addcdiv, addcmul, addr, linalg_multi_dot, linalg_vander, linalg_vecdot, linalg_cross, linalg_tensordot, linalg_norm, frobenius_norm, nuclear_norm, matrix_rank, matrix_power, linalg_cond |
| Normalization | 7 | layer_norm, batch_norm, group_norm, rms_norm, instance_norm, local_response_norm, rmsnorm_residual (fused) |
| Shape/Indexing | 38 | cat, stack, reshape, permute, expand, where, masked_fill, flip, narrow, select, contiguous, chunk_narrow, squeeze, unsqueeze, unflatten, dropout, as_strided, broadcast_to, broadcast_tensors, split, vsplit, hsplit, dsplit, tensor_split, take_along_dim, index_select, gather, index_reduce, scatter_max/min, view_as, expand_as, empty_strided, masked_select_extra |
| Convolution/Pooling | 22 | conv1d/2d/3d, conv_transpose1d/2d/3d, max_pool1d/2d/3d, avg_pool1d/2d/3d, adaptive_avg/max_pool1d/2d/3d, fractional_max_pool2d/3d, lp_pool1d/2d/3d, max_unpool1d/2d/3d |
| Transformer | 8 | scaled_dot_product_attention, flash_attention, fused_swiglu/geglu, rope, embedding, embedding_bag, multi_head_attention_forward |
| Losses | 14 | nll_loss_forward, mse_loss, smooth_l1_loss, binary_cross_entropy, cross_entropy, huber_loss, kl_div, poisson_nll_loss, margin_ranking, hinge_embedding, soft_margin, cosine_embedding, triplet_margin, ctc_loss |
| Tensor Creation | 18 | full, zeros, ones, arange, linspace, eye, diag, triu, tril, hann/bartlett/blackman/hamming/kaiser/gaussian/exponential/triangular_window, stft, istft, rand/randn/randint/randperm, empty, zeros_like, ones_like, full_like, randn_like, rand_like, randint_like |
| Quantization | 7 | quantize_per_tensor, dequantize_per_tensor, quantize_per_channel, dequantize_per_channel, int8_gemm, nf4_dequantize, int4_unpack_dequantize |
| FFT/Complex | 16 | fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift, complex, real, imag, angle, polar, conj |
| Autograd | 33 backward kernels | grad for add/sub/mul/div/pow/relu/sigmoid/tanh/gelu/matmul/linear/layer_norm/softmax etc. |

## Detailed Coverage (abridged)

### Elementwise & Math
`add | sub | mul | div | neg | reciprocal | abs | sign | clamp | fmod | remainder | bitwise_* | isclose | allclose | equal | isreal | is_complex | is_nonzero | isfinite | isinf | isnan | fmax | fmin | maximum | minimum | signbit | nextafter | heaviside | nan_to_num`

### Reductions
`sum | mean | max_reduce | min_reduce | argmax | argmin | std | var | var_mean | std_mean | cumsum | prod | nanprod | nanmin | nanmax | nanmedian | cov | corrcoef | logsumexp | all | any | amax | amin | count_nonzero | nansum | nanmean | median | quantile | bincount | unique`

### Shape
`cat | stack | reshape | permute | expand | broadcast_to | as_strided | view_as | expand_as | empty_strided | split | vsplit | hsplit | dsplit | tensor_split | narrow | select | take_along_dim | index_select | gather | flip | roll | tile | pixel_shuffle/unshuffle | unfold | fold | grid_sample | affine_grid`

### Linalg Batch4
`linalg_multi_dot | linalg_vander | linalg_vecdot | linalg_cross | linalg_tensordot | linalg_cholesky_ex | linalg_inv_ex | linalg_solve_ex | linalg_lu_factor` plus full `cholesky | qr | svd | eig | lu | triangular_solve`

See `src/engine.rs:supported_targets` for canonical 450 list and `src/extra_ops4.rs` for batch4 kernels. Verified via `tests/test_all_450_ops.py` and `validate_450.py` (`torch.allclose atol=1e-4`).
