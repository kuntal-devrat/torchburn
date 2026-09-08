//! Per-op dispatch: route a plan Node to its native kernel.
//!
//! Uses a pre-built HashMap for O(1) target→module lookup instead of
//! linear scanning through all 22 modules. The HashMap is built once
//! at first use via OnceLock.

use super::*;
use std::sync::OnceLock;

pub mod d_act_loss;
pub mod d_batch2;
pub mod d_batch4;
pub mod d_blas_extra;
pub mod d_decomp;
pub mod d_elementwise;
pub mod d_fft;
pub mod d_float_special;
pub mod d_fused;
pub mod d_math_extra;
pub mod d_nn;
pub mod d_nn3d;
pub mod d_norm;
pub mod d_quant;
pub mod d_random;
pub mod d_reducers;
pub mod d_rnn;
pub mod d_scatter;
pub mod d_scatter2;
pub mod d_shape;
pub mod d_shape2;
pub mod d_solve;

/// Module index constants — order matches MODULE_FNS below.
const M_ELEMENTWISE: usize = 0;
const M_REDUCERS: usize = 1;
const M_NORM: usize = 2;
const M_SHAPE: usize = 3;
const M_NN: usize = 4;
const M_SCATTER: usize = 5;
const M_MATH_EXTRA: usize = 6;
const M_BATCH2: usize = 7;
const M_FLOAT_SPECIAL: usize = 8;
const M_BLAS_EXTRA: usize = 9;
const M_ACT_LOSS: usize = 10;
const M_DECOMP: usize = 11;
const M_SCATTER2: usize = 12;
const M_SHAPE2: usize = 13;
const M_NN3D: usize = 14;
const M_RANDOM: usize = 15;
const M_RNN: usize = 16;
const M_SOLVE: usize = 17;
const M_FUSED: usize = 18;
const M_QUANT: usize = 19;
const M_FFT: usize = 20;
const M_BATCH4: usize = 21;

type DispatchFn = fn(&Node, &mut Vec<Slot>, &[CapsuleRef]) -> PyResult<bool>;

/// Module dispatch functions indexed by module constant.
const MODULE_FNS: &[DispatchFn] = &[
    d_elementwise::try_dispatch,   // 0
    d_reducers::try_dispatch,      // 1
    d_norm::try_dispatch,          // 2
    d_shape::try_dispatch,         // 3
    d_nn::try_dispatch,            // 4
    d_scatter::try_dispatch,       // 5
    d_math_extra::try_dispatch,    // 6
    d_batch2::try_dispatch,        // 7
    d_float_special::try_dispatch, // 8
    d_blas_extra::try_dispatch,    // 9
    d_act_loss::try_dispatch,      // 10
    d_decomp::try_dispatch,        // 11
    d_scatter2::try_dispatch,      // 12
    d_shape2::try_dispatch,        // 13
    d_nn3d::try_dispatch,          // 14
    d_random::try_dispatch,        // 15
    d_rnn::try_dispatch,           // 16
    d_solve::try_dispatch,         // 17
    d_fused::try_dispatch,         // 18
    d_quant::try_dispatch,         // 19
    d_fft::try_dispatch,           // 20
    d_batch4::try_dispatch,        // 21
];

/// Static (target, module_index) pairs — one entry per supported target.
/// Sorted alphabetically within each module group for readability.
static TARGET_MODULE_PAIRS: &[(&str, usize)] = &[
    // d_elementwise (37 targets)
    ("add", M_ELEMENTWISE),
    ("sub", M_ELEMENTWISE),
    ("mul", M_ELEMENTWISE),
    ("div", M_ELEMENTWISE),
    ("relu", M_ELEMENTWISE),
    ("logical_and", M_ELEMENTWISE),
    ("logical_or", M_ELEMENTWISE),
    ("logical_not", M_ELEMENTWISE),
    ("to_dtype", M_ELEMENTWISE),
    ("eq", M_ELEMENTWISE),
    ("ne", M_ELEMENTWISE),
    ("lt", M_ELEMENTWISE),
    ("le", M_ELEMENTWISE),
    ("gt", M_ELEMENTWISE),
    ("ge", M_ELEMENTWISE),
    ("abs", M_ELEMENTWISE),
    ("neg", M_ELEMENTWISE),
    ("sign", M_ELEMENTWISE),
    ("sqrt", M_ELEMENTWISE),
    ("rsqrt", M_ELEMENTWISE),
    ("exp", M_ELEMENTWISE),
    ("log", M_ELEMENTWISE),
    ("reciprocal", M_ELEMENTWISE),
    ("ceil", M_ELEMENTWISE),
    ("floor", M_ELEMENTWISE),
    ("clamp", M_ELEMENTWISE),
    ("clamp_min", M_ELEMENTWISE),
    ("clamp_max", M_ELEMENTWISE),
    ("sin", M_ELEMENTWISE),
    ("cos", M_ELEMENTWISE),
    ("round", M_ELEMENTWISE),
    ("pow", M_ELEMENTWISE),
    ("sigmoid", M_ELEMENTWISE),
    ("tanh", M_ELEMENTWISE),
    ("gelu", M_ELEMENTWISE),
    ("silu", M_ELEMENTWISE),
    ("leaky_relu", M_ELEMENTWISE),
    ("elu", M_ELEMENTWISE),
    ("selu", M_ELEMENTWISE),
    ("softplus", M_ELEMENTWISE),
    ("hardswish", M_ELEMENTWISE),
    ("mish", M_ELEMENTWISE),
    ("softmax", M_ELEMENTWISE),
    ("log_softmax", M_ELEMENTWISE),
    ("threshold_backward", M_ELEMENTWISE),
    // d_reducers (19 targets)
    ("sum", M_REDUCERS),
    ("mean", M_REDUCERS),
    ("max", M_REDUCERS),
    ("max_reduce", M_REDUCERS),
    ("min", M_REDUCERS),
    ("min_reduce", M_REDUCERS),
    ("argmax", M_REDUCERS),
    ("argmin", M_REDUCERS),
    ("std", M_REDUCERS),
    ("var", M_REDUCERS),
    ("cumsum", M_REDUCERS),
    ("prod", M_REDUCERS),
    ("norm", M_REDUCERS),
    ("linalg_vector_norm", M_REDUCERS),
    ("matmul", M_REDUCERS),
    ("bmm", M_REDUCERS),
    ("linear", M_REDUCERS),
    ("dot", M_REDUCERS),
    ("addmm", M_REDUCERS),
    // d_norm (4 targets)
    ("layer_norm", M_NORM),
    ("batch_norm", M_NORM),
    ("group_norm", M_NORM),
    ("rms_norm", M_NORM),
    // d_shape (25 targets)
    ("cat", M_SHAPE),
    ("stack", M_SHAPE),
    ("reshape", M_SHAPE),
    ("permute", M_SHAPE),
    ("transpose", M_SHAPE),
    ("index_select", M_SHAPE),
    ("gather", M_SHAPE),
    ("chunk", M_SHAPE),
    ("unbind", M_SHAPE),
    ("t", M_SHAPE),
    ("expand", M_SHAPE),
    ("where", M_SHAPE),
    ("masked_fill", M_SHAPE),
    ("flip", M_SHAPE),
    ("narrow", M_SHAPE),
    ("select", M_SHAPE),
    ("getitem", M_SHAPE),
    ("chunk_narrow", M_SHAPE),
    ("contiguous", M_SHAPE),
    ("squeeze", M_SHAPE),
    ("unsqueeze", M_SHAPE),
    ("unflatten", M_SHAPE),
    ("dropout", M_SHAPE),
    ("flatten", M_SHAPE),
    // d_nn (26 targets)
    ("scalar_tensor", M_NN),
    ("full", M_NN),
    ("zeros", M_NN),
    ("ones", M_NN),
    ("arange", M_NN),
    ("linspace", M_NN),
    ("embedding", M_NN),
    ("scaled_dot_product_attention", M_NN),
    ("rope", M_NN),
    ("nll_loss_forward", M_NN),
    ("mse_loss", M_NN),
    ("smooth_l1_loss", M_NN),
    ("binary_cross_entropy", M_NN),
    ("conv1d", M_NN),
    ("conv2d", M_NN),
    ("conv_transpose1d", M_NN),
    ("conv_transpose2d", M_NN),
    ("max_pool2d", M_NN),
    ("avg_pool2d", M_NN),
    ("adaptive_avg_pool2d", M_NN),
    ("adaptive_max_pool2d", M_NN),
    ("max_pool1d", M_NN),
    ("avg_pool1d", M_NN),
    ("upsample_nearest2d", M_NN),
    ("upsample_bilinear2d", M_NN),
    ("interpolate", M_NN),
    // d_scatter (11 targets)
    ("scatter", M_SCATTER),
    ("scatter_add", M_SCATTER),
    ("topk", M_SCATTER),
    ("sort", M_SCATTER),
    ("argsort", M_SCATTER),
    ("repeat_interleave", M_SCATTER),
    ("repeat", M_SCATTER),
    ("einsum", M_SCATTER),
    ("prelu", M_SCATTER),
    ("nonzero", M_SCATTER),
    ("clamp_tensor", M_SCATTER),
    // d_math_extra (50 targets)
    ("atan", M_MATH_EXTRA),
    ("asin", M_MATH_EXTRA),
    ("acos", M_MATH_EXTRA),
    ("sinh", M_MATH_EXTRA),
    ("cosh", M_MATH_EXTRA),
    ("asinh", M_MATH_EXTRA),
    ("acosh", M_MATH_EXTRA),
    ("atanh", M_MATH_EXTRA),
    ("erf", M_MATH_EXTRA),
    ("erfc", M_MATH_EXTRA),
    ("expm1", M_MATH_EXTRA),
    ("log1p", M_MATH_EXTRA),
    ("log2", M_MATH_EXTRA),
    ("log10", M_MATH_EXTRA),
    ("trunc", M_MATH_EXTRA),
    ("frac", M_MATH_EXTRA),
    ("square", M_MATH_EXTRA),
    ("exp2", M_MATH_EXTRA),
    ("atan2", M_MATH_EXTRA),
    ("hypot", M_MATH_EXTRA),
    ("fmod", M_MATH_EXTRA),
    ("remainder", M_MATH_EXTRA),
    ("copysign", M_MATH_EXTRA),
    ("ldexp", M_MATH_EXTRA),
    ("lerp", M_MATH_EXTRA),
    ("bitwise_and", M_MATH_EXTRA),
    ("bitwise_or", M_MATH_EXTRA),
    ("bitwise_xor", M_MATH_EXTRA),
    ("bitwise_not", M_MATH_EXTRA),
    ("isfinite", M_MATH_EXTRA),
    ("isinf", M_MATH_EXTRA),
    ("isnan", M_MATH_EXTRA),
    ("all", M_MATH_EXTRA),
    ("any", M_MATH_EXTRA),
    ("amax", M_MATH_EXTRA),
    ("amin", M_MATH_EXTRA),
    ("count_nonzero", M_MATH_EXTRA),
    ("nansum", M_MATH_EXTRA),
    ("nanmean", M_MATH_EXTRA),
    ("tile", M_MATH_EXTRA),
    ("roll", M_MATH_EXTRA),
    ("pixel_shuffle", M_MATH_EXTRA),
    ("instance_norm", M_MATH_EXTRA),
    ("cross_entropy", M_MATH_EXTRA),
    ("huber_loss", M_MATH_EXTRA),
    ("hardtanh", M_MATH_EXTRA),
    ("hardsigmoid", M_MATH_EXTRA),
    ("glu", M_MATH_EXTRA),
    ("bucketize", M_MATH_EXTRA),
    ("histc", M_MATH_EXTRA),
    // d_batch2 (49 targets)
    ("embedding_bag", M_BATCH2),
    ("unfold", M_BATCH2),
    ("fold", M_BATCH2),
    ("grid_sample", M_BATCH2),
    ("affine_grid", M_BATCH2),
    ("pixel_unshuffle", M_BATCH2),
    ("channel_shuffle", M_BATCH2),
    ("cummax", M_BATCH2),
    ("cummin", M_BATCH2),
    ("logcumsumexp", M_BATCH2),
    ("scatter_reduce", M_BATCH2),
    ("index_put", M_BATCH2),
    ("index_add", M_BATCH2),
    ("masked_scatter", M_BATCH2),
    ("take", M_BATCH2),
    ("put", M_BATCH2),
    ("masked_select", M_BATCH2),
    ("index_fill", M_BATCH2),
    ("bincount", M_BATCH2),
    ("unique", M_BATCH2),
    ("kthvalue", M_BATCH2),
    ("median", M_BATCH2),
    ("quantile", M_BATCH2),
    ("histogram", M_BATCH2),
    ("searchsorted", M_BATCH2),
    ("meshgrid", M_BATCH2),
    ("cdist", M_BATCH2),
    ("pdist", M_BATCH2),
    ("renorm", M_BATCH2),
    ("bernoulli", M_BATCH2),
    ("multinomial", M_BATCH2),
    ("logspace", M_BATCH2),
    ("eye", M_BATCH2),
    ("diag", M_BATCH2),
    ("diagonal", M_BATCH2),
    ("trace", M_BATCH2),
    ("matrix_exp", M_BATCH2),
    ("slogdet", M_BATCH2),
    ("det", M_BATCH2),
    ("lstsq", M_BATCH2),
    ("pinverse", M_BATCH2),
    ("normal", M_BATCH2),
    ("uniform", M_BATCH2),
    ("triu", M_BATCH2),
    ("tril", M_BATCH2),
    ("hann_window", M_BATCH2),
    ("bartlett_window", M_BATCH2),
    ("blackman_window", M_BATCH2),
    ("stft", M_BATCH2),
    // d_float_special (35 targets)
    ("nextafter", M_FLOAT_SPECIAL),
    ("heaviside", M_FLOAT_SPECIAL),
    ("nan_to_num", M_FLOAT_SPECIAL),
    ("logaddexp", M_FLOAT_SPECIAL),
    ("logaddexp2", M_FLOAT_SPECIAL),
    ("sinc", M_FLOAT_SPECIAL),
    ("i0", M_FLOAT_SPECIAL),
    ("i1", M_FLOAT_SPECIAL),
    ("i0e", M_FLOAT_SPECIAL),
    ("i1e", M_FLOAT_SPECIAL),
    ("bessel_j0", M_FLOAT_SPECIAL),
    ("bessel_j1", M_FLOAT_SPECIAL),
    ("bessel_y0", M_FLOAT_SPECIAL),
    ("bessel_y1", M_FLOAT_SPECIAL),
    ("digamma", M_FLOAT_SPECIAL),
    ("lgamma", M_FLOAT_SPECIAL),
    ("polygamma", M_FLOAT_SPECIAL),
    ("mvlgamma", M_FLOAT_SPECIAL),
    ("erfinv", M_FLOAT_SPECIAL),
    ("erfcinv", M_FLOAT_SPECIAL),
    ("ndtri", M_FLOAT_SPECIAL),
    ("ndtr", M_FLOAT_SPECIAL),
    ("log_ndtr", M_FLOAT_SPECIAL),
    ("logit", M_FLOAT_SPECIAL),
    ("expit", M_FLOAT_SPECIAL),
    ("rad2deg", M_FLOAT_SPECIAL),
    ("deg2rad", M_FLOAT_SPECIAL),
    ("gcd", M_FLOAT_SPECIAL),
    ("lcm", M_FLOAT_SPECIAL),
    ("fmax", M_FLOAT_SPECIAL),
    ("fmin", M_FLOAT_SPECIAL),
    ("maximum", M_FLOAT_SPECIAL),
    ("minimum", M_FLOAT_SPECIAL),
    ("signbit", M_FLOAT_SPECIAL),
    // d_blas_extra (15 targets)
    ("addcdiv", M_BLAS_EXTRA),
    ("addcmul", M_BLAS_EXTRA),
    ("addr", M_BLAS_EXTRA),
    ("outer", M_BLAS_EXTRA),
    ("ger", M_BLAS_EXTRA),
    ("mv", M_BLAS_EXTRA),
    ("vdot", M_BLAS_EXTRA),
    ("baddbmm", M_BLAS_EXTRA),
    ("addbmm", M_BLAS_EXTRA),
    ("addmv", M_BLAS_EXTRA),
    ("kron", M_BLAS_EXTRA),
    ("inner", M_BLAS_EXTRA),
    ("trapz", M_BLAS_EXTRA),
    ("trapezoid", M_BLAS_EXTRA),
    ("cumulative_trapezoid", M_BLAS_EXTRA),
    // d_act_loss (22 targets)
    ("celu", M_ACT_LOSS),
    ("hardshrink", M_ACT_LOSS),
    ("softshrink", M_ACT_LOSS),
    ("tanhshrink", M_ACT_LOSS),
    ("threshold", M_ACT_LOSS),
    ("logsigmoid", M_ACT_LOSS),
    ("rrelu", M_ACT_LOSS),
    ("kl_div", M_ACT_LOSS),
    ("poisson_nll_loss", M_ACT_LOSS),
    ("margin_ranking_loss", M_ACT_LOSS),
    ("hinge_embedding_loss", M_ACT_LOSS),
    ("multilabel_margin_loss", M_ACT_LOSS),
    ("soft_margin_loss", M_ACT_LOSS),
    ("multilabel_soft_margin_loss", M_ACT_LOSS),
    ("cosine_embedding_loss", M_ACT_LOSS),
    ("triplet_margin_loss", M_ACT_LOSS),
    ("ctc_loss", M_ACT_LOSS),
    ("hamming_window", M_ACT_LOSS),
    ("kaiser_window", M_ACT_LOSS),
    ("gaussian_window", M_ACT_LOSS),
    ("exponential_window", M_ACT_LOSS),
    ("triangular_window", M_ACT_LOSS),
    // d_decomp (18 targets)
    ("cross", M_DECOMP),
    ("linalg_norm", M_DECOMP),
    ("frobenius_norm", M_DECOMP),
    ("nuclear_norm", M_DECOMP),
    ("matrix_rank", M_DECOMP),
    ("matrix_power", M_DECOMP),
    ("cholesky", M_DECOMP),
    ("cholesky_inverse", M_DECOMP),
    ("cholesky_solve", M_DECOMP),
    ("qr", M_DECOMP),
    ("svd", M_DECOMP),
    ("svdvals", M_DECOMP),
    ("eig", M_DECOMP),
    ("eigh", M_DECOMP),
    ("eigvals", M_DECOMP),
    ("eigvalsh", M_DECOMP),
    ("lu", M_DECOMP),
    ("triangular_solve", M_DECOMP),
    // d_scatter2 (5 targets)
    ("select_scatter", M_SCATTER2),
    ("slice_scatter", M_SCATTER2),
    ("diagonal_scatter", M_SCATTER2),
    ("index_copy", M_SCATTER2),
    ("narrow_copy", M_SCATTER2),
    // d_shape2 (22 targets)
    ("movedim", M_SHAPE2),
    ("moveaxis", M_SHAPE2),
    ("swapdims", M_SHAPE2),
    ("swapaxes", M_SHAPE2),
    ("column_stack", M_SHAPE2),
    ("row_stack", M_SHAPE2),
    ("dstack", M_SHAPE2),
    ("hstack", M_SHAPE2),
    ("vstack", M_SHAPE2),
    ("atleast_1d", M_SHAPE2),
    ("atleast_2d", M_SHAPE2),
    ("atleast_3d", M_SHAPE2),
    ("block_diag", M_SHAPE2),
    ("cartesian_prod", M_SHAPE2),
    ("combinations", M_SHAPE2),
    ("pad", M_SHAPE2),
    ("constant_pad_nd", M_SHAPE2),
    ("reflection_pad1d", M_SHAPE2),
    ("reflection_pad2d", M_SHAPE2),
    ("replication_pad1d", M_SHAPE2),
    ("replication_pad2d", M_SHAPE2),
    ("zero_pad2d", M_SHAPE2),
    // d_nn3d (13 targets)
    ("conv3d", M_NN3D),
    ("conv_transpose3d", M_NN3D),
    ("max_pool3d", M_NN3D),
    ("avg_pool3d", M_NN3D),
    ("adaptive_max_pool3d", M_NN3D),
    ("adaptive_avg_pool3d", M_NN3D),
    ("fractional_max_pool2d", M_NN3D),
    ("fractional_max_pool3d", M_NN3D),
    ("lp_pool1d", M_NN3D),
    ("lp_pool2d", M_NN3D),
    ("max_unpool1d", M_NN3D),
    ("max_unpool2d", M_NN3D),
    ("max_unpool3d", M_NN3D),
    // d_random (8 targets)
    ("rand", M_RANDOM),
    ("randn", M_RANDOM),
    ("randint", M_RANDOM),
    ("randperm", M_RANDOM),
    ("empty", M_RANDOM),
    ("zeros_like", M_RANDOM),
    ("ones_like", M_RANDOM),
    ("full_like", M_RANDOM),
    // d_rnn (6 targets)
    ("rnn_tanh_cell", M_RNN),
    ("rnn_relu_cell", M_RNN),
    ("gru_cell", M_RNN),
    ("lstm_cell", M_RNN),
    ("multi_head_attention_forward", M_RNN),
    ("transformer_encoder_layer_fwd", M_RNN),
    // d_solve (8 targets)
    ("lu_solve", M_SOLVE),
    ("lu_unpack", M_SOLVE),
    ("linalg_solve", M_SOLVE),
    ("linalg_inv", M_SOLVE),
    ("linalg_pinv", M_SOLVE),
    ("linalg_det", M_SOLVE),
    ("linalg_slogdet", M_SOLVE),
    ("linalg_cond", M_SOLVE),
    // d_fused (4 targets)
    ("flash_attention", M_FUSED),
    ("fused_swiglu", M_FUSED),
    ("fused_geglu", M_FUSED),
    ("fused_rmsnorm_residual", M_FUSED),
    // d_quant (9 targets)
    ("quantize_per_tensor", M_QUANT),
    ("dequantize_per_tensor", M_QUANT),
    ("quantize_per_channel", M_QUANT),
    ("dequantize_per_channel", M_QUANT),
    ("int8_gemm", M_QUANT),
    ("nf4_dequantize", M_QUANT),
    ("int4_unpack_dequantize", M_QUANT),
    ("w8a32_linear", M_QUANT),
    ("w4a32_linear", M_QUANT),
    // d_fft (16 targets)
    ("fft", M_FFT),
    ("ifft", M_FFT),
    ("rfft", M_FFT),
    ("irfft", M_FFT),
    ("fft2", M_FFT),
    ("ifft2", M_FFT),
    ("fftn", M_FFT),
    ("ifftn", M_FFT),
    ("fftshift", M_FFT),
    ("ifftshift", M_FFT),
    ("complex", M_FFT),
    ("real", M_FFT),
    ("imag", M_FFT),
    ("angle", M_FFT),
    ("polar", M_FFT),
    ("conj", M_FFT),
    // d_batch4 (48 targets)
    ("isclose", M_BATCH4),
    ("allclose", M_BATCH4),
    ("equal", M_BATCH4),
    ("isreal", M_BATCH4),
    ("is_complex", M_BATCH4),
    ("is_nonzero", M_BATCH4),
    ("nanprod", M_BATCH4),
    ("nanmin", M_BATCH4),
    ("nanmax", M_BATCH4),
    ("var_mean", M_BATCH4),
    ("std_mean", M_BATCH4),
    ("nanmedian", M_BATCH4),
    ("cov", M_BATCH4),
    ("corrcoef", M_BATCH4),
    ("as_strided", M_BATCH4),
    ("broadcast_to", M_BATCH4),
    ("broadcast_tensors", M_BATCH4),
    ("split", M_BATCH4),
    ("vsplit", M_BATCH4),
    ("hsplit", M_BATCH4),
    ("dsplit", M_BATCH4),
    ("tensor_split", M_BATCH4),
    ("take_along_dim", M_BATCH4),
    ("index_reduce", M_BATCH4),
    ("scatter_max", M_BATCH4),
    ("scatter_min", M_BATCH4),
    ("linalg_multi_dot", M_BATCH4),
    ("linalg_vander", M_BATCH4),
    ("linalg_vecdot", M_BATCH4),
    ("linalg_cross", M_BATCH4),
    ("linalg_tensordot", M_BATCH4),
    ("linalg_cholesky_ex", M_BATCH4),
    ("linalg_inv_ex", M_BATCH4),
    ("linalg_solve_ex", M_BATCH4),
    ("linalg_lu_factor", M_BATCH4),
    ("local_response_norm", M_BATCH4),
    ("adaptive_avg_pool1d", M_BATCH4),
    ("adaptive_max_pool1d", M_BATCH4),
    ("lp_pool3d", M_BATCH4),
    ("logsumexp", M_BATCH4),
    ("randn_like", M_BATCH4),
    ("rand_like", M_BATCH4),
    ("randint_like", M_BATCH4),
    ("empty_strided", M_BATCH4),
    ("view_as", M_BATCH4),
    ("expand_as", M_BATCH4),
    ("masked_select_extra", M_BATCH4),
    ("istft", M_BATCH4),
];

/// Lazily-built HashMap: target string → module index.
fn dispatch_table() -> &'static HashMap<&'static str, usize> {
    static TABLE: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = HashMap::with_capacity(TARGET_MODULE_PAIRS.len());
        for &(target, module_idx) in TARGET_MODULE_PAIRS {
            map.insert(target, module_idx);
        }
        map
    })
}

/// Execute a node by dispatching to the appropriate kernel.
/// Uses O(1) HashMap lookup instead of linear scanning through 22 modules.
pub(crate) fn dispatch_node(
    node: &Node,
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<()> {
    let table = dispatch_table();
    let target = node.target.as_str();

    match table.get(target) {
        Some(&module_idx) => {
            let handled = MODULE_FNS[module_idx](node, slots, capsules)?;
            if handled {
                Ok(())
            } else {
                // Target is in the table but the module didn't handle it.
                // This shouldn't happen if the table is consistent with the modules.
                Err(unsupported(&format!(
                    "internal error: target '{}' mapped to module {module_idx} but not handled",
                    node.target
                )))
            }
        }
        None => Err(unsupported(&format!("unknown target {:?}", node.target))),
    }
}
