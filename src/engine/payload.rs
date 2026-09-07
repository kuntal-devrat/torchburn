//! Payload wire types and supported-target registry.
//!
//! Wire format between the Python parser (\_parser.py\) and the Rust
//! execution engine. Extracted from \engine.rs\; the engine root
//! re-exports these so \crate::engine::{Payload, Node}\ keeps working.

use crate::dlpack::{dtype_from_spec, unsupported, DType, OwnedTensor};
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Payload {
    #[serde(default)]
    pub inputs: Vec<InputSpec>,
    pub nodes: Vec<Node>,
    /// Node ids whose output capsules the caller wants back.
    #[serde(default)]
    pub outputs: Vec<u32>,
}

#[derive(Deserialize)]
pub struct InputSpec {
    pub shape: Vec<i64>,
    pub dtype: String,
}

#[derive(Deserialize, Clone)]
pub struct Node {
    pub id: u32,
    pub target: String,
    #[serde(default)]
    pub args: Vec<ArgRef>,
    #[serde(default)]
    pub kwargs: HashMap<String, serde_json::Value>,
}

/// One node argument reference. `index` points at a slot (input capsule or
/// owned intermediate); `value` carries serialised constants / index lists.
/// The Python side also emits a `kind` field ("slot"/"input"/"const"/...)
/// for protocol documentation; serde ignores it here since it is not read.
#[derive(Deserialize, Clone)]
pub struct ArgRef {
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
}

/// One live tensor slot: either an input capsule, a Rust-owned intermediate,
/// or a tuple of tensors (for multi-output ops like unbind, chunk, sort).
pub(crate) enum Slot {
    Input(usize),
    Owned(OwnedTensor),
    View {
        data: *const u8,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: DType,
    },
    Tuple(Vec<OwnedTensor>),
}
impl Slot {
    /// Take an element from a Tuple slot, leaving a None placeholder.
    /// Returns None if the slot is not a Tuple or the index is out of bounds.
    pub(crate) fn take_tuple_elem(&mut self, index: usize) -> Option<OwnedTensor> {
        match self {
            Slot::Tuple(elems) => {
                if index < elems.len() {
                    Some(std::mem::take(&mut elems[index]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(crate) fn is_tuple(&self) -> bool {
        matches!(self, Slot::Tuple(_))
    }

    pub(crate) fn tuple_len(&self) -> usize {
        match self {
            Slot::Tuple(elems) => elems.len(),
            _ => 0,
        }
    }
}

/// All supported targets — keep in sync with _parser.py
pub fn supported_targets() -> Vec<String> {
    vec![
        // Phase 1: elementwise
        "add".into(),
        "sub".into(),
        "mul".into(),
        "div".into(),
        "relu".into(),
        // Phase 2: math/comparison
        "eq".into(),
        "ne".into(),
        "lt".into(),
        "le".into(),
        "gt".into(),
        "ge".into(),
        "abs".into(),
        "neg".into(),
        "sign".into(),
        "sqrt".into(),
        "rsqrt".into(),
        "exp".into(),
        "log".into(),
        "reciprocal".into(),
        "ceil".into(),
        "floor".into(),
        "clamp".into(),
        "clamp_min".into(),
        "clamp_max".into(),
        "pow".into(),
        "sin".into(),
        "cos".into(),
        "round".into(),
        // Phase 2: logical + dtype cast
        "logical_and".into(),
        "logical_or".into(),
        "logical_not".into(),
        "to_dtype".into(),
        // Phase 2: activations
        "sigmoid".into(),
        "tanh".into(),
        "gelu".into(),
        "silu".into(),
        "leaky_relu".into(),
        "elu".into(),
        "selu".into(),
        "softplus".into(),
        "hardswish".into(),
        "mish".into(),
        "softmax".into(),
        "log_softmax".into(),
        "threshold_backward".into(),
        // Phase 2: reductions
        "sum".into(),
        "mean".into(),
        "max".into(),
        "max_reduce".into(),
        "min".into(),
        "min_reduce".into(),
        "argmax".into(),
        "argmin".into(),
        "std".into(),
        "var".into(),
        "cumsum".into(),
        "prod".into(),
        "norm".into(),
        "linalg_vector_norm".into(),
        // Phase 2: linalg
        "matmul".into(),
        "bmm".into(),
        "linear".into(),
        "dot".into(),
        "addmm".into(),
        // Phase 2: shape ops
        "t".into(),
        "transpose".into(),
        "index_select".into(),
        "gather".into(),
        // Phase 2: norm
        "layer_norm".into(),
        "batch_norm".into(),
        "group_norm".into(),
        "rms_norm".into(),
        // Phase 2: shape ops
        "cat".into(),
        "stack".into(),
        "reshape".into(),
        "permute".into(),
        "expand".into(),
        "where".into(),
        "masked_fill".into(),
        "flip".into(),
        "narrow".into(),
        "select".into(),
        "contiguous".into(),
        "chunk_narrow".into(),
        "squeeze".into(),
        "unsqueeze".into(),
        "unflatten".into(),
        "dropout".into(),
        // Phase 10: tensor creation
        "full".into(),
        "zeros".into(),
        "ones".into(),
        "arange".into(),
        "linspace".into(),
        // Phase 3: convolution & pooling & upsampling
        "conv1d".into(),
        "conv2d".into(),
        "conv_transpose1d".into(),
        "conv_transpose2d".into(),
        "max_pool2d".into(),
        "avg_pool2d".into(),
        "adaptive_avg_pool2d".into(),
        "adaptive_max_pool2d".into(),
        "max_pool1d".into(),
        "avg_pool1d".into(),
        "upsample_nearest2d".into(),
        "upsample_bilinear2d".into(),
        "interpolate".into(),
        "flatten".into(),
        // Phase 4: transformer stack
        "scalar_tensor".into(),
        "embedding".into(),
        "scaled_dot_product_attention".into(),
        "rope".into(),
        "nll_loss_forward".into(),
        "mse_loss".into(),
        "smooth_l1_loss".into(),
        "binary_cross_entropy".into(),
        // Phase 7: extended ops
        "scatter".into(),
        "scatter_add".into(),
        "topk".into(),
        "sort".into(),
        "argsort".into(),
        "unbind".into(),
        "chunk".into(),
        "getitem".into(),
        "repeat_interleave".into(),
        "repeat".into(),
        "einsum".into(),
        "prelu".into(),
        "nonzero".into(),
        "clamp_tensor".into(),
        // v0.2 extra 50 ops batch 1
        "atan".into(),
        "asin".into(),
        "acos".into(),
        "sinh".into(),
        "cosh".into(),
        "asinh".into(),
        "acosh".into(),
        "atanh".into(),
        "erf".into(),
        "erfc".into(),
        "expm1".into(),
        "log1p".into(),
        "log2".into(),
        "log10".into(),
        "atan2".into(),
        "hypot".into(),
        "fmod".into(),
        "remainder".into(),
        "copysign".into(),
        "lerp".into(),
        "bitwise_and".into(),
        "bitwise_or".into(),
        "bitwise_xor".into(),
        "bitwise_not".into(),
        "isfinite".into(),
        "isinf".into(),
        "isnan".into(),
        "all".into(),
        "any".into(),
        "amax".into(),
        "amin".into(),
        "count_nonzero".into(),
        "nansum".into(),
        "nanmean".into(),
        "tile".into(),
        "roll".into(),
        "pixel_shuffle".into(),
        "instance_norm".into(),
        "cross_entropy".into(),
        "huber_loss".into(),
        "hardtanh".into(),
        "hardsigmoid".into(),
        "glu".into(),
        "trunc".into(),
        "frac".into(),
        "square".into(),
        "exp2".into(),
        "ldexp".into(),
        "bucketize".into(),
        "histc".into(),
        // Extra ops batch 2 (49 ops)
        "embedding_bag".into(),
        "unfold".into(),
        "fold".into(),
        "grid_sample".into(),
        "affine_grid".into(),
        "pixel_unshuffle".into(),
        "channel_shuffle".into(),
        "cummax".into(),
        "cummin".into(),
        "logcumsumexp".into(),
        "scatter_reduce".into(),
        "index_put".into(),
        "index_add".into(),
        "masked_scatter".into(),
        "take".into(),
        "put".into(),
        "masked_select".into(),
        "index_fill".into(),
        "bincount".into(),
        "unique".into(),
        "kthvalue".into(),
        "median".into(),
        "quantile".into(),
        "histogram".into(),
        "searchsorted".into(),
        "meshgrid".into(),
        "cdist".into(),
        "pdist".into(),
        "renorm".into(),
        "bernoulli".into(),
        "multinomial".into(),
        "logspace".into(),
        "eye".into(),
        "diag".into(),
        "diagonal".into(),
        "trace".into(),
        "matrix_exp".into(),
        "slogdet".into(),
        "det".into(),
        "lstsq".into(),
        "pinverse".into(),
        "normal".into(),
        "uniform".into(),
        "triu".into(),
        "tril".into(),
        "hann_window".into(),
        "bartlett_window".into(),
        "blackman_window".into(),
        "stft".into(),
        // Extra ops batch 3 (149 ops) -> Total exactly 375 ops!
        "nextafter".into(),
        "heaviside".into(),
        "nan_to_num".into(),
        "logaddexp".into(),
        "logaddexp2".into(),
        "sinc".into(),
        "i0".into(),
        "i1".into(),
        "i0e".into(),
        "i1e".into(),
        "bessel_j0".into(),
        "bessel_j1".into(),
        "bessel_y0".into(),
        "bessel_y1".into(),
        "digamma".into(),
        "lgamma".into(),
        "polygamma".into(),
        "mvlgamma".into(),
        "erfinv".into(),
        "erfcinv".into(),
        "ndtri".into(),
        "ndtr".into(),
        "log_ndtr".into(),
        "logit".into(),
        "expit".into(),
        "rad2deg".into(),
        "deg2rad".into(),
        "gcd".into(),
        "lcm".into(),
        "fmax".into(),
        "fmin".into(),
        "maximum".into(),
        "minimum".into(),
        "signbit".into(),
        "addcdiv".into(),
        "addcmul".into(),
        "addr".into(),
        "outer".into(),
        "mv".into(),
        "vdot".into(),
        "baddbmm".into(),
        "addbmm".into(),
        "addmv".into(),
        "kron".into(),
        "inner".into(),
        "trapz".into(),
        "trapezoid".into(),
        "cumulative_trapezoid".into(),
        "celu".into(),
        "hardshrink".into(),
        "softshrink".into(),
        "tanhshrink".into(),
        "threshold".into(),
        "logsigmoid".into(),
        "rrelu".into(),
        "kl_div".into(),
        "poisson_nll_loss".into(),
        "margin_ranking_loss".into(),
        "hinge_embedding_loss".into(),
        "multilabel_margin_loss".into(),
        "soft_margin_loss".into(),
        "multilabel_soft_margin_loss".into(),
        "cosine_embedding_loss".into(),
        "triplet_margin_loss".into(),
        "ctc_loss".into(),
        "hamming_window".into(),
        "kaiser_window".into(),
        "gaussian_window".into(),
        "exponential_window".into(),
        "triangular_window".into(),
        "cross".into(),
        "linalg_norm".into(),
        "frobenius_norm".into(),
        "nuclear_norm".into(),
        "matrix_rank".into(),
        "matrix_power".into(),
        "cholesky".into(),
        "cholesky_inverse".into(),
        "cholesky_solve".into(),
        "qr".into(),
        "svd".into(),
        "svdvals".into(),
        "eig".into(),
        "eigh".into(),
        "eigvals".into(),
        "eigvalsh".into(),
        "lu".into(),
        "triangular_solve".into(),
        "select_scatter".into(),
        "slice_scatter".into(),
        "diagonal_scatter".into(),
        "index_copy".into(),
        "narrow_copy".into(),
        "movedim".into(),
        "moveaxis".into(),
        "swapdims".into(),
        "swapaxes".into(),
        "column_stack".into(),
        "row_stack".into(),
        "dstack".into(),
        "hstack".into(),
        "vstack".into(),
        "atleast_1d".into(),
        "atleast_2d".into(),
        "atleast_3d".into(),
        "block_diag".into(),
        "cartesian_prod".into(),
        "combinations".into(),
        "pad".into(),
        "constant_pad_nd".into(),
        "reflection_pad1d".into(),
        "reflection_pad2d".into(),
        "replication_pad1d".into(),
        "replication_pad2d".into(),
        "zero_pad2d".into(),
        "conv3d".into(),
        "conv_transpose3d".into(),
        "max_pool3d".into(),
        "avg_pool3d".into(),
        "adaptive_max_pool3d".into(),
        "adaptive_avg_pool3d".into(),
        "fractional_max_pool2d".into(),
        "fractional_max_pool3d".into(),
        "lp_pool1d".into(),
        "lp_pool2d".into(),
        "max_unpool1d".into(),
        "max_unpool2d".into(),
        "max_unpool3d".into(),
        "rand".into(),
        "randn".into(),
        "randint".into(),
        "randperm".into(),
        "empty".into(),
        "zeros_like".into(),
        "ones_like".into(),
        "full_like".into(),
        "rnn_tanh_cell".into(),
        "rnn_relu_cell".into(),
        "gru_cell".into(),
        "lstm_cell".into(),
        "multi_head_attention_forward".into(),
        "transformer_encoder_layer_fwd".into(),
        "lu_solve".into(),
        "lu_unpack".into(),
        "linalg_solve".into(),
        "linalg_inv".into(),
        "linalg_pinv".into(),
        "linalg_det".into(),
        "linalg_slogdet".into(),
        "linalg_cond".into(),
        // Advanced LLM & FlashAttention
        "flash_attention".into(),
        "fused_swiglu".into(),
        "fused_geglu".into(),
        "fused_rmsnorm_residual".into(),
        // Universal Low-Bit Quantization & GEMM
        "quantize_per_tensor".into(),
        "dequantize_per_tensor".into(),
        "quantize_per_channel".into(),
        "dequantize_per_channel".into(),
        "int8_gemm".into(),
        "nf4_dequantize".into(),
        "int4_unpack_dequantize".into(),
        // Universal FFT & Complex Suite
        "fft".into(),
        "ifft".into(),
        "rfft".into(),
        "irfft".into(),
        "fft2".into(),
        "ifft2".into(),
        "fftn".into(),
        "ifftn".into(),
        "fftshift".into(),
        "ifftshift".into(),
        "complex".into(),
        "real".into(),
        "imag".into(),
        "angle".into(),
        "polar".into(),
        "conj".into(),
        // Extra ops batch 4 (48 ops) -> Total 450 ops
        "isclose".into(),
        "allclose".into(),
        "equal".into(),
        "isreal".into(),
        "is_complex".into(),
        "is_nonzero".into(),
        "nanprod".into(),
        "nanmin".into(),
        "nanmax".into(),
        "var_mean".into(),
        "std_mean".into(),
        "nanmedian".into(),
        "cov".into(),
        "corrcoef".into(),
        "as_strided".into(),
        "broadcast_to".into(),
        "broadcast_tensors".into(),
        "split".into(),
        "vsplit".into(),
        "hsplit".into(),
        "dsplit".into(),
        "tensor_split".into(),
        "take_along_dim".into(),
        "index_reduce".into(),
        "scatter_max".into(),
        "scatter_min".into(),
        "linalg_multi_dot".into(),
        "linalg_vander".into(),
        "linalg_vecdot".into(),
        "linalg_cross".into(),
        "linalg_tensordot".into(),
        "linalg_cholesky_ex".into(),
        "linalg_inv_ex".into(),
        "linalg_solve_ex".into(),
        "linalg_lu_factor".into(),
        "local_response_norm".into(),
        "adaptive_avg_pool1d".into(),
        "adaptive_max_pool1d".into(),
        "lp_pool3d".into(),
        "logsumexp".into(),
        "randn_like".into(),
        "rand_like".into(),
        "randint_like".into(),
        "empty_strided".into(),
        "view_as".into(),
        "expand_as".into(),
        "masked_select_extra".into(),
        "istft".into(),
    ]
}

// ---------------------------------------------------------------------------
// Direct dict-to-payload conversion (bypasses JSON serialisation)
// ---------------------------------------------------------------------------

use pyo3::types::{PyDict, PyList};

/// Convert a Python dict directly to a `Payload`, bypassing JSON.
/// This eliminates the ~15us Python json.dumps overhead per call.
pub fn dict_to_payload(dict: &Bound<'_, PyDict>) -> PyResult<Payload> {
    // --- inputs ---
    let inputs: Vec<InputSpec> = match dict.get_item("inputs")? {
        Some(obj) => {
            let list: &Bound<'_, PyList> = obj.downcast().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("payload 'inputs' must be a list")
            })?;
            let mut v = Vec::with_capacity(list.len());
            for item in list.iter() {
                let d: &Bound<'_, PyDict> = item.downcast().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("input spec must be a dict")
                })?;
                let shape: Vec<i64> = match d.get_item("shape")? {
                    Some(o) => {
                        let l: &Bound<'_, PyList> = o.downcast().map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err("input 'shape' must be a list")
                        })?;
                        l.iter().map(|x| x.extract::<i64>().unwrap_or(0)).collect()
                    }
                    None => vec![],
                };
                let dtype: String = match d.get_item("dtype")? {
                    Some(o) => o.extract::<String>().unwrap_or_else(|_| "f32".to_string()),
                    None => "f32".to_string(),
                };
                if shape.iter().any(|&d| d < 0) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "input shape contains negative dim",
                    ));
                }
                if dtype_from_spec(&dtype).is_none() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown dtype '{dtype}'"
                    )));
                }
                v.push(InputSpec { shape, dtype });
            }
            v
        }
        None => vec![],
    };

    // --- nodes ---
    let nodes: Vec<Node> = match dict.get_item("nodes")? {
        Some(obj) => {
            let list: &Bound<'_, PyList> = obj.downcast().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("payload 'nodes' must be a list")
            })?;
            let mut v = Vec::with_capacity(list.len());
            for item in list.iter() {
                let d: &Bound<'_, PyDict> = item
                    .downcast()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("node must be a dict"))?;
                let id: u32 = d
                    .get_item("id")?
                    .map(|o| o.extract().unwrap_or(0))
                    .unwrap_or(0);
                let target: String = d
                    .get_item("target")?
                    .map(|o| o.extract::<String>().unwrap_or_default())
                    .unwrap_or_default();
                if target.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "node {id} missing target"
                    )));
                }

                // Parse args: list of dicts with optional index/value
                let args: Vec<ArgRef> = match d.get_item("args")? {
                    Some(ao) => {
                        let al: &Bound<'_, PyList> = ao.downcast().map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err("node 'args' must be a list")
                        })?;
                        al.iter()
                            .map(|a| {
                                if let Ok(ad) = a.downcast::<PyDict>() {
                                    let index: Option<usize> =
                                        ad.get_item("index")?.and_then(|o| o.extract().ok());
                                    let value: Option<serde_json::Value> = ad
                                        .get_item("value")?
                                        .map(|o| py_to_json(&o).ok())
                                        .flatten();
                                    Ok(ArgRef { index, value })
                                } else {
                                    Ok(ArgRef {
                                        index: None,
                                        value: None,
                                    })
                                }
                            })
                            .collect::<PyResult<Vec<_>>>()?
                    }
                    None => vec![],
                };

                // Parse kwargs: dict of string -> JSON value
                let kwargs: HashMap<String, serde_json::Value> = match d.get_item("kwargs")? {
                    Some(ko) => {
                        let kd: &Bound<'_, PyDict> = ko.downcast().map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err("node 'kwargs' must be a dict")
                        })?;
                        let mut m = HashMap::new();
                        for (k, v) in kd.iter() {
                            if let (Ok(ks), Some(jv)) = (k.extract::<String>(), py_to_json(&v).ok())
                            {
                                m.insert(ks, jv);
                            }
                        }
                        m
                    }
                    None => HashMap::new(),
                };

                v.push(Node {
                    id,
                    target,
                    args,
                    kwargs,
                });
            }
            v
        }
        None => vec![],
    };

    // --- outputs ---
    let outputs: Vec<u32> = match dict.get_item("outputs")? {
        Some(obj) => {
            let list: &Bound<'_, PyList> = obj.downcast().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("payload 'outputs' must be a list")
            })?;
            list.iter()
                .filter_map(|o| o.extract::<u32>().ok())
                .collect()
        }
        None => vec![],
    };
    if nodes.is_empty() && !inputs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "payload has inputs but no nodes",
        ));
    }
    // DoS protection: same 10 MB limit as JSON path – estimate from counts.
    if nodes.len() > 100_000 {
        return Err(unsupported(&format!(
            "payload too large ({} nodes > limit)",
            nodes.len()
        )));
    }
    if inputs.len() > 1024 {
        return Err(unsupported(&format!(
            "payload too large ({} inputs > limit)",
            inputs.len()
        )));
    }

    Ok(Payload {
        inputs,
        nodes,
        outputs,
    })
}

/// Recursively convert a Python object to a serde_json::Value.
fn py_to_json(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<serde_json::Value> {
    if let Ok(v) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(v));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(serde_json::Value::Number(v.into()));
    }
    if let Ok(v) = obj.extract::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(v) {
            return Ok(serde_json::Value::Number(n));
        }
        return Ok(serde_json::Value::String(v.to_string()));
    }
    if let Ok(v) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(v));
    }
    if let Ok(lst) = obj.downcast::<PyList>() {
        let arr: Vec<serde_json::Value> = lst.iter().filter_map(|x| py_to_json(&x).ok()).collect();
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(d) = obj.downcast::<PyDict>() {
        let mut m = serde_json::Map::new();
        for (k, v) in d.iter() {
            if let Ok(ks) = k.extract::<String>() {
                if let Ok(jv) = py_to_json(&v) {
                    m.insert(ks, jv);
                }
            }
        }
        return Ok(serde_json::Value::Object(m));
    }
    // Fallback: string representation
    Ok(serde_json::Value::String(obj.to_string()))
}
