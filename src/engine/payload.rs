//! Payload wire types and supported-target registry.
//!
//! Wire format between the Python parser (\_parser.py\) and the Rust
//! execution engine. Extracted from \engine.rs\; the engine root
//! re-exports these so \crate::engine::{Payload, Node}\ keeps working.

use crate::dlpack::{dtype_from_spec, unsupported, DType, OwnedTensor};
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

#[derive(Deserialize, Clone)]
pub struct Payload {
    #[serde(default)]
    pub inputs: Vec<InputSpec>,
    pub nodes: Vec<Node>,
    /// Node ids whose output capsules the caller wants back.
    #[serde(default)]
    pub outputs: Vec<u32>,
}

#[derive(Deserialize, Clone)]
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
        shape: Arc<[i64]>,
        strides: Arc<[i64]>,
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

/// All supported targets — keep in sync with _parser.py.
/// Uses OnceLock to avoid allocating ~450 Strings on every call.
pub fn supported_targets() -> Vec<String> {
    static TARGETS: OnceLock<Vec<String>> = OnceLock::new();
    TARGETS
        .get_or_init(|| {
            SUPPORTED_TARGETS_LIST
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
        .clone()
}

/// Static list of all supported target strings (zero-allocation).
static SUPPORTED_TARGETS_LIST: &[&str] = &[
    // Phase 1: elementwise
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    // Phase 2: math/comparison
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "abs",
    "neg",
    "sign",
    "sqrt",
    "rsqrt",
    "exp",
    "log",
    "reciprocal",
    "ceil",
    "floor",
    "clamp",
    "clamp_min",
    "clamp_max",
    "pow",
    "sin",
    "cos",
    "round",
    // Phase 2: logical + dtype cast
    "logical_and",
    "logical_or",
    "logical_not",
    "to_dtype",
    // Phase 2: activations
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "leaky_relu",
    "elu",
    "selu",
    "softplus",
    "hardswish",
    "mish",
    "softmax",
    "log_softmax",
    "threshold_backward",
    // Phase 2: reductions
    "sum",
    "mean",
    "max",
    "max_reduce",
    "min",
    "min_reduce",
    "argmax",
    "argmin",
    "std",
    "var",
    "cumsum",
    "prod",
    "norm",
    "linalg_vector_norm",
    // Phase 2: linalg
    "matmul",
    "bmm",
    "linear",
    "dot",
    "addmm",
    // Phase 2: shape ops
    "t",
    "transpose",
    "index_select",
    "gather",
    // Phase 2: norm
    "layer_norm",
    "batch_norm",
    "group_norm",
    "rms_norm",
    // Phase 2: shape ops
    "cat",
    "stack",
    "reshape",
    "permute",
    "expand",
    "where",
    "masked_fill",
    "flip",
    "narrow",
    "select",
    "contiguous",
    "chunk_narrow",
    "squeeze",
    "unsqueeze",
    "unflatten",
    "dropout",
    // Phase 10: tensor creation
    "full",
    "zeros",
    "ones",
    "arange",
    "linspace",
    // Phase 3: convolution & pooling & upsampling
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    "conv_transpose2d",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "max_pool1d",
    "avg_pool1d",
    "upsample_nearest2d",
    "upsample_bilinear2d",
    "interpolate",
    "flatten",
    // Phase 4: transformer stack
    "scalar_tensor",
    "embedding",
    "scaled_dot_product_attention",
    "rope",
    "nll_loss_forward",
    "mse_loss",
    "smooth_l1_loss",
    "binary_cross_entropy",
    // Phase 7: extended ops
    "scatter",
    "scatter_add",
    "topk",
    "sort",
    "argsort",
    "unbind",
    "chunk",
    "getitem",
    "repeat_interleave",
    "repeat",
    "einsum",
    "prelu",
    "nonzero",
    "clamp_tensor",
    // v0.2 extra 50 ops batch 1
    "atan",
    "asin",
    "acos",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "expm1",
    "log1p",
    "log2",
    "log10",
    "atan2",
    "hypot",
    "fmod",
    "remainder",
    "copysign",
    "lerp",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "isfinite",
    "isinf",
    "isnan",
    "all",
    "any",
    "amax",
    "amin",
    "count_nonzero",
    "nansum",
    "nanmean",
    "tile",
    "roll",
    "pixel_shuffle",
    "instance_norm",
    "cross_entropy",
    "huber_loss",
    "hardtanh",
    "hardsigmoid",
    "glu",
    "trunc",
    "frac",
    "square",
    "exp2",
    "ldexp",
    "bucketize",
    "histc",
    // Extra ops batch 2 (49 ops)
    "embedding_bag",
    "unfold",
    "fold",
    "grid_sample",
    "affine_grid",
    "pixel_unshuffle",
    "channel_shuffle",
    "cummax",
    "cummin",
    "logcumsumexp",
    "scatter_reduce",
    "index_put",
    "index_add",
    "masked_scatter",
    "take",
    "put",
    "masked_select",
    "index_fill",
    "bincount",
    "unique",
    "kthvalue",
    "median",
    "quantile",
    "histogram",
    "searchsorted",
    "meshgrid",
    "cdist",
    "pdist",
    "renorm",
    "bernoulli",
    "multinomial",
    "logspace",
    "eye",
    "diag",
    "diagonal",
    "trace",
    "matrix_exp",
    "slogdet",
    "det",
    "lstsq",
    "pinverse",
    "normal",
    "uniform",
    "triu",
    "tril",
    "hann_window",
    "bartlett_window",
    "blackman_window",
    "stft",
    // Extra ops batch 3 (149 ops) -> Total exactly 375 ops!
    "nextafter",
    "heaviside",
    "nan_to_num",
    "logaddexp",
    "logaddexp2",
    "sinc",
    "i0",
    "i1",
    "i0e",
    "i1e",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "digamma",
    "lgamma",
    "polygamma",
    "mvlgamma",
    "erfinv",
    "erfcinv",
    "ndtri",
    "ndtr",
    "log_ndtr",
    "logit",
    "expit",
    "rad2deg",
    "deg2rad",
    "gcd",
    "lcm",
    "fmax",
    "fmin",
    "maximum",
    "minimum",
    "signbit",
    "addcdiv",
    "addcmul",
    "addr",
    "outer",
    "mv",
    "vdot",
    "baddbmm",
    "addbmm",
    "addmv",
    "kron",
    "inner",
    "trapz",
    "trapezoid",
    "cumulative_trapezoid",
    "celu",
    "hardshrink",
    "softshrink",
    "tanhshrink",
    "threshold",
    "logsigmoid",
    "rrelu",
    "kl_div",
    "poisson_nll_loss",
    "margin_ranking_loss",
    "hinge_embedding_loss",
    "multilabel_margin_loss",
    "soft_margin_loss",
    "multilabel_soft_margin_loss",
    "cosine_embedding_loss",
    "triplet_margin_loss",
    "ctc_loss",
    "hamming_window",
    "kaiser_window",
    "gaussian_window",
    "exponential_window",
    "triangular_window",
    "cross",
    "linalg_norm",
    "frobenius_norm",
    "nuclear_norm",
    "matrix_rank",
    "matrix_power",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "qr",
    "svd",
    "svdvals",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "lu",
    "triangular_solve",
    "select_scatter",
    "slice_scatter",
    "diagonal_scatter",
    "index_copy",
    "narrow_copy",
    "movedim",
    "moveaxis",
    "swapdims",
    "swapaxes",
    "column_stack",
    "row_stack",
    "dstack",
    "hstack",
    "vstack",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block_diag",
    "cartesian_prod",
    "combinations",
    "pad",
    "constant_pad_nd",
    "reflection_pad1d",
    "reflection_pad2d",
    "replication_pad1d",
    "replication_pad2d",
    "zero_pad2d",
    "conv3d",
    "conv_transpose3d",
    "max_pool3d",
    "avg_pool3d",
    "adaptive_max_pool3d",
    "adaptive_avg_pool3d",
    "fractional_max_pool2d",
    "fractional_max_pool3d",
    "lp_pool1d",
    "lp_pool2d",
    "max_unpool1d",
    "max_unpool2d",
    "max_unpool3d",
    "rand",
    "randn",
    "randint",
    "randperm",
    "empty",
    "zeros_like",
    "ones_like",
    "full_like",
    "rnn_tanh_cell",
    "rnn_relu_cell",
    "gru_cell",
    "lstm_cell",
    "multi_head_attention_forward",
    "transformer_encoder_layer_fwd",
    "lu_solve",
    "lu_unpack",
    "linalg_solve",
    "linalg_inv",
    "linalg_pinv",
    "linalg_det",
    "linalg_slogdet",
    "linalg_cond",
    // Advanced LLM & FlashAttention
    "flash_attention",
    "fused_swiglu",
    "fused_geglu",
    "fused_rmsnorm_residual",
    // Universal Low-Bit Quantization & GEMM
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantize_per_channel",
    "dequantize_per_channel",
    "int8_gemm",
    "nf4_dequantize",
    "int4_unpack_dequantize",
    // Universal FFT & Complex Suite
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "fftshift",
    "ifftshift",
    "complex",
    "real",
    "imag",
    "angle",
    "polar",
    "conj",
    // Extra ops batch 4 (48 ops) -> Total 450 ops
    "isclose",
    "allclose",
    "equal",
    "isreal",
    "is_complex",
    "is_nonzero",
    "nanprod",
    "nanmin",
    "nanmax",
    "var_mean",
    "std_mean",
    "nanmedian",
    "cov",
    "corrcoef",
    "as_strided",
    "broadcast_to",
    "broadcast_tensors",
    "split",
    "vsplit",
    "hsplit",
    "dsplit",
    "tensor_split",
    "take_along_dim",
    "index_reduce",
    "scatter_max",
    "scatter_min",
    "linalg_multi_dot",
    "linalg_vander",
    "linalg_vecdot",
    "linalg_cross",
    "linalg_tensordot",
    "linalg_cholesky_ex",
    "linalg_inv_ex",
    "linalg_solve_ex",
    "linalg_lu_factor",
    "local_response_norm",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "lp_pool3d",
    "logsumexp",
    "randn_like",
    "rand_like",
    "randint_like",
    "empty_strided",
    "view_as",
    "expand_as",
    "masked_select_extra",
    "istft",
];

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
