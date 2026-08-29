//! Payload execution engine.
//!
//! The Python side slices a `torch.fx` graph into runs of supported nodes
//! (REQ-002) and sends each run as a compact JSON payload alongside the
//! DLPack capsules backing its tensor inputs. This module walks the payload,
//! dispatches to the native zero-copy kernels (or the optional Burn engine),
//! and returns one output capsule per requested node.

use crate::dlpack::{BorrowedTensor, CapsuleRef, DType, OwnedTensor, contiguous_strides, dtype_from_spec, unsupported};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

use crate::{ops, activations, math_ops, reductions, linalg, norm, shape_ops, convolution, pooling, upsample, embedding, losses, attention, fusion, ops_phase7, extra_ops, extra_ops2, extra_ops3, quantization, fft_complex};
use std::sync::{RwLock, OnceLock};

// ---------------------------------------------------------------------------
// Global graph cache: prepare_graph / execute_prepared
// ---------------------------------------------------------------------------

/// A prepared graph: parsed nodes + metadata, ready for execution.
pub(crate) struct PreparedGraph {
    payload: Payload,
}

struct GraphCache {
    graphs: HashMap<i64, PreparedGraph>,
    order: std::collections::VecDeque<i64>,
}

fn graph_cache() -> &'static RwLock<GraphCache> {
    static INSTANCE: OnceLock<RwLock<GraphCache>> = OnceLock::new();
    INSTANCE.get_or_init(|| RwLock::new(GraphCache {
        graphs: HashMap::new(),
        order: std::collections::VecDeque::new(),
    }))
}

static NEXT_HANDLE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1);

/// Maximum payload JSON size accepted (DoS protection).
pub const MAX_PAYLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Parse a Python dict into a PreparedGraph and cache it. Returns a handle.
pub fn prepare_graph(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<i64> {
    let payload = dict_to_payload(dict)?;
    let handle = NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut cache = graph_cache().write().unwrap_or_else(|e| e.into_inner());
    cache.graphs.insert(handle, PreparedGraph { payload });
    cache.order.push_back(handle);
    // Evict oldest if over 1024 prepared graphs.
    while cache.graphs.len() > 1024 {
        if let Some(old_handle) = cache.order.pop_front() {
            cache.graphs.remove(&old_handle);
        } else {
            break;
        }
    }
    Ok(handle)
}

/// Release a prepared graph from the cache.
pub fn release_graph(handle: i64) {
    let mut cache = graph_cache().write().unwrap_or_else(|e| e.into_inner());
    cache.graphs.remove(&handle);
    cache.order.retain(|&h| h != handle);
}

/// Execute a prepared graph with new input tensors.
pub fn execute_prepared(
    py: Python<'_>,
    handle: i64,
    capsules: &[Bound<'_, PyCapsule>],
) -> PyResult<Vec<Py<PyCapsule>>> {
    let cache = graph_cache().read().unwrap_or_else(|e| e.into_inner());
    let graph = cache.graphs.get(&handle).ok_or_else(||
        pyo3::exceptions::PyValueError::new_err(format!("invalid graph handle {handle}"))
    )?;

    #[cfg(feature = "burn")]
    if engine_is_burn() {
        return burn_impl::execute_burn(py, &graph.payload, capsules);
    }

    let refs: Vec<CapsuleRef> = capsules
        .iter()
        .map(crate::dlpack::capsule_ref)
        .collect::<PyResult<_>>()?;
    let native_out = py.allow_threads(|| execute_native(&graph.payload, &refs))?;

    let mut out = Vec::with_capacity(native_out.len());
    for owned in native_out {
        out.push(crate::dlpack::owned_to_capsule_owned(py, owned)?);
    }
    Ok(out)
}
use crate::fusion::Step;

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
    fn take_tuple_elem(&mut self, index: usize) -> Option<OwnedTensor> {
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

    fn is_tuple(&self) -> bool {
        matches!(self, Slot::Tuple(_))
    }

    fn tuple_len(&self) -> usize {
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
        "add".into(), "sub".into(), "mul".into(), "div".into(), "relu".into(),
        // Phase 2: math/comparison
        "eq".into(), "ne".into(), "lt".into(), "le".into(), "gt".into(), "ge".into(),
        "abs".into(), "neg".into(), "sign".into(), "sqrt".into(), "rsqrt".into(),
        "exp".into(), "log".into(), "reciprocal".into(), "ceil".into(), "floor".into(),
        "clamp".into(), "clamp_min".into(), "clamp_max".into(), "pow".into(),
        "sin".into(), "cos".into(), "round".into(),
        // Phase 2: logical + dtype cast
        "logical_and".into(), "logical_or".into(), "logical_not".into(), "to_dtype".into(),
        // Phase 2: activations
        "sigmoid".into(), "tanh".into(), "gelu".into(), "silu".into(),
        "leaky_relu".into(), "elu".into(), "selu".into(), "softplus".into(),
        "hardswish".into(), "mish".into(), "softmax".into(), "log_softmax".into(),
        "threshold_backward".into(),
        // Phase 2: reductions
        "sum".into(), "mean".into(), "max_reduce".into(), "min_reduce".into(),
        "argmax".into(), "argmin".into(), "std".into(), "var".into(),
        "cumsum".into(), "prod".into(), "norm".into(), "linalg_vector_norm".into(),
        // Phase 2: linalg
        "matmul".into(), "bmm".into(), "linear".into(), "dot".into(), "addmm".into(),
        // Phase 2: shape ops
        "t".into(), "transpose".into(), "index_select".into(), "gather".into(),
        // Phase 2: norm
        "layer_norm".into(), "batch_norm".into(), "group_norm".into(), "rms_norm".into(),
        // Phase 2: shape ops
        "cat".into(), "stack".into(), "reshape".into(), "permute".into(),
        "expand".into(), "where".into(), "masked_fill".into(), "flip".into(),
        "narrow".into(), "select".into(), "contiguous".into(), "chunk_narrow".into(),
        "squeeze".into(), "unsqueeze".into(), "unflatten".into(), "dropout".into(),
        // Phase 10: tensor creation
        "full".into(), "zeros".into(), "ones".into(), "arange".into(), "linspace".into(),
        // Phase 3: convolution & pooling & upsampling
        "conv1d".into(), "conv2d".into(),
        "conv_transpose1d".into(), "conv_transpose2d".into(),
        "max_pool2d".into(), "avg_pool2d".into(),
        "adaptive_avg_pool2d".into(), "adaptive_max_pool2d".into(),
        "max_pool1d".into(), "avg_pool1d".into(),
        "upsample_nearest2d".into(), "upsample_bilinear2d".into(), "interpolate".into(),
        "flatten".into(),
        // Phase 4: transformer stack
        "scalar_tensor".into(),
        "embedding".into(), "scaled_dot_product_attention".into(), "rope".into(),
        "nll_loss_forward".into(), "mse_loss".into(), "smooth_l1_loss".into(),
        "binary_cross_entropy".into(),
        // Phase 7: extended ops
        "scatter".into(), "scatter_add".into(),
        "topk".into(), "sort".into(), "argsort".into(),
        "unbind".into(), "chunk".into(), "getitem".into(),
        "repeat_interleave".into(), "repeat".into(),
        "einsum".into(), "prelu".into(), "nonzero".into(), "clamp_tensor".into(),
        // v0.2 extra 50 ops batch 1
        "atan".into(), "asin".into(), "acos".into(), "sinh".into(), "cosh".into(),
        "asinh".into(), "acosh".into(), "atanh".into(), "erf".into(), "erfc".into(),
        "expm1".into(), "log1p".into(), "log2".into(), "log10".into(),
        "atan2".into(), "hypot".into(), "fmod".into(), "remainder".into(), "copysign".into(), "lerp".into(),
        "bitwise_and".into(), "bitwise_or".into(), "bitwise_xor".into(), "bitwise_not".into(),
        "isfinite".into(), "isinf".into(), "isnan".into(),
        "all".into(), "any".into(), "amax".into(), "amin".into(), "count_nonzero".into(), "nansum".into(), "nanmean".into(),
        "tile".into(), "roll".into(), "pixel_shuffle".into(), "instance_norm".into(),
        "cross_entropy".into(), "huber_loss".into(),
        "hardtanh".into(), "hardsigmoid".into(), "glu".into(),
        "trunc".into(), "frac".into(), "square".into(), "exp2".into(), "ldexp".into(),
        "bucketize".into(), "histc".into(),
        // Extra ops batch 2 (49 ops)
        "embedding_bag".into(), "unfold".into(), "fold".into(), "grid_sample".into(), "affine_grid".into(),
        "pixel_unshuffle".into(), "channel_shuffle".into(), "cummax".into(), "cummin".into(), "logcumsumexp".into(),
        "scatter_reduce".into(), "index_put".into(), "index_add".into(), "masked_scatter".into(), "take".into(), "put".into(), "masked_select".into(), "index_fill".into(),
        "bincount".into(), "unique".into(), "kthvalue".into(), "median".into(), "quantile".into(), "histogram".into(), "searchsorted".into(), "meshgrid".into(), "cdist".into(), "pdist".into(), "renorm".into(),
        "bernoulli".into(), "multinomial".into(), "logspace".into(), "eye".into(), "diag".into(), "diagonal".into(), "trace".into(), "matrix_exp".into(), "slogdet".into(), "det".into(), "lstsq".into(), "pinverse".into(),
        "normal".into(), "uniform".into(), "triu".into(), "tril".into(), "hann_window".into(), "bartlett_window".into(), "blackman_window".into(), "stft".into(),
        // Extra ops batch 3 (149 ops) -> Total exactly 375 ops!
        "nextafter".into(), "heaviside".into(), "nan_to_num".into(), "logaddexp".into(), "logaddexp2".into(),
        "sinc".into(), "i0".into(), "i1".into(), "i0e".into(), "i1e".into(),
        "bessel_j0".into(), "bessel_j1".into(), "bessel_y0".into(), "bessel_y1".into(),
        "digamma".into(), "lgamma".into(), "polygamma".into(), "mvlgamma".into(),
        "erfinv".into(), "erfcinv".into(), "ndtri".into(), "ndtr".into(), "log_ndtr".into(),
        "logit".into(), "expit".into(), "rad2deg".into(), "deg2rad".into(), "gcd".into(), "lcm".into(),
        "fmax".into(), "fmin".into(), "maximum".into(), "minimum".into(), "signbit".into(),
        "addcdiv".into(), "addcmul".into(), "addr".into(), "outer".into(),
        "mv".into(), "vdot".into(), "baddbmm".into(), "addbmm".into(), "addmv".into(),
        "kron".into(), "inner".into(), "trapz".into(), "trapezoid".into(), "cumulative_trapezoid".into(),
        "celu".into(), "hardshrink".into(), "softshrink".into(), "tanhshrink".into(), "threshold".into(),
        "logsigmoid".into(), "rrelu".into(), "kl_div".into(), "poisson_nll_loss".into(), "margin_ranking_loss".into(),
        "hinge_embedding_loss".into(), "multilabel_margin_loss".into(), "soft_margin_loss".into(), "multilabel_soft_margin_loss".into(), "cosine_embedding_loss".into(),
        "triplet_margin_loss".into(), "ctc_loss".into(), "hamming_window".into(), "kaiser_window".into(), "gaussian_window".into(),
        "exponential_window".into(), "triangular_window".into(), "cross".into(), "linalg_norm".into(), "frobenius_norm".into(),
        "nuclear_norm".into(), "matrix_rank".into(), "matrix_power".into(), "cholesky".into(), "cholesky_inverse".into(),
        "cholesky_solve".into(), "qr".into(), "svd".into(), "svdvals".into(), "eig".into(),
        "eigh".into(), "eigvals".into(), "eigvalsh".into(), "lu".into(), "triangular_solve".into(),
        "select_scatter".into(), "slice_scatter".into(), "diagonal_scatter".into(), "index_copy".into(), "narrow_copy".into(),
        "movedim".into(), "moveaxis".into(), "swapdims".into(), "swapaxes".into(), "column_stack".into(),
        "row_stack".into(), "dstack".into(), "hstack".into(), "vstack".into(), "atleast_1d".into(),
        "atleast_2d".into(), "atleast_3d".into(), "block_diag".into(), "cartesian_prod".into(), "combinations".into(),
        "pad".into(), "constant_pad_nd".into(), "reflection_pad1d".into(), "reflection_pad2d".into(), "replication_pad1d".into(),
        "replication_pad2d".into(), "zero_pad2d".into(), "conv3d".into(), "conv_transpose3d".into(), "max_pool3d".into(),
        "avg_pool3d".into(), "adaptive_max_pool3d".into(), "adaptive_avg_pool3d".into(), "fractional_max_pool2d".into(), "fractional_max_pool3d".into(),
        "lp_pool1d".into(), "lp_pool2d".into(), "max_unpool1d".into(), "max_unpool2d".into(), "max_unpool3d".into(),
        "rand".into(), "randn".into(), "randint".into(), "randperm".into(), "empty".into(),
        "zeros_like".into(), "ones_like".into(), "full_like".into(), "rnn_tanh_cell".into(), "rnn_relu_cell".into(),
        "gru_cell".into(), "lstm_cell".into(), "multi_head_attention_forward".into(), "lu_solve".into(), "lu_unpack".into(),
        "linalg_solve".into(), "linalg_inv".into(), "linalg_pinv".into(), "linalg_det".into(), "linalg_slogdet".into(),
        "linalg_cond".into(),
        // Advanced LLM & FlashAttention
        "flash_attention".into(), "fused_swiglu".into(), "fused_geglu".into(), "fused_rmsnorm_residual".into(),
        // Universal Low-Bit Quantization & GEMM
        "quantize_per_tensor".into(), "dequantize_per_tensor".into(), "quantize_per_channel".into(), "dequantize_per_channel".into(),
        "int8_gemm".into(), "nf4_dequantize".into(), "int4_unpack_dequantize".into(),
        // Universal FFT & Complex Suite
        "fft".into(), "ifft".into(), "rfft".into(), "irfft".into(), "fft2".into(), "ifft2".into(), "fftn".into(), "ifftn".into(),
        "fftshift".into(), "ifftshift".into(), "complex".into(), "real".into(), "imag".into(), "angle".into(), "polar".into(), "conj".into(),
    ]
}

/// View a slot as a borrowed tensor (zero-copy for inputs).
pub(crate) fn slot_view<'p>(
    slots: &[Slot],
    capsules: &'p [CapsuleRef],
    index: usize,
) -> PyResult<BorrowedTensor> {
    match slots.get(index) {
        Some(Slot::Input(i)) => unsafe { BorrowedTensor::from_managed(capsules[*i].0) },
        Some(Slot::Owned(t)) => Ok(BorrowedTensor::from_owned(t)),
        Some(Slot::View { data, shape, strides, dtype }) => Ok(BorrowedTensor {
            data: *data,
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: *dtype,
        }),
        Some(Slot::Tuple(_)) => Err(unsupported(&format!("slot {index} is a tuple; use slot_view_tuple"))),
        None => Err(unsupported(&format!("argument references missing slot {index}"))),
    }
}

/// View an element of a tuple slot.
pub(crate) fn slot_view_tuple<'p>(
    slots: &[Slot],
    _capsules: &'p [CapsuleRef],
    tuple_slot: usize,
    elem_index: usize,
) -> PyResult<BorrowedTensor> {
    match slots.get(tuple_slot) {
        Some(Slot::Tuple(elems)) => elems
            .get(elem_index)
            .map(|t| BorrowedTensor::from_owned(t))
            .ok_or_else(|| unsupported(&format!(
                "tuple slot {tuple_slot}: index {elem_index} out of range (len={})",
                elems.len()
            ))),
        Some(Slot::Owned(_)) => Err(unsupported(&format!(
            "slot {tuple_slot} is not a tuple; cannot index"
        ))),
        Some(Slot::View { .. }) => Err(unsupported(&format!(
            "slot {tuple_slot} is a view; cannot index into tuple"
        ))),
        Some(Slot::Input(_)) => Err(unsupported(&format!(
            "slot {tuple_slot} is an input; cannot index into tuple"
        ))),
        None => Err(unsupported(&format!("tuple slot {tuple_slot} does not exist"))),
    }
}

/// Resolve a node argument to a slot index.
fn arg_index(node: &Node, position: usize) -> PyResult<usize> {
    let arg = node
        .args
        .get(position)
        .ok_or_else(|| unsupported(&format!("node '{}' missing argument #{position}", node.target)))?;
    arg.index
        .ok_or_else(|| unsupported(&format!("node '{}' has an unindexed argument", node.target)))
}

/// Get a scalar f64 from kwargs or a default.
fn kw_f64(node: &Node, key: &str, default: f64) -> f64 {
    node.kwargs.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Read a scalar from kwargs, decoding the "inf"/"-inf"/"nan" string
/// tokens that the parser emits for non-finite constants (serde_json rejects
/// raw Infinity/NaN literals).
fn kw_f64_allow_inf(node: &Node, key: &str, default: f64) -> f64 {
    match node.kwargs.get(key) {
        Some(v) => {
            if let Some(x) = v.as_f64() {
                x
            } else if let Some(s) = v.as_str() {
                match s {
                    "inf" => f64::INFINITY,
                    "-inf" => f64::NEG_INFINITY,
                    "nan" => f64::NAN,
                    _ => default,
                }
            } else {
                default
            }
        }
        None => default,
    }
}

fn kw_bool(node: &Node, key: &str, default: bool) -> bool {
    node.kwargs.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn kw_isize(node: &Node, key: &str, default: isize) -> isize {
    node.kwargs.get(key).and_then(|v| v.as_i64()).map(|v| v as isize).unwrap_or(default)
}

/// Read an optional reduction dim from kwargs.  Returns None when absent.
/// Supports single dim (scalar) or multi-dim (list of ints).
/// For multi-dim, returns the list via kw_isize_vec("dim").
fn kw_opt_dim(node: &Node) -> PyResult<Option<isize>> {
    match node.kwargs.get("dim") {
        None => Ok(None),
        Some(v) => match v.as_i64() {
            Some(x) => Ok(Some(x as isize)),
            None => match v.as_array() {
                Some(arr) if arr.len() == 1 => {
                    Ok(arr[0].as_i64().map(|x| x as isize))
                }
                Some(arr) => {
                    // Multi-dim: for now, if it's [dim], treat as scalar;
                    // otherwise signal the caller to do iterative reduction.
                    let dims: Vec<isize> = arr.iter()
                        .filter_map(|v| v.as_i64().map(|x| x as isize))
                        .collect();
                    if dims.is_empty() {
                        return Ok(None);
                    }
                    if dims.len() > 1 {
                        return Err(unsupported(
                            "multi-dim reductions (dim as a list) are not supported by the native engine",
                        ));
                    }
                    Ok(Some(dims[0]))
                }
                None => Ok(None),
            }
        },
    }
}

fn kw_i64_vec(node: &Node, key: &str) -> Vec<i64> {
    node.kwargs.get(key).and_then(|v| {
        v.as_array().map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
    }).unwrap_or_default()
}

fn kw_isize_vec(node: &Node, key: &str) -> Vec<isize> {
    node.kwargs.get(key).and_then(|v| {
        v.as_array().map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as isize)).collect())
    }).unwrap_or_default()
}

fn kw_usize(node: &Node, key: &str, default: usize) -> usize {
    node.kwargs.get(key).and_then(|v| v.as_i64()).map(|v| v as usize).unwrap_or(default)
}

fn kw_i64(node: &Node, key: &str, default: i64) -> i64 {
    node.kwargs.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
}

fn kw_str<'a>(node: &'a Node, key: &str, default: &'a str) -> &'a str {
    node.kwargs.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

/// Execute a node by dispatching to the appropriate kernel.
fn dispatch_node(node: &Node, slots: &mut Vec<Slot>, capsules: &[CapsuleRef]) -> PyResult<()> {
    let target = node.target.as_str();

    match target {
        // Phase 1: binary elementwise
        "add" | "sub" | "mul" | "div" => {
            let op = ops::BinaryOp::from_target(target).expect("binary op target already validated");
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(ops::binary(op, &a, &b)?));
        }
        "relu" => {
            let ai = arg_index(node, 0)?;
            let a = slot_view(slots, capsules, ai)?;
            slots.push(Slot::Owned(ops::relu(&a)?));
        }

        // Phase 2: logical ops
        "logical_and" | "logical_or" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            let out = if target == "logical_and" {
                math_ops::logical_and(&a, &b)?
            } else {
                math_ops::logical_or(&a, &b)?
            };
            slots.push(Slot::Owned(out));
        }
        "logical_not" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(math_ops::logical_not(&a)?));
        }
        "to_dtype" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            // dtype comes from the "dtype" kwarg (aten._to_copy) or from a
            // positional const arg (x.to(torch.float64) -> args[1].value).
            let dtype_str: Option<&str> = node.kwargs.get("dtype").and_then(|v| v.as_str());
            let dtype_str = dtype_str.or_else(|| {
                node.args.get(1).and_then(|arg| arg.value.as_ref()).and_then(|v| v.as_str())
            });
            let dtype_str = dtype_str.ok_or_else(|| unsupported("to_dtype: missing dtype"))?;
            let target = crate::dlpack::dtype_from_spec(dtype_str)
                .ok_or_else(|| unsupported(&format!("to_dtype: unknown dtype '{dtype_str}'")))?;
            slots.push(Slot::Owned(math_ops::to_dtype(&a, target)?));
        }

        // Phase 2: comparison ops
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(math_ops::comparison(target, &a, &b)?));
        }

        // Phase 2: unary math
        "abs" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::abs(&a)?)); }
        "neg" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::neg(&a)?)); }
        "sign" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::sign(&a)?)); }
        "sqrt" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::sqrt(&a)?)); }
        "rsqrt" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::rsqrt(&a)?)); }
        "exp" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::exp(&a)?)); }
        "log" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::log(&a)?)); }
        "reciprocal" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::reciprocal(&a)?)); }
        "ceil" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::ceil(&a)?)); }
        "floor" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::floor(&a)?)); }
        "clamp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let min = kw_f64(node, "min", f64::NEG_INFINITY);
            let max = kw_f64(node, "max", f64::INFINITY);
            slots.push(Slot::Owned(math_ops::clamp(&a, min, max)?));
        }
        "clamp_min" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let min = kw_f64(node, "min", 0.0);
            slots.push(Slot::Owned(math_ops::clamp_min(&a, min)?));
        }
        "clamp_max" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let max = kw_f64(node, "max", 0.0);
            slots.push(Slot::Owned(math_ops::clamp_max(&a, max)?));
        }
        "sin" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::sin(&a)?)); }
        "cos" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::cos(&a)?)); }
        "round" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(math_ops::round(&a)?)); }
        "pow" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let exp = kw_f64(node, "exp", 2.0);
            slots.push(Slot::Owned(math_ops::pow_scalar(&a, exp)?));
        }

        // Phase 2: activations
        "sigmoid" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::sigmoid(&a)?)); }
        "tanh" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::tanh_act(&a)?)); }
        "gelu" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::gelu(&a)?)); }
        "silu" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::silu(&a)?)); }
        "leaky_relu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ns = kw_f64(node, "negative_slope", 0.01);
            slots.push(Slot::Owned(activations::leaky_relu(&a, ns)?));
        }
        "elu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(activations::elu(&a, alpha)?));
        }
        "selu" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::selu(&a)?)); }
        "softplus" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::softplus(&a)?)); }
        "hardswish" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::hardswish(&a)?)); }
        "mish" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(activations::mish(&a)?)); }
        "softmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(activations::softmax(&a, dim)?));
        }
        "log_softmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(activations::log_softmax(&a, dim)?));
        }
        "threshold_backward" => {
            let grad = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let threshold = kw_f64(node, "threshold", 0.0);
            slots.push(Slot::Owned(activations::threshold_backward(&grad, &x, threshold)?));
        }

        // Phase 2: reductions
        "sum" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::sum(&a, dim, keepdim)?));
        }
        "mean" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::mean(&a, dim, keepdim)?));
        }
        "max_reduce" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let (val, idx) = reductions::max_reduce(&a, dim, keepdim)?;
            slots.push(Slot::Tuple(vec![val, idx]));
        }
        "min_reduce" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let (val, idx) = reductions::min_reduce(&a, dim, keepdim)?;
            slots.push(Slot::Tuple(vec![val, idx]));
        }
        "argmax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::argmax(&a, dim, keepdim)?));
        }
        "argmin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::argmin(&a, dim, keepdim)?));
        }
        "std" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let unbiased = kw_bool(node, "unbiased", true);
            slots.push(Slot::Owned(reductions::std_dev(&a, dim, keepdim, unbiased)?));
        }
        "var" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            let unbiased = kw_bool(node, "unbiased", true);
            slots.push(Slot::Owned(reductions::var(&a, dim, keepdim, unbiased)?));
        }
        "cumsum" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(reductions::cumsum(&a, dim)?));
        }
        "prod" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::prod(&a, dim, keepdim)?));
        }
        "norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64_allow_inf(node, "p", 2.0);
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::p_norm(&a, p, dim, keepdim)?));
        }
        "linalg_vector_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64_allow_inf(node, "ord", 2.0);
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(reductions::p_norm(&a, p, dim, keepdim)?));
        }

        // Phase 2: linalg
        "matmul" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(linalg::matmul(&a, &b)?));
        }
        "bmm" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(linalg::bmm(&a, &b)?));
        }
        "linear" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = if node.args.len() > 2 {
                Some(slot_view(slots, capsules, arg_index(node, 2)?)?)
            } else {
                None
            };
            slots.push(Slot::Owned(linalg::linear(&input, &weight, bias.as_ref(), None)?));
        }
        "dot" => {
            let ai = arg_index(node, 0)?;
            let bi = arg_index(node, 1)?;
            let a = slot_view(slots, capsules, ai)?;
            let b = slot_view(slots, capsules, bi)?;
            slots.push(Slot::Owned(linalg::dot(&a, &b)?));
        }
        "addmm" => {
            // aten.addmm(bias, mat1, mat2) — mat2 is NOT transposed.
            let bias = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mat1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let mat2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(linalg::addmm(&bias, &mat1, &mat2, None)?));
        }

        // Phase 2: norm
        "layer_norm" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(norm::layer_norm(&input, &weight, &bias, eps)?));
        }
        "batch_norm" => {
            // torch signature: batch_norm(x, running_mean, running_var, weight, bias, ...)
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let running_mean = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let running_var = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let bias = slot_view(slots, capsules, arg_index(node, 4)?)?;
            let eps = kw_f64(node, "eps", 1e-5);
            let training = kw_bool(node, "training", false);
            slots.push(Slot::Owned(norm::batch_norm(&input, &weight, &bias, &running_mean, &running_var, eps, training)?));
        }
        "group_norm" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let num_groups = kw_usize(node, "num_groups", 32);
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(norm::group_norm(&input, &weight, &bias, num_groups, eps)?));
        }
        "rms_norm" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let eps = kw_f64(node, "eps", 1e-6);
            slots.push(Slot::Owned(norm::rms_norm(&input, &weight, eps)?));
        }

        // Phase 2: shape ops
        "cat" => {
            // Two calling conventions:
            // 1. args = [{kind:"slot", value:[i,j,...]}, {dim kwarg}] — list-in-first-arg
            // 2. args = [{kind:"slot", index:i}, {kind:"slot", index:j}, ...] — flat args
            let indices_arg = node.args.get(0).ok_or_else(|| unsupported("cat: missing tensors argument"))?;
            let tensor_indices: Vec<usize> = if let Some(arr) = indices_arg.value.as_ref().and_then(|v| v.as_array()) {
                // Convention 1: first arg holds a JSON array of slot indices
                arr.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect()
            } else {
                // Convention 2: all positional args are individual slot refs
                node.args.iter().filter_map(|a| a.index).collect()
            };
            if tensor_indices.is_empty() {
                return Err(unsupported("cat: no tensor slot indices found"));
            }
            let tensors: Vec<BorrowedTensor> = tensor_indices.iter()
                .map(|&i| slot_view(slots, capsules, i))
                .collect::<PyResult<_>>()?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::cat(&tensors, dim)?));
        }
        "stack" => {
            let indices_arg = node.args.get(0).ok_or_else(|| unsupported("stack: missing tensors argument"))?;
            let tensor_indices: Vec<usize> = if let Some(arr) = indices_arg.value.as_ref().and_then(|v| v.as_array()) {
                arr.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect()
            } else {
                node.args.iter().filter_map(|a| a.index).collect()
            };
            if tensor_indices.is_empty() {
                return Err(unsupported("stack: no tensor slot indices found"));
            }
            let tensors: Vec<BorrowedTensor> = tensor_indices.iter()
                .map(|&i| slot_view(slots, capsules, i))
                .collect::<PyResult<_>>()?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::stack(&tensors, dim)?));
        }
        "reshape" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            // Shape comes from kwargs["shape"] (set by the parser's seq/const
            // promotion).  A missing shape means the parser failed to convey
            // it — reject loudly rather than silently returning a copy.
            let shape = kw_i64_vec(node, "shape");
            if shape.is_empty() {
                return Err(unsupported("reshape: shape not conveyed via kwargs"));
            }
            let resolved = shape_ops::resolve_shape(&a.shape, &shape)?;
            if a.strides == contiguous_strides(&a.shape) {
                let new_strides = contiguous_strides(&resolved);
                slots.push(Slot::View {
                    data: a.data,
                    shape: resolved,
                    strides: new_strides,
                    dtype: a.dtype,
                });
            } else {
                slots.push(Slot::Owned(shape_ops::reshape(&a, &shape)?));
            }
        }
        "permute" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dims = kw_isize_vec(node, "dims");
            if dims.is_empty() {
                return Err(unsupported("permute: no dims provided"));
            }
            let (new_shape, new_strides) = shape_ops::permute_view(&a, &dims)?;
            slots.push(Slot::View {
                data: a.data,
                shape: new_shape,
                strides: new_strides,
                dtype: a.dtype,
            });
        }
        "transpose" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d0 = kw_isize(node, "d0", 0);
            let d1 = kw_isize(node, "d1", 1);
            let (new_shape, new_strides) = shape_ops::transpose_view(&a, d0, d1)?;
            slots.push(Slot::View {
                data: a.data,
                shape: new_shape,
                strides: new_strides,
                dtype: a.dtype,
            });
        }
        "index_select" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let index = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::index_select(&a, dim, &index)?));
        }
        "gather" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let index = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(shape_ops::gather(&a, dim, &index)?));
        }
        "chunk" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let num_chunks = kw_usize(node, "chunks", 1);
            let dim = kw_isize(node, "dim", 0);
            let ndim = a.shape.len() as isize;
            let normalized_dim = if dim < 0 { dim + ndim } else { dim };
            let dim_size = a.shape[normalized_dim as usize] as usize;
            let chunk_size = dim_size / num_chunks;
            let mut parts = Vec::with_capacity(num_chunks);
            for i in 0..num_chunks {
                let start = i * chunk_size;
                let length = if i == num_chunks - 1 {
                    dim_size - start
                } else {
                    chunk_size
                };
                parts.push(shape_ops::narrow(&a, dim, start, length)?);
            }
            slots.push(Slot::Tuple(parts));
        }
        "unbind" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let ndim = a.shape.len() as isize;
            let normalized_dim = if dim < 0 { dim + ndim } else { dim };
            let dim_size = a.shape[normalized_dim as usize] as usize;
            let mut parts = Vec::with_capacity(dim_size);
            for i in 0..dim_size {
                parts.push(shape_ops::select(&a, dim, i)?);
            }
            slots.push(Slot::Tuple(parts));
        }
        "t" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            if a.shape.len() > 2 {
                return Err(unsupported("t: input must be <= 2D"));
            }
            if a.shape.len() <= 1 {
                slots.push(Slot::View {
                    data: a.data,
                    shape: a.shape.clone(),
                    strides: a.strides.clone(),
                    dtype: a.dtype,
                });
            } else {
                let (new_shape, new_strides) = shape_ops::transpose_view(&a, 0, 1)?;
                slots.push(Slot::View {
                    data: a.data,
                    shape: new_shape,
                    strides: new_strides,
                    dtype: a.dtype,
                });
            }
        }
        "expand" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let shape = kw_i64_vec(node, "shape");
            if shape.is_empty() {
                return Err(unsupported("expand: no shape provided"));
            }
            slots.push(Slot::Owned(shape_ops::expand(&a, &shape)?));
        }
        "where" => {
            let cond = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let y = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(shape_ops::where_op(&cond, &x, &y)?));
        }
        "masked_fill" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mask = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let value = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(shape_ops::masked_fill(&a, &mask, value)?));
        }
        "flip" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dims: Vec<isize> = kw_isize_vec(node, "dims");
            slots.push(Slot::Owned(shape_ops::flip(&a, &dims)?));
        }
        "narrow" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let start = kw_usize(node, "start", 0);
            let length = kw_usize(node, "length", 0);
            slots.push(Slot::Owned(shape_ops::narrow(&a, dim, start, length)?));
        }
        "select" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let index = kw_usize(node, "index", 0);
            slots.push(Slot::Owned(shape_ops::select(&a, dim, index)?));
        }
        "getitem" => {
            // getitem(tuple_slot, index) -> tuple_slot[index]
            let slot_idx = arg_index(node, 0)?;
            let elem = kw_usize(node, "index", 0);
            let len = slots[slot_idx].tuple_len();
            if len == 0 {
                return Err(unsupported(&format!("getitem: slot {slot_idx} is not a tuple")));
            }
            if elem >= len {
                return Err(unsupported(&format!("getitem: index {elem} out of range for tuple of len {len}")));
            }
            if let Some(taken) = slots[slot_idx].take_tuple_elem(elem) {
                slots.push(Slot::Owned(taken));
            }
        }
        "chunk_narrow" => {
            // chunk decomposition: getitem(chunk(x, N), i) -> narrow(x, dim, start, length)
            // Compute start and length at runtime from the tensor's actual shape.
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let chunk_index = kw_usize(node, "chunk_index", 0);
            let num_chunks = kw_usize(node, "num_chunks", 1);
            let ndim = a.shape.len() as isize;
            let normalized_dim = if dim < 0 { dim + ndim } else { dim };
            let dim_size = a.shape[normalized_dim as usize] as usize;
            let chunk_size = dim_size / num_chunks;
            let start = chunk_index * chunk_size;
            let length = if chunk_index == num_chunks - 1 {
                dim_size - start  // last chunk gets remainder
            } else {
                chunk_size
            };
            slots.push(Slot::Owned(shape_ops::narrow(&a, dim, start, length)?));
        }
        // contiguous() is a no-op for already-contiguous inputs (the interpreter
        // ensures inputs are contiguous before passing to the Rust engine).
        "contiguous" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            if a.strides == contiguous_strides(&a.shape) {
                slots.push(Slot::View {
                    data: a.data,
                    shape: a.shape.clone(),
                    strides: a.strides.clone(),
                    dtype: a.dtype,
                });
            } else {
                slots.push(Slot::Owned(shape_ops::to_contiguous(&a)?));
            }
        }
        "squeeze" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let (new_shape, new_strides) = shape_ops::squeeze_view(&a, dim)?;
            slots.push(Slot::View {
                data: a.data,
                shape: new_shape,
                strides: new_strides,
                dtype: a.dtype,
            });
        }
        "unsqueeze" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let (new_shape, new_strides) = shape_ops::unsqueeze_view(&a, dim)?;
            slots.push(Slot::View {
                data: a.data,
                shape: new_shape,
                strides: new_strides,
                dtype: a.dtype,
            });
        }
        "unflatten" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let sizes = kw_i64_vec(node, "sizes");
            slots.push(Slot::Owned(shape_ops::unflatten(&a, dim, &sizes)?));
        }
        // dropout is a no-op during inference (training=false)
        "dropout" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::View {
                data: a.data,
                shape: a.shape.clone(),
                strides: a.strides.clone(),
                dtype: a.dtype,
            });
        }

        // Phase 3: flatten
        "flatten" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let start_dim = kw_isize(node, "start_dim", 0);
            let end_dim = kw_isize(node, "end_dim", -1);
            slots.push(Slot::Owned(shape_ops::flatten(&a, start_dim, end_dim)?));
        }

        // Phase 4: scalar constants used in mask graphs (aten.scalar_tensor)
        "scalar_tensor" => {
            let v = kw_f64_allow_inf(node, "value", 0.0);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                _ => DType::F32,
            };
            let mut out = OwnedTensor::new(dtype, vec![]);
            match dtype {
                DType::F32 => {
                    let d = unsafe {
                        std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, out.elem_count())
                    };
                    d[0] = v as f32;
                }
                _ => {
                    let d = unsafe {
                        std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f64, out.elem_count())
                    };
                    d[0] = v;
                }
            }
            slots.push(Slot::Owned(out));
        }

        // Phase 10: tensor creation ops
        "full" => {
            let shape = kw_i64_vec(node, "shape");
            let value = kw_f64(node, "value", 0.0);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                Some("i32") => DType::I32,
                Some("bool") => DType::Bool,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::full(&shape, value, dtype)?));
        }
        "zeros" => {
            let shape = kw_i64_vec(node, "shape");
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::zeros(&shape, dtype)?));
        }
        "ones" => {
            let shape = kw_i64_vec(node, "shape");
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::ones(&shape, dtype)?));
        }
        "arange" => {
            let start = kw_f64(node, "start", 0.0);
            let end = kw_f64(node, "end", 0.0);
            let step = kw_f64(node, "step", 1.0);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                Some("i64") => DType::I64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::arange(start, end, step, dtype)?));
        }
        "linspace" => {
            let start = kw_f64(node, "start", 0.0);
            let end = kw_f64(node, "end", 0.0);
            let steps = kw_usize(node, "steps", 100);
            let dtype = match node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                Some("f64") => DType::F64,
                _ => DType::F32,
            };
            slots.push(Slot::Owned(shape_ops::linspace(start, end, steps, dtype)?));
        }

        // Phase 4: embedding (weight, indices) — int64/int32 indices
        "embedding" => {
            let weight = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let indices = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(embedding::embedding(&weight, &indices)?));
        }

        // Phase 4: attention
        "scaled_dot_product_attention" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let mask = if node.args.len() > 3 {
                Some(slot_view(slots, capsules, arg_index(node, 3)?)?)
            } else {
                None
            };
            let is_causal = kw_bool(node, "is_causal", false);
            slots.push(Slot::Owned(attention::scaled_dot_product_attention(
                &q, &k, &v, mask.as_ref(), is_causal,
            )?));
        }
        "rope" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let cos = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let sin = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(attention::rope(&x, &cos, &sin)?));
        }

        // Phase 4: losses
        "nll_loss_forward" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let target = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let reduction = kw_isize(node, "reduction", 1);
            let ignore_index = kw_isize(node, "ignore_index", -100);
            slots.push(Slot::Owned(losses::nll_loss_forward(
                &input, &target, reduction as i64, ignore_index as i64,
            )?));
        }
        "mse_loss" | "smooth_l1_loss" | "binary_cross_entropy" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let reduction = kw_isize(node, "reduction", 1);
            let beta = kw_f64(node, "beta", 1.0);
            match target {
                "mse_loss" => {
                    slots.push(Slot::Owned(losses::mse_loss(&a, &b, reduction as i64)?));
                }
                "smooth_l1_loss" => {
                    slots.push(Slot::Owned(losses::smooth_l1_loss(&a, &b, reduction as i64, beta)?));
                }
                _ => {
                    slots.push(Slot::Owned(losses::binary_cross_entropy(&a, &b, reduction as i64)?));
                }
            }
        }

        // Phase 3: convolution
        "conv1d" | "conv2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = if node.args.len() > 2 {
                Some(slot_view(slots, capsules, arg_index(node, 2)?)?)
            } else {
                None
            };
            let groups = kw_usize(node, "groups", 1) as i64;
            if target == "conv2d" {
                slots.push(Slot::Owned(convolution::conv2d(
                    &input, &weight, bias.as_ref(),
                    node.kwargs.get("stride"), node.kwargs.get("padding"),
                    node.kwargs.get("dilation"), groups,
                )?));
            } else {
                slots.push(Slot::Owned(convolution::conv1d(
                    &input, &weight, bias.as_ref(),
                    node.kwargs.get("stride"), node.kwargs.get("padding"),
                    node.kwargs.get("dilation"), groups,
                )?));
            }
        }
        "conv_transpose1d" | "conv_transpose2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let bias = if node.args.len() > 2 {
                Some(slot_view(slots, capsules, arg_index(node, 2)?)?)
            } else {
                None
            };
            let groups = kw_usize(node, "groups", 1) as i64;
            let (stride, padding, output_padding, dilation) = (
                node.kwargs.get("stride"), node.kwargs.get("padding"),
                node.kwargs.get("output_padding"), node.kwargs.get("dilation"),
            );
            slots.push(Slot::Owned(if target == "conv_transpose1d" {
                convolution::conv_transpose1d(
                    &input, &weight, bias.as_ref(),
                    stride, padding, output_padding, dilation, groups,
                )?
            } else {
                convolution::conv_transpose2d(
                    &input, &weight, bias.as_ref(),
                    stride, padding, output_padding, dilation, groups,
                )?
            }));
        }

        // Phase 3: pooling
        "max_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            slots.push(Slot::Owned(pooling::max_pool2d(
                &input,
                node.kwargs.get("kernel"), node.kwargs.get("stride"),
                node.kwargs.get("padding"), node.kwargs.get("dilation"), ceil_mode,
            )?));
        }
        "avg_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            let count_include_pad = kw_bool(node, "count_include_pad", true);
            slots.push(Slot::Owned(pooling::avg_pool2d(
                &input,
                node.kwargs.get("kernel"), node.kwargs.get("stride"),
                node.kwargs.get("padding"), ceil_mode, count_include_pad,
            )?));
        }
        "adaptive_avg_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(pooling::adaptive_avg_pool2d(
                &input, node.kwargs.get("output_size"),
            )?));
        }
        "adaptive_max_pool2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(pooling::adaptive_max_pool2d(
                &input, node.kwargs.get("output_size"),
            )?));
        }
        "max_pool1d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            slots.push(Slot::Owned(pooling::max_pool1d(
                &input,
                node.kwargs.get("kernel"), node.kwargs.get("stride"),
                node.kwargs.get("padding"), ceil_mode,
            )?));
        }
        "avg_pool1d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ceil_mode = kw_bool(node, "ceil_mode", false);
            let count_include_pad = kw_bool(node, "count_include_pad", true);
            slots.push(Slot::Owned(pooling::avg_pool1d(
                &input,
                node.kwargs.get("kernel"), node.kwargs.get("stride"),
                node.kwargs.get("padding"), ceil_mode, count_include_pad,
            )?));
        }

        // Phase 3: upsampling
        "upsample_nearest2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(upsample::upsample_nearest2d(
                &input, node.kwargs.get("size"),
            )?));
        }
        "upsample_bilinear2d" => {
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(upsample::upsample_bilinear2d(
                &input, node.kwargs.get("size"),
            )?));
        }
        "interpolate" => {
            // F.interpolate(x, size=..., mode=...) — route by mode. Only the
            // 4-D (N,C,H,W) case with an explicit size is supported here;
            // scale_factor and other modes fall back to eager (REQ-002).
            let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mode = node.kwargs.get("mode").and_then(|v| v.as_str()).unwrap_or("nearest");
            match mode {
                "nearest" | "nearest-exact" => slots.push(Slot::Owned(upsample::upsample_nearest2d(
                    &input, node.kwargs.get("size"),
                )?)),
                "bilinear" => {
                    // Only align_corners=false is implemented; reject the
                    // true variant rather than silently returning wrong values.
                    if node.kwargs.get("align_corners").and_then(|v| v.as_bool()) == Some(true) {
                        return Err(unsupported("interpolate: align_corners=true not supported"));
                    }
                    slots.push(Slot::Owned(upsample::upsample_bilinear2d(
                        &input, node.kwargs.get("size"),
                    )?))
                }
                _ => return Err(unsupported(&format!("interpolate: unsupported mode '{mode}'"))),
            }
        }

        // Phase 7: extended ops
        "scatter" => {
            // call_method: scatter(self, dim, index, src) -> 4 args
            // call_function: scatter(src, dim, index) -> 3 args
            let is_method = node.args.len() >= 4;
            if is_method {
                // Read dim from slot (arg 1 is a scalar tensor)
                let dim = node.args.get(1).and_then(|a| a.index).and_then(|idx| {
                    match slots.get(idx) {
                        Some(Slot::Owned(t)) => unsafe { match t.dtype {
                            DType::I64 => Some(*(t.data.as_ptr() as *const i64) as isize),
                            DType::F32 => Some(*(t.data.as_ptr() as *const f32) as isize),
                            _ => None,
                        }},
                        Some(Slot::Input(i)) => unsafe {
                            let bt = BorrowedTensor::from_managed(capsules[*i].0);
                            bt.ok().and_then(|b| match b.dtype {
                                DType::I64 => Some(*(b.data as *const i64) as isize),
                                DType::F32 => Some(*(b.data as *const f32) as isize),
                                _ => None,
                            })
                        },
                        _ => None,
                    }
                }).unwrap_or(0);
                let self_tensor = slot_view(slots, capsules, arg_index(node, 0)?)?;
                let index = slot_view(slots, capsules, arg_index(node, 2)?)?;
                let src = slot_view(slots, capsules, arg_index(node, 3)?)?;
                slots.push(Slot::Owned(ops_phase7::scatter_method(&self_tensor, dim, &index, &src)?));
            } else {
                let dim = kw_isize(node, "dim", 0);
                let src = slot_view(slots, capsules, arg_index(node, 0)?)?;
                let index = slot_view(slots, capsules, arg_index(node, 1)?)?;
                slots.push(Slot::Owned(ops_phase7::scatter(&src, dim, &index)?));
            }
        }
        "scatter_add" => {
            let (dim, idx_pos, src_pos): (isize, usize, usize) = if node.args.len() >= 4 {
                (kw_isize(node, "dim", -1), 2, 3)
            } else {
                (kw_isize(node, "dim", 0), 1, 0)
            };
            let dim = if dim >= 0 { dim } else {
                node.args.get(1).and_then(|a| a.index).and_then(|idx| {
                    match slots.get(idx) {
                        Some(Slot::Owned(t)) => unsafe { match t.dtype {
                            DType::I64 => Some(*(t.data.as_ptr() as *const i64) as isize),
                            DType::F32 => Some(*(t.data.as_ptr() as *const f32) as isize),
                            _ => None,
                        }},
                        Some(Slot::Input(i)) => unsafe {
                            let bt = BorrowedTensor::from_managed(capsules[*i].0);
                            bt.ok().and_then(|b| match b.dtype {
                                DType::I64 => Some(*(b.data as *const i64) as isize),
                                DType::F32 => Some(*(b.data as *const f32) as isize),
                                _ => None,
                            })
                        },
                        _ => None,
                    }
                }).unwrap_or(0)
            };
            let index = slot_view(slots, capsules, arg_index(node, idx_pos)?)?;
            let src = slot_view(slots, capsules, arg_index(node, src_pos)?)?;
            slots.push(Slot::Owned(ops_phase7::scatter_add(&src, dim, &index)?));
        }
        "topk" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_usize(node, "k", 1);
            let dim = kw_isize(node, "dim", -1);
            let largest = kw_bool(node, "largest", true);
            let (values, indices) = ops_phase7::topk(&a, k, dim, largest)?;
            slots.push(Slot::Tuple(vec![values, indices]));
        }
        // sort returns only values (indices dropped by parser aliasing)
        "sort" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            let descending = kw_bool(node, "descending", false);
            let (values, indices) = ops_phase7::sort(&a, dim, descending)?;
            slots.push(Slot::Tuple(vec![values, indices]));
        }
        "argsort" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", -1);
            let descending = kw_bool(node, "descending", false);
            slots.push(Slot::Owned(ops_phase7::argsort(&a, dim, descending)?));
        }
        "repeat_interleave" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let reps = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(ops_phase7::repeat_interleave(&a, &reps, dim)?));
        }
        "repeat" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            // repeat() is emitted as call_method with positional args: [self, *repeats]
            // The repeats come from slots (constant tensors promoted to capsules).
            let reps: Vec<i64> = node.args[1..].iter().filter_map(|arg| {
                // Try slot reference first (constant tensor in capsule)
                if let Some(idx) = arg.index {
                    match slots.get(idx) {
                        Some(Slot::Owned(t)) => {
                            return unsafe { match t.dtype {
                                DType::F32 => Some(*(t.data.as_ptr() as *const f32) as i64),
                                DType::F64 => Some(*(t.data.as_ptr() as *const f64) as i64),
                                DType::I64 => Some(*(t.data.as_ptr() as *const i64)),
                                _ => None,
                            }};
                        }
                        Some(Slot::Input(i)) => {
                            let bt = unsafe { BorrowedTensor::from_managed(capsules[*i].0) };
                            if let Ok(bt) = bt {
                                return unsafe { match bt.dtype {
                                    DType::F32 => Some(*(bt.data as *const f32) as i64),
                                    DType::F64 => Some(*(bt.data as *const f64) as i64),
                                    DType::I64 => Some(*(bt.data as *const i64)),
                                    _ => None,
                                }};
                            }
                        }
                        _ => {}
                    }
                }
                // Fall back to value field
                arg.value.as_ref().and_then(|v| {
                    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))
                })
            }).collect();
            slots.push(Slot::Owned(ops_phase7::repeat(&a, &reps)?));
        }
        "einsum" => {
            let eq = node.kwargs.get("equation").and_then(|v| v.as_str()).unwrap_or("");
            let tensor_indices: Vec<usize> = node.args.iter().filter_map(|a| a.index).collect();
            let tensors: Vec<BorrowedTensor> = tensor_indices.iter()
                .map(|&i| slot_view(slots, capsules, i))
                .collect::<PyResult<_>>()?;
            let refs: Vec<&BorrowedTensor> = tensors.iter().collect();
            slots.push(Slot::Owned(ops_phase7::einsum(eq, &refs)?));
        }
        "prelu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(ops_phase7::prelu(&a, &w)?));
        }
        "nonzero" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(ops_phase7::nonzero(&a)?));
        }
        "clamp_tensor" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lo = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let hi = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(ops_phase7::clamp_tensor(&a, &lo, &hi)?));
        }
        // ── 50 extra super ops (SIMD + tiled) ──
        "atan" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::atan(&a)?)); }
        "asin" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::asin(&a)?)); }
        "acos" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::acos(&a)?)); }
        "sinh" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::sinh(&a)?)); }
        "cosh" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::cosh(&a)?)); }
        "asinh" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::asinh(&a)?)); }
        "acosh" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::acosh(&a)?)); }
        "atanh" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::atanh(&a)?)); }
        "erf" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::erf(&a)?)); }
        "erfc" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::erfc(&a)?)); }
        "expm1" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::expm1(&a)?)); }
        "log1p" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::log1p(&a)?)); }
        "log2" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::log2(&a)?)); }
        "log10" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::log10(&a)?)); }
        "trunc" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::trunc(&a)?)); }
        "frac" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::frac(&a)?)); }
        "square" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::square(&a)?)); }
        "exp2" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::exp2(&a)?)); }
        "atan2" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::atan2(&a, &b)?)); }
        "hypot" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::hypot(&a, &b)?)); }
        "fmod" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::fmod(&a, &b)?)); }
        "remainder" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::remainder(&a, &b)?)); }
        "copysign" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::copysign(&a, &b)?)); }
        "ldexp" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::ldexp(&a, &b)?)); }
        "lerp" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; let w = kw_f64(node, "weight", kw_f64(node, "w", 0.5)); slots.push(Slot::Owned(extra_ops::lerp(&a, &b, w)?)); }
        "bitwise_and" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::bitwise_and(&a, &b)?)); }
        "bitwise_or" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::bitwise_or(&a, &b)?)); }
        "bitwise_xor" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::bitwise_xor(&a, &b)?)); }
        "bitwise_not" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::bitwise_not(&a)?)); }
        "isfinite" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::isfinite(&a)?)); }
        "isinf" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::isinf(&a)?)); }
        "isnan" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::isnan(&a)?)); }
        "all" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::all(&a)?)); }
        "any" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::any(&a)?)); }
        "amax" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::amax(&a)?)); }
        "amin" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::amin(&a)?)); }
        "count_nonzero" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::count_nonzero(&a)?)); }
        "nansum" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::nansum(&a)?)); }
        "nanmean" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::nanmean(&a)?)); }
        "tile" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let repeats = kw_i64_vec(node, "repeats"); let repeats = if repeats.is_empty() { kw_i64_vec(node, "dims") } else { repeats }; slots.push(Slot::Owned(extra_ops::tile(&a, &repeats)?)); }
        "roll" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let shift = node.kwargs.get("shifts")
                .or_else(|| node.kwargs.get("shift"))
                .and_then(|v| v.as_i64().or_else(|| v.as_array().and_then(|arr| arr.first().and_then(|x| x.as_i64()))))
                .unwrap_or(1);
            let dim = node.kwargs.get("dims")
                .or_else(|| node.kwargs.get("dim"))
                .and_then(|v| v.as_i64().or_else(|| v.as_array().and_then(|arr| arr.first().and_then(|x| x.as_i64()))))
                .unwrap_or(0) as isize;
            slots.push(Slot::Owned(extra_ops::roll(&a, shift, dim)?));
        }
        "pixel_shuffle" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let r = kw_isize(node, "upscale_factor", kw_isize(node, "upscale", 2)) as i64; slots.push(Slot::Owned(extra_ops::pixel_shuffle(&a, r)?)); }
        "instance_norm" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let eps = kw_f64(node, "eps", 1e-5); slots.push(Slot::Owned(extra_ops::instance_norm(&a, eps)?)); }
        "cross_entropy" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::cross_entropy(&a, &b)?)); }
        "huber_loss" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; let d = kw_f64(node, "delta", 1.0); slots.push(Slot::Owned(extra_ops::huber_loss(&a, &b, d)?)); }
        "hardtanh" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let lo = kw_f64(node, "min_val", kw_f64(node, "min", -1.0)); let hi = kw_f64(node, "max_val", kw_f64(node, "max", 1.0)); slots.push(Slot::Owned(extra_ops::hardtanh(&a, lo, hi)?)); }
        "hardsigmoid" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops::hardsigmoid(&a)?)); }
        "glu" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let dim = kw_isize(node, "dim", -1); slots.push(Slot::Owned(extra_ops::glu(&a, dim)?)); }
        "bucketize" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops::bucketize(&a, &b)?)); }
        "histc" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let bins = kw_usize(node, "bins", 100); let min = kw_f64(node, "min", 0.0); let max = kw_f64(node, "max", 0.0); slots.push(Slot::Owned(extra_ops::histc(&a, bins, min, max)?)); }

        // ── Batch 2 operations (49 ops) ──
        "embedding_bag" => {
            let w = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let mode = kw_str(node, "mode", "mean");
            slots.push(Slot::Owned(extra_ops2::embedding_bag(&w, &idx, mode)?));
        }
        "unfold" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dimension", kw_isize(node, "dim", 0));
            let size = kw_i64(node, "size", 1);
            let step = kw_i64(node, "step", 1);
            slots.push(Slot::Owned(extra_ops2::unfold(&a, dim, size, step)?));
        }
        "fold" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops2::fold(&a, &out_sz)?));
        }
        "grid_sample" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let g = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops2::grid_sample(&a, &g)?));
        }
        "affine_grid" => {
            let theta = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(extra_ops2::affine_grid(&theta, &sz)?));
        }
        "pixel_unshuffle" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let r = kw_i64(node, "downscale_factor", kw_i64(node, "downscale", 2));
            slots.push(Slot::Owned(extra_ops2::pixel_unshuffle(&a, r)?));
        }
        "channel_shuffle" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let g = kw_i64(node, "groups", 2);
            slots.push(Slot::Owned(extra_ops2::channel_shuffle(&a, g)?));
        }
        "cummax" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(extra_ops2::cummax(&a, dim)?));
        }
        "cummin" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(extra_ops2::cummin(&a, dim)?));
        }
        "logcumsumexp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            slots.push(Slot::Owned(extra_ops2::logcumsumexp(&a, dim)?));
        }
        "scatter_reduce" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let reduce = kw_str(node, "reduce", "sum");
            slots.push(Slot::Owned(extra_ops2::scatter_reduce(&a, dim, &idx, &src, reduce)?));
        }
        "index_put" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let val = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(extra_ops2::index_put(&a, &idx, &val)?));
        }
        "index_add" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(extra_ops2::index_add(&a, dim, &idx, &src)?));
        }
        "masked_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let m = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(extra_ops2::masked_scatter(&a, &m, &src)?));
        }
        "take" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops2::take(&a, &idx)?));
        }
        "put" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let acc = kw_bool(node, "accumulate", false);
            slots.push(Slot::Owned(extra_ops2::put(&a, &idx, &src, acc)?));
        }
        "masked_select" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let m = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops2::masked_select(&a, &m)?));
        }
        "index_fill" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = kw_i64(node, "index", 0);
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(extra_ops2::index_fill(&a, dim, idx, val)?));
        }
        "bincount" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = if node.args.len() > 1 { slot_view(slots, capsules, arg_index(node, 1)?).ok() } else { None };
            slots.push(Slot::Owned(extra_ops2::bincount(&a, w.as_ref())?));
        }
        "unique" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::unique(&a)?));
        }
        "kthvalue" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_usize(node, "k", 1);
            slots.push(Slot::Owned(extra_ops2::kthvalue(&a, k)?));
        }
        "median" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::median(&a)?));
        }
        "quantile" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let q = kw_f64(node, "q", 0.5);
            let dim = kw_opt_dim(node)?;
            let keepdim = kw_bool(node, "keepdim", false);
            slots.push(Slot::Owned(extra_ops2::quantile(&a, q, dim, keepdim)?));
        }
        "histogram" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let bins = kw_usize(node, "bins", 100);
            slots.push(Slot::Owned(extra_ops2::histogram(&a, bins)?));
        }
        "searchsorted" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops2::searchsorted(&a, &v)?));
        }
        "meshgrid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let (o1, o2) = extra_ops2::meshgrid(&a, &b)?;
            slots.push(Slot::Tuple(vec![o1, o2]));
        }
        "cdist" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops2::cdist(&a, &b)?));
        }
        "pdist" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::pdist(&a)?));
        }
        "renorm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64(node, "p", 2.0);
            let dim = kw_isize(node, "dim", 0);
            let maxnorm = kw_f64(node, "maxnorm", 1.0);
            slots.push(Slot::Owned(extra_ops2::renorm(&a, p, dim, maxnorm)?));
        }
        "bernoulli" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let p = kw_f64(node, "p", 0.5);
            slots.push(Slot::Owned(extra_ops2::bernoulli(&a, p)?));
        }
        "multinomial" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let num_samples = kw_usize(node, "num_samples", 1);
            slots.push(Slot::Owned(extra_ops2::multinomial(&a, num_samples)?));
        }
        "logspace" => {
            let start = kw_f64(node, "start", 0.0);
            let end = kw_f64(node, "end", 1.0);
            let steps = kw_usize(node, "steps", 100);
            slots.push(Slot::Owned(extra_ops2::logspace(start, end, steps)?));
        }
        "eye" => {
            let n = kw_i64(node, "n", kw_i64_vec(node, "shape").first().copied().unwrap_or(3));
            slots.push(Slot::Owned(extra_ops2::eye(n)?));
        }
        "diag" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::diag(&a)?));
        }
        "diagonal" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let offset = kw_i64(node, "offset", 0);
            let dim1 = kw_isize(node, "dim1", 0);
            let dim2 = kw_isize(node, "dim2", 1);
            slots.push(Slot::Owned(extra_ops2::diagonal(&a, offset, dim1, dim2)?));
        }
        "trace" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::trace(&a)?));
        }
        "matrix_exp" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::matrix_exp(&a)?));
        }
        "slogdet" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (s, l) = extra_ops2::slogdet(&a)?;
            slots.push(Slot::Tuple(vec![s, l]));
        }
        "det" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::det(&a)?));
        }
        "lstsq" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops2::lstsq(&a, &b)?));
        }
        "pinverse" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops2::pinverse(&a)?));
        }
        "normal" => {
            let mean = kw_f64(node, "mean", 0.0);
            let std = kw_f64(node, "std", 1.0);
            let size = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(extra_ops2::normal(mean, std, &size)?));
        }
        "uniform" => {
            let from = kw_f64(node, "from", 0.0);
            let to = kw_f64(node, "to", 1.0);
            let size = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(extra_ops2::uniform(from, to, &size)?));
        }
        "triu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d = kw_i64(node, "diagonal", 0);
            slots.push(Slot::Owned(extra_ops2::triu(&a, d)?));
        }
        "tril" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d = kw_i64(node, "diagonal", 0);
            slots.push(Slot::Owned(extra_ops2::tril(&a, d)?));
        }
        "hann_window" => {
            let win_len = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(extra_ops2::hann_window(win_len, periodic)?));
        }
        "bartlett_window" => {
            let win_len = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(extra_ops2::bartlett_window(win_len, periodic)?));
        }
        "blackman_window" => {
            let win_len = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(extra_ops2::blackman_window(win_len, periodic)?));
        }
        "stft" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n_fft = kw_usize(node, "n_fft", 256);
            let hop = kw_usize(node, "hop_length", n_fft / 4);
            let win = kw_usize(node, "win_length", n_fft);
            slots.push(Slot::Owned(extra_ops2::stft(&a, n_fft, hop, win)?));
        }

        // ── Batch 3 operations (149 ops) ──
        "nextafter" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::nextafter(&a, &b)?)); }
        "heaviside" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::heaviside(&a, &b)?)); }
        "nan_to_num" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let nan = kw_f64(node, "nan", 0.0); let posinf = kw_f64(node, "posinf", f64::MAX); let neginf = kw_f64(node, "neginf", f64::MIN); slots.push(Slot::Owned(extra_ops3::nan_to_num(&a, nan, Some(posinf), Some(neginf))?)); }
        "logaddexp" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::logaddexp(&a, &b)?)); }
        "logaddexp2" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::logaddexp2(&a, &b)?)); }
        "sinc" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::sinc(&a)?)); }
        "i0" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::i0(&a)?)); }
        "i1" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::i1(&a)?)); }
        "i0e" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::i0e(&a)?)); }
        "i1e" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::i1e(&a)?)); }
        "bessel_j0" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::bessel_j0(&a)?)); }
        "bessel_j1" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::bessel_j1(&a)?)); }
        "bessel_y0" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::bessel_y0(&a)?)); }
        "bessel_y1" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::bessel_y1(&a)?)); }
        "digamma" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::digamma(&a)?)); }
        "lgamma" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::lgamma(&a)?)); }
        "polygamma" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let n = kw_i64(node, "n", 1); slots.push(Slot::Owned(extra_ops3::polygamma(n, &a)?)); }
        "mvlgamma" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let p = kw_i64(node, "p", 1); slots.push(Slot::Owned(extra_ops3::mvlgamma(&a, p)?)); }
        "erfinv" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::erfinv(&a)?)); }
        "erfcinv" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::erfcinv(&a)?)); }
        "ndtri" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::ndtri(&a)?)); }
        "ndtr" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::ndtr(&a)?)); }
        "log_ndtr" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::log_ndtr(&a)?)); }
        "logit" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let eps = kw_f64(node, "eps", -1.0); slots.push(Slot::Owned(extra_ops3::logit(&a, if eps < 0.0 { None } else { Some(eps) })?)); }
        "expit" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::expit(&a)?)); }
        "rad2deg" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::rad2deg(&a)?)); }
        "deg2rad" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::deg2rad(&a)?)); }
        "gcd" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::gcd(&a, &b)?)); }
        "lcm" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::lcm(&a, &b)?)); }
        "fmax" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::fmax(&a, &b)?)); }
        "fmin" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::fmin(&a, &b)?)); }
        "maximum" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::maximum(&a, &b)?)); }
        "minimum" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; let b = slot_view(slots, capsules, arg_index(node, 1)?)?; slots.push(Slot::Owned(extra_ops3::minimum(&a, &b)?)); }
        "signbit" => { let a = slot_view(slots, capsules, arg_index(node, 0)?)?; slots.push(Slot::Owned(extra_ops3::signbit(&a)?)); }
        "addcdiv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let t1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let val = kw_f64(node, "value", 1.0);
            slots.push(Slot::Owned(extra_ops3::addcdiv(&a, &t1, &t2, val)?));
        }
        "addcmul" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let t1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let val = kw_f64(node, "value", 1.0);
            slots.push(Slot::Owned(extra_ops3::addcmul(&a, &t1, &t2, val)?));
        }
        "addr" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let v1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(extra_ops3::addr(&a, &v1, &v2, beta, alpha)?));
        }
        "outer" | "ger" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::outer(&a, &b)?));
        }
        "mv" => {
            let mat = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let vec = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::mv(&mat, &vec)?));
        }
        "vdot" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::vdot(&a, &b)?));
        }
        "baddbmm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(extra_ops3::baddbmm(&a, &b1, &b2, beta, alpha)?));
        }
        "addbmm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(extra_ops3::addbmm(&a, &b1, &b2, beta, alpha)?));
        }
        "addmv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let mat = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let vec = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let beta = kw_f64(node, "beta", 1.0);
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(extra_ops3::addmv(&a, &mat, &vec, beta, alpha)?));
        }
        "kron" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::kron(&a, &b)?));
        }
        "inner" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::inner(&a, &b)?));
        }
        "trapz" | "trapezoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dx = kw_f64(node, "dx", 1.0);
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(extra_ops3::trapezoid(&a, None, dx, dim)?));
        }
        "cumulative_trapezoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dx = kw_f64(node, "dx", 1.0);
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(extra_ops3::cumulative_trapezoid(&a, dx, dim)?));
        }
        "celu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let alpha = kw_f64(node, "alpha", 1.0);
            slots.push(Slot::Owned(extra_ops3::celu(&a, alpha)?));
        }
        "hardshrink" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lambd = kw_f64(node, "lambd", 0.5);
            slots.push(Slot::Owned(extra_ops3::hardshrink(&a, lambd)?));
        }
        "softshrink" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lambd = kw_f64(node, "lambd", 0.5);
            slots.push(Slot::Owned(extra_ops3::softshrink(&a, lambd)?));
        }
        "tanhshrink" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::tanhshrink(&a)?));
        }
        "threshold" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let th = kw_f64(node, "threshold", 0.0);
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(extra_ops3::threshold(&a, th, val)?));
        }
        "logsigmoid" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::logsigmoid(&a)?));
        }
        "rrelu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lower = kw_f64(node, "lower", 1.0 / 8.0);
            let upper = kw_f64(node, "upper", 1.0 / 3.0);
            slots.push(Slot::Owned(extra_ops3::rrelu(&a, lower, upper)?));
        }
        "kl_div" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let log_target = kw_bool(node, "log_target", false);
            slots.push(Slot::Owned(extra_ops3::kl_div(&a, &b, log_target)?));
        }
        "poisson_nll_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let log_input = kw_bool(node, "log_input", true);
            let full = kw_bool(node, "full", false);
            let eps = kw_f64(node, "eps", 1e-8);
            slots.push(Slot::Owned(extra_ops3::poisson_nll_loss(&a, &b, log_input, full, eps)?));
        }
        "margin_ranking_loss" => {
            let x1 = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x2 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let margin = kw_f64(node, "margin", 0.0);
            slots.push(Slot::Owned(extra_ops3::margin_ranking_loss(&x1, &x2, &t, margin)?));
        }
        "hinge_embedding_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let margin = kw_f64(node, "margin", 1.0);
            slots.push(Slot::Owned(extra_ops3::hinge_embedding_loss(&a, &b, margin)?));
        }
        "multilabel_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::multilabel_margin_loss(&a, &b)?));
        }
        "soft_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::soft_margin_loss(&a, &b)?));
        }
        "multilabel_soft_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::multilabel_soft_margin_loss(&a, &b)?));
        }
        "cosine_embedding_loss" => {
            let x1 = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let x2 = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let t = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let margin = kw_f64(node, "margin", 0.0);
            slots.push(Slot::Owned(extra_ops3::cosine_embedding_loss(&x1, &x2, &t, margin)?));
        }
        "triplet_margin_loss" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pos = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let neg = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let margin = kw_f64(node, "margin", 1.0);
            slots.push(Slot::Owned(extra_ops3::triplet_margin_loss(&a, &pos, &neg, margin)?));
        }
        "ctc_loss" => {
            let log_p = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let tgt = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::ctc_loss(&log_p, &tgt)?));
        }
        "hamming_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(extra_ops3::hamming_window(n, periodic)?));
        }
        "kaiser_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            let beta = kw_f64(node, "beta", 12.0);
            slots.push(Slot::Owned(extra_ops3::kaiser_window(n, beta, periodic)?));
        }
        "gaussian_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            let std = kw_f64(node, "std", 1.0);
            slots.push(Slot::Owned(extra_ops3::gaussian_window(n, std, periodic)?));
        }
        "exponential_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            let tau = kw_f64(node, "tau", 1.0);
            slots.push(Slot::Owned(extra_ops3::exponential_window(n, tau, periodic)?));
        }
        "triangular_window" => {
            let n = kw_i64(node, "window_length", 100);
            let periodic = kw_bool(node, "periodic", true);
            slots.push(Slot::Owned(extra_ops3::triangular_window(n, periodic)?));
        }
        "cross" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", -1);
            slots.push(Slot::Owned(extra_ops3::cross(&a, &b, dim)?));
        }
        "linalg_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ord = kw_f64(node, "ord", 2.0);
            slots.push(Slot::Owned(extra_ops3::linalg_norm(&a, Some(ord))?));
        }
        "frobenius_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::frobenius_norm(&a)?));
        }
        "nuclear_norm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::nuclear_norm(&a)?));
        }
        "matrix_rank" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::matrix_rank(&a)?));
        }
        "matrix_power" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = kw_i64(node, "n", 1);
            slots.push(Slot::Owned(extra_ops3::matrix_power(&a, n)?));
        }
        "cholesky" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::cholesky(&a)?));
        }
        "cholesky_inverse" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::cholesky_inverse(&a)?));
        }
        "cholesky_solve" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::cholesky_solve(&a, &b)?));
        }
        "qr" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (q, r) = extra_ops3::qr(&a)?;
            slots.push(Slot::Tuple(vec![q, r]));
        }
        "svd" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (u, s, v) = extra_ops3::svd(&a)?;
            slots.push(Slot::Tuple(vec![u, s, v]));
        }
        "svdvals" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::svdvals(&a)?));
        }
        "eig" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (vals, vecs) = extra_ops3::eig(&a)?;
            slots.push(Slot::Tuple(vec![vals, vecs]));
        }
        "eigh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (vals, vecs) = extra_ops3::eigh(&a)?;
            slots.push(Slot::Tuple(vec![vals, vecs]));
        }
        "eigvals" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::eigvals(&a)?));
        }
        "eigvalsh" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::eigvalsh(&a)?));
        }
        "lu" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (p, l, u) = extra_ops3::lu(&a)?;
            slots.push(Slot::Tuple(vec![p, l, u]));
        }
        "triangular_solve" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let sol = extra_ops3::triangular_solve(&a, &b)?;
            slots.push(Slot::Owned(sol));
        }
        "select_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = kw_i64(node, "index", 0);
            slots.push(Slot::Owned(extra_ops3::select_scatter(&a, &src, dim, idx)?));
        }
        "slice_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let dim = kw_isize(node, "dim", 0);
            let start = kw_isize(node, "start", 0);
            let end = kw_isize(node, "end", 0);
            let step = kw_isize(node, "step", 1);
            slots.push(Slot::Owned(extra_ops3::slice_scatter(&a, &src, dim, Some(start as i64), Some(end as i64), step as i64)?));
        }
        "diagonal_scatter" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let offset = kw_i64(node, "offset", 0);
            slots.push(Slot::Owned(extra_ops3::diagonal_scatter(&a, &src, offset)?));
        }
        "index_copy" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let idx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let src = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(extra_ops3::index_copy(&a, dim, &idx, &src)?));
        }
        "narrow_copy" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let dim = kw_isize(node, "dim", 0);
            let start = kw_i64(node, "start", 0);
            let len = kw_i64(node, "length", 1);
            slots.push(Slot::Owned(extra_ops3::narrow_copy(&a, dim, start as usize, len as usize)?));
        }
        "movedim" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = kw_isize_vec(node, "source");
            let dst = kw_isize_vec(node, "destination");
            slots.push(Slot::Owned(extra_ops3::movedim(&a, &src, &dst)?));
        }
        "moveaxis" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let src = kw_isize_vec(node, "source");
            let dst = kw_isize_vec(node, "destination");
            slots.push(Slot::Owned(extra_ops3::moveaxis(&a, &src, &dst)?));
        }
        "swapdims" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d0 = kw_isize(node, "dim0", 0);
            let d1 = kw_isize(node, "dim1", 1);
            slots.push(Slot::Owned(extra_ops3::swapdims(&a, d0, d1)?));
        }
        "swapaxes" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let d0 = kw_isize(node, "axis0", kw_isize(node, "dim0", 0));
            let d1 = kw_isize(node, "axis1", kw_isize(node, "dim1", 1));
            slots.push(Slot::Owned(extra_ops3::swapaxes(&a, d0, d1)?));
        }
        "column_stack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::column_stack(&tens)?));
        }
        "row_stack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::row_stack(&tens)?));
        }
        "dstack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::dstack(&tens)?));
        }
        "hstack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::hstack(&tens)?));
        }
        "vstack" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::vstack(&tens)?));
        }
        "atleast_1d" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::atleast_1d(&tens)?));
        }
        "atleast_2d" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::atleast_2d(&tens)?));
        }
        "atleast_3d" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::atleast_3d(&tens)?));
        }
        "block_diag" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::block_diag(&tens)?));
        }
        "cartesian_prod" => {
            let mut tens = Vec::new();
            for i in 0..node.args.len() {
                if let Ok(idx) = arg_index(node, i) {
                    if let Ok(v) = slot_view(slots, capsules, idx) {
                        tens.push(v);
                    }
                }
            }
            slots.push(Slot::Owned(extra_ops3::cartesian_prod(&tens)?));
        }
        "combinations" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let r = kw_usize(node, "r", 2);
            slots.push(Slot::Owned(extra_ops3::combinations(&a, r)?));
        }
        "pad" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            let mode = kw_str(node, "mode", "constant");
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(extra_ops3::pad(&a, &pad, mode, val)?));
        }
        "constant_pad_nd" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            let val = kw_f64(node, "value", 0.0);
            slots.push(Slot::Owned(extra_ops3::constant_pad_nd(&a, &pad, val)?));
        }
        "reflection_pad1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(extra_ops3::reflection_pad1d(&a, &pad)?));
        }
        "reflection_pad2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(extra_ops3::reflection_pad2d(&a, &pad)?));
        }
        "replication_pad1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(extra_ops3::replication_pad1d(&a, &pad)?));
        }
        "replication_pad2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(extra_ops3::replication_pad2d(&a, &pad)?));
        }
        "zero_pad2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let pad = kw_i64_vec(node, "pad");
            slots.push(Slot::Owned(extra_ops3::zero_pad2d(&a, &pad)?));
        }
        "conv3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b = if node.args.len() > 2 { slot_view(slots, capsules, arg_index(node, 2)?).ok() } else { None };
            slots.push(Slot::Owned(extra_ops3::conv3d(&a, &w, b.as_ref())?));
        }
        "conv_transpose3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let b = if node.args.len() > 2 { slot_view(slots, capsules, arg_index(node, 2)?).ok() } else { None };
            slots.push(Slot::Owned(extra_ops3::conv_transpose3d(&a, &w, b.as_ref())?));
        }
        "max_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_i64_vec(node, "kernel_size");
            let s = kw_i64_vec(node, "stride");
            slots.push(Slot::Owned(extra_ops3::max_pool3d(&a, &k, &s)?));
        }
        "avg_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = kw_i64_vec(node, "kernel_size");
            let s = kw_i64_vec(node, "stride");
            slots.push(Slot::Owned(extra_ops3::avg_pool3d(&a, &k, &s)?));
        }
        "adaptive_max_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops3::adaptive_max_pool3d(&a, &out_sz)?));
        }
        "adaptive_avg_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops3::adaptive_avg_pool3d(&a, &out_sz)?));
        }
        "fractional_max_pool2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops3::fractional_max_pool2d(&a, &out_sz)?));
        }
        "fractional_max_pool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops3::fractional_max_pool3d(&a, &out_sz)?));
        }
        "lp_pool1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let norm = kw_f64(node, "norm_type", 2.0);
            slots.push(Slot::Owned(extra_ops3::lp_pool1d(&a, norm)?));
        }
        "lp_pool2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let norm = kw_f64(node, "norm_type", 2.0);
            slots.push(Slot::Owned(extra_ops3::lp_pool2d(&a, norm)?));
        }
        "max_unpool1d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops3::max_unpool1d(&a, &out_sz)?));
        }
        "max_unpool2d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops3::max_unpool2d(&a, &out_sz)?));
        }
        "max_unpool3d" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let out_sz = kw_i64_vec(node, "output_size");
            slots.push(Slot::Owned(extra_ops3::max_unpool3d(&a, &out_sz)?));
        }
        "rand" => {
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(extra_ops3::rand(&sz)?));
        }
        "randn" => {
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(extra_ops3::randn(&sz)?));
        }
        "randint" => {
            let low = kw_i64(node, "low", 0);
            let high = kw_i64(node, "high", 100);
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(extra_ops3::randint(low, high, &sz)?));
        }
        "randperm" => {
            let n = kw_i64(node, "n", 10);
            slots.push(Slot::Owned(extra_ops3::randperm(n)?));
        }
        "empty" => {
            let sz = kw_i64_vec(node, "size");
            slots.push(Slot::Owned(extra_ops3::empty(&sz, DType::F32)?));
        }
        "zeros_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::zeros_like(&a)?));
        }
        "ones_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::ones_like(&a)?));
        }
        "full_like" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let val = kw_f64(node, "fill_value", 0.0);
            slots.push(Slot::Owned(extra_ops3::full_like(&a, val)?));
        }
        "rnn_tanh_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let b_ih = if node.args.len() > 4 { slot_view(slots, capsules, arg_index(node, 4)?).ok() } else { None };
            let b_hh = if node.args.len() > 5 { slot_view(slots, capsules, arg_index(node, 5)?).ok() } else { None };
            slots.push(Slot::Owned(extra_ops3::rnn_tanh_cell(&x, &hx, &w_ih, &w_hh, b_ih.as_ref(), b_hh.as_ref())?));
        }
        "rnn_relu_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let b_ih = if node.args.len() > 4 { slot_view(slots, capsules, arg_index(node, 4)?).ok() } else { None };
            let b_hh = if node.args.len() > 5 { slot_view(slots, capsules, arg_index(node, 5)?).ok() } else { None };
            slots.push(Slot::Owned(extra_ops3::rnn_relu_cell(&x, &hx, &w_ih, &w_hh, b_ih.as_ref(), b_hh.as_ref())?));
        }
        "gru_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let b_ih = if node.args.len() > 4 { slot_view(slots, capsules, arg_index(node, 4)?).ok() } else { None };
            let b_hh = if node.args.len() > 5 { slot_view(slots, capsules, arg_index(node, 5)?).ok() } else { None };
            slots.push(Slot::Owned(extra_ops3::gru_cell(&x, &hx, &w_ih, &w_hh, b_ih.as_ref(), b_hh.as_ref())?));
        }
        "lstm_cell" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let hx = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let cx = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let w_ih = slot_view(slots, capsules, arg_index(node, 3)?)?;
            let w_hh = slot_view(slots, capsules, arg_index(node, 4)?)?;
            let b_ih = if node.args.len() > 5 { slot_view(slots, capsules, arg_index(node, 5)?).ok() } else { None };
            let b_hh = if node.args.len() > 6 { slot_view(slots, capsules, arg_index(node, 6)?).ok() } else { None };
            let (h, c) = extra_ops3::lstm_cell(&x, &hx, &cx, &w_ih, &w_hh, b_ih.as_ref(), b_hh.as_ref())?;
            slots.push(Slot::Tuple(vec![h, c]));
        }
        "multi_head_attention_forward" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(extra_ops3::multi_head_attention_forward(&q, &k, &v)?));
        }
        "lu_solve" => {
            let b = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let lu_d = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::lu_solve(&b, &lu_d)?));
        }
        "lu_unpack" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (p, l, u) = extra_ops3::lu_unpack(&a)?;
            slots.push(Slot::Tuple(vec![p, l, u]));
        }
        "linalg_solve" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(extra_ops3::linalg_solve(&a, &b)?));
        }
        "linalg_inv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::linalg_inv(&a)?));
        }
        "linalg_pinv" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::linalg_pinv(&a)?));
        }
        "linalg_det" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::linalg_det(&a)?));
        }
        "linalg_slogdet" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let (s, l) = extra_ops3::linalg_slogdet(&a)?;
            slots.push(Slot::Tuple(vec![s, l]));
        }
        "linalg_cond" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(extra_ops3::linalg_cond(&a)?));
        }

        // Advanced LLM & FlashAttention
        "flash_attention" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let k = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let v = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let mask = if let Ok(idx) = arg_index(node, 3) {
                Some(slot_view(slots, capsules, idx)?)
            } else {
                None
            };
            let is_causal = kw_bool(node, "is_causal", false);
            let scale = node.kwargs.get("scale").and_then(|v| v.as_f64());
            slots.push(Slot::Owned(attention::flash_attention_forward(&q, &k, &v, mask.as_ref(), is_causal, scale)?));
        }
        "fused_swiglu" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let gate_w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let up_w = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(attention::fused_swiglu(&x, &gate_w, &up_w)?));
        }
        "fused_geglu" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let gate_w = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let up_w = slot_view(slots, capsules, arg_index(node, 2)?)?;
            slots.push(Slot::Owned(attention::fused_geglu(&x, &gate_w, &up_w)?));
        }
        "fused_rmsnorm_residual" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let residual = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let weight = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let eps = kw_f64(node, "eps", 1e-5);
            slots.push(Slot::Owned(attention::fused_rmsnorm_residual(&x, &residual, &weight, eps)?));
        }

        // Universal Low-Bit Quantization & GEMM
        "quantize_per_tensor" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scale = kw_f64(node, "scale", 1.0);
            let zero_point = node.kwargs.get("zero_point").and_then(|v| v.as_i64()).unwrap_or(0);
            let dtype = if let Some(s) = node.kwargs.get("dtype").and_then(|v| v.as_str()) {
                dtype_from_spec(s).unwrap_or(DType::I32)
            } else {
                DType::I32
            };
            slots.push(Slot::Owned(quantization::quantize_per_tensor(&x, scale, zero_point, dtype)?));
        }
        "dequantize_per_tensor" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scale = kw_f64(node, "scale", 1.0);
            let zero_point = node.kwargs.get("zero_point").and_then(|v| v.as_i64()).unwrap_or(0);
            slots.push(Slot::Owned(quantization::dequantize_per_tensor(&q, scale, zero_point)?));
        }
        "quantize_per_channel" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let zero_points = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let axis = kw_isize(node, "axis", 0) as usize;
            slots.push(Slot::Owned(quantization::quantize_per_channel(&x, &scales, &zero_points, axis)?));
        }
        "dequantize_per_channel" => {
            let q = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let zero_points = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let axis = kw_isize(node, "axis", 0) as usize;
            slots.push(Slot::Owned(quantization::dequantize_per_channel(&q, &scales, &zero_points, axis)?));
        }
        "int8_gemm" => {
            let a = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let b = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let scale_a = kw_f64(node, "scale_a", 1.0);
            let scale_b = kw_f64(node, "scale_b", 1.0);
            slots.push(Slot::Owned(quantization::int8_gemm(&a, &b, scale_a, scale_b)?));
        }
        "nf4_dequantize" => {
            let packed = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let absmax = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let group_size = kw_isize(node, "group_size", 64) as usize;
            slots.push(Slot::Owned(quantization::nf4_dequantize(&packed, &absmax, group_size)?));
        }
        "int4_unpack_dequantize" => {
            let packed = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let scales = slot_view(slots, capsules, arg_index(node, 1)?)?;
            let zeros = slot_view(slots, capsules, arg_index(node, 2)?)?;
            let group_size = kw_isize(node, "group_size", 128) as usize;
            slots.push(Slot::Owned(quantization::int4_unpack_dequantize(&packed, &scales, &zeros, group_size)?));
        }

        // Universal FFT & Complex Suite
        "fft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::fft(&x, n, dim)?));
        }
        "ifft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::ifft(&x, n, dim)?));
        }
        "rfft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::rfft(&x, n, dim)?));
        }
        "irfft" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let n = node.kwargs.get("n").and_then(|v| v.as_i64());
            let dim = node.kwargs.get("dim").and_then(|v| v.as_i64());
            slots.push(Slot::Owned(fft_complex::irfft(&x, n, dim)?));
        }
        "fft2" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::fft2(&x)?));
        }
        "ifft2" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::ifft2(&x)?));
        }
        "fftn" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::fftn(&x)?));
        }
        "ifftn" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::ifftn(&x)?));
        }
        "fftshift" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::fftshift(&x)?));
        }
        "ifftshift" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::ifftshift(&x)?));
        }
        "complex" => {
            let re = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let im = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(fft_complex::complex(&re, &im)?));
        }
        "real" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::real(&x)?));
        }
        "imag" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::imag(&x)?));
        }
        "angle" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::angle(&x)?));
        }
        "polar" => {
            let abs = slot_view(slots, capsules, arg_index(node, 0)?)?;
            let ang = slot_view(slots, capsules, arg_index(node, 1)?)?;
            slots.push(Slot::Owned(fft_complex::polar(&abs, &ang)?));
        }
        "conj" => {
            let x = slot_view(slots, capsules, arg_index(node, 0)?)?;
            slots.push(Slot::Owned(fft_complex::conj(&x)?));
        }

        _ => {
            return Err(unsupported(&format!("unknown target {:?}", target)));
        }
    }
    Ok(())
}

/// Execute a payload with the native zero-copy engine.
/// Initialise the input slots from the capsules, validating shape/dtype.
fn init_input_slots(payload: &Payload, capsules: &[CapsuleRef]) -> PyResult<Vec<Slot>> {
    if payload.inputs.len() != capsules.len() {
        return Err(unsupported(&format!(
            "payload declares {} inputs but {} capsules were passed",
            payload.inputs.len(),
            capsules.len()
        )));
    }
    let mut slots: Vec<Slot> = Vec::with_capacity(payload.nodes.len() + payload.inputs.len());
    for (i, cap) in capsules.iter().enumerate() {
        let spec = &payload.inputs[i];
        let t = unsafe { BorrowedTensor::from_managed(cap.0) }?;
        let want = dtype_from_spec(&spec.dtype)
            .ok_or_else(|| unsupported(&format!("unknown input dtype '{}'", spec.dtype)))?;
        if t.dtype != want {
            return Err(unsupported(&format!(
                "input {i} dtype {} does not match payload spec {}",
                t.dtype.name(),
                spec.dtype
            )));
        }
        if !spec.shape.is_empty() && spec.shape != [0] && t.shape != spec.shape {
            return Err(unsupported(&format!(
                "input {i} shape {:?} does not match payload spec {:?}",
                t.shape, spec.shape
            )));
        }
        slots.push(Slot::Input(i));
    }
    Ok(slots)
}

/// Collect the requested node outputs out of the slot table.
/// For tuple slots, each requested element index is encoded as
/// (node_id << 16) | element_index.  The caller passes these as
/// output IDs; the high 16 bits select the node, the low 16 the element.
/// Plain node IDs (no element encoding) request the full tuple or single tensor.
fn collect_outputs(
    payload: &Payload,
    node_slot: &HashMap<u32, usize>,
    slots: &mut [Slot],
) -> PyResult<Vec<OwnedTensor>> {
    let mut ref_counts: HashMap<usize, usize> = HashMap::new();
    for id in &payload.outputs {
        let elem = (*id >> 16) as usize;
        let node_id = (*id & 0xFFFF) as u32;
        let use_tuple_elem = node_id != *id || elem > 0;
        let effective_id = if use_tuple_elem { node_id } else { *id };
        if let Some(&idx) = node_slot.get(&effective_id) {
            *ref_counts.entry(idx).or_insert(0) += 1;
        }
    }

    let mut out = Vec::with_capacity(payload.outputs.len());
    for id in &payload.outputs {
        // Check if this is an element-encoded output: high 16 bits = node_id,
        // low 16 bits = element index within the tuple.
        let elem = (*id >> 16) as usize;
        let node_id = (*id & 0xFFFF) as u32;
        let use_tuple_elem = node_id != *id || elem > 0;
        let effective_id = if use_tuple_elem { node_id } else { *id };

        let slot_idx = node_slot
            .get(&effective_id)
            .ok_or_else(|| unsupported(&format!("output references unknown node {effective_id}")))?;
        match &mut slots[*slot_idx] {
            Slot::Owned(t) => {
                let count = ref_counts.get_mut(slot_idx).unwrap();
                if *count > 1 {
                    *count -= 1;
                    out.push(t.clone());
                } else {
                    out.push(std::mem::take(t));
                }
            }
            Slot::View { data, shape, strides, dtype } => {
                let borrowed = BorrowedTensor {
                    data: *data,
                    shape: shape.clone(),
                    strides: strides.clone(),
                    dtype: *dtype,
                };
                out.push(shape_ops::to_contiguous(&borrowed)?);
            }
            Slot::Tuple(elems) => {
                if use_tuple_elem {
                    if elem >= elems.len() {
                        return Err(unsupported(&format!(
                            "tuple node {effective_id}: element {elem} out of range (len={})",
                            elems.len()
                        )));
                    }
                    out.push(std::mem::take(&mut elems[elem]));
                } else {
                    if let Some(t) = elems.first_mut() {
                        out.push(std::mem::take(t));
                    }
                }
            }
            Slot::Input(_) => {
                return Err(unsupported(&format!(
                    "node {effective_id} output aliases an input; handle passthrough in the interpreter"
                )));
            }
        }
    }
    // Recycle all remaining intermediate slots back into the thread memory pool
    for slot in slots.iter_mut() {
        if let Slot::Owned(t) = slot {
            if !t.data.is_empty() {
                let taken = std::mem::take(t);
                crate::pool::recycle_tensor(taken);
            }
        }
    }
    Ok(out)
}

/// Execute one fusion step, pushing exactly one output slot.
///
/// `nodes` are the remapped payload nodes (a fused group's members share the
/// group's output slot).
fn execute_step(
    step: &Step,
    nodes: &[Node],
    slots: &mut Vec<Slot>,
    capsules: &[CapsuleRef],
) -> PyResult<()> {
    let out_slot = slots.len();
    match step {
        Step::Node(i) => dispatch_node(&nodes[*i], slots, capsules)?,
        Step::Chain(plan) => {
            let out = fusion::run_chain(plan, nodes, slots, capsules)?;
            slots.push(Slot::Owned(out));
        }
        Step::Gemm { linear, spec, .. } => {
            let node = &nodes[*linear];
            let out = if node.target == "addmm" {
                // aten.addmm(bias, mat1, mat2) — mat2 is NOT transposed.
                let bias = slot_view(slots, capsules, arg_index(node, 0)?)?;
                let mat1 = slot_view(slots, capsules, arg_index(node, 1)?)?;
                let mat2 = slot_view(slots, capsules, arg_index(node, 2)?)?;
                linalg::addmm(&bias, &mat1, &mat2, Some(spec))?
            } else {
                let input = slot_view(slots, capsules, arg_index(node, 0)?)?;
                let weight = slot_view(slots, capsules, arg_index(node, 1)?)?;
                let bias = if node.args.len() > 2 {
                    Some(slot_view(slots, capsules, arg_index(node, 2)?)?)
                } else {
                    None
                };
                linalg::linear(&input, &weight, bias.as_ref(), Some(spec))?
            };
            slots.push(Slot::Owned(out));
        }
        Step::ConvBnRelu(spec) => {
            // 1) Run conv2d normally.
            let conv_node = &nodes[spec.conv];
            dispatch_node(conv_node, slots, capsules)?;

            // 2) Read BN parameters from input slots.
            let bn_node = &nodes[spec.bn];
            let w_slot = arg_index(bn_node, 1)?;
            let b_slot = arg_index(bn_node, 2)?;
            let rm_slot = arg_index(bn_node, 3)?;
            let rv_slot = arg_index(bn_node, 4)?;
            let w_view = slot_view(slots, capsules, w_slot)?;
            let b_view = slot_view(slots, capsules, b_slot)?;
            let rm_view = slot_view(slots, capsules, rm_slot)?;
            let rv_view = slot_view(slots, capsules, rv_slot)?;

            if w_view.dtype != DType::F32 {
                return Err(fusion::fusion_skip("conv_bn_relu: BN params must be f32"));
            }

            let c = w_view.shape[0] as usize;
            let w_data = unsafe { std::slice::from_raw_parts(w_view.data as *const f32, w_view.buffer_len()) };
            let b_data = unsafe { std::slice::from_raw_parts(b_view.data as *const f32, b_view.buffer_len()) };
            let rm_data = unsafe { std::slice::from_raw_parts(rm_view.data as *const f32, rm_view.buffer_len()) };
            let rv_data = unsafe { std::slice::from_raw_parts(rv_view.data as *const f32, rv_view.buffer_len()) };

            // 3) Precompute fused scale/bias per channel.
            let mut fused_scale = Vec::with_capacity(c);
            let mut fused_bias = Vec::with_capacity(c);
            for ch in 0..c {
                let inv_std = 1.0 / (rv_data[ch] + spec.eps).sqrt();
                fused_scale.push(w_data[ch] * inv_std);
                fused_bias.push(b_data[ch] - rm_data[ch] * w_data[ch] * inv_std);
            }

            // 4) Get conv output and apply fused BN+ReLU in single pass.
            let conv_out_slot = slots.len() - 1;
            let conv_view = slot_view(slots, capsules, conv_out_slot)?;
            let shape = conv_view.shape.clone();
            let n = shape[0] as usize;
            let spatial: usize = shape[2..].iter().map(|&d| d.max(0) as usize).product();
            let total = n * c * spatial;
            let in_data = unsafe { std::slice::from_raw_parts(conv_view.data as *const f32, conv_view.buffer_len()) };
            let mut out = OwnedTensor::new(DType::F32, shape.clone());
            let out_data = unsafe {
                std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, total)
            };
            for i in 0..n {
                for ch in 0..c {
                    let scale = fused_scale[ch];
                    let bias = fused_bias[ch];
                    let base_idx = i * c * spatial + ch * spatial;
                    for s in 0..spatial {
                        let val = in_data[base_idx + s] * scale + bias;
                        out_data[base_idx + s] = if val > 0.0 { val } else { 0.0 };
                    }
                }
            }
            // Replace the conv output slot with the fused result.
            slots[conv_out_slot] = Slot::Owned(out);
        }
    }
    debug_assert_eq!(slots.len(), out_slot + 1, "each step pushes exactly one slot");
    Ok(())
}

/// Execute a payload with the native zero-copy engine, fusing contiguous
/// supported runs into single-pass kernels (REQ-004).
///
/// Fusion is a pure execution-level rewrite; if a fused kernel cannot run
/// (e.g. an i64 elementwise chain), it raises `TB_FUSION_SKIP` and this
/// function falls back to the classic per-node path — observable behaviour
/// never changes.
pub fn execute_native(payload: &Payload, capsules: &[CapsuleRef]) -> PyResult<Vec<OwnedTensor>> {
    let base = payload.inputs.len();

    // Plan fusion on a clone (the planner rewrites arg slots to group slots).
    // TORCHBURN_NO_FUSION=1 skips the planner for benchmarking.
    let mut nodes = payload.nodes.clone();
    let no_fusion = std::env::var("TORCHBURN_NO_FUSION").map_or(false, |v| v == "1" || v == "true");
    let fp = if no_fusion {
        fusion::FusionPlan { steps: (0..nodes.len()).map(|i| Step::Node(i)).collect(), node_step: (0..nodes.len()).collect() }
    } else {
        fusion::plan(&nodes, base)
    };

    // Safety: a fused chain's *intermediate* members don't materialise their
    // own output (the group output is the chain's last node).  If the caller
    // explicitly requested one, fusion would return the wrong tensor — refuse
    // it and run unfused.  (GEMM epilogue members are safe: the group output
    // IS the activation's output.)
    let requested: std::collections::HashSet<u32> = payload.outputs.iter().copied().collect();
    let mut unsafe_output = false;
    for step in &fp.steps {
        if let Step::Chain(plan) = step {
            for &m in &plan.nodes[..plan.nodes.len() - 1] {
                if requested.contains(&nodes[m].id) {
                    unsafe_output = true;
                }
            }
        }
        if let Step::Gemm { linear, .. } = step {
            if requested.contains(&nodes[*linear].id) {
                unsafe_output = true;
            }
        }
    }
    if !unsafe_output {
        // Remap every argument slot to the step that produces it.
        let mut remap = Vec::with_capacity(base + nodes.len());
        remap.extend(0..base);
        remap.extend((0..nodes.len()).map(|i| base + fp.node_step[i]));
        for node in nodes.iter_mut() {
            for arg in node.args.iter_mut() {
                if let Some(s) = arg.index {
                    if s < remap.len() {
                        arg.index = Some(remap[s]);
                    }
                }
                if let Some(Value::Array(arr)) = arg.value.as_mut() {
                    for v in arr.iter_mut() {
                        if let Some(u) = v.as_u64() {
                            let s = u as usize;
                            if s < remap.len() {
                                *v = Value::from(remap[s] as u64);
                            }
                        }
                    }
                }
            }
        }

        let mut slots = init_input_slots(payload, capsules)?;
        let mut node_slot: HashMap<u32, usize> = HashMap::with_capacity(nodes.len());
        let mut ok = true;
        for (si, step) in fp.steps.iter().enumerate() {
            let out_slot = base + si;
            match execute_step(step, &nodes, &mut slots, capsules) {
                Ok(()) => {}
                Err(e) if e.to_string().contains(fusion::FUSION_SKIP_MARKER) => {
                    ok = false;
                    break;
                }
                Err(e) => return Err(e),
            }
            for member in step.member_nodes() {
                node_slot.insert(nodes[member].id, out_slot);
            }
        }
        if ok {
            return collect_outputs(payload, &node_slot, &mut slots);
        }
    }

    // Classic unfused path (also the fallback after a fusion skip): one
    // dispatch + allocation per node.
    let mut slots = init_input_slots(payload, capsules)?;
    let mut node_slot: HashMap<u32, usize> = HashMap::with_capacity(payload.nodes.len());
    for node in &payload.nodes {
        dispatch_node(node, &mut slots, capsules)?;
        node_slot.insert(node.id, slots.len() - 1);
    }
    collect_outputs(payload, &node_slot, &mut slots)
}

#[cfg(feature = "burn")]
mod burn_impl {
    use super::*;

    pub fn execute_burn(
        py: Python<'_>,
        payload: &Payload,
        capsules: &[Bound<'_, PyCapsule>],
    ) -> PyResult<Vec<Py<PyCapsule>>> {
        crate::burn_engine::execute_plan(py, payload, capsules)
    }
}

/// Parse the payload and run it on the selected engine.
pub fn execute_plan(
    py: Python<'_>,
    payload_json: &str,
    capsules: &[Bound<'_, PyCapsule>],
) -> PyResult<Vec<Py<PyCapsule>>> {
    if payload_json.len() > MAX_PAYLOAD_BYTES {
        return Err(unsupported(&format!(
            "payload too large ({} bytes > {} limit)",
            payload_json.len(),
            MAX_PAYLOAD_BYTES
        )));
    }
    let payload: Payload = serde_json::from_str(payload_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid payload: {e}")))?;

    #[cfg(feature = "burn")]
    if engine_is_burn() {
        return burn_impl::execute_burn(py, &payload, capsules);
    }

    let refs: Vec<CapsuleRef> = capsules
        .iter()
        .map(crate::dlpack::capsule_ref)
        .collect::<PyResult<_>>()?;
    let native_out = py.allow_threads(|| execute_native(&payload, &refs))?;

    let mut out = Vec::with_capacity(native_out.len());
    for owned in native_out {
        out.push(crate::dlpack::owned_to_capsule_owned(py, owned)?);
    }
    Ok(out)
}

/// Which execution engine is active?
pub fn engine_name() -> &'static str {
    #[cfg(feature = "burn")]
    {
        if engine_is_burn() {
            match crate::burn_engine::backend_choice() {
                #[cfg(feature = "burn-wgpu")]
                crate::burn_engine::BurnBackendChoice::Wgpu => {
                    return if crate::wgpu_backend::gpu_available() {
                        "burn_wgpu"
                    } else {
                        "burn_ndarray"
                    };
                }
                _ => return "burn_ndarray",
            }
        }
    }
    "native_cpu"
}

#[cfg(feature = "burn")]
fn engine_is_burn() -> bool {
    // If the user explicitly chose CPU, respect that choice:
    if matches!(std::env::var("TORCHBURN_DEVICE").as_deref(), Ok("cpu"))
        || matches!(
            std::env::var("TORCHBURN_ENGINE").as_deref(),
            Ok("native_cpu") | Ok("cpu")
        )
    {
        return false;
    }
    // Accept explicit Burn / WGPU engine selections:
    if matches!(
        std::env::var("TORCHBURN_ENGINE").as_deref(),
        Ok("burn") | Ok("burn-wgpu") | Ok("wgpu") | Ok("burn_gpu")
    ) {
        return true;
    }
    // GPU First: Default to Burn WGPU if available on the system
    #[cfg(feature = "burn-wgpu")]
    {
        if crate::wgpu_backend::gpu_available() {
            return true;
        }
    }
    false
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
                    return Err(pyo3::exceptions::PyValueError::new_err("input shape contains negative dim"));
                }
                if dtype_from_spec(&dtype).is_none() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown dtype '{dtype}'")));
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
                let d: &Bound<'_, PyDict> = item.downcast().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("node must be a dict")
                })?;
                let id: u32 = d.get_item("id")?.map(|o| o.extract().unwrap_or(0)).unwrap_or(0);
                let target: String = d.get_item("target")?.map(|o| {
                    o.extract::<String>().unwrap_or_default()
                }).unwrap_or_default();
                if target.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("node {id} missing target")));
                }

                // Parse args: list of dicts with optional index/value
                let args: Vec<ArgRef> = match d.get_item("args")? {
                    Some(ao) => {
                        let al: &Bound<'_, PyList> = ao.downcast().map_err(|_| {
                            pyo3::exceptions::PyValueError::new_err("node 'args' must be a list")
                        })?;
                        al.iter().map(|a| {
                            if let Ok(ad) = a.downcast::<PyDict>() {
                                let index: Option<usize> = ad.get_item("index")?.and_then(|o| o.extract().ok());
                                let value: Option<serde_json::Value> = ad.get_item("value")?.map(|o| {
                                    py_to_json(&o).ok()
                                }).flatten();
                                Ok(ArgRef { index, value })
                            } else {
                                Ok(ArgRef { index: None, value: None })
                            }
                        }).collect::<PyResult<Vec<_>>>()?
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
                            if let (Ok(ks), Some(jv)) = (k.extract::<String>(), py_to_json(&v).ok()) {
                                m.insert(ks, jv);
                            }
                        }
                        m
                    }
                    None => HashMap::new(),
                };

                v.push(Node { id, target, args, kwargs });
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
            list.iter().filter_map(|o| o.extract::<u32>().ok()).collect()
        }
        None => vec![],
    };
    if nodes.is_empty() && !inputs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("payload has inputs but no nodes"));
    }

    Ok(Payload { inputs, nodes, outputs })
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

/// Execute a payload directly from a Python dict, bypassing JSON.
pub fn execute_from_dict(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    capsules: &[Bound<'_, PyCapsule>],
) -> PyResult<Vec<Py<PyCapsule>>> {
    let payload = dict_to_payload(dict)?;

    #[cfg(feature = "burn")]
    if engine_is_burn() {
        return burn_impl::execute_burn(py, &payload, capsules);
    }

    let refs: Vec<CapsuleRef> = capsules
        .iter()
        .map(crate::dlpack::capsule_ref)
        .collect::<PyResult<_>>()?;
    let native_out = py.allow_threads(|| execute_native(&payload, &refs))?;

    let mut out = Vec::with_capacity(native_out.len());
    for owned in native_out {
        out.push(crate::dlpack::owned_to_capsule_owned(py, owned)?);
    }
    Ok(out)
}
