//! Optional Burn execution engine (feature `burn` / `burn-wgpu`).
//!
//! Maps a payload onto [`burn::tensor::Tensor`] operations.  The op bodies are
//! backend-agnostic and run on either the pure-CPU `NdArray<f32>` backend or
//! the `Wgpu` backend (Metal on macOS, Vulkan on Linux/Windows, DX12 via the
//! same wgpu stack).  The backend is selected at runtime:
//!
//! * `TORCHBURN_ENGINE=burn`      → `NdArray` (CPU, works everywhere)
//! * `TORCHBURN_ENGINE=burn-wgpu` → `Wgpu` GPU, with automatic CPU fallback
//!   when no GPU adapter is available (e.g. headless CI, VMs).
//! * `TORCHBURN_WGPU_BACKEND=dx12|vulkan|metal` forces a graphics API.
//!
//! Semantics:
//! * The Burn path executes the ops it supports natively (elementwise math,
//!   activations, comparisons, and same-rank broadcasting for binary ops).
//! * Ops outside that scope raise `TB_UNSUPPORTED`, and `execute_plan`
//!   delegates the whole payload to the native zero-copy engine instead of
//!   failing — so every Burn engine is a strict superset of native
//!   correctness (REQ-002 fallback philosophy).
//! * f32 only, contiguous inputs; anything else is delegated to native.
//! * The Burn path copies data into Burn-managed buffers at the boundary —
//!   the zero-copy guarantee (REQ-003) belongs to the native engine; the
//!   fused wgpu engine will own its device buffers instead.

use crate::dlpack::{
    self, contiguous_strides, dtype_from_spec, unsupported, BorrowedTensor, DType, OwnedTensor,
};
use crate::engine::{self, Node, Payload};
use burn::backend::NdArray;
use burn::tensor::activation::{
    leaky_relu, log_softmax, mish, sigmoid, silu, softmax, softplus, tanh,
};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Tensor, TensorData};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Read all capsule inputs into contiguous f32 buffers.
fn read_inputs(payload: &Payload, capsules: &[Bound<'_, PyCapsule>]) -> PyResult<Vec<Vec<f32>>> {
    if payload.inputs.len() != capsules.len() {
        return Err(unsupported(&format!(
            "payload declares {} inputs but {} capsules were passed",
            payload.inputs.len(),
            capsules.len()
        )));
    }
    let mut inputs = Vec::with_capacity(capsules.len());
    for (i, cap) in capsules.iter().enumerate() {
        let t = BorrowedTensor::from_capsule(cap)?;
        let want = dtype_from_spec(&payload.inputs[i].dtype).ok_or_else(|| {
            unsupported(&format!(
                "unknown input dtype '{}'",
                payload.inputs[i].dtype
            ))
        })?;
        if t.dtype != want {
            return Err(unsupported(&format!(
                "input {i} dtype {} does not match payload spec {}",
                t.dtype.name(),
                payload.inputs[i].dtype
            )));
        }
        if t.shape != payload.inputs[i].shape {
            return Err(unsupported(&format!(
                "input {i} shape {:?} does not match payload spec {:?}",
                t.shape, payload.inputs[i].shape
            )));
        }
        if t.dtype != DType::F32 {
            return Err(unsupported("burn engine currently supports f32 only"));
        }
        if t.strides != contiguous_strides(&t.shape) {
            return Err(unsupported("burn engine requires contiguous inputs"));
        }
        let n = t.elem_count();
        if n == 0 {
            return Err(unsupported(
                "burn engine does not support empty tensors; fallback to native",
            ));
        }
        // SAFETY: the DLPack buffer is alive for this call and holds n f32s.
        let slice = unsafe { std::slice::from_raw_parts(t.data as *const f32, n) };
        inputs.push(slice.to_vec());
    }
    Ok(inputs)
}

/// Normalize a dim (handle negatives) to [0, R).
fn norm_dim(dim: isize, rank: usize) -> PyResult<usize> {
    let d = if dim < 0 { rank as isize + dim } else { dim };
    if d < 0 || d as usize >= rank {
        return Err(unsupported(&format!(
            "dim {dim} out of range for rank {rank}"
        )));
    }
    Ok(d as usize)
}

/// Execute a rank-R payload: elementwise ops over same-shape tensors.
fn run_rank<B, const R: usize>(
    payload: &Payload,
    inputs: Vec<Vec<f32>>,
) -> PyResult<Vec<OwnedTensor>>
where
    B: BurnBackend<FloatElem = f32>,
{
    let mut env: Vec<Option<Tensor<B, R>>> = Vec::with_capacity(payload.nodes.len() + inputs.len());
    // Original rank of every env slot (inputs as declared; node outputs are R).
    // linalg arms use it to reject matrix-vector forms burn cannot express.
    let mut orig_ranks: Vec<usize> = Vec::with_capacity(env.capacity());
    let device = B::Device::default();

    for (i, values) in inputs.into_iter().enumerate() {
        let mut dims: Vec<usize> = payload.inputs[i]
            .shape
            .iter()
            .map(|&d| d as usize)
            .collect();
        let orig_rank = dims.len();
        // Effective scalars (0-d or all-1 dims) must have rank-R dims for
        // TensorData; run_node expands them to sibling shapes before ops.
        if dims.iter().all(|&d| d == 1) {
            dims = vec![1; R];
        }
        if dims.len() != R {
            if dims.len() == 1 && R >= 2 {
                // Rank-1 operands (e.g. linear/addmm bias [N]) are unsqueezed
                // to rank R with leading 1s: [N] -> [1, ..., 1, N].  This lets
                // burn represent them in the uniform-rank env; broadcast_pair
                // expands them to sibling shapes at the op site.
                while dims.len() < R {
                    dims.insert(0, 1);
                }
            } else {
                // Rank mismatch (e.g. 2-D + 1-D in a way we can't broadcast):
                // delegate to the native engine.
                return Err(unsupported("burn engine requires same-rank operands"));
            }
        }
        // Move `values` directly into TensorData without redundant cloning.
        env.push(Some(Tensor::from_data(
            TensorData::new(values, dims),
            &device,
        )));
        orig_ranks.push(orig_rank);
    }

    let mut node_slot: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for node in &payload.nodes {
        let out = run_node::<B, R>(node, &env, &orig_ranks)?;
        env.push(Some(out));
        orig_ranks.push(R);
        node_slot.insert(node.id, env.len() - 1);
    }

    let mut ref_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for id in &payload.outputs {
        if let Some(idx) = node_slot.get(id) {
            *ref_counts.entry(*idx).or_insert(0) += 1;
        }
    }

    let mut out = Vec::with_capacity(payload.outputs.len());
    for id in &payload.outputs {
        let idx = *node_slot
            .get(id)
            .ok_or_else(|| unsupported(&format!("output references unknown node {id}")))?;
        let count = ref_counts.get_mut(&idx).unwrap();
        *count -= 1;
        let tensor = if *count == 0 {
            // Last reference: avoid cloning and avoid creating a 0-sized tensor on GPU device.
            env[idx]
                .take()
                .ok_or_else(|| unsupported("slot already consumed"))?
        } else {
            env[idx]
                .as_ref()
                .ok_or_else(|| unsupported("slot already consumed"))?
                .clone()
        };
        let data = tensor.into_data();
        let shape: Vec<i64> = data.shape.iter().map(|&d| d as i64).collect();
        let values = data
            .to_vec::<f32>()
            .map_err(|_| unsupported("burn engine: could not read output data"))?;
        out.push(owned_from_values(values, shape));
    }
    drop(env);
    drop(node_slot);
    drop(ref_counts);
    B::sync(&device);
    Ok(out)
}

/// Broadcast two same-rank tensors to their common shape (1-dims expand).
fn broadcast_pair<B, const R: usize>(
    a: &Tensor<B, R>,
    b: &Tensor<B, R>,
) -> PyResult<(Tensor<B, R>, Tensor<B, R>)>
where
    B: BurnBackend<FloatElem = f32>,
{
    let da = a.shape().dims::<R>();
    let db = b.shape().dims::<R>();
    let mut target = [0usize; R];
    let mut compatible = true;
    for i in 0..R {
        target[i] = if da[i] == db[i] {
            da[i]
        } else if da[i] == 1 {
            db[i]
        } else if db[i] == 1 {
            da[i]
        } else {
            compatible = false;
            break;
        };
    }
    if !compatible {
        return Err(unsupported("burn engine: incompatible broadcast shapes"));
    }
    let mut a_out = a.clone();
    let mut b_out = b.clone();
    if a.shape().dims::<R>() != target {
        a_out = a.clone().expand(target);
    }
    if b.shape().dims::<R>() != target {
        b_out = b.clone().expand(target);
    }
    Ok((a_out, b_out))
}

fn run_node<B, const R: usize>(
    node: &Node,
    env: &[Option<Tensor<B, R>>],
    orig_ranks: &[usize],
) -> PyResult<Tensor<B, R>>
where
    B: BurnBackend<FloatElem = f32>,
{
    let arg = |pos: usize| -> PyResult<usize> {
        node.args.get(pos).and_then(|a| a.index).ok_or_else(|| {
            unsupported(&format!("node '{}' has an unindexed argument", node.target))
        })
    };
    let unary = |pos: usize| -> PyResult<Tensor<B, R>> {
        let idx = arg(pos)?;
        env.get(idx)
            .and_then(|t| t.as_ref())
            .cloned()
            .ok_or_else(|| unsupported(&format!("slot {idx} is empty or out of bounds")))
    };
    let binary = |pos: usize| -> PyResult<Tensor<B, R>> {
        let idx = arg(pos)?;
        env.get(idx)
            .and_then(|t| t.as_ref())
            .cloned()
            .ok_or_else(|| unsupported(&format!("slot {idx} is empty or out of bounds")))
    };
    let kw_f64 = |key: &str, default: f64| -> f64 {
        node.kwargs
            .get(key)
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    };
    let kw_isize = |key: &str, default: isize| -> isize {
        node.kwargs
            .get(key)
            .and_then(|v| v.as_i64())
            .map(|v| v as isize)
            .unwrap_or(default)
    };

    match node.target.as_str() {
        // Phase 1: binary elementwise (with same-rank broadcasting)
        "add" | "sub" | "mul" | "div" => {
            let a = binary(0)?;
            let b = binary(1)?;
            let (a, b) = broadcast_pair(&a, &b)?;
            let out = match node.target.as_str() {
                "add" => a.add(b),
                "sub" => a.sub(b),
                "mul" => a.mul(b),
                _ => a.div(b),
            };
            Ok(out)
        }

        // Phase 2: comparisons (Bool -> f32)
        // Delegated to the native engine: burn 0.18's Wgpu/Metal backend
        // returns garbage for equal()/lower()/... bool-to-f32 readback (seen
        // on macOS CI). Native comparison is correct on every platform and
        // zero-copy, so the superset fallback handles it.
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
            return Err(unsupported("comparisons run on the native engine"));
        }

        // Phase 2: unary math
        "abs" => Ok(unary(0)?.abs()),
        "neg" => Ok(unary(0)?.neg()),
        "sign" => Ok(unary(0)?.sign()),
        "sqrt" => Ok(unary(0)?.sqrt()),
        "rsqrt" => Ok(unary(0)?.sqrt().recip()),
        "exp" => Ok(unary(0)?.exp()),
        "log" => Ok(unary(0)?.log()),
        "reciprocal" => Ok(unary(0)?.recip()),
        "ceil" => Ok(unary(0)?.ceil()),
        "floor" => Ok(unary(0)?.floor()),
        "clamp" => {
            let x = unary(0)?;
            let min = kw_f64("min", f64::NEG_INFINITY);
            let max = kw_f64("max", f64::INFINITY);
            Ok(x.clamp(min as f32, max as f32))
        }
        "pow" => {
            let x = unary(0)?;
            let exp = kw_f64("exp", 2.0);
            Ok(x.powf_scalar(exp as f32))
        }

        // Phase 2: activations
        "sigmoid" => Ok(sigmoid(unary(0)?)),
        "tanh" => Ok(tanh(unary(0)?)),
        "gelu" => {
            // tanh approximation (matches the native engine and torch's
            // approximate="tanh"): 0.5*x*(1 + tanh(0.7978845608*(x + 0.044715*x^3)))
            let x = unary(0)?;
            let x3 = x.clone().powf_scalar(3.0);
            let inner = x
                .clone()
                .add(x3.mul_scalar(0.044715))
                .mul_scalar(0.7978845608);
            Ok(x.mul(inner.tanh().add_scalar(1.0)).mul_scalar(0.5))
        }
        "silu" => Ok(silu(unary(0)?)),
        "mish" => Ok(mish(unary(0)?)),
        "leaky_relu" => {
            let slope = kw_f64("negative_slope", 0.01);
            Ok(leaky_relu(unary(0)?, slope))
        }
        "softplus" => {
            // softplus(x, beta) = 1/beta * log(1 + exp(beta * x))
            let beta = kw_f64("beta", 1.0);
            Ok(softplus(unary(0)?, beta))
        }
        "softmax" => {
            let dim = norm_dim(kw_isize("dim", -1), R)?;
            Ok(softmax(unary(0)?, dim))
        }
        "log_softmax" => {
            let dim = norm_dim(kw_isize("dim", -1), R)?;
            Ok(log_softmax(unary(0)?, dim))
        }

        // Phase 2: linalg — executed natively on the burn backend (GPU when
        // wgpu is selected).  burn's `matmul` handles 2-D matmul and 3-D
        // batched matmul (bmm) in one op.
        "matmul" | "bmm" => {
            let ai = arg(0)?;
            let bi = arg(1)?;
            // A rank-1 operand (e.g. matrix-vector [K] @ [K,N]) cannot be
            // represented exactly in the uniform-rank env — delegate to native.
            if orig_ranks.get(ai).copied().unwrap_or(R) != R
                || orig_ranks.get(bi).copied().unwrap_or(R) != R
            {
                return Err(unsupported(
                    "burn engine: matmul/bmm needs same-rank 2-D/3-D operands",
                ));
            }
            // Pre-validate shapes: burn's `matmul` panics (not returns) on
            // mismatches, so reject cleanly to enable the native fallback.
            let sa = unary(ai)?.shape().dims::<R>();
            let sb = unary(bi)?.shape().dims::<R>();
            if sa[R - 1] != sb[R - 2] {
                return Err(unsupported(&format!(
                    "burn engine: matmul inner dims mismatch: {} vs {}",
                    sa[R - 1],
                    sb[R - 2]
                )));
            }
            if node.target == "bmm" && R >= 3 && sa[..R - 2] != sb[..R - 2] {
                return Err(unsupported(&format!(
                    "burn engine: bmm batch dims mismatch: {:?} vs {:?}",
                    &sa[..R - 2],
                    &sb[..R - 2]
                )));
            }
            Ok(unary(ai)?.matmul(unary(bi)?))
        }
        // linear(input, weight, bias?) = input @ weight^T (+ bias)
        "linear" => {
            let input = unary(0)?;
            let weight = unary(1)?;
            if orig_ranks.get(arg(0)?).copied().unwrap_or(R) != R
                || orig_ranks.get(arg(1)?).copied().unwrap_or(R) != R
            {
                return Err(unsupported(
                    "burn engine: linear needs 2-D input and weight",
                ));
            }
            // weight is [O, I]; after swap_dims it is [I, O], so the input's
            // last dim must equal weight[1] — validate to avoid burn's panic.
            let si = input.shape().dims::<R>();
            let sw = weight.shape().dims::<R>();
            if si[R - 1] != sw[1] {
                return Err(unsupported(&format!(
                    "burn engine: linear dim mismatch: input last dim {} vs weight in-dim {}",
                    si[R - 1],
                    sw[1]
                )));
            }
            let mut out = input.matmul(weight.swap_dims(0, 1));
            if node.args.len() > 2 {
                let bias = unary(2)?;
                let (o, b) = broadcast_pair(&out, &bias)?;
                out = o.add(b);
            }
            Ok(out)
        }
        // aten.addmm(bias, mat1, mat2) = mat1 @ mat2 + bias (mat2 NOT transposed)
        "addmm" => {
            let bias = unary(0)?;
            let mat1 = unary(1)?;
            let mat2 = unary(2)?;
            if orig_ranks.get(arg(1)?).copied().unwrap_or(R) != R
                || orig_ranks.get(arg(2)?).copied().unwrap_or(R) != R
            {
                return Err(unsupported("burn engine: addmm needs 2-D mat1 and mat2"));
            }
            let sm1 = mat1.shape().dims::<R>();
            let sm2 = mat2.shape().dims::<R>();
            if sm1[R - 1] != sm2[R - 2] {
                return Err(unsupported(&format!(
                    "burn engine: addmm inner dims mismatch: {} vs {}",
                    sm1[R - 1],
                    sm2[R - 2]
                )));
            }
            let out = mat1.matmul(mat2);
            let (o, b) = broadcast_pair(&out, &bias)?;
            Ok(o.add(b))
        }

        // Shape ops (rank-preserving, GPU-accelerated)
        "reshape" => {
            let t = unary(0)?;
            let shape_vec: Vec<usize> = node
                .kwargs
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_i64())
                        .map(|x| x as usize)
                        .collect()
                })
                .unwrap_or_default();
            if shape_vec.len() != R {
                return Err(unsupported(
                    "burn engine: reshape rank mismatch (GPU fallback to native)",
                ));
            }
            let old_numel: usize = t.shape().dims.iter().product();
            let new_numel: usize = shape_vec.iter().product();
            if old_numel != new_numel {
                return Err(unsupported(
                    "burn engine: reshape numel mismatch (fallback to native)",
                ));
            }
            // Convert Vec to array [usize; R]
            let shape_arr: [usize; R] = shape_vec
                .try_into()
                .map_err(|_| unsupported("reshape shape conversion failed"))?;
            Ok(t.reshape(shape_arr))
        }
        "permute" => {
            let t = unary(0)?;
            let dims_vec: Vec<isize> = node
                .kwargs
                .get("dims")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_i64())
                        .map(|x| x as isize)
                        .collect()
                })
                .unwrap_or((0..R as isize).collect());
            if dims_vec.len() != R {
                return Err(unsupported("burn engine: permute dims mismatch"));
            }
            let dims_arr: [isize; R] = dims_vec
                .try_into()
                .map_err(|_| unsupported("permute dims conversion failed"))?;
            Ok(t.permute(dims_arr))
        }
        "transpose" => {
            let t = unary(0)?;
            let d0 = kw_isize("d0", 0);
            let d1 = kw_isize("d1", 1);
            let dim0 = if d0 < 0 {
                (R as isize + d0) as usize
            } else {
                d0 as usize
            };
            let dim1 = if d1 < 0 {
                (R as isize + d1) as usize
            } else {
                d1 as usize
            };
            Ok(t.swap_dims(dim0, dim1))
        }
        "t" => {
            let t = unary(0)?;
            if R != 2 {
                return Err(unsupported("burn engine: t() only for 2D"));
            }
            Ok(t.transpose())
        }
        "expand" => {
            let t = unary(0)?;
            let shape_vec: Vec<usize> = node
                .kwargs
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_i64())
                        .map(|x| x as usize)
                        .collect()
                })
                .unwrap_or_default();
            if shape_vec.len() != R {
                return Err(unsupported("burn engine: expand rank mismatch"));
            }
            let shape_arr: [usize; R] = shape_vec
                .try_into()
                .map_err(|_| unsupported("expand shape conversion failed"))?;
            Ok(t.expand(shape_arr))
        }
        "squeeze" | "unsqueeze" => {
            // Implemented via reshape for GPU (rank-preserving only)
            let _ = unary(0)?;
            // For GPU, only handle when rank doesn't change (e.g., squeeze dim of size 1 where shape still R)
            // Otherwise fallback
            return Err(unsupported(
                "burn engine: squeeze/unsqueeze rank change fallback to native",
            ));
        }
        "sum" => {
            let t = unary(0)?;
            if let Some(dim_val) = node.kwargs.get("dim").and_then(|v| v.as_i64()) {
                let dim = if dim_val < 0 {
                    (R as i64 + dim_val) as usize
                } else {
                    dim_val as usize
                };
                if dim >= R {
                    return Err(unsupported("burn engine: sum dim out of range"));
                }
                let keepdim = node
                    .kwargs
                    .get("keepdim")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if keepdim {
                    Ok(t.sum_dim(dim))
                } else {
                    // sum_dim keeps dim, need to squeeze if not keepdim — fallback for now
                    return Err(unsupported(
                        "burn engine: sum without keepdim fallback to native",
                    ));
                }
            } else {
                // sum all -> scalar (rank 0) - fallback
                return Err(unsupported("burn engine: sum all fallback to native"));
            }
        }
        "mean" => {
            let t = unary(0)?;
            if let Some(dim_val) = node.kwargs.get("dim").and_then(|v| v.as_i64()) {
                let dim = if dim_val < 0 {
                    (R as i64 + dim_val) as usize
                } else {
                    dim_val as usize
                };
                if dim >= R {
                    return Err(unsupported("burn engine: mean dim out of range"));
                }
                let keepdim = node
                    .kwargs
                    .get("keepdim")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if keepdim {
                    Ok(t.mean_dim(dim))
                } else {
                    return Err(unsupported(
                        "burn engine: mean without keepdim fallback to native",
                    ));
                }
            } else {
                return Err(unsupported("burn engine: mean all fallback to native"));
            }
        }
        "sin" => {
            let t = unary(0)?;
            // Burn's Tensor has sin via `sin()` if available, else use float_sin via low-level
            // Try high-level first
            Ok(t.sin())
        }
        "cos" => {
            let t = unary(0)?;
            Ok(t.cos())
        }
        "round" => {
            let t = unary(0)?;
            Ok(t.round())
        }

        other => Err(unsupported(&format!(
            "burn engine: unknown target {other:?}"
        ))),
    }
}

/// Convert a Burn `Data<f32, R>` payload into a Rust-owned tensor.
fn owned_from_values(values: Vec<f32>, shape: Vec<i64>) -> OwnedTensor {
    let bytes = values.len() * 4;
    let words = bytes.div_ceil(8);
    let mut data = vec![0u64; words];
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            data.as_mut_ptr() as *mut u8,
            bytes,
        );
    }
    OwnedTensor {
        data,
        shape,
        dtype: DType::F32,
    }
}

/// Which Burn backend should run a payload?
#[derive(Clone, Copy, PartialEq)]
pub enum BurnBackendChoice {
    /// Pure-CPU `NdArray` (works everywhere).
    NdArray,
    /// `Wgpu` GPU (Metal/Vulkan/DX12), with automatic CPU fallback.
    Wgpu,
}

/// Parse `TORCHBURN_ENGINE` and `TORCHBURN_WGPU_BACKEND` into a backend choice.
///
/// * `TORCHBURN_ENGINE=burn`      → NdArray
/// * `TORCHBURN_ENGINE=burn-wgpu` → Wgpu (falls back to CPU if no GPU)
/// * anything else / unset        → NdArray (the original `burn` behaviour)
pub fn backend_choice() -> BurnBackendChoice {
    // TORCHBURN_DEVICE overrides TORCHBURN_ENGINE for device selection.
    #[cfg(feature = "burn-wgpu")]
    {
        if crate::wgpu_backend::force_cpu() {
            return BurnBackendChoice::NdArray;
        }
        if crate::wgpu_backend::force_gpu() {
            return BurnBackendChoice::Wgpu;
        }
    }
    match std::env::var("TORCHBURN_ENGINE").as_deref() {
        Ok("native_cpu") | Ok("cpu") | Ok("burn") => BurnBackendChoice::NdArray,
        Ok("burn-wgpu") | Ok("wgpu") | Ok("burn_gpu") => BurnBackendChoice::Wgpu,
        _ => {
            // Default to GPU first if available on this system
            #[cfg(feature = "burn-wgpu")]
            {
                if crate::wgpu_backend::gpu_available() {
                    return BurnBackendChoice::Wgpu;
                }
            }
            BurnBackendChoice::NdArray
        }
    }
}

/// Execute a payload with the Burn engine, delegating unsupported payloads to
/// the native zero-copy engine (strict superset semantics).
///
/// The backend is selected at runtime: wgpu when `TORCHBURN_ENGINE=burn-wgpu`
/// and a GPU adapter is available, otherwise the pure-CPU ndarray backend.
pub fn execute_plan(
    py: Python<'_>,
    payload: &Payload,
    capsules: &[Bound<'_, PyCapsule>],
) -> PyResult<Vec<Py<PyCapsule>>> {
    let choice = backend_choice();
    // Release GIL during Rust/Burn computation for better concurrency.
    // The DLPack capsule creation at the end requires the GIL.
    let burn_result = match choice {
        BurnBackendChoice::Wgpu => {
            #[cfg(feature = "burn-wgpu")]
            {
                crate::wgpu_backend::init_wgpu_runtime();
                if crate::wgpu_backend::gpu_available() {
                    // wgpu panics (not returns) on OOM, device loss and validation
                    // errors.  Catch the panic so a GPU failure degrades to the
                    // CPU backend instead of killing the Python process.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute_burn_generic::<crate::wgpu_backend::Backend>(py, payload, capsules)
                    }));
                    match result {
                        Ok(Ok(out)) => Ok(out),
                        // GPU unavailable at runtime (adapter lost / device init failed).
                        Ok(Err(err)) if err.to_string().contains("TB_WGPU_UNAVAILABLE") => {
                            eprintln!(
                                "torchburn: wgpu runtime failed, falling back to the ndarray CPU backend"
                            );
                            execute_burn_generic::<NdArray<f32>>(py, payload, capsules)
                        }
                        Ok(Err(err)) => Err(err),
                        Err(_) => {
                            eprintln!(
                                "torchburn: wgpu execution panicked (OOM or device failure), \
                                 falling back to the ndarray CPU backend"
                            );
                            execute_burn_generic::<NdArray<f32>>(py, payload, capsules)
                        }
                    }
                } else {
                    eprintln!(
                        "torchburn: wgpu unavailable on this machine, falling back to the ndarray CPU backend"
                    );
                    execute_burn_generic::<NdArray<f32>>(py, payload, capsules)
                }
            }
            #[cfg(not(feature = "burn-wgpu"))]
            {
                // Feature not compiled in: CPU is the only burn backend.
                execute_burn_generic::<NdArray<f32>>(py, payload, capsules)
            }
        }
        BurnBackendChoice::NdArray => execute_burn_generic::<NdArray<f32>>(py, payload, capsules),
    };
    match burn_result {
        Ok(out) => Ok(out),
        Err(err) if err.to_string().contains("TB_UNSUPPORTED") => {
            // Delegate to the native engine: burn is a superset of native.
            let refs = capsules
                .iter()
                .map(crate::dlpack::capsule_ref)
                .collect::<PyResult<Vec<_>>>()?;
            let native_out = py.allow_threads(|| engine::execute_native(payload, &refs))?;
            let mut out = Vec::with_capacity(native_out.len());
            for owned in native_out {
                out.push(dlpack::owned_to_capsule_owned(py, owned)?);
            }
            Ok(out)
        }
        Err(err) => Err(err),
    }
}

fn execute_burn_generic<B>(
    py: Python<'_>,
    payload: &Payload,
    capsules: &[Bound<'_, PyCapsule>],
) -> PyResult<Vec<Py<PyCapsule>>>
where
    B: BurnBackend<FloatElem = f32>,
{
    let inputs = read_inputs(payload, capsules)?;
    let rank = payload.inputs.first().map(|i| i.shape.len()).unwrap_or(2);
    // Release GIL during the Burn computation (pure Rust, no Python interaction).
    let owned = py.allow_threads(|| match rank {
        1 => run_rank::<B, 1>(payload, inputs),
        2 => run_rank::<B, 2>(payload, inputs),
        3 => run_rank::<B, 3>(payload, inputs),
        4 => run_rank::<B, 4>(payload, inputs),
        r => Err(unsupported(&format!(
            "burn engine supports ranks 1-4, got {r}"
        ))),
    })?;
    let mut out = Vec::with_capacity(owned.len());
    for tensor in owned {
        out.push(dlpack::owned_to_capsule_owned(py, tensor)?);
    }
    Ok(out)
}
