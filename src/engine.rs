//! Payload execution engine.
//!
//! The Python side slices a `torch.fx` graph into runs of supported nodes
//! (REQ-002) and sends each run as a compact JSON payload alongside the
//! DLPack capsules backing its tensor inputs. This module walks the payload,
//! dispatches to the native zero-copy kernels (or the optional Burn engine),
//! and returns one output capsule per requested node.

use crate::dlpack::{
    contiguous_strides, dtype_from_spec, unsupported, BorrowedTensor, CapsuleRef, DType,
    OwnedTensor,
};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use serde_json::Value;
use std::collections::HashMap;

use crate::{
    fft_complex, fusion, kernels, linalg, math_ops,
    nn::{activations, attention, convolution, embedding, losses, norm, pooling, upsample},
    quantization, reductions, shape_ops,
};
use std::sync::{OnceLock, RwLock};

// ---------------------------------------------------------------------------
// Global graph cache: prepare_graph / execute_prepared
// ---------------------------------------------------------------------------

use crate::fusion::Step;

pub mod graph_cache;

pub use self::graph_cache::{execute_prepared, prepare_graph, release_graph, MAX_PAYLOAD_BYTES};

pub mod payload;

pub(crate) use self::payload::Slot;
pub use self::payload::{dict_to_payload, supported_targets, ArgRef, Node, Payload};

pub mod helpers;

pub(crate) use self::helpers::{
    arg_index, kw_bool, kw_f64, kw_f64_allow_inf, kw_i64, kw_i64_vec, kw_isize, kw_isize_vec,
    kw_opt_dim, kw_opt_dims, kw_str, kw_usize, slot_view,
};

pub mod dispatch_op;

pub(crate) use self::dispatch_op::dispatch_node;

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
    let parse_output_id = |id: u32| -> (u32, bool, usize) {
        if node_slot.contains_key(&id) {
            (id, false, 0)
        } else if node_slot.contains_key(&(id >> 16)) {
            (id >> 16, true, (id & 0xFFFF) as usize)
        } else if node_slot.contains_key(&(id & 0xFFFF)) {
            (id & 0xFFFF, true, (id >> 16) as usize)
        } else {
            (id, false, 0)
        }
    };

    let mut ref_counts: HashMap<usize, usize> = HashMap::new();
    for id in &payload.outputs {
        let (effective_id, _, _) = parse_output_id(*id);
        if let Some(&idx) = node_slot.get(&effective_id) {
            *ref_counts.entry(idx).or_insert(0) += 1;
        }
    }

    let mut out = Vec::with_capacity(payload.outputs.len());
    for id in &payload.outputs {
        let (effective_id, use_tuple_elem, elem) = parse_output_id(*id);

        let slot_idx = node_slot.get(&effective_id).ok_or_else(|| {
            unsupported(&format!("output references unknown node {effective_id}"))
        })?;
        match &mut slots[*slot_idx] {
            Slot::Owned(t) => {
                if let Some(count) = ref_counts.get_mut(slot_idx) {
                    if *count > 1 {
                        *count -= 1;
                        out.push(t.clone());
                    } else {
                        out.push(std::mem::take(t));
                    }
                } else {
                    out.push(t.clone());
                }
            }
            Slot::View {
                data,
                shape,
                strides,
                dtype,
            } => {
                let borrowed = BorrowedTensor {
                    data: *data,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
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
                crate::memory_pool::recycle_tensor(taken);
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
            let w_data = unsafe {
                std::slice::from_raw_parts(w_view.data as *const f32, w_view.buffer_len())
            };
            let b_data = unsafe {
                std::slice::from_raw_parts(b_view.data as *const f32, b_view.buffer_len())
            };
            let rm_data = unsafe {
                std::slice::from_raw_parts(rm_view.data as *const f32, rm_view.buffer_len())
            };
            let rv_data = unsafe {
                std::slice::from_raw_parts(rv_view.data as *const f32, rv_view.buffer_len())
            };

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
            let in_data = unsafe {
                std::slice::from_raw_parts(conv_view.data as *const f32, conv_view.buffer_len())
            };
            let mut out = OwnedTensor::new(DType::F32, shape.clone());
            let out_data =
                unsafe { std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut f32, total) };
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
    debug_assert_eq!(
        slots.len(),
        out_slot + 1,
        "each step pushes exactly one slot"
    );
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
    let mut fp = if no_fusion {
        fusion::FusionPlan {
            steps: (0..nodes.len()).map(|i| Step::Node(i)).collect(),
            node_step: (0..nodes.len()).collect(),
        }
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
        if let Step::ConvBnRelu(spec) = step {
            if requested.contains(&nodes[spec.conv].id) || requested.contains(&nodes[spec.bn].id) {
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
        fp.remap(&remap);

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
                    return if crate::wgpu::backend::gpu_available() {
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
    // Accept explicit GPU device selection:
    if matches!(
        std::env::var("TORCHBURN_DEVICE").as_deref(),
        Ok("gpu") | Ok("wgpu") | Ok("cuda")
    ) {
        #[cfg(feature = "burn-wgpu")]
        {
            if crate::wgpu::backend::gpu_available() {
                return true;
            }
        }
    }
    // Default: native CPU execution
    false
}

// ---------------------------------------------------------------------------
// Direct dict-to-payload conversion (bypasses JSON serialisation)
// ---------------------------------------------------------------------------

use pyo3::types::PyDict;

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
