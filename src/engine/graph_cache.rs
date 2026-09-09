//! Prepared-graph cache: prepare_graph / execute_prepared.
//!
//! Parsed payloads are cached under an i64 handle so repeat calls
//! skip JSON parsing and fusion planning. Inherits the engine root
//! imports via super; pure move, no logic changes.

use super::*;
pub(crate) struct PreplannedExecution {
    pub nodes: Vec<Node>,
    pub fp: fusion::FusionPlan,
    pub node_slot: HashMap<u32, usize>,
}

/// A prepared graph: parsed nodes + metadata, ready for execution.
pub(crate) struct PreparedGraph {
    payload: Payload,
    preplanned: Option<PreplannedExecution>,
}

struct GraphCache {
    graphs: HashMap<i64, PreparedGraph>,
    order: std::collections::VecDeque<i64>,
}

impl GraphCache {
    /// Promote a handle to the back of the eviction queue (most recently used).
    /// If the handle is not in the queue, this is a no-op.
    fn touch(&mut self, handle: i64) {
        // Remove from current position and push to back (most recently used).
        // VecDeque::retain is O(n) but n <= 1024, which is negligible.
        let len = self.order.len();
        self.order.retain(|&h| h != handle);
        // Only re-add if it was actually in the queue (i.e., we removed something).
        if self.order.len() < len {
            self.order.push_back(handle);
        }
    }
}

fn graph_cache() -> &'static RwLock<GraphCache> {
    static INSTANCE: OnceLock<RwLock<GraphCache>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        RwLock::new(GraphCache {
            graphs: HashMap::new(),
            order: std::collections::VecDeque::new(),
        })
    })
}

static NEXT_HANDLE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1);

/// Maximum payload JSON size accepted (DoS protection).
pub const MAX_PAYLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Parse a Python dict into a PreparedGraph and cache it. Returns a handle.
pub fn prepare_graph(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<i64> {
    let payload = dict_to_payload(dict)?;
    let base = payload.inputs.len();
    let mut nodes = payload.nodes.clone();
    let mut fp = fusion::plan(&nodes, base);
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
    let preplanned = if !unsafe_output {
        fn is_slot_list_target(t: &str) -> bool {
            matches!(t, "cat" | "stack" | "concat" | "unbind" | "split_with_sizes")
        }
        let mut remap = Vec::with_capacity(base + nodes.len());
        remap.extend(0..base);
        remap.extend((0..nodes.len()).map(|i| base + fp.node_step[i]));
        for node in nodes.iter_mut() {
            let remap_arr = is_slot_list_target(node.target.as_str());
            for arg in node.args.iter_mut() {
                if let Some(s) = arg.index {
                    if s < remap.len() {
                        arg.index = Some(remap[s]);
                    }
                }
                if remap_arr {
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
        }
        fp.remap(&remap);
        let mut node_slot = HashMap::with_capacity(nodes.len());
        for (si, step) in fp.steps.iter().enumerate() {
            let out_slot = base + si;
            for member in step.member_nodes() {
                node_slot.insert(nodes[member].id, out_slot);
            }
        }
        Some(PreplannedExecution {
            nodes,
            fp,
            node_slot,
        })
    } else {
        None
    };

    let handle = NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut cache = graph_cache().write().unwrap_or_else(|e| e.into_inner());
    cache.graphs.insert(
        handle,
        PreparedGraph {
            payload,
            preplanned,
        },
    );
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

pub(crate) fn execute_prepared_native(
    graph: &PreparedGraph,
    capsules: &[CapsuleRef],
) -> PyResult<Vec<OwnedTensor>> {
    if let Some(ref plan) = graph.preplanned {
        let mut slots = init_input_slots(&graph.payload, capsules)?;
        let mut ok = true;
        for step in &plan.fp.steps {
            match execute_step(step, &plan.nodes, &mut slots, capsules) {
                Ok(()) => {}
                Err(e) if e.to_string().contains(fusion::FUSION_SKIP_MARKER) => {
                    ok = false;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        if ok {
            return collect_outputs(&graph.payload, &plan.node_slot, &mut slots);
        }
    }
    execute_native(&graph.payload, capsules)
}

/// Execute a prepared graph with new input tensors.
pub fn execute_prepared(
    py: Python<'_>,
    handle: i64,
    capsules: &[Bound<'_, PyCapsule>],
) -> PyResult<Vec<Py<PyCapsule>>> {
    // LRU promotion: take write lock briefly to promote, then downcast to read lock.
    {
        let mut cache = graph_cache().write().unwrap_or_else(|e| e.into_inner());
        if !cache.graphs.contains_key(&handle) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid graph handle {handle}"
            )));
        }
        cache.touch(handle);
    }

    let cache = graph_cache().read().unwrap_or_else(|e| e.into_inner());
    let graph = cache.graphs.get(&handle).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid graph handle {handle}"))
    })?;

    #[cfg(feature = "burn")]
    if engine_is_burn() {
        return burn_impl::execute_burn(py, &graph.payload, capsules);
    }

    let refs: Vec<CapsuleRef> = capsules
        .iter()
        .map(crate::dlpack::capsule_ref)
        .collect::<PyResult<_>>()?;
    let native_out = py.allow_threads(|| execute_prepared_native(graph, &refs))?;

    let mut out = Vec::with_capacity(native_out.len());
    for owned in native_out {
        out.push(crate::dlpack::owned_to_capsule_owned(py, owned)?);
    }
    Ok(out)
}
