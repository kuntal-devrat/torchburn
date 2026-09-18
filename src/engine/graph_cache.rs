//! Prepared-graph cache: prepare_graph / execute_prepared.
//!
//! Parsed payloads are cached under an i64 handle so repeat calls
//! skip JSON parsing and fusion planning. Inherits the engine root
//! imports via super; pure move, no logic changes.

use super::*;
use std::sync::Arc;

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

struct LruEntry {
    handle: i64,
    graph: Option<Arc<PreparedGraph>>,
    prev: usize,
    next: usize,
}

const NIL: usize = usize::MAX;

struct GraphCache {
    map: HashMap<i64, usize>,
    entries: Vec<LruEntry>,
    free: Vec<usize>,
    head: usize, // LRU (oldest)
    tail: usize, // MRU (newest)
}

impl GraphCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            entries: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    #[inline]
    fn is_mru(&self, handle: i64) -> bool {
        self.tail != NIL && self.entries[self.tail].handle == handle
    }

    #[inline]
    fn get(&self, handle: i64) -> Option<Arc<PreparedGraph>> {
        let &idx = self.map.get(&handle)?;
        self.entries[idx].graph.clone()
    }

    /// Promote a handle to MRU in O(1).
    fn touch(&mut self, handle: i64) {
        let &idx = match self.map.get(&handle) {
            Some(i) => i,
            None => return,
        };
        if idx == self.tail {
            return;
        }

        // Unlink from current position
        let prev = self.entries[idx].prev;
        let next = self.entries[idx].next;
        if prev != NIL {
            self.entries[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.entries[next].prev = prev;
        }

        // Attach to tail (MRU)
        self.entries[idx].prev = self.tail;
        self.entries[idx].next = NIL;
        if self.tail != NIL {
            self.entries[self.tail].next = idx;
        }
        self.tail = idx;
    }

    fn insert(&mut self, handle: i64, graph: Arc<PreparedGraph>) {
        if let Some(&idx) = self.map.get(&handle) {
            self.entries[idx].graph = Some(graph);
            self.touch(handle);
            return;
        }

        let idx = if let Some(i) = self.free.pop() {
            self.entries[i] = LruEntry {
                handle,
                graph: Some(graph),
                prev: self.tail,
                next: NIL,
            };
            i
        } else {
            let i = self.entries.len();
            self.entries.push(LruEntry {
                handle,
                graph: Some(graph),
                prev: self.tail,
                next: NIL,
            });
            i
        };

        if self.tail != NIL {
            self.entries[self.tail].next = idx;
        }
        self.tail = idx;
        if self.head == NIL {
            self.head = idx;
        }
        self.map.insert(handle, idx);

        // Evict LRU (head) if over 1024 prepared graphs
        while self.map.len() > 1024 && self.head != NIL {
            let evict_idx = self.head;
            let evict_handle = self.entries[evict_idx].handle;
            self.remove(evict_handle);
        }
    }

    fn remove(&mut self, handle: i64) {
        let idx = match self.map.remove(&handle) {
            Some(i) => i,
            None => return,
        };

        let prev = self.entries[idx].prev;
        let next = self.entries[idx].next;
        if prev != NIL {
            self.entries[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.entries[next].prev = prev;
        } else {
            self.tail = prev;
        }

        self.entries[idx].graph = None;
        self.free.push(idx);
    }
}

fn graph_cache() -> &'static RwLock<GraphCache> {
    static INSTANCE: OnceLock<RwLock<GraphCache>> = OnceLock::new();
    INSTANCE.get_or_init(|| RwLock::new(GraphCache::new()))
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
            matches!(
                t,
                "cat" | "stack" | "concat" | "unbind" | "split_with_sizes"
            )
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
    cache.insert(
        handle,
        Arc::new(PreparedGraph {
            payload,
            preplanned,
        }),
    );
    Ok(handle)
}

/// Release a prepared graph from the cache.
pub fn release_graph(handle: i64) {
    let mut cache = graph_cache().write().unwrap_or_else(|e| e.into_inner());
    cache.remove(handle);
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
    // Check MRU and fetch Arc<PreparedGraph> under read lock first
    let (graph, needs_touch) = {
        let cache = graph_cache().read().unwrap_or_else(|e| e.into_inner());
        match cache.get(handle) {
            Some(g) => {
                let is_mru = cache.is_mru(handle);
                (g, !is_mru)
            }
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid graph handle {handle}"
                )));
            }
        }
    };

    if needs_touch {
        let mut cache = graph_cache().write().unwrap_or_else(|e| e.into_inner());
        cache.touch(handle);
    }

    #[cfg(feature = "burn")]
    if engine_is_burn() {
        return burn_impl::execute_burn(py, &graph.payload, capsules);
    }

    let refs: Vec<CapsuleRef> = capsules
        .iter()
        .map(crate::dlpack::capsule_ref)
        .collect::<PyResult<_>>()?;
    let native_out = py.allow_threads(|| execute_prepared_native(&graph, &refs))?;

    let mut out = Vec::with_capacity(native_out.len());
    for owned in native_out {
        out.push(crate::dlpack::owned_to_capsule_owned(py, owned)?);
    }
    Ok(out)
}
