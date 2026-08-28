//! Graph-level operator fusion (REQ-004).
//!
//! A payload is a run of supported nodes the Python interpreter sliced out of
//! the FX graph (REQ-002).  Without fusion each node dispatches to its own
//! kernel and materialises an intermediate allocation — the classic
//! "read-write per op" data-bus bottleneck the PRD targets.  This module
//! fuses contiguous runs into *single-pass* kernels:
//!
//! 1. **Elementwise chains** — maximal runs of single-consumer unary/binary
//!    elementwise ops (`add → relu → mul`, `x + bias → sigmoid`, ...)
//!    compile into one kernel that walks the output once, evaluating the
//!    whole expression tree per element with no intermediate tensors.
//! 2. **GEMM epilogues** — `linear`/`addmm` followed by a single-consumer
//!    activation (`linear → relu` in MLP blocks, `linear → gelu` in BERT)
//!    fuse the activation into the matmul's output write.
//!
//! Execution uses a *step* model: after planning, each step produces exactly
//! one output slot (an unfused node, a fused chain, or a fused GEMM), and all
//! references to absorbed nodes are remapped to their group's slot.  The
//! engine therefore executes strictly fewer, larger kernels per call.
//!
//! Correctness contract: fusion is purely an execution-level rewrite of
//! elementwise-identical math with the same broadcasting rules.  If a fused
//! kernel cannot run (dtype/shape guard), it raises `TB_FUSION_SKIP` and the
//! engine falls back to the classic per-node path — fusion never changes
//! observable behaviour.

use crate::dlpack::{DType, OwnedTensor, elem_count, unsupported};
use crate::engine::{ArgRef, Node, Slot};
use pyo3::prelude::*;
use serde_json::Value;

/// Marker error: a fused kernel cannot run; the engine re-executes unfused.
pub const FUSION_SKIP_MARKER: &str = "TB_FUSION_SKIP:";

pub fn fusion_skip(msg: &str) -> PyErr {
    unsupported(&format!("{FUSION_SKIP_MARKER} {msg}"))
}

// ---------------------------------------------------------------------------
// Elementwise op classification
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryKind {
    Relu,
    Abs,
    Neg,
    Sign,
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Reciprocal,
    Ceil,
    Floor,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    LeakyRelu,
    Elu,
    Selu,
    Softplus,
    Hardswish,
    Mish,
    Pow,
    Clamp,
}

impl UnaryKind {
    /// Canonical op name -> unary kind (elementwise, dtype-preserving).
    pub fn from_target(target: &str) -> Option<UnaryKind> {
        match target {
            "relu" => Some(UnaryKind::Relu),
            "abs" => Some(UnaryKind::Abs),
            "neg" => Some(UnaryKind::Neg),
            "sign" => Some(UnaryKind::Sign),
            "sqrt" => Some(UnaryKind::Sqrt),
            "rsqrt" => Some(UnaryKind::Rsqrt),
            "exp" => Some(UnaryKind::Exp),
            "log" => Some(UnaryKind::Log),
            "reciprocal" => Some(UnaryKind::Reciprocal),
            "ceil" => Some(UnaryKind::Ceil),
            "floor" => Some(UnaryKind::Floor),
            "sigmoid" => Some(UnaryKind::Sigmoid),
            "tanh" => Some(UnaryKind::Tanh),
            "gelu" => Some(UnaryKind::Gelu),
            "silu" => Some(UnaryKind::Silu),
            "leaky_relu" => Some(UnaryKind::LeakyRelu),
            "elu" => Some(UnaryKind::Elu),
            "selu" => Some(UnaryKind::Selu),
            "softplus" => Some(UnaryKind::Softplus),
            "hardswish" => Some(UnaryKind::Hardswish),
            "mish" => Some(UnaryKind::Mish),
            "pow" => Some(UnaryKind::Pow),
            "clamp" => Some(UnaryKind::Clamp),
            _ => None,
        }
    }

    /// Kwarg parameter defaults, in `[p0, p1]` form.
    fn defaults(self) -> [f64; 2] {
        match self {
            UnaryKind::LeakyRelu => [0.01, 0.0],
            UnaryKind::Elu => [1.0, 0.0],
            UnaryKind::Softplus => [1.0, 0.0],
            UnaryKind::Pow => [2.0, 0.0],
            UnaryKind::Clamp => [f64::NEG_INFINITY, f64::INFINITY],
            _ => [0.0, 0.0],
        }
    }

    /// Read this op's parameters from its node kwargs (defaults when absent).
    pub fn params(self, kwargs: &std::collections::HashMap<String, Value>) -> [f64; 2] {
        let mut p = self.defaults();
        let f = |key: &str| kwargs.get(key).and_then(|v| v.as_f64());
        match self {
            UnaryKind::LeakyRelu => p[0] = f("negative_slope").unwrap_or(p[0]),
            UnaryKind::Elu => p[0] = f("alpha").unwrap_or(p[0]),
            UnaryKind::Softplus => p[0] = f("beta").unwrap_or(p[0]),
            UnaryKind::Pow => p[0] = f("exp").unwrap_or(p[0]),
            UnaryKind::Clamp => {
                if let Some(v) = f("min") {
                    p[0] = v;
                }
                if let Some(v) = f("max") {
                    p[1] = v;
                }
            }
            _ => {}
        }
        p
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryKind {
    pub fn from_target(target: &str) -> Option<BinaryKind> {
        match target {
            "add" => Some(BinaryKind::Add),
            "sub" => Some(BinaryKind::Sub),
            "mul" => Some(BinaryKind::Mul),
            "div" => Some(BinaryKind::Div),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar plumbing for f32/f64 kernels
// ---------------------------------------------------------------------------

/// Minimal numeric surface the fused kernels need (f32 and f64).
pub trait Fp:
    Copy
    + PartialOrd
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const ZERO: Self;
    fn from_f64(v: f64) -> Self;
    fn fp_exp(self) -> Self;
    fn fp_ln(self) -> Self;
    fn fp_sqrt(self) -> Self;
    fn fp_tanh(self) -> Self;
    fn fp_powf(self, e: Self) -> Self;
    fn fp_abs(self) -> Self;
    fn fp_neg(self) -> Self;
    fn fp_ceil(self) -> Self;
    fn fp_floor(self) -> Self;
}

impl Fp for f32 {
    const ZERO: f32 = 0.0;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn fp_exp(self) -> Self {
        self.exp()
    }
    fn fp_ln(self) -> Self {
        self.ln()
    }
    fn fp_sqrt(self) -> Self {
        self.sqrt()
    }
    fn fp_tanh(self) -> Self {
        self.tanh()
    }
    fn fp_powf(self, e: Self) -> Self {
        self.powf(e)
    }
    fn fp_abs(self) -> Self {
        self.abs()
    }
    fn fp_neg(self) -> Self {
        -self
    }
    fn fp_ceil(self) -> Self {
        self.ceil()
    }
    fn fp_floor(self) -> Self {
        self.floor()
    }
}

impl Fp for f64 {
    const ZERO: f64 = 0.0;
    fn from_f64(v: f64) -> Self {
        v
    }
    fn fp_exp(self) -> Self {
        self.exp()
    }
    fn fp_ln(self) -> Self {
        self.ln()
    }
    fn fp_sqrt(self) -> Self {
        self.sqrt()
    }
    fn fp_tanh(self) -> Self {
        self.tanh()
    }
    fn fp_powf(self, e: Self) -> Self {
        self.powf(e)
    }
    fn fp_abs(self) -> Self {
        self.abs()
    }
    fn fp_neg(self) -> Self {
        -self
    }
    fn fp_ceil(self) -> Self {
        self.ceil()
    }
    fn fp_floor(self) -> Self {
        self.floor()
    }
}

/// Apply a unary elementwise op.  `p` carries op parameters ([p0, p1]).
pub fn apply_unary<T: Fp>(k: UnaryKind, x: T, p: [f64; 2]) -> T {
    let one = T::from_f64(1.0);
    match k {
        UnaryKind::Relu => {
            if x > T::ZERO {
                x
            } else {
                T::ZERO
            }
        }
        UnaryKind::Abs => x.fp_abs(),
        UnaryKind::Neg => x.fp_neg(),
        UnaryKind::Sign => {
            if x > T::ZERO {
                one
            } else if x < T::ZERO {
                one.fp_neg()
            } else {
                T::ZERO
            }
        }
        UnaryKind::Sqrt => x.fp_sqrt(),
        UnaryKind::Rsqrt => one / x.fp_sqrt(),
        UnaryKind::Exp => x.fp_exp(),
        UnaryKind::Log => x.fp_ln(),
        UnaryKind::Reciprocal => one / x,
        UnaryKind::Ceil => x.fp_ceil(),
        UnaryKind::Floor => x.fp_floor(),
        UnaryKind::Sigmoid => one / (one + x.fp_neg().fp_exp()),
        UnaryKind::Tanh => x.fp_tanh(),
        // gelu (tanh approximation — matches torch's approximate="tanh" and
        // the native engine's activation kernel).
        UnaryKind::Gelu => {
            let c = T::from_f64(0.7978845608028654);
            let b = T::from_f64(0.044715);
            let inner = c * (x + b * x.fp_powf(T::from_f64(3.0)));
            T::from_f64(0.5) * x * (one + inner.fp_tanh())
        }
        UnaryKind::Silu => x * (one / (one + x.fp_neg().fp_exp())),
        UnaryKind::LeakyRelu => {
            if x > T::ZERO {
                x
            } else {
                x * T::from_f64(p[0])
            }
        }
        UnaryKind::Elu => {
            if x > T::ZERO {
                x
            } else {
                T::from_f64(p[0]) * (x.fp_exp() - one)
            }
        }
        UnaryKind::Selu => {
            // torch's SELU constants (matches the native kernel).
            let alpha = T::from_f64(1.6732632423543772);
            let scale = T::from_f64(1.0507009873554805);
            if x > T::ZERO {
                scale * x
            } else {
                scale * alpha * (x.fp_exp() - one)
            }
        }
        UnaryKind::Softplus => {
            let beta = T::from_f64(p[0]);
            (one / beta) * (one + (beta * x).fp_exp()).fp_ln()
        }
        UnaryKind::Hardswish => {
            x * clamp_scalar(x + T::from_f64(3.0), T::ZERO, T::from_f64(6.0)) / T::from_f64(6.0)
        }
        UnaryKind::Mish => x * (one + x.fp_exp()).fp_ln().fp_tanh(),
        UnaryKind::Pow => x.fp_powf(T::from_f64(p[0])),
        UnaryKind::Clamp => clamp_scalar(x, T::from_f64(p[0]), T::from_f64(p[1])),
    }
}

fn clamp_scalar<T: Fp>(x: T, lo: T, hi: T) -> T {
    let mut v = x;
    if v < lo {
        v = lo;
    }
    if v > hi {
        v = hi;
    }
    v
}

/// Concrete f32/f64 entry points (used by the engine's fused GEMM epilogue).
pub fn apply_unary_f32(k: UnaryKind, x: f32, p: [f64; 2]) -> f32 {
    apply_unary(k, x, p)
}
pub fn apply_unary_f64(k: UnaryKind, x: f64, p: [f64; 2]) -> f64 {
    apply_unary(k, x, p)
}

fn apply_binary<T: Fp>(k: BinaryKind, a: T, b: T) -> T {
    match k {
        BinaryKind::Add => a + b,
        BinaryKind::Sub => a - b,
        BinaryKind::Mul => a * b,
        BinaryKind::Div => a / b,
    }
}

// ---------------------------------------------------------------------------
// Fusion planning
// ---------------------------------------------------------------------------

/// One argument of a fused chain expression: a runtime slot (leaf) or the
/// output of an earlier chain expression.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arg {
    Leaf(usize),
    Chain(usize),
}

#[derive(Clone, Copy, Debug)]
pub enum ChainOp {
    Unary(UnaryKind),
    Binary(BinaryKind),
}

/// One expression node of a fused elementwise chain.
#[derive(Clone, Debug)]
pub struct ChainExpr {
    pub op: ChainOp,
    pub a: Arg,
    pub b: Option<Arg>,
    /// Payload node index (used to read op kwargs like `negative_slope`).
    pub node: usize,
}

/// A compiled elementwise chain: node indices in payload order + the
/// expression DAG (evaluated bottom-up, one value per element).
#[derive(Clone, Debug)]
pub struct ChainPlan {
    pub nodes: Vec<usize>,
    pub exprs: Vec<ChainExpr>,
}

/// An activation fused into a GEMM's output write.
#[derive(Clone, Copy, Debug)]
pub struct ActSpec {
    pub kind: UnaryKind,
    pub params: [f64; 2],
}

/// Precomputed fused BN parameters (per-channel scale and bias) for
/// inference-time conv+bn fusion.
#[derive(Clone, Debug)]
pub struct ConvBnReluSpec {
    /// Payload index of the conv2d node.
    pub conv: usize,
    /// Payload index of the batch_norm node.
    pub bn: usize,
    /// Payload index of the relu node.
    pub relu: usize,
    /// Precomputed fused_weight[c] = bn_weight[c] / sqrt(running_var[c] + eps).
    pub fused_scale: Vec<f32>,
    /// Precomputed fused_bias[c] = bn_bias[c] - running_mean[c] * fused_scale[c].
    pub fused_bias: Vec<f32>,
    /// Epsilon used in BN.
    pub eps: f32,
}

#[derive(Clone, Debug)]
pub enum Step {
    /// Single unfused node (payload index).
    Node(usize),
    /// Fused elementwise chain.
    Chain(ChainPlan),
    /// `linear`/`addmm` node with a single-consumer activation absorbed.
    Gemm { linear: usize, act: usize, spec: ActSpec },
    /// `conv2d → batch_norm(training=false) → relu` fused into a single pass.
    ConvBnRelu(ConvBnReluSpec),
}

impl Step {
    pub(crate) fn member_nodes(&self) -> Vec<usize> {
        match self {
            Step::Node(i) => vec![*i],
            Step::Chain(p) => p.nodes.clone(),
            Step::Gemm { linear, act, .. } => vec![*linear, *act],
            Step::ConvBnRelu(spec) => vec![spec.conv, spec.bn, spec.relu],
        }
    }
}

/// The full execution plan for a payload: steps in payload order plus, for
/// every payload node, the step index whose output slot materialises it.
pub struct FusionPlan {
    pub steps: Vec<Step>,
    pub node_step: Vec<usize>,
}

fn is_fusable_unary(target: &str) -> bool {
    UnaryKind::from_target(target).is_some()
}

fn is_fusable_binary(target: &str) -> bool {
    BinaryKind::from_target(target).is_some()
}

fn is_fusable_node(node: &Node) -> bool {
    is_fusable_unary(&node.target) || is_fusable_binary(&node.target)
}

/// Count, per payload node, how many *later* nodes reference its output slot.
fn consumer_counts(nodes: &[Node], base: usize) -> Vec<usize> {
    let n = nodes.len();
    let mut counts = vec![0usize; n];
    for node in nodes {
        for arg in &node.args {
            if let Some(s) = arg.index {
                if s >= base && s - base < n {
                    counts[s - base] += 1;
                }
            }
            if let Some(Value::Array(arr)) = &arg.value {
                for v in arr {
                    if let Some(u) = v.as_u64() {
                        let s = u as usize;
                        if s >= base && s - base < n {
                            counts[s - base] += 1;
                        }
                    }
                }
            }
        }
    }
    counts
}

/// The single consumer of node `i`, if any (a later node referencing its slot).
fn single_consumer(nodes: &[Node], i: usize, base: usize) -> Option<usize> {
    let mut found = None;
    for (j, node) in nodes.iter().enumerate().skip(i + 1) {
        let mut refs = Vec::new();
        for arg in &node.args {
            if let Some(s) = arg.index {
                refs.push(s);
            }
            if let Some(Value::Array(arr)) = &arg.value {
                refs.extend(arr.iter().filter_map(|v| v.as_u64().map(|u| u as usize)));
            }
        }
        for s in refs {
            if s >= base && s - base == i {
                if found.is_some() {
                    return None; // more than one consumer
                }
                found = Some(j);
            }
        }
    }
    found
}

/// Build the fused execution plan for a payload's node list.
///
/// `nodes` are the payload nodes (slot indices: inputs are `0..base`, node
/// outputs are `base + node_position`).  The returned plan owns its steps;
/// the caller (engine) is responsible for executing them and remapping slots.
pub fn plan(nodes: &[Node], base: usize) -> FusionPlan {
    let n = nodes.len();
    let mut node_step = vec![0usize; n];
    let mut steps: Vec<Step> = Vec::new();
    let consumers = consumer_counts(nodes, base);
    let mut i = 0usize;
    while i < n {
        let node = &nodes[i];
        // Conv+BN+ReLU fusion: conv2d → batch_norm(training=false) → relu
        // DISABLED: currently produces incorrect output (all 1s) — needs debug.
        // The BN parameters (running_mean, running_var, weight, bias) are
        // precomputed into fused per-channel scale/bias at plan time, so the
        // runtime kernel is a single multiply+add+relu per output element.
        if false && node.target == "conv2d"
            && i + 2 < n
            && nodes[i + 1].target == "batch_norm"
            && nodes[i + 2].target == "relu"
            && consumers[i] == 1
            && consumers[i + 1] == 1
            && single_consumer(nodes, i, base) == Some(i + 1)
            && single_consumer(nodes, i + 1, base) == Some(i + 2)
        {
            // Check that BN is in inference mode (training=false).
            let bn_node = &nodes[i + 1];
            let training = bn_node.kwargs.get("training")
                .and_then(|v| v.as_bool()).unwrap_or(false);
            if !training {
                // Try to precompute fused BN scale/bias from the plan's
                // constant slots.  The BN node args are:
                //   [input, weight, bias, running_mean, running_var]
                // We need weight, bias, running_mean, running_var — these are
                // typically input slots (model params), not intra-graph nodes.
                if let Some(spec) = build_conv_bn_relu_spec(nodes, i, i + 1, i + 2, base) {
                    steps.push(Step::ConvBnRelu(spec));
                    node_step[i] = steps.len() - 1;
                    node_step[i + 1] = steps.len() - 1;
                    node_step[i + 2] = steps.len() - 1;
                    i += 3;
                    continue;
                }
            }
        }
        // GEMM epilogue: linear/addmm whose output is consumed ONLY by an
        // adjacent unary elementwise node -> fuse the activation into the
        // matmul output write.
        if (node.target == "linear" || node.target == "addmm")
            && consumers[i] == 1
            && i + 1 < n
            && is_fusable_unary(&nodes[i + 1].target)
            && single_consumer(nodes, i, base) == Some(i + 1)
        {
            let act = &nodes[i + 1];
            let kind = UnaryKind::from_target(&act.target).expect("fusable unary target already validated");
            let spec = ActSpec {
                kind,
                params: kind.params(&act.kwargs),
            };
            steps.push(Step::Gemm { linear: i, act: i + 1, spec });
            node_step[i] = steps.len() - 1;
            node_step[i + 1] = steps.len() - 1;
            i += 2;
            continue;
        }
        // Elementwise chain: grow while the node is fusable, has exactly one
        // consumer, and that consumer is the adjacent node and also fusable.
        if is_fusable_node(node) {
            let mut chain = vec![i];
            let mut j = i;
            while j + 1 < n
                && is_fusable_node(&nodes[j])
                && consumers[j] == 1
                && single_consumer(nodes, j, base) == Some(j + 1)
                && is_fusable_node(&nodes[j + 1])
            {
                j += 1;
                chain.push(j);
            }
            if chain.len() >= 2 {
                match build_chain_exprs(nodes, &chain, base) {
                    Ok(exprs) => {
                        let cplan = ChainPlan { nodes: chain.clone(), exprs };
                        steps.push(Step::Chain(cplan));
                        for &m in &chain {
                            node_step[m] = steps.len() - 1;
                        }
                        i = j + 1;
                        continue;
                    }
                    // Malformed chain (missing args): fall back to unfused.
                    Err(_) => {}
                }
            }
        }
        steps.push(Step::Node(i));
        node_step[i] = steps.len() - 1;
        i += 1;
    }
    FusionPlan { steps, node_step }
}

/// Build the expression DAG for a chain.
///
/// Chain nodes are consecutive payload indices; node `chain[k]`'s arguments
/// reference either runtime slots (leaves) or earlier chain members.  A
/// reference to chain member m (< k) becomes `Arg::Chain(m)`; anything else
/// is a leaf slot.
fn build_chain_exprs(nodes: &[Node], chain: &[usize], base: usize) -> PyResult<Vec<ChainExpr>> {
    let mut exprs = Vec::with_capacity(chain.len());
    for (k, &idx) in chain.iter().enumerate() {
        let node = &nodes[idx];
        let classify = |arg: &ArgRef| -> Arg {
            match arg.index {
                Some(s) if s >= base => {
                    let member = s - base;
                    match chain.iter().position(|&m| m == member) {
                        Some(mk) if mk < k => Arg::Chain(mk),
                        _ => Arg::Leaf(s),
                    }
                }
                Some(s) => Arg::Leaf(s),
                None => Arg::Leaf(usize::MAX), // missing arg — rejected below
            }
        };
        let (op, a, b) = if let Some(u) = UnaryKind::from_target(&node.target) {
            match node.args.get(0) {
                Some(arg) => (ChainOp::Unary(u), classify(arg), None),
                None => return Err(fusion_skip("chain unary missing its argument")),
            }
        } else if let Some(bk) = BinaryKind::from_target(&node.target) {
            match (node.args.get(0), node.args.get(1)) {
                (Some(aa), Some(bb)) => (
                    ChainOp::Binary(bk),
                    classify(aa),
                    Some(classify(bb)),
                ),
                _ => return Err(fusion_skip("chain binary missing its arguments")),
            }
        } else {
            return Err(fusion_skip("chain member is not elementwise"));
        };
        exprs.push(ChainExpr { op, a, b, node: idx });
    }
    Ok(exprs)
}

// ---------------------------------------------------------------------------
// Conv + BN + ReLU fusion
// ---------------------------------------------------------------------------

/// Try to build a fused conv→bn→relu spec by reading the BN parameters
/// from the plan's argument slots.  Returns `None` if the BN params are
/// intra-graph nodes (not constant/input slots) — the planner falls back
/// to unfused execution in that case.
fn build_conv_bn_relu_spec(
    nodes: &[Node],
    conv_idx: usize,
    bn_idx: usize,
    _relu_idx: usize,
    base: usize,
) -> Option<ConvBnReluSpec> {
    let bn = &nodes[bn_idx];
    // BN args: [input, weight, bias, running_mean, running_var]
    // We need weight (1), bias (2), running_mean (3), running_var (4)
    // as constant input slots whose values we can read at plan time.
    // In practice these are model parameters — input slots (indices < base).
    let get_slot = |pos: usize| -> Option<usize> {
        bn.args.get(pos).and_then(|a| a.index)
    };

    let weight_slot = get_slot(1)?;
    let bias_slot = get_slot(2)?;
    let rm_slot = get_slot(3)?;
    let rv_slot = get_slot(4)?;

    // All four must be input slots (indices < base) — they are model
    // parameters that don't change between calls.
    if weight_slot >= base || bias_slot >= base || rm_slot >= base || rv_slot >= base {
        return None;
    }

    let eps = bn.kwargs.get("eps")
        .and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;

    // We can't read the actual tensor values at plan time (they live in
    // DLPack capsules that only exist at runtime).  Instead, store the
    // slot indices and precompute at first execution.  For simplicity,
    // we store placeholder vectors and fill them at runtime.
    //
    // However, since the fusion planner runs at prepare_graph time (which
    // is init time), and the actual values are only available at execute
    // time, we store the slot indices and do the fusion in the engine.
    //
    // Actually, the cleanest approach: store the slot indices in the spec
    // and do the per-channel scale/bias computation at execution time
    // (it's O(C) — negligible vs the O(N*C*H*W) conv compute).
    Some(ConvBnReluSpec {
        conv: conv_idx,
        bn: bn_idx,
        relu: _relu_idx,
        fused_scale: Vec::new(), // filled at runtime
        fused_bias: Vec::new(),  // filled at runtime
        eps,
    })
}

// ---------------------------------------------------------------------------
// Fused elementwise chain kernel
// ---------------------------------------------------------------------------

/// How a leaf's flat index maps from the fused output's flat index.
enum LeafMap {
    /// Same shape: index == output index.
    Identity,
    /// Single-element tensor (scalar broadcast): index == 0.
    Scalar,
    /// General broadcasting: per output dim, the leaf's stride (0 where the
    /// leaf broadcasts).  Resolved with a per-element coordinate walk.
    General { strides: Vec<usize> },
}

struct LeafInfo {
    slot: usize,
    map: LeafMap,
}

/// Broadcast strides of a leaf shape relative to an output shape (right-aligned).
fn broadcast_strides(out_shape: &[i64], leaf_shape: &[i64]) -> Option<Vec<usize>> {
    let rank = out_shape.len();
    let lrank = leaf_shape.len();
    let mut strides = vec![0usize; rank];
    let mut acc = 1usize;
    let mut leaf_strides = vec![0usize; lrank];
    for d in (0..lrank).rev() {
        leaf_strides[d] = acc;
        acc *= leaf_shape[d].max(0) as usize;
    }
    for d in 0..rank {
        let ld = d + lrank - rank;
        if ld < lrank {
            let ldim = leaf_shape[ld].max(0) as usize;
            let odim = out_shape[d].max(0) as usize;
            if ldim == odim {
                strides[d] = leaf_strides[ld];
            } else if ldim == 1 {
                strides[d] = 0; // broadcast
            } else {
                return None; // incompatible (ops validated this earlier)
            }
        } else {
            strides[d] = 0; // absent leading dim -> broadcast
        }
    }
    Some(strides)
}

/// Resolved per-expression argument: which leaf (index into `leaves`) or
/// which earlier chain expression's value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RArg {
    Chain(usize),
    Leaf(usize),
}

struct RExpr {
    op: ChainOp,
    a: RArg,
    b: Option<RArg>,
    params: [f64; 2],
}

/// Decompose flat index `f` into row-major coordinates for `out_shape`.
fn coords_of(f: usize, out_shape: &[i64], coords: &mut [usize]) {
    let mut rem = f;
    for d in (0..out_shape.len()).rev() {
        let dim = out_shape[d].max(1) as usize;
        coords[d] = rem % dim;
        rem /= dim;
    }
}

/// Leaf value at output element `f`.
#[inline]
fn leaf_at<T: Fp>(leaves: &[LeafInfo], data: &[&[T]], li: usize, f: usize, coords: &[usize]) -> T {
    let idx = match &leaves[li].map {
        LeafMap::Identity => f,
        LeafMap::Scalar => 0,
        LeafMap::General { strides } => {
            let mut idx = 0usize;
            for d in 0..strides.len() {
                idx += coords[d] * strides[d];
            }
            idx
        }
    };
    data[li][idx]
}

/// Elements per parallel work chunk; keeps chunks within 32 KB L1 data cache
/// while splitting large tensors across cores.
const CHAIN_PAR_CHUNK: usize = 4 * 1024;
/// Number of elements above which the fused kernel parallelises.
const CHAIN_PAR_THRESHOLD: usize = 16 * 1024;

/// Execute a fused elementwise chain: one pass over the output, evaluating
/// the whole expression tree per element.  Returns `TB_FUSION_SKIP` when the
/// chain cannot run fused (dtype/broadcast guard) so the engine falls back.
pub fn run_chain(
    plan: &ChainPlan,
    nodes: &[Node],
    slots: &[Slot],
    capsules: &[crate::dlpack::CapsuleRef],
) -> PyResult<OwnedTensor> {
    // 1) Collect unique leaf slots (first-use order) and resolve shapes/dtypes.
    let mut leaf_slots: Vec<usize> = Vec::new();
    let mut leaf_shapes: Vec<Vec<i64>> = Vec::new();
    let mut leaf_dtypes: Vec<DType> = Vec::new();
    for expr in &plan.exprs {
        let mut args = vec![expr.a];
        if let Some(b) = expr.b {
            args.push(b);
        }
        for arg in args {
            if let Arg::Leaf(slot) = arg {
                if !leaf_slots.contains(&slot) {
                    let view = crate::engine::slot_view(slots, capsules, slot)?;
                    leaf_slots.push(slot);
                    leaf_shapes.push(view.shape.clone());
                    leaf_dtypes.push(view.dtype);
                }
            }
        }
    }
    if leaf_slots.is_empty() {
        return Err(fusion_skip("chain has no leaf inputs"));
    }
    let dtype = leaf_dtypes[0];
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(fusion_skip("chain leaves are not f32/f64"));
    }
    if leaf_dtypes.iter().any(|&d| d != dtype) {
        return Err(fusion_skip("chain leaves have mixed dtypes"));
    }

    // 2) Propagate shapes bottom-up to get the output shape.
    let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(plan.exprs.len());
    let shape_of = |arg: Arg, shapes: &[Vec<i64>]| -> PyResult<Vec<i64>> {
        match arg {
            Arg::Chain(m) => Ok(shapes[m].clone()),
            Arg::Leaf(slot) => {
                let li = leaf_slots.iter().position(|&s| s == slot)
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("fusion: slot {} not in leaf_slots", slot)))?;
                Ok(leaf_shapes[li].clone())
            }
        }
    };
    for e in &plan.exprs {
        let s = match e.op {
            ChainOp::Unary(_) => shape_of(e.a, &shapes)?,
            ChainOp::Binary(_) => {
                let sa = shape_of(e.a, &shapes)?;
                let sb = shape_of(e.b.expect("binary op has second operand"), &shapes)?;
                crate::ops::broadcast_shape(&sa, &sb)?
            }
        };
        shapes.push(s);
    }
    let out_shape = shapes[plan.exprs.len() - 1].clone();
    let out_n = elem_count(&out_shape);

    // 3) Broadcast maps for each leaf against the output shape.
    let mut leaves: Vec<LeafInfo> = Vec::with_capacity(leaf_slots.len());
    for (slot, shape) in leaf_slots.iter().zip(leaf_shapes.iter()) {
        let map = if *shape == out_shape {
            LeafMap::Identity
        } else if elem_count(shape) == 1 {
            LeafMap::Scalar
        } else {
            match broadcast_strides(&out_shape, shape) {
                Some(strides) => LeafMap::General { strides },
                None => return Err(fusion_skip("chain leaf not broadcast-compatible")),
            }
        };
        leaves.push(LeafInfo { slot: *slot, map });
    }

    // 4) Resolved expressions (leaf indices + params baked in).
    let rexprs: Vec<RExpr> = plan
        .exprs
        .iter()
        .map(|e| {
            let rarg = |a: Arg| -> RArg {
                match a {
                    Arg::Chain(m) => RArg::Chain(m),
                    Arg::Leaf(slot) => {
                        let li = leaves.iter().position(|l| l.slot == slot)
                            .expect("fusion: slot not in leaves");
                        RArg::Leaf(li)
                    }
                }
            };
            let params = match e.op {
                ChainOp::Unary(u) => u.params(&nodes[e.node].kwargs),
                ChainOp::Binary(_) => [0.0, 0.0],
            };
            RExpr {
                op: e.op,
                a: rarg(e.a),
                b: e.b.map(rarg),
                params,
            }
        })
        .collect();

    let out_shape_rt = out_shape.clone();
    let mut out = OwnedTensor::new(dtype, out_shape);
    match dtype {
        DType::F32 => run_chain_typed::<f32>(&rexprs, &leaves, slots, capsules, &mut out, out_n, &out_shape_rt)?,
        DType::F64 => run_chain_typed::<f64>(&rexprs, &leaves, slots, capsules, &mut out, out_n, &out_shape_rt)?,
        _ => unreachable!(),
    }
    Ok(out)
}

#[inline(always)]
fn eval_unary_slice<T: Fp>(u: UnaryKind, src: &[T], dst: &mut [T], p: [f64; 2]) {
    match u {
        UnaryKind::Relu => {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = if *s > T::ZERO { *s } else { T::ZERO };
            }
        }
        UnaryKind::Abs => {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.fp_abs();
            }
        }
        UnaryKind::Neg => {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.fp_neg();
            }
        }
        UnaryKind::Sigmoid => {
            let one = T::from_f64(1.0);
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = one / (one + s.fp_neg().fp_exp());
            }
        }
        UnaryKind::Tanh => {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.fp_tanh();
            }
        }
        _ => {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = apply_unary(u, *s, p);
            }
        }
    }
}

#[inline(always)]
fn eval_binary_slices<T: Fp>(b: BinaryKind, src_a: &[T], src_b: &[T], dst: &mut [T]) {
    match b {
        BinaryKind::Add => {
            for ((a, b), d) in src_a.iter().zip(src_b.iter()).zip(dst.iter_mut()) {
                *d = *a + *b;
            }
        }
        BinaryKind::Sub => {
            for ((a, b), d) in src_a.iter().zip(src_b.iter()).zip(dst.iter_mut()) {
                *d = *a - *b;
            }
        }
        BinaryKind::Mul => {
            for ((a, b), d) in src_a.iter().zip(src_b.iter()).zip(dst.iter_mut()) {
                *d = *a * *b;
            }
        }
        BinaryKind::Div => {
            for ((a, b), d) in src_a.iter().zip(src_b.iter()).zip(dst.iter_mut()) {
                *d = *a / *b;
            }
        }
    }
}

#[inline(always)]
fn eval_binary_scalar_rhs<T: Fp>(b: BinaryKind, src_a: &[T], scalar: T, dst: &mut [T]) {
    match b {
        BinaryKind::Add => {
            for (a, d) in src_a.iter().zip(dst.iter_mut()) {
                *d = *a + scalar;
            }
        }
        BinaryKind::Sub => {
            for (a, d) in src_a.iter().zip(dst.iter_mut()) {
                *d = *a - scalar;
            }
        }
        BinaryKind::Mul => {
            for (a, d) in src_a.iter().zip(dst.iter_mut()) {
                *d = *a * scalar;
            }
        }
        BinaryKind::Div => {
            for (a, d) in src_a.iter().zip(dst.iter_mut()) {
                *d = *a / scalar;
            }
        }
    }
}

#[inline(always)]
fn eval_binary_scalar_lhs<T: Fp>(b: BinaryKind, scalar: T, src_b: &[T], dst: &mut [T]) {
    match b {
        BinaryKind::Add => {
            for (b_val, d) in src_b.iter().zip(dst.iter_mut()) {
                *d = scalar + *b_val;
            }
        }
        BinaryKind::Sub => {
            for (b_val, d) in src_b.iter().zip(dst.iter_mut()) {
                *d = scalar - *b_val;
            }
        }
        BinaryKind::Mul => {
            for (b_val, d) in src_b.iter().zip(dst.iter_mut()) {
                *d = scalar * *b_val;
            }
        }
        BinaryKind::Div => {
            for (b_val, d) in src_b.iter().zip(dst.iter_mut()) {
                *d = scalar / *b_val;
            }
        }
    }
}

fn run_chunk_vectorized<T: Fp>(
    rexprs: &[RExpr],
    leaves: &[LeafInfo],
    leaf_data: &[&[T]],
    start: usize,
    chunk: &mut [T],
    scratch: &mut [Vec<T>],
) {
    let len = chunk.len();
    for (k, e) in rexprs.iter().enumerate() {
        let is_last = k == rexprs.len() - 1;
        let (prior, rest) = scratch.split_at_mut(k);
        let dst: &mut [T] = if is_last {
            chunk
        } else {
            &mut rest[0][..len]
        };

        match e.op {
            ChainOp::Unary(u) => {
                let src: &[T] = match e.a {
                    RArg::Chain(m) => &prior[m][..len],
                    RArg::Leaf(li) => match leaves[li].map {
                        LeafMap::Identity => &leaf_data[li][start..start + len],
                        LeafMap::Scalar => {
                            let scalar = leaf_data[li][0];
                            let val = apply_unary(u, scalar, e.params);
                            dst.fill(val);
                            continue;
                        }
                        _ => unreachable!(),
                    },
                };
                eval_unary_slice(u, src, dst, e.params);
            }
            ChainOp::Binary(b) => {
                let (src_a_slice, scalar_a) = match e.a {
                    RArg::Chain(m) => (Some(&prior[m][..len]), None),
                    RArg::Leaf(li) => match leaves[li].map {
                        LeafMap::Identity => (Some(&leaf_data[li][start..start + len]), None),
                        LeafMap::Scalar => (None, Some(leaf_data[li][0])),
                        _ => unreachable!(),
                    },
                };
                let (src_b_slice, scalar_b) = match e.b.expect("binary has second operand") {
                    RArg::Chain(m) => (Some(&prior[m][..len]), None),
                    RArg::Leaf(li) => match leaves[li].map {
                        LeafMap::Identity => (Some(&leaf_data[li][start..start + len]), None),
                        LeafMap::Scalar => (None, Some(leaf_data[li][0])),
                        _ => unreachable!(),
                    },
                };

                if let (Some(sa), Some(sb)) = (src_a_slice, src_b_slice) {
                    eval_binary_slices(b, sa, sb, dst);
                } else if let (Some(sa), Some(sc_b)) = (src_a_slice, scalar_b) {
                    eval_binary_scalar_rhs(b, sa, sc_b, dst);
                } else if let (Some(sc_a), Some(sb)) = (scalar_a, src_b_slice) {
                    eval_binary_scalar_lhs(b, sc_a, sb, dst);
                } else if let (Some(sc_a), Some(sc_b)) = (scalar_a, scalar_b) {
                    let res = apply_binary(b, sc_a, sc_b);
                    dst.fill(res);
                }
            }
        }
    }
}

fn run_chain_typed<T: Fp>(
    rexprs: &[RExpr],
    leaves: &[LeafInfo],
    slots: &[Slot],
    capsules: &[crate::dlpack::CapsuleRef],
    out: &mut OwnedTensor,
    out_n: usize,
    out_shape: &[i64],
) -> PyResult<()> {
    // Resolve leaf buffers (all validated T-typed in `run_chain`).
    let mut leaf_data: Vec<&[T]> = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        let view = crate::engine::slot_view(slots, capsules, leaf.slot)?;
        // SAFETY: `run_chain` verified every leaf is T; buffers are alive for
        // the whole execute_native call.
        leaf_data.push(unsafe { std::slice::from_raw_parts(view.data as *const T, view.buffer_len()) });
    }
    let out_data = unsafe {
        std::slice::from_raw_parts_mut(out.data.as_mut_ptr() as *mut T, out_n)
    };
    let needs_coords = leaves
        .iter()
        .any(|l| matches!(l.map, LeafMap::General { .. }));

    // Fast cache-tiled SIMD path for contiguous inputs (Identity/Scalar)
    if !needs_coords {
        if out_n >= CHAIN_PAR_THRESHOLD {
            use rayon::prelude::*;
            out_data
                .par_chunks_mut(CHAIN_PAR_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * CHAIN_PAR_CHUNK;
                    let mut scratch: Vec<Vec<T>> = (0..rexprs.len())
                        .map(|_| vec![T::ZERO; chunk.len()])
                        .collect();
                    run_chunk_vectorized(rexprs, leaves, &leaf_data, start, chunk, &mut scratch);
                });
        } else {
            let mut scratch: Vec<Vec<T>> = (0..rexprs.len())
                .map(|_| vec![T::ZERO; out_data.len()])
                .collect();
            run_chunk_vectorized(rexprs, leaves, &leaf_data, 0, out_data, &mut scratch);
        }
        return Ok(());
    }

    // Fallback: general strided broadcast path with per-element coordinates
    if out_n >= CHAIN_PAR_THRESHOLD {
        use rayon::prelude::*;
        out_data
            .par_chunks_mut(CHAIN_PAR_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let start = ci * CHAIN_PAR_CHUNK;
                let mut vals = vec![T::ZERO; rexprs.len()];
                let mut coords = vec![0usize; out_shape.len()];
                for (off, o) in chunk.iter_mut().enumerate() {
                    let f = start + off;
                    coords_of(f, out_shape, &mut coords);
                    for (k, e) in rexprs.iter().enumerate() {
                        let av = match e.a {
                            RArg::Chain(m) => vals[m],
                            RArg::Leaf(li) => leaf_at(leaves, &leaf_data, li, f, &coords),
                        };
                        let v = match e.op {
                            ChainOp::Unary(u) => apply_unary(u, av, e.params),
                            ChainOp::Binary(b) => {
                                let bv = match e.b.expect("binary op has second operand") {
                                    RArg::Chain(m) => vals[m],
                                    RArg::Leaf(li) => leaf_at(leaves, &leaf_data, li, f, &coords),
                                };
                                apply_binary(b, av, bv)
                            }
                        };
                        vals[k] = v;
                    }
                    *o = vals[rexprs.len() - 1];
                }
            });
    } else {
        let mut vals = vec![T::ZERO; rexprs.len()];
        let mut coords = vec![0usize; out_shape.len()];
        for (f, o) in out_data.iter_mut().enumerate() {
            coords_of(f, out_shape, &mut coords);
            for (k, e) in rexprs.iter().enumerate() {
                let av = match e.a {
                    RArg::Chain(m) => vals[m],
                    RArg::Leaf(li) => leaf_at(leaves, &leaf_data, li, f, &coords),
                };
                let v = match e.op {
                    ChainOp::Unary(u) => apply_unary(u, av, e.params),
                    ChainOp::Binary(b) => {
                        let bv = match e.b.expect("binary op has second operand") {
                            RArg::Chain(m) => vals[m],
                            RArg::Leaf(li) => leaf_at(leaves, &leaf_data, li, f, &coords),
                        };
                        apply_binary(b, av, bv)
                    }
                };
                vals[k] = v;
            }
            *o = vals[rexprs.len() - 1];
        }
    }
    Ok(())
}
