# FX graph walker: traversal, kwarg promotion, payload construction.

from __future__ import annotations

import json
from typing import Any, Callable

import torch

from .op_registry import (
    _TUPLE_OUTPUT_OPS,
    _getitem_as_select,
    _target_key,
    canonical_op,
)

# Shape ops whose positional const args (after the tensor) should be promoted
# to kwargs so the engine can read them without needing to inspect slot values.
# Maps op name -> kwarg name that receives the list/scalar of const args.


# Shape ops whose positional const args (after the tensor) should be promoted
# to kwargs so the engine can read them without needing to inspect slot values.
# Maps op name -> kwarg name that receives the list/scalar of const args.
_SHAPE_OP_CONST_KWARGS: dict[str, str] = {
    "permute": "dims",
    "reshape": "shape",
    "expand": "shape",
    "flip": "dims",
    "getitem": "index",
    "broadcast_to": "shape",
    "view_as": "shape",
    "expand_as": "shape",
    # Tensor creation ops: positional shape/dims become kwargs
    "zeros": "shape",
    "ones": "shape",
    "full": "shape",
    "arange": "start",  # handled specially below
    "empty_strided": "size",
}

# Phase 3 ops whose positional const/seq args (kernel, stride, padding, ...)
# are promoted to kwargs. Maps op name -> kwarg name per positional slot.
# Values may be lists (seq refs) or scalars (const refs); the engine reads
# them via kwargs and applies torch defaults for anything absent.
_CONV_POOL_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "conv1d": ["stride", "padding", "dilation", "groups"],
    "conv2d": ["stride", "padding", "dilation", "groups"],
    "conv_transpose1d": ["stride", "padding", "output_padding", "groups", "dilation"],
    "conv_transpose2d": ["stride", "padding", "output_padding", "groups", "dilation"],
    "max_pool1d": ["kernel", "stride", "padding", "dilation", "ceil_mode"],
    "max_pool2d": ["kernel", "stride", "padding", "dilation", "ceil_mode"],
    "avg_pool1d": ["kernel", "stride", "padding", "ceil_mode", "count_include_pad"],
    "avg_pool2d": ["kernel", "stride", "padding", "ceil_mode", "count_include_pad"],
    "adaptive_avg_pool2d": ["output_size"],
    "adaptive_max_pool2d": ["output_size"],
    "upsample_nearest2d": ["size"],
    "upsample_bilinear2d": ["size"],
    "flatten": ["start_dim", "end_dim"],
    # F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    "batch_norm": ["training", "momentum", "eps"],
}

# Reductions/activations whose positional const/seq args (dim, keepdim) are
# promoted to kwargs.  Export graphs emit e.g. ``aten.sum.dim_IntList(x, [1])``
# with dim as a positional seq; the engine only reads these from kwargs.
_REDUCE_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "sum": ["dim", "keepdim"],
    "mean": ["dim", "keepdim"],
    "max_reduce": ["dim", "keepdim"],
    "min_reduce": ["dim", "keepdim"],
    "argmax": ["dim", "keepdim"],
    "argmin": ["dim", "keepdim"],
    "std": ["dim", "keepdim"],
    "var": ["dim", "keepdim"],
    "cumsum": ["dim"],
    "prod": ["dim", "keepdim"],
    "norm": ["p", "dim", "keepdim"],
    "linalg_vector_norm": ["ord", "dim", "keepdim"],
    "softmax": ["dim"],
    "log_softmax": ["dim"],
    "threshold_backward": ["threshold"],
    "clamp": ["min", "max"],
    "clamp_min": ["min"],
    "clamp_max": ["max"],
    "gelu": ["approximate"],
    "isclose": ["rtol", "atol", "equal_nan"],
    "allclose": ["rtol", "atol", "equal_nan"],
    "nanprod": ["dim", "keepdim"],
    "nanmin": ["dim", "keepdim"],
    "nanmax": ["dim", "keepdim"],
    "var_mean": ["dim", "keepdim"],
    "std_mean": ["dim", "keepdim"],
    "nanmedian": ["dim", "keepdim"],
    "logsumexp": ["dim", "keepdim"],
    "cov": ["correction"],
    "linalg_vecdot": ["dim"],
    "linalg_cross": ["dim"],
    "linalg_tensordot": ["dims"],
    "adaptive_avg_pool1d": ["output_size"],
    "adaptive_max_pool1d": ["output_size"],
    "lp_pool3d": ["norm_type", "kernel_size", "stride"],
    "local_response_norm": ["size", "alpha", "beta", "k"],
    "pow": ["exp"],
}

# aten.transpose(x, d0, d1) — the two dims are positional consts.
_TRANSPOSE_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "transpose": ["d0", "d1"],
    "index_select": ["dim"],
    "gather": ["dim"],
    "narrow": ["dim", "start", "length"],
    "select": ["dim", "index"],
    "roll": ["shift", "dim"],
    "tile": ["repeats"],
    "pixel_shuffle": ["upscale_factor"],
    "instance_norm": ["eps"],
    "ldexp": ["other"],
    "split": ["split_size", "dim"],
    "vsplit": ["sections"],
    "hsplit": ["sections"],
    "dsplit": ["sections"],
    "tensor_split": ["indices", "dim"],
    "take_along_dim": ["dim"],
    "index_reduce": ["dim", "reduce"],
    "scatter_max": ["dim"],
    "scatter_min": ["dim"],
    "as_strided": ["size", "stride", "storage_offset"],
    "broadcast_to": ["shape"],
    "linalg_vander": ["N"],
    "linalg_cholesky_ex": ["upper"],
    "linalg_inv_ex": ["check_errors"],
    "linalg_solve_ex": ["check_errors"],
    "linalg_lu_factor": ["pivot"],
    "logsumexp": ["dim", "keepdim"],
}

# Phase 4: SDPA positional consts.  Export graphs emit either
# ``aten.scaled_dot_product_attention(q, k, v, mask, dropout_p, is_causal)``
# or the fused variant ``(q, k, v, dropout_p, is_causal)`` — promote the
# trailing scalars so the engine reads them from kwargs.
_SDPA_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "scaled_dot_product_attention": ["dropout_p", "is_causal"],
}

# Fused TransformerEncoderLayer (_transformer_encoder_layer_fwd).  The scalar
# params (embed_dim, num_heads, use_gelu, norm_first, eps) arrive as positional
# consts interleaved with tensor args; promote them to kwargs.  Tensor args stay
# positional in original order (src, qkv_w, qkv_b, proj_w, proj_b, nw1, nb1,
# nw2, nb2, ffn_w1, ffn_b1, ffn_w2, ffn_b2, mask?, bias?).
_TRANSFORMER_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "transformer_encoder_layer_fwd": ["embed_dim", "num_heads", "use_gelu", "norm_first", "eps"],
}

# Ops whose engine kernel consumes a fixed prefix of tensor args.  Dynamo
# replays a function's full positional signature (defaults included), so e.g.
# F.embedding arrives with padding_idx/max_norm/norm_type/scale_grad_by_freq/
# sparse as trailing consts; they are dead for the engine and must be trimmed
# (bool consts would also trip the runtime bool-const guard in _compiled.py).
_FIXED_TENSOR_ARITY: dict[str, int] = {
    "embedding": 2,  # (weight, indices)
}

# Phase 4: loss positional consts (reduction enum, ignore_index, beta).
_LOSS_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "nll_loss_forward": ["reduction", "ignore_index"],
    "mse_loss": ["reduction"],
    "smooth_l1_loss": ["reduction", "beta"],
    "binary_cross_entropy": ["reduction"],
    # aten.scalar_tensor(-inf, dtype=...) — the value is a positional const.
    "scalar_tensor": ["value"],
}


def _promote_positional_args_to_kwargs(
    op: str, args: list[dict[str, Any]], existing_kwargs: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Promote positional const/seq args of Phase 3 ops into kwargs.

    e.g. F.max_pool2d(x, 2, 2, 0, 1) -> args=[x], kwargs={"kernel":2,"stride":2,"padding":0,"dilation":1}
         torch.conv2d(x, w, b, [1,1], [1,1], [1,1], 1) -> args=[x,w,b], kwargs={...}

    Tensor refs (input/node/attr) are kept as positional args; every other
    positional arg is mapped to the next named kwarg for that op.
    """
    names = (
        _CONV_POOL_POSITIONAL_KWARGS.get(op)
        or _REDUCE_POSITIONAL_KWARGS.get(op)
        or _SDPA_POSITIONAL_KWARGS.get(op)
        or _LOSS_POSITIONAL_KWARGS.get(op)
        or _TRANSPOSE_POSITIONAL_KWARGS.get(op)
        or _TRANSFORMER_POSITIONAL_KWARGS.get(op)
    )
    if names is None:
        return args, existing_kwargs

    new_args: list[dict[str, Any]] = []
    new_kwargs = dict(existing_kwargs)
    name_iter = iter(names)
    optional_tensor_ops = {
        "conv1d", "conv2d", "conv_transpose1d", "conv_transpose2d",
        "nll_loss_forward", "cross_entropy_loss", "linear", "addmm"
    }
    for arg in args:
        if arg.get("kind") in ("input", "node", "attr"):
            new_args.append(arg)
            continue
        if op in optional_tensor_ops and arg.get("kind") == "const" and arg.get("value") is None:
            # Explicit None for optional tensor parameter (e.g. bias=None or weight=None):
            # drop it without advancing the kwarg name iterator.
            continue
        name = next(name_iter, None)
        if name is None:
            continue
        if arg.get("kind") == "const" and arg.get("value") is None:
            # Explicit None for named kwarg (e.g. clamp min=None): advance kwarg slot without setting
            continue
        if name in new_kwargs:
            continue  # an explicit kwarg already wins
        if arg.get("kind") == "seq":
            vals = [item["value"] for item in arg.get("value", []) if item.get("kind") == "const"]
            new_kwargs[name] = vals if len(vals) > 1 else (vals[0] if vals else None)
        else:
            new_kwargs[name] = arg.get("value")
    return new_args, new_kwargs


def _promote_const_args_to_kwargs(
    op: str, args: list[dict[str, Any]], existing_kwargs: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """For shape ops, extract positional const args after the tensor into kwargs.

    e.g. permute(tensor, 1, 0) -> args=[tensor_ref], kwargs={"dims": [1, 0]}
         reshape(tensor, -1)   -> args=[tensor_ref], kwargs={"shape": [-1]}

    This makes the Rust engine dispatch straightforward regardless of whether
    the op was called as a method or a function.
    """
    # select() has position-dependent const args: getitem(t, i) -> dim=0,index=i;
    # aten.select.int(t, dim, index) -> dim, index as written.
    if op == "select" and "dim" not in existing_kwargs:
        rest = [a for a in args[1:] if a.get("kind") in ("const", "seq")]
        values: list[Any] = []
        for a in rest:
            if a.get("kind") == "const":
                values.append(a["value"])
            elif a.get("kind") == "seq":
                values.extend(v for v in a.get("value", []) if isinstance(v, (int, float)))
        if len(values) == 1 and isinstance(values[0], int):
            return [args[0]], {**existing_kwargs, "dim": 0, "index": values[0]}
        if len(values) >= 2 and isinstance(values[0], int) and isinstance(values[1], int):
            return [args[0]], {**existing_kwargs, "dim": values[0], "index": values[1]}

    kwarg_key = _SHAPE_OP_CONST_KWARGS.get(op)
    if kwarg_key is None or kwarg_key in existing_kwargs:
        return args, existing_kwargs

    # Collect trailing const/seq args (everything after the first tensor ref).
    # e.g. view(x, (B, T, H, D)) arrives as a positional seq of consts.
    tensor_args = []
    const_vals = []
    seen_tensor = False
    for arg in args:
        kind = arg.get("kind")
        if seen_tensor and kind == "const":
            const_vals.append(arg["value"])
        elif seen_tensor and kind == "seq":
            items = arg.get("value", [])
            if all(item.get("kind") == "const" for item in items):
                const_vals.extend(item["value"] for item in items)
            else:
                tensor_args.append(arg)  # seq holding tensor refs: keep positional
        else:
            tensor_args.append(arg)
            if kind in ("input", "node", "attr"):
                seen_tensor = True

    if not const_vals:
        return args, existing_kwargs

    new_kwargs = dict(existing_kwargs)
    new_kwargs[kwarg_key] = const_vals
    return tensor_args, new_kwargs


def _dtype_str(dtype: torch.dtype) -> str | None:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float64:
        return "f64"
    return None


def parse_graph(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> tuple[dict[str, Any], dict[str, Callable]]:
    """Convert a GraphModule into a JSON-serializable plan.

    Returns ``(plan, function_map)`` where ``function_map`` maps target keys
    to the original callables (needed to re-invoke unsupported nodes eagerly;
    callables are not JSON-serializable so they travel separately).
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    tensor_inputs = [t for t in example_inputs if isinstance(t, torch.Tensor)]
    if len(tensor_inputs) == 0 or len(tensor_inputs) > len(placeholders):
        raise ValueError("torchburn: no tensor example inputs found for graph")
    tensor_placeholder_ids: set[int] = set()
    scalar_placeholder_ids: set[int] = set()

    node_id: dict[torch.fx.Node, int] = {}
    nodes: list[dict[str, Any]] = []
    function_map: dict[str, Callable] = {}
    next_id = 0
    tensor_index = 0

    # ------------------------------------------------------------------ pre-pass
    # Ops whose engine implementation returns ONE tensor while the aten op
    # returns a TUPLE unpacked by getitem(0)/getitem(1):
    #   * max_reduce / min_reduce -> (values, indices)
    #   * nll_loss_forward -> (loss, total_weight)
    #   * scaled_dot_product_attention (fused variants) -> (out, attn_weights)
    #   * unbind -> (slice_0, slice_1, ...)
    #   * chunk -> (part_0, part_1, ...)
    #   * sort -> (values, indices)
    #
    # For N-element tuples: if any non-zero-index getitem is consumed by
    # downstream nodes, the whole tuple node runs eagerly (the engine can
    # produce tuples but the interpreter doesn't map element-encoded capsule
    # outputs back to individual getitem results).  When only getitem(0) is
    # used (the common pattern for max/min/sort), we alias it to the node's
    # output slot and drop the dead getitem nodes.
    getitem_alias: dict[torch.fx.Node, torch.fx.Node] = {}
    drop_nodes: set[torch.fx.Node] = set()  # dead getitem nodes — never execute
    force_eager: set[torch.fx.Node] = set()
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        mapped = canonical_op(n)
        if mapped is None or mapped[0] not in _TUPLE_OUTPUT_OPS:
            continue
        # Collect ALL getitem consumers with their element indices.
        getitem_by_idx: dict[int, torch.fx.Node] = {}
        for user in n.users:
            if user.op != "call_function":
                continue
            if _target_key(user.target) != "_operator.getitem" and str(user.target) not in ("<built-in function getitem>", "operator.getitem"):
                continue
            if len(user.args) != 2 or not isinstance(user.args[1], int):
                continue
            getitem_by_idx[user.args[1]] = user
        # If ANY non-zero-index getitem is consumed downstream, force eager.
        has_consumed_nonzero = False
        for idx, gi_node in getitem_by_idx.items():
            if idx > 0 and any(u is not n for u in gi_node.users):
                has_consumed_nonzero = True
                break
        if has_consumed_nonzero:
            force_eager.add(n)
            for gi_node in getitem_by_idx.values():
                force_eager.add(gi_node)
        else:
            # Only element-0 is consumed (or no consumers at all).
            # Alias getitem(0) to the tuple-producing node; drop all dead getitems.
            gi0 = getitem_by_idx.get(0)
            if gi0 is not None:
                getitem_alias[gi0] = n
            for idx, gi_node in getitem_by_idx.items():
                if gi_node is not gi0 or gi0 is None:
                    drop_nodes.add(gi_node)



    for n in gm.graph.nodes:
        if n.op == "placeholder":
            pos = placeholders.index(n)
            node_id[n] = next_id
            if pos < len(example_inputs) and isinstance(example_inputs[pos], torch.Tensor):
                nodes.append({"id": next_id, "op": "placeholder", "pos": pos, "index": tensor_index})
                tensor_placeholder_ids.add(next_id)
                tensor_index += 1
            else:
                nodes.append({"id": next_id, "op": "placeholder", "pos": pos, "index": -1})
                scalar_placeholder_ids.add(next_id)
            next_id += 1
        elif n.op == "get_attr":
            node_id[n] = next_id
            nodes.append({"id": next_id, "op": "get_attr", "target": n.target})
            next_id += 1
        elif n.op == "output":
            node_id[n] = next_id
            nodes.append({"id": next_id, "op": "output", "args": [_ref(a, node_id) for a in n.args]})
            next_id += 1
        elif n.op in ("call_function", "call_method", "call_module"):
            # getitem(0) on a max/min_reduce: alias to the reduce node's slot
            # (the reduce output IS the values tensor).  Dead getitem(1) nodes
            # are dropped — running them eagerly would index the values tensor
            # instead of the (nonexistent) indices tuple.
            if n in drop_nodes:
                continue
            if n in getitem_alias:
                node_id[n] = node_id[getitem_alias[n]]
                continue

            mapped = canonical_op(n)
            if mapped is None:
                mapped = _getitem_as_select(n)
            target_key = _target_key(n.target)
            args = [_ref(a, node_id) for a in n.args]
            # The fused SDPA variants pass the attention mask as a kwarg
            # (attn_mask=<node>); move it into position 3 so the engine sees
            # q, k, v, mask, with dropout_p/is_causal promoted to kwargs.
            if mapped is not None and mapped[0] == "scaled_dot_product_attention":
                mask_node = n.kwargs.get("attn_mask")
                if isinstance(mask_node, torch.fx.Node):
                    args.append(_ref(mask_node, node_id))
            # to_dtype: method calls like x.float() / x.double() / x.to(dtype)
            # don't carry a serializable dtype kwarg.  Infer from the method name.
            _DTYPE_FROM_METHOD: dict[str, str] | None = None
            if mapped is not None and mapped[0] == "to_dtype" and "dtype" not in (n.kwargs or {}):
                _DTYPE_FROM_METHOD = {"float": "f32", "double": "f64", "half": "f16",
                                      "bfloat16": "bf16", "int": "i32", "long": "i64"}
            # Allow call_function and call_method nodes with kwargs — the engine
            # reads kwargs for dim/keepdim/eps/etc.  call_module nodes keep the
            # old strict check since their kwargs wiring is not yet mapped.
            kwargs_ok = (not n.kwargs) if n.op == "call_module" else True
            if mapped is not None and n not in force_eager and kwargs_ok and _arg_count_ok(mapped[0], args):
                op, key = mapped
                # For shape ops called as methods, promote trailing const args
                # (e.g. permute(1, 0), reshape(-1)) into kwargs so the engine
                # can read them without decoding scalar slot values.
                extracted_kwargs = _extract_kwargs(n.kwargs, node_id)
                # Inject dtype for method calls like x.float() / x.double()
                if _DTYPE_FROM_METHOD is not None and "dtype" not in extracted_kwargs:
                    method_name = str(n.target)
                    dtype = _DTYPE_FROM_METHOD.get(method_name)
                    if dtype is not None:
                        extracted_kwargs["dtype"] = dtype
                args, extracted_kwargs = _promote_const_args_to_kwargs(op, args, extracted_kwargs)
                args, extracted_kwargs = _promote_positional_args_to_kwargs(op, args, extracted_kwargs)
                # Ops whose engine kernel consumes a fixed prefix of tensor args:
                # drop trailing const defaults dynamo injects (e.g. F.embedding's
                # padding_idx/max_norm/norm_type/scale_grad_by_freq/sparse). They
                # are dead for the engine and bool consts would trip the runtime
                # bool-const guard, forcing an eager fallback.
                fixed = _FIXED_TENSOR_ARITY.get(op)
                if fixed is not None and len(args) > fixed:
                    args = args[:fixed]
                # F.embedding(input, weight) has the Python signature, but the
                # ATen schema (and the engine kernel) is embedding(weight,
                # indices) — make_fx emits the ATen order, dynamo emits the
                # Python order.  Normalise to (weight, indices).
                if op == "embedding" and target_key.startswith("torch.nn.functional"):
                    if len(args) == 2:
                        args = [args[1], args[0]]
                # aten *_native_batch_norm* variants carry schema order
                # (input, weight, bias, running_mean, running_var); the engine
                # kernel expects F.batch_norm order (input, running_mean,
                # running_var, weight, bias).  Consts were already promoted to
                # kwargs, so only the 5 tensor refs remain.
                if op == "batch_norm" and target_key.startswith("aten.") and "native_batch_norm" in target_key:
                    if len(args) >= 5:
                        args = [args[0], args[3], args[4], args[1], args[2]]
                # aten.std.correction / aten.var.correction pass ``correction``
                # (torch's divisor adjustment) as a kwarg, but the engine reads
                # ``unbiased`` (bool).  Map 0 -> False, 1 -> True; anything else
                # (correction=2 etc.) is unsupported and falls back to eager.
                if op in ("std", "var") and "correction" in extracted_kwargs:
                    corr = extracted_kwargs.pop("correction")
                    if corr not in (0, 1):
                        mapped = None
                    else:
                        extracted_kwargs["unbiased"] = bool(corr)
                # Tensor creation ops: all args are shape consts, no tensor input.
                # Collect all const args into a single "shape" list kwarg.
                if op in ("zeros", "ones", "full") and "shape" not in extracted_kwargs:
                    shape_vals = []
                    for a in args:
                        if isinstance(a, dict) and a.get("kind") == "const":
                            shape_vals.append(a["value"])
                    if shape_vals:
                        args = [args[0]] if args else []  # keep first ref (even if const)
                        extracted_kwargs["shape"] = shape_vals
                if op == "arange" and not extracted_kwargs:
                    # torch.arange(end) or torch.arange(start, end, step)
                    vals = []
                    for a in args:
                        if isinstance(a, dict) and a.get("kind") == "const":
                            vals.append(a["value"])
                    if len(vals) >= 1:
                        extracted_kwargs["end"] = vals[-1]
                    if len(vals) >= 2:
                        extracted_kwargs["start"] = vals[0]
                    if len(vals) >= 3:
                        extracted_kwargs["step"] = vals[1]
                    args = []
                    # torch.arange defaults to int64; the engine defaults to f32.
                    # Emit i64 when every bound is integral so index tensors fed
                    # to embedding stay integer (the kernel rejects float ids).
                    if "dtype" not in extracted_kwargs and vals and all(
                        isinstance(v, int) and not isinstance(v, bool) for v in vals
                    ):
                        extracted_kwargs["dtype"] = "i64"
                # einsum: equation string is args[0] (const), tensors start at args[1]
                if op == "einsum" and args and isinstance(args[0], dict) and args[0].get("kind") == "const":
                    extracted_kwargs["equation"] = args[0]["value"]
                    args = args[1:]
                if mapped is not None and n.op == "call_function":
                    function_map[key] = n.target
                if mapped is None:
                    if n.op == "call_function":
                        function_map[target_key] = n.target
                    node_id[n] = next_id
                    nodes.append(
                        {
                            "id": next_id,
                            "op": "unsupported",
                            "fx_op": n.op,
                            "fx_target": target_key,
                            "args": args,
                            "fx_args": [_ref(a, node_id) for a in n.args],
                            # kwargs feeds the JSON signature payload: keep only
                            # serializable primitives (device/dtype objects and
                            # tensor refs live in fx_kwargs for eager replay).
                            "kwargs": _extract_kwargs(n.kwargs, node_id),
                            "fx_kwargs": _ref_kwargs(n.kwargs, node_id),
                        }
                    )
                else:
                    node_id[n] = next_id
                    nodes.append(
                        {
                            "id": next_id,
                            "op": "supported",
                            "target": op,
                            "fx_op": n.op,
                            "fx_target": target_key,
                            "args": args,
                            "fx_args": [_ref(a, node_id) for a in n.args],
                            "kwargs": extracted_kwargs,
                            "fx_kwargs": _ref_kwargs(n.kwargs, node_id),
                        }
                    )
            else:
                if n.op == "call_function":
                    function_map[target_key] = n.target
                node_id[n] = next_id
                nodes.append(
                    {
                        "id": next_id,
                        "op": "unsupported",
                        "fx_op": n.op,
                        "fx_target": target_key,
                        "args": args,
                        "fx_args": [_ref(a, node_id) for a in n.args],
                        "kwargs": _extract_kwargs(n.kwargs, node_id),
                        "fx_kwargs": _ref_kwargs(n.kwargs, node_id),
                    }
                )
            next_id += 1
        else:
            raise ValueError(f"torchburn: unexpected FX node op {n.op!r}")

    input_spec = [
        {"shape": [int(s) for s in t.shape], "dtype": _dtype_str(t.dtype)}
        for t in tensor_inputs
    ]
    return {
        "nodes": nodes,
        "input_spec": input_spec,
        "tensor_placeholders": sorted(tensor_placeholder_ids),
        "scalar_placeholders": sorted(scalar_placeholder_ids),
    }, function_map


def _ref_kwargs(fx_kwargs: Any, node_id: dict) -> dict[str, Any]:
    """Convert FX node kwargs for the eager fallback (REQ-002).

    Tensor-valued kwargs (e.g. ``attn_mask=<Node>``) must become refs so
    ``_run_eager`` can resolve them from the env; raw ``torch.fx.Node``
    objects are not serialisable and would otherwise reach eager as Nodes.
    """
    out: dict[str, Any] = {}
    for k, v in fx_kwargs.items():
        if isinstance(v, torch.fx.Node):
            out[k] = _ref(v, node_id)
        elif isinstance(v, (list, tuple)) and any(isinstance(i, torch.fx.Node) for i in v):
            out[k] = {"kind": "seq", "value": [_ref(i, node_id) if isinstance(i, torch.fx.Node) else i for i in v],
                       "type": "tuple" if isinstance(v, tuple) else "list"}
        else:
            out[k] = v
    return out


def _extract_kwargs(fx_kwargs: Any, node_id: dict) -> dict[str, Any]:
    """Convert FX node kwargs into JSON-serializable primitives for the engine.

    Tensor-valued kwargs are not expected at the ops we support; they come as
    positional args or get_attr refs.  We serialise scalars, booleans, and
    simple sequences; anything we can't represent becomes None (safe: the
    engine falls back to its own defaults).
    """
    out: dict[str, Any] = {}
    for k, v in fx_kwargs.items():
        if isinstance(v, torch.dtype):
            # dtype kwarg (e.g. aten._to_copy(dtype=torch.float64)) -> "f32"/"f64"
            if v == torch.float32:
                out[k] = "f32"
            elif v == torch.float64:
                out[k] = "f64"
            continue
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            # e.g. dims=[0, 1], normalized_shape=(16,)
            serialized = []
            for item in v:
                if isinstance(item, (bool, int, float)):
                    serialized.append(item)
                else:
                    serialized = None  # unserialisable element
                    break
            if serialized is not None:
                out[k] = serialized
        # Skip tensors and other complex objects; the engine uses defaults.
    return out


def _arg_count_ok(op: str, args: list[dict[str, Any]]) -> bool:
    """Check if the arg count is reasonable for this op."""
    # Unary ops: 1 arg
    unary_ops = {"relu", "abs", "neg", "sign", "sqrt", "rsqrt", "exp", "log",
                 "reciprocal", "ceil", "floor", "round", "sin", "cos",
                 "sigmoid", "tanh", "gelu", "silu",
                 "elu", "selu", "softplus", "hardswish", "mish", "softmax",
                 "log_softmax", "norm"}
    # Binary ops: 2 args
    binary_ops = {"add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge"}
    # Reduction ops: 1-2 args (tensor + optional dim)
    reduce_ops = {"sum", "mean", "max_reduce", "min_reduce", "argmax", "argmin", "std", "var",
                  "cumsum", "prod"}
    # Linalg ops: 2-3 args
    linalg_ops = {"matmul", "bmm", "dot"}
    # 3-arg ops
    ternary_ops = {"where", "masked_fill"}
    # clamp: 1 tensor arg + kwargs (min/max)
    clamp_ops = {"clamp", "clamp_min", "clamp_max"}

    if op in unary_ops:
        return len(args) >= 1
    if op in binary_ops:
        return len(args) >= 2
    if op in reduce_ops:
        return len(args) >= 1
    if op in linalg_ops:
        return len(args) >= 2
    if op in ternary_ops:
        return len(args) >= 3
    if op in clamp_ops:
        return len(args) >= 1
    # ops with variable args (cat, stack, layer_norm, batch_norm, etc.)
    return True


def _ref(a: Any, node_id: dict[torch.fx.Node, int]) -> Any:
    """Serialize an FX node argument into a plan reference."""
    if isinstance(a, torch.fx.Node):
        if a.op == "placeholder":
            return {"kind": "input", "index": node_id[a]}
        if a.op == "get_attr":
            return {"kind": "attr", "index": node_id[a]}
        return {"kind": "node", "index": node_id[a]}
    if a is Ellipsis:
        # x[..., :half] — ``...`` must round-trip as Ellipsis, not None (None
        # would add a new axis in eager getitem).
        return {"kind": "const", "value": "__ellipsis__"}
    if isinstance(a, slice):
        # x[..., :half] — slice args must round-trip for eager fallback.
        return {
            "kind": "slice",
            "start": _ref(a.start, node_id) if isinstance(a.start, torch.fx.Node) else a.start,
            "stop": _ref(a.stop, node_id) if isinstance(a.stop, torch.fx.Node) else a.stop,
            "step": a.step,
        }
    if isinstance(a, (list, tuple)):
        return {"kind": "seq", "type": type(a).__name__, "value": [_ref(x, node_id) for x in a]}
    if isinstance(a, torch.dtype):
        # dtype arg (e.g. x.to(torch.float64)) -> "f32"/"f64" string
        return {"kind": "const", "value": "f32" if a == torch.float32 else "f64"}
    if a is None or isinstance(a, (bool, int, float, str)):
        return {"kind": "const", "value": a}
    return {"kind": "const", "value": None}


def _sanitize_nonfinite(obj: Any) -> Any:
    """Replace non-finite floats with string tokens.

    Python's json emits ``Infinity``/``NaN`` which serde_json rejects, so
    masks built from ``-inf`` scalars would break the payload.  The engine
    decodes the ``"inf"``/``"-inf"``/``"nan"`` strings back to floats.
    """
    import math

    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)  # 'inf', '-inf', 'nan'
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


def payload_json(plan: dict[str, Any]) -> str:
    """Canonical JSON form of a plan (sorted keys => stable BLAKE3 signature).

    ``fx_kwargs`` holds the ORIGINAL FX kwargs (may contain non-serializable
    values like torch.dtype) for eager fallback replay; it is excluded from
    the signature payload because it is never sent to the Rust engine.
    """
    # Deep-copy so we never mutate the live plan (which carries fx_kwargs).
    nodes = []
    for node in plan.get("nodes", []):
        copy = dict(node)
        copy.pop("fx_kwargs", None)
        nodes.append(copy)
    sig = dict(plan)
    sig.pop("fx_kwargs", None)
    sig["nodes"] = nodes
    sig = _sanitize_nonfinite(sig)
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))
