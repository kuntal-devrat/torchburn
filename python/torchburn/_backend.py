"""Backend registration for ``torch.compile(..., backend="torchburn")`` (REQ-001)."""

from __future__ import annotations

import logging

import torch

from ._compiled import BurnCompiledCallable
from ._interpreter import _warn_fallback
from ._parser.op_registry import (
    canonical_op,
    _getitem_as_select,
    _SUPPORTED_OPS,
    _TUPLE_OUTPUT_OPS,
    _target_key,
)

_LOG = logging.getLogger("torchburn")


class _BurnSubmoduleWrapper(torch.nn.Module):
    """Wraps BurnCompiledCallable in an nn.Module so it can be assigned as an FX submodule."""

    def __init__(self, compiled: BurnCompiledCallable):
        super().__init__()
        self._compiled = compiled

    def forward(self, *args: any, **kwargs: any) -> any:
        return self._compiled(*args, **kwargs)


def _is_node_supported(node: torch.fx.Node) -> bool:
    """Check if a specific FX node is supported by TorchBurn native execution."""
    if node.op in ("placeholder", "output", "get_attr"):
        return True
    if node.op not in ("call_function", "call_method"):
        return False
    mapped = canonical_op(node)
    if mapped is None:
        mapped = _getitem_as_select(node)
    if mapped is None:
        return False
    op_name, _ = mapped
    if op_name not in _SUPPORTED_OPS:
        return False
    # If the op returns a tuple (e.g. native_layer_norm, max_reduce),
    # TorchBurn engine natively only produces element 0. If any downstream
    # node consumes element > 0 (e.g. mean/rstd saved for backward),
    # TorchBurn cannot execute this node natively without falling back.
    if op_name in _TUPLE_OUTPUT_OPS:
        for user in node.users:
            if user.op == "call_function":
                target_k = _target_key(user.target)
                if target_k in ("_operator.getitem", "operator.getitem", "<built-in function getitem>"):
                    if len(user.args) >= 2 and isinstance(user.args[1], int) and user.args[1] > 0:
                        if any(u is not node for u in user.users):
                            return False
    return True


def _has_unsupported_nodes(gm: torch.fx.GraphModule) -> bool:
    """Check if a graph module contains any nodes unsupported by TorchBurn."""
    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method"):
            if not _is_node_supported(node):
                return True
    return False


def _partition_graph(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """Partition an FX graph into TorchBurn-native and PyTorch-eager subgraphs.

    Clusters contiguous supported nodes into compiled TorchBurn subgraphs and
    unsupported nodes into PyTorch FX submodules. This eliminates the per-node
    Python interpreter fallback loop that causes GIL contention.
    """
    call_nodes = [n for n in gm.graph.nodes if n.op in ("call_function", "call_method")]
    if not call_nodes:
        return BurnCompiledCallable(gm, example_inputs)

    # If all nodes are supported, compile the entire graph
    if not any(not _is_node_supported(n) for n in call_nodes):
        return BurnCompiledCallable(gm, example_inputs)

    # If all nodes are unsupported, keep eager PyTorch graph
    if all(not _is_node_supported(n) for n in call_nodes):
        for n in call_nodes:
            _warn_fallback(_target_key(n.target))
        return gm

    for n in call_nodes:
        if not _is_node_supported(n):
            _warn_fallback(_target_key(n.target))

    try:
        from torch.fx.passes.split_module import split_module
    except ImportError:
        _LOG.debug("torch.fx.passes.split_module not available; skipping partitioning")
        return BurnCompiledCallable(gm, example_inputs)

    # Build monotonic partition tags along topological order to guarantee NO cycles.
    curr_id = 0
    last_status = None
    node_tags: dict[torch.fx.Node, int] = {}

    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output", "get_attr"):
            continue
        # Direct getitem unpacking of a tuple node must stay in the same partition as the parent
        target_k = _target_key(n.target)
        if target_k in ("_operator.getitem", "operator.getitem", "<built-in function getitem>"):
            parent = n.args[0] if n.args else None
            if isinstance(parent, torch.fx.Node) and parent in node_tags:
                node_tags[n] = node_tags[parent]
                continue

        status = _is_node_supported(n)
        if last_status is not None and status != last_status:
            curr_id += 1
        last_status = status
        node_tags[n] = curr_id

    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output", "get_attr"):
            node_tags[n] = 0

    try:
        split_gm = split_module(gm, gm, lambda n: node_tags[n])
    except Exception as exc:
        _LOG.debug("split_module failed: %s; falling back to single-graph compilation", exc)
        return BurnCompiledCallable(gm, example_inputs)

    # Capture runtime example inputs for each submodule
    submod_inputs: dict[str, tuple[any, ...]] = {}
    handles = []
    for name, child in split_gm.named_children():
        def make_hook(submod_name: str):
            def hook(mod: torch.nn.Module, inputs: tuple[any, ...]):
                if submod_name not in submod_inputs:
                    submod_inputs[submod_name] = inputs
            return hook
        h = child.register_forward_pre_hook(make_hook(name))
        handles.append(h)

    try:
        split_gm(*example_inputs)
    except Exception as exc:
        _LOG.debug("Submodule input tracing pass failed: %s", exc)
    finally:
        for h in handles:
            h.remove()

    # Compile supported submodules with TorchBurn
    for name, child in list(split_gm.named_children()):
        if not isinstance(child, torch.fx.GraphModule):
            continue
        call_sub = [n for n in child.graph.nodes if n.op in ("call_function", "call_method")]
        if call_sub and all(_is_node_supported(n) for n in call_sub):
            inputs = submod_inputs.get(name)
            if inputs and all(isinstance(x, torch.Tensor) for x in inputs):
                try:
                    compiled = BurnCompiledCallable(child, list(inputs))
                    setattr(split_gm, name, _BurnSubmoduleWrapper(compiled))
                except Exception as exc:
                    _LOG.debug("Failed to compile submodule %s: %s", name, exc)

    return split_gm


def _compile_boxed(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    try:
        from torch._functorch.aot_autograd import make_boxed_func
    except ImportError:
        return BurnCompiledCallable(gm, example_inputs)

    # Check if graph has unsupported ops and partition if needed
    if _has_unsupported_nodes(gm):
        partitioned = _partition_graph(gm, example_inputs)
        try:
            return make_boxed_func(partitioned)
        except Exception:
            return partitioned
    else:
        try:
            return make_boxed_func(BurnCompiledCallable(gm, example_inputs))
        except Exception:
            return BurnCompiledCallable(gm, example_inputs)


def torchburn_backend(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
):
    """Dynamo compilation backend entrypoint with native AOTAutograd training integration.

    Receives the traced ``torch.fx.GraphModule`` plus sample inputs, partitions
    forward and backward graphs via AOTAutograd for training or compiles the
    module directly for inference, returning high-performance native TorchBurn callables.
    """
    needs_training = (
        torch.is_grad_enabled()
        and (
            any(getattr(t, "requires_grad", False) for t in example_inputs if isinstance(t, torch.Tensor))
            or any(p.requires_grad for p in gm.parameters())
        )
    )
    if needs_training:
        try:
            from torch._functorch.aot_autograd import aot_module_simplified
            return aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=_compile_boxed,
                bw_compiler=_compile_boxed,
            )
        except Exception as exc:
            _LOG.debug("aot_module_simplified fallback: %s", exc)

    # For inference: partition if there are unsupported ops
    if _has_unsupported_nodes(gm):
        return _partition_graph(gm, example_inputs)

    return BurnCompiledCallable(gm, example_inputs)


def register() -> None:
    """Register the ``torchburn`` backend so ``torch.compile(backend=...)`` works."""
    if getattr(torch, "_dynamo", None) is not None:
        try:
            torch._dynamo.register_backend(name="torchburn")(torchburn_backend)
            return
        except Exception as exc:  # pragma: no cover - API drift fallback
            _LOG.debug("torch._dynamo.register_backend failed: %s", exc)
    try:
        torch.compiler.register_backend(name="torchburn")(torchburn_backend)
    except Exception as exc:  # pragma: no cover
        _LOG.warning("torchburn: could not register the 'torchburn' backend: %s", exc)
