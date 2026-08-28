"""Shared interpreter for torchburn compiled callables (REQ-001).

Eliminates duplication between `BurnCompiledCallable` (`_compiled.py`) and
`TorchBurnModule` (`capture.py`). Both need:

* pre-compute (`_precompute_plan`) — slice FX graph into native/eager phases,
  build a single combined payload and `prepare_graph` it,
* runtime (`_interpret`, `_exec_all_native`, `_exec_native_phase`) — zero-copy
  DLPack dispatch with fallback,
* helpers (`_resolve`, `_run_eager`, `_tensor_for`, etc.)

The mixin holds all shared state; subclasses provide `gm`/`function_map`/`plan`.
"""

from __future__ import annotations

import threading
import warnings
import weakref
from typing import Any

import torch

from . import _torchburn as _native
from ._cache import _LOCK, lookup, store
from ._parser import payload_json, parse_graph

_WARNED: set[tuple[str, str]] = set()
_WARN_LOCK = threading.Lock()

_F32_F64 = (torch.float32, torch.float64)
_INT_BOOL = (torch.int64, torch.int32, torch.bool)
_MIXED_FLOAT = (torch.float16, torch.bfloat16)


def _warn_fallback(target: str, reason: str = "") -> None:
    key = (target, reason)
    with _WARN_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    detail = f" ({reason})" if reason else ""
    warnings.warn(
        UserWarning(
            f"torchburn: falling back to eager PyTorch for unsupported "
            f"operator {target!r}{detail}. See torchburn.cache_stats() for coverage."
        ),
        stacklevel=3,
    )


def _arg_key(arg: dict[str, Any]) -> tuple:
    """Hashable dedup key for a node argument."""
    if arg.get("kind") == "const":
        return ("const", type(arg["value"]).__name__, arg["value"])
    return (arg.get("kind"), arg.get("index"))


class _BaseInterpreter:
    """Mixin with shared torchburn interpreter logic.

    Subclasses must set before calling `_init_interpreter`:
      - self.gm: torch.fx.GraphModule
      - self._input_count (optional)
    After init, provides `_phases`, `_combined_input_keys`, etc., and
    `._interpret(env)`, `._graph_handle`, etc.
    """

    _MAX_CHUNK_NODES = 128

    # To be filled by _init_interpreter
    plan: dict[str, Any]
    function_map: dict[str, Any]
    signature: str
    _phases: list[dict[str, Any]]
    _combined_input_keys: list[tuple]
    _combined_output_ids: list[int]
    _graph_handle: int | None

    def _init_interpreter(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> None:
        self.gm = gm
        self._input_count = len(example_inputs)
        plan, function_map = parse_graph(gm, example_inputs)
        signature = _native.signature(payload_json(plan))
        with _LOCK:
            cached = lookup(signature)
            if cached is not None:
                plan = cached["plan"]
            else:
                store(signature, plan)
        self.signature = signature
        self.plan = plan
        self.function_map = function_map
        self._phases = []
        self._combined_input_keys = []
        self._combined_output_ids = []
        self._node_phase = {}
        self._needed_cache: dict[tuple, set[int]] = {}
        self._graph_handle = None
        self._precompute_plan()
        if self._graph_handle is not None:
            weakref.finalize(self, _native.release_graph, self._graph_handle)

    # --------------------------------------------- pre-compute at init time
    def _precompute_plan(self) -> None:
        """Walk the plan once and build phases + combined payload."""
        phases: list[dict[str, Any]] = []
        native_batch: list[dict[str, Any]] = []

        def _flush_native():
            if not native_batch:
                return
            phases.append({"kind": "native", "nodes": native_batch[:]})
            native_batch.clear()

        for node in self.plan["nodes"]:
            if node["op"] == "supported":
                if len(native_batch) >= self._MAX_CHUNK_NODES:
                    _flush_native()
                native_batch.append(node)
            elif node["op"] == "unsupported":
                _flush_native()
                phases.append({"kind": "eager", "node": node})
        _flush_native()

        native_nodes = [n for p in phases if p["kind"] == "native" for n in p["nodes"]]
        if not native_nodes:
            self._phases = phases
            return

        all_input_keys: list[tuple] = []
        seen: dict[tuple, int] = {}
        all_native_ids = {n["id"] for n in native_nodes}
        global_pos = {n["id"]: i for i, n in enumerate(native_nodes)}

        def _reg(arg: dict) -> int:
            k = _arg_key(arg)
            if k in seen:
                return seen[k]
            kind = arg.get("kind", "")
            if kind == "node" and arg.get("index") in all_native_ids:
                return -1
            if kind == "seq":
                return -1
            if kind == "const" and arg.get("value") is None:
                return -1
            idx = len(all_input_keys)
            all_input_keys.append(k)
            seen[k] = idx
            return idx

        for node in native_nodes:
            for arg in node["args"]:
                kind = arg.get("kind", "")
                if kind == "seq":
                    if node.get("target") in ("cat", "stack"):
                        for item in arg.get("value", []):
                            _reg(item)
                    continue
                _reg(arg)

        node_base = len(all_input_keys)

        payload_nodes = []
        for node in native_nodes:
            pargs = []
            for arg in node["args"]:
                kind = arg.get("kind", "")
                if kind == "seq":
                    if node.get("target") in ("cat", "stack"):
                        indices = []
                        for item in arg.get("value", []):
                            if item.get("kind") in ("input", "node", "attr") and item.get("index") in all_native_ids:
                                indices.append(node_base + global_pos[item["index"]])
                            else:
                                idx = seen.get(_arg_key(item))
                                if idx is not None:
                                    indices.append(idx)
                        pargs.append({"kind": "slot", "value": indices})
                    continue
                if kind == "node" and arg["index"] in all_native_ids:
                    pargs.append({"kind": "slot", "index": node_base + global_pos[arg["index"]]})
                else:
                    idx = seen.get(_arg_key(arg), -1)
                    if idx >= 0:
                        pargs.append({"kind": "slot", "index": idx})
            payload_nodes.append({
                "id": node["id"],
                "target": node["target"],
                "args": pargs,
                "kwargs": node.get("kwargs", {}),
            })

        needed = self._needed_outputs(all_native_ids)
        input_specs = [{"shape": [0], "dtype": "f32"} for _ in all_input_keys]
        payload = {"inputs": input_specs, "nodes": payload_nodes, "outputs": sorted(needed)}
        self._phases = phases
        self._combined_input_keys = all_input_keys
        self._combined_output_ids = sorted(needed)
        self._combined_payload_template = payload
        try:
            self._graph_handle = _native.prepare_graph(payload)
        except Exception:
            self._graph_handle = None

    # ------------------------------------------------------------ interpreter
    def _interpret(self, env: dict[int, Any]) -> Any:
        if not self._phases:
            return None
        has_eager = any(p["kind"] == "eager" for p in self._phases)
        if not has_eager:
            native_nodes = [n for p in self._phases if p["kind"] == "native" for n in p["nodes"]]
            if native_nodes:
                self._exec_all_native(env)
        else:
            for p in self._phases:
                if p["kind"] == "native":
                    self._exec_native_phase(p["nodes"], env)
                elif p["kind"] == "eager":
                    env[p["node"]["id"]] = self._run_eager(p["node"], env)
        for node in self.plan["nodes"]:
            if node["op"] == "output":
                return self._resolve(node["args"][0], env)
        return None

    def _exec_all_native(self, env: dict[int, Any]) -> None:
        run_inputs: list[torch.Tensor] = []
        cast_map: dict[int, torch.dtype] = {}
        bad_keys: set[int] = set()
        for i, key in enumerate(self._combined_input_keys):
            kind = key[0]
            if kind == "const":
                val = key[2]
                tensor = torch.tensor(val, dtype=torch.float32)
                run_inputs.append(tensor)
            else:
                index = key[1]
                val = env.get(index)
                if val is None or not isinstance(val, torch.Tensor):
                    bad_keys.add(i)
                    run_inputs.append(torch.empty(0))
                    continue
                tensor = val.detach() if val.requires_grad else val
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                if tensor.dtype in _MIXED_FLOAT:
                    cast_map[len(run_inputs)] = tensor.dtype
                    tensor = tensor.to(torch.float32)
                run_inputs.append(tensor)
        if bad_keys:
            self._exec_phases_sequentially(env)
            return
        if self._graph_handle is None:
            self._exec_phases_sequentially(env)
            return
        capsules = [t.__dlpack__() for t in run_inputs]
        try:
            out_capsules = _native.execute_prepared(self._graph_handle, capsules)
        except Exception:
            self._exec_phases_sequentially(env)
            return
        by_id = dict(zip(self._combined_output_ids, out_capsules))
        all_native_ids = set()
        for p in self._phases:
            if p["kind"] == "native":
                for n in p["nodes"]:
                    all_native_ids.add(n["id"])
        for node_id in all_native_ids:
            capsule = by_id.get(node_id)
            if capsule is not None:
                t = torch.from_dlpack(capsule)
                if cast_map and t.dtype == torch.float32:
                    dtype_counts: dict[torch.dtype, int] = {}
                    for dt in cast_map.values():
                        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                    if dtype_counts:
                        t = t.to(max(dtype_counts, key=dtype_counts.get))
                env[node_id] = t

    def _exec_phases_sequentially(self, env: dict[int, Any]) -> None:
        for p in self._phases:
            if p["kind"] == "native":
                self._exec_native_phase(p["nodes"], env)
            elif p["kind"] == "eager":
                node = p["node"]
                env[node["id"]] = self._run_eager(node, env)

    def _exec_native_phase(self, nodes: list[dict[str, Any]], env: dict[int, Any]) -> None:
        if not nodes:
            return
        chunk_ids = {n["id"] for n in nodes}
        pos = {n["id"]: p for p, n in enumerate(nodes)}
        run_inputs: list[torch.Tensor] = []
        seen: dict[tuple, int] = {}
        cast_map: dict[int, torch.dtype] = {}

        def _add(arg, node):
            k = _arg_key(arg)
            if k in seen:
                return seen[k]
            kind = arg.get("kind", "")
            if kind == "node" and arg["index"] in chunk_ids:
                return -1
            if kind == "seq":
                return -1
            val = env.get(arg["index"]) if kind != "const" else arg["value"]
            if val is None or (kind != "const" and not isinstance(val, torch.Tensor)):
                seen[k] = -1
                return -1
            tensor = self._tensor_for(arg, node, env)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tensor.dtype in _MIXED_FLOAT:
                cast_map[len(run_inputs)] = tensor.dtype
                tensor = tensor.to(torch.float32)
            idx = len(run_inputs)
            run_inputs.append(tensor)
            seen[k] = idx
            return idx

        for node in nodes:
            for arg in node["args"]:
                kind = arg.get("kind", "")
                if kind == "seq":
                    if node.get("target") in ("cat", "stack"):
                        for item in arg.get("value", []):
                            if item.get("kind") in ("input", "node", "attr") and item.get("index") in chunk_ids:
                                continue
                            _add(item, node)
                    continue
                _add(arg, node)

        base = len(run_inputs)
        payload_nodes = []
        for node in nodes:
            pargs = []
            for arg in node["args"]:
                kind = arg.get("kind", "")
                if kind == "seq":
                    if node.get("target") in ("cat", "stack"):
                        indices = []
                        for item in arg.get("value", []):
                            if item.get("kind") in ("input", "node", "attr") and item.get("index") in chunk_ids:
                                indices.append(base + pos[item["index"]])
                            else:
                                idx = seen.get(_arg_key(item), -1)
                                if idx >= 0:
                                    indices.append(idx)
                        pargs.append({"kind": "slot", "value": indices})
                    continue
                if kind == "node" and arg["index"] in chunk_ids:
                    pargs.append({"kind": "slot", "index": base + pos[arg["index"]]})
                else:
                    idx = _add(arg, node) if _arg_key(arg) not in seen else seen[_arg_key(arg)]
                    if idx >= 0:
                        pargs.append({"kind": "slot", "index": idx})
            payload_nodes.append({"id": node["id"], "target": node["target"], "args": pargs, "kwargs": node.get("kwargs", {}),})

        needed = self._needed_outputs(chunk_ids)
        payload = {"inputs": [self._spec(t) for t in run_inputs], "nodes": payload_nodes, "outputs": sorted(needed),}
        capsules = [t.__dlpack__() for t in run_inputs]
        try:
            out_capsules = _native.execute_from_dict(payload, capsules)
        except Exception as exc:
            msg = str(exc)
            reason = msg.split("TB_UNSUPPORTED:", 1)[1].strip() if "TB_UNSUPPORTED:" in msg else msg
            for node in nodes:
                _warn_fallback(node["fx_target"], reason)
                env[node["id"]] = self._run_eager(node, env)
            return
        by_id = dict(zip(sorted(needed), out_capsules))
        for node in nodes:
            capsule = by_id.get(node["id"])
            if capsule is not None:
                t = torch.from_dlpack(capsule)
                if cast_map:
                    dtype_counts2: dict[torch.dtype, int] = {}
                    for dt in cast_map.values():
                        dtype_counts2[dt] = dtype_counts2.get(dt, 0) + 1
                    if dtype_counts2 and t.dtype == torch.float32:
                        t = t.to(max(dtype_counts2, key=dtype_counts2.get))
                env[node["id"]] = t

    # -------------------------------------------------------- output helpers
    def _needed_outputs(self, chunk_ids: set[int]) -> set[int]:
        key = tuple(sorted(chunk_ids))
        cached = self._needed_cache.get(key)
        if cached is not None:
            return cached
        needed: set[int] = set()
        for node in self.plan["nodes"]:
            if node["id"] in chunk_ids:
                continue
            self._collect_refs(node.get("args", []), chunk_ids, needed)
        self._needed_cache[key] = needed
        return needed

    def _collect_refs(self, args: Any, chunk_ids: set[int], needed: set[int]) -> None:
        if isinstance(args, dict):
            if args.get("kind") == "node" and args.get("index") in chunk_ids:
                needed.add(args["index"])
            elif args.get("kind") == "seq":
                for item in args.get("value", []):
                    self._collect_refs(item, chunk_ids, needed)
        elif isinstance(args, (list, tuple)):
            for item in args:
                self._collect_refs(item, chunk_ids, needed)

    def _node_ok(self, node: dict[str, Any], env: dict[int, Any]) -> bool:
        for arg in node["args"]:
            kind = arg.get("kind", "")
            if kind == "const":
                if isinstance(arg.get("value"), bool):
                    return False
                continue
            if kind == "seq":
                continue
            if kind not in ("input", "node", "attr"):
                continue
            value = env.get(arg["index"])
            if isinstance(value, (tuple, list)):
                return False
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    return False
                if value.dtype not in _F32_F64 + _INT_BOOL + _MIXED_FLOAT:
                    return False
        return True

    def _tensor_for(self, arg: dict[str, Any], node: dict[str, Any], env: dict[int, Any]) -> torch.Tensor:
        if arg["kind"] == "const":
            return torch.tensor(arg["value"], dtype=self._scalar_dtype(node, env))
        value = env[arg["index"]]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"torchburn: expected tensor operand, got {type(value).__name__}")
        return value.detach() if value.requires_grad else value

    @staticmethod
    def _scalar_dtype(node: dict[str, Any], env: dict[int, Any]) -> torch.dtype:
        for arg in node["args"]:
            if arg["kind"] in ("input", "node", "attr"):
                value = env.get(arg["index"])
                if isinstance(value, torch.Tensor) and value.dtype in _F32_F64:
                    return value.dtype
        return torch.float32

    @staticmethod
    def _spec(t: torch.Tensor) -> dict[str, Any]:
        dtype_map = {torch.float32: "f32", torch.float64: "f64", torch.int64: "i64", torch.int32: "i32", torch.bool: "bool",}
        return {"shape": [int(s) for s in t.shape], "dtype": dtype_map.get(t.dtype, "f32")}

    def _resolve(self, refs: Any, env: dict[int, Any]) -> Any:
        if isinstance(refs, dict):
            kind = refs["kind"]
            if kind == "seq":
                resolved = [self._resolve(x, env) for x in refs["value"]]
                return tuple(resolved) if refs.get("type") == "tuple" else resolved
            if kind == "slice":
                return slice(self._resolve(refs["start"], env) if isinstance(refs["start"], dict) else refs["start"], self._resolve(refs["stop"], env) if isinstance(refs["stop"], dict) else refs["stop"], refs.get("step"),)
            if kind == "const" and refs.get("value") == "__ellipsis__":
                return Ellipsis
            if kind in ("input", "node", "attr"):
                try:
                    return env[refs["index"]]
                except KeyError:
                    raise RuntimeError("torchburn: graph references an unsupported placeholder (dynamic-shape symbolic input). Not supported yet.") from None
            if kind == "const":
                return refs["value"]
            raise KeyError(f"torchburn: unknown reference kind {kind!r}")
        if isinstance(refs, (list, tuple)):
            return [self._resolve(x, env) for x in refs]
        return refs

    def _run_eager(self, node: dict[str, Any], env: dict[int, Any]) -> Any:
        _warn_fallback(node["fx_target"])
        args = self._resolve(node.get("fx_args", node["args"]), env)
        kwargs: dict[str, Any] = {}
        raw_kwargs = node.get("fx_kwargs", node.get("kwargs", {}))
        for key, value in raw_kwargs.items():
            kwargs[key] = self._resolve(value, env)
        op = node["fx_op"]
        if op == "call_function":
            fn = self.function_map[node["fx_target"]]
            return fn(*args, **kwargs)
        if op == "call_method":
            return getattr(args[0], node["fx_target"])(*args[1:], **kwargs)
        if op == "call_module":
            return getattr(self.gm, node["fx_target"])(*args, **kwargs)
        if op == "get_attr":
            return getattr(self.gm, node["fx_target"])
        raise RuntimeError(f"torchburn: cannot execute node with fx_op {op!r}")
