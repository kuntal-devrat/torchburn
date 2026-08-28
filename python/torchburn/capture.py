"""Direct graph capture path that bypasses ``torch._dynamo`` entirely."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._interpreter import _BaseInterpreter


class TorchBurnModule(_BaseInterpreter, nn.Module):
    """An ``nn.Module`` wrapper that executes via the Rust engine."""

    def __init__(
        self,
        original: nn.Module,
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
        nn.Module.__init__(self)
        self._original = original
        # _BaseInterpreter handles parse/cache/precompute and graph_handle cleanup
        self._init_interpreter(gm, example_inputs)

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(
                f"TorchBurnModule received unexpected keyword arguments {sorted(kwargs)}"
            )
        with torch.inference_mode():
            env: dict[int, Any] = {}
            for node in self.plan["nodes"]:
                if node["op"] == "placeholder":
                    if node["index"] >= 0:
                        env[node["id"]] = args[node["pos"]]
                elif node["op"] == "get_attr":
                    env[node["id"]] = getattr(self.gm, node["target"])
            return self._interpret(env)


def capture(
    model: nn.Module,
    *example_inputs: torch.Tensor,
    strict: bool = True,
) -> TorchBurnModule:
    """Capture a model's computation graph and return a Rust-executed module."""
    if strict:
        try:
            ep = torch.export.export(model, tuple(example_inputs))
            gm = ep.graph_module
            # Export lifts parameters as placeholders; we use symbolic_trace path
            # which correctly models them as get_attr. If lifted, fallback.
            num_ph = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
            if num_ph != len(example_inputs):
                raise RuntimeError(
                    f"export lifted {num_ph - len(example_inputs)} parameters as placeholders; "
                    f"torchburn capture currently expects get_attr for parameters"
                )
        except Exception as e:
            try:
                gm = torch.fx.symbolic_trace(model)
            except Exception as e2:
                raise RuntimeError(
                    f"torchburn.capture: both export and symbolic_trace failed. "
                    f"Use torch.compile(model, backend='torchburn') instead. "
                    f"Export error: {e}; Trace error: {e2}"
                ) from e2
    else:
        gm = torch.fx.symbolic_trace(model)
    try:
        gm.graph.lint()
    except Exception as e:
        raise RuntimeError(f"torchburn.capture: traced graph is invalid: {e}") from e
    return TorchBurnModule(model, gm, list(example_inputs))
