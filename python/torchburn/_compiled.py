"""The compiled callable that replaces eager execution (REQ-001)."""

from __future__ import annotations

from typing import Any

import torch

from ._interpreter import _BaseInterpreter


class BurnCompiledCallable(_BaseInterpreter):
    """Replaces eager-mode execution for a traced ``torch.fx`` graph."""

    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        # _BaseInterpreter handles parse, cache, precompute, and graph_handle cleanup
        self._init_interpreter(gm, example_inputs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(
                f"torchburn compiled callable received unexpected keyword arguments {sorted(kwargs)}"
            )
        env: dict[int, Any] = {}
        for node in self.plan["nodes"]:
            if node["op"] == "placeholder":
                if node["pos"] < len(args):
                    env[node["id"]] = args[node["pos"]]
            elif node["op"] == "get_attr":
                env[node["id"]] = getattr(self.gm, node["target"])
        return self._interpret(env)
