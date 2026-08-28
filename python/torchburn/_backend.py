"""Backend registration for ``torch.compile(..., backend="torchburn")`` (REQ-001)."""

from __future__ import annotations

import logging

import torch

from ._compiled import BurnCompiledCallable

_LOG = logging.getLogger("torchburn")


def _compile_boxed(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    try:
        from torch._functorch.aot_autograd import make_boxed_func
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
