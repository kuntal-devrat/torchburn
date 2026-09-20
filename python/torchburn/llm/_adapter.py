"""Adapter that normalizes custom transformer architectures to TorchBurn's expected interface.

Handles:
- Attribute aliasing: maps common naming conventions (wte/embed_tokens, blocks/layers, ln_f/norm)
- Flexible forward() output signatures: extracts logits from various return types
- Graceful fallback when expected attributes are missing
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn


# Attribute alias groups: (canonical_name, [possible_names])
_EMBED_ALIASES = ("embed_tokens", "wte", "token_embeddings", "word_embeddings",
                  "embeddings", "embed", "token_embed", "wte_embed")
_LAYERS_ALIASES = ("layers", "blocks", "h", "transformer_blocks", "decoder_layers",
                   "encoder_layers", "layer", "block")
_NORM_ALIASES = ("norm", "ln_f", "final_layernorm", "final_norm", "model_norm",
                 "layer_norm", "final_layer_norm", "norm_f")
_LM_HEAD_ALIASES = ("lm_head", "head", "output", "output_proj", "fc_out",
                    "linear", "classifier")

_ALIAS_GROUPS = [
    ("embed_tokens", _EMBED_ALIASES),
    ("layers", _LAYERS_ALIASES),
    ("norm", _NORM_ALIASES),
    ("lm_head", _LM_HEAD_ALIASES),
]


def _find_attr(model: nn.Module, aliases: tuple[str, ...]) -> Optional[str]:
    """Find the first matching attribute name from a list of aliases."""
    # Direct attributes
    for name in aliases:
        if hasattr(model, name):
            return name
    # Check inside a nested 'model' or 'transformer' wrapper
    for wrapper_name in ("model", "transformer", "gpt", "backbone"):
        wrapper = getattr(model, wrapper_name, None)
        if wrapper is not None:
            for name in aliases:
                if hasattr(wrapper, name):
                    return f"{wrapper_name}.{name}"
    return None


def _get_nested_attr(model: nn.Module, dotted_name: str) -> Any:
    """Get an attribute that may be nested (e.g. 'model.layers')."""
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def _alias_attr(model: nn.Module, canonical: str, aliases: tuple[str, ...]) -> None:
    """Add a canonical attribute name as an alias if it doesn't exist."""
    if hasattr(model, canonical):
        return  # Already has the canonical name

    found = _find_attr(model, aliases)
    if found is None:
        return  # No match found; skip silently

    try:
        val = _get_nested_attr(model, found)
        # Only set if the model doesn't already have this attribute
        if not hasattr(model, canonical):
            setattr(model, canonical, val)
    except (AttributeError, TypeError):
        pass


class _FlexibleOutputWrapper(nn.Module):
    """Wraps a model's forward() to always return (logits, kv_caches).

    Handles models that return:
    - logits (single tensor)
    - (logits, loss)
    - (logits, loss, aux_loss)
    - (logits, kv_caches)
    - a dataclass/namedtuple with .logits
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self._wrapped = model
        # Proxy all attributes except forward
        self.__dict__["_wrapped"] = model

    def __getattr__(self, name: str) -> Any:
        if name == "_wrapped":
            return self.__dict__["_wrapped"]
        try:
            return getattr(self._wrapped, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Any]:
        result = self._wrapped(*args, **kwargs)

        # Case 1: Single tensor output (logits)
        if isinstance(result, torch.Tensor):
            return result, None

        # Case 2: Dataclass or object with .logits attribute
        if hasattr(result, "logits"):
            logits = result.logits
            kv = getattr(result, "past_key_values", None) or getattr(result, "kv_caches", None)
            return logits, kv

        # Case 3: Tuple/list output
        if isinstance(result, (tuple, list)):
            if len(result) == 0:
                raise ValueError("Model returned empty tuple")
            if len(result) == 1:
                return result[0], None
            # Handle (loss, logits, ...) order common when labels are passed
            if (
                isinstance(result[0], torch.Tensor)
                and isinstance(result[1], torch.Tensor)
                and result[0].ndim <= 1
                and result[1].ndim >= 2
            ):
                return result[1], None
            # Standard (logits, ...)
            logits = result[0]
            second = result[1]
            # If second element is a list of tensors (KV caches), return as-is
            if isinstance(second, (list, tuple)) and len(second) > 0:
                return logits, second
            # Otherwise it's likely a loss or aux output; return logits with no kv
            return logits, None

        # Fallback: assume it's logits
        return result, None

    def parameters(self, recurse: bool = True):
        return self._wrapped.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._wrapped.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self._wrapped.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._wrapped.load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self._wrapped.train(mode)
        return self

    def eval(self):
        self._wrapped.eval()
        return self


def adapt_model_interface(model: nn.Module) -> nn.Module:
    """Adapt a custom model to TorchBurn's expected interface.

    Adds canonical attribute aliases and wraps the forward() output
    to always return (logits, kv_caches).
    """
    for canonical, aliases in _ALIAS_GROUPS:
        _alias_attr(model, canonical, aliases)

    return _FlexibleOutputWrapper(model)
