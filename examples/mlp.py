"""Minimal end-to-end example: compile a small model with TorchBurn.

The model uses only elementwise ops (mul/add/relu) so the whole graph runs
inside the Rust engine with zero fallbacks. Run with:

    python examples/mlp.py
"""

from __future__ import annotations

import torch
import torchburn


def weight_scaled_mlp(x, w1, b1, w2, b2):
    h = torch.relu(x * w1 + b1)
    return h * w2 + b2


def main() -> None:
    torch.manual_seed(0)
    n = 512
    x = torch.randn(8, n)
    w1 = torch.randn(1, n)
    b1 = torch.randn(1, n)
    w2 = torch.randn(1, n)
    b2 = torch.randn(1, n)

    compiled = torch.compile(weight_scaled_mlp, backend="torchburn")

    out = compiled(x, w1, b1, w2, b2)
    ref = weight_scaled_mlp(x, w1, b1, w2, b2)

    print(f"engine:            {torchburn._torchburn.active_engine()}")
    print(f"outputs match:     {torch.allclose(out, ref)}")
    print(f"cache stats:       {torchburn.cache_stats()}")
    print(f"signature:         {torchburn.cache_stats()['size']} compiled graph(s) cached")


if __name__ == "__main__":
    main()
