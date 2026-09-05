import torch
import torch.nn as nn
import torchburn
from torchburn._compiled import BurnCompiledCallable


def test_compiled_callable_direct_autograd():
    """Verify that calling BurnCompiledCallable directly with requires_grad=True
    does not detach the graph or cause silent grad_fn=None errors."""
    class SimpleModule(nn.Module):
        def forward(self, x):
            return torch.relu(x * 2.0)

    gm = torch.fx.symbolic_trace(SimpleModule())
    x = torch.randn(4, 4, requires_grad=True)
    compiled = BurnCompiledCallable(gm, [x])

    out = compiled(x)
    assert out.requires_grad, "Output should retain requires_grad=True when grad is enabled"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient should not be None"
    assert not torch.all(x.grad == 0), "Gradients should be non-zero"
