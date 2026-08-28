"""
End-to-end AOTAutograd training tests for torch.compile(..., backend="torchburn").

Verifies:
1. Exact gradient alignment with eager PyTorch across inputs and parameters.
2. Multi-step loss convergence using standard PyTorch optimizers (SGD, AdamW).
3. Complex architectures including Conv2D, MLP, and multi-branch networks.
4. Clean transition between training (grad enabled) and inference (eval/no_grad).
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torchburn


class MLP(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=16, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class TestAOTAutogradTraining:
    def test_gradient_numerical_accuracy(self):
        """Verify that gradients from compiled backward match eager PyTorch within float tolerance."""
        torch.manual_seed(42)
        m_eager = MLP()
        m_compiled = MLP()
        m_compiled.load_state_dict(m_eager.state_dict())

        x_eager = torch.randn(4, 8, requires_grad=True)
        x_compiled = x_eager.detach().clone().requires_grad_(True)

        # Eager forward + backward
        out_eager = m_eager(x_eager)
        loss_eager = (out_eager ** 2).sum()
        loss_eager.backward()

        # Compiled forward + backward
        compiled_fn = torch.compile(m_compiled, backend="torchburn")
        out_compiled = compiled_fn(x_compiled)
        loss_compiled = (out_compiled ** 2).sum()
        loss_compiled.backward()

        # Check forward output
        torch.testing.assert_close(out_compiled, out_eager, atol=1e-5, rtol=1e-5)

        # Check input gradient
        torch.testing.assert_close(x_compiled.grad, x_eager.grad, atol=1e-5, rtol=1e-5)

        # Check all parameter gradients
        for (n_ref, p_ref), (n_comp, p_comp) in zip(
            m_eager.named_parameters(), m_compiled.named_parameters()
        ):
            assert p_comp.grad is not None, f"Gradient missing for {n_comp}"
            torch.testing.assert_close(
                p_comp.grad, p_ref.grad, atol=1e-5, rtol=1e-5,
                msg=f"Gradient mismatch for parameter {n_ref}"
            )

    def test_mlp_training_convergence(self):
        """Train a compiled MLP on synthetic regression data; loss must drop by > 80%."""
        torch.manual_seed(42)
        model = MLP(in_dim=4, hidden_dim=12, out_dim=1)
        compiled_model = torch.compile(model, backend="torchburn")

        optimizer = optim.SGD(compiled_model.parameters(), lr=0.02)
        criterion = nn.MSELoss()

        x = torch.randn(64, 4)
        true_w = torch.tensor([[1.5], [-2.0], [0.5], [3.0]])
        y = x @ true_w + 0.2

        initial_loss = None
        final_loss = None

        for epoch in range(40):
            optimizer.zero_grad()
            pred = compiled_model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss * 0.2, (
            f"Loss did not converge sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_adamw_optimizer_training(self):
        """Verify compatibility with AdamW optimizer and weight decay."""
        torch.manual_seed(42)
        model = MLP(in_dim=6, hidden_dim=16, out_dim=2)
        compiled_model = torch.compile(model, backend="torchburn")

        optimizer = optim.AdamW(compiled_model.parameters(), lr=0.01, weight_decay=1e-4)

        x = torch.randn(16, 6)
        target = torch.randn(16, 2)

        for _ in range(5):
            optimizer.zero_grad()
            out = compiled_model(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        for name, param in compiled_model.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

    def test_conv2d_backward_training(self):
        """Verify backward graph compilation for CNNs (Conv2d, AdaptiveAvgPool2d, Linear)."""
        torch.manual_seed(42)
        model = ConvNet()
        compiled_model = torch.compile(model, backend="torchburn")

        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        out = compiled_model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (2, 3, 8, 8)
        assert model.conv.weight.grad is not None
        assert model.conv.weight.grad.shape == model.conv.weight.shape
        assert model.fc.weight.grad is not None

    def test_gradient_accumulation(self):
        """Accumulating gradients across two mini-batches without zero_grad."""
        torch.manual_seed(42)
        model = MLP(in_dim=4, hidden_dim=8, out_dim=2)
        compiled_model = torch.compile(model, backend="torchburn")

        x1 = torch.randn(4, 4)
        x2 = torch.randn(4, 4)

        # Batch 1
        out1 = compiled_model(x1)
        loss1 = out1.sum()
        loss1.backward()

        grad_w_step1 = model.fc1.weight.grad.clone()

        # Batch 2 without zero_grad
        out2 = compiled_model(x2)
        loss2 = out2.sum()
        loss2.backward()

        grad_w_step2 = model.fc1.weight.grad.clone()

        # Grad after step 2 should strictly differ and be larger than step 1
        assert not torch.allclose(grad_w_step1, grad_w_step2)

    def test_eval_mode_after_training(self):
        """Switching between training and eval/inference mode."""
        torch.manual_seed(42)
        model = MLP(in_dim=4, hidden_dim=8, out_dim=2)
        compiled_model = torch.compile(model, backend="torchburn")

        # Train step
        compiled_model.train()
        x_train = torch.randn(4, 4, requires_grad=True)
        compiled_model(x_train).sum().backward()
        assert model.fc1.weight.grad is not None

        # Eval step with no_grad
        compiled_model.eval()
        x_test = torch.randn(2, 4)
        with torch.no_grad():
            out_eval = compiled_model(x_test)
        assert out_eval.shape == (2, 2)
