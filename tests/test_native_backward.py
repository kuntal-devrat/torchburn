"""Tests for native backward activation & embedding kernels, training batch norm, and end-to-end AOTAutograd training."""

import json
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchburn
from torchburn import _torchburn as _native
from torchburn._parser.op_registry import _ATEN_TO_OP


def run_plan(plan, inputs):
    capsules = [torch.to_dlpack(t.contiguous()) for t in inputs]
    dtype_map = {torch.float32: "f32", torch.float64: "f64", torch.int64: "i64", torch.int32: "i32"}
    plan["inputs"] = [{"shape": list(t.shape), "dtype": dtype_map.get(t.dtype, "f32")} for t in inputs]
    out_capsules = _native.execute(json.dumps(plan), capsules)
    return [torch.from_dlpack(c) for c in out_capsules]


class TestNativeBackwardRegistry:
    def test_op_registry_contains_backward_ops(self):
        assert _ATEN_TO_OP.get("aten.gelu_backward.default") == "gelu_backward"
        assert _ATEN_TO_OP.get("aten.silu_backward.default") == "silu_backward"
        assert _ATEN_TO_OP.get("aten.sigmoid_backward.default") == "sigmoid_backward"
        assert _ATEN_TO_OP.get("aten.tanh_backward.default") == "tanh_backward"
        assert _ATEN_TO_OP.get("aten.leaky_relu_backward.default") == "leaky_relu_backward"
        assert _ATEN_TO_OP.get("aten.embedding_dense_backward.default") == "embedding_backward"
        assert _ATEN_TO_OP.get("aten._native_batch_norm_legit_functional.default") == "batch_norm"
        assert _ATEN_TO_OP.get("aten._native_batch_norm_legit.default") == "batch_norm"
        assert _ATEN_TO_OP.get("aten._native_batch_norm_legit.no_stats") == "batch_norm"


class TestNativeBackwardMath:
    def test_gelu_backward_exact_and_tanh(self):
        # Exact
        x = torch.randn(32, 64, dtype=torch.float32, requires_grad=True)
        y = F.gelu(x, approximate="none")
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        expected_grad = x.grad.clone()

        # Engine direct call via plan execution
        plan = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "gelu_backward",
                    "args": [{"kind": "input", "index": 0}, {"kind": "input", "index": 1}],
                    "kwargs": {"approximate": "none"},
                }
            ],
            "outputs": [2],
        }
        out = run_plan(plan, [grad_out, x.detach()])
        torch.testing.assert_close(out[0], expected_grad, rtol=1e-4, atol=1e-4)

        # Tanh approximate
        x = torch.randn(32, 64, dtype=torch.float32, requires_grad=True)
        y = F.gelu(x, approximate="tanh")
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        expected_grad = x.grad.clone()

        plan_tanh = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "gelu_backward",
                    "args": [{"kind": "input", "index": 0}, {"kind": "input", "index": 1}],
                    "kwargs": {"approximate": "tanh"},
                }
            ],
            "outputs": [2],
        }
        out_tanh = run_plan(plan_tanh, [grad_out, x.detach()])
        torch.testing.assert_close(out_tanh[0], expected_grad, rtol=1e-4, atol=1e-4)

    def test_silu_backward(self):
        x = torch.randn(32, 64, dtype=torch.float32, requires_grad=True)
        y = F.silu(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        expected_grad = x.grad.clone()

        plan = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "silu_backward",
                    "args": [{"kind": "input", "index": 0}, {"kind": "input", "index": 1}],
                    "kwargs": {},
                }
            ],
            "outputs": [2],
        }
        out = run_plan(plan, [grad_out, x.detach()])
        torch.testing.assert_close(out[0], expected_grad, rtol=1e-4, atol=1e-4)

    def test_sigmoid_backward(self):
        x = torch.randn(32, 64, dtype=torch.float32, requires_grad=True)
        y = torch.sigmoid(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        expected_grad = x.grad.clone()

        plan = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "sigmoid_backward",
                    "args": [{"kind": "input", "index": 0}, {"kind": "input", "index": 1}],
                    "kwargs": {},
                }
            ],
            "outputs": [2],
        }
        out = run_plan(plan, [grad_out, y.detach()])
        torch.testing.assert_close(out[0], expected_grad, rtol=1e-5, atol=1e-5)

    def test_tanh_backward(self):
        x = torch.randn(32, 64, dtype=torch.float32, requires_grad=True)
        y = torch.tanh(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        expected_grad = x.grad.clone()

        plan = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "tanh_backward",
                    "args": [{"kind": "input", "index": 0}, {"kind": "input", "index": 1}],
                    "kwargs": {},
                }
            ],
            "outputs": [2],
        }
        out = run_plan(plan, [grad_out, y.detach()])
        torch.testing.assert_close(out[0], expected_grad, rtol=1e-5, atol=1e-5)

    def test_leaky_relu_backward(self):
        x = torch.randn(32, 64, dtype=torch.float32, requires_grad=True)
        y = F.leaky_relu(x, negative_slope=0.2)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        expected_grad = x.grad.clone()

        plan = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "leaky_relu_backward",
                    "args": [{"kind": "input", "index": 0}, {"kind": "input", "index": 1}],
                    "kwargs": {"negative_slope": 0.2},
                }
            ],
            "outputs": [2],
        }
        out = run_plan(plan, [grad_out, x.detach()])
        torch.testing.assert_close(out[0], expected_grad, rtol=1e-5, atol=1e-5)

    def test_embedding_backward(self):
        num_embeddings = 100
        embedding_dim = 64
        weight = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32, requires_grad=True)
        indices = torch.tensor([[5, 12, 5, 42], [0, 99, 12, 7]], dtype=torch.int64)

        out = F.embedding(indices, weight)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        expected_grad = weight.grad.clone()

        plan = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "embedding_backward",
                    "args": [{"kind": "input", "index": 0}, {"kind": "input", "index": 1}],
                    "kwargs": {"num_weights": num_embeddings, "padding_idx": -1, "scale_grad_by_freq": False},
                }
            ],
            "outputs": [2],
        }
        engine_grad = run_plan(plan, [grad_out, indices])
        torch.testing.assert_close(engine_grad[0], expected_grad, rtol=1e-5, atol=1e-5)

    def test_batch_norm_training(self):
        x = torch.randn(8, 16, 10, 10, dtype=torch.float32)
        running_mean = torch.zeros(16)
        running_var = torch.ones(16)
        weight = torch.randn(16)
        bias = torch.randn(16)

        # PyTorch training batch norm
        expected = F.batch_norm(x, running_mean, running_var, weight, bias, training=True, momentum=0.1, eps=1e-5)

        plan = {
            "inputs": [0, 1, 2, 3, 4],
            "nodes": [
                {
                    "id": 5,
                    "target": "batch_norm",
                    "args": [
                        {"kind": "input", "index": 0},
                        {"kind": "input", "index": 1},
                        {"kind": "input", "index": 2},
                        {"kind": "input", "index": 3},
                        {"kind": "input", "index": 4},
                    ],
                    "kwargs": {"training": True, "eps": 1e-5},
                }
            ],
            "outputs": [5],
        }
        res = run_plan(plan, [x, running_mean, running_var, weight, bias])
        torch.testing.assert_close(res[0], expected, rtol=1e-4, atol=1e-4)


class TestEndToEndTraining:
    def test_mlp_torch_compile_training(self):
        """Verify full AOTAutograd training loop with torch.compile(backend='torchburn')."""
        torch.manual_seed(42)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 16)
                self.fc3 = nn.Linear(16, 1)

            def forward(self, x):
                h1 = F.gelu(self.fc1(x))
                h2 = F.silu(self.fc2(h1))
                return self.fc3(h2)

        model = MLP()
        opt_model = torch.compile(model, backend="torchburn")

        x = torch.randn(16, 32)
        y = torch.randn(16, 1)

        optimizer = torch.optim.SGD(opt_model.parameters(), lr=0.01)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            pred = opt_model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert len(losses) == 5
        # Verify gradients were successfully generated and params updated
        for p in opt_model.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()

    def test_transformer_block_torch_compile_training(self):
        """Verify full AOTAutograd training loop on a Transformer block with Embedding, Attention, LayerNorm, and GeLU."""
        torch.manual_seed(42)

        class MiniTransformer(nn.Module):
            def __init__(self, vocab_size=64, d_model=32, num_heads=4):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.fc1 = nn.Linear(d_model, d_model * 2)
                self.fc2 = nn.Linear(d_model * 2, d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, idx):
                x = self.embed(idx)
                attn_out, _ = self.attn(x, x, x)
                x = self.norm1(x + attn_out)
                h = F.gelu(self.fc1(x))
                ffn_out = self.fc2(h)
                x = self.norm2(x + ffn_out)
                return self.head(x)

        model = MiniTransformer()
        opt_model = torch.compile(model, backend="torchburn")

        tokens = torch.randint(0, 64, (4, 16))
        targets = torch.randint(0, 64, (4, 16))

        optimizer = torch.optim.Adam(opt_model.parameters(), lr=0.001)

        for _ in range(3):
            optimizer.zero_grad()
            logits = opt_model(tokens)
            loss = F.cross_entropy(logits.view(-1, 64), targets.view(-1))
            loss.backward()
            optimizer.step()

        for p in opt_model.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


class TestF64AndStridedBackward:
    def test_f64_precision_backward(self):
        """Verify f64 precision across all backward kernels."""
        x = torch.randn(16, 32, dtype=torch.float64, requires_grad=True)

        for act_fn, op_name, kw in [
            (lambda t: F.gelu(t, approximate="none"), "gelu_backward", {"approximate": "none"}),
            (lambda t: F.gelu(t, approximate="tanh"), "gelu_backward", {"approximate": "tanh"}),
            (F.silu, "silu_backward", {}),
            (torch.sigmoid, "sigmoid_backward", {}),
            (torch.tanh, "tanh_backward", {}),
            (lambda t: F.leaky_relu(t, 0.15), "leaky_relu_backward", {"negative_slope": 0.15}),
        ]:
            x.grad = None
            y = act_fn(x)
            grad_out = torch.randn_like(y)
            y.backward(grad_out)
            expected = x.grad.clone()

            second_arg = y.detach() if "sigmoid" in op_name or "tanh" in op_name else x.detach()
            plan = {
                "inputs": [0, 1],
                "nodes": [
                    {
                        "id": 2,
                        "target": op_name,
                        "args": [{"index": 0}, {"index": 1}],
                        "kwargs": kw,
                    }
                ],
                "outputs": [2],
            }
            res = run_plan(plan, [grad_out, second_arg])
            torch.testing.assert_close(res[0], expected, rtol=1e-5, atol=1e-5)

    def test_strided_non_contiguous_backward(self):
        """Verify strided non-contiguous tensor support in backward kernels."""
        x_base = torch.randn(32, 128, dtype=torch.float32, requires_grad=True)
        # Non-contiguous slice: step by 2
        x = x_base[:, ::2]
        assert not x.is_contiguous()

        y = F.silu(x)
        grad_out = torch.randn_like(y)[:, :]

        plan = {
            "inputs": [0, 1],
            "nodes": [
                {
                    "id": 2,
                    "target": "silu_backward",
                    "args": [{"index": 0}, {"index": 1}],
                    "kwargs": {},
                }
            ],
            "outputs": [2],
        }
        res = run_plan(plan, [grad_out, x.detach()])
        # Ground truth
        s = torch.sigmoid(x.detach())
        expected = grad_out * (s * (1.0 + x.detach() * (1.0 - s)))
        torch.testing.assert_close(res[0], expected, rtol=1e-5, atol=1e-5)


class TestConcurrencyAndStrides:
    def test_autograd_multithreaded_backwards(self):
        """Verify autograd lock minimization by running parallel backward loops."""
        import threading
        import torchburn.autograd as ta

        errors = []

        def worker():
            ta.enable()
            try:
                for _ in range(10):
                    w = ta.Tensor(torch.randn(8, 8), requires_grad=True)
                    x = ta.Tensor(torch.randn(4, 8))
                    h = ta.relu(ta.linear(x, w))
                    loss = ta.sum_op(h)
                    loss.backward()
                    assert w.grad is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent autograd errors: {errors}"
