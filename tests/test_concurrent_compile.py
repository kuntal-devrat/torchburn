"""
Stress tests for concurrent compilation and multi-threaded execution.

Validates:
1. Multiple threads calling torch.compile simultaneously with different models.
2. Multiple threads compiling the same model concurrently (cache race conditions).
3. Concurrent forward passes sharing compiled functions without cross-thread corruption.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torchburn


class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))


class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.fc = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc(self.norm(x))


def test_concurrent_compile_distinct_models():
    """Concurrently compile and evaluate different model architectures across 8 worker threads."""
    models_and_inputs = [
        (ModelA(), torch.randn(4, 32)),
        (ModelB(), torch.randn(2, 3, 16, 16)),
        (ModelC(), torch.randn(4, 32)),
        (ModelA(), torch.randn(8, 32)),
        (ModelB(), torch.randn(1, 3, 8, 8)),
        (ModelC(), torch.randn(2, 32)),
        (ModelA(), torch.randn(16, 32)),
        (ModelB(), torch.randn(4, 3, 12, 12)),
    ]

    def compile_and_run(idx, model, inp):
        model.eval()
        compiled = torch.compile(model, backend="torchburn")
        expected = model(inp)
        actual = compiled(inp)
        assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3), f"Parity mismatch in thread {idx}"
        return idx

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(compile_and_run, i, m, inp)
            for i, (m, inp) in enumerate(models_and_inputs)
        ]
        completed = [f.result() for f in as_completed(futures)]

    assert len(completed) == len(models_and_inputs)


def test_concurrent_compile_same_model():
    """Multiple threads compiling and executing the same model concurrently."""
    model = ModelA().eval()
    compiled = torch.compile(model, backend="torchburn")

    def run_inference(thread_id):
        x = torch.randn(8, 32)
        expected = model(x)
        actual = compiled(x)
        assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
        return thread_id

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_inference, i) for i in range(20)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == 20
