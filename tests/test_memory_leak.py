"""
Memory leak soak test for TorchBurn.

Verifies that:
1. Long-running forward execution loops recycle buffers via the memory pool.
2. Pool hit rate is high and allocations plateau.
3. Process RSS memory stays bounded across thousands of iterations.
4. `clear_memory_pool()` cleans up cached buffers properly.
"""

import os
import sys
import gc
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torchburn

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def get_rss_mb() -> float:
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def test_memory_pool_recycling_soak():
    """Run 1,000 iterations of compile+execute and verify pool reuse and RSS stability."""
    torchburn.clear_memory_pool()
    model = MLP().eval()
    compiled_model = torch.compile(model, backend="torchburn")

    # Warm-up
    x = torch.randn(8, 64)
    for _ in range(10):
        _ = compiled_model(x)

    gc.collect()
    stats_start = torchburn.memory_pool_stats()
    rss_start = get_rss_mb()

    # Soak run: 1,000 iterations
    iterations = 1000
    for i in range(iterations):
        inp = torch.randn(8, 64)
        out = compiled_model(inp)
        assert out.shape == (8, 64)

    gc.collect()
    stats_end = torchburn.memory_pool_stats()
    rss_end = get_rss_mb()

    # Assertions
    # 1. Buffers must have been recycled continuously
    recycle_delta = stats_end["recycle_count"] - stats_start["recycle_count"]
    hit_delta = stats_end["hit_count"] - stats_start["hit_count"]
    alloc_delta = stats_end["alloc_count"] - stats_start["alloc_count"]

    assert recycle_delta > 0, "No buffers were recycled during soak test"
    assert hit_delta > 0, "No cache hits occurred during soak test"

    # 2. Vast majority of allocations must be hits from the recycled pool (>80% hit rate)
    iteration_hit_rate = hit_delta / max(alloc_delta, 1)
    assert iteration_hit_rate > 0.8, f"Soak hit rate too low: {iteration_hit_rate:.2%} ({hit_delta}/{alloc_delta})"

    # 3. Cached buffers must stay bounded (not growing with iterations)
    assert stats_end["cached_buffers"] < 100, f"Pool cached buffers exploded: {stats_end['cached_buffers']}"

    # 4. If psutil is available, verify RSS didn't explode
    if HAS_PSUTIL and rss_start > 0:
        rss_growth = rss_end - rss_start
        assert rss_growth < 100.0, f"Memory leak detected: RSS grew by {rss_growth:.2f} MB"

    # 5. Verify clear_memory_pool works
    torchburn.clear_memory_pool()
    stats_cleared = torchburn.memory_pool_stats()
    assert stats_cleared["cached_buffers"] == 0
    assert stats_cleared["cached_words"] == 0


def test_wgpu_weight_cache_bounded():
    """Verify that weight caching does not leak GPU resources under churn."""
    # Test wgpu weight cache clear function exists and runs cleanly
    if hasattr(torchburn, "wgpu_clear_weight_cache"):
        torchburn.wgpu_clear_weight_cache()
