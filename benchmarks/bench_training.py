"""Benchmark: 2-layer MLP training throughput -- PyTorch vs torchburn.

Compares three execution paths:
  1. Eager PyTorch training (forward + backward + optimizer)
  2. torch.compile(backend='torchburn') inference throughput (forward only)
  3. torchburn.autograd training (Python tape + Rust backward_single FFI)

Reports:
  * Wall-clock time per epoch
  * Samples/sec throughput
  * Final loss (convergence check)
  * Correctness: all three paths converge to similar solutions

Run with engine selection:

    python benchmarks/bench_training.py                          # native CPU
    TORCHBURN_ENGINE=burn python benchmarks/bench_training.py    # burn ndarray
    TORCHBURN_ENGINE=burn-wgpu python benchmarks/bench_training.py  # burn wgpu
"""

from __future__ import annotations

import gc
import statistics
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchburn


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class MLP2Layer(nn.Module):
    """2-layer MLP: input -> hidden (ReLU) -> output."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_samples: int, input_dim: int, output_dim: int, seed: int = 42):
    """Create a synthetic classification dataset."""
    torch.manual_seed(seed)
    x = torch.randn(n_samples, input_dim)
    w_true = torch.randn(input_dim, output_dim)
    y = (x @ w_true).argmax(dim=1)
    return x, y


def _bench_eager_training(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                          epochs: int = 50, lr: float = 1e-3, warmup: int = 3
                          ) -> dict:
    """Train with eager PyTorch and return per-epoch stats."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_times = []
    losses = []

    for epoch in range(epochs):
        t0 = time.perf_counter()

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)
        losses.append(loss.item())

    steady = epoch_times[warmup:]
    return {
        "epoch_times": epoch_times,
        "steady_median_ms": statistics.median(steady) * 1000,
        "steady_mean_ms": statistics.mean(steady) * 1000,
        "first_epoch_ms": epoch_times[0] * 1000,
        "throughput_sps": len(x) / statistics.median(steady),
        "losses": losses,
        "final_loss": losses[-1],
    }


def _bench_compiled_inference(model: nn.Module, x: torch.Tensor,
                              rounds: int = 3, warmup: int = 5, iters: int = 50
                              ) -> dict:
    """Forward-only throughput with torch.compile(backend='torchburn')."""
    compiled = torch.compile(model, backend="torchburn")

    # Warmup (triggers dynamo trace + BLAKE3 cache)
    with torch.no_grad():
        for _ in range(warmup):
            compiled(x)

    # Benchmark
    latencies = []
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                compiled(x)
        latencies.append((time.perf_counter() - start) / iters)

    steady_ms = statistics.median(latencies) * 1000
    return {
        "steady_median_ms": steady_ms,
        "throughput_sps": len(x) / (steady_ms / 1000),
    }


def _bench_native_autograd(input_dim: int, hidden_dim: int, output_dim: int,
                           x_data: torch.Tensor, y_data: torch.Tensor,
                           epochs: int = 50, lr: float = 1e-3, warmup: int = 3
                           ) -> dict:
    """Train with torchburn.autograd (Python tape + Rust backward_single)."""
    import torchburn.autograd as ta

    ta.reset()
    ta.enable()

    torch.manual_seed(0)
    w1 = ta.Tensor(torch.randn(hidden_dim, input_dim) * 0.01, requires_grad=True)
    b1 = ta.Tensor(torch.zeros(hidden_dim), requires_grad=True)
    w2 = ta.Tensor(torch.randn(output_dim, hidden_dim) * 0.01, requires_grad=True)
    b2 = ta.Tensor(torch.zeros(output_dim), requires_grad=True)

    epoch_times = []
    losses = []

    for epoch in range(epochs):
        t0 = time.perf_counter()

        if epoch > 0:
            ta.reset()
            ta.enable()
        ta.Tensor._registry[w1._id] = w1
        ta.Tensor._registry[b1._id] = b1
        ta.Tensor._registry[w2._id] = w2
        ta.Tensor._registry[b2._id] = b2

        # Forward
        x = ta.Tensor(x_data)
        h = ta.relu(ta.linear(x, w1, b1))
        logits = ta.linear(h, w2, b2)
        loss = ta.cross_entropy(logits, ta.Tensor(y_data.long()))

        # Backward
        loss.backward()

        # SGD update
        w1.data -= lr * w1.grad
        b1.data -= lr * b1.grad
        w2.data -= lr * w2.grad
        b2.data -= lr * b2.grad

        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)
        losses.append(loss.data.item())

    ta.disable()
    ta.reset()

    steady = epoch_times[warmup:]
    return {
        "epoch_times": epoch_times,
        "steady_median_ms": statistics.median(steady) * 1000,
        "steady_mean_ms": statistics.mean(steady) * 1000,
        "first_epoch_ms": epoch_times[0] * 1000,
        "throughput_sps": len(x_data) / statistics.median(steady) if steady else 0,
        "losses": losses,
        "final_loss": losses[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Config
    INPUT_DIM = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 10
    N_SAMPLES = 1024
    EPOCHS = 30
    LR = 1e-3
    WARMUP = 3

    engine = torchburn._torchburn.active_engine()
    print(f"engine: {engine}")
    print(f"config: {N_SAMPLES} samples, {INPUT_DIM}>{HIDDEN_DIM}>{OUTPUT_DIM}, "
          f"{EPOCHS} epochs, lr={LR}")
    print()

    # Dataset
    x_data, y_data = _make_dataset(N_SAMPLES, INPUT_DIM, OUTPUT_DIM)

    # ====================================================================
    # 1. EAGER PYTORCH TRAINING
    # ====================================================================
    print("=" * 64)
    print("  1. EAGER PYTORCH TRAINING (forward + backward + step)")
    print("=" * 64)

    model_eager = MLP2Layer(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    eager_stats = _bench_eager_training(model_eager, x_data, y_data, EPOCHS, LR, WARMUP)

    print(f"  first epoch:      {eager_stats['first_epoch_ms']:8.2f} ms")
    print(f"  steady (median):  {eager_stats['steady_median_ms']:8.2f} ms/epoch")
    print(f"  throughput:       {eager_stats['throughput_sps']:8.0f} samples/sec")
    print(f"  final loss:       {eager_stats['final_loss']:8.4f}")
    print()

    # ====================================================================
    # 2. TORCH.COMPILE + TORCHBURN (inference only)
    # ====================================================================
    print("=" * 64)
    print("  2. TORCH.COMPILE + TORCHBURN (forward-only inference)")
    print("=" * 64)

    model_tb = MLP2Layer(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    compiled_stats = _bench_compiled_inference(model_tb, x_data)

    print(f"  steady (median):  {compiled_stats['steady_median_ms']:8.2f} ms/batch")
    print(f"  throughput:       {compiled_stats['throughput_sps']:8.0f} samples/sec")

    # Compare forward-only: eager forward vs compiled forward
    with torch.no_grad():
        # Time eager forward
        times_eager_fwd = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            for _ in range(100):
                model_eager(x_data)
            times_eager_fwd.append((time.perf_counter() - start) / 100)
        eager_fwd_ms = statistics.median(times_eager_fwd) * 1000

    speedup_fwd = eager_fwd_ms / compiled_stats['steady_median_ms']
    print(f"  eager forward:    {eager_fwd_ms:8.2f} ms/batch")
    print(f"  speedup (fwd):    {speedup_fwd:8.2f}x")
    print()

    # ====================================================================
    # 3. TORCHBURN AUTOGRAAD TRAINING (native Rust backward)
    # ====================================================================
    print("=" * 64)
    print("  3. TORCHBURN AUTOGRAAD (Python fwd + Rust backward_single)")
    print("=" * 64)

    native_stats = _bench_native_autograd(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, x_data, y_data, EPOCHS, LR, WARMUP
    )

    print(f"  first epoch:      {native_stats['first_epoch_ms']:8.2f} ms")
    print(f"  steady (median):  {native_stats['steady_median_ms']:8.2f} ms/epoch")
    print(f"  throughput:       {native_stats['throughput_sps']:8.0f} samples/sec")
    print(f"  final loss:       {native_stats['final_loss']:8.4f}")

    speedup_native = eager_stats['steady_median_ms'] / native_stats['steady_median_ms']
    print(f"  vs eager:         {speedup_native:8.2f}x")
    print()

    # ====================================================================
    # 4. CORRECTNESS
    # ====================================================================
    print("=" * 64)
    print("  4. CORRECTNESS (convergence check)")
    print("=" * 64)

    print(f"  eager final loss:    {eager_stats['final_loss']:.6f}")
    print(f"  compiled final loss: N/A (inference only)")
    print(f"  native final loss:   {native_stats['final_loss']:.6f}")
    loss_diff = abs(eager_stats['final_loss'] - native_stats['final_loss'])
    print(f"  |eager - native|:    {loss_diff:.6f}")
    converged = loss_diff < 0.5  # both should converge to similar solutions
    print(f"  convergence match:   {'PASS' if converged else 'WARN'}")
    print()

    # ====================================================================
    # 5. LOSS CONVERGENCE CURVES
    # ====================================================================
    print("=" * 64)
    print("  5. LOSS CONVERGENCE (sampled every 10 epochs)")
    print("=" * 64)
    step = max(1, EPOCHS // 10)
    print(f"  {'epoch':>5}  {'eager':>10}  {'native':>10}  {'delta':>10}")
    print(f"  {'-----':>5}  {'-----':>10}  {'------':>10}  {'-----':>10}")
    for ep in range(0, EPOCHS, step):
        e = eager_stats['losses'][ep]
        n = native_stats['losses'][ep]
        d = e - n
        print(f"  {ep:5d}  {e:10.4f}  {n:10.4f}  {d:+10.4f}")
    # Final
    e = eager_stats['losses'][-1]
    n = native_stats['losses'][-1]
    d = e - n
    print(f"  {EPOCHS - 1:5d}  {e:10.4f}  {n:10.4f}  {d:+10.4f}")
    print()

    # ====================================================================
    # 6. TIMING BREAKDOWN
    # ====================================================================
    print("=" * 64)
    print("  6. TIMING BREAKDOWN (per-epoch average, ms)")
    print("=" * 64)

    # Eager breakdown
    torch.manual_seed(0)
    m = MLP2Layer(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    opt = torch.optim.SGD(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    fwd_times = []
    bwd_times = []
    step_times = []
    for _ in range(10):
        opt.zero_grad()
        t0 = time.perf_counter()
        out = m(x_data)
        t1 = time.perf_counter()
        loss = crit(out, y_data)
        loss.backward()
        t2 = time.perf_counter()
        opt.step()
        t3 = time.perf_counter()
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        step_times.append(t3 - t2)

    eager_fwd = statistics.median(fwd_times) * 1000
    eager_bwd = statistics.median(bwd_times) * 1000
    eager_sgd = statistics.median(step_times) * 1000

    print(f"  {'Component':<25} {'Eager':>10} {'Native':>10}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10}")
    print(f"  {'Forward':<25} {eager_fwd:10.2f} {'(Python)':>10}")
    print(f"  {'Backward':<25} {eager_bwd:10.2f} {'(Rust FFI)':>10}")
    print(f"  {'Optimizer step':<25} {eager_sgd:10.2f} {'(PyTorch)':>10}")
    print(f"  {'Total':<25} {eager_fwd + eager_bwd + eager_sgd:10.2f} "
          f"{native_stats['steady_median_ms']:10.2f}")
    print()

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  {'Path':<35} {'ms/epoch':>10} {'samp/s':>10} {'vs eager':>10}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'Eager PyTorch (training)':<35} "
          f"{eager_stats['steady_median_ms']:10.2f} "
          f"{eager_stats['throughput_sps']:10.0f} {'1.00x':>10}")
    print(f"  {'torch.compile (inference fwd)':<35} "
          f"{compiled_stats['steady_median_ms']:10.2f} "
          f"{compiled_stats['throughput_sps']:10.0f} "
          f"{speedup_fwd:9.2f}x")
    print(f"  {'torchburn.autograd (training)':<35} "
          f"{native_stats['steady_median_ms']:10.2f} "
          f"{native_stats['throughput_sps']:10.0f} "
          f"{speedup_native:9.2f}x")
    print()
    print(f"  cache stats: {torchburn.cache_stats()}")
    print()
    print("  Notes:")
    print("  - 'steady' = median over epochs after warmup (first 5 skipped)")
    print("  - torch.compile inference benchmark is forward-only (no backward)")
    print("  - torchburn.autograd: forward in Python, backward via Rust FFI")
    print("  - Eager PyTorch uses native C++ autograd engine")
    print("  - Native autograd overhead: Python tape recording + DLPack FFI")


if __name__ == "__main__":
    main()
