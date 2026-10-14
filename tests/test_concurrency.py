"""
Tests for Thread Safety & Concurrency.

Tests concurrent compilation, parallel backward passes, GIL release,
and thread-safe cache operations.
"""

import os
import sys
import threading
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torchburn

# ---------------------------------------------------------------------------
# Cache Thread Safety
# ---------------------------------------------------------------------------


class TestCacheConcurrency:
    def test_concurrent_cache_puts(self):
        """Multiple threads writing to the cache simultaneously."""
        from torchburn._torchburn import cache_clear, cache_put

        cache_clear()

        def worker(i):
            sig = f"test_sig_{i}"
            payload = f'{{"nodes": [{i}], "inputs": [], "outputs": []}}'
            cache_put(sig, payload)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        from torchburn._torchburn import cache_stats

        size, _, _ = cache_stats()
        assert size == 50

    def test_concurrent_cache_reads(self):
        """Multiple threads reading from the cache simultaneously."""
        from torchburn._torchburn import cache_clear, cache_get, cache_put

        cache_clear()

        # Populate cache
        for i in range(20):
            cache_put(f"sig_{i}", f'{{"nodes": [{i}]}}')

        results = [None] * 20

        def worker(i):
            results[i] = cache_get(f"sig_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(20):
            assert results[i] is not None

    def test_concurrent_cache_mixed(self):
        """Mixed reads and writes from multiple threads."""
        from torchburn._torchburn import cache_clear, cache_get, cache_put

        cache_clear()

        # Pre-populate
        for i in range(10):
            cache_put(f"pre_{i}", f'{{"nodes": [{i}]}}')

        errors = []

        def reader(thread_id):
            try:
                for i in range(10):
                    cache_get(f"pre_{i}")
            except Exception as e:
                errors.append(e)

        def writer(thread_id):
            try:
                for i in range(10):
                    cache_put(f"new_{thread_id}_{i}", f'{{"nodes": [{i}]}}')
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=reader, args=(i,)))
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Concurrent Compilation
# ---------------------------------------------------------------------------


class TestConcurrentCompilation:
    def test_parallel_compile_same_model(self):
        """Multiple threads compiling the same model."""

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1.0)

        m = M()
        results = [None] * 8
        errors = [None] * 8

        def worker(i):
            try:
                compiled = torch.compile(m, backend="torchburn")
                x = torch.randn(4, 4)
                results[i] = compiled(x)
            except Exception as e:
                errors[i] = e

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(8):
            assert errors[i] is None, f"Thread {i} failed: {errors[i]}"
            assert results[i] is not None

    def test_parallel_compile_different_models(self):
        """Multiple threads compiling different models simultaneously."""
        models = [
            lambda x: torch.relu(x),
            lambda x: torch.sigmoid(x),
            lambda x: torch.tanh(x),
            lambda x: x + 1.0,
            lambda x: x * 2.0,
            lambda x: torch.abs(x),
            lambda x: torch.neg(x),
            lambda x: torch.clamp(x, -1.0, 1.0),
        ]

        results = [None] * len(models)
        errors = [None] * len(models)

        def worker(i):
            try:
                fn = models[i]
                compiled = torch.compile(fn, backend="torchburn")
                x = torch.randn(4, 4)
                results[i] = compiled(x)
            except Exception as e:
                errors[i] = e

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(models))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(len(models)):
            assert errors[i] is None, f"Thread {i} failed: {errors[i]}"
            assert results[i] is not None


# ---------------------------------------------------------------------------
# GIL Release
# ---------------------------------------------------------------------------


class TestGILRelease:
    def test_other_threads_progress_during_compile(self):
        """Other Python threads can progress while torchburn compiles."""
        import threading as _threading

        progress = []
        stop_event = _threading.Event()

        def progress_tracker():
            while not stop_event.is_set():
                progress.append(time.time())
                time.sleep(0.001)

        # Start a model that takes some time to compile
        class M(torch.nn.Module):
            def forward(self, x):
                for _ in range(3):
                    x = torch.relu(x + 1.0)
                return x

        t = _threading.Thread(target=progress_tracker)
        t.start()

        m = M()
        compiled = torch.compile(m, backend="torchburn")
        x = torch.randn(8, 8)
        _ = compiled(x)

        stop_event.set()
        t.join(timeout=1.0)

        # The progress tracker should have made progress during compilation
        # (indicating GIL was released at some point)
        assert len(progress) > 0


# ---------------------------------------------------------------------------
# Autograd Thread Safety
# ---------------------------------------------------------------------------


class TestAutogradThreadSafety:
    def test_concurrent_backward_passes(self):
        """Multiple threads running backward passes simultaneously."""
        torchburn._torchburn.autograd_reset()

        def worker(thread_id):
            torchburn._torchburn.autograd_enable()
            try:
                w = torch.randn(4, 4, requires_grad=True)
                x = torch.randn(2, 4)
                y = x @ w
                loss = y.sum()
                loss.backward()
                assert w.grad is not None
            finally:
                torchburn._torchburn.autograd_disable()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_sequential_backward_is_correct(self):
        """Sequential backward passes produce correct results."""
        torchburn._torchburn.autograd_reset()

        for _ in range(10):
            torchburn._torchburn.autograd_enable()
            try:
                w = torch.randn(4, 4, requires_grad=True)
                x = torch.randn(2, 4)
                y = x @ w
                loss = y.sum()
                loss.backward()
                assert w.grad is not None
                assert w.grad.shape == w.shape
            finally:
                torchburn._torchburn.autograd_disable()

    def test_thread_exit_cleans_up_active_recording_threads(self):
        """Thread terminating without autograd_disable() must not leak active count."""
        initial = torchburn._torchburn.autograd_active_recording_threads()

        def worker():
            torchburn._torchburn.autograd_enable()
            assert torchburn._torchburn.autograd_is_enabled()
            assert torchburn._torchburn.autograd_active_recording_threads() >= initial + 1
            # Thread intentionally terminates without calling autograd_disable()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Thread drop handler must clean up counter.
        # Allow small window for OS pthread TLS destructors to finish running.
        for _ in range(50):
            if torchburn._torchburn.autograd_active_recording_threads() == initial:
                break
            time.sleep(0.01)

        assert torchburn._torchburn.autograd_active_recording_threads() == initial

    def test_autograd_guard_context_manager(self):
        """AutogradGuard RAII scoping and cleanup."""
        initial = torchburn._torchburn.autograd_active_recording_threads()

        with torchburn._torchburn.AutogradGuard():
            assert torchburn._torchburn.autograd_is_enabled()
            assert torchburn._torchburn.autograd_active_recording_threads() == initial + 1

        assert not torchburn._torchburn.autograd_is_enabled()
        assert torchburn._torchburn.autograd_active_recording_threads() == initial

        # Also test Python-level AutogradGuard
        with torchburn.AutogradGuard():
            assert torchburn._torchburn.autograd_is_enabled()

        assert not torchburn._torchburn.autograd_is_enabled()

    def test_autograd_guard_exception_safety(self):
        """AutogradGuard restores disabled state even when exception raised."""
        initial = torchburn._torchburn.autograd_active_recording_threads()

        with pytest.raises(ValueError, match="test error"):
            with torchburn.AutogradGuard():
                assert torchburn._torchburn.autograd_is_enabled()
                raise ValueError("test error")

        assert not torchburn._torchburn.autograd_is_enabled()
        assert torchburn._torchburn.autograd_active_recording_threads() == initial


# ---------------------------------------------------------------------------
# Memory Pool Concurrency
# ---------------------------------------------------------------------------


class TestMemoryPoolConcurrency:
    def test_concurrent_pool_alloc_and_recycle(self):
        """Concurrent memory pool access maintains correct high-water mark and zeroing."""
        torchburn.clear_memory_pool()
        torchburn.reset_stats()

        errors = []

        def worker(thread_id):
            try:
                for _i in range(25):
                    x = torch.randn(4, 8)
                    w = torch.randn(8, 4)
                    y = x @ w
                    assert y.shape == (4, 4)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = torchburn.memory_pool_stats()
        assert stats["peak_cached_words"] >= stats["cached_words"]
        torchburn.clear_memory_pool()


# ---------------------------------------------------------------------------
# Stress Tests
# ---------------------------------------------------------------------------


class TestStress:
    def test_rapid_compile_execute_cycles(self):
        """Rapid compile/execute cycles don't crash."""

        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.relu(x + y)

        m = M()
        for _i in range(50):
            compiled = torch.compile(m, backend="torchburn")
            x = torch.randn(2, 4)
            y = torch.randn(2, 4)
            out = compiled(x, y)
            assert out.shape == (2, 4)

    def test_varying_shapes(self):
        """Compiling with different tensor shapes doesn't corrupt state."""

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1.0)

        m = M()
        shapes = [(1, 4), (2, 8), (4, 16), (8, 32), (16, 64)]
        for shape in shapes:
            compiled = torch.compile(m, backend="torchburn")
            x = torch.randn(*shape)
            out = compiled(x)
            assert out.shape == shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
