"""Production Readiness & Packaging Assessment Tests.

Validates deep non-trivial multi-dimensional tensor assertions for:
1. linalg_solve: non-trivial 2D (multiple RHS columns) and 3D batched solve.
2. combinations: 1D/multi-element combinations with and without replacement.
3. pad: 3D and 4D tensor padding across various spatial specifications.
4. gru_cell: batched multi-dimensional GRU recurrent steps via eager fallback.
5. Fallback safety and exception inheritance hierarchy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchburn
from torchburn import UnsupportedOpError
from torchburn import _torchburn as native


class TestLinalgSolveMultiDim:
    """Non-trivial multi-dimensional assertions for linalg_solve."""

    def test_linalg_solve_non_trivial_2d_multiple_rhs(self):
        """Test non-trivial 8x8 matrix with 5 RHS columns (8, 5) natively."""
        torch.manual_seed(42)
        # Construct well-conditioned positive definite matrix
        M = torch.randn(8, 8)
        A = M @ M.T + torch.eye(8) * 2.0
        B = torch.randn(8, 5)

        payload = {
            "inputs": [
                {"shape": list(A.shape), "dtype": "f32"},
                {"shape": list(B.shape), "dtype": "f32"},
            ],
            "nodes": [
                {
                    "id": 0,
                    "target": "linalg_solve",
                    "args": [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}],
                }
            ],
            "outputs": [0],
        }
        res = native.execute_from_dict(payload, [A.__dlpack__(), B.__dlpack__()])
        out = torch.from_dlpack(res[0])
        expected = torch.linalg.solve(A, B)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_linalg_solve_torch_compile_e2e(self):
        """Test linalg_solve compiled via torch.compile with non-trivial inputs."""
        torch.manual_seed(123)
        M = torch.randn(6, 6)
        A = M @ M.T + torch.eye(6) * 1.5
        B = torch.randn(6, 4)

        def solve_fn(a, b):
            return torch.linalg.solve(a, b)

        compiled_fn = torch.compile(solve_fn, backend="torchburn")
        out = compiled_fn(A, B)
        expected = torch.linalg.solve(A, B)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_linalg_solve_3d_batched_fallback(self):
        """Verify 3D batched linalg_solve (4, 6, 6) cleanly routes to eager fallback."""
        torch.manual_seed(99)
        # Batched positive-definite matrices (4, 6, 6)
        M = torch.randn(4, 6, 6)
        A = torch.matmul(M, M.transpose(-2, -1)) + torch.eye(6).unsqueeze(0) * 2.0
        B = torch.randn(4, 6, 3)

        def batched_solve_fn(a, b):
            return torch.linalg.solve(a, b)

        compiled_fn = torch.compile(batched_solve_fn, backend="torchburn")
        out = compiled_fn(A, B)
        expected = torch.linalg.solve(A, B)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


class TestCombinationsMultiDim:
    """Non-trivial assertions for combinations and cartesian products."""

    def test_combinations_1d_non_trivial_e2e(self):
        """Verify torch.combinations on 10 elements (r=3, 120 tuples) under torch.compile."""
        x = torch.arange(10, dtype=torch.float32)

        def comb_fn(t):
            return torch.combinations(t, r=3)

        compiled_fn = torch.compile(comb_fn, backend="torchburn")
        out = compiled_fn(x)
        expected = torch.combinations(x, r=3)
        assert out.shape == (120, 3)
        torch.testing.assert_close(out, expected)

    def test_combinations_with_replacement(self):
        """Verify torch.combinations with replacement under torch.compile."""
        x = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])

        def comb_fn(t):
            return torch.combinations(t, r=2, with_replacement=True)

        compiled_fn = torch.compile(comb_fn, backend="torchburn")
        out = compiled_fn(x)
        expected = torch.combinations(x, r=2, with_replacement=True)
        assert out.shape == (15, 2)
        torch.testing.assert_close(out, expected)

    def test_cartesian_prod_fallback(self):
        """Verify torch.cartesian_prod under torch.compile."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        c = torch.tensor([6.0, 7.0, 8.0, 9.0])

        def prod_fn(x, y, z):
            return torch.cartesian_prod(x, y, z)

        compiled_fn = torch.compile(prod_fn, backend="torchburn")
        out = compiled_fn(a, b, c)
        expected = torch.cartesian_prod(a, b, c)
        assert out.shape == (24, 3)
        torch.testing.assert_close(out, expected)


class TestPadMultiDim:
    """Non-trivial multi-dimensional assertions for pad operators."""

    def test_pad_3d_constant_native(self):
        """Test constant padding on 3D tensor (3, 7, 11) with asymmetric padding."""
        torch.manual_seed(77)
        x = torch.randn(3, 7, 11)
        # Pad last two dimensions: (pad_left, pad_right, pad_top, pad_bottom) = (2, 3, 1, 4)
        pad_spec = [2, 3, 1, 4]

        payload = {
            "inputs": [{"shape": list(x.shape), "dtype": "f32"}],
            "nodes": [
                {
                    "id": 0,
                    "target": "constant_pad_nd",
                    "args": [{"kind": "slot", "index": 0}],
                    "kwargs": {"pad": pad_spec, "value": 1.25},
                }
            ],
            "outputs": [0],
        }
        res = native.execute_from_dict(payload, [x.__dlpack__()])
        out = torch.from_dlpack(res[0])
        expected = F.pad(x, pad_spec, mode="constant", value=1.25)
        assert out.shape == (3, 7 + 1 + 4, 11 + 2 + 3)
        torch.testing.assert_close(out, expected)

    def test_pad_4d_nchw_spatial_padding_e2e(self):
        """Test 4D NCHW spatial padding (2, 4, 16, 16) under torch.compile."""
        torch.manual_seed(88)
        x = torch.randn(2, 4, 16, 16)
        pad_spec = [1, 2, 3, 4]  # left=1, right=2, top=3, bottom=4

        def pad_fn(t):
            return F.pad(t, pad_spec, mode="constant", value=0.5)

        compiled_fn = torch.compile(pad_fn, backend="torchburn")
        out = compiled_fn(x)
        expected = F.pad(x, pad_spec, mode="constant", value=0.5)
        assert out.shape == (2, 4, 16 + 3 + 4, 16 + 1 + 2)
        torch.testing.assert_close(out, expected)

    def test_pad_non_constant_modes_fallback(self):
        """Verify non-constant padding modes (reflect, replicate) fall back safely."""
        torch.manual_seed(99)
        x = torch.randn(2, 3, 8, 8)

        def reflect_pad(t):
            return F.pad(t, (1, 1, 1, 1), mode="reflect")

        def replicate_pad(t):
            return F.pad(t, (2, 2, 2, 2), mode="replicate")

        compiled_reflect = torch.compile(reflect_pad, backend="torchburn")
        compiled_replicate = torch.compile(replicate_pad, backend="torchburn")

        out_reflect = compiled_reflect(x)
        expected_reflect = F.pad(x, (1, 1, 1, 1), mode="reflect")
        torch.testing.assert_close(out_reflect, expected_reflect)

        out_replicate = compiled_replicate(x)
        expected_replicate = F.pad(x, (2, 2, 2, 2), mode="replicate")
        torch.testing.assert_close(out_replicate, expected_replicate)


class TestGruCellMultiDim:
    """Non-trivial multi-dimensional assertions for GRU cell."""

    def test_gru_cell_batched_e2e(self):
        """Verify batched GRUCell forward pass (batch_size=8, input_dim=32, hidden_dim=64)."""
        torch.manual_seed(55)
        cell = nn.GRUCell(input_size=32, hidden_size=64)
        x = torch.randn(8, 32)
        h = torch.randn(8, 64)

        compiled_cell = torch.compile(cell, backend="torchburn")
        h_next = compiled_cell(x, h)
        expected = cell(x, h)
        torch.testing.assert_close(h_next, expected, atol=1e-5, rtol=1e-5)

    def test_gru_cell_unrolled_sequence_fallback(self):
        """Verify unrolling a sequence of 5 time-steps through GRUCell under torch.compile."""
        torch.manual_seed(66)
        cell = nn.GRUCell(input_size=16, hidden_size=32)
        seq = torch.randn(5, 4, 16)  # (seq_len=5, batch_size=4, input_size=16)
        h = torch.zeros(4, 32)

        def rnn_loop(inputs, h_init):
            hidden = h_init
            outputs = []
            for t in range(inputs.shape[0]):
                hidden = cell(inputs[t], hidden)
                outputs.append(hidden)
            return torch.stack(outputs, dim=0)

        compiled_rnn = torch.compile(rnn_loop, backend="torchburn")
        out = compiled_rnn(seq, h)
        expected = rnn_loop(seq, h)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestProductionReadinessContracts:
    """Verification of production readiness contracts: error hierarchy and fallback safety."""

    def test_unsupported_op_error_hierarchy(self):
        """Verify UnsupportedOpError inherits from RuntimeError for clean catchability."""
        err = UnsupportedOpError("test_op: custom reasoning")
        assert isinstance(err, RuntimeError)
        assert isinstance(err, Exception)
        assert "test_op: custom reasoning" in str(err)

    def test_cache_stats_contract(self):
        """Verify cache_stats returns valid structured telemetry."""
        stats = torchburn.cache_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats or "cache_hits" in stats or "cached_graphs" in stats

    def test_supported_targets_non_empty(self):
        """Verify native engine reports supported targets."""
        targets = native.supported_targets()
        assert isinstance(targets, list)
        assert len(targets) > 100
        assert "matmul" in targets
        assert "add" in targets
        assert "relu" in targets
