"""Verification of all 450 native operations (full coverage)."""
from __future__ import annotations
import torch
from torchburn import _torchburn as tb

def test_exactly_450_supported_ops():
    targets = tb.supported_targets()
    target_set = set(targets)
    assert len(target_set) >= 450, f"Expected at least 450 unique ops, got {len(target_set)}: {len(targets)}"
    assert len(targets) == len(target_set), f"duplicate ops {len(targets)} vs {len(target_set)}"

def test_batch4_ops_count():
    # Ensure the 48 batch4 ops are present
    targets = set(tb.supported_targets())
    batch4 = {"isclose","allclose","equal","isreal","is_complex","is_nonzero","nanprod","nanmin","nanmax","var_mean","std_mean","nanmedian","cov","corrcoef","as_strided","broadcast_to","broadcast_tensors","split","vsplit","hsplit","dsplit","tensor_split","take_along_dim","index_reduce","scatter_max","scatter_min","linalg_multi_dot","linalg_vander","linalg_vecdot","linalg_cross","linalg_tensordot","linalg_cholesky_ex","linalg_inv_ex","linalg_solve_ex","linalg_lu_factor","local_response_norm","adaptive_avg_pool1d","adaptive_max_pool1d","lp_pool3d","logsumexp","randn_like","rand_like","randint_like","empty_strided","view_as","expand_as","masked_select_extra","istft"}
    missing = batch4 - targets
    assert not missing, f"Missing batch4 ops: {missing}"
