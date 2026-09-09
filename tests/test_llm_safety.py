"""Regression tests for explicit failure handling in LLM utilities."""

import numpy as np
import pytest

from torchburn.llm.gguf_loader import (
    GGUFLoader,
    GGML_TYPE_Q5_0,
    _dequant_q4_k,
    _dequant_q6_k,
)
from torchburn.llm.loader import ModelLoader
from torchburn.llm.tokenizer import UniversalTokenizer


class _FailingTokenizer:
    def encode(self, text, add_special_tokens=False):
        raise ValueError("bad tokenizer state")


class _InvalidResultTokenizer:
    def encode(self, text, add_special_tokens=False):
        return object()


def test_tokenizer_encode_reports_backend_errors():
    tokenizer = UniversalTokenizer(_FailingTokenizer())
    with pytest.raises(RuntimeError, match="failed to encode"):
        tokenizer.encode("hello")


def test_tokenizer_rejects_invalid_backend_result():
    tokenizer = UniversalTokenizer(_InvalidResultTokenizer())
    with pytest.raises(TypeError, match="unsupported encode result"):
        tokenizer.encode("hello")


def test_q6_k_uses_the_complete_210_byte_block():
    values = _dequant_q6_k(bytes(210), 256)
    assert values.shape == (256,)
    assert values.dtype == np.float32
    assert np.all(values == 0)


def test_quantized_dequantizers_reject_truncated_data():
    with pytest.raises(ValueError, match="Q4_K data"):
        _dequant_q4_k(bytes(143), 256)
    with pytest.raises(ValueError, match="Q6_K data"):
        _dequant_q6_k(bytes(209), 256)


def test_local_only_model_resolution_never_falls_back_to_network(tmp_path):
    with pytest.raises(FileNotFoundError, match="local files only"):
        ModelLoader._resolve_files(str(tmp_path / "missing-model"), local_files_only=True)


def test_local_only_tokenizer_resolution_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="local files only"):
        UniversalTokenizer.from_pretrained(str(tmp_path / "missing-tokenizer"), local_files_only=True)


def test_gguf_unsupported_types_are_not_zero_filled():
    loader = object.__new__(GGUFLoader)
    loader._data_offset = 0
    loader._raw = bytes(128)
    loader.tensor_info = {
        "weights": {
            "offset": 0,
            "n_elem": 32,
            "ggml_type": GGML_TYPE_Q5_0,
            "shape": [32],
        }
    }

    with pytest.raises(NotImplementedError, match="not supported"):
        loader.get_tensor("weights")
