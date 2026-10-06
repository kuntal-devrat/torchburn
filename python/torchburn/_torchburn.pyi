"""Type stubs for the native _torchburn PyO3 extension module."""

from __future__ import annotations
from typing import Any, Sequence, Tuple, Dict, List, Optional

__version__: str

class UnsupportedOpError(RuntimeError): ...

class RustQwenDecoder:
    def __init__(
        self,
        embed_tokens: Any,
        layers_data: List[List[Any]],
        final_norm_w: Any,
        lm_head_w: Any,
        lm_head_s: Any,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        group_size: int = 64,
        rms_norm_eps: float = 1e-6,
        max_seq_len: int = 2048,
        rope_theta: float = 1000000.0,
    ) -> None: ...
    def step(self, token_id: int, offset: int) -> None: ...
    def sample(self, temperature: float = 0.0, top_p: float = 1.0) -> int: ...
    def sample_and_step(self, token_id: int, offset: int, temperature: float = 0.0, top_p: float = 1.0) -> int: ...
    def reset_kv_cache(self) -> None: ...
    def logits_slice(self) -> Any: ...

class WgpuQwenDecoder:
    def __init__(
        self,
        embed_tokens: Any,
        layers_data: List[List[Any]],
        final_norm_w: Any,
        lm_head_w: Any,
        lm_head_s: Any,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        group_size: int = 64,
        rms_norm_eps: float = 1e-6,
        max_seq_len: int = 2048,
        rope_theta: float = 1000000.0,
        layers_per_pass: Optional[int] = None,
    ) -> None: ...
    def step(self, token_id: int, offset: int) -> None: ...
    def sample(self, temperature: float = 0.0, top_p: float = 1.0) -> int: ...
    def sample_and_step(self, token_id: int, offset: int, temperature: float = 0.0, top_p: float = 1.0) -> int: ...
    def reset_kv_cache(self) -> None: ...
    def logits_slice(self) -> Any: ...
    @property
    def is_device_lost(self) -> bool: ...
    def stats(self) -> Dict[str, Any]: ...

# Core engine FFI
def execute(payload: str, inputs: Sequence[Any]) -> List[Any]: ...
def execute_from_dict(dict: Dict[str, Any], inputs: Sequence[Any]) -> List[Any]: ...
def prepare_graph(dict: Dict[str, Any]) -> int: ...
def execute_prepared(handle: int, inputs: Sequence[Any]) -> List[Any]: ...
def release_graph(handle: int) -> None: ...
def signature(payload: str) -> str: ...
def supported_targets() -> List[str]: ...
def active_engine() -> str: ...
def rayon_threads() -> int: ...
def dropout_forward(input: Any, p: float, training: bool) -> Any: ...
def memory_pool_stats() -> Dict[str, Any]: ...
def clear_memory_pool() -> None: ...

# GPU FFI
def gpu_info() -> Dict[str, Any]: ...
def gpu_backend() -> str: ...
def gpu_available() -> bool: ...
def wgpu_clear_weight_cache() -> None: ...
def wgpu_clear_buffer_pool() -> None: ...

# Debug FFI
def cpu_features_report() -> Dict[str, Any]: ...
def data_ptr(capsule: Any) -> int: ...
def capsule_dump(capsule: Any) -> Dict[str, Any]: ...

# Autograd FFI
def autograd_enable() -> None: ...
def autograd_disable() -> None: ...
def autograd_is_enabled() -> bool: ...
def autograd_backward(loss: Any) -> None: ...
def autograd_reset() -> None: ...
def autograd_tape_len() -> int: ...
def backward_native(op: str, grad_out: Any, inputs: Sequence[Any], kwargs_json: str) -> List[Any]: ...
def backward_single(op: str, grad_out: Any, inputs: Sequence[Any], kwargs_json: str) -> List[Any]: ...
def backward_batch(
    targets: Sequence[str],
    all_inputs: Sequence[Sequence[Any]],
    all_kwargs: Sequence[str],
    output_ids: Sequence[int],
    input_ids_all: Sequence[Sequence[int]],
    saved_shapes_all: Sequence[Sequence[Sequence[int]]],
    upstream_capsule: Any,
    initial_output_id: int,
) -> List[Tuple[int, Any]]: ...

# Cache FFI
def cache_get(signature: str) -> Optional[str]: ...
def cache_contains(signature: str) -> bool: ...
def cache_put(signature: str, payload: str) -> None: ...
def cache_stats() -> Tuple[int, int, int]: ...
def cache_evictions() -> int: ...
def cache_clear() -> None: ...

# Quantization FFI
def w8a32_linear(x: Any, w: Any, scales: Any, bias: Optional[Any] = None) -> Any: ...
def w4a32_linear(x: Any, w_packed: Any, scales: Any, bias: Optional[Any] = None) -> Any: ...
def w4a32_grouped_linear(x: Any, w_packed: Any, scales: Any, bias: Optional[Any] = None, group_size: int = 64) -> Any: ...
def w4a32_grouped_linear_v2(x: Any, w_blocked: Any, bias: Optional[Any] = None, group_size: int = 64) -> Any: ...
def fused_swiglu_mlp_w8a32(
    x: Any,
    gate_w: Any,
    gate_s: Any,
    gate_b: Optional[Any],
    up_w: Any,
    up_s: Any,
    up_b: Optional[Any],
    down_w: Any,
    down_s: Any,
    down_b: Optional[Any],
) -> Any: ...
def fused_swiglu_mlp_w4a32(
    x: Any,
    gate_w: Any,
    gate_s: Any,
    gate_b: Optional[Any],
    up_w: Any,
    up_s: Any,
    up_b: Optional[Any],
    down_w: Any,
    down_s: Any,
    down_b: Optional[Any],
    group_size: int = 64,
) -> Any: ...
def fused_swiglu_mlp_batched_w4a32(
    x: Any,
    gate_w: Any,
    gate_s: Any,
    gate_b: Optional[Any],
    up_w: Any,
    up_s: Any,
    up_b: Optional[Any],
    down_w: Any,
    down_s: Any,
    down_b: Optional[Any],
    group_size: int = 64,
) -> Any: ...
def fused_attention_step_w8a32(
    x: Any,
    qkv_w: Any,
    qkv_s: Any,
    qkv_b: Optional[Any],
    o_w: Any,
    o_s: Any,
    o_b: Optional[Any],
    k_cache: Any,
    v_cache: Any,
    cos: Any,
    sin: Any,
    offset: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Any: ...
def fused_attention_step_w4a32(
    x: Any,
    qkv_w: Any,
    qkv_s: Any,
    qkv_b: Optional[Any],
    o_w: Any,
    o_s: Any,
    o_b: Optional[Any],
    k_cache: Any,
    v_cache: Any,
    cos: Any,
    sin: Any,
    offset: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group_size: int = 64,
) -> Any: ...
def fused_transformer_layer_step_w4a32(
    x: Any,
    input_norm_w: Any,
    qkv_w: Any,
    qkv_s: Any,
    qkv_b: Optional[Any],
    o_w: Any,
    o_s: Any,
    o_b: Optional[Any],
    post_norm_w: Any,
    gate_w: Any,
    gate_s: Any,
    gate_b: Optional[Any],
    up_w: Any,
    up_s: Any,
    up_b: Optional[Any],
    down_w: Any,
    down_s: Any,
    down_b: Optional[Any],
    k_cache: Any,
    v_cache: Any,
    cos: Any,
    sin: Any,
    offset: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group_size: int = 64,
    eps: float = 1e-6,
) -> None: ...
def quantize_linear_int8(w: Any) -> Tuple[Any, Any]: ...
def quantize_linear_int4(w: Any) -> Tuple[Any, Any]: ...
def wgpu_w4a32_grouped_linear(x: Any, w_packed: Any, scales: Any, bias: Optional[Any] = None, group_size: int = 64) -> Any: ...

# GGUF FFI
def gguf_info(path: str) -> Dict[str, Any]: ...
def gguf_tensors(path: str) -> List[Dict[str, Any]]: ...
def gguf_metadata(path: str) -> Dict[str, Any]: ...

# Profiler FFI
def profiler_enable() -> None: ...
def profiler_disable() -> None: ...
def profiler_reset() -> None: ...
def profiler_report() -> List[Tuple[str, int, int, int, int]]: ...
def profiler_print() -> None: ...
