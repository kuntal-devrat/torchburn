"""FX graph -> structured payload parser (REQ-001 / REQ-002).

Walks a ``torch.fx.GraphModule`` and produces a JSON-serializable plan where
every node is classified as:

* ``placeholder`` / ``get_attr`` — graph inputs and model state,
* ``supported``   — routable to the Rust engine (native kernels),
* ``unsupported`` — must run in native PyTorch eager contexts,
* ``output``      — the graph's return references.

The parser never raises on unknown operators: they become ``unsupported``
nodes that the interpreter executes eagerly (REQ-002 safety loop).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import torch

from . import _torchburn as _native

# torch function targets -> canonical Rust op names
_FUNCTION_TO_OP: dict[str, str] = {
    # Phase 1: elementwise
    "torch.add": "add",
    "torch.sub": "sub",
    "torch.mul": "mul",
    "torch.div": "div",
    "torch.relu": "relu",
    "torch.nn.functional.relu": "relu",
    "operator.add": "add",
    "operator.sub": "sub",
    "operator.mul": "mul",
    "operator.div": "div",
    "_operator.add": "add",
    "_operator.sub": "sub",
    "_operator.mul": "mul",
    "_operator.iadd": "add",
    "_operator.isub": "sub",
    "_operator.imul": "mul",
    "_operator.itruediv": "div",
    "_operator.div": "div",
    # Phase 2: math/comparison
    "torch.eq": "eq",
    "torch.ne": "ne",
    "torch.lt": "lt",
    "torch.le": "le",
    "torch.gt": "gt",
    "torch.ge": "ge",
    "torch.abs": "abs",
    "torch.neg": "neg",
    "torch.sign": "sign",
    "torch.sqrt": "sqrt",
    "torch.rsqrt": "rsqrt",
    "torch.exp": "exp",
    "torch.log": "log",
    "torch.reciprocal": "reciprocal",
    "torch.ceil": "ceil",
    "torch.floor": "floor",
    "torch.clamp": "clamp",
    "torch.clamp_min": "clamp_min",
    "torch.clamp_max": "clamp_max",
    "torch.clip": "clamp",
    "torch.pow": "pow",
    "torch.sin": "sin",
    "torch.cos": "cos",
    "torch.round": "round",
    # Phase 2: activations
    "torch.sigmoid": "sigmoid",
    "torch.tanh": "tanh",
    "torch.nn.functional.gelu": "gelu",
    "torch.nn.functional.silu": "silu",
    "torch.nn.functional.leaky_relu": "leaky_relu",
    "torch.nn.functional.elu": "elu",
    "torch.nn.functional.selu": "selu",
    "torch.nn.functional.softplus": "softplus",
    "torch.nn.functional.hardswish": "hardswish",
    "torch.nn.functional.mish": "mish",
    "torch.softmax": "softmax",
    "torch.nn.functional.softmax": "softmax",
    "torch.log_softmax": "log_softmax",
    "torch.nn.functional.log_softmax": "log_softmax",
    # Phase 2: reductions
    "torch.sum": "sum",
    "sum": "sum",
    "torch.mean": "mean",
    "torch.max": "max_reduce",
    "torch.min": "min_reduce",
    "torch.argmax": "argmax",
    "torch.argmin": "argmin",
    "torch.std": "std",
    "torch.var": "var",
    "torch.cumsum": "cumsum",
    "torch.prod": "prod",
    "torch.norm": "norm",
    # Phase 2: linalg
    "torch.matmul": "matmul",
    "torch.mm": "matmul",
    "torch.bmm": "bmm",
    "torch.linear": "linear",
    "torch.nn.functional.linear": "linear",
    "torch.nn.functional.embedding": "embedding",
    "torch._C._nn.linear": "linear",
    "torch._C._nn.gelu": "gelu",
    "torch.dot": "dot",
    # Phase 2: norm
    "torch.nn.functional.layer_norm": "layer_norm",
    "torch.nn.functional.batch_norm": "batch_norm",
    "torch.nn.functional.group_norm": "group_norm",
    "torch.nn.functional.rms_norm": "rms_norm",
    # Phase 2: shape ops
    "torch.cat": "cat",
    "torch.stack": "stack",
    "torch.reshape": "reshape",
    "torch.flatten": "flatten",
    "torch.nn.functional.flatten": "flatten",
    "torch.permute": "permute",
    "torch.expand": "expand",
    "torch.index_select": "index_select",
    "torch.gather": "gather",
    "torch.where": "where",
    "torch.masked_fill": "masked_fill",
    "torch.flip": "flip",
    "torch.narrow": "narrow",
    "torch.unbind": "unbind",
    # Phase 3: convolution
    "torch.conv1d": "conv1d",
    "torch.conv2d": "conv2d",
    "torch.conv_transpose1d": "conv_transpose1d",
    "torch.conv_transpose2d": "conv_transpose2d",
    "torch.nn.functional.conv1d": "conv1d",
    "torch.nn.functional.conv2d": "conv2d",
    "torch.nn.functional.conv_transpose1d": "conv_transpose1d",
    "torch.nn.functional.conv_transpose2d": "conv_transpose2d",
    # Phase 3: pooling
    "torch.nn.functional.max_pool1d": "max_pool1d",
    "torch.nn.functional.max_pool2d": "max_pool2d",
    "torch._C._nn.max_pool2d": "max_pool2d",
    "torch.nn.functional.avg_pool1d": "avg_pool1d",
    "torch.nn.functional.avg_pool2d": "avg_pool2d",
    "torch._C._nn.avg_pool2d": "avg_pool2d",
    "torch.nn.functional.adaptive_avg_pool2d": "adaptive_avg_pool2d",
    "torch._C._nn.adaptive_avg_pool2d": "adaptive_avg_pool2d",
    "torch.nn.functional.adaptive_max_pool2d": "adaptive_max_pool2d",
    # Phase 3: upsampling
    "torch.nn.functional.interpolate": "interpolate",
    # operator overloading
    "operator.eq": "eq",
    "operator.ne": "ne",
    "operator.lt": "lt",
    "operator.le": "le",
    "operator.gt": "gt",
    "operator.ge": "ge",
    "_operator.matmul": "matmul",
    "_operator.pow": "pow",
    "_operator.neg": "neg",
    "operator.neg": "neg",
    # Phase 2: logical ops + dtype cast
    "torch.logical_and": "logical_and",
    "torch.logical_or": "logical_or",
    "torch.logical_not": "logical_not",
    # Phase 7: extended ops
    "torch.scatter": "scatter",
    "torch.scatter_add": "scatter_add",
    "torch.topk": "topk",
    "torch.sort": "sort",
    "torch.argsort": "argsort",
    "torch.repeat_interleave": "repeat_interleave",
    "torch.Tensor.repeat_interleave": "repeat_interleave",
    "torch.Tensor.repeat": "repeat",
    "torch.prelu": "prelu",
    "torch.nn.functional.prelu": "prelu",
    "torch.nonzero": "nonzero",
    "torch.Tensor.nonzero": "nonzero",
    "torch.einsum": "einsum",
    # Phase 10: missing operator targets
    "_operator.truediv": "div",
    "operator.truediv": "div",
    # bare targets emitted by torch.compile
    "transpose": "transpose",
    "repeat": "repeat",
    "scatter": "scatter",
    "scatter_add": "scatter_add",
    "squeeze": "squeeze",
    "unsqueeze": "unsqueeze",
    "sort": "sort",

    "chunk": "chunk",

    "unbind": "unbind",
    "topk": "topk",
    # Tensor creation functions (call_function targets)
    "torch.zeros": "zeros",
    "torch.arange": "arange",
    "torch.functional.einsum": "einsum",
    # v0.2 extra 50 — high-coverage math/shape
    "torch.atan": "atan",
    "torch.asin": "asin",
    "torch.acos": "acos",
    "torch.sinh": "sinh",
    "torch.cosh": "cosh",
    "torch.asinh": "asinh",
    "torch.acosh": "acosh",
    "torch.atanh": "atanh",
    "torch.erf": "erf",
    "torch.erfc": "erfc",
    "torch.expm1": "expm1",
    "torch.log1p": "log1p",
    "torch.log2": "log2",
    "torch.log10": "log10",
    "torch.trunc": "trunc",
    "torch.frac": "frac",
    "torch.square": "square",
    "torch.exp2": "exp2",
    "torch.atan2": "atan2",
    "torch.hypot": "hypot",
    "torch.fmod": "fmod",
    "torch.remainder": "remainder",
    "torch.copysign": "copysign",
    "torch.lerp": "lerp",
    "torch.bitwise_and": "bitwise_and",
    "torch.bitwise_or": "bitwise_or",
    "torch.bitwise_xor": "bitwise_xor",
    "torch.bitwise_not": "bitwise_not",
    "torch.isfinite": "isfinite",
    "torch.isinf": "isinf",
    "torch.isnan": "isnan",
    "torch.all": "all",
    "torch.any": "any",
    "torch.amax": "amax",
    "torch.amin": "amin",
    "torch.count_nonzero": "count_nonzero",
    "torch.nansum": "nansum",
    "torch.nanmean": "nanmean",
    "torch.tile": "tile",
    "torch.roll": "roll",
    "torch.nn.functional.pixel_shuffle": "pixel_shuffle",
    "torch.nn.functional.instance_norm": "instance_norm",
    "torch.nn.functional.cross_entropy": "cross_entropy",
    "torch.nn.functional.huber_loss": "huber_loss",
    "torch.nn.functional.hardtanh": "hardtanh",
    "torch.nn.functional.hardsigmoid": "hardsigmoid",
    "torch.nn.functional.glu": "glu",
    "torch.bucketize": "bucketize",
    "torch.histc": "histc",
    "torch.nn.functional.lerp": "lerp",
    # Batch 2 operations
    "torch.nn.functional.embedding_bag": "embedding_bag",
    "torch.embedding_bag": "embedding_bag",
    "torch.nn.functional.unfold": "unfold",
    "torch.nn.functional.fold": "fold",
    "torch.nn.functional.grid_sample": "grid_sample",
    "torch.nn.functional.affine_grid": "affine_grid",
    "torch.nn.functional.pixel_unshuffle": "pixel_unshuffle",
    "torch.nn.functional.channel_shuffle": "channel_shuffle",
    "torch.channel_shuffle": "channel_shuffle",
    "torch.cummax": "cummax",
    "torch.cummin": "cummin",
    "torch.logcumsumexp": "logcumsumexp",
    "torch.scatter_reduce": "scatter_reduce",
    "torch.index_put": "index_put",
    "torch.index_add": "index_add",
    "torch.masked_scatter": "masked_scatter",
    "torch.take": "take",
    "torch.put": "put",
    "torch.masked_select": "masked_select",
    "torch.index_fill": "index_fill",
    "torch.bincount": "bincount",
    "torch.unique": "unique",
    "torch.kthvalue": "kthvalue",
    "torch.median": "median",
    "torch.quantile": "quantile",
    "torch.histogram": "histogram",
    "torch.searchsorted": "searchsorted",
    "torch.meshgrid": "meshgrid",
    "torch.cdist": "cdist",
    "torch.nn.functional.pdist": "pdist",
    "torch.pdist": "pdist",
    "torch.renorm": "renorm",
    "torch.bernoulli": "bernoulli",
    "torch.multinomial": "multinomial",
    "torch.logspace": "logspace",
    "torch.eye": "eye",
    "torch.diag": "diag",
    "torch.diagonal": "diagonal",
    "torch.trace": "trace",
    "torch.linalg.matrix_exp": "matrix_exp",
    "torch.matrix_exp": "matrix_exp",
    "torch.linalg.slogdet": "slogdet",
    "torch.slogdet": "slogdet",
    "torch.linalg.det": "det",
    "torch.det": "det",
    "torch.linalg.lstsq": "lstsq",
    "torch.lstsq": "lstsq",
    "torch.linalg.pinv": "pinverse",
    "torch.pinverse": "pinverse",
    "torch.normal": "normal",
    "torch.triu": "triu",
    "torch.tril": "tril",
    "torch.hann_window": "hann_window",
    "torch.bartlett_window": "bartlett_window",
    "torch.blackman_window": "blackman_window",
    "torch.stft": "stft",
    # Batch 3 operations
    "torch.nextafter": "nextafter",
    "torch.heaviside": "heaviside",
    "torch.nan_to_num": "nan_to_num",
    "torch.logaddexp": "logaddexp",
    "torch.logaddexp2": "logaddexp2",
    "torch.sinc": "sinc",
    "torch.special.sinc": "sinc",
    "torch.i0": "i0",
    "torch.special.i0": "i0",
    "torch.special.i0e": "i0e",
    "torch.special.i1": "i1",
    "torch.special.i1e": "i1e",
    "torch.special.bessel_j0": "bessel_j0",
    "torch.special.bessel_j1": "bessel_j1",
    "torch.special.bessel_y0": "bessel_y0",
    "torch.special.bessel_y1": "bessel_y1",
    "torch.digamma": "digamma",
    "torch.special.digamma": "digamma",
    "torch.lgamma": "lgamma",
    "torch.special.gammaln": "lgamma",
    "torch.special.polygamma": "polygamma",
    "torch.special.multigammaln": "mvlgamma",
    "torch.mvlgamma": "mvlgamma",
    "torch.erfinv": "erfinv",
    "torch.special.erfinv": "erfinv",
    "torch.special.erfcinv": "erfcinv",
    "torch.special.ndtri": "ndtri",
    "torch.special.ndtr": "ndtr",
    "torch.special.log_ndtr": "log_ndtr",
    "torch.logit": "logit",
    "torch.special.logit": "logit",
    "torch.special.expit": "expit",
    "torch.rad2deg": "rad2deg",
    "torch.deg2rad": "deg2rad",
    "torch.gcd": "gcd",
    "torch.lcm": "lcm",
    "torch.fmax": "fmax",
    "torch.fmin": "fmin",
    "torch.maximum": "maximum",
    "torch.minimum": "minimum",
    "torch.signbit": "signbit",
    "torch.addcdiv": "addcdiv",
    "torch.addcmul": "addcmul",
    "torch.addr": "addr",
    "torch.outer": "outer",
    "torch.ger": "ger",
    "torch.mv": "mv",
    "torch.vdot": "vdot",
    "torch.baddbmm": "baddbmm",
    "torch.addbmm": "addbmm",
    "torch.addmv": "addmv",
    "torch.kron": "kron",
    "torch.inner": "inner",
    "torch.trapz": "trapz",
    "torch.trapezoid": "trapezoid",
    "torch.cumulative_trapezoid": "cumulative_trapezoid",
    "torch.nn.functional.celu": "celu",
    "torch.nn.functional.hardshrink": "hardshrink",
    "torch.nn.functional.softshrink": "softshrink",
    "torch.nn.functional.tanhshrink": "tanhshrink",
    "torch.nn.functional.threshold": "threshold",
    "torch.nn.functional.logsigmoid": "logsigmoid",
    "torch.nn.functional.rrelu": "rrelu",
    "torch.nn.functional.kl_div": "kl_div",
    "torch.nn.functional.poisson_nll_loss": "poisson_nll_loss",
    "torch.nn.functional.margin_ranking_loss": "margin_ranking_loss",
    "torch.nn.functional.hinge_embedding_loss": "hinge_embedding_loss",
    "torch.nn.functional.multilabel_margin_loss": "multilabel_margin_loss",
    "torch.nn.functional.soft_margin_loss": "soft_margin_loss",
    "torch.nn.functional.multilabel_soft_margin_loss": "multilabel_soft_margin_loss",
    "torch.nn.functional.cosine_embedding_loss": "cosine_embedding_loss",
    "torch.nn.functional.triplet_margin_loss": "triplet_margin_loss",
    "torch.nn.functional.ctc_loss": "ctc_loss",
    "torch.hamming_window": "hamming_window",
    "torch.kaiser_window": "kaiser_window",
    "torch.gaussian_window": "gaussian_window",
    "torch.exponential_window": "exponential_window",
    "torch.triangular_window": "triangular_window",
    "torch.cross": "cross",
    "torch.linalg.norm": "linalg_norm",
    "torch.linalg.matrix_rank": "matrix_rank",
    "torch.linalg.matrix_power": "matrix_power",
    "torch.linalg.cholesky": "cholesky",
    "torch.linalg.cholesky_inverse": "cholesky_inverse",
    "torch.linalg.cholesky_solve": "cholesky_solve",
    "torch.linalg.qr": "qr",
    "torch.linalg.svd": "svd",
    "torch.linalg.svdvals": "svdvals",
    "torch.linalg.eig": "eig",
    "torch.linalg.eigh": "eigh",
    "torch.linalg.eigvals": "eigvals",
    "torch.linalg.eigvalsh": "eigvalsh",
    "torch.linalg.lu": "lu",
    "torch.linalg.triangular_solve": "triangular_solve",
    "torch.select_scatter": "select_scatter",
    "torch.slice_scatter": "slice_scatter",
    "torch.diagonal_scatter": "diagonal_scatter",
    "torch.index_copy": "index_copy",
    "torch.narrow_copy": "narrow_copy",
    "torch.movedim": "movedim",
    "torch.moveaxis": "moveaxis",
    "torch.swapdims": "swapdims",
    "torch.swapaxes": "swapaxes",
    "torch.column_stack": "column_stack",
    "torch.row_stack": "row_stack",
    "torch.dstack": "dstack",
    "torch.hstack": "hstack",
    "torch.vstack": "vstack",
    "torch.atleast_1d": "atleast_1d",
    "torch.atleast_2d": "atleast_2d",
    "torch.atleast_3d": "atleast_3d",
    "torch.block_diag": "block_diag",
    "torch.cartesian_prod": "cartesian_prod",
    "torch.combinations": "combinations",
    "torch.nn.functional.pad": "pad",
    "torch.nn.functional.conv3d": "conv3d",
    "torch.nn.functional.conv_transpose3d": "conv_transpose3d",
    "torch.nn.functional.max_pool3d": "max_pool3d",
    "torch.nn.functional.avg_pool3d": "avg_pool3d",
    "torch.nn.functional.adaptive_max_pool3d": "adaptive_max_pool3d",
    "torch.nn.functional.adaptive_avg_pool3d": "adaptive_avg_pool3d",
    "torch.nn.functional.fractional_max_pool2d": "fractional_max_pool2d",
    "torch.nn.functional.fractional_max_pool3d": "fractional_max_pool3d",
    "torch.nn.functional.lp_pool1d": "lp_pool1d",
    "torch.nn.functional.lp_pool2d": "lp_pool2d",
    "torch.nn.functional.max_unpool1d": "max_unpool1d",
    "torch.nn.functional.max_unpool2d": "max_unpool2d",
    "torch.nn.functional.max_unpool3d": "max_unpool3d",
    "torch.rand": "rand",
    "torch.randn": "randn",
    "torch.randint": "randint",
    "torch.randperm": "randperm",
    "torch.empty": "empty",
    "torch.zeros_like": "zeros_like",
    "torch.ones_like": "ones_like",
    "torch.full_like": "full_like",
    "torch.nn.functional.rnn_tanh_cell": "rnn_tanh_cell",
    "torch.nn.functional.rnn_relu_cell": "rnn_relu_cell",
    "torch.nn.functional.gru_cell": "gru_cell",
    "torch.nn.functional.lstm_cell": "lstm_cell",
    "torch.nn.functional.multi_head_attention_forward": "multi_head_attention_forward",
    "torch.linalg.lu_solve": "lu_solve",
    "torch.lu_unpack": "lu_unpack",
    "torch.linalg.solve": "linalg_solve",
    "torch.linalg.inv": "linalg_inv",
    "torch.linalg.cond": "linalg_cond",
    # C-extension aliases emitted by PyTorch FX tracer
    "torch._C._special.special_logit": "logit",
    "torch._C._linalg.linalg_det": "det",
    "torch._C._linalg.linalg_pinv": "pinverse",
    "torch._C._nn.softshrink": "softshrink",
    "torch._C._nn.celu": "celu",
    "torch._C._nn.hardshrink": "hardshrink",
    "torch._C._nn.tanhshrink": "tanhshrink",
    # Advanced LLM & FlashAttention
    "torch.nn.functional.scaled_dot_product_attention": "scaled_dot_product_attention",
    # Universal Low-Bit Quantization
    "torch.quantize_per_tensor": "quantize_per_tensor",
    "torch.dequantize": "dequantize_per_tensor",
    "torch.ao.quantization.quantize_per_tensor": "quantize_per_tensor",
    "torch.ao.quantization.dequantize_per_tensor": "dequantize_per_tensor",
    # Universal FFT & Complex Suite
    "torch.fft.fft": "fft",
    "torch.fft.ifft": "ifft",
    "torch.fft.rfft": "rfft",
    "torch.fft.irfft": "irfft",
    "torch.fft.fft2": "fft2",
    "torch.fft.ifft2": "ifft2",
    "torch.fft.fftn": "fftn",
    "torch.fft.ifftn": "ifftn",
    "torch.fft.fftshift": "fftshift",
    "torch.fft.ifftshift": "ifftshift",
    "torch.complex": "complex",
    "torch.real": "real",
    "torch.imag": "imag",
    "torch.angle": "angle",
    "torch.polar": "polar",
    "torch.conj": "conj",
    "torch.conj_physical": "conj",
    "torch._C._fft.fft_fft": "fft",
    "torch._C._fft.fft_ifft": "ifft",
    "torch._C._fft.fft_rfft": "rfft",
    "torch._C._fft.fft_irfft": "irfft",
    # Batch 4 — 48 ops for 450 total
    "torch.isclose": "isclose",
    "torch.allclose": "allclose",
    "torch.equal": "equal",
    "torch.isreal": "isreal",
    "torch.is_complex": "is_complex",
    "torch.is_nonzero": "is_nonzero",
    "torch.nanprod": "nanprod",
    "torch.nanmin": "nanmin",
    "torch.nanmax": "nanmax",
    "torch.var_mean": "var_mean",
    "torch.std_mean": "std_mean",
    "torch.nanmedian": "nanmedian",
    "torch.cov": "cov",
    "torch.corrcoef": "corrcoef",
    "torch.as_strided": "as_strided",
    "torch.broadcast_to": "broadcast_to",
    "torch.broadcast_tensors": "broadcast_tensors",
    "torch.split": "split",
    "torch.vsplit": "vsplit",
    "torch.hsplit": "hsplit",
    "torch.dsplit": "dsplit",
    "torch.tensor_split": "tensor_split",
    "torch.take_along_dim": "take_along_dim",
    "torch.index_reduce": "index_reduce",
    "torch.scatter": "scatter_max",
    "torch.linalg.multi_dot": "linalg_multi_dot",
    "torch.linalg.vander": "linalg_vander",
    "torch.linalg.vecdot": "linalg_vecdot",
    "torch.linalg.cross": "linalg_cross",
    "torch.linalg.tensordot": "linalg_tensordot",
    "torch.linalg.cholesky_ex": "linalg_cholesky_ex",
    "torch.linalg.inv_ex": "linalg_inv_ex",
    "torch.linalg.solve_ex": "linalg_solve_ex",
    "torch.linalg.lu_factor": "linalg_lu_factor",
    "torch.nn.functional.local_response_norm": "local_response_norm",
    "torch.nn.functional.adaptive_avg_pool1d": "adaptive_avg_pool1d",
    "torch.nn.functional.adaptive_max_pool1d": "adaptive_max_pool1d",
    "torch.nn.functional.lp_pool3d": "lp_pool3d",
    "torch.logsumexp": "logsumexp",
    "torch.randn_like": "randn_like",
    "torch.rand_like": "rand_like",
    "torch.randint_like": "randint_like",
    "torch.empty_strided": "empty_strided",
    "torch.Tensor.view_as": "view_as",
    "torch.Tensor.expand_as": "expand_as",
    "torch.masked_select": "masked_select_extra",
    "torch.Tensor.masked_select": "masked_select_extra",
    "torch.istft": "istft",
    # In-place method targets (call_method)
    "add_": "add",
}

# aten OpOverload targets (as emitted by torch._dynamo) -> canonical op names
_ATEN_TO_OP: dict[str, str] = {
    "aten.add.Tensor": "add",
    "aten.sub.Tensor": "sub",
    "aten.mul.Tensor": "mul",
    "aten.div.Tensor": "div",
    "aten.relu": "relu",
    "aten.relu.default": "relu",
    "aten.detach.default": "contiguous",
    "aten.clone.default": "contiguous",
    "aten.threshold_backward.default": "threshold_backward",
    # Phase 2: math/comparison
    "aten.eq.Tensor": "eq",
    "aten.ne.Tensor": "ne",
    "aten.lt.Tensor": "lt",
    "aten.le.Tensor": "le",
    "aten.gt.Tensor": "gt",
    "aten.ge.Tensor": "ge",
    "aten.abs.default": "abs",
    "aten.neg.default": "neg",
    "aten.sign.default": "sign",
    "aten.sqrt.default": "sqrt",
    "aten.rsqrt.default": "rsqrt",
    "aten.exp.default": "exp",
    "aten.log.default": "log",
    "aten.reciprocal.default": "reciprocal",
    "aten.ceil.default": "ceil",
    "aten.floor.default": "floor",
    "aten.clamp.default": "clamp",
    "aten.pow.Tensor_Scalar": "pow",
    # Phase 2: logical ops + dtype cast
    "aten.logical_and.default": "logical_and",
    "aten.logical_or.default": "logical_or",
    "aten.logical_not.default": "logical_not",
    "aten._to_copy.default": "to_dtype",
    "aten.to.dtype": "to_dtype",
    "aten.to.dtype_layout": "to_dtype",
    # Phase 2: activations
    "aten.sigmoid.default": "sigmoid",
    "aten.tanh.default": "tanh",
    "aten.gelu.default": "gelu",
    # Phase 7: extended ops
    "aten.scatter.src": "scatter",
    "aten.scatter_add.src": "scatter_add",
    "aten.topk.default": "topk",
    "aten.sort.default": "sort",
    "aten.argsort.default": "argsort",
    "aten.repeat_interleave.Tensor": "repeat_interleave",
    "aten.repeat_interleave.self_Tensor": "repeat_interleave",
    "aten.prelu.default": "prelu",
    "aten.prelu.ndarray": "prelu",
    "aten.nonzero.default": "nonzero",
    "aten.nonzero_numpy.default": "nonzero",
    "aten.einsum.default": "einsum",
    "aten.silu.default": "silu",
    "aten.silu_": "silu",
    # Phase 10: in-place op aliases (aten.X_.default variants)
    "aten.add_.Tensor": "add",
    "aten.mul_.Tensor": "mul",
    "aten.sub_.Tensor": "sub",
    "aten.div_.Tensor": "div",
    "aten.relu_.default": "relu",
    "aten.abs_.default": "abs",
    "aten.neg_.default": "neg",
    "aten.clamp_.default": "clamp",
    "aten.clamp_min_.default": "clamp_min",
    "aten.clamp_max_.default": "clamp_max",
    "aten.sqrt_.default": "sqrt",
    "aten.exp_.default": "exp",
    "aten.log_.default": "log",
    "aten.sin_.default": "sin",
    "aten.cos_.default": "cos",
    "aten.ceil_.default": "ceil",
    "aten.floor_.default": "floor",
    "aten.round_.default": "round",
    "aten.reciprocal_.default": "reciprocal",
    # Phase 10: new aten targets for trig ops
    "aten.sin.default": "sin",
    "aten.cos.default": "cos",
    "aten.round.default": "round",
    "aten.clamp_min.default": "clamp_min",
    "aten.clamp_max.default": "clamp_max",
    "aten.clip.default": "clamp",
    "aten.full.default": "full",
    "aten.zeros.default": "zeros",
    "aten.ones.default": "ones",
    "aten.arange.default": "arange",
    "aten.linspace.default": "linspace",
    "aten.leaky_relu.default": "leaky_relu",
    "aten.elu.default": "elu",
    "aten.selu.default": "selu",
    "aten.softplus.default": "softplus",
    "aten.hardswish_.default": "hardswish",
    "aten.hardswish.default": "hardswish",
    "aten.mish_.default": "mish",
    "aten.mish.default": "mish",
    "aten.softmax.int": "softmax",
    "aten._softmax.default": "softmax",
    "aten.log_softmax.int": "log_softmax",
    # Phase 2: reductions
    "aten.sum.default": "sum",
    "aten.sum.dim_IntList": "sum",
    "aten.mean.default": "mean",
    "aten.mean.dim": "mean",
    "aten.max.default": "max_reduce",
    "aten.max.dim": "max_reduce",
    "aten.min.default": "min_reduce",
    "aten.min.dim": "min_reduce",
    "aten.argmax.default": "argmax",
    "aten.argmax.dim": "argmax",
    "aten.argmin.default": "argmin",
    "aten.argmin.dim": "argmin",
    "aten.std.default": "std",
    "aten.std.correction": "std",
    "aten.var.default": "var",
    "aten.var.correction": "var",
    "aten.cumsum.default": "cumsum",
    "aten.cumsum.dim": "cumsum",
    "aten.prod.default": "prod",
    "aten.prod.dim_int": "prod",
    "aten.linalg_vector_norm.default": "linalg_vector_norm",
    # Phase 2: linalg
    "aten.mm": "matmul",
    "aten.mm.default": "matmul",
    "aten.addmm": "addmm",
    "aten.addmm.default": "addmm",
    "aten.t": "t",
    "aten.t.default": "t",
    "aten.bmm": "bmm",
    "aten.bmm.default": "bmm",
    "aten.dot.default": "dot",
    # Phase 2: norm
    "aten.layer_norm.default": "layer_norm",
    "aten.native_layer_norm.default": "layer_norm",
    "aten.native_batch_norm.default": "batch_norm",
    "aten.batch_norm.default": "batch_norm",
    "aten.native_group_norm.default": "group_norm",
    "aten.native_rms_norm.default": "rms_norm",
    # Phase 2: shape ops
    "aten.cat.default": "cat",
    "aten.stack.default": "stack",
    "aten.reshape.default": "reshape",
    "aten.view.default": "reshape",
    "aten.permute_copy.default": "permute",
    "aten.permute.default": "permute",
    "aten.transpose.int": "transpose",
    "aten.transpose.int.int": "transpose",
    "aten.index_select": "index_select",
    "aten.index_select.default": "index_select",
    "aten.gather": "gather",
    "aten.gather.default": "gather",
    "aten.expand.default": "expand",
    "aten.index_select.default": "index_select",
    "aten.gather.default": "gather",
    "aten.where.self": "where",
    "aten.where.default": "where",
    "aten.masked_fill_.default": "masked_fill",
    "aten.masked_fill.default": "masked_fill",
    "aten.flip.default": "flip",
    "aten.narrow.default": "narrow",
    "aten.slice_copy.Tensor": "narrow",
    # Phase 3: convolution (export path uses aten.convolution.default)
    "aten.convolution.default": "conv2d",
    "aten.conv1d.default": "conv1d",
    "aten.conv2d.default": "conv2d",
    # Phase 3: pooling
    "aten.max_pool2d_with_indices.default": "max_pool2d",
    "aten.max_pool1d.default": "max_pool1d",
    "aten.avg_pool2d.default": "avg_pool2d",
    "aten.avg_pool1d.default": "avg_pool1d",
    "aten.adaptive_avg_pool2d.default": "adaptive_avg_pool2d",
    "aten.adaptive_max_pool2d.default": "adaptive_max_pool2d",
    # Phase 3: upsampling
    "aten.upsample_nearest2d.default": "upsample_nearest2d",
    "aten.upsample_bilinear2d.default": "upsample_bilinear2d",
    "aten.upsample_nearest2d.vec": "upsample_nearest2d",
    "aten.upsample_bilinear2d.vec": "upsample_bilinear2d",
    # Phase 3: flatten
    "aten.flatten.using_ints": "flatten",
    "aten.flatten.default": "flatten",
    # views emitted by dynamo/AOTAutograd (reshape alias)
    "aten._unsafe_view": "reshape",
    "aten._unsafe_view.default": "reshape",
    "aten.reshape.default": "reshape",
    # contiguous() — layout-only, treated as reshape in the engine
    "aten.contiguous.memory_format": "contiguous",
    # Phase 13: shape ops for model coverage
    "aten.squeeze.dim": "squeeze",
    "aten.squeeze.default": "squeeze",
    "aten.unsqueeze.default": "unsqueeze",
    "aten.unsqueeze.dim": "unsqueeze",
    "aten.unflatten.int": "unflatten",
    "aten.unbind.int": "unbind",
    "aten.unflatten.default": "unflatten",
    "aten.dropout.default": "dropout",
    "aten.dropout_.default": "dropout",
    # Phase 4: transformer stack (embedding, attention, losses)
    "aten.scalar_tensor": "scalar_tensor",
    "aten.scalar_tensor.default": "scalar_tensor",
    "aten.embedding": "embedding",
    "aten.embedding.default": "embedding",
    "torch._C._nn.scaled_dot_product_attention": "scaled_dot_product_attention",
    "aten.scaled_dot_product_attention": "scaled_dot_product_attention",
    "aten.scaled_dot_product_attention.default": "scaled_dot_product_attention",
    "aten._scaled_dot_product_flash_attention_for_cpu": "scaled_dot_product_attention",
    "aten._scaled_dot_product_flash_attention_for_cpu.default": "scaled_dot_product_attention",
    "aten._scaled_dot_product_flash_attention": "scaled_dot_product_attention",
    "aten._scaled_dot_product_flash_attention.default": "scaled_dot_product_attention",
    "aten._scaled_dot_product_efficient_attention": "scaled_dot_product_attention",
    "aten._scaled_dot_product_efficient_attention.default": "scaled_dot_product_attention",
    "aten.rotary_embedding": "rope",
    "aten.rotary_embedding.default": "rope",
    "aten._log_softmax": "log_softmax",
    "aten._log_softmax.default": "log_softmax",
    "torch.nn.functional.mse_loss": "mse_loss",
    "torch.nn.functional.binary_cross_entropy": "binary_cross_entropy",
    "torch.nn.functional.nll_loss": "nll_loss_forward",
    "torch.nn.functional.smooth_l1_loss": "smooth_l1_loss",
    "aten.nll_loss_forward": "nll_loss_forward",
    "aten.nll_loss_forward.default": "nll_loss_forward",
    "aten.mse_loss": "mse_loss",
    "aten.mse_loss.default": "mse_loss",
    "aten.smooth_l1_loss": "smooth_l1_loss",
    "aten.smooth_l1_loss.default": "smooth_l1_loss",
    "aten.binary_cross_entropy": "binary_cross_entropy",
    "aten.binary_cross_entropy.default": "binary_cross_entropy",
    # v0.2 extra 50
    "aten.atan.default": "atan",
    "aten.asin.default": "asin",
    "aten.acos.default": "acos",
    "aten.sinh.default": "sinh",
    "aten.cosh.default": "cosh",
    "aten.asinh.default": "asinh",
    "aten.acosh.default": "acosh",
    "aten.atanh.default": "atanh",
    "aten.erf.default": "erf",
    "aten.erfc.default": "erfc",
    "aten.expm1.default": "expm1",
    "aten.log1p.default": "log1p",
    "aten.log2.default": "log2",
    "aten.log10.default": "log10",
    "aten.trunc.default": "trunc",
    "aten.frac.default": "frac",
    "aten.square.default": "square",
    "aten.exp2.default": "exp2",
    "aten.atan2.default": "atan2",
    "aten.hypot.default": "hypot",
    "aten.fmod.Tensor": "fmod",
    "aten.remainder.Tensor": "remainder",
    "aten.copysign.default": "copysign",
    "aten.copysign.Tensor": "copysign",
    "aten.lerp.Tensor": "lerp",
    "aten.bitwise_and.Tensor": "bitwise_and",
    "aten.bitwise_or.Tensor": "bitwise_or",
    "aten.bitwise_xor.Tensor": "bitwise_xor",
    "aten.bitwise_not.default": "bitwise_not",
    "aten.isfinite.default": "isfinite",
    "aten.isinf.default": "isinf",
    "aten.isnan.default": "isnan",
    "aten.all.default": "all",
    "aten.any.default": "any",
    "aten.amax.default": "amax",
    "aten.amin.default": "amin",
    "aten.count_nonzero.default": "count_nonzero",
    "aten.nansum.default": "nansum",
    "aten.nanmean.default": "nanmean",
    "aten.tile.default": "tile",
    "aten.roll.default": "roll",
    "aten.pixel_shuffle.default": "pixel_shuffle",
    "aten.instance_norm.default": "instance_norm",
    "aten.cross_entropy_loss.default": "cross_entropy",
    "aten.huber_loss.default": "huber_loss",
    "aten.hardtanh.default": "hardtanh",
    "aten.hardsigmoid.default": "hardsigmoid",
    "aten.glu.default": "glu",
    "aten.bucketize.Tensor": "bucketize",
    "aten.histc.default": "histc",
    "aten.ldexp.Tensor": "ldexp",
    # Universal FFT & Complex Suite
    "aten.fft_fft.default": "fft",
    "aten.fft_ifft.default": "ifft",
    "aten.fft_rfft.default": "rfft",
    "aten.fft_irfft.default": "irfft",
    "aten.fft_fft2.default": "fft2",
    "aten.fft_ifft2.default": "ifft2",
    "aten.fft_fftn.default": "fftn",
    "aten.fft_ifftn.default": "ifftn",
    "aten.fft_fftshift.default": "fftshift",
    "aten.fft_ifftshift.default": "ifftshift",
    "aten.complex.default": "complex",
    "aten.real.default": "real",
    "aten.imag.default": "imag",
    "aten.angle.default": "angle",
    "aten.polar.default": "polar",
    "aten.conj.default": "conj",
    "aten.conj_physical.default": "conj",
    # Quantization
    "aten.quantize_per_tensor.default": "quantize_per_tensor",
    "aten.dequantize.self": "dequantize_per_tensor",
    # Batch 4 — 48 ops for 450
    "aten.isclose.default": "isclose",
    "aten.allclose.default": "allclose",
    "aten.equal.default": "equal",
    "aten.isreal.default": "isreal",
    "aten.is_complex.default": "is_complex",
    "aten.is_nonzero.default": "is_nonzero",
    "aten.nanprod.default": "nanprod",
    "aten.nanmin.default": "nanmin",
    "aten.nanmax.default": "nanmax",
    "aten.var_mean.correction": "var_mean",
    "aten.std_mean.correction": "std_mean",
    "aten.nanmedian.default": "nanmedian",
    "aten.cov.default": "cov",
    "aten.corrcoef.default": "corrcoef",
    "aten.as_strided.default": "as_strided",
    "aten.broadcast_to.default": "broadcast_to",
    "aten.broadcast_tensors.default": "broadcast_tensors",
    "aten.split.Tensor": "split",
    "aten.split_with_sizes.default": "split",
    "aten.vsplit.default": "vsplit",
    "aten.hsplit.default": "hsplit",
    "aten.dsplit.default": "dsplit",
    "aten.tensor_split.sections": "tensor_split",
    "aten.tensor_split.indices": "tensor_split",
    "aten.take_along_dim.default": "take_along_dim",
    "aten.index_reduce.default": "index_reduce",
    "aten.scatter_reduce.two": "scatter_max",
    "aten.scatter_max.default": "scatter_max",
    "aten.scatter_min.default": "scatter_min",
    "aten.linalg.multi_dot": "linalg_multi_dot",
    "aten.linalg.vander.default": "linalg_vander",
    "aten.linalg.vecdot.default": "linalg_vecdot",
    "aten.linalg.cross.default": "linalg_cross",
    "aten.linalg.tensordot.default": "linalg_tensordot",
    "aten.linalg.cholesky_ex.default": "linalg_cholesky_ex",
    "aten.linalg.inv_ex.default": "linalg_inv_ex",
    "aten.linalg.solve_ex.default": "linalg_solve_ex",
    "aten.linalg.lu_factor.default": "linalg_lu_factor",
    "aten.linalg.lu_factor_ex.default": "linalg_lu_factor",
    "aten.local_response_norm.default": "local_response_norm",
    "aten.adaptive_avg_pool1d.default": "adaptive_avg_pool1d",
    "aten.adaptive_max_pool1d.default": "adaptive_max_pool1d",
    "aten.adaptive_avg_pool1d.default": "adaptive_avg_pool1d",
    "aten.lp_pool3d.default": "lp_pool3d",
    "aten.logsumexp.default": "logsumexp",
    "aten.randn_like.default": "randn_like",
    "aten.rand_like.default": "rand_like",
    "aten.randint_like.default": "randint_like",
    "aten.empty_strided.default": "empty_strided",
    "aten.view_as.default": "view_as",
    "aten.expand_as.default": "expand_as",
    "aten.masked_select.default": "masked_select_extra",
    "aten.isfinite.default": "isfinite",
    "aten.istft.default": "istft",
    # Attention
    "aten._scaled_dot_product_flash_attention.default": "scaled_dot_product_attention",
    "aten._scaled_dot_product_efficient_attention.default": "scaled_dot_product_attention",
}

# Tensor method names (call_method nodes) -> canonical op names
_METHOD_TO_OP: dict[str, str] = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "abs": "abs",
    "neg": "neg",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "softmax": "softmax",
    "view": "reshape",
    "reshape": "reshape",
    "permute": "permute",
    "expand": "expand",
    "clamp": "clamp",
    "clamp_min": "clamp_min",
    "clamp_max": "clamp_max",
    "sin": "sin",
    "cos": "cos",
    "round": "round",
    "logical_and": "logical_and",
    "logical_or": "logical_or",
    "logical_not": "logical_not",
    "to": "to_dtype",
    "float": "to_dtype",
    "double": "to_dtype",
    # contiguous() is a layout-only op; map it so the engine handles it as a
    # shape-preserving no-op (copies data if non-contiguous, which the engine
    # enforces anyway before dispatch).
    "contiguous": "contiguous",
    # Phase 10: missing method targets
    "transpose": "transpose",
    "repeat": "repeat",
    "sort": "sort",
    "scatter": "scatter",
    "scatter_add": "scatter_add",
    "squeeze": "squeeze",
    "unsqueeze": "unsqueeze",
    "topk": "topk",
    "sin": "sin",
    "cos": "cos",
    "round": "round",
    # Phase 10: in-place method aliases
    "relu_": "relu",
    "abs_": "abs",
    "neg_": "neg",
    "clamp_": "clamp",
    "clamp_min_": "clamp_min",
    "clamp_max_": "clamp_max",
    "sqrt_": "sqrt",
    "exp_": "exp",
    "log_": "log",
    "sin_": "sin",
    "cos_": "cos",
    "ceil_": "ceil",
    "add_": "add",
    "sub_": "sub",
    "mul_": "mul",
    "div_": "div",
    "floor_": "floor",
    "round_": "round",
    "reciprocal_": "reciprocal",
    "sign_": "sign",
    "add_": "add",
    "sub_": "sub",
    "mul_": "mul",
    "div_": "div",
    "view_as": "view_as",
    "expand_as": "expand_as",
    "isreal": "isreal",
    "is_complex": "is_complex",
    "is_nonzero": "is_nonzero",
    "split": "split",
    "vsplit": "vsplit",
    "hsplit": "hsplit",
    "dsplit": "dsplit",
}

_SUPPORTED_OPS = set(_native.supported_targets())


def _target_key(target: Any) -> str:
    """Stable string key for an FX node target (function / OpOverload / str)."""
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    module = getattr(target, "__module__", None)
    name = getattr(target, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return str(target)


def canonical_op(node: torch.fx.Node) -> tuple[str, str] | None:
    """Map an FX node to ``(canonical_op, target_key)`` or ``None`` if unsupported."""
    if node.op == "call_function":
        key = _target_key(node.target)
        op = _FUNCTION_TO_OP.get(key) or _ATEN_TO_OP.get(key)
        if op is not None:
            return op, key
        return None
    if node.op == "call_method":
        op = _METHOD_TO_OP.get(str(node.target))
        if op is not None:
            return op, str(node.target)
        return None
    return None


def _is_tuple_source(node: torch.fx.Node) -> bool:
    """True if the node produces a tuple (shape/size) rather than a tensor.

    ``x.shape`` / ``x.size()`` / ``x.device`` return Python tuples/objects;
    indexing them with getitem must NOT be routed to the tensor ``select`` op.
    """
    if node.op == "get_attr":
        return False
    if node.op == "call_function":
        key = _target_key(node.target)
        if key == "builtins.getattr":
            args = list(node.args)
            if len(args) >= 2 and isinstance(args[1], str) and args[1] in ("shape", "device", "dtype", "layout"):
                return True
            return False
        if key in ("torch.Tensor.size", "torch.size", "_operator.getitem", "operator.getitem"):
            return True
        # aten ops producing a tuple
        if "aten." in key and any(s in key for s in ("_shape_as_tensor", "max", "min", "sort", "topk", "unbind", "chunk")):
            return True
        return False
    if node.op == "call_method":
        return str(node.target) in ("size", "shape", "unbind", "chunk")
    return False


def _getitem_as_select(node: torch.fx.Node) -> tuple[str, str] | None:
    """Map ``tensor[int]`` (getitem / aten.select.int) to the ``select`` op,
    or tuple indexing to the native ``getitem`` op.

    Returns ``(op, target_key)`` or ``None``.
    """
    key = _target_key(node.target)
    if key in ("_operator.getitem", "operator.getitem", "<built-in function getitem>"):
        args = list(node.args)
        if len(args) == 2 and isinstance(args[1], int):
            if _is_tuple_source(args[0]):
                return "getitem", key
            return "select", key
        return None
    if key in ("aten.select.int", "aten.select"):
        return "select", key
    return None


# Shape ops whose positional const args (after the tensor) should be promoted
# to kwargs so the engine can read them without needing to inspect slot values.
# Maps op name -> kwarg name that receives the list/scalar of const args.


# Shape ops whose positional const args (after the tensor) should be promoted
# to kwargs so the engine can read them without needing to inspect slot values.
# Maps op name -> kwarg name that receives the list/scalar of const args.
_SHAPE_OP_CONST_KWARGS: dict[str, str] = {
    "permute": "dims",
    "reshape": "shape",
    "expand": "shape",
    "flip": "dims",
    "getitem": "index",
    "broadcast_to": "shape",
    "view_as": "shape",
    "expand_as": "shape",
    # Tensor creation ops: positional shape/dims become kwargs
    "zeros": "shape",
    "ones": "shape",
    "full": "shape",
    "arange": "start",  # handled specially below
    "empty_strided": "size",
}

# Phase 3 ops whose positional const/seq args (kernel, stride, padding, ...)
# are promoted to kwargs. Maps op name -> kwarg name per positional slot.
# Values may be lists (seq refs) or scalars (const refs); the engine reads
# them via kwargs and applies torch defaults for anything absent.
_CONV_POOL_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "conv1d": ["stride", "padding", "dilation", "groups"],
    "conv2d": ["stride", "padding", "dilation", "groups"],
    "conv_transpose1d": ["stride", "padding", "output_padding", "groups", "dilation"],
    "conv_transpose2d": ["stride", "padding", "output_padding", "groups", "dilation"],
    "max_pool1d": ["kernel", "stride", "padding", "dilation", "ceil_mode"],
    "max_pool2d": ["kernel", "stride", "padding", "dilation", "ceil_mode"],
    "avg_pool1d": ["kernel", "stride", "padding", "ceil_mode", "count_include_pad"],
    "avg_pool2d": ["kernel", "stride", "padding", "ceil_mode", "count_include_pad"],
    "adaptive_avg_pool2d": ["output_size"],
    "adaptive_max_pool2d": ["output_size"],
    "upsample_nearest2d": ["size"],
    "upsample_bilinear2d": ["size"],
    "flatten": ["start_dim", "end_dim"],
    # F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    "batch_norm": ["training", "momentum", "eps"],
}

# Reductions/activations whose positional const/seq args (dim, keepdim) are
# promoted to kwargs.  Export graphs emit e.g. ``aten.sum.dim_IntList(x, [1])``
# with dim as a positional seq; the engine only reads these from kwargs.
_REDUCE_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "sum": ["dim", "keepdim"],
    "mean": ["dim", "keepdim"],
    "max_reduce": ["dim", "keepdim"],
    "min_reduce": ["dim", "keepdim"],
    "argmax": ["dim", "keepdim"],
    "argmin": ["dim", "keepdim"],
    "std": ["dim", "keepdim"],
    "var": ["dim", "keepdim"],
    "cumsum": ["dim"],
    "prod": ["dim", "keepdim"],
    "norm": ["p", "dim", "keepdim"],
    "linalg_vector_norm": ["ord", "dim", "keepdim"],
    "softmax": ["dim"],
    "log_softmax": ["dim"],
    "threshold_backward": ["threshold"],
    "clamp": ["min", "max"],
    "clamp_min": ["min"],
    "clamp_max": ["max"],
    "isclose": ["rtol", "atol", "equal_nan"],
    "allclose": ["rtol", "atol", "equal_nan"],
    "nanprod": ["dim", "keepdim"],
    "nanmin": ["dim", "keepdim"],
    "nanmax": ["dim", "keepdim"],
    "var_mean": ["dim", "keepdim"],
    "std_mean": ["dim", "keepdim"],
    "nanmedian": ["dim", "keepdim"],
    "logsumexp": ["dim", "keepdim"],
    "cov": ["correction"],
    "linalg_vecdot": ["dim"],
    "linalg_cross": ["dim"],
    "linalg_tensordot": ["dims"],
    "adaptive_avg_pool1d": ["output_size"],
    "adaptive_max_pool1d": ["output_size"],
    "lp_pool3d": ["norm_type", "kernel_size", "stride"],
    "local_response_norm": ["size", "alpha", "beta", "k"],
}

# aten.transpose(x, d0, d1) — the two dims are positional consts.
_TRANSPOSE_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "transpose": ["d0", "d1"],
    "index_select": ["dim"],
    "gather": ["dim"],
    "narrow": ["dim", "start", "length"],
    "select": ["dim", "index"],
    "roll": ["shift", "dim"],
    "tile": ["repeats"],
    "pixel_shuffle": ["upscale_factor"],
    "instance_norm": ["eps"],
    "ldexp": ["other"],
    "split": ["split_size", "dim"],
    "vsplit": ["sections"],
    "hsplit": ["sections"],
    "dsplit": ["sections"],
    "tensor_split": ["indices", "dim"],
    "take_along_dim": ["dim"],
    "index_reduce": ["dim", "reduce"],
    "scatter_max": ["dim"],
    "scatter_min": ["dim"],
    "as_strided": ["size", "stride", "storage_offset"],
    "broadcast_to": ["shape"],
    "linalg_vander": ["N"],
    "linalg_cholesky_ex": ["upper"],
    "linalg_inv_ex": ["check_errors"],
    "linalg_solve_ex": ["check_errors"],
    "linalg_lu_factor": ["pivot"],
    "logsumexp": ["dim", "keepdim"],
}

# Phase 4: SDPA positional consts.  Export graphs emit either
# ``aten.scaled_dot_product_attention(q, k, v, mask, dropout_p, is_causal)``
# or the fused variant ``(q, k, v, dropout_p, is_causal)`` — promote the
# trailing scalars so the engine reads them from kwargs.
_SDPA_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "scaled_dot_product_attention": ["dropout_p", "is_causal"],
}

# Ops whose engine kernel consumes a fixed prefix of tensor args.  Dynamo
# replays a function's full positional signature (defaults included), so e.g.
# F.embedding arrives with padding_idx/max_norm/norm_type/scale_grad_by_freq/
# sparse as trailing consts; they are dead for the engine and must be trimmed
# (bool consts would also trip the runtime bool-const guard in _compiled.py).
_FIXED_TENSOR_ARITY: dict[str, int] = {
    "embedding": 2,  # (weight, indices)
}

# Phase 4: loss positional consts (reduction enum, ignore_index, beta).
_LOSS_POSITIONAL_KWARGS: dict[str, list[str]] = {
    "nll_loss_forward": ["reduction", "ignore_index"],
    "mse_loss": ["reduction"],
    "smooth_l1_loss": ["reduction", "beta"],
    "binary_cross_entropy": ["reduction"],
    # aten.scalar_tensor(-inf, dtype=...) — the value is a positional const.
    "scalar_tensor": ["value"],
}

# Engine ops whose aten counterpart returns a TUPLE (see the parser pre-pass):
# the engine produces element 0 natively; getitem(0) aliases it, a consumed
# getitem(1) forces eager.
_TUPLE_OUTPUT_OPS = frozenset({"max_reduce", "min_reduce", "nll_loss_forward", "scaled_dot_product_attention", "sort", "unbind", "chunk",
"var_mean","std_mean","linalg_cholesky_ex","linalg_inv_ex","linalg_solve_ex","linalg_lu_factor","split","vsplit","hsplit","dsplit","tensor_split","broadcast_tensors","qr","svd","eig","eigh","lu","linalg_slogdet","linalg_cholesky","lstm_cell"})


def _promote_positional_args_to_kwargs(
    op: str, args: list[dict[str, Any]], existing_kwargs: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Promote positional const/seq args of Phase 3 ops into kwargs.

    e.g. F.max_pool2d(x, 2, 2, 0, 1) -> args=[x], kwargs={"kernel":2,"stride":2,"padding":0,"dilation":1}
         torch.conv2d(x, w, b, [1,1], [1,1], [1,1], 1) -> args=[x,w,b], kwargs={...}

    Tensor refs (input/node/attr) are kept as positional args; every other
    positional arg is mapped to the next named kwarg for that op.
    """
    names = (
        _CONV_POOL_POSITIONAL_KWARGS.get(op)
        or _REDUCE_POSITIONAL_KWARGS.get(op)
        or _SDPA_POSITIONAL_KWARGS.get(op)
        or _LOSS_POSITIONAL_KWARGS.get(op)
        or _TRANSPOSE_POSITIONAL_KWARGS.get(op)
    )
    if names is None:
        return args, existing_kwargs

    new_args: list[dict[str, Any]] = []
    new_kwargs = dict(existing_kwargs)
    name_iter = iter(names)
    optional_tensor_ops = {
        "conv1d", "conv2d", "conv_transpose1d", "conv_transpose2d",
        "nll_loss_forward", "cross_entropy_loss", "linear", "addmm"
    }
    for arg in args:
        if arg.get("kind") in ("input", "node", "attr"):
            new_args.append(arg)
            continue
        if op in optional_tensor_ops and arg.get("kind") == "const" and arg.get("value") is None:
            # Explicit None for optional tensor parameter (e.g. bias=None or weight=None):
            # drop it without advancing the kwarg name iterator.
            continue
        name = next(name_iter, None)
        if name is None:
            continue
        if arg.get("kind") == "const" and arg.get("value") is None:
            # Explicit None for named kwarg (e.g. clamp min=None): advance kwarg slot without setting
            continue
        if name in new_kwargs:
            continue  # an explicit kwarg already wins
        if arg.get("kind") == "seq":
            vals = [item["value"] for item in arg.get("value", []) if item.get("kind") == "const"]
            new_kwargs[name] = vals if len(vals) > 1 else (vals[0] if vals else None)
        else:
            new_kwargs[name] = arg.get("value")
    return new_args, new_kwargs


def _promote_const_args_to_kwargs(
    op: str, args: list[dict[str, Any]], existing_kwargs: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """For shape ops, extract positional const args after the tensor into kwargs.

    e.g. permute(tensor, 1, 0) -> args=[tensor_ref], kwargs={"dims": [1, 0]}
         reshape(tensor, -1)   -> args=[tensor_ref], kwargs={"shape": [-1]}

    This makes the Rust engine dispatch straightforward regardless of whether
    the op was called as a method or a function.
    """
    # select() has position-dependent const args: getitem(t, i) -> dim=0,index=i;
    # aten.select.int(t, dim, index) -> dim, index as written.
    if op == "select" and "dim" not in existing_kwargs:
        rest = [a for a in args[1:] if a.get("kind") in ("const", "seq")]
        values: list[Any] = []
        for a in rest:
            if a.get("kind") == "const":
                values.append(a["value"])
            elif a.get("kind") == "seq":
                values.extend(v for v in a.get("value", []) if isinstance(v, (int, float)))
        if len(values) == 1 and isinstance(values[0], int):
            return [args[0]], {**existing_kwargs, "dim": 0, "index": values[0]}
        if len(values) >= 2 and isinstance(values[0], int) and isinstance(values[1], int):
            return [args[0]], {**existing_kwargs, "dim": values[0], "index": values[1]}

    kwarg_key = _SHAPE_OP_CONST_KWARGS.get(op)
    if kwarg_key is None or kwarg_key in existing_kwargs:
        return args, existing_kwargs

    # Collect trailing const/seq args (everything after the first tensor ref).
    # e.g. view(x, (B, T, H, D)) arrives as a positional seq of consts.
    tensor_args = []
    const_vals = []
    seen_tensor = False
    for arg in args:
        kind = arg.get("kind")
        if seen_tensor and kind == "const":
            const_vals.append(arg["value"])
        elif seen_tensor and kind == "seq":
            items = arg.get("value", [])
            if all(item.get("kind") == "const" for item in items):
                const_vals.extend(item["value"] for item in items)
            else:
                tensor_args.append(arg)  # seq holding tensor refs: keep positional
        else:
            tensor_args.append(arg)
            if kind in ("input", "node", "attr"):
                seen_tensor = True

    if not const_vals:
        return args, existing_kwargs

    new_kwargs = dict(existing_kwargs)
    new_kwargs[kwarg_key] = const_vals
    return tensor_args, new_kwargs


def _dtype_str(dtype: torch.dtype) -> str | None:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float64:
        return "f64"
    return None


def parse_graph(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> tuple[dict[str, Any], dict[str, Callable]]:
    """Convert a GraphModule into a JSON-serializable plan.

    Returns ``(plan, function_map)`` where ``function_map`` maps target keys
    to the original callables (needed to re-invoke unsupported nodes eagerly;
    callables are not JSON-serializable so they travel separately).
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    tensor_inputs = [t for t in example_inputs if isinstance(t, torch.Tensor)]
    if len(tensor_inputs) == 0 or len(tensor_inputs) > len(placeholders):
        raise ValueError("torchburn: no tensor example inputs found for graph")
    tensor_placeholder_ids: set[int] = set()
    scalar_placeholder_ids: set[int] = set()

    node_id: dict[torch.fx.Node, int] = {}
    nodes: list[dict[str, Any]] = []
    function_map: dict[str, Callable] = {}
    next_id = 0
    tensor_index = 0

    # ------------------------------------------------------------------ pre-pass
    # Ops whose engine implementation returns ONE tensor while the aten op
    # returns a TUPLE unpacked by getitem(0)/getitem(1):
    #   * max_reduce / min_reduce -> (values, indices)
    #   * nll_loss_forward -> (loss, total_weight)
    #   * scaled_dot_product_attention (fused variants) -> (out, attn_weights)
    #   * unbind -> (slice_0, slice_1, ...)
    #   * chunk -> (part_0, part_1, ...)
    #   * sort -> (values, indices)
    #
    # For N-element tuples: if any non-zero-index getitem is consumed by
    # downstream nodes, the whole tuple node runs eagerly (the engine can
    # produce tuples but the interpreter doesn't map element-encoded capsule
    # outputs back to individual getitem results).  When only getitem(0) is
    # used (the common pattern for max/min/sort), we alias it to the node's
    # output slot and drop the dead getitem nodes.
    getitem_alias: dict[torch.fx.Node, torch.fx.Node] = {}
    drop_nodes: set[torch.fx.Node] = set()  # dead getitem nodes — never execute
    force_eager: set[torch.fx.Node] = set()
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        mapped = canonical_op(n)
        if mapped is None or mapped[0] not in _TUPLE_OUTPUT_OPS:
            continue
        # Collect ALL getitem consumers with their element indices.
        getitem_by_idx: dict[int, torch.fx.Node] = {}
        for user in n.users:
            if user.op != "call_function":
                continue
            if _target_key(user.target) != "_operator.getitem" and str(user.target) not in ("<built-in function getitem>", "operator.getitem"):
                continue
            if len(user.args) != 2 or not isinstance(user.args[1], int):
                continue
            getitem_by_idx[user.args[1]] = user
        # If ANY non-zero-index getitem is consumed downstream, force eager.
        has_consumed_nonzero = False
        for idx, gi_node in getitem_by_idx.items():
            if idx > 0 and any(u is not n for u in gi_node.users):
                has_consumed_nonzero = True
                break
        if has_consumed_nonzero:
            force_eager.add(n)
        else:
            # Only element-0 is consumed (or no consumers at all).
            # Alias getitem(0) to the tuple-producing node; drop all dead getitems.
            gi0 = getitem_by_idx.get(0)
            if gi0 is not None:
                getitem_alias[gi0] = n
            for idx, gi_node in getitem_by_idx.items():
                if gi_node is not gi0 or gi0 is None:
                    drop_nodes.add(gi_node)



    for n in gm.graph.nodes:
        if n.op == "placeholder":
            pos = placeholders.index(n)
            node_id[n] = next_id
            if pos < len(example_inputs) and isinstance(example_inputs[pos], torch.Tensor):
                nodes.append({"id": next_id, "op": "placeholder", "pos": pos, "index": tensor_index})
                tensor_placeholder_ids.add(next_id)
                tensor_index += 1
            else:
                nodes.append({"id": next_id, "op": "placeholder", "pos": pos, "index": -1})
                scalar_placeholder_ids.add(next_id)
            next_id += 1
        elif n.op == "get_attr":
            node_id[n] = next_id
            nodes.append({"id": next_id, "op": "get_attr", "target": n.target})
            next_id += 1
        elif n.op == "output":
            node_id[n] = next_id
            nodes.append({"id": next_id, "op": "output", "args": [_ref(a, node_id) for a in n.args]})
            next_id += 1
        elif n.op in ("call_function", "call_method", "call_module"):
            # getitem(0) on a max/min_reduce: alias to the reduce node's slot
            # (the reduce output IS the values tensor).  Dead getitem(1) nodes
            # are dropped — running them eagerly would index the values tensor
            # instead of the (nonexistent) indices tuple.
            if n in drop_nodes:
                continue
            if n in getitem_alias:
                node_id[n] = node_id[getitem_alias[n]]
                continue

            mapped = canonical_op(n)
            if mapped is None:
                mapped = _getitem_as_select(n)
            target_key = _target_key(n.target)
            args = [_ref(a, node_id) for a in n.args]
            # The fused SDPA variants pass the attention mask as a kwarg
            # (attn_mask=<node>); move it into position 3 so the engine sees
            # q, k, v, mask, with dropout_p/is_causal promoted to kwargs.
            if mapped is not None and mapped[0] == "scaled_dot_product_attention":
                mask_node = n.kwargs.get("attn_mask")
                if isinstance(mask_node, torch.fx.Node):
                    args.append(_ref(mask_node, node_id))
            # to_dtype: method calls like x.float() / x.double() / x.to(dtype)
            # don't carry a serializable dtype kwarg.  Infer from the method name.
            _DTYPE_FROM_METHOD: dict[str, str] | None = None
            if mapped is not None and mapped[0] == "to_dtype" and "dtype" not in (n.kwargs or {}):
                _DTYPE_FROM_METHOD = {"float": "f32", "double": "f64", "half": "f16",
                                      "bfloat16": "bf16", "int": "i32", "long": "i64"}
            # Allow call_function and call_method nodes with kwargs — the engine
            # reads kwargs for dim/keepdim/eps/etc.  call_module nodes keep the
            # old strict check since their kwargs wiring is not yet mapped.
            kwargs_ok = (not n.kwargs) if n.op == "call_module" else True
            if mapped is not None and n not in force_eager and kwargs_ok and _arg_count_ok(mapped[0], args):
                op, key = mapped
                # For shape ops called as methods, promote trailing const args
                # (e.g. permute(1, 0), reshape(-1)) into kwargs so the engine
                # can read them without decoding scalar slot values.
                extracted_kwargs = _extract_kwargs(n.kwargs, node_id)
                # Inject dtype for method calls like x.float() / x.double()
                if _DTYPE_FROM_METHOD is not None and "dtype" not in extracted_kwargs:
                    method_name = str(n.target)
                    dtype = _DTYPE_FROM_METHOD.get(method_name)
                    if dtype is not None:
                        extracted_kwargs["dtype"] = dtype
                args, extracted_kwargs = _promote_const_args_to_kwargs(op, args, extracted_kwargs)
                args, extracted_kwargs = _promote_positional_args_to_kwargs(op, args, extracted_kwargs)
                # Ops whose engine kernel consumes a fixed prefix of tensor args:
                # drop trailing const defaults dynamo injects (e.g. F.embedding's
                # padding_idx/max_norm/norm_type/scale_grad_by_freq/sparse). They
                # are dead for the engine and bool consts would trip the runtime
                # bool-const guard, forcing an eager fallback.
                fixed = _FIXED_TENSOR_ARITY.get(op)
                if fixed is not None and len(args) > fixed:
                    args = args[:fixed]
                # F.embedding(input, weight) has the Python signature, but the
                # ATen schema (and the engine kernel) is embedding(weight,
                # indices) — make_fx emits the ATen order, dynamo emits the
                # Python order.  Normalise to (weight, indices).
                if op == "embedding" and target_key.startswith("torch.nn.functional"):
                    if len(args) == 2:
                        args = [args[1], args[0]]
                # aten.std.correction / aten.var.correction pass ``correction``
                # (torch's divisor adjustment) as a kwarg, but the engine reads
                # ``unbiased`` (bool).  Map 0 -> False, 1 -> True; anything else
                # (correction=2 etc.) is unsupported and falls back to eager.
                if op in ("std", "var") and "correction" in extracted_kwargs:
                    corr = extracted_kwargs.pop("correction")
                    if corr not in (0, 1):
                        mapped = None
                    else:
                        extracted_kwargs["unbiased"] = bool(corr)
                # Tensor creation ops: all args are shape consts, no tensor input.
                # Collect all const args into a single "shape" list kwarg.
                if op in ("zeros", "ones", "full") and "shape" not in extracted_kwargs:
                    shape_vals = []
                    for a in args:
                        if isinstance(a, dict) and a.get("kind") == "const":
                            shape_vals.append(a["value"])
                    if shape_vals:
                        args = [args[0]] if args else []  # keep first ref (even if const)
                        extracted_kwargs["shape"] = shape_vals
                if op == "arange" and not extracted_kwargs:
                    # torch.arange(end) or torch.arange(start, end, step)
                    vals = []
                    for a in args:
                        if isinstance(a, dict) and a.get("kind") == "const":
                            vals.append(a["value"])
                    if len(vals) >= 1:
                        extracted_kwargs["end"] = vals[-1]
                    if len(vals) >= 2:
                        extracted_kwargs["start"] = vals[0]
                    if len(vals) >= 3:
                        extracted_kwargs["step"] = vals[1]
                    args = []
                # einsum: equation string is args[0] (const), tensors start at args[1]
                if op == "einsum" and args and isinstance(args[0], dict) and args[0].get("kind") == "const":
                    extracted_kwargs["equation"] = args[0]["value"]
                    args = args[1:]
                if mapped is not None and n.op == "call_function":
                    function_map[key] = n.target
                if mapped is None:
                    if n.op == "call_function":
                        function_map[target_key] = n.target
                    node_id[n] = next_id
                    nodes.append(
                        {
                            "id": next_id,
                            "op": "unsupported",
                            "fx_op": n.op,
                            "fx_target": target_key,
                            "args": args,
                            "fx_args": [_ref(a, node_id) for a in n.args],
                            # kwargs feeds the JSON signature payload: keep only
                            # serializable primitives (device/dtype objects and
                            # tensor refs live in fx_kwargs for eager replay).
                            "kwargs": _extract_kwargs(n.kwargs, node_id),
                            "fx_kwargs": _ref_kwargs(n.kwargs, node_id),
                        }
                    )
                else:
                    node_id[n] = next_id
                    nodes.append(
                        {
                            "id": next_id,
                            "op": "supported",
                            "target": op,
                            "fx_op": n.op,
                            "fx_target": target_key,
                            "args": args,
                            "fx_args": [_ref(a, node_id) for a in n.args],
                            "kwargs": extracted_kwargs,
                            "fx_kwargs": _ref_kwargs(n.kwargs, node_id),
                        }
                    )
            else:
                if n.op == "call_function":
                    function_map[target_key] = n.target
                node_id[n] = next_id
                nodes.append(
                    {
                        "id": next_id,
                        "op": "unsupported",
                        "fx_op": n.op,
                        "fx_target": target_key,
                        "args": args,
                        "fx_args": [_ref(a, node_id) for a in n.args],
                        "kwargs": _extract_kwargs(n.kwargs, node_id),
                        "fx_kwargs": _ref_kwargs(n.kwargs, node_id),
                    }
                )
            next_id += 1
        else:
            raise ValueError(f"torchburn: unexpected FX node op {n.op!r}")

    input_spec = [
        {"shape": [int(s) for s in t.shape], "dtype": _dtype_str(t.dtype)}
        for t in tensor_inputs
    ]
    return {
        "nodes": nodes,
        "input_spec": input_spec,
        "tensor_placeholders": sorted(tensor_placeholder_ids),
        "scalar_placeholders": sorted(scalar_placeholder_ids),
    }, function_map


def _ref_kwargs(fx_kwargs: Any, node_id: dict) -> dict[str, Any]:
    """Convert FX node kwargs for the eager fallback (REQ-002).

    Tensor-valued kwargs (e.g. ``attn_mask=<Node>``) must become refs so
    ``_run_eager`` can resolve them from the env; raw ``torch.fx.Node``
    objects are not serialisable and would otherwise reach eager as Nodes.
    """
    out: dict[str, Any] = {}
    for k, v in fx_kwargs.items():
        if isinstance(v, torch.fx.Node):
            out[k] = _ref(v, node_id)
        elif isinstance(v, (list, tuple)) and any(isinstance(i, torch.fx.Node) for i in v):
            out[k] = {"kind": "seq", "value": [_ref(i, node_id) if isinstance(i, torch.fx.Node) else i for i in v],
                       "type": "tuple" if isinstance(v, tuple) else "list"}
        else:
            out[k] = v
    return out


def _extract_kwargs(fx_kwargs: Any, node_id: dict) -> dict[str, Any]:
    """Convert FX node kwargs into JSON-serializable primitives for the engine.

    Tensor-valued kwargs are not expected at the ops we support; they come as
    positional args or get_attr refs.  We serialise scalars, booleans, and
    simple sequences; anything we can't represent becomes None (safe: the
    engine falls back to its own defaults).
    """
    out: dict[str, Any] = {}
    for k, v in fx_kwargs.items():
        if isinstance(v, torch.dtype):
            # dtype kwarg (e.g. aten._to_copy(dtype=torch.float64)) -> "f32"/"f64"
            if v == torch.float32:
                out[k] = "f32"
            elif v == torch.float64:
                out[k] = "f64"
            continue
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            # e.g. dims=[0, 1], normalized_shape=(16,)
            serialized = []
            for item in v:
                if isinstance(item, (bool, int, float)):
                    serialized.append(item)
                else:
                    serialized = None  # unserialisable element
                    break
            if serialized is not None:
                out[k] = serialized
        # Skip tensors and other complex objects; the engine uses defaults.
    return out


def _arg_count_ok(op: str, args: list[dict[str, Any]]) -> bool:
    """Check if the arg count is reasonable for this op."""
    # Unary ops: 1 arg
    unary_ops = {"relu", "abs", "neg", "sign", "sqrt", "rsqrt", "exp", "log",
                 "reciprocal", "ceil", "floor", "round", "sin", "cos",
                 "sigmoid", "tanh", "gelu", "silu",
                 "elu", "selu", "softplus", "hardswish", "mish", "softmax",
                 "log_softmax", "norm"}
    # Binary ops: 2 args
    binary_ops = {"add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge"}
    # Reduction ops: 1-2 args (tensor + optional dim)
    reduce_ops = {"sum", "mean", "max_reduce", "min_reduce", "argmax", "argmin", "std", "var",
                  "cumsum", "prod"}
    # Linalg ops: 2-3 args
    linalg_ops = {"matmul", "bmm", "dot"}
    # 3-arg ops
    ternary_ops = {"where", "masked_fill"}
    # clamp: 1 tensor arg + kwargs (min/max)
    clamp_ops = {"clamp", "clamp_min", "clamp_max"}

    if op in unary_ops:
        return len(args) >= 1
    if op in binary_ops:
        return len(args) >= 2
    if op in reduce_ops:
        return len(args) >= 1
    if op in linalg_ops:
        return len(args) >= 2
    if op in ternary_ops:
        return len(args) >= 3
    if op in clamp_ops:
        return len(args) >= 1
    # ops with variable args (cat, stack, layer_norm, batch_norm, etc.)
    return True


def _ref(a: Any, node_id: dict[torch.fx.Node, int]) -> Any:
    """Serialize an FX node argument into a plan reference."""
    if isinstance(a, torch.fx.Node):
        if a.op == "placeholder":
            return {"kind": "input", "index": node_id[a]}
        if a.op == "get_attr":
            return {"kind": "attr", "index": node_id[a]}
        return {"kind": "node", "index": node_id[a]}
    if a is Ellipsis:
        # x[..., :half] — ``...`` must round-trip as Ellipsis, not None (None
        # would add a new axis in eager getitem).
        return {"kind": "const", "value": "__ellipsis__"}
    if isinstance(a, slice):
        # x[..., :half] — slice args must round-trip for eager fallback.
        return {
            "kind": "slice",
            "start": _ref(a.start, node_id) if isinstance(a.start, torch.fx.Node) else a.start,
            "stop": _ref(a.stop, node_id) if isinstance(a.stop, torch.fx.Node) else a.stop,
            "step": a.step,
        }
    if isinstance(a, (list, tuple)):
        return {"kind": "seq", "type": type(a).__name__, "value": [_ref(x, node_id) for x in a]}
    if isinstance(a, torch.dtype):
        # dtype arg (e.g. x.to(torch.float64)) -> "f32"/"f64" string
        return {"kind": "const", "value": "f32" if a == torch.float32 else "f64"}
    if a is None or isinstance(a, (bool, int, float, str)):
        return {"kind": "const", "value": a}
    return {"kind": "const", "value": None}


def _sanitize_nonfinite(obj: Any) -> Any:
    """Replace non-finite floats with string tokens.

    Python's json emits ``Infinity``/``NaN`` which serde_json rejects, so
    masks built from ``-inf`` scalars would break the payload.  The engine
    decodes the ``"inf"``/``"-inf"``/``"nan"`` strings back to floats.
    """
    import math

    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)  # 'inf', '-inf', 'nan'
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


def payload_json(plan: dict[str, Any]) -> str:
    """Canonical JSON form of a plan (sorted keys => stable BLAKE3 signature).

    ``fx_kwargs`` holds the ORIGINAL FX kwargs (may contain non-serializable
    values like torch.dtype) for eager fallback replay; it is excluded from
    the signature payload because it is never sent to the Rust engine.
    """
    # Deep-copy so we never mutate the live plan (which carries fx_kwargs).
    nodes = []
    for node in plan.get("nodes", []):
        copy = dict(node)
        copy.pop("fx_kwargs", None)
        nodes.append(copy)
    sig = dict(plan)
    sig.pop("fx_kwargs", None)
    sig["nodes"] = nodes
    sig = _sanitize_nonfinite(sig)
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))
