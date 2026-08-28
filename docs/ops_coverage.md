# TorchBurn Operator Coverage

## Summary

TorchBurn supports **130+ native operators** covering the most common neural network architectures.

| Category | Native Ops | Fallback |
|----------|-----------|----------|
| Elementwise | 20+ | None |
| Math/Comparison | 25+ | None |
| Activations | 15+ | None |
| Reductions | 12+ | None |
| Linalg | 8+ | None |
| Normalization | 4 | None |
| Shape Ops | 15+ | None |
| Convolution | 6 | None |
| Pooling | 8+ | None |
| Upsampling | 4 | None |
| Transformer | 6 | None |
| Embedding | 3 | None |
| Losses | 4 | None |
| Autograd | 33 backward | None |

## Detailed Coverage

### Elementwise Operations
| Op | Parser Target | Backward |
|----|---------------|----------|
| `add` | `aten.add.Tensor` | `grad + other` |
| `sub` | `aten.sub.Tensor` | `grad, -grad` |
| `mul` | `aten.mul.Tensor` | `grad * other, grad * input` |
| `div` | `aten.div.Tensor` | `-grad / other, grad * input / other²` |
| `relu` | `aten.relu.default` | `grad * (x > 0)` |
| `abs` | `aten.abs.default` | `grad * sign(x)` |
| `neg` | `aten.neg.default` | `-grad` |
| `exp` | `aten.exp.default` | `grad * exp(x)` |
| `log` | `aten.log.default` | `grad / x` |
| `sqrt` | `aten.sqrt.default` | `0.5 / sqrt(x) * grad` |
| `clamp` | `aten.clamp.default` | `grad * mask` |
| `where` | `aten.where.self` | `grad * mask, grad * ~mask` |
| `eq` | `aten.eq.Tensor` | None (bool) |
| `ne` | `aten.ne.Tensor` | None (bool) |
| `lt` | `aten.lt.Tensor` | None (bool) |
| `le` | `aten.le.Tensor` | None (bool) |
| `gt` | `aten.gt.Tensor` | None (bool) |
| `ge` | `aten.ge.Tensor` | None (bool) |
| `logical_and` | `aten.logical_and` | None (bool) |
| `logical_or` | `aten.logical_or` | None (bool) |
| `logical_not` | `aten.logical_not` | None (bool) |

### Math Operations
| Op | Parser Target | Notes |
|----|---------------|-------|
| `pow` | `aten.pow.Tensor` | Elementwise power |
| `sin` | `aten.sin` | Trig |
| `cos` | `aten.cos` | Trig |
| `tan` | `aten.tan` | Trig |
| `asin` | `aten.asin` | Inverse trig |
| `acos` | `aten.acos` | Inverse trig |
| `atan` | `aten.atan` | Inverse trig |
| `sinh` | `aten.sinh` | Hyperbolic |
| `cosh` | `aten.cosh` | Hyperbolic |
| `tanh` | `aten.tanh` | Hyperbolic |
| `ceil` | `aten.ceil` | Rounding |
| `floor` | `aten.floor` | Rounding |
| `round` | `aten.round` | Rounding |
| `sign` | `aten.sign` | Sign |
| `reciprocal` | `aten.reciprocal` | 1/x |
| `isinf` | `aten.isinf` | Check |
| `isnan` | `aten.isnan` | Check |

### Activations
| Op | Parser Target | Notes |
|----|---------------|-------|
| `sigmoid` | `aten.sigmoid.default` | 1 / (1 + exp(-x)) |
| `tanh` | `aten.tanh.default` | tanh(x) |
| `gelu` | `aten.gelu.default` | Gaussian error linear |
| `silu` | `aten.silu.default` | x * sigmoid(x) |
| `leaky_relu` | `aten.leaky_relu.default` | max(x, α*x) |
| `elu` | `aten.elu.default` | x if x>0, α(exp(x)-1) |
| `selu` | `aten.selu.default` | Self-normalizing ELU |
| `softplus` | `aten.softplus.default` | log(1 + exp(x)) |
| `hardswish` | `aten.hardswish.default` | x * ReLU6(x+3)/6 |
| `mish` | `aten.mish.default` | x * tanh(softplus(x)) |
| `softmax` | `aten.softmax.int` | exp(x) / sum(exp(x)) |
| `log_softmax` | `aten.log_softmax.int` | log(softmax(x)) |

### Reductions
| Op | Parser Target | Notes |
|----|---------------|-------|
| `sum` | `aten.sum.dim_IntList` | Reduce along dims |
| `mean` | `aten.mean.dim` | Average along dims |
| `max` | `aten.max.dim` | Max along dims |
| `min` | `aten.min.dim` | Min along dims |
| `argmax` | `aten.argmax.default` | Index of max |
| `argmin` | `aten.argmin.default` | Index of min |
| `std` | `aten.std.dim` | Standard deviation |
| `var` | `aten.var.dim` | Variance |
| `cumsum` | `aten.cumsum.default` | Cumulative sum |
| `prod` | `aten.prod.dim_int` | Product along dims |
| `norm` | `aten.norm.default` | L2 norm |

### Linear Algebra
| Op | Parser Target | Notes |
|----|---------------|-------|
| `matmul` | `aten.matmul` | Matrix multiply |
| `bmm` | `aten.bmm` | Batched matmul |
| `linear` | `aten.linear` | Linear layer |
| `dot` | `aten.dot` | Vector dot product |
| `addmm` | `aten.addmm` | mat + matmul |
| `t` | `aten.t` | Transpose 2D |

### Normalization
| Op | Parser Target | Notes |
|----|---------------|-------|
| `layer_norm` | `aten.layer_norm` | Layer normalization |
| `batch_norm` | `aten.batch_norm` | Batch normalization |
| `group_norm` | `aten.group_norm` | Group normalization |
| `rms_norm` | `aten.rms_norm` | RMS normalization |

### Shape Operations
| Op | Parser Target | Notes |
|----|---------------|-------|
| `cat` | `aten.cat` | Concatenate tensors |
| `stack` | `aten.stack` | Stack along new dim |
| `reshape` | `aten.reshape` | Reshape tensor |
| `permute` | `aten.permute` | Permute dimensions |
| `expand` | `aten.expand` | Expand tensor |
| `flip` | `aten.flip` | Flip dimensions |
| `narrow` | `aten.narrow` | Slice along dim |
| `select` | `aten.select.int` | Select element along dim |
| `contiguous` | `aten.contiguous` | Make contiguous |
| `squeeze` | `aten.squeeze` | Remove dim of size 1 |
| `unsqueeze` | `aten.unsqueeze` | Add dim of size 1 |
| `flatten` | `aten.flatten` | Flatten dimensions |
| `unflatten` | `aten.unflatten.int` | Unflatten dimension |

### Convolution
| Op | Parser Target | Notes |
|----|---------------|-------|
| `conv1d` | `aten.conv1d` | 1D convolution |
| `conv2d` | `aten.conv2d` | 2D convolution |
| `conv_transpose1d` | `aten.conv_transpose1d` | Transposed 1D conv |
| `conv_transpose2d` | `aten.conv_transpose2d` | Transposed 2D conv |

### Pooling
| Op | Parser Target | Notes |
|----|---------------|-------|
| `max_pool1d` | `aten.max_pool1d` | Max pooling 1D |
| `max_pool2d` | `aten.max_pool2d` | Max pooling 2D |
| `avg_pool1d` | `aten.avg_pool1d` | Average pooling 1D |
| `avg_pool2d` | `aten.avg_pool2d` | Average pooling 2D |
| `adaptive_avg_pool1d` | `aten.adaptive_avg_pool1d` | Adaptive avg pool 1D |
| `adaptive_avg_pool2d` | `aten.adaptive_avg_pool2d` | Adaptive avg pool 2D |
| `adaptive_max_pool1d` | `aten.adaptive_max_pool1d` | Adaptive max pool 1D |
| `adaptive_max_pool2d` | `aten.adaptive_max_pool2d` | Adaptive max pool 2D |

### Upsampling
| Op | Parser Target | Notes |
|----|---------------|-------|
| `nearest` | `aten.upsample_nearest` | Nearest neighbor |
| `bilinear` | `aten.upsample_bilinear` | Bilinear interpolation |

### Transformer
| Op | Parser Target | Notes |
|----|---------------|-------|
| `scaled_dot_product_attention` | `aten.scaled_dot_product_attention` | Fused attention |
| `rope` | `aten.rope` | Rotary position embedding |
| `embedding` | `aten.embedding` | Embedding lookup |
| `index_select` | `aten.index_select` | Index select |
| `gather` | `aten.gather` | Gather along dim |

### Losses
| Op | Parser Target | Notes |
|----|---------------|-------|
| `nll_loss` | `aten.nll_loss` | Negative log likelihood |
| `mse_loss` | `aten.mse_loss` | Mean squared error |
| `smooth_l1_loss` | `aten.smooth_l1_loss` | Smooth L1 (Huber) |
| `binary_cross_entropy` | `aten.binary_cross_entropy` | BCE |

### Phase 7 Extensions
| Op | Parser Target | Notes |
|----|---------------|-------|
| `scatter` | `aten.scatter.src` | Scatter values |
| `sort` | `aten.sort` | Sort along dim |
| `repeat` | `aten.repeat` | Repeat tensor |
| `prelu` | `aten.prelu` | Parametric ReLU |
| `nonzero` | `aten.nonzero` | Non-zero indices |
| `einsum` | `aten.einsum` | Einstein summation |
| `clamp_tensor` | `aten.clamp.Tensor` | Clamp with tensor bounds |
| `topk` | `aten.topk` | Top K elements |
| `argsort` | `aten.argsort` | Indices of sorted tensor |

### In-Place Aliases
| Op | Maps To |
|----|---------|
| `aten.add_.Tensor` | `add` |
| `aten.mul_.Tensor` | `mul` |
| `aten.sub_.Tensor` | `sub` |
| `aten.div_.Tensor` | `div` |
| `aten.relu_.default` | `relu` |
| `aten.abs_.default` | `abs` |
| `aten.neg_.default` | `neg` |
| `aten.clamp_.default` | `clamp` |

### Fallback Operations
| Op | Reason | Fallback Behavior |
|----|--------|-------------------|
| `dropout` | Inference only, passthrough | Copies tensor unchanged |
| `aten.zeros` | Constant creation | Falls back to eager |
| `aten.full` | Constant creation | Falls back to eager |
