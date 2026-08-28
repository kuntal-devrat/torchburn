"""Phase 3 operator coverage tests: convolution, pooling, upsampling.

Each test validates a native-engine kernel against PyTorch reference output,
exercising strides, padding, dilation, groups, dtypes, and edge cases. All
tests run against the same payload protocol as Phase 1/2 tests.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from torchburn import _torchburn as tb


def _spec(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "dtype": "f32" if t.dtype == torch.float32 else "f64"}


def run_conv(target: str, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, **kwargs) -> torch.Tensor:
    args = [{"kind": "slot", "index": 0}, {"kind": "slot", "index": 1}]
    inputs = [x, w]
    if bias is not None:
        args.append({"kind": "slot", "index": 2})
        inputs.append(bias)
    payload = json.dumps({
        "inputs": [_spec(t) for t in inputs],
        "nodes": [{"id": 0, "target": target, "args": args, "kwargs": kwargs}],
        "outputs": [0],
    }, sort_keys=True)
    (cap,) = tb.execute(payload, [t.__dlpack__() for t in inputs])
    return torch.from_dlpack(cap)


def run_pool(target: str, x: torch.Tensor, **kwargs) -> torch.Tensor:
    payload = json.dumps({
        "inputs": [_spec(x)],
        "nodes": [{"id": 0, "target": target,
                   "args": [{"kind": "slot", "index": 0}], "kwargs": kwargs}],
        "outputs": [0],
    }, sort_keys=True)
    (cap,) = tb.execute(payload, [x.__dlpack__()])
    return torch.from_dlpack(cap)


# ---------------------------------------------------------------------------
# 1. Convolution
# ---------------------------------------------------------------------------

class TestConv2d:
    @pytest.mark.parametrize("k,s,p", [
        (1, 1, 0), (3, 1, 1), (5, 2, 2), (7, 1, 3), (3, 2, 0),
    ])
    def test_conv2d_variants(self, k, s, p):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 16, 16)
        w = torch.randn(8, 3, k, k)
        b = torch.randn(8)
        got = run_conv("conv2d", x, w, b, stride=s, padding=p)
        ref = F.conv2d(x, w, b, stride=s, padding=p)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv2d_no_bias(self):
        x = torch.randn(2, 3, 16, 16)
        w = torch.randn(8, 3, 3, 3)
        got = run_conv("conv2d", x, w, None, stride=1, padding=1)
        ref = F.conv2d(x, w, None, stride=1, padding=1)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv2d_f64(self):
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
        w = torch.randn(4, 3, 3, 3, dtype=torch.float64)
        got = run_conv("conv2d", x, w, None, stride=1, padding=1)
        ref = F.conv2d(x, w, None, stride=1, padding=1)
        assert got.dtype == torch.float64
        assert torch.allclose(got, ref, atol=1e-10)

    def test_conv2d_dilation(self):
        x = torch.randn(1, 2, 10, 10)
        w = torch.randn(4, 2, 3, 3)
        got = run_conv("conv2d", x, w, None, stride=1, padding=2, dilation=2)
        ref = F.conv2d(x, w, None, stride=1, padding=2, dilation=2)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv2d_groups(self):
        # depthwise separable (groups == channels)
        x = torch.randn(1, 4, 8, 8)
        w = torch.randn(4, 1, 3, 3)
        got = run_conv("conv2d", x, w, None, stride=1, padding=1, groups=4)
        ref = F.conv2d(x, w, None, stride=1, padding=1, groups=4)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv2d_groups_half(self):
        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(6, 2, 3, 3)
        got = run_conv("conv2d", x, w, None, stride=1, padding=1, groups=2)
        ref = F.conv2d(x, w, None, stride=1, padding=1, groups=2)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv2d_asymmetric_stride_pad(self):
        x = torch.randn(2, 3, 12, 14)
        w = torch.randn(6, 3, 3, 5)
        got = run_conv("conv2d", x, w, None, stride=[2, 3], padding=[1, 2])
        ref = F.conv2d(x, w, None, stride=(2, 3), padding=(1, 2))
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv2d_groups_mismatch_rejected(self):
        x = torch.randn(1, 4, 8, 8)
        w = torch.randn(4, 1, 3, 3)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_conv("conv2d", x, w, None, groups=3)  # 3 ∤ 4

    def test_conv2d_dtype_mismatch_rejected(self):
        x = torch.randn(1, 3, 8, 8, dtype=torch.float32)
        w = torch.randn(4, 3, 3, 3, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_conv("conv2d", x, w, None)


class TestConv1d:
    def test_conv1d(self):
        x = torch.randn(2, 4, 20)
        w = torch.randn(8, 4, 3)
        got = run_conv("conv1d", x, w, None, stride=1, padding=1)
        ref = F.conv1d(x, w, None, stride=1, padding=1)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv1d_stride_pad(self):
        x = torch.randn(2, 3, 16)
        w = torch.randn(6, 3, 5)
        b = torch.randn(6)
        got = run_conv("conv1d", x, w, b, stride=2, padding=3)
        ref = F.conv1d(x, w, b, stride=2, padding=3)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv1d_dilation_groups(self):
        x = torch.randn(1, 4, 12)
        w = torch.randn(4, 1, 3)
        got = run_conv("conv1d", x, w, None, stride=1, padding=2, dilation=2, groups=4)
        ref = F.conv1d(x, w, None, stride=1, padding=2, dilation=2, groups=4)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv1d_f64(self):
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        w = torch.randn(5, 3, 3, dtype=torch.float64)
        got = run_conv("conv1d", x, w, None, stride=1, padding=1)
        ref = F.conv1d(x, w, None, stride=1, padding=1)
        assert got.dtype == torch.float64
        assert torch.allclose(got, ref, atol=1e-10)


class TestConvTranspose1d:
    def test_conv_transpose1d_basic(self):
        x = torch.randn(2, 4, 16)
        w = torch.randn(4, 6, 3)
        b = torch.randn(6)
        got = run_conv("conv_transpose1d", x, w, b, stride=2, padding=1, output_padding=1)
        ref = F.conv_transpose1d(x, w, b, stride=2, padding=1, output_padding=1)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv_transpose1d_stride1_no_bias(self):
        x = torch.randn(1, 3, 8)
        w = torch.randn(3, 5, 3)
        got = run_conv("conv_transpose1d", x, w, None, stride=1, padding=1)
        ref = F.conv_transpose1d(x, w, None, stride=1, padding=1)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv_transpose1d_dilation_groups(self):
        x = torch.randn(1, 4, 12)
        w = torch.randn(4, 3, 3)
        got = run_conv("conv_transpose1d", x, w, None, stride=2, padding=2, dilation=2, groups=2)
        ref = F.conv_transpose1d(x, w, None, stride=2, padding=2, dilation=2, groups=2)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv_transpose1d_f64(self):
        x = torch.randn(1, 2, 10, dtype=torch.float64)
        w = torch.randn(2, 4, 3, dtype=torch.float64)
        got = run_conv("conv_transpose1d", x, w, None, stride=2, padding=1, output_padding=1)
        ref = F.conv_transpose1d(x, w, None, stride=2, padding=1, output_padding=1)
        assert torch.allclose(got, ref, atol=1e-10)


class TestConvTranspose2d:
    def test_conv_transpose2d_basic(self):
        x = torch.randn(1, 4, 8, 8)
        w = torch.randn(4, 6, 3, 3)
        b = torch.randn(6)
        got = run_conv("conv_transpose2d", x, w, b, stride=2, padding=1, output_padding=1)
        ref = F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv_transpose2d_stride1(self):
        x = torch.randn(2, 3, 6, 6)
        w = torch.randn(3, 5, 3, 3)
        got = run_conv("conv_transpose2d", x, w, None, stride=1, padding=1)
        ref = F.conv_transpose2d(x, w, None, stride=1, padding=1)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv_transpose2d_no_pad(self):
        x = torch.randn(1, 2, 4, 4)
        w = torch.randn(2, 3, 3, 3)
        got = run_conv("conv_transpose2d", x, w, None, stride=2, padding=0)
        ref = F.conv_transpose2d(x, w, None, stride=2, padding=0)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_conv_transpose2d_f64(self):
        x = torch.randn(1, 2, 5, 5, dtype=torch.float64)
        w = torch.randn(2, 4, 3, 3, dtype=torch.float64)
        got = run_conv("conv_transpose2d", x, w, None, stride=2, padding=1, output_padding=1)
        ref = F.conv_transpose2d(x, w, None, stride=2, padding=1, output_padding=1)
        assert torch.allclose(got, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Pooling
# ---------------------------------------------------------------------------

class TestMaxPool2d:
    @pytest.mark.parametrize("k", [2, 3])
    def test_max_pool2d(self, k):
        x = torch.randn(2, 4, 12, 12)
        got = run_pool("max_pool2d", x, kernel=k)
        ref = F.max_pool2d(x, k)
        assert torch.allclose(got, ref)

    def test_max_pool2d_stride_pad(self):
        x = torch.randn(2, 3, 10, 10)
        got = run_pool("max_pool2d", x, kernel=3, stride=2, padding=1)
        ref = F.max_pool2d(x, 3, stride=2, padding=1)
        assert torch.allclose(got, ref)

    def test_max_pool2d_dilation(self):
        x = torch.randn(1, 2, 8, 8)
        got = run_pool("max_pool2d", x, kernel=2, stride=1, dilation=2)
        ref = F.max_pool2d(x, 2, stride=1, dilation=2)
        assert torch.allclose(got, ref)

    def test_max_pool2d_f64(self):
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
        got = run_pool("max_pool2d", x, kernel=2)
        ref = F.max_pool2d(x, 2)
        assert got.dtype == torch.float64
        assert torch.allclose(got, ref)

    def test_max_pool2d_kernel_defaults_to_stride(self):
        # torch: stride defaults to kernel
        x = torch.randn(2, 3, 8, 8)
        got = run_pool("max_pool2d", x, kernel=2)
        ref = F.max_pool2d(x, 2)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref)


class TestAvgPool2d:
    def test_avg_pool2d(self):
        x = torch.randn(2, 4, 10, 10)
        got = run_pool("avg_pool2d", x, kernel=2)
        ref = F.avg_pool2d(x, 2)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_avg_pool2d_stride_pad(self):
        x = torch.randn(2, 3, 9, 9)
        got = run_pool("avg_pool2d", x, kernel=3, stride=2, padding=1)
        ref = F.avg_pool2d(x, 3, stride=2, padding=1)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_avg_pool2d_count_include_pad_false(self):
        x = torch.randn(1, 2, 5, 5)
        got = run_pool("avg_pool2d", x, kernel=3, stride=1, padding=1, count_include_pad=False)
        ref = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_avg_pool2d_f64(self):
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
        got = run_pool("avg_pool2d", x, kernel=2)
        ref = F.avg_pool2d(x, 2)
        assert got.dtype == torch.float64
        assert torch.allclose(got, ref, atol=1e-10)


class TestAdaptivePool:
    def test_adaptive_avg_pool2d_global(self):
        # Global average pooling — critical for ResNet classifiers
        x = torch.randn(2, 16, 7, 7)
        got = run_pool("adaptive_avg_pool2d", x, output_size=[1, 1])
        ref = F.adaptive_avg_pool2d(x, (1, 1))
        assert got.shape == (2, 16, 1, 1)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_adaptive_avg_pool2d_rect(self):
        x = torch.randn(2, 4, 6, 10)
        got = run_pool("adaptive_avg_pool2d", x, output_size=[2, 3])
        ref = F.adaptive_avg_pool2d(x, (2, 3))
        assert torch.allclose(got, ref, atol=1e-5)

    def test_adaptive_avg_pool2d_single_int(self):
        x = torch.randn(1, 3, 8, 8)
        got = run_pool("adaptive_avg_pool2d", x, output_size=3)
        ref = F.adaptive_avg_pool2d(x, 3)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_adaptive_max_pool2d(self):
        x = torch.randn(2, 8, 7, 7)
        got = run_pool("adaptive_max_pool2d", x, output_size=[1, 1])
        ref = F.adaptive_max_pool2d(x, (1, 1))
        assert torch.allclose(got, ref)

    def test_adaptive_max_pool2d_f64(self):
        x = torch.randn(1, 3, 6, 6, dtype=torch.float64)
        got = run_pool("adaptive_max_pool2d", x, output_size=[2, 2])
        ref = F.adaptive_max_pool2d(x, (2, 2))
        assert torch.allclose(got, ref)


class TestPool1d:
    def test_max_pool1d(self):
        x = torch.randn(2, 4, 20)
        got = run_pool("max_pool1d", x, kernel=2)
        ref = F.max_pool1d(x, 2)
        assert torch.allclose(got, ref)

    def test_max_pool1d_stride_pad(self):
        x = torch.randn(2, 3, 16)
        got = run_pool("max_pool1d", x, kernel=3, stride=2, padding=1)
        ref = F.max_pool1d(x, 3, stride=2, padding=1)
        assert torch.allclose(got, ref)

    def test_avg_pool1d(self):
        x = torch.randn(2, 4, 20)
        got = run_pool("avg_pool1d", x, kernel=2)
        ref = F.avg_pool1d(x, 2)
        assert torch.allclose(got, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Upsampling + flatten
# ---------------------------------------------------------------------------

class TestUpsample:
    def test_upsample_nearest2d(self):
        x = torch.randn(1, 3, 8, 8)
        got = run_pool("upsample_nearest2d", x, size=[16, 16])
        ref = F.interpolate(x, size=(16, 16), mode="nearest")
        assert got.shape == (1, 3, 16, 16)
        assert torch.allclose(got, ref)

    def test_upsample_nearest2d_downscale(self):
        x = torch.randn(1, 2, 10, 10)
        got = run_pool("upsample_nearest2d", x, size=[5, 5])
        ref = F.interpolate(x, size=(5, 5), mode="nearest")
        assert torch.allclose(got, ref)

    def test_interpolate_align_corners_true_rejected(self):
        # align_corners=True would silently produce wrong values; must fall back.
        x = torch.randn(1, 2, 8, 8)
        payload = json.dumps({
            "inputs": [_spec(x)],
            "nodes": [{"id": 0, "target": "interpolate",
                       "args": [{"kind": "slot", "index": 0}],
                       "kwargs": {"size": [16, 16], "mode": "bilinear",
                                   "align_corners": True}}],
            "outputs": [0],
        }, sort_keys=True)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            tb.execute(payload, [x.__dlpack__()])

    def test_upsample_bilinear2d(self):
        x = torch.randn(1, 3, 8, 8)
        got = run_pool("upsample_bilinear2d", x, size=[16, 16])
        ref = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_upsample_bilinear2d_f64(self):
        x = torch.randn(1, 2, 6, 6, dtype=torch.float64)
        got = run_pool("upsample_bilinear2d", x, size=[12, 12])
        ref = F.interpolate(x, size=(12, 12), mode="bilinear", align_corners=False)
        assert got.dtype == torch.float64
        assert torch.allclose(got, ref, atol=1e-10)


class TestFlatten:
    def test_flatten_full(self):
        x = torch.randn(2, 3, 4, 5)
        got = run_pool("flatten", x, start_dim=0, end_dim=-1)
        ref = torch.flatten(x, 0, -1)
        assert got.shape == (120,)
        assert torch.allclose(got, ref)

    def test_flatten_start_dim1(self):
        # The classic classifier pattern: flatten(1) after conv/pool
        x = torch.randn(2, 3, 4, 5)
        got = run_pool("flatten", x, start_dim=1, end_dim=-1)
        ref = torch.flatten(x, 1)
        assert got.shape == (2, 60)
        assert torch.allclose(got, ref)

    def test_flatten_middle_dims(self):
        x = torch.randn(2, 3, 4, 5, 6)
        got = run_pool("flatten", x, start_dim=1, end_dim=3)
        ref = torch.flatten(x, 1, 3)
        assert got.shape == (2, 60, 6)
        assert torch.allclose(got, ref)

    def test_flatten_invalid_dims_rejected(self):
        x = torch.randn(2, 3)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_pool("flatten", x, start_dim=2, end_dim=1)


# ---------------------------------------------------------------------------
# 4. End-to-end torch.compile: mini-CNN with no eager fallbacks
# ---------------------------------------------------------------------------

class TestEndToEndPhase3:
    def test_mini_cnn_via_compile(self):
        """Conv → ReLU → MaxPool → Flatten → Linear: every node native."""
        torch.manual_seed(1)
        conv_w = torch.randn(16, 3, 3, 3)
        conv_b = torch.randn(16)
        # 8x8 in -> conv(pad=1) 8x8 -> maxpool(2) 4x4 -> flatten 16*4*4 = 256
        fc_w = torch.randn(10, 16 * 4 * 4)
        fc_b = torch.randn(10)

        def model(x):
            h = torch.nn.functional.conv2d(x, conv_w, conv_b, stride=1, padding=1)
            h = torch.relu(h)
            h = torch.nn.functional.max_pool2d(h, 2)
            h = torch.flatten(h, 1)
            return torch.nn.functional.linear(h, fc_w, fc_b)

        compiled = torch.compile(model, backend="torchburn")
        x = torch.randn(2, 3, 8, 8)
        out = compiled(x)
        ref = model(x)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_resnet_style_block_via_compile(self):
        """conv → bn → relu → pool → adaptive pool → flatten → linear."""
        torch.manual_seed(2)
        conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        bn = torch.nn.BatchNorm2d(8)
        fc = torch.nn.Linear(8, 4)

        def model(x):
            h = conv(x)
            h = bn(h)
            h = torch.relu(h)
            h = torch.nn.functional.max_pool2d(h, 2)
            h = torch.nn.functional.adaptive_avg_pool2d(h, (1, 1))
            h = torch.flatten(h, 1)
            return fc(h)

        compiled = torch.compile(model, backend="torchburn")
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            out = compiled(x)
            ref = model(x)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_conv_no_fallback_warning(self):
        """Ensure conv graphs don't emit fallback warnings (all native)."""
        import warnings

        conv_w = torch.randn(4, 3, 3, 3)
        conv_b = torch.randn(4)

        def model(x):
            return torch.nn.functional.conv2d(x, conv_w, conv_b, padding=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compiled = torch.compile(model, backend="torchburn")
            compiled(torch.randn(1, 3, 8, 8))
            fallbacks = [m for m in w if "falling back to eager" in str(m.message)]
        assert len(fallbacks) == 0, f"unexpected fallbacks: {fallbacks}"

    def test_non_contiguous_input_falls_back_safely(self):
        """Strided conv input must fall back (TB_UNSUPPORTED), not crash."""
        base = torch.randn(2, 3, 10, 10)
        x = base[:, :, ::2, ::2]  # non-contiguous
        w = torch.randn(4, 3, 3, 3)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_conv("conv2d", x, w, None, stride=1, padding=1)


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestConvEdgeCases:
    def test_conv2d_kernel_larger_than_input(self):
        # torch rejects this too: padded input (4x4) < kernel (7x7) -> negative out
        x = torch.randn(1, 2, 4, 4)
        w = torch.randn(3, 2, 7, 7)
        with pytest.raises(RuntimeError):
            F.conv2d(x, w, None, stride=1, padding=0)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_conv("conv2d", x, w, None, stride=1, padding=0)

    def test_conv2d_zero_padding_no_negative_size(self):
        # 4x4 input, 5x5 kernel, no padding → output would be 0 → must raise
        x = torch.randn(1, 1, 4, 4)
        w = torch.randn(2, 1, 5, 5)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_conv("conv2d", x, w, None, stride=1, padding=0)

    def test_max_pool2d_kernel_too_big_falls_back(self):
        x = torch.randn(1, 2, 3, 3)
        with pytest.raises(RuntimeError, match="TB_UNSUPPORTED"):
            run_pool("max_pool2d", x, kernel=4)

    def test_batch_size_gt1_conv(self):
        x = torch.randn(4, 3, 8, 8)
        w = torch.randn(5, 3, 3, 3)
        b = torch.randn(5)
        got = run_conv("conv2d", x, w, b, stride=1, padding=1)
        ref = F.conv2d(x, w, b, stride=1, padding=1)
        assert torch.allclose(got, ref, atol=1e-4)

    def test_channels_gt1_conv(self):
        x = torch.randn(1, 12, 8, 8)
        w = torch.randn(20, 12, 3, 3)
        got = run_conv("conv2d", x, w, None, stride=1, padding=1)
        ref = F.conv2d(x, w, None, stride=1, padding=1)
        assert torch.allclose(got, ref, atol=1e-4)
