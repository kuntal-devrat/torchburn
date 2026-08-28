"""
Tests for Phase 13: Model-Level Validation.

Tests ResNet-18 and BERT-Tiny correctness through torch.compile with torchburn.
Verifies output matches PyTorch eager within atol=1e-3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torchburn


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestResNet18:
    @pytest.fixture(autouse=True)
    def _reset_dynamo(self):
        torch._dynamo.reset()
    def test_resnet18_inference(self):
        """ResNet-18 inference matches PyTorch eager."""
        model = ResNet18(num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            ref = model(x)
            compiled = torch.compile(model, backend="torchburn")
            out = compiled(x)

        assert torch.allclose(ref, out, atol=1e-3), \
            f"max diff: {(ref - out).abs().max().item()}"

    def test_resnet18_batch(self):
        """ResNet-18 with batch size > 1."""
        model = ResNet18(num_classes=10)
        model.eval()
        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            ref = model(x)
            compiled = torch.compile(model, backend="torchburn")
            out = compiled(x)

        assert torch.allclose(ref, out, atol=1e-3), \
            f"max diff: {(ref - out).abs().max().item()}"

    def test_resnet18_fallback_coverage(self):
        """Check what percentage of ops are native."""
        model = ResNet18(num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 224, 224)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compiled = torch.compile(model, backend="torchburn")
            with torch.no_grad():
                _ = compiled(x)

        fallback_warnings = [x for x in w if "falling back" in str(x.message)]
        print(f"\nResNet-18 fallback count: {len(fallback_warnings)}")
        for fw in fallback_warnings:
            print(f"  {fw.message}")


# ---------------------------------------------------------------------------
# BERT-Tiny
# ---------------------------------------------------------------------------

class BertTiny(nn.Module):
    def __init__(self, vocab=30522, hidden=128, num_heads=2, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.pos_embed = nn.Embedding(512, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 4, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.transformer(x)
        x = x[:, 0]
        return self.classifier(x)


class TestBertTiny:
    @pytest.fixture(autouse=True)
    def _reset_dynamo(self):
        torch._dynamo.reset()
    def test_bert_tiny_inference(self):
        """BERT-Tiny inference matches PyTorch eager."""
        model = BertTiny()
        model.eval()
        input_ids = torch.randint(0, 30522, (2, 128))

        with torch.no_grad():
            ref = model(input_ids)
            compiled = torch.compile(model, backend="torchburn")
            out = compiled(input_ids)

        assert torch.allclose(ref, out, atol=1e-3), \
            f"max diff: {(ref - out).abs().max().item()}"

    def test_bert_tiny_single(self):
        """BERT-Tiny with batch size 1."""
        model = BertTiny()
        model.eval()
        input_ids = torch.randint(0, 30522, (1, 64))

        with torch.no_grad():
            ref = model(input_ids)
            compiled = torch.compile(model, backend="torchburn")
            out = compiled(input_ids)

        assert torch.allclose(ref, out, atol=1e-3), \
            f"max diff: {(ref - out).abs().max().item()}"

    def test_bert_tiny_fallback_coverage(self):
        """Check what percentage of ops are native."""
        model = BertTiny()
        model.eval()
        input_ids = torch.randint(0, 30522, (2, 128))

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compiled = torch.compile(model, backend="torchburn")
            with torch.no_grad():
                _ = compiled(input_ids)

        fallback_warnings = [x for x in w if "falling back" in str(x.message)]
        print(f"\nBERT-Tiny fallback count: {len(fallback_warnings)}")
        for fw in fallback_warnings:
            print(f"  {fw.message}")


# ---------------------------------------------------------------------------
# Small Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        h = self.ff(h)
        return x + h


class TestTransformerBlock:
    @pytest.fixture(autouse=True)
    def _reset_dynamo(self):
        torch._dynamo.reset()
    def test_transformer_block(self):
        """Transformer block inference matches PyTorch eager."""
        model = TransformerBlock(d_model=128, n_heads=4)
        model.eval()
        x = torch.randn(2, 16, 128)

        with torch.no_grad():
            ref = model(x)
            compiled = torch.compile(model, backend="torchburn")
            out = compiled(x)

        assert torch.allclose(ref, out, atol=1e-3), \
            f"max diff: {(ref - out).abs().max().item()}"


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    @pytest.fixture(autouse=True)
    def _reset_dynamo(self):
        torch._dynamo.reset()
    def test_benchmark_vision(self):
        """Benchmark ResNet-18 inference."""
        model = ResNet18(num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            # Warmup
            compiled = torch.compile(model, backend="torchburn")
            for _ in range(1):
                _ = compiled(x)

            # Benchmark (small N to stay within timeout)
            start = time.perf_counter()
            N = 3
            for _ in range(N):
                _ = compiled(x)
            elapsed = time.perf_counter() - start

            # Also time eager
            start_eager = time.perf_counter()
            for _ in range(N):
                _ = model(x)
            elapsed_eager = time.perf_counter() - start_eager

            ms_per_iter = elapsed / N * 1000
            ms_per_iter_eager = elapsed_eager / N * 1000
            print(f"\nResNet-18 benchmark:")
            print(f"  torchburn: {ms_per_iter:.1f} ms/iter")
            print(f"  eager:     {ms_per_iter_eager:.1f} ms/iter")
            print(f"  ratio:     {ms_per_iter / ms_per_iter_eager:.2f}x")

    def test_benchmark_nlp(self):
        """Benchmark BERT-Tiny inference."""
        model = BertTiny()
        model.eval()
        input_ids = torch.randint(0, 30522, (2, 128))

        with torch.no_grad():
            # Warmup
            compiled = torch.compile(model, backend="torchburn")
            for _ in range(1):
                _ = compiled(input_ids)

            # Benchmark (small N to stay within timeout)
            start = time.perf_counter()
            N = 3
            for _ in range(N):
                _ = compiled(input_ids)
            elapsed = time.perf_counter() - start

            # Also time eager
            start_eager = time.perf_counter()
            for _ in range(N):
                _ = model(input_ids)
            elapsed_eager = time.perf_counter() - start_eager

            ms_per_iter = elapsed / N * 1000
            ms_per_iter_eager = elapsed_eager / N * 1000
            print(f"\nBERT-Tiny benchmark:")
            print(f"  torchburn: {ms_per_iter:.1f} ms/iter")
            print(f"  eager:     {ms_per_iter_eager:.1f} ms/iter")
            print(f"  ratio:     {ms_per_iter / ms_per_iter_eager:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
