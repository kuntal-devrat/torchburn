#!/usr/bin/env python3
"""Fusion benchmark: fused vs unfused on MLP, Transformer, Conv blocks."""

import os, sys, time, statistics, warnings
import torch, torch.nn as nn, torch.nn.functional as F

warnings.filterwarnings("ignore")
import torchburn  # noqa

WARMUP = 2
ITERS = 8

def median_us(fn, n=ITERS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)

class MLPBlock(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_out)
    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))

class MLPDeep(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])
    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d=128, h=4, dff=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.out_proj = nn.Linear(d, d)
        self.n_heads, self.d_head = h, d//h
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))
    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) * (self.d_head ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, T, D)
        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x

class ConvBnRelu(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(oc)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

def main():
    print("=" * 90)
    print("TORCHBURN FUSION BENCHMARK")
    print("=" * 90)
    torch.manual_seed(42)

    # Section 1: MLP (GEMM epilogue fusion)
    print("\n--- MLP BLOCKS (linear->relu GEMM epilogue fusion) ---")
    print("  %-36s %10s %10s  %8s" % ("Model", "Eager", "TorchBurn", "ratio"))
    for label, model, shape in [
        ("MLP 256->512->256", MLPBlock(256,512,256), (1,256)),
        ("MLP 512->1024->512", MLPBlock(512,1024,512), (1,512)),
        ("MLP 4-layer 256", MLPDeep(256), (1,256)),
        ("MLP batch=32 512", MLPBlock(512,1024,512), (32,512)),
    ]:
        model.eval()
        x = torch.randn(*shape)
        compiled = torch.compile(model, backend="torchburn")
        with torch.no_grad():
            try:
                compiled(x)
            except Exception as e:
                print("  SKIP %-36s %s" % (label, str(e)[:50]))
                continue
            e_us = median_us(lambda: model(x))
            c_us = median_us(lambda: compiled(x))
            print("  %-36s %10.1f %10.1f  %6.2fx" % (label, e_us, c_us, e_us/c_us))

    # Section 2: Transformer
    print("\n--- TRANSFORMER (elementwise + GEMM fusion) ---")
    print("  %-36s %10s %10s  %8s" % ("Model", "Eager", "TorchBurn", "ratio"))
    for label, model, shape in [
        ("Transformer d=128", TransformerBlock(128,4,512), (1,16,128)),
        ("Transformer d=256", TransformerBlock(256,4,1024), (1,16,256)),
    ]:
        model.eval()
        x = torch.randn(*shape)
        compiled = torch.compile(model, backend="torchburn")
        with torch.no_grad():
            try:
                compiled(x)
            except Exception as e:
                print("  SKIP %-36s %s" % (label, str(e)[:50]))
                continue
            e_us = median_us(lambda: model(x))
            c_us = median_us(lambda: compiled(x))
            print("  %-36s %10.1f %10.1f  %6.2fx" % (label, e_us, c_us, e_us/c_us))

    # Section 3: Conv blocks
    print("\n--- CONV BLOCKS (conv->bn->relu) ---")
    print("  %-36s %10s %10s  %8s" % ("Model", "Eager", "TorchBurn", "ratio"))
    for label, model, shape in [
        ("ConvBnRelu 3->16", ConvBnRelu(3,16), (1,3,32,32)),
        ("ConvBnRelu 3->64", ConvBnRelu(3,64), (1,3,64,64)),
    ]:
        model.eval()
        x = torch.randn(*shape)
        compiled = torch.compile(model, backend="torchburn")
        with torch.no_grad():
            try:
                compiled(x)
            except Exception as e:
                print("  SKIP %-36s %s" % (label, str(e)[:50]))
                continue
            e_us = median_us(lambda: model(x))
            c_us = median_us(lambda: compiled(x))
            print("  %-36s %10.1f %10.1f  %6.2fx" % (label, e_us, c_us, e_us/c_us))

    # Section 4: Fused vs Unfused
    print("\n--- FUSION IMPACT (fused vs TORCHBURN_NO_FUSION=1) ---")
    print("  %-36s %10s %10s %10s  %8s" % ("Model", "Fused", "Unfused", "Fusion", "speedup"))
    for label, model, shape in [
        ("MLP 4-layer 256", MLPDeep(256), (1,256)),
        ("MLP 4-layer 512", MLPDeep(512), (1,512)),
        ("Transformer d=128", TransformerBlock(128,4,512), (1,16,128)),
    ]:
        model.eval()
        x = torch.randn(*shape)

        # Fused
        torch._dynamo.reset()
        fused = torch.compile(model, backend="torchburn")
        with torch.no_grad():
            try:
                fused(x)
            except:
                print("  SKIP %-36s (fused failed)" % label)
                continue

        # Unfused
        os.environ["TORCHBURN_NO_FUSION"] = "1"
        torch._dynamo.reset()
        unfused_model = None
        try:
            unfused_model = torch.compile(model, backend="torchburn")
            with torch.no_grad():
                unfused_model(x)
        except Exception as e:
            print("  SKIP %-36s (unfused compile: %s)" % (label, str(e)[:50]))
            os.environ.pop("TORCHBURN_NO_FUSION", None)
            continue
        os.environ.pop("TORCHBURN_NO_FUSION", None)

        _fused = fused
        _unfused = unfused_model
        def _run_fused(): return _fused(x)
        def _run_unfused(): return _unfused(x)
        try:
            f_us = median_us(_run_fused)
            u_us = median_us(_run_unfused)
        except Exception as e:
            print("  SKIP %-36s (bench error: %s)" % (label, str(e)[:50]))
            continue
        ratio = u_us / f_us if f_us > 0 else 0
        print("  %-36s %10.1f %10.1f %10.1f  %6.2fx" % (label, f_us, u_us, ratio, ratio))

    print("\n" + "=" * 90)
    print("Fusion = unfused/fused (>1.0 = fusion helps)")
    print("=" * 90)

if __name__ == "__main__":
    main()
