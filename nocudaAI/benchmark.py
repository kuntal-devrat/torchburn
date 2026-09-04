"""Comprehensive CPU Inference Benchmark Suite for NoCudaAI.

Directly compares PyTorch Eager (CPU) vs TorchBurn Native CPU Engine across:
- Prefill Latency & Throughput (Prompt Processing)
- Step-by-Step Decoding Latency (Autoregressive Token Generation)
- Sustained Tokens/sec (End-to-End Throughput)
"""

from __future__ import annotations
import gc
import sys
import time
from typing import Dict, Any, List
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .model import NoCudaModel
from .tokenizer import NoCudaTokenizer
from .config import ModelConfig, MODEL_PROFILES, GenerationConfig, EngineConfig
from .tokenizer import NoCudaTokenizer, get_tokenizer
from .engine import NoCudaEngine


def run_benchmark(
    profile_name: str = "micro",
    prompt: str = "The future of CPU-native artificial intelligence with TorchBurn acceleration is",
    gen_tokens: int = 32,
    warmup: int = 2,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Executes a comparative benchmark between PyTorch Eager and TorchBurn."""
    is_pretrained = any(q in profile_name.lower() for q in ("qwen", "0.5b", "0_5b")) or (
        isinstance(profile_name, str) and (profile_name.endswith(".safetensors") or "/" in profile_name)
    )

    if is_pretrained:
        base_model = NoCudaModel.from_pretrained(profile_name)
        tokenizer = get_tokenizer(profile_name)
        config = base_model.config
    else:
        config = MODEL_PROFILES.get(profile_name, MODEL_PROFILES["micro"])
        tokenizer = get_tokenizer(profile_name, vocab_size=config.vocab_size)
        torch.manual_seed(42)
        base_model = NoCudaModel(config).eval()

    print("\n" + "=" * 75)
    print(f"  NoCudaAI / TorchBurn CPU Benchmark: Profile '{profile_name}'")
    print(f"  Architecture: {config.num_hidden_layers} Layers, {config.hidden_size} Hidden Dim, {config.num_attention_heads} Heads")
    print(f"  Context: Generating {gen_tokens} tokens | Threads: {torch.get_num_threads()}")
    print("=" * 75 + "\n")

    num_params = base_model.get_num_params()
    print(f"  Total Parameters: {num_params / 1e6:.2f}M\n")

    results = {}

    # 1. Benchmark PyTorch Eager
    print("▶ Testing [PyTorch Eager (CPU)]...")
    eager_engine = NoCudaEngine(base_model, tokenizer, EngineConfig(engine="eager"))
    
    # Warmup
    for _ in range(warmup):
        eager_engine.generate(prompt, GenerationConfig(max_new_tokens=8, stream=False))

    eager_prefill_times = []
    eager_decode_times = []
    eager_tokens_sec = []

    for run_idx in range(num_runs):
        gc.collect()
        summary = None
        prefill_ms = 0.0
        for pkt in eager_engine.generate_stream(prompt, GenerationConfig(max_new_tokens=gen_tokens, stream=False)):
            if pkt["type"] == "prefill":
                prefill_ms = pkt["prefill_time_ms"]
            elif pkt["type"] == "summary":
                summary = pkt
        if summary:
            eager_prefill_times.append(prefill_ms)
            eager_decode_times.append(summary["avg_ms_per_token"])
            eager_tokens_sec.append(summary["decode_tok_sec"])

    results["eager"] = {
        "prefill_ms": sum(eager_prefill_times) / len(eager_prefill_times),
        "decode_ms_per_tok": sum(eager_decode_times) / len(eager_decode_times),
        "tok_per_sec": sum(eager_tokens_sec) / len(eager_tokens_sec),
    }

    # 2. Benchmark TorchBurn Native CPU
    print("[*] Testing [TorchBurn Native CPU (SIMD AVX2)]...")
    cpu_engine = NoCudaEngine(base_model, tokenizer, EngineConfig(engine="cpu", torchburn_engine="native_cpu"))

    for _ in range(warmup):
        cpu_engine.generate(prompt, GenerationConfig(max_new_tokens=8, stream=False))

    cpu_prefill_times, cpu_decode_times, cpu_tokens_sec = [], [], []
    for run_idx in range(num_runs):
        gc.collect()
        summary, prefill_ms = None, 0.0
        for pkt in cpu_engine.generate_stream(prompt, GenerationConfig(max_new_tokens=gen_tokens, stream=False)):
            if pkt["type"] == "prefill":
                prefill_ms = pkt["prefill_time_ms"]
            elif pkt["type"] == "summary":
                summary = pkt
        if summary:
            cpu_prefill_times.append(prefill_ms)
            cpu_decode_times.append(summary["avg_ms_per_token"])
            cpu_tokens_sec.append(summary["decode_tok_sec"])

    results["cpu"] = {
        "prefill_ms": sum(cpu_prefill_times) / max(len(cpu_prefill_times), 1),
        "decode_ms_per_tok": sum(cpu_decode_times) / max(len(cpu_decode_times), 1),
        "tok_per_sec": sum(cpu_tokens_sec) / max(len(cpu_tokens_sec), 1),
    }

    # 3. Benchmark TorchBurn iGPU (Vulkan Shaders)
    has_igpu = False
    try:
        from torchburn import _torchburn as _native
        has_igpu = _native.gpu_available()
    except Exception:
        pass

    if has_igpu:
        print("[*] Testing [TorchBurn iGPU (Vulkan / burn-wgpu)]...")
        igpu_engine = NoCudaEngine(base_model, tokenizer, EngineConfig(engine="igpu", torchburn_engine="burn-wgpu"))

        for _ in range(warmup):
            igpu_engine.generate(prompt, GenerationConfig(max_new_tokens=8, stream=False))

        igpu_prefill_times, igpu_decode_times, igpu_tokens_sec = [], [], []
        for run_idx in range(num_runs):
            gc.collect()
            summary, prefill_ms = None, 0.0
            for pkt in igpu_engine.generate_stream(prompt, GenerationConfig(max_new_tokens=gen_tokens, stream=False)):
                if pkt["type"] == "prefill":
                    prefill_ms = pkt["prefill_time_ms"]
                elif pkt["type"] == "summary":
                    summary = pkt
            if summary:
                igpu_prefill_times.append(prefill_ms)
                igpu_decode_times.append(summary["avg_ms_per_token"])
                igpu_tokens_sec.append(summary["decode_tok_sec"])

        results["igpu"] = {
            "prefill_ms": sum(igpu_prefill_times) / max(len(igpu_prefill_times), 1),
            "decode_ms_per_tok": sum(igpu_decode_times) / max(len(igpu_decode_times), 1),
            "tok_per_sec": sum(igpu_tokens_sec) / max(len(igpu_tokens_sec), 1),
        }

    # Print Comparative Results
    eg = results["eager"]
    tb_cpu = results["cpu"]

    print("\n" + "=" * 80)
    print(f"{'Execution Mode':<28} | {'Prefill (ms)':<15} | {'Decode (ms/tok)':<16} | {'Throughput':<14}")
    print("-" * 80)
    print(f"{'PyTorch Eager (CPU)':<28} | {eg['prefill_ms']:>10.2f} ms   | {eg['decode_ms_per_tok']:>11.2f} ms   | {eg['tok_per_sec']:>9.1f} t/s")
    print(f"{'TorchBurn CPU (AVX2)':<28} | {tb_cpu['prefill_ms']:>10.2f} ms   | {tb_cpu['decode_ms_per_tok']:>11.2f} ms   | {tb_cpu['tok_per_sec']:>9.1f} t/s")
    if "igpu" in results:
        tb_igpu = results["igpu"]
        print(f"{'TorchBurn iGPU (Vulkan)':<28} | {tb_igpu['prefill_ms']:>10.2f} ms   | {tb_igpu['decode_ms_per_tok']:>11.2f} ms   | {tb_igpu['tok_per_sec']:>9.1f} t/s")
    print("=" * 80)

    cpu_speedup = eg["decode_ms_per_tok"] / max(tb_cpu["decode_ms_per_tok"], 1e-6)
    print(f"  [+] TorchBurn CPU Speedup vs Eager: {cpu_speedup:.2f}x")
    if "igpu" in results:
        print(f"  [+] iGPU Hardware Accelerated via Vulkan / WGPU compute shaders (No CUDA / No llama.cpp)")
    print("-" * 80 + "\n")

    return results
