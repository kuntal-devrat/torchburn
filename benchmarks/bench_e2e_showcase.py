#!/usr/bin/env python3
"""TorchBurn Comprehensive Production Showcase & Latency Benchmark.

Measures:
1. First-run compilation latency vs steady-state execution (BLAKE3 cache).
2. Graph visualizer diagnostic overhead (torchburn.visualize).
3. Memory pool recycling efficiency.
4. Single-pass LLM token decoder throughput (CPU SIMD vs WGPU GPU).
"""

import os
import sys
import time
import statistics
import torch
import torch.nn as nn
import torchburn as tb

def banner(text: str):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def main():
    banner("TORCHBURN PRODUCTION PERFORMANCE SHOWCASE")
    gpu_info = tb.gpu_info()
    print(f"  Hardware Adapter:   {gpu_info.get('adapter_name', 'CPU')}")
    print(f"  Graphics Backend:   {gpu_info.get('backend', 'None')}")
    print(f"  GPU Available:      {gpu_info.get('available', False)}")
    print(f"  Rayon CPU Threads:  {tb.rayon_threads()}")
    print(f"  PyTorch Version:    {torch.__version__}")
    print(f"  TorchBurn Version:  {tb.__version__}")

    # -------------------------------------------------------------------------
    # 1. Compilation Latency & Caching Benchmark
    # -------------------------------------------------------------------------
    banner("1. COMPILATION LATENCY & STEADY-STATE REPLAY")
    model = nn.Sequential(
        nn.Linear(256, 1024),
        nn.GELU(),
        nn.Linear(1024, 256),
        nn.LayerNorm(256),
    )
    x = torch.randn(16, 256)

    # Warmup eager
    _ = model(x)

    t0 = time.perf_counter()
    compiled = torch.compile(model, backend="torchburn")
    # First call: traces + parses + compiles + executes
    _ = compiled(x)
    first_call_ms = (time.perf_counter() - t0) * 1000.0

    # Steady state (cached BLAKE3 lookup)
    steady_times = []
    for _ in range(50):
        t_start = time.perf_counter()
        _ = compiled(x)
        steady_times.append((time.perf_counter() - t_start) * 1000.0)
    
    steady_median_ms = statistics.median(steady_times)
    stats = tb.cache_stats()

    print(f"  First Forward Pass (Trace + Compile + Execute):  {first_call_ms:8.2f} ms")
    print(f"  Steady-State Replay (BLAKE3 Hash Hit):           {steady_median_ms:8.3f} ms")
    print(f"  Speedup vs First Call:                            {first_call_ms / steady_median_ms:8.1f}x")
    print(f"  Graph Cache Hits:                                {stats.get('hits', 0)}")
    print(f"  Graph Cache Size:                                {stats.get('size', 0)}")

    # -------------------------------------------------------------------------
    # 2. Graph Inspector & Visualizer Benchmark
    # -------------------------------------------------------------------------
    banner("2. COMPILER OBSERVABILITY: GRAPH VISUALIZER")
    t0 = time.perf_counter()
    viz = tb.visualize(model, x, print_summary=False)
    viz_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Visualizer Inspection Latency:                   {viz_ms:8.2f} ms")
    print(f"  Native Fast-Path Coverage:                       {viz.native_ratio * 100.0:.1f}% ({len(viz.native_nodes)}/{viz.total_compute_nodes} nodes)")
    print(f"  Mermaid Diagram Size:                            {len(viz.to_mermaid())} bytes")
    print(f"  Interactive HTML Dashboard Size:                 {len(viz.to_html())} bytes")

    # -------------------------------------------------------------------------
    # 3. Buffer Pool Zero-Overhead Recycling
    # -------------------------------------------------------------------------
    banner("3. MEMORY BUFFER POOL DIAGNOSTICS")
    pool_stats = tb.memory_pool_stats()
    print(f"  Buffer Pool Allocations:                         {pool_stats.get('alloc_count', 0)}")
    print(f"  Buffer Pool Hits:                                {pool_stats.get('hit_count', 0)}")
    print(f"  Buffer Hit Rate:                                 {pool_stats.get('hit_rate', 0.0) * 100.0:.1f}%")
    print(f"  Recycled Tensor Buffers:                         {pool_stats.get('cached_buffers', 0)}")

    # -------------------------------------------------------------------------
    # 4. LLM Single-Pass Decode Step Benchmark (if checkpoint available)
    # -------------------------------------------------------------------------
    banner("4. SINGLE-PASS LLM TOKEN GENERATION BENCHMARK")
    model_path = os.environ.get("TORCHBURN_TEST_MODEL", r"d:\torchburn\models\qwen_0_5b")
    if os.path.isdir(model_path):
        from torchburn.llm.loader import ModelLoader
        from torchburn.quantization import create_rust_qwen_decoder, create_wgpu_qwen_decoder

        print(f"  Loading INT4 model: {model_path}...")
        qmodel, _cfg, _root = ModelLoader.load(model_path, quant="int4", device="cpu")

        # CPU SIMD Benchmark
        print("\n  [CPU Single-Pass AVX/SIMD Decoder]")
        cpu_dec = create_rust_qwen_decoder(qmodel, max_seq_len=1024)
        # Warmup
        tok = 100
        # Warmup
        tok = 100
        for i in range(10):
            tok = cpu_dec.decode_and_sample(tok, i, 0.0, 1, 1.0, None)
        
        n_steps = 64
        t0 = time.perf_counter()
        for i in range(10, 10 + n_steps):
            tok = cpu_dec.decode_and_sample(tok, i, 0.0, 1, 1.0, None)
        cpu_elapsed = time.perf_counter() - t0
        cpu_tok_s = n_steps / cpu_elapsed
        cpu_ms_per_tok = (cpu_elapsed / n_steps) * 1000.0
        print(f"    Latency per token:                             {cpu_ms_per_tok:8.2f} ms")
        print(f"    Throughput:                                    {cpu_tok_s:8.2f} tokens/sec")

        # WGPU GPU Benchmark
        if tb.gpu_available():
            print(f"\n  [WGPU Universal GPU Compute Shader Decoder ({gpu_info.get('backend')})]")
            gpu_dec = create_wgpu_qwen_decoder(qmodel, max_seq_len=1024)
            tok = 100
            for i in range(10):
                tok = gpu_dec.decode_and_sample(tok, i, 0.0, 1, 1.0, None)
            
            t0 = time.perf_counter()
            for i in range(10, 10 + n_steps):
                tok = gpu_dec.decode_and_sample(tok, i, 0.0, 1, 1.0, None)
            gpu_elapsed = time.perf_counter() - t0
            gpu_tok_s = n_steps / gpu_elapsed
            gpu_ms_per_tok = (gpu_elapsed / n_steps) * 1000.0
            print(f"    Latency per token:                             {gpu_ms_per_tok:8.2f} ms")
            print(f"    Throughput:                                    {gpu_tok_s:8.2f} tokens/sec")
            print(f"    GPU Speedup vs CPU:                            {gpu_tok_s / cpu_tok_s:8.2f}x")
    else:
        print(f"  Notice: Local INT4 checkpoint at '{model_path}' not found; skipping LLM decode steps.")

    banner("BENCHMARK COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
