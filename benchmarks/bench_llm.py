#!/usr/bin/env python3
"""End-to-end LLM decode benchmark (Phase 0.1).

Measures real decode tok/s for the CPU Rust decoder and (optionally) the
iGPU wgpu decoder on a real model, saves a JSON result with a hardware
fingerprint to `benchmarks/results/{date}_{hostname}.json`, and can compare
against a baseline file for regression gating.

Usage:
    python benchmarks/bench_llm.py --model models/qwen_0_5b --device auto
    python benchmarks/bench_llm.py --model models/qwen_0_5b --device auto --runs 3 --tokens 64
    python benchmarks/bench_llm.py --compare benchmarks/results/2026-09-06_host.json

The reproducibility gate is: median tok/s must be within 10% across `--runs`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

import torchburn


def hardware_fingerprint() -> Dict[str, Any]:
    """CPU model, adapter name, backend, driver-ish info, and versions."""
    gpu = {}
    try:
        gpu = torchburn.gpu_info()
    except Exception:
        gpu = {"available": False, "adapter_name": "n/a", "backend": "n/a"}
    dispatch = {}
    try:
        dispatch = torchburn._torchburn.cpu_features_report()
    except Exception:
        dispatch = {"tier": "unknown"}
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "n/a",
        "cpu_count_logical": os.cpu_count() or 0,
        "rayon_threads": torchburn.rayon_threads(),
        "adapter_name": gpu.get("adapter_name", "n/a"),
        "gpu_backend": gpu.get("backend", "n/a"),
        "gpu_available": bool(gpu.get("available", False)),
        "torch_version": torch.__version__,
        "torchburn_version": torchburn.__version__,
        "cpu_tier": dispatch.get("tier", "unknown"),
        "num_threads": torch.get_num_threads(),
    }


def _warmup_and_time(decoder, n_warmup: int, n_tokens: int) -> float:
    """Returns per-token seconds for a decode loop on an already-quantized decoder."""
    tok = 100
    for i in range(n_warmup):
        tok = decoder.decode_and_sample(tok, i, 0.0, 1, 1.0, None, 1.0)
    t0 = time.perf_counter()
    for i in range(n_warmup, n_warmup + n_tokens):
        tok = decoder.decode_and_sample(tok, i, 0.0, 1, 1.0, None, 1.0)
    return (time.perf_counter() - t0) / n_tokens


def bench_cpu_decoder(qmodel, tokens: int, warmup: int, runs: int) -> Dict[str, Any]:
    from torchburn.quantization import create_rust_qwen_decoder

    decoder = create_rust_qwen_decoder(qmodel, max_seq_len=1024)
    per_token = [_warmup_and_time(decoder, warmup, tokens) for _ in range(runs)]
    med = statistics.median(per_token)
    return {
        "cpu_tok_per_s": 1.0 / med if med > 0 else 0.0,
        "cpu_ms_per_token": med * 1000.0,
        "cpu_runs_ms": [t * 1000.0 for t in per_token],
    }


def bench_igpu_decoder(qmodel, tokens: int, warmup: int, runs: int) -> Dict[str, Any]:
    from torchburn.quantization import create_wgpu_qwen_decoder

    decoder = create_wgpu_qwen_decoder(qmodel, max_seq_len=1024)
    # warm the pipelines / JIT shaders once
    decoder.step(0, 0)
    decoder.reset_kv_cache()
    per_token = [_warmup_and_time(decoder, warmup, tokens) for _ in range(runs)]
    med = statistics.median(per_token)
    return {
        "igpu_tok_per_s": 1.0 / med if med > 0 else 0.0,
        "igpu_ms_per_token": med * 1000.0,
        "igpu_runs_ms": [t * 1000.0 for t in per_token],
    }


def load_quantized_model(model_path: str):
    from torchburn.llm.loader import ModelLoader

    qmodel, _cfg, _root = ModelLoader.load(model_path, quant="int4", device="cpu")
    return qmodel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("TORCHBURN_TEST_MODEL", "models/qwen_0_5b"))
    parser.add_argument("--device", choices=["auto", "cpu", "igpu"], default="auto")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3, help="repeats for the 10% reproducibility gate")
    parser.add_argument("--out-dir", default="benchmarks/results")
    parser.add_argument(
        "--compare",
        metavar="BASELINE.json",
        help="compare against a baseline result file; exit 1 if any tok/s regressed >10%%",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"[bench_llm] model dir not found: {args.model}", file=sys.stderr)
        return 2

    fp = hardware_fingerprint()
    print(f"[bench_llm] {fp['platform']} / {fp['machine']}")
    print(f"[bench_llm] adapter={fp['adapter_name']} backend={fp['gpu_backend']} tier={fp['cpu_tier']}")
    print(f"[bench_llm] loading + quantizing int4: {args.model}")
    t0 = time.perf_counter()
    qmodel = load_quantized_model(args.model)
    print(f"[bench_llm] model ready in {time.perf_counter() - t0:.1f}s")

    result: Dict[str, Any] = {"fingerprint": fp, "args": vars(args)}

    if args.device in ("auto", "cpu"):
        try:
            result["cpu"] = bench_cpu_decoder(qmodel, args.tokens, args.warmup, args.runs)
            print(f"[bench_llm] CPU decode: {result['cpu']['cpu_tok_per_s']:.1f} tok/s "
                  f"({result['cpu']['cpu_ms_per_token']:.2f} ms/tok, runs={args.runs})")
        except Exception as exc:
            print(f"[bench_llm] CPU bench failed: {exc}", file=sys.stderr)
            result["cpu"] = {"error": str(exc)}

    if args.device in ("auto", "igpu") and fp["gpu_available"]:
        try:
            result["igpu"] = bench_igpu_decoder(qmodel, args.tokens, args.warmup, args.runs)
            print(f"[bench_llm] iGPU decode: {result['igpu']['igpu_tok_per_s']:.1f} tok/s "
                  f"({result['igpu']['igpu_ms_per_token']:.2f} ms/tok)")
        except Exception as exc:
            print(f"[bench_llm] iGPU bench failed: {exc}", file=sys.stderr)
            result["igpu"] = {"error": str(exc)}
    elif args.device == "igpu" and not fp["gpu_available"]:
        print("[bench_llm] no GPU available; skipping iGPU", file=sys.stderr)

    # Reproducibility gate: median within 10% across runs.
    for backend in ("cpu", "igpu"):
        entry = result.get(backend, {})
        runs_ms = entry.get(f"{backend}_runs_ms")
        if runs_ms and len(runs_ms) > 1:
            spread = (max(runs_ms) - min(runs_ms)) / statistics.median(runs_ms)
            if spread > 0.10:
                print(f"[bench_llm] WARNING: {backend} spread {spread:.1%} > 10% "
                      f"(runs: {[f'{t:.2f}' for t in runs_ms]})", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    date = datetime.date.today().isoformat()
    host = platform.node() or "unknown"
    out_path = Path(args.out_dir) / f"{date}_{host}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[bench_llm] wrote {out_path}")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        return compare_results(baseline, result)
    return 0


def compare_results(baseline: Dict[str, Any], current: Dict[str, Any]) -> int:
    """Print deltas for every tracked metric; exit 1 if any tok/s regressed >10%."""
    keys = [
        ("cpu", "cpu_tok_per_s"),
        ("igpu", "igpu_tok_per_s"),
    ]
    regressed = False
    print("[bench_llm] comparison vs baseline:")
    for backend, metric in keys:
        b = baseline.get(backend, {}).get(metric)
        c = current.get(backend, {}).get(metric)
        if b is None or c is None or b == 0:
            print(f"  {metric:16s} n/a")
            continue
        delta = (c - b) / b
        flag = ""
        if delta < -0.10:
            flag = "  <-- REGRESSION >10%"
            regressed = True
        print(f"  {metric:16s} {b:8.2f} -> {c:8.2f} ({delta:+.1%}){flag}")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())