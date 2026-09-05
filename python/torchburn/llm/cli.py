"""Command-line interface for TorchBurn LLM engine."""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Optional

import torchburn as tb
from torchburn.llm.config import EngineConfig, GenerationConfig


def main(args: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="torchburn.llm",
        description="TorchBurn Universal LLM Engine — Fast, Hardware-Agnostic, Zero-CUDA Inference.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")


    # --- chat ---
    chat_parser = subparsers.add_parser("chat", help="Launch interactive multi-turn chat")
    chat_parser.add_argument("--model", "-m", type=str, default=r"models/qwen_0_5b", help="Model path or HuggingFace repo ID")
    chat_parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cpu", "igpu", "dgpu", "gpu", "vulkan"], help="Execution target")
    chat_parser.add_argument("--quant", "-q", type=str, default="int4", choices=["int4", "int8", "none"], help="Weight quantization")
    chat_parser.add_argument("--system", "-s", type=str, default=None, help="System prompt")
    chat_parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens per turn")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--token", type=str, default=None, help="Hugging Face API token for gated models")
    chat_parser.add_argument("--threads", type=int, default=None, help="CPU thread count")

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate completion for a prompt")
    gen_parser.add_argument("prompt", type=str, help="Input prompt text")
    gen_parser.add_argument("--model", "-m", type=str, default=r"models/qwen_0_5b", help="Model path or HuggingFace repo ID")
    gen_parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cpu", "igpu", "dgpu", "gpu", "vulkan"], help="Execution target")
    gen_parser.add_argument("--quant", "-q", type=str, default="int4", choices=["int4", "int8", "none"], help="Weight quantization")
    gen_parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--stream", action="store_true", help="Stream tokens as they are produced")
    gen_parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    gen_parser.add_argument("--threads", type=int, default=None, help="CPU thread count")

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark generation speed (tok/s)")
    bench_parser.add_argument("--model", "-m", type=str, default=r"models/qwen_0_5b", help="Model path or HuggingFace repo ID")
    bench_parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cpu", "igpu", "dgpu", "gpu", "vulkan"], help="Execution target")
    bench_parser.add_argument("--quant", "-q", type=str, default="int4", choices=["int4", "int8", "none"], help="Weight quantization")
    bench_parser.add_argument("--tokens", type=int, default=64, help="Tokens to generate in benchmark")
    bench_parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    bench_parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    bench_parser.add_argument("--threads", type=int, default=None, help="CPU thread count")

    # --- download ---
    dl_parser = subparsers.add_parser("download", help="Pre-download and cache a Hugging Face model")
    dl_parser.add_argument("model_id", type=str, help="Hugging Face repository ID (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    dl_parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    dl_parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory")

    parsed = parser.parse_args(args)
    if not parsed.command:
        parser.print_help()
        return 1

    if parsed.command == "download":
        from torchburn.llm.loader import ModelLoader
        print(f"Downloading model '{parsed.model_id}' from Hugging Face Hub...")
        path = ModelLoader.download_hf_model(parsed.model_id, token=parsed.token, cache_dir=parsed.cache_dir)
        print(f"Successfully downloaded to: {path}")
        return 0

    # Load model
    print(f"\n[TorchBurn] Initializing LLM from '{parsed.model}' [device={parsed.device}, quant={parsed.quant}]...")
    t0 = time.perf_counter()
    llm = tb.LLM.from_pretrained(
        parsed.model,
        quant=parsed.quant,
        device=parsed.device,
        token=parsed.token,
        num_threads=parsed.threads,
    )
    load_time = time.perf_counter() - t0
    print(f"[TorchBurn] Loaded in {load_time:.2f}s ({llm.model.get_num_params() / 1e6:.1f}M parameters).\n")

    if parsed.command == "chat":
        llm.chat(
            system_prompt=parsed.system,
            max_tokens=parsed.max_tokens,
            temperature=parsed.temperature,
        )
        return 0

    elif parsed.command == "generate":
        if parsed.stream:
            print(f"\033[1mPrompt:\033[0m {parsed.prompt}\n")
            print("\033[1mResponse:\033[0m ", end="", flush=True)
            for chunk in llm.stream(parsed.prompt, max_tokens=parsed.max_tokens, temperature=parsed.temperature):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            t_gen0 = time.perf_counter()
            resp = llm.generate(parsed.prompt, max_tokens=parsed.max_tokens, temperature=parsed.temperature)
            gen_time = time.perf_counter() - t_gen0
            print(f"\033[1mPrompt:\033[0m {parsed.prompt}\n")
            print(f"\033[1mResponse:\033[0m\n{resp}\n")
            print(f"\033[2m[Generated in {gen_time:.2f}s]\033[0m")
        return 0

    elif parsed.command == "benchmark":
        prompt = "Write a comprehensive summary of artificial intelligence and machine learning principles:"
        print(f"Running benchmark with prompt: \"{prompt}\" ({parsed.tokens} tokens)...")
        for w in range(parsed.warmup):
            _ = llm.generate(prompt, max_tokens=16, temperature=0.0)

        t_start = time.perf_counter()
        pkt = None
        count = 0
        cfg = GenerationConfig(max_new_tokens=parsed.tokens, temperature=0.0)
        for p in llm.engine.generate_stream(prompt, cfg):
            if p["type"] == "token":
                count += 1
            elif p["type"] == "summary":
                pkt = p

        total_time = time.perf_counter() - t_start
        print("\n" + "=" * 50)
        print("          TORCHBURN LLM BENCHMARK RESULTS")
        print("=" * 50)
        if pkt:
            print(f"  Prefill tokens       : {pkt.get('prefill_tokens', 0)}")
            print(f"  Prefill latency      : {pkt.get('prefill_latency_ms', 0):.2f} ms")
            print(f"  Tokens generated     : {pkt.get('tokens_generated', count)}")
            print(f"  Decode latency       : {pkt.get('decode_latency_sec', total_time):.3f} s")
            print(f"  Decode throughput    : {pkt.get('decode_tok_sec', count / total_time):.1f} tokens/sec")
            print(f"  Average step latency : {pkt.get('avg_ms_per_token', 0):.2f} ms/token")
        else:
            print(f"  Tokens generated     : {count}")
            print(f"  Total time           : {total_time:.3f} s")
            print(f"  Throughput           : {count / total_time:.1f} tokens/sec")
        print("=" * 50 + "\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
