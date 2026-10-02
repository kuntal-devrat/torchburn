#!/usr/bin/env python3
"""Run Llama 3.2 1B with TorchBurn INT4 Quantization on CPU or GPU.

Usage examples:
    # 1. Interactive device selection:
    python examples/run_llama.py

    # 2. Run directly on GPU (Vulkan / DirectX 12 / Metal) with INT4:
    python examples/run_llama.py --device gpu

    # 3. Run directly on CPU (AVX2 / AVX-512 SIMD Pure-Rust decoder) with INT4:
    python examples/run_llama.py --device cpu

    # 4. Interactive multi-turn chat:
    python examples/run_llama.py --device gpu --chat

    # 5. Specify custom prompt and token limit:
    python examples/run_llama.py --device gpu --prompt "Explain quantum computing in 2 sentences." --max-tokens 64

    # 6. Use a local checkpoint directory or GGUF file:
    python examples/run_llama.py --model path/to/Llama-3.2-1B-Instruct --device gpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

# Enable high-speed Rust-based multi-threaded parallel downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

try:
    import torchburn as tb
except ImportError:
    print("\033[91mError: torchburn is not installed in the active environment.\033[0m")
    print("Please install or run from the repo: pip install -e .")
    sys.exit(1)


DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
OPEN_MIRROR = "unsloth/Llama-3.2-1B-Instruct"


def check_and_display_gpu_info() -> bool:
    """Inspect and display GPU hardware capabilities."""
    try:
        available = bool(tb.gpu_available())
        info = tb.gpu_info()
        if isinstance(info, dict):
            adapter = info.get("adapter_name", "Unknown GPU")
            backend = info.get("backend", "Unknown Backend")
        elif isinstance(info, (tuple, list)):
            adapter = info[1] if len(info) > 1 else "Unknown GPU"
            backend = info[2] if len(info) > 2 else "Unknown Backend"
        else:
            adapter, backend = "Unknown GPU", "WGPU"

        if available:
            print(f"[\033[92mGPU Detected\033[0m] {adapter} via {backend}")
        else:
            print("[\033[93mGPU Not Available\033[0m] No compatible WGPU adapter detected.")
        return available
    except Exception as e:
        print(f"[\033[93mGPU Probe Warning\033[0m] {e}")
        return False


def select_device_interactively(gpu_available: bool) -> str:
    """Prompt the user interactively to pick CPU or GPU if not passed via CLI."""
    print("\n\033[1mSelect Execution Device for TorchBurn Engine:\033[0m")
    print("  \033[94m[1]\033[0m CPU  - Zero-Python Pure-Rust Decoder (AVX-512 / AVX2 SIMD)")
    if gpu_available:
        print("  \033[95m[2]\033[0m GPU  - End-to-End WGPU Compute Graph (Vulkan / DirectX 12 / Metal)")
        default_choice = "2"
    else:
        print("  \033[90m[2] GPU  - (WGPU adapter unavailable, CPU fallback will be used)\033[0m")
        default_choice = "1"

    prompt = f"Enter choice [1/2, default={default_choice}]: "
    try:
        choice = input(prompt).strip() or default_choice
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if choice == "2":
        return "gpu"
    return "cpu"


def resolve_hf_token(cli_token: Optional[str]) -> Optional[str]:
    """Retrieve Hugging Face token from CLI, environment, cache, or interactive input."""
    if cli_token:
        return cli_token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass

    # Prompt user interactively if no token is configured
    print("\n\033[93mHugging Face Authentication\033[0m (Unlocks full download speeds & gated repos):")
    try:
        user_tok = input("Enter HF Token (optional, press Enter to skip): ").strip()
        if user_tok:
            os.environ["HF_TOKEN"] = user_tok
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(user_tok)
            except Exception:
                pass
            return user_tok
    except (EOFError, KeyboardInterrupt):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Llama 1B with TorchBurn INT4 quantization on CPU or GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID or local directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "igpu", "dgpu", "auto"],
        default=None,
        help="Target compute device: 'cpu' or 'gpu' (or 'igpu' / 'dgpu' / 'auto')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why the sky is blue in three simple, engaging sentences.",
        help="Input text prompt for text generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token for gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch interactive multi-turn chat session",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU worker threads (default: all physical cores)",
    )

    args = parser.parse_args()

    print("=" * 68)
    print(" \033[1;36mTorchBurn Universal LLM Engine — Llama 1B (INT4 Quantized)\033[0m")
    print("=" * 68)

    # 1. Hardware Detection & Device Resolution
    gpu_available = check_and_display_gpu_info()
    device = args.device
    if device is None:
        device = select_device_interactively(gpu_available)
    else:
        device = device.lower()

    print(f"\n[\033[1mConfiguration\033[0m]")
    print(f"  - Model:        \033[94m{args.model}\033[0m")
    print(f"  - Quantization: \033[92mINT4\033[0m (grouped SIMD weights)")
    print(f"  - Device:       \033[95m{device.upper()}\033[0m")
    if args.threads:
        print(f"  - CPU Threads:  {args.threads}")

    token = resolve_hf_token(args.token)
    if token:
        print("  - HF Auth:      \033[92mAuthenticated\033[0m (high-speed downloads active)")
    else:
        print("  - HF Auth:      \033[90mAnonymous\033[0m")

    # 2. Load Model & Engine
    print(f"\n[\033[94m1/2\033[0m] Initializing TorchBurn INT4 Engine...")
    t0 = time.perf_counter()
    model_to_load = args.model
    try:
        llm = tb.LLM.from_pretrained(
            model_id_or_path=model_to_load,
            quant="int4",
            device=device,
            token=token,
            num_threads=args.threads,
        )
    except Exception as e:
        err_msg = str(e)
        is_gated = any(k in err_msg.lower() for k in ("401", "403", "gated", "restricted"))
        if is_gated and "meta-llama" in model_to_load:
            print(f"\n\033[93mNotice: Access to '{model_to_load}' is restricted on this Hugging Face account.\033[0m")
            print(f"[\033[94mAuto-Fallback\033[0m] Switching to open un-gated weights mirror: \033[92m{OPEN_MIRROR}\033[0m...\n")
            model_to_load = OPEN_MIRROR
            llm = tb.LLM.from_pretrained(
                model_id_or_path=model_to_load,
                quant="int4",
                device=device,
                token=token,
                num_threads=args.threads,
            )
        else:
            print(f"\n\033[91mFailed to load model: {err_msg}\033[0m")
            if is_gated:
                print("\n\033[93mTip: This model requires an authorized Hugging Face token.\033[0m")
                print(f"Run with open mirror: python examples/run_llama.py --model {OPEN_MIRROR}")
                print("Or set:               $env:HF_TOKEN=\"<YOUR_HF_TOKEN>\"")
            sys.exit(1)

    load_time = time.perf_counter() - t0
    print(f"[\033[92mReady\033[0m] Model initialized in {load_time:.2f}s.\n")

    # 3. Interactive Chat Mode or Single Prompt Generation
    if args.chat:
        print("=" * 68)
        print(" \033[1mInteractive Chat Session (type 'exit' or 'quit' to end)\033[0m")
        print("=" * 68)
        llm.chat(system_prompt="You are a helpful, brilliant, and concise AI assistant.")
    else:
        print("=" * 68)
        print(f"\033[1mPrompt:\033[0m {args.prompt}")
        print("=" * 68)
        print("\033[1;32mResponse:\033[0m ", end="", flush=True)

        token_count = 0
        gen_start = time.perf_counter()
        first_token_time = None

        for chunk in llm.stream(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(chunk, end="", flush=True)
            token_count += 1

        gen_total_time = time.perf_counter() - gen_start
        print("\n" + "=" * 68)

        # Performance telemetry
        if token_count > 0 and gen_total_time > 0:
            tok_per_sec = token_count / gen_total_time
            ttft_ms = (first_token_time - gen_start) * 1000 if first_token_time else 0
            print(f"\033[90mGenerated {token_count} tokens in {gen_total_time:.2f}s "
                  f"({tok_per_sec:.1f} tok/s) | TTFT: {ttft_ms:.1f}ms | Device: {device.upper()}\033[0m")
        print("=" * 68)


if __name__ == "__main__":
    main()
