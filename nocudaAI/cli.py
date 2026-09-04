"""Command-Line Interface for NoCudaAI: Zero-CUDA Local AI Assistant."""

from __future__ import annotations
import argparse
import sys
import torch

import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TORCHBURN_SUPPRESS_FALLBACK_WARNINGS", "1")
warnings.filterwarnings("ignore", message=".*Failed to find CUDA.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*torchburn: falling back to eager PyTorch.*")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .config import MODEL_PROFILES, ModelConfig, GenerationConfig, EngineConfig
from .model import NoCudaModel
from .tokenizer import NoCudaTokenizer, get_tokenizer
from .engine import NoCudaEngine
from .agent import NoCudaAgent, TerminalColors
from .benchmark import run_benchmark


BANNER = r"""
  _   _         ____          _         _    ___ 
 | \ | | ___   / ___|   _  __| | __ _  / \  |_ _|
 |  \| |/ _ \ | |  | | | |/ _` |/ _` |/ _ \  | | 
 | |\  | (_) || |__| |_| | (_| | (_| / ___ \ | | 
 |_| \_|\___/  \____\__,_|\__,_|\__,_/_/   \_\___|
   [>>] No CUDA * No llama.cpp * iGPU Acceleration via TorchBurn [<<]
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nocudaAI",
        description="NoCudaAI: Small Models running directly on your iGPU via TorchBurn (Vulkan/WGPU). No CUDA, No llama.cpp.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Global options on parent parser or subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", choices=list(MODEL_PROFILES.keys()), default="micro", help="Model size profile")
    common.add_argument("--model", default=None, help="Pretrained model ID or path (e.g. 'qwen_0_5b', 'Qwen/Qwen2.5-0.5B-Instruct')")
    common.add_argument("--engine", choices=["igpu", "cpu", "eager"], default="igpu", help="Execution engine (default: igpu)")
    common.add_argument("--tb-engine", default=None, help="TorchBurn engine target override")
    common.add_argument("--threads", type=int, default=None, help="Number of CPU threads to utilize")
    common.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    common.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    common.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # 1. Chat Command
    subparsers.add_parser("chat", parents=[common], help="Interactive conversational chat")

    # 2. Prompt Command
    prompt_p = subparsers.add_parser("prompt", parents=[common], help="Single-shot prompt completion")
    prompt_p.add_argument("text", type=str, help="Input prompt text")

    # 3. Benchmark Command
    bench_p = subparsers.add_parser("bench", parents=[common], help="Compare Eager vs TorchBurn CPU vs iGPU")
    bench_p.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")

    # 4. Agent Command
    subparsers.add_parser("agent", parents=[common], help="Interactive terminal agent with autonomous tool calling")

    # 5. Download Command
    dl_p = subparsers.add_parser("download", help="Download pretrained model weights (e.g. Qwen 0.5B)")
    dl_p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID to download")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    C = TerminalColors
    print(C.CYAN + BANNER + C.RESET)

    if getattr(args, "threads", None):
        torch.set_num_threads(args.threads)

    # Dispatch commands
    if args.command == "download":
        print(f"Resolving and downloading pretrained weights for: {args.model}")
        model = NoCudaModel.from_pretrained(args.model)
        print(f"{C.GREEN}Download complete! Model ready for inference.{C.RESET}")
        return

    model_target = getattr(args, "model", None) or args.profile
    if args.command == "bench":
        run_benchmark(
            profile_name=model_target,
            gen_tokens=args.max_tokens,
            num_runs=args.runs,
        )
        return

    # Initialize model and tokenizer
    is_pretrained = any(q in model_target.lower() for q in ("qwen", "0.5b", "0_5b")) or (
        isinstance(model_target, str) and (model_target.endswith(".safetensors") or "/" in model_target)
    )

    if is_pretrained:
        print(f"{C.BLUE}[NoCudaAI]{C.RESET} Loading pretrained model '{model_target}'...")
        model = NoCudaModel.from_pretrained(model_target)
        tokenizer = get_tokenizer(model_target)
    else:
        config = MODEL_PROFILES.get(model_target, MODEL_PROFILES["micro"])
        tokenizer = get_tokenizer(model_target, vocab_size=config.vocab_size)
        torch.manual_seed(args.seed)
        model = NoCudaModel(config).eval()

    engine_cfg = EngineConfig(
        engine=args.engine,
        torchburn_engine=args.tb_engine,
        num_threads=args.threads,
    )
    engine = NoCudaEngine(model, tokenizer, engine_cfg)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    if args.command == "prompt":
        print(f"{C.BOLD}Prompt:{C.RESET} {args.text}\n")
        print(f"{C.CYAN}{C.BOLD}NoCuda{C.RESET}: ", end="", flush=True)
        summary = None
        for pkt in engine.generate_stream(args.text, gen_cfg):
            if pkt["type"] == "token":
                print(pkt["text"], end="", flush=True)
            elif pkt["type"] == "summary":
                summary = pkt
        print()
        if summary:
            print(
                f"\n{C.DIM}⚡ [{summary['tokens_generated']} tokens | {summary['avg_ms_per_token']:.1f} ms/tok | {summary['decode_tok_sec']:.1f} tok/s]{C.RESET}\n"
            )

    elif args.command == "chat":
        print(f"{C.GREEN}Interactive Chat Session Started.{C.RESET}")
        print(f"Profile: {args.profile} ({model.get_num_params() / 1e6:.2f}M params) | Engine: {args.engine}\nType 'exit' or 'quit' to end.\n")
        agent = NoCudaAgent(engine, tokenizer)
        while True:
            try:
                user_msg = input(f"{C.BOLD}You:{C.RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{C.YELLOW}Session ended.{C.RESET}")
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", "q"):
                print(f"{C.YELLOW}Goodbye!{C.RESET}")
                break
            agent.chat_round(user_msg, gen_cfg, max_tool_iterations=1)
            print()

    elif args.command == "agent":
        print(f"{C.GREEN}Autonomous Terminal Agent Activated.{C.RESET}")
        print(f"Available tools: calc, sys_info, file_view, run_cmd\nType 'exit' to quit.\n")
        agent = NoCudaAgent(engine, tokenizer)
        while True:
            try:
                user_msg = input(f"{C.BOLD}You:{C.RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{C.YELLOW}Session ended.{C.RESET}")
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", "q"):
                print(f"{C.YELLOW}Goodbye!{C.RESET}")
                break
            agent.chat_round(user_msg, gen_cfg, max_tool_iterations=3)
            print()


if __name__ == "__main__":
    main()
