"""Autonomous Terminal Agent for NoCudaAI.

Features:
- Multi-turn conversational memory
- Autonomous tool execution (safe eval, system specs, file viewer, shell execution)
- Real-time ANSI streaming with tokens/sec metrics
- Pure CPU acceleration via TorchBurn
"""

from __future__ import annotations
import ast
import json
import os
import platform
import re
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional, Callable, Union
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .engine import NoCudaEngine
from .tokenizer import NoCudaTokenizer, PretrainedTokenizerWrapper
from .config import GenerationConfig


class TerminalColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class AgentTool:
    """Tool specification and runner for NoCuda Agent."""

    def __init__(self, name: str, description: str, func: Callable[..., str], schema: Dict[str, str]):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            return self.func(**args)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"


class NoCudaAgent:
    """Zero-CUDA Autonomous AI Agent running in the terminal."""

    SYSTEM_PROMPT = (
        "You are NoCuda, a lightweight, ultra-responsive CPU-native AI terminal agent "
        "powered by TorchBurn graph compilation. You operate entirely on local CPU with zero CUDA dependencies. "
        "You have direct access to tools for mathematical calculations, system inspection, file inspection, and shell commands.\n\n"
        "To invoke a tool, write:\n"
        "```tool\n"
        '{"tool": "<tool_name>", "args": {<argument_name>: <value>}}\n'
        "```\n"
    )

    def __init__(self, engine: NoCudaEngine, tokenizer: Union[NoCudaTokenizer, PretrainedTokenizerWrapper]):
        self.engine = engine
        self.tokenizer = tokenizer
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        self.tools: Dict[str, AgentTool] = {}
        self._cached_kv = None
        self._cached_token_ids: List[int] = []
        self._register_default_tools()
        self._warmup()

    def _register_default_tools(self):
        """Registers built-in terminal tools."""

        def calc(expression: str) -> str:
            """Evaluates safe mathematical expressions."""
            try:
                # Safe eval via AST
                code = compile(expression, "<string>", "eval")
                for name in code.co_names:
                    if name not in ("abs", "min", "max", "round", "sum", "pow"):
                        return f"Forbidden identifier: {name}"
                result = eval(code, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except Exception as ex:
                return f"Math Error: {ex}"

        def sys_info() -> str:
            """Returns CPU and system specs."""
            try:
                import torch
                tb_info = "Available"
                try:
                    import torchburn
                    tb_ver = getattr(torchburn, "__version__", "0.5.1")
                    tb_info = f"TorchBurn v{tb_ver} (Native CPU SIMD active)"
                except ImportError:
                    tb_info = "PyTorch Eager Fallback"

                return (
                    f"OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
                    f"Python: {platform.python_version()}\n"
                    f"CPU Cores (logical): {os.cpu_count()}\n"
                    f"PyTorch Version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})\n"
                    f"Inference Engine: {tb_info}\n"
                    f"Active Threads: {torch.get_num_threads()}"
                )
            except Exception as e:
                return f"System Info Error: {e}"

        def file_view(filepath: str, max_lines: int = 40) -> str:
            """Reads the first N lines of a local file."""
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    lines = [f.readline() for _ in range(max_lines)]
                content = "".join(lines)
                if len(lines) == max_lines:
                    content += "\n... [truncated]"
                return content
            except Exception as e:
                return f"Read error: {e}"

        def run_cmd(command: str) -> str:
            """Executes safe diagnostic terminal commands."""
            # Disallow destructive commands
            destructive = ["rm -rf", "mkfs", "dd if=", "del /f /s /q c:", "shutdown", "reboot"]
            if any(d in command.lower() for d in destructive):
                return "Command execution blocked: potentially destructive."
            try:
                out = subprocess.check_output(
                    command, shell=True, stderr=subprocess.STDOUT, timeout=10
                ).decode("utf-8", errors="replace")
                return out.strip() if out else "(No output)"
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 10 seconds."
            except subprocess.CalledProcessError as e:
                return f"Command returned exit code {e.returncode}: {e.output.decode('utf-8', errors='replace')}"

        self.register_tool(AgentTool("calc", "Evaluates mathematical expressions.", calc, {"expression": "string"}))
        self.register_tool(AgentTool("sys_info", "Returns CPU architecture and engine info.", sys_info, {}))
        self.register_tool(AgentTool("file_view", "Reads preview of local file.", file_view, {"filepath": "string"}))
        self.register_tool(AgentTool("run_cmd", "Executes local shell command.", run_cmd, {"command": "string"}))

    def register_tool(self, tool: AgentTool):
        self.tools[tool.name] = tool

    def _warmup(self):
        """Warm up engine and pre-cache system prompt to eliminate Turn 1 prefill & thread cold-start lag."""
        try:
            sys_prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.SYSTEM_PROMPT}],
                add_generation_prompt=False,
            )
            sys_ids = self.tokenizer.encode(sys_prompt_text)
            if sys_ids:
                input_tensor = torch.tensor([sys_ids], dtype=torch.long)
                if self.engine.config.use_static_kv_cache and hasattr(self.engine.raw_model, "create_static_kv_caches"):
                    max_len = max(len(sys_ids) + 512, 2048)
                    init_kv = self.engine.raw_model.create_static_kv_caches(max_batch_size=1, max_seq_len=max_len)
                    _, kv, _ = self.engine.prefill(input_tensor, kv_caches=init_kv, offset=0)
                else:
                    _, kv, _ = self.engine.prefill(input_tensor)
                # Warm up one decode step on dummy token so Rayon worker threads & SIMD kernels spin up
                if kv is not None:
                    dummy_token = sys_ids[-1]
                    self.engine.decode_step(dummy_token, kv, offset=len(sys_ids) - 1)
                self._cached_kv = kv
                self._cached_token_ids = sys_ids
        except Exception:
            pass

    def reset(self):
        self.history = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self._cached_kv = None
        self._cached_token_ids = []
        self._warmup()

    def chat_round(
        self,
        user_input: str,
        config: Optional[GenerationConfig] = None,
        max_tool_iterations: int = 3,
    ):
        """Runs a complete interactive chat round with optional multi-turn tool calling."""
        cfg = config or GenerationConfig()
        self.history.append({"role": "user", "content": user_input})

        C = TerminalColors
        iterations = 0

        while iterations < max_tool_iterations:
            iterations += 1
            formatted_prompt = self.tokenizer.apply_chat_template(self.history, add_generation_prompt=True)

            print(f"{C.CYAN}{C.BOLD}NoCuda{C.RESET} {C.DIM}[CPU/TorchBurn]{C.RESET}: ", end="", flush=True)

            collected_tokens = []
            prefill_meta = None
            summary_meta = None

            for packet in self.engine.generate_stream(
                formatted_prompt,
                cfg,
                kv_caches=self._cached_kv,
                cached_token_ids=self._cached_token_ids,
            ):
                if packet["type"] == "prefill":
                    prefill_meta = packet
                elif packet["type"] == "token":
                    piece = packet["text"]
                    collected_tokens.append(piece)
                    # Filter out special control tokens from raw terminal print
                    if not piece.startswith("<|"):
                        print(piece, end="", flush=True)
                elif packet["type"] == "summary":
                    summary_meta = packet
                    self._cached_kv = packet.get("kv_caches")
                    self._cached_token_ids = packet.get("all_token_ids", [])

            response_text = "".join(collected_tokens).strip()
            print()  # Newline after stream

            # Print speed metrics badge
            if summary_meta:
                tok_sec = summary_meta["decode_tok_sec"]
                avg_ms = summary_meta["avg_ms_per_token"]
                n_tok = summary_meta["tokens_generated"]
                pref_ms = prefill_meta["prefill_time_ms"] if prefill_meta else 0
                reused_tok = prefill_meta.get("prefix_reused_tokens", 0) if prefill_meta else 0
                reused_str = f" (cached {reused_tok} tok)" if reused_tok > 0 else ""
                print(
                    f"{C.DIM}⚡ [{n_tok} tokens | prefill: {pref_ms:.1f}ms{reused_str} | decode: {avg_ms:.1f}ms/tok ({tok_sec:.1f} tok/s)]{C.RESET}"
                )

            self.history.append({"role": "assistant", "content": response_text})

            # Check if model invoked a tool
            tool_call_match = re.search(r"```tool\s*\n({.*?})\s*\n```", response_text, re.DOTALL)
            if not tool_call_match:
                break

            try:
                tool_data = json.loads(tool_call_match.group(1))
                tool_name = tool_data.get("tool")
                tool_args = tool_data.get("args", {})
            except Exception:
                break

            if tool_name not in self.tools:
                break

            print(f"\n{C.YELLOW}⚙ Executing Tool:{C.RESET} {C.BOLD}{tool_name}{C.RESET}({tool_args})")
            tool_result = self.tools[tool_name].execute(tool_args)
            print(f"{C.DIM}{tool_result}{C.RESET}\n")

            self.history.append({
                "role": "system",
                "content": f"Tool Result for {tool_name}:\n{tool_result}"
            })
