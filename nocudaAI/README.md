# NoCudaAI ⚡ (Showcase #2: Local LLM on iGPU with TorchBurn)

> **Run PyTorch LLMs directly on your Integrated GPU (Intel Iris Xe, AMD Radeon, Apple Silicon) — No CUDA, No llama.cpp required.**

NoCudaAI is Showcase #2 of **TorchBurn**: a demonstration of running modern Transformer Language Models directly on consumer **Integrated GPUs (iGPUs)** using standard PyTorch models, completely bypassing NVIDIA CUDA and without requiring GGUF conversion or `llama.cpp`.

---

## 💡 Why Showcase #2 Matters

Historically, running local LLMs without dedicated NVIDIA hardware meant either:
1. Converting PyTorch weights to GGUF format and compiling `llama.cpp`.
2. Enduring slow, unoptimized CPU eager execution.
3. Dealing with complex OpenCL/Vulkan custom runtimes.

**With TorchBurn, none of that is needed.**
Standard PyTorch neural network modules (`torch.nn.Module`) are JIT compiled using `torchburn.compile(model)` directly into **Vulkan / DirectX 12 / Metal compute shaders via WGPU**.

```
┌────────────────────────────────────────┐
│       Standard PyTorch nn.Module       │
│  (RMSNorm + RoPE + SwiGLU + Attention) │
└───────────────────┬────────────────────┘
                    │ torchburn.compile()
                    ▼
┌────────────────────────────────────────┐
│          TorchBurn JIT Engine          │
│   Single-pass loop & activation fusion │
└───────────────────┬────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│   iGPU (WGPU)   │   │  CPU Fallback   │
│ Intel Iris Xe / │   │ AVX2 / SIMD     │
│ Vulkan Shaders  │   │ Rayon Multicore │
│ (Zero CUDA!)    │   │ (Zero Crash!)   │
└─────────────────┘   └─────────────────┘
```

---

## 🌟 Key Highlights

- 🎮 **Integrated GPU Acceleration**: Discovers and runs on your local iGPU (tested on **Intel(R) Iris(R) Xe Graphics** via Vulkan).
- 🚫 **Zero CUDA Dependencies**: Zero NVIDIA GPU requirements. Runs on thin laptops, ultrabooks, and everyday developer machines.
- 🦙 **No llama.cpp or GGUF Needed**: Keep your native PyTorch model definitions and weights; no third-party quantization format or C++ bindings required.
- 🛡️ **Seamless Fallback**: If a tensor shape or memory limit exceeds the iGPU device pool, TorchBurn automatically falls back to optimized AVX2 SIMD CPU kernels without crashing.
- 🤖 **Autonomous Terminal Agent**: Interactive chat, streaming token generation, and multi-tool agent execution (math evaluation, system diagnostics, and shell execution).
- 📊 **3-Way Comparative Benchmark**: Built-in side-by-side benchmark comparing **TorchBurn iGPU**, **TorchBurn Native CPU**, and **PyTorch Eager**.

---

## 🚀 Quickstart

### 1. Installation

Ensure you have Python 3.10+ and a Rust toolchain:

```bash
# In the repository root
pip install -r nocudaAI/requirements.txt
maturin develop --release
```

### 2. Run Interactive Chat on iGPU

Launch an interactive conversation powered by your integrated graphics:

```bash
python -m nocudaAI.cli chat --profile pico --engine igpu
```

Available model profiles:
- `pico`: Ultra-fast test profile (~1M params, instant response)
- `micro`: Balanced laptop profile (~5M params)
- `nano`: Higher capacity profile (~18M params)
- `smollm_tiny`: Grouped-Query Attention profile (~30M params)
- `qwen_0_5b` / `qwen2_5_0_5b`: **Pretrained Qwen 0.5B** (~494M params, high-quality instruction following)

### 3. Pretrained Qwen 0.5B Inference (No CUDA, No llama.cpp)

Run official pretrained **Qwen 0.5B** (`Qwen/Qwen2.5-0.5B-Instruct`) weights directly on your iGPU or CPU:

```bash
# Single-shot prompt with Qwen 0.5B
python -m nocudaAI.main prompt "Explain why integrated GPUs are great for AI" --model qwen_0_5b --engine igpu

# Interactive chat with Qwen 0.5B
python -m nocudaAI.main chat --model qwen_0_5b --engine igpu

# Download weights locally
python -m nocudaAI.main download --model Qwen/Qwen2.5-0.5B-Instruct
```

### 3. Single-Shot Prompt Generation

Stream a completion directly to your console:

```bash
python -m nocudaAI.cli prompt "Explain how TorchBurn enables iGPU inference without CUDA" --profile pico --engine igpu --max-tokens 32
```

### 4. 3-Way Benchmark (Eager vs CPU vs iGPU)

Compare your system's performance across PyTorch Eager, TorchBurn Native CPU, and TorchBurn iGPU:

```bash
python -m nocudaAI.cli bench --profile pico --max-tokens 16
```

### 5. Autonomous Terminal Agent Mode

Start an agent with real-time tool calling running on your iGPU:

```bash
python -m nocudaAI.cli agent --engine igpu
```

Try asking:
- *"Inspect my current hardware and GPU status"*
- *"Calculate 2 ** 32 / (1024 * 1024)"*
- *"Show the first 10 lines of README.md"*

---

## 📊 Technical Specifications

| Feature | PyTorch Vanilla | llama.cpp | TorchBurn (Showcase #2) |
| :--- | :---: | :---: | :---: |
| **Input Format** | PyTorch `nn.Module` | GGUF binary format | **Native PyTorch `nn.Module`** |
| **iGPU Acceleration** | ❌ (CUDA only) | ⚠️ (Requires Vulkan build) | **✅ (Built-in via WGPU / Vulkan)** |
| **Compilation API** | `torch.compile()` | N/A (Offline conversion) | **`torchburn.compile(model)`** |
| **Fallback Strategy** | Crash on non-CUDA | Fails if backend unsupported | **Seamless AVX2 CPU SIMD fallback** |
| **Kernel Fusion** | Heavy Triton/CUDA deps | Static hand-written kernels | **Single-pass JIT Rust fusion** |

---

## 📜 License

Apache-2.0. Built with ❤️ for the TorchBurn and Rust machine learning ecosystem.
