"""Universal LLM Inference in 5–9 lines with TorchBurn.

Run directly:
    python examples/llm_inference.py
"""

import torchburn as tb

# 1. Load any model (HuggingFace repo, local directory, or .safetensors) in 1 line
llm = tb.LLM.from_pretrained(
    "models/qwen_0_5b",    # Or "Qwen/Qwen2.5-0.5B-Instruct" directly from HuggingFace
    quant="int4",          # "int4", "int8", or "none"
    device="auto",         # Auto-dispatches to Intel Iris Xe / Apple Silicon / AMD / CPU
)

# 2. Simple one-line prompt completion
print("=== Single Generation ===")
response = llm.generate("Explain black holes in two simple sentences.", max_tokens=64)
print(response)

# 3. Real-time token streaming in 3 lines
print("\n=== Real-time Streaming ===")
for token in llm.stream("Write a short three-line poem about coding:", max_tokens=48):
    print(token, end="", flush=True)
print("\n")
