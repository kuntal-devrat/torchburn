"""Interactive multi-turn chat in 3 lines with TorchBurn LLM.

Run directly:
    python examples/llm_chat.py
"""

import torchburn as tb

# Load model and launch interactive terminal chat
llm = tb.LLM.from_pretrained("models/qwen_0_5b", quant="int4", device="auto")
llm.chat(system_prompt="You are a brilliant, concise AI assistant.")
