"""Central model registry shared by the loader and tokenizer.

Holds the single source of truth for:
  * short-name aliases -> canonical Hugging Face repo ids,
  * where offline model checkpoints are searched,
  * family fallbacks (special-token ids) when metadata files are absent.

Everything else in the package should import from here instead of re-encoding
alias tables, machine-specific paths, or magic token ids.
"""

from __future__ import annotations

import os
from typing import List

# Convenience aliases -> canonical Hugging Face repo ids.
MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen_0_5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "default": "Qwen/Qwen2.5-0.5B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

# Fallback special-token ids used only when a model dir ships no
# tokenizer_config.json with usable ids. Keyed by repo-id substring so a
# fallback never silently applies a foreign id to an unrelated family.
FALLBACK_EOS_IDS = {
    "qwen": 151643,
    "deepseek": 151645,
}


def resolve_repo_id(model_id_or_path: str) -> str:
    """Map a short alias to its canonical Hugging Face repo id."""
    return MODEL_ALIASES.get(model_id_or_path, model_id_or_path)


def repo_snapshot_dir(repo_id: str) -> str:
    """Directory of the newest local snapshot of a HF repo, if cached."""
    snap_root = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots"
    )
    try:
        snaps = sorted(os.listdir(snap_root))
        if snaps:
            return os.path.join(snap_root, snaps[-1])
    except OSError:
        pass
    return snap_root


def local_model_roots() -> List[str]:
    """Directory roots searched for offline checkpoints (configurable)."""
    roots = []
    env_dir = os.environ.get("TORCHBURN_MODELS_DIR", "").strip()
    if env_dir:
        roots.append(env_dir)
    roots.append(os.path.join(os.path.expanduser("~/.cache/torchburn"), "models"))
    return roots


def offline_candidates(model_id_or_path: str) -> List[str]:
    """Local paths probed by the loader before hitting the network."""
    repo_id = resolve_repo_id(model_id_or_path)
    cands = [model_id_or_path]
    for root in local_model_roots():
        cands.append(os.path.join(root, repo_id.split("/")[-1]))
        cands.append(os.path.join(root, model_id_or_path))
    cands.append(repo_snapshot_dir(repo_id))
    return cands


def fallback_eos_id(model_id_or_path: str) -> int:
    """Best-effort EOS id for tokenizer_config-less checkpoints."""
    low = (model_id_or_path or "").lower()
    for family, eos in FALLBACK_EOS_IDS.items():
        if family in low:
            return eos
    return 151643
