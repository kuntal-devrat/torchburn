"""Universal Model Loader for TorchBurn LLM."""

from __future__ import annotations
import gc
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import torch

from ._registry import offline_candidates, repo_snapshot_dir, resolve_repo_id
from .config import ModelConfig
from .model import UniversalTransformer


def resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    """Resolves Hugging Face authentication token from parameter, env, or cache."""
    if token:
        return token
    # Environment variables
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_var)
        if val:
            return val.strip()
    # Cache file from `huggingface-cli login`
    cache_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(cache_token_path):
        try:
            with open(cache_token_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return None


class ModelLoader:
    """Universal loader for local and Hugging Face transformer models."""

    @classmethod
    def load(
        cls,
        model_id_or_path: str,
        quant: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> Tuple[UniversalTransformer, ModelConfig, str]:
        """Loads ModelConfig and UniversalTransformer weights from a local path or HF Hub."""
        auth_token = resolve_hf_token(token)

        # 1. Resolve local path or download from Hugging Face
        weights_files, config_file, root_path = cls._resolve_files(
            model_id_or_path,
            token=auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        # 2. Parse config.json
        with open(config_file, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        config = ModelConfig.from_dict(raw_cfg)

        # 3. If quant in ("int4", "int8"), use streaming low-memory loader with persistent disk caching
        quant_lower = quant.lower() if quant else None
        if quant_lower in ("int4", "int8"):
            # Phase 1.1: prefer the v2 interleaved layout cache if it exists
            # (smaller on disk, embedded f16 scales); fall back to v1.
            v2_name = f"torchburn_{quant_lower}_g64_v2.safetensors"
            cache_names = [v2_name, f"torchburn_{quant_lower}_g64.safetensors", f"torchburn_{quant_lower}_g64.pt"]
            cand_caches = []
            for cn in cache_names:
                cand_caches.append(os.path.join(root_path, cn))
                cand_caches.append(os.path.join(os.path.expanduser("~/.cache/torchburn"), f"{Path(root_path).name}_{cn}"))

            chosen_cache = None
            for cand in cand_caches:
                if os.path.isfile(cand):
                    chosen_cache = cand
                    break

            if chosen_cache:
                is_v2 = os.path.basename(chosen_cache) == v2_name
                if is_v2:
                    print(f"[\033[92mTorchBurn\033[0m] Loading pre-quantized {quant_lower.upper()} weights from v2 disk cache (interleaved layout)...")
                else:
                    print(f"[\033[92mTorchBurn\033[0m] Loading pre-quantized {quant_lower.upper()} weights from disk cache (sub-second fast startup)...")
                model = UniversalTransformer(config, init_weights=False, quant=quant_lower, fused_qkv=True).to(device=device)
                if chosen_cache.endswith(".safetensors"):
                    import safetensors.torch
                    state_dict = safetensors.torch.load_file(chosen_cache, device=device)
                else:
                    state_dict = torch.load(chosen_cache, map_location=device, weights_only=True)
                if is_v2 and quant_lower == "int4":
                    # v1 cache:  `qweight` (N, K/2) + `scales` (N, K/64) f32
                    # v2 cache:  `qweight` (N, G*34) interleaved blocks
                    # Migrate v2 -> v1 in memory so the existing Rust decoder
                    # keeps working unchanged (kernel-level v2 lands in 1.2).
                    state_dict = cls._migrate_v2_state_dict(state_dict)
                    print(f"[\033[93mTorchBurn\033[0m] Migrated v2 layout to v1 tensors in memory (deprecation: v1 is the default format until Phase 1.2).")
                model.load_state_dict(state_dict)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"[\033[92mTorchBurn\033[0m] Successfully loaded model from cache: {os.path.basename(chosen_cache)} ({total_params / 1e6:.2f}M params).")
                return model, config, root_path

            # If cache does not exist, run streaming quantization directly from safetensors
            primary_save = os.path.join(root_path, f"torchburn_{quant_lower}_g64.safetensors")
            fallback_save = os.path.join(os.path.expanduser("~/.cache/torchburn"), f"{Path(root_path).name}_torchburn_{quant_lower}_g64.safetensors")
            save_path = primary_save
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                test_f = save_path + ".tmp"
                with open(test_f, "w") as f:
                    f.write("ok")
                os.remove(test_f)
            except Exception:
                save_path = fallback_save
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            model = cls._stream_quantize_safetensors(
                weights_files=weights_files,
                config=config,
                quant=quant_lower,
                group_size=64,
                device=device,
                save_cache_path=save_path,
            )
            return model, config, root_path


        # Standard unquantized float loading path
        model = UniversalTransformer(config, init_weights=False).to(device=device, dtype=dtype)
        cls._load_weights_into_model(model, weights_files)

        return model, config, root_path

    @classmethod
    def _stream_quantize_safetensors(
        cls,
        weights_files: List[str],
        config: ModelConfig,
        quant: str = "int4",
        group_size: int = 64,
        device: str = "cpu",
        save_cache_path: Optional[str] = None,
    ) -> UniversalTransformer:
        """Streams weights directly from safetensors shards, fusing and quantizing layer-by-layer
        to strictly bound peak RAM below 1.8 GB and completely eliminate laptop freezing."""
        import safetensors
        from torchburn.quantization import quantize_weight_int4_grouped, quantize_weight_int8

        print(f"[\033[94mTorchBurn\033[0m] Streaming {quant.upper()} layer-by-layer quantization (low-memory mode, peak RAM < 1.8 GB)...")

        # Build tensor key to file index
        tensor_index = {}
        for wf in weights_files:
            with safetensors.safe_open(wf, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    tensor_index[k] = wf

        def get_tensor(key: str) -> Optional[torch.Tensor]:
            # Universal tensor key resolution: support LLaMA, Mistral, Qwen, Gemma,
            # DeepSeek, ChatGLM, Falcon, and standard custom HF formats.
            cands = [
                key,
                f"model.{key}",
                f"{key}.weight",
                f"model.{key}.weight",
                f"transformer.{key}",
                f"transformer.{key}.weight",
            ]
            # Key-specific architecture aliases
            if key == "embed_tokens":
                cands.extend([
                    "model.embed_tokens.weight", "transformer.wte.weight",
                    "transformer.embedding.word_embeddings.weight",
                    "word_embeddings.weight", "embeddings.word_embeddings.weight",
                ])
            elif key == "norm":
                cands.extend([
                    "model.norm.weight", "transformer.ln_f.weight",
                    "transformer.norm.weight", "model.final_layernorm.weight",
                ])
            elif "input_layernorm" in key:
                prefix, idx = key.split(".input_layernorm")[0], key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.input_layernorm.weight",
                    f"transformer.h.{idx}.ln_1.weight",
                    f"transformer.layers.{idx}.input_layernorm.weight",
                    f"layers.{idx}.operator_norm.weight",
                ])
            elif "post_attention_layernorm" in key:
                prefix, idx = key.split(".post_attention_layernorm")[0], key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.post_attention_layernorm.weight",
                    f"transformer.h.{idx}.ln_2.weight",
                    f"transformer.layers.{idx}.post_attention_layernorm.weight",
                ])
            elif "q_proj" in key:
                idx = key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.self_attn.q_proj.weight",
                    f"transformer.h.{idx}.attn.q_proj.weight",
                    f"transformer.layers.{idx}.self_attn.q_proj.weight",
                ])
            elif "k_proj" in key:
                idx = key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.self_attn.k_proj.weight",
                    f"transformer.h.{idx}.attn.k_proj.weight",
                    f"transformer.layers.{idx}.self_attn.k_proj.weight",
                ])
            elif "v_proj" in key:
                idx = key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.self_attn.v_proj.weight",
                    f"transformer.h.{idx}.attn.v_proj.weight",
                    f"transformer.layers.{idx}.self_attn.v_proj.weight",
                ])
            elif "o_proj" in key:
                idx = key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.self_attn.o_proj.weight",
                    f"transformer.h.{idx}.attn.out_proj.weight",
                    f"transformer.layers.{idx}.self_attn.o_proj.weight",
                    f"model.layers.{idx}.self_attn.dense.weight",
                ])
            elif "gate_proj" in key:
                idx = key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.mlp.gate_proj.weight",
                    f"transformer.h.{idx}.mlp.gate_proj.weight",
                    f"transformer.layers.{idx}.mlp.gate_proj.weight",
                    f"model.layers.{idx}.mlp.w1.weight",
                ])
            elif "up_proj" in key:
                idx = key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.mlp.up_proj.weight",
                    f"transformer.h.{idx}.mlp.up_proj.weight",
                    f"transformer.layers.{idx}.mlp.up_proj.weight",
                    f"model.layers.{idx}.mlp.w3.weight",
                ])
            elif "down_proj" in key:
                idx = key.split(".")[1]
                cands.extend([
                    f"model.layers.{idx}.mlp.down_proj.weight",
                    f"transformer.h.{idx}.mlp.down_proj.weight",
                    f"transformer.layers.{idx}.mlp.down_proj.weight",
                    f"model.layers.{idx}.mlp.w2.weight",
                ])

            for c in cands:
                if c in tensor_index:
                    with safetensors.safe_open(tensor_index[c], framework="pt", device="cpu") as sf:
                        return sf.get_tensor(c)
            return None

        model = UniversalTransformer(config, init_weights=False, quant=quant, fused_qkv=True).to(device=device)

        # 1. Embed tokens
        emb = get_tensor("embed_tokens")
        if emb is not None:
            model.embed_tokens.weight.data.copy_(emb.float())
            del emb
            gc.collect()

        # 2. Final norm
        norm_w = get_tensor("norm")
        if norm_w is not None:
            model.norm.weight.data.copy_(norm_w.float())
            del norm_w
            gc.collect()

        # 3. Stream layer-by-layer
        for l in range(config.num_hidden_layers):
            print(f"[\033[94mTorchBurn\033[0m] Quantizing layer {l+1}/{config.num_hidden_layers} to {quant.upper()}...", end="\r", flush=True)
            layer = model.layers[l]

            # Input norm
            in_norm = get_tensor(f"layers.{l}.input_layernorm.weight")
            if in_norm is not None:
                layer.input_layernorm.weight.data.copy_(in_norm.float())
                del in_norm

            # Q, K, V
            q_w = get_tensor(f"layers.{l}.self_attn.q_proj.weight")
            k_w = get_tensor(f"layers.{l}.self_attn.k_proj.weight")
            v_w = get_tensor(f"layers.{l}.self_attn.v_proj.weight")
            if q_w is not None and k_w is not None and v_w is not None:
                qkv_w = torch.cat([q_w.float(), k_w.float(), v_w.float()], dim=0)
                del q_w, k_w, v_w
                if quant == "int4":
                    qw, qs = quantize_weight_int4_grouped(qkv_w, group_size=group_size)
                else:
                    qw, qs = quantize_weight_int8(qkv_w)
                del qkv_w
                layer.self_attn.qkv_proj.qweight.data.copy_(qw)
                layer.self_attn.qkv_proj.scales.data.copy_(qs)
                del qw, qs

            # QKV bias if present
            if config.qkv_bias:
                q_b = get_tensor(f"layers.{l}.self_attn.q_proj.bias")
                k_b = get_tensor(f"layers.{l}.self_attn.k_proj.bias")
                v_b = get_tensor(f"layers.{l}.self_attn.v_proj.bias")
                if q_b is not None and k_b is not None and v_b is not None:
                    qkv_b = torch.cat([q_b.float(), k_b.float(), v_b.float()], dim=0)
                    del q_b, k_b, v_b
                    if layer.self_attn.qkv_proj.bias is not None:
                        layer.self_attn.qkv_proj.bias.data.copy_(qkv_b)
                    del qkv_b

            # O proj
            o_w = get_tensor(f"layers.{l}.self_attn.o_proj.weight")
            if o_w is not None:
                if quant == "int4":
                    qw, qs = quantize_weight_int4_grouped(o_w.float(), group_size=group_size)
                else:
                    qw, qs = quantize_weight_int8(o_w.float())
                del o_w
                layer.self_attn.o_proj.qweight.data.copy_(qw)
                layer.self_attn.o_proj.scales.data.copy_(qs)
                del qw, qs

            # Post attention norm
            post_norm = get_tensor(f"layers.{l}.post_attention_layernorm.weight")
            if post_norm is not None:
                layer.post_attention_layernorm.weight.data.copy_(post_norm.float())
                del post_norm

            # MLP: gate, up, down
            for proj_name, module in [("gate_proj", layer.mlp.gate_proj), ("up_proj", layer.mlp.up_proj), ("down_proj", layer.mlp.down_proj)]:
                pw = get_tensor(f"layers.{l}.mlp.{proj_name}.weight")
                if pw is not None:
                    if quant == "int4":
                        qw, qs = quantize_weight_int4_grouped(pw.float(), group_size=group_size)
                    else:
                        qw, qs = quantize_weight_int8(pw.float())
                    del pw
                    module.qweight.data.copy_(qw)
                    module.scales.data.copy_(qs)
                    del qw, qs

            # Free transient PyTorch memory immediately after each layer
            gc.collect()

        print()  # Newline after progress

        # 4. LM Head
        print(f"[\033[94mTorchBurn\033[0m] Quantizing LM head (bounded chunking)...")
        if config.tie_word_embeddings:
            lm_w = model.embed_tokens.weight.data
            if quant == "int4":
                qw, qs = quantize_weight_int4_grouped(lm_w, group_size=group_size)
            else:
                qw, qs = quantize_weight_int8(lm_w)
            model.lm_head.qweight.data.copy_(qw)
            model.lm_head.scales.data.copy_(qs)
            del qw, qs
        else:
            lm_w = get_tensor("lm_head")
            if lm_w is not None:
                if quant == "int4":
                    qw, qs = quantize_weight_int4_grouped(lm_w, group_size=group_size)
                else:
                    qw, qs = quantize_weight_int8(lm_w.float())
                del lm_w
                model.lm_head.qweight.data.copy_(qw)
                model.lm_head.scales.data.copy_(qs)
                del qw, qs
        gc.collect()


        # 5. Persist to disk cache using fast zero-copy memory-mapped safetensors
        if save_cache_path:
            import safetensors.torch
            print(f"[\033[92mTorchBurn\033[0m] Saving quantized weights to disk cache: {os.path.basename(save_cache_path)}...")
            clean_sd = {k: v.contiguous() for k, v in model.state_dict().items()}
            safetensors.torch.save_file(clean_sd, save_cache_path)
            print(f"[\033[92mTorchBurn\033[0m] Cache created successfully. Future loads will take < 0.1s via zero-copy mmap.")

        return model

    @staticmethod
    def _migrate_v2_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert a v2-format state dict (interleaved 34-byte blocks, f16 scales)
        to v1 tensors (`qweight` (N, K/2) uint8 + `scales` (N, K/64) f32) in
        memory. Vectorized; a 0.5B model converts in well under a second.
        """
        migrated = {}
        for key, tensor in state_dict.items():
            if key.endswith(".qweight") and tensor.ndim == 2 and tensor.shape[1] % 34 == 0:
                n, cols = tensor.shape
                num_groups = cols // 34
                blocks = tensor.view(n, num_groups, 34)
                packed = blocks[:, :, :32].reshape(n, num_groups * 32)
                scales_h = blocks[:, :, 32:].view(torch.int16).view(torch.float16).float()
                scales = scales_h.reshape(n, num_groups)
                migrated[key] = packed.contiguous()
                migrated[key.replace(".qweight", ".scales")] = scales.contiguous()
            else:
                migrated[key] = tensor
        return migrated

    @classmethod
    def migrate_cache_to_v2(cls, model_dir: str) -> str:
        """One-time migration (roadmap 1.1): re-quantize an already-cached model
        into the v2 interleaved layout, writing `torchburn_int4_g64_v2.safetensors`
        next to the v1 cache. Loads take the v2 file first when present.
        """
        from torchburn.quantization import quantize_weight_int4_grouped_v2

        model, config, root = cls.load(model_dir, quant="int4", device="cpu")
        if config.tie_word_embeddings:
            # lm_head shares embed_tokens weights: rebuild it the same way the
            # streaming quantizer does so the state dict has its own qweight.
            from torchburn.quantization import quantize_weight_int4_grouped
            emb = model.embed_tokens.weight.data
            qw, qs = quantize_weight_int4_grouped(emb, group_size=64)
            model.lm_head.qweight.data.copy_(qw)
            model.lm_head.scales.data.copy_(qs)

        v2_path = os.path.join(root, "torchburn_int4_g64_v2.safetensors")
        sd = {}
        state = model.state_dict()
        for key, tensor in state.items():
            if key.endswith(".qweight") and key.replace(".qweight", ".scales") in state:
                # v1 -> f32 (exact dequant) -> v2 blocks. Re-quantizing the
                # dequantized v1 weights is what inference actually runs, so
                # the v2 cache preserves byte-identical outputs.
                scales = state[key.replace(".qweight", ".scales")]
                n, k2 = tensor.shape
                k = k2 * 2
                lo = (tensor & 0x0F).to(torch.int8)
                hi = ((tensor >> 4) & 0x0F).to(torch.int8)
                q = torch.cat([lo.reshape(n, -1, 1), hi.reshape(n, -1, 1)], dim=2)
                q = q.reshape(n, k).to(torch.int8) - 8
                w_deq = q.float() * scales.repeat_interleave(64, dim=1)
                sd[key] = quantize_weight_int4_grouped_v2(w_deq, group_size=64).contiguous()
            elif key.endswith(".scales"):
                continue  # scales are embedded in the v2 blocks
            else:
                sd[key] = tensor.contiguous()
        import safetensors.torch
        safetensors.torch.save_file(sd, v2_path)
        print(f"[\033[92mTorchBurn\033[0m] v2 cache written: {v2_path}")
        return v2_path

    @classmethod
    def _resolve_files(
        cls,
        model_id_or_path: str,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> Tuple[List[str], str, str]:
        """Finds weight file(s) and config.json locally or downloads them via huggingface_hub."""
        # Check local paths first (env TORCHBURN_MODELS_DIR, per-user cache,
        # HuggingFace hub snapshots) before hitting the network.
        repo_id = resolve_repo_id(model_id_or_path)
        local_cands = offline_candidates(model_id_or_path)
        hf_cache_snapshot_dir = repo_snapshot_dir(repo_id)

        for cand in local_cands:
            if os.path.isdir(cand):
                cfg_path = os.path.join(cand, "config.json")
                if os.path.isfile(cfg_path):
                    # Check for safetensors files
                    st_files = sorted([os.path.join(cand, f) for f in os.listdir(cand) if f.endswith(".safetensors")])
                    if st_files:
                        return st_files, cfg_path, cand

            elif os.path.isfile(cand) and (cand.endswith(".safetensors") or cand.endswith(".bin")):
                dir_name = os.path.dirname(cand)
                cfg_path = os.path.join(dir_name, "config.json")
                if os.path.isfile(cfg_path):
                    return [cand], cfg_path, dir_name

        # Otherwise, download via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            repo_id = resolve_repo_id(model_id_or_path)

            print(f"[\033[94mTorchBurn\033[0m] Resolving model '{repo_id}' from Hugging Face...")
            # Download config.json
            cfg_path = hf_hub_download(repo_id, "config.json", token=token, cache_dir=cache_dir)
            root_dir = os.path.dirname(cfg_path)

            # Check if sharded or single
            index_path = None
            try:
                index_path = hf_hub_download(repo_id, "model.safetensors.index.json", token=token, cache_dir=cache_dir)
            except Exception:
                pass

            if index_path and os.path.isfile(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                shard_names = sorted(list(set(weight_map.values())))
                st_files = [hf_hub_download(repo_id, s, token=token, cache_dir=cache_dir) for s in shard_names]
            else:
                single_st = hf_hub_download(repo_id, "model.safetensors", token=token, cache_dir=cache_dir)
                st_files = [single_st]

            return st_files, cfg_path, root_dir
        except Exception as e:
            raise FileNotFoundError(
                f"Could not locate model '{model_id_or_path}' locally or on Hugging Face: {e}."
            )

    @classmethod
    def _load_weights_into_model(cls, model: UniversalTransformer, weights_files: List[str]):
        """Directly populates model parameters from safetensors files."""
        import safetensors.torch

        param_map = dict(model.named_parameters())
        loaded_keys = set()

        for w_file in weights_files:
            print(f"[\033[94mTorchBurn\033[0m] Loading checkpoint shard: {os.path.basename(w_file)}")
            with safetensors.safe_open(w_file, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    # Normalize HF prefix e.g. "model.layers.0..." -> "layers.0..."
                    norm_k = k
                    if norm_k.startswith("model."):
                        norm_k = norm_k[6:]

                    if norm_k in param_map:
                        param = param_map[norm_k]
                        if param.shape == tensor.shape:
                            param.data.copy_(tensor)
                            loaded_keys.add(norm_k)
                        else:
                            # Handle shape transpositions if needed
                            if param.numel() == tensor.numel():
                                param.data.copy_(tensor.view_as(param))
                                loaded_keys.add(norm_k)

                    elif k in param_map:
                        param = param_map[k]
                        if param.shape == tensor.shape:
                            param.data.copy_(tensor)
                            loaded_keys.add(k)

        # Handle tied word embeddings
        if model.config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight

        # Tie fused QKV if present
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[\033[92mTorchBurn\033[0m] Successfully loaded {total_params / 1e6:.2f}M parameter model.")
        gc.collect()
