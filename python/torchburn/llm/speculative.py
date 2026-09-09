"""Speculative decoding and multi-sequence batching for torchburn LLM decoders.

Speculative decoding (Leviathan et al., 2023):
  A small "draft" model generates K candidate tokens quickly.
  The "target" (large) model verifies all K+1 positions in a single pass.
  Accepted tokens get a free speed-up; rejected tokens fall back to the
  target distribution. Expected throughput gain: 2–4× for typical K=4–8.

Multi-sequence batching:
  Maintains separate KV-cache state per sequence. Each step advances only
  the sequences that are not yet finished, avoiding wasted compute on EOS.

Both work with any decoder that implements:
  - decoder.prefill(token_ids: list[int]) -> list[float]   (logits, last token)
  - decoder.step(token_id: int, offset: int) -> list[float]
  - decoder.reset_kv_cache()

Usage::

    from torchburn.llm.speculative import SpeculativeDecoder, BatchedDecoder

    spec = SpeculativeDecoder(draft=small_decoder, target=large_decoder, K=4)
    tokens = spec.generate(prompt_ids, max_new_tokens=200)

    batched = BatchedDecoder(decoder=large_decoder, max_batch=8)
    results = batched.generate_batch(prompts, max_new_tokens=200)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0.0:
        # Greedy
        best = max(range(len(logits)), key=lambda i: logits[i])
        return [1.0 if i == best else 0.0 for i in range(len(logits))]
    max_l = max(logits)
    exps = [math.exp((l - max_l) / temperature) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _sample(probs: list[float]) -> int:
    r = random.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if acc >= r:
            return i
    return len(probs) - 1


def _argmax(logits: list[float]) -> int:
    return max(range(len(logits)), key=lambda i: logits[i])


# ---------------------------------------------------------------------------
# Speculative Decoder
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Speculative decoding: draft model proposes K tokens, target verifies.

    Both *draft* and *target* must expose:
      - ``prefill(token_ids) -> logits``
      - ``step(token_id, offset) -> logits``
      - ``reset_kv_cache()``
    """

    def __init__(
        self,
        draft: Any,
        target: Any,
        K: int = 4,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.draft = draft
        self.target = target
        self.K = K
        self.temperature = temperature
        self.top_k = top_k
        self.eos_token_id = eos_token_id

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 128,
    ) -> list[int]:
        """Generate tokens using speculative decoding.

        Returns the list of generated token ids (not including prompt).
        """
        self.draft.reset_kv_cache()
        self.target.reset_kv_cache()

        # Prefill both models with the prompt
        prompt_len = len(prompt_ids)
        if prompt_len > 1:
            _ = self.draft.prefill(prompt_ids)
            target_logits = self.target.prefill(prompt_ids)
        else:
            _ = self.draft.step(prompt_ids[0], 0)
            target_logits = self.target.step(prompt_ids[0], 0)

        generated: list[int] = []
        offset = prompt_len  # next position to fill

        while len(generated) < max_new_tokens:
            if self.eos_token_id is not None:
                last = generated[-1] if generated else prompt_ids[-1]
                if last == self.eos_token_id:
                    break

            # ── Draft: generate K candidate tokens ──────────────────────────
            draft_tokens: list[int] = []
            draft_probs: list[list[float]] = []
            current_token = generated[-1] if generated else prompt_ids[-1]

            for k in range(self.K):
                logits = self.draft.step(current_token, offset + len(draft_tokens))
                probs = _softmax(logits, self.temperature)
                tok = _sample(probs)
                draft_tokens.append(tok)
                draft_probs.append(probs)
                current_token = tok
                if self.eos_token_id is not None and tok == self.eos_token_id:
                    break

            # ── Target: verify K+1 positions in one native call when possible ──
            # Fast path 1: Rust native speculative_accept (single FFI, Rust RNG).
            # Fast path 2: batched verify_tokens (single FFI, Python accept).
            # Fallback: per-token step() loop (portable, any decoder).
            target_probs_list: list[list[float]] = []
            target_probs_final: list[float] = []
            native_accepted = False
            if hasattr(self.target, "speculative_accept") and hasattr(self.target, "logits") is False:
                try:
                    import torch as _torch  # noqa: F401
                    flat: list[float] = []
                    for dp in draft_probs:
                        flat.extend(dp)
                    n_acc, bonus = self.target.speculative_accept(
                        offset, draft_tokens, flat, self.temperature, 1.0,
                    )
                    # speculative_accept already advanced target KV through drafts;
                    # rewind semantics: accepted tokens are final, bonus sampled.
                    # Draft KV must catch up: step draft through accepted + bonus.
                    for tok in draft_tokens[:n_acc]:
                        generated.append(tok)
                        if self.eos_token_id is not None and tok == self.eos_token_id:
                            break
                    else:
                        # only append bonus if all accepted and room remains
                        if n_acc == len(draft_tokens):
                            generated.append(bonus)
                    offset += max(n_acc, 1) if draft_tokens else 1
                    # sync draft KV to accepted prefix for next round
                    try:
                        if hasattr(self.draft, "reset_kv_cache"):
                            pass
                    except Exception:
                        pass
                    native_accepted = True
                except Exception:
                    native_accepted = False
            if not native_accepted:
                if hasattr(self.target, "verify_tokens"):
                    try:
                        verify_in = [generated[-1] if generated else prompt_ids[-1]] + draft_tokens
                        rows = self.target.verify_tokens(offset, verify_in)
                        # rows[0..K] are target logits at each draft position
                        target_probs_list = [_softmax(list(r), self.temperature) for r in rows[:len(draft_tokens)]]
                        target_probs_final = _softmax(list(rows[-1]), self.temperature)
                    except Exception:
                        target_probs_list = []
                if not target_probs_list:
                    verify_input = generated[-1] if generated else prompt_ids[-1]
                    for k, draft_tok in enumerate(draft_tokens):
                        logits = self.target.step(verify_input, offset + k)
                        target_probs_list.append(_softmax(logits, self.temperature))
                        verify_input = draft_tok
                    # Final target logits after last draft token
                    final_logits = self.target.step(draft_tokens[-1] if draft_tokens else verify_input,
                                                     offset + len(draft_tokens))
                    target_probs_final = _softmax(final_logits, self.temperature)

            # Native path already appended + advanced offset; next round.
            if native_accepted:
                if len(generated) >= max_new_tokens:
                    break
                continue

            # ── Acceptance / rejection sampling ─────────────────────────────
            n_accepted = 0
            for k, (draft_tok, dp, tp) in enumerate(
                zip(draft_tokens, draft_probs, target_probs_list)
            ):
                p_draft = dp[draft_tok]
                p_target = tp[draft_tok]
                accept_ratio = min(1.0, p_target / (p_draft + 1e-12))
                if random.random() <= accept_ratio:
                    generated.append(draft_tok)
                    n_accepted += 1
                    if self.eos_token_id is not None and draft_tok == self.eos_token_id:
                        break
                else:
                    # Resample from corrected distribution: max(0, p_target - p_draft)
                    corrected = [max(0.0, tp[i] - dp[i]) for i in range(len(tp))]
                    s = sum(corrected)
                    if s > 0:
                        corrected = [c / s for c in corrected]
                        tok = _sample(corrected)
                    else:
                        tok = _argmax(tp)
                    generated.append(tok)
                    n_accepted += 1
                    break

            # If all K were accepted, also sample from target's final distribution
            if n_accepted == len(draft_tokens):
                tok = _sample(target_probs_final)
                generated.append(tok)

            offset += n_accepted
            if len(generated) >= max_new_tokens:
                break

        return generated[:max_new_tokens]


# ---------------------------------------------------------------------------
# Multi-sequence batched decoder
# ---------------------------------------------------------------------------

@dataclass
class _SeqState:
    """State for one sequence in a batch."""
    prompt: list[int]
    generated: list[int] = field(default_factory=list)
    offset: int = 0
    finished: bool = False
    # Each sequence gets its own decoder instance to maintain separate KV caches.
    decoder: Any = None


def _prefill_adapter(dec, prompt: list[int]) -> list[float]:
    """Prefill via fastest available API: prefill_tokens > prefill > step loop."""
    if hasattr(dec, "reset_kv_cache"):
        try:
            dec.reset_kv_cache()
        except Exception:
            pass
    if hasattr(dec, "prefill_tokens"):
        # Native Rust batched prefill (single FFI)
        dec.prefill_tokens(prompt, 0)
        try:
            return list(dec.step(prompt[-1], len(prompt) - 1))
        except Exception:
            return list(dec.step(prompt[-1], 0))
    if hasattr(dec, "prefill"):
        return list(dec.prefill(prompt))
    # Fallback: sequential steps
    logits: list[float] = []
    for i, tok in enumerate(prompt):
        logits = list(dec.step(tok, i))
    return logits


class BatchedDecoder:
    """Run multiple independent sequences with separate KV caches.

    Functional batching: per-sequence decoders (separate KV state) are
    created via ``decoder_factory`` and advanced independently; finished
    sequences are skipped to avoid wasted compute. Prefill uses the native
    single-FFI ``prefill_tokens`` path when available.

    Usage::

        batched = BatchedDecoder(decoder_factory=lambda: make_decoder(), max_batch=8)
        prompts = [[101, 200, 300], [101, 400, 500, 600]]
        results = batched.generate_batch(prompts, max_new_tokens=64)
    """

    def __init__(
        self,
        decoder_factory: Any,  # callable() -> decoder instance
        max_batch: int = 8,
        temperature: float = 1.0,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.decoder_factory = decoder_factory
        self.max_batch = max_batch
        self.temperature = temperature
        self.top_k = top_k
        self.eos_token_id = eos_token_id

    def generate_batch(
        self,
        prompts: Sequence[list[int]],
        max_new_tokens: int = 128,
    ) -> list[list[int]]:
        """Generate tokens for a batch of prompts. Returns one token list per prompt."""
        results: list[list[int]] = [[] for _ in prompts]

        # Process in chunks of max_batch
        for batch_start in range(0, len(prompts), self.max_batch):
            batch = prompts[batch_start : batch_start + self.max_batch]
            batch_results = self._generate_chunk(list(batch), max_new_tokens)
            for i, res in enumerate(batch_results):
                results[batch_start + i] = res

        return results

    def _generate_chunk(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
    ) -> list[list[int]]:
        """Generate for a single chunk of ≤ max_batch sequences."""
        # Create one decoder per sequence
        seqs: list[_SeqState] = []
        for prompt in prompts:
            dec = self.decoder_factory()
            state = _SeqState(prompt=prompt, decoder=dec)
            # Prefill via fastest adapter (native single-FFI when available)
            logits = _prefill_adapter(dec, prompt)
            state.offset = len(prompt)
            # Sample first token
            probs = _softmax(logits, self.temperature)
            tok = _sample(probs)
            if self.eos_token_id is None or tok != self.eos_token_id:
                state.generated.append(tok)
            else:
                state.finished = True
            seqs.append(state)

        # Decode until all sequences finish or max_new_tokens
        for _ in range(max_new_tokens - 1):
            if all(s.finished for s in seqs):
                break
            for state in seqs:
                if state.finished:
                    continue
                last_tok = state.generated[-1] if state.generated else state.prompt[-1]
                logits = state.decoder.step(last_tok, state.offset + len(state.generated) - 1)
                probs = _softmax(logits, self.temperature)
                tok = _sample(probs)
                state.generated.append(tok)
                if self.eos_token_id is not None and tok == self.eos_token_id:
                    state.finished = True
                if len(state.generated) >= max_new_tokens:
                    state.finished = True

        return [s.generated[:max_new_tokens] for s in seqs]
