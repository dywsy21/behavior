"""Explicit static-format constrained AR, never target actions or FM fallback.

This is a separate inference variant, NOT evidence that the model learned its
syntax. All neural codebook payloads remain model predictions. Joint gripper
indices are constrained using the actual codec's radix and finite code count,
so its legacy safe=True decoder cannot silently clamp an invalid sequence.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Slot:
    kind: str
    group: str
    level: int = 0
    token_id: int | None = None
    payload_start: int = 0
    payload_index: int = 0
    payload_count: int = 0


class ActionSchemaSampler:
    def __init__(self, tokenizer, stop_token_id):
        serializer = tokenizer.serializer
        self.offset, self.codebook = int(tokenizer.action_token_begin_idx), int(tokenizer._codebook_size)
        self.end = int(tokenizer.action_token_end_idx)
        self.rule = getattr(tokenizer.action_tokenizer, "_rule_tokenizer", None)
        if serializer.rule_key_names and (self.rule is None or self.rule.num_tokens != serializer.rule_tokens_per_key
                or self.rule.vocab_size != self.codebook or not 1 <= self.rule.num_valid_sequences <= self.codebook ** self.rule.num_tokens):
            raise ValueError("Need exact rule radix/length/valid-sequence metadata; never guess gripper codes")
        if type(stop_token_id) is not int or not 0 <= stop_token_id < self.offset:
            raise ValueError("Require the actual single-token text action delimiter")
        self.slots = []
        for level in range(serializer.num_residuals):
            for key in serializer.nn_key_names:
                name = f"<{key}_{level}>" if serializer.max_residuals > 1 else f"<{key}>"
                marker = self.offset + int(serializer.group_marker_action_indices[name])
                self.slots.append(Slot("marker", key, level, marker))
                self.slots.extend(Slot("code", key, level, payload_index=index)
                                  for index in range(serializer.code_len))
        for key in serializer.rule_key_names:
            marker = self.offset + int(serializer.group_marker_action_indices[f"<{key}>"])
            self.slots.append(Slot("marker", key, token_id=marker))
            start = len(self.slots)
            self.slots.extend(Slot("rule", key, payload_start=start, payload_index=index,
                                   payload_count=serializer.rule_tokens_per_key)
                              for index in range(serializer.rule_tokens_per_key))
        markers = [slot.token_id for slot in self.slots if slot.kind == "marker"]
        if (not markers or len(set(markers)) != len(markers)
                or any(not self.offset + self.codebook <= token < self.end for token in markers)
                or not 1 <= len(self.slots) < 96):
            raise ValueError("Invalid, duplicate, or over-budget actual codec layout")
        self.slots.append(Slot("stop", "end", token_id=stop_token_id))
        self.generated, self.trace = [], []

    def allowed_range(self):
        index = len(self.generated)
        if index >= len(self.slots):
            raise RuntimeError("AR decoder continued beyond its complete static layout")
        slot = self.slots[index]
        if slot.kind in {"marker", "stop"}:
            return slot.token_id, slot.token_id + 1
        upper = self.codebook - 1
        if slot.kind == "rule":
            prefix = 0
            for token in self.generated[slot.payload_start:index]:
                digit = token - self.offset
                if not 0 <= digit < self.codebook:
                    raise RuntimeError("Invalid generated rule prefix")
                prefix = prefix * self.codebook + digit
            remaining = slot.payload_count - slot.payload_index - 1
            upper = min(upper, (self.rule.num_valid_sequences - 1
                        - prefix * self.codebook ** (remaining + 1)) // self.codebook ** remaining)
            if upper < 0:
                raise RuntimeError("Generated gripper prefix has no valid continuation")
        return self.offset, self.offset + upper + 1

    def sample(self, logits, original_sampler, **kwargs):
        if (logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < self.end
                or torch.isnan(logits).any() or torch.isposinf(logits).any()):
            raise RuntimeError("Schema decoder requires single-environment valid full-vocabulary logits")
        lower, upper = self.allowed_range()
        if not torch.isfinite(logits[:, lower:upper]).any():
            raise RuntimeError("Model masks every codec-legal token; refusing invented action scores")
        restricted = torch.full_like(logits, -torch.inf)
        restricted[:, lower:upper] = logits[:, lower:upper]
        sampled = original_sampler(restricted, **kwargs)
        if sampled.shape != (1,) or not lower <= int(sampled[0]) < upper:
            raise RuntimeError("Actual sampler returned a codec-illegal token")
        token = int(sampled[0])
        slot = self.slots[len(self.generated)]
        raw = logits[0].float()
        self.trace.append(dict(index=len(self.generated), kind=slot.kind, group=slot.group, level=slot.level,
            allowed_range=[lower, upper], unconstrained_argmax=int(raw.argmax()), selected_id=token,
            selected_raw_ce=float(torch.logsumexp(raw, dim=0) - raw[token]),
            selected_raw_rank=int((raw > raw[token]).sum()) + 1))
        self.generated.append(token)
        return sampled

    def require_complete(self):
        if len(self.generated) != len(self.slots):
            raise RuntimeError("AR generation ended before the full constrained codec layout")
        if self.rule is not None:
            for slot in self.slots:
                if slot.kind == "rule" and slot.payload_index == 0:
                    tokens = [token - self.offset for token in
                              self.generated[slot.payload_start:slot.payload_start + slot.payload_count]]
                    self.rule.decode(tokens, safe=False)
        return dict(complete=True, action_tokens=len(self.slots) - 1,
                    static_format_forced=True, payload_source="model_ar_logits", rule_safe_clamp=False,
                    raw_argmax_overruled=sum(row["unconstrained_argmax"] != row["selected_id"] for row in self.trace))


@contextmanager
def constrained_action_schema(policy):
    helper = policy.model.ar_helper
    if (policy.action_training.route != "ar" or policy.action_training.predicts_cot
            or helper.block_wise_autoregressive or getattr(helper, "_schema_sampling_active", False)):
        raise RuntimeError("Initial schema variant is one pure-AR, non-CoT, non-BAR call at a time")
    delimiter = policy.processor.tokenizer.encode("|", add_special_tokens=False)
    if len(delimiter) != 1 or delimiter[0] not in (policy._get_action_stop_token_ids() or []):
        raise RuntimeError("Codec delimiter does not match actual action stopping configuration")
    sampler = ActionSchemaSampler(policy.action_tokenizer, int(delimiter[0]))
    original, was_overridden = helper._sample, "_sample" in helper.__dict__
    helper._schema_sampling_active = True
    helper._sample = lambda logits, **kwargs: sampler.sample(logits, original, **kwargs)
    try:
        yield sampler
        sampler.require_complete()
    finally:
        if was_overridden:
            helper._sample = original
        else:
            del helper._sample
        del helper._schema_sampling_active
