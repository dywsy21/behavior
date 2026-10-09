"""Restore masked planner-only grammar constants at inference.

Only UNKNOWN and false fields are forced: neither is a physical claim.
Decision, parent, skills and memory remain model-generated and strictly parsed.
This is a local deployment repair, not recovered original training source.
"""
from contextlib import contextmanager
from .memlite_planner_fields import split_planner_event_fields


@contextmanager
def planner_only_format_constants(ar_helper, processor, *, batch_size, hl_end_id, enabled):
    receipt = {"enabled": bool(enabled), "implementation": "local_masked_constant_repair_v1",
               "forced_tokens": 0}
    if not enabled:
        yield receipt
        return
    text_processor = processor.modality_processors["text"]
    def encode(text):
        ids, _, _ = text_processor.process(text, is_masked=False, training=False,
            tokenizer=processor.tokenizer, sample_dict={})
        return list(ids)
    prefix = encode("Previous outcome: UNKNOWN") + encode("|")
    suffix = encode("Task complete: false") + encode("|") + [int(hl_end_id)]
    pending = [list(prefix) for _ in range(batch_size)]
    suffix_started = [False] * batch_size
    original = ar_helper._sample
    had_instance_value = "_sample" in ar_helper.__dict__
    instance_value = ar_helper.__dict__.get("_sample")

    def sample(logits, prev_tokens=None, **kwargs):
        if logits.shape[0] != batch_size:
            raise ValueError("Planner format batch size mismatch")
        for i in range(batch_size):
            if pending[i] or suffix_started[i] or prev_tokens is None:
                continue
            text = processor.tokenizer.decode(prev_tokens[i].tolist(), skip_special_tokens=False)
            try:
                fields = split_planner_event_fields(text)
            except ValueError:
                continue  # Partial JSON during autoregressive decoding is normal.
            if text.endswith("|") and len(fields) == 5:
                pending[i] = list(suffix)
                suffix_started[i] = True
        result = original(logits, prev_tokens=prev_tokens, **kwargs)
        for i in range(batch_size):
            if pending[i]:
                result[i] = pending[i].pop(0)
                receipt["forced_tokens"] += 1
        return result

    ar_helper._sample = sample
    try:
        yield receipt
    finally:
        if had_instance_value:
            ar_helper._sample = instance_value
        else:
            del ar_helper._sample
