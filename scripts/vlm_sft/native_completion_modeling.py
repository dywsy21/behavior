"""Separate status encoder/decoder; motion messages are byte-compatible with H09Z."""
import hashlib
import time

from modeling import collate, eos_id, load_model, supervised_loss
from native_completion_protocol import messages, validate_query, vocabulary


def encode(processor, row, images, *, supervised):
    import torch
    validate_query(row, supervised=supervised)
    prefix = processor.apply_chat_template(messages(row, images), tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    if prefix["input_ids"].shape[1] > 1800: raise ValueError("Registered context budget")
    if not supervised: return prefix
    response = processor.tokenizer.encode(row["target"], add_special_tokens=False) + [eos_id(processor)]
    if not 2 <= len(response) <= 16 or any(type(t) is not int or t < 0 for t in response):
        raise ValueError("Unexpected response encoding")
    result = {k: v.clone() for k, v in prefix.items()}; old = prefix["input_ids"].shape[1]
    result["input_ids"] = torch.cat([prefix["input_ids"], torch.tensor([response], dtype=torch.long)], dim=1)
    for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
        if key in result:
            result[key] = torch.cat([result[key], torch.full((1, len(response)),
                1 if key == "attention_mask" else 0, dtype=result[key].dtype)], dim=1)
    labels = torch.full_like(result["input_ids"], -100); labels[:, old:] = result["input_ids"][:, old:]
    result["labels"] = labels
    if not torch.equal(result["input_ids"][:, :old], prefix["input_ids"]):
        raise RuntimeError("Train/inference prefix mismatch")
    return result


def check_mask(processor, row, encoded):
    labels = encoded["labels"][encoded["labels"] != -100]
    expected = processor.tokenizer.encode(row["target"], add_special_tokens=False) + [eos_id(processor)]
    if labels.tolist() != expected or processor.tokenizer.decode(labels.tolist(), skip_special_tokens=True) != row["target"]:
        raise RuntimeError("Real response-only/EOS label gate failed")
    if (encoded["labels"][:, :-len(expected)] != -100).any():
        raise RuntimeError("Prompt or image token was supervised")


def decode(model, processor, encoded, kind):
    import torch
    inputs = {k: v.to("cuda") for k, v in encoded.items() if k != "labels"}
    prefix = inputs["input_ids"].shape[1]; trie = {}; eos = eos_id(processor)
    for token in vocabulary(kind):
        node = trie
        for value in processor.tokenizer.encode(token, add_special_tokens=False): node = node.setdefault(value, {})
        node[eos] = {}
    def allowed(_batch, ids):
        node = trie
        for value in ids[prefix:].tolist(): node = node[value]
        return list(node) or [eos]
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=20, do_sample=False, use_cache=True,
            prefix_allowed_tokens_fn=allowed, eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id)
    torch.cuda.synchronize()
    answer = processor.tokenizer.decode(result[0, prefix:], skip_special_tokens=True).strip()
    if answer not in vocabulary(kind): raise RuntimeError("Constrained query vocabulary violated")
    return {"prediction": answer, "latency_s": time.perf_counter()-start, "input_tokens": prefix,
        "output_tokens": int(result.shape[1]-prefix),
        "input_ids_sha256": hashlib.sha256(inputs["input_ids"].cpu().numpy().tobytes()).hexdigest()}
