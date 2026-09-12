"""Real checkpoint parity and bounded training-instance adapter-gradient smoke.

Updates are confined to an in-memory candidate adapter. This is not the formal
5000-step run, not high-level training and not a task success-rate evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from memlite_fm_v11_common import config, sha, source_hashes, TASK5_RUN
from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint


def batch(samples, device):
    return {"samples": [deepcopy(s["samples"]) for s in samples],
            "pixel_values": {key: torch.stack([s["pixel_values"][key] for s in samples]).to(device)
                             for key in samples[0]["pixel_values"]},
            "actions": torch.stack([s["action"] for s in samples]).to(device),
            "action_pad_masks": torch.stack([s["action_is_pad"] for s in samples]).to(device),
            "action_dim_is_pad": torch.stack([s["action_dim_is_pad"] for s in samples]).to(device)}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def forbid_action_codec(policy):
    """Guard the real VQ backend, not nonexistent wrapper encode/decode APIs."""
    with ExitStack() as stack:
        backend = policy.action_tokenizer.action_tokenizer
        for method in ("encode", "decode"):
            stack.enter_context(patch.object(backend, method,
                side_effect=AssertionError(f"FM must not call action codec {method}")))
        yield


def compare_parameters(low, old):
    a, b = dict(low.model.named_parameters()), dict(old.model.named_parameters())
    if a.keys() != b.keys():
        raise ValueError(f"Baseline model structure changed: new={a.keys()-b.keys()}, old={b.keys()-a.keys()}")
    mismatches = [key for key in a if a[key].shape != b[key].shape or not torch.equal(a[key], b[key])]
    if mismatches:
        raise ValueError(f"Frozen baseline weights changed: {mismatches[:10]}")
    return len(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--optimizer-steps", type=int, default=12)
    args = ap.parse_args()
    if args.output.exists() or not 3 <= args.optimizer_steps <= 32:
        raise ValueError("Use a new output and a bounded 3..32-step smoke")
    torch.set_num_threads(2)
    device = "cuda:0"
    torch.cuda.set_device(device)
    cfg = config()
    manifest = json.loads((args.data / "manifest.json").read_text())
    if not manifest.get("passed"):
        raise ValueError("Real source data gate did not pass")
    checkpoint = TASK5_RUN / "checkpoints/step_5000.pt"
    set_seed(1708)
    low = load_model_from_checkpoint(cfg.model.model_arch, str(checkpoint),
        device=device, extra_prefixes=["normalizer."], eval_mode=False)
    low.apply_fp32_params()
    low.eval()
    old_cfg = OmegaConf.load(TASK5_RUN / ".hydra/config.yaml")
    old_cfg.model.model_arch.attn_implementation = "sdpa"
    old = load_model_from_checkpoint(old_cfg.model.model_arch, str(checkpoint),
        device=device, extra_prefixes=["normalizer."], eval_mode=False)
    old.apply_fp32_params()
    old.requires_grad_(False).eval()
    compared = compare_parameters(low, old)
    named = dict(low.named_parameters())
    trainable = {key: value for key, value in named.items() if value.requires_grad}
    assert trainable and all(key.startswith("fm_intent_adapter.") for key in trainable)
    rows = manifest["rows"]
    load = lambda row: torch.load(args.data / row["file"], weights_only=False, map_location="cpu")["sample"]
    eval_rows = []
    for task in range(5):
        candidates = [r for r in rows if r["split"] == "eval" and r["task_id"] == task]
        eval_rows.extend(candidates[:2])
    assert len(eval_rows) == 10
    parity = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in eval_rows:
            data = batch([load(row)], device)
            baseline = low.baseline_samples(data["samples"])
            old_state = old.prefill(baseline, data["pixel_values"])
            new_state = low.prefill(baseline, data["pixel_values"])
            assert torch.equal(old_state.input_ids, new_state.input_ids)
            assert torch.equal(old_state.attention_mask, new_state.attention_mask)
            assert torch.equal(old_state.position_ids, new_state.position_ids)
            seed = 1800 + row["id"]
            set_seed(seed)
            expected = old.model.inference_fm(attention_mask=old_state.attention_mask,
                pixel_values=old_state.pixel_values, past_key_values=old_state.kv_cache,
                action_dim_is_pad=data["action_dim_is_pad"], position_ids_override=old_state.position_ids,
                embodiment_types=["galaxea_r1pro"])
            outputs = {}
            for enabled in (False, True):
                low.intent_enabled = enabled
                set_seed(seed)
                with forbid_action_codec(low):
                    actual = low.generate_low_level_action(data["samples"], data["pixel_values"],
                        intent_text=[row["intent"]], action_dim_is_pad=data["action_dim_is_pad"])
                assert actual["selected_action_source"] == "fm"
                outputs[str(enabled)] = float((actual["action"]-expected).abs().max())
                torch.testing.assert_close(actual["action"], expected, rtol=0, atol=0)
            parity.append({"id": row["id"], "task_id": row["task_id"], "prefix_exact": True, "max_abs_diff": outputs})
            del old_state, new_state, expected, actual, data

    # Only reviewed TRAIN source IDs may enter a real optimizer update here.
    chosen = []
    for task in range(4):
        for kind in ("yaw_stop", "recovery"):
            chosen.append(next(r for r in rows if r["split"] == "train" and r["task_id"] == task and r["category"] == kind))
    assert len(chosen) == 8 and all(row["split"] == "train" for row in chosen)
    training = batch([load(row) for row in chosen], device)
    optimizer = torch.optim.AdamW(low.get_optim_param_groups(lr=1e-4, weight_decay=.01))
    low.train()
    low.intent_dropout = 0.0  # deterministic diagnostic, recipe dropout is separately tested
    low.intent_enabled = True
    losses = []
    for step in range(args.optimizer_steps):
        set_seed(314159)
        optimizer.zero_grad(set_to_none=True)
        with forbid_action_codec(low):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = low.forward_train(**training)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(value.grad is None for key, value in named.items() if not key.startswith("fm_intent_adapter."))
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in trainable.values())
        if step >= 2:
            assert low.fm_intent_adapter.token_projection.weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        optimizer.step()
        losses.append({"step": step+1, "loss": float(loss.detach()),
                       **{key: float(value) for key, value in metrics.items()}})
        print(json.dumps(losses[-1]), flush=True)
    compare_parameters(low, old)
    low.eval()
    row = chosen[0]
    probe = batch([load(row)], device)
    predictions = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tag, text in (("correct", row["intent"]), ("different", chosen[-1]["intent"]), ("disabled", row["intent"])):
            low.intent_enabled = tag != "disabled"
            set_seed(1709)
            predictions[tag] = low.generate_low_level_action(probe["samples"], probe["pixel_values"],
                intent_text=[text], action_dim_is_pad=probe["action_dim_is_pad"])["action"]
        state = old.prefill(low.baseline_samples(probe["samples"]), probe["pixel_values"])
        set_seed(1709)
        preserved = old.model.inference_fm(attention_mask=state.attention_mask, pixel_values=state.pixel_values,
            past_key_values=state.kv_cache, position_ids_override=state.position_ids,
            action_dim_is_pad=probe["action_dim_is_pad"], embodiment_types=["galaxea_r1pro"])
        torch.testing.assert_close(predictions["disabled"], preserved, rtol=0, atol=0)
        sensitivity = float((predictions["correct"]-predictions["different"]).abs().max())
        assert sensitivity > 0, "Intent adapter is decorative: actions do not depend on intent"
    result = {"passed": True, "checkpoint": str(checkpoint), "data_manifest_sha256": sha(args.data/"manifest.json"),
        "model_sources": {k: v for k, v in source_hashes(Path(__file__).resolve().parents[1]).items()
                          if k.startswith(("src/", "configs/"))},
        "frozen_low_parameters_compared": compared, "frozen_parameters_unchanged_after_updates": True,
        "trainable_parameters": list(trainable), "trainable_numel": sum(p.numel() for p in trainable.values()),
        "initial_parity": parity, "temporary_optimizer_steps": args.optimizer_steps,
        "optimizer_train_source_ids": [row["id"] for row in chosen], "losses": losses,
        "post_smoke_disabled_max_abs_diff": 0.0, "post_smoke_intent_sensitivity_max_abs": sensitivity,
        "formal_training": False, "task_success_evaluated": False}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"passed": True, "trainable_numel": result["trainable_numel"],
                      "frozen_parameters_unchanged": True, "intent_sensitivity": sensitivity}), flush=True)


if __name__ == "__main__":
    main()
