"""Read-only, same-history teacher-forcing vs actual cached-AR diagnostics.

Teacher tokens are forced ONLY inside this offline diagnostic, never a deployed
actor or physics loop. A separate target-free call records real generation.
No optimizer, checkpoint publication, dataset release or simulation is allowed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from probe_action_training_gpu import architecture_for, prepare_action_updates, take_row
from probe_ar_execution_codec import publish
from train_action_method_probe import now, read
from train_fm_method_probe import BASE, PARENT, PARENT_SHA, verify_parent


def compare_logits(reference, actual, target):
    """CPU float32 statistics, before any sampling/penalty/filtering."""
    import torch
    import torch.nn.functional as F
    if (reference.ndim != 1 or reference.shape != actual.shape
            or not torch.isfinite(reference).all() or not torch.isfinite(actual).all()
            or not 0 <= target < reference.numel()):
        raise ValueError("Need finite, matching full-vocabulary logits and valid target")
    a, b = reference.float(), actual.float()
    label = torch.tensor([target], dtype=torch.long)
    return dict(target_id=target, full_argmax=int(a.argmax()), cached_argmax=int(b.argmax()),
        argmax_equal=bool(a.argmax() == b.argmax()), max_abs_diff=float((a - b).abs().max()),
        rms_diff=float((a - b).square().mean().sqrt()),
        full_target_ce=float(F.cross_entropy(a[None], label)),
        cached_target_ce=float(F.cross_entropy(b[None], label)),
        full_target_rank=int((a > a[target]).sum()) + 1,
        cached_target_rank=int((b > b[target]).sum()) + 1)


def verify_probe_checkpoint(policy, payload, kind):
    import torch
    if kind == "a4":
        verify_parent(policy, payload)
        return dict(step=2500, exact_entries=1138, lora_entries=192)
    expected, actual = payload["model_state_dict"], policy.state_dict()
    if (payload["step"] != 500 or set(expected) != set(actual) or len(actual) != 1138
            or sum("lora_" in key for key in actual) != 192):
        raise RuntimeError("AR500 full checkpoint schema changed")
    for name, value in actual.items():
        value, saved = value.detach().cpu(), expected[name]
        if (value.dtype != saved.dtype or value.shape != saved.shape or not torch.isfinite(value).all()
                or not torch.equal(value.contiguous().view(torch.uint8), saved.contiguous().view(torch.uint8))):
            raise RuntimeError("AR500 checkpoint not restored exactly: " + name)
    return dict(step=500, exact_entries=1138, lora_entries=192)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--checkpoint", choices=("a4", "ar500"), required=True)
    args = parser.parse_args()
    if args.output.parent != BASE or args.gpu not in range(4):
        raise ValueError("Use a new direct child run and one explicit robo GPU")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean, pinned worktree")
    checkpoint, checkpoint_sha = PARENT, PARENT_SHA
    if args.checkpoint == "ar500":
        run = BASE / "ar_a4_fulltrain_v2"
        spec, status = read(run / "method_spec.json"), read(run / "status.json")
        inspection = read(run / "formal/checkpoint_inspection.json")
        if (spec["route"] != "ar" or spec["parent_sha256"] != PARENT_SHA
                or spec["max_updates"] != 500 or status.get("state") != "complete"
                or not inspection.get("passed") or inspection.get("actual_updates") != 500):
            raise RuntimeError("The declared AR500 predecessor has not completed verification")
        checkpoint = run / "formal/checkpoints/step_500.pt"
        if inspection["checkpoint"] != str(checkpoint):
            raise RuntimeError("AR500 checkpoint path differs from its verified inspection")
        checkpoint_sha = inspection["checkpoint_sha256"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output.mkdir(exist_ok=False)
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.training.ar_training_methods import action_prefix_samples, ActionTrainingSettings

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Insufficient GPU headroom; do not evict other work")
    torch.cuda.set_per_process_memory_fraction(.4)
    config = INPUT / "diagnostic_processor_config.yaml"
    source_receipt = read(INPUT / "result.json")
    if (sha(checkpoint) != checkpoint_sha or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or source_receipt["processor_config_sha256"] != sha(config)
            or source_receipt["status"] != "complete" or not source_receipt["train_only"]):
        raise RuntimeError("Checkpoint/original train input identity changed")
    cfg = OmegaConf.load(config)
    settings = ActionTrainingSettings(route="ar")
    arch = architecture_for(cfg, settings)
    identity.update(start_time=now(), gpu=args.gpu, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        checkpoint_kind=args.checkpoint, original_input_sha256=INPUT_SHA, config_sha256=sha(config),
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), original_train_rows=2, maximum_forced_tokens_per_row=96,
        prefix_and_full_forwards_per_row=3, target_free_calls=2, optimizer_updates=0, simulator_controls=0,
        settings=settings.as_dict(), timing_comparison_valid=False, no_policy_publication=True)
    publish(args.output / "manifest.json", identity)
    model, payload = load_model_from_checkpoint(arch, str(checkpoint), device="cuda:0",
                                               eval_mode=False, return_full_checkpoint=True)
    restoration = verify_probe_checkpoint(model, payload, args.checkpoint)
    del payload
    gc.collect()
    prepare_action_updates(model, "cuda:0")
    model.eval()
    if model.model.ar_helper.block_wise_autoregressive:
        raise RuntimeError("This diagnostic explicitly assumes the original non-BAR codec")
    publish(args.output / "restoration.json", restoration)
    originals = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_index in range(2):
            original = take_row(originals[batch_index])
            source = source_receipt["batches"][batch_index]["sources"][0]
            batch = dict_apply(deepcopy(original), lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x)
            calls, stage, captured = defaultdict(lambda: defaultdict(int)), ["full"], {}
            handles = []
            for name, module in model.model.vlm.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    def count(_module, _args, _output, name=name):
                        calls[stage[0]][name] += 1
                    handles.append(module.register_forward_hook(count))
            original_prefill = model.model.vlm_prefill
            def capture_prefill(input_ids, attention_mask, *a, **kw):
                value = original_prefill(input_ids, attention_mask, *a, **kw)
                captured.update(ids=input_ids.detach().clone(), mask=attention_mask.detach().clone(),
                                hidden=value[0].detach().clone(), positions=value[2].detach().clone())
                return value
            model.model.vlm_prefill = capture_prefill
            try:
                _, metrics = model(batch)
            finally:
                model.model.vlm_prefill = original_prefill
            prefix_samples = action_prefix_samples(batch["samples"], settings)
            stage[0] = "forced"
            state = model.prefill(prefix_samples, batch["pixel_values"])
            split = state.input_ids.shape[1]
            ids, mask, positions = captured["ids"], captured["mask"], captured["positions"]
            targets = ids[0, split:]
            if (not torch.equal(ids[:, :split], state.input_ids)
                    or not torch.equal(mask[:, :split], state.attention_mask)
                    or not torch.equal(positions[..., :split], state.position_ids)
                    or not 1 <= targets.numel() <= 96 or not mask[:, split:].bool().all()):
                raise RuntimeError("Same-history diagnostic has unequal prefixes, padding, or excess targets")
            full_logits = model.model.vlm.decode(captured["hidden"][:, split - 1:-1]).float().cpu()[0]
            initial_hidden_rms = float((captured["hidden"][:, split - 1].float() - state.last_hidden.float()).square().mean().sqrt())
            del captured["hidden"]
            helper, core = model.model.ar_helper, model.model
            original_sample, original_mask = helper._sample, core.build_causal_mask_and_position_ids
            token_rows, position_rows = [], []
            def forced_sample(logits, **kwargs):
                index = len(token_rows)
                if index >= targets.numel():
                    raise RuntimeError("Actual AR loop exceeded forced-history budget")
                token_rows.append(compare_logits(full_logits[index], logits[0].float().cpu(), int(targets[index])))
                return targets[index:index + 1]
            def traced_mask(input_ids, attention_mask, *a, **kw):
                value = original_mask(input_ids, attention_mask, *a, **kw)
                end = attention_mask.shape[1]
                position_rows.append(dict(index=end - 1,
                    mask_equal=bool(torch.equal(attention_mask, mask[:, :end])),
                    position_equal=bool(torch.equal(value[1], positions[..., end - 1:end]))))
                return value
            helper._sample, core.build_causal_mask_and_position_ids = forced_sample, traced_mask
            try:
                forced = core.inference_ar(state.last_hidden, state.attention_mask, state.pixel_values,
                    past_key_values=state.kv_cache, return_kv_cache=True, stop_token_ids=[],
                    max_new_tokens=int(targets.numel()), do_sample=False)
                if not torch.equal(forced["generated_ids"][0], targets):
                    raise RuntimeError("Actual loop stopped before the declared teacher diagnostic suffix")
            finally:
                helper._sample, core.build_causal_mask_and_position_ids = original_sample, original_mask
            del forced, state, full_logits
            stage[0] = "free"
            raw = {}
            original_ar = core.inference_ar
            def traced_ar(*a, **kw):
                value = original_ar(*a, **kw)
                raw["ids"] = value["generated_ids"].detach().cpu().tolist()
                return value
            core.inference_ar = traced_ar
            torch.manual_seed(17)
            try:
                prediction = model.forward_inference(batch["samples"], batch["pixel_values"],
                    action_dim_is_pad=batch["action_dim_is_pad"])
                free = dict(valid=True, selected_action_source=prediction["selected_action_source"], raw=raw)
            except RuntimeError as exc:
                if not str(exc).startswith("Free AR generation"):
                    raise
                free = dict(valid=False, error=str(exc), raw=raw)
            finally:
                core.inference_ar = original_ar
                for handle in handles:
                    handle.remove()
            row = dict(source=source, prefix_length=split, forced_suffix_length=len(token_rows),
                teacher_policy_metrics={k: float(v) for k, v in metrics.items()},
                prefix_ids_and_positions_exact=True, first_hidden_rms=initial_hidden_rms,
                token_comparisons=token_rows, cached_position_comparisons=position_rows,
                actual_lora_module_calls={key: dict(value) for key, value in calls.items()}, free_generation=free)
            rows.append(row)
            publish(args.output / f"row_{batch_index}.json", row)
            print(json.dumps(dict(source=source, tokens=len(token_rows), initial_hidden_rms=initial_hidden_rms,
                max_logit_diff=max(r["max_abs_diff"] for r in token_rows),
                same_argmax=sum(r["argmax_equal"] for r in token_rows), free_valid=free["valid"])), flush=True)
    publish(args.output / "result.json", dict(complete=True, identity=identity, rows=rows,
        optimizer_updates=0, simulator_controls=0, no_success_rate_claim=True,
        limitations=["Teacher-forced cached decoding is an offline numerical diagnostic, never an actor repair.",
                     "Two original train windows are not method-effect or heldout evidence.",
                     "Numerical differences are recorded, not automatically labelled a bug or a pass."]))


if __name__ == "__main__":
    main()
