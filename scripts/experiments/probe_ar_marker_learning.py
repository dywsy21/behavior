"""Twenty-update paired train-cache diagnostic, NOT a deployable policy.

Run control and markers separately from the exact completed AR500 parent. Both
use eager CE and the same original ten train rows. Only the markers arm trains
the shared eight-row vocabulary delta. No simulator or constraint decoder.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import traceback

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from probe_action_training_gpu import architecture_for, prepare_action_updates, take_row, task_generation_rows
from probe_ar_execution_codec import publish
from train_action_method_probe import now, read
from train_fm_method_probe import BASE, PARENT_SHA

AR500_SHA = "51bacc1d9ec1f6d19c7e82e93bed60b8a1eb5d5ffbab5337c9b7ba47a84e90aa"
AUDIT_SHA = "88c4a83e5f11e5df8f967966de1aaa92db2be1ecec1d33b3948c3d7f49778edb"
MAX_UPDATES = 20


def marker_architecture(cfg, settings, arm):
    if arm not in {"control", "markers"}:
        raise ValueError("Only matched control and required-marker arms are declared")
    arch = architecture_for(cfg, settings)
    arch._target_ = "g05.models.g05.g05_policy_memlite_action_rows.G05PolicyMEMLiteActionRows"
    arch.ar.use_fused_ce = False
    arch.marker_row_adaptation = dict(train_rows=arm == "markers", mode="required_codec_markers")
    return arch


def marker_cache_metrics(cache, marker_ids):
    """Use teacher logits only for diagnostics, never to fill actor outputs."""
    import torch
    labels, logits, losses = (cache[k] for k in ("shift_labels_masked", "shift_logits", "token_loss"))
    if (logits is None or labels.ndim != 1 or logits.shape[0] != labels.numel()
            or losses.shape != labels.shape or not torch.isfinite(logits).all()):
        raise RuntimeError("Finite eager CE cache required for marker diagnostics")
    rows = []
    for token in marker_ids:
        positions = torch.where(labels == token)[0]
        if positions.numel() != 1:
            raise RuntimeError("Each single-row target must contain every actual marker exactly once")
        index = int(positions[0])
        row = logits[index].float()
        rows.append(dict(token_id=token, target_index=index, ce=float(losses[index]),
            rank=1 + int((row > row[token]).sum()), argmax=int(row.argmax())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "markers"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.parent != BASE or args.gpu not in range(4):
        raise ValueError("Use a new direct child run and explicit robo GPU")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned worktree, not active training source")
    run = BASE / "ar_a4_fulltrain_v2"
    checkpoint = run / "formal/checkpoints/step_500.pt"
    inspection, spec, status = (read(run / name) for name in
        ("formal/checkpoint_inspection.json", "method_spec.json", "status.json"))
    audit_path = BASE / "ar_marker_embedding_audit_v1/result.json"
    if (status.get("state") != "complete" or not inspection.get("passed")
            or inspection.get("actual_updates") != 500 or inspection["checkpoint"] != str(checkpoint)
            or inspection["checkpoint_sha256"] != AR500_SHA or spec["route"] != "ar"
            or spec["parent_sha256"] != PARENT_SHA or spec["recipe"]["max_updates"] != 500
            or sha(audit_path) != AUDIT_SHA or not read(audit_path)["complete"]):
        raise RuntimeError("Completed AR500 parent or actual marker audit identity changed")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output.mkdir(exist_ok=False)
    identity = bootstrap()
    import torch
    from torch.nn import functional as F
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    from g05.utils.training.coordination_grad_audit import CoordinationGradientAudit
    from g05.models.g05.helpers.action_marker_rows import restore_marker_rows, assert_marker_bindings

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Require 40 GiB free; do not evict other work")
    torch.cuda.set_per_process_memory_fraction(.4)
    config = INPUT / "diagnostic_processor_config.yaml"
    receipt = read(INPUT / "result.json")
    if (sha(checkpoint) != AR500_SHA or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or receipt["processor_config_sha256"] != sha(config)
            or receipt["status"] != "complete" or not receipt["train_only"]):
        raise RuntimeError("Parent, audited original train cache, or processor changed")
    cfg = OmegaConf.load(config)
    settings = ActionTrainingSettings(route="ar", conditioning="skills")
    arch = marker_architecture(cfg, settings, args.arm)
    identity.update(start_time=now(), arm=args.arm, gpu=args.gpu,
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), checkpoint=str(checkpoint), checkpoint_sha256=AR500_SHA,
        original_input_sha256=INPUT_SHA, input_config_sha256=sha(config), marker_audit_sha256=AUDIT_SHA,
        settings=settings.as_dict(), architecture=OmegaConf.to_container(arch, resolve=True),
        max_updates=MAX_UPDATES, microbatch=2, original_train_rows=10, actual_train_draws=40,
        seed=41, lora_lr=1e-5, marker_lr=1e-3 if args.arm == "markers" else None,
        betas=[.9, .95], weight_decay=.03, clip_norm=1., scheduler="constant",
        original_cache_order=True, simulator_controls=0, standalone_policy_saved=False,
        generalization_experiment=False, timing_comparison_valid=False)
    publish(args.output / "manifest.json", identity)
    print(json.dumps(dict(stage="load", arm=args.arm, checkpoint_sha256=AR500_SHA)), flush=True)
    model, payload = load_model_from_checkpoint(arch, str(checkpoint), device="cuda:0",
                                               eval_mode=False, return_full_checkpoint=True)
    expected, actual = payload["model_state_dict"], model.state_dict()
    extra = {"model.action_marker_rows.delta", "model.action_marker_rows.token_ids"}
    if (payload["step"] != 500 or len(expected) != 1138 or set(actual) - set(expected) != extra
            or set(expected) - set(actual) or sum("lora_" in key for key in expected) != 192):
        raise RuntimeError("AR500 full state or explicit two-entry adapter schema changed")
    for key, value in expected.items():
        loaded = actual[key].detach().cpu()
        if (loaded.dtype != value.dtype or loaded.shape != value.shape
                or not torch.equal(loaded.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8))):
            raise RuntimeError("Parent not restored exactly: " + key)
    del expected, actual, payload
    gc.collect()
    contract = prepare_action_updates(model, "cuda:0")
    adapter, vlm = model.model.action_marker_rows, model._marker_vlm()
    marker_ids = adapter.token_ids.cpu().tolist()
    audited_ids = sorted(row["token_id"] for row in read(audit_path)["matrices"][0]["rows"])
    if (marker_ids != audited_ids or adapter.delta.numel() != 16384
            or adapter.delta.dtype != torch.float32 or torch.count_nonzero(adapter.delta)):
        raise RuntimeError("Required markers, capacity, fp32 storage or zero initialization changed")
    publish(args.output / "restoration.json", dict(passed=True, exact_parent_entries=1138,
        exact_parent_lora_entries=192, new_adapter_entries=2, trainability=contract))
    originals = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    if len(originals) != 5 or any(len(batch["samples"]) != 2 for batch in originals):
        raise RuntimeError("Twenty-update probe requires the same five original two-row batches")
    keys = ("samples", "action", "action_is_pad", "action_dim_is_pad", "pixel_values")
    cpu_batches = [{key: batch[key] for key in keys} for batch in originals]
    generation_rows = task_generation_rows(originals, receipt)
    def move(batch):
        return dict_apply(deepcopy(batch), lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x)
    groups = model.coordination_trainable_parameter_groups()
    params = [p for _, entries in groups.items() for _, p in entries]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    audit = CoordinationGradientAudit(model, expected_groups=sorted(groups), frozen_prefixes=frozen,
                                     output_dir=args.output, rank=0)
    torch.cuda.reset_peak_memory_stats()

    # Actual first neural forward: compare BOTH invoked projection outputs to
    # the restored unmodified parent operations, before any optimizer update.
    zero_calls = dict(input=0, output=0)
    def zero_projection(module, inputs, output, kind):
        original = (F.embedding(inputs[0], module.weight, module.padding_idx) if kind == "input"
                    else F.linear(inputs[0], module.weight))
        if not torch.equal(original, output):
            raise RuntimeError("Zero marker delta changed the actual parent " + kind + " projection")
        zero_calls[kind] += 1
    handles = [vlm.input_proj.register_forward_hook(lambda m, i, o: zero_projection(m, i, o, "input")),
               vlm.output_proj.register_forward_hook(lambda m, i, o: zero_projection(m, i, o, "output"))]
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, zero_metrics = model(move(take_row(originals[0])))
    for handle in handles:
        handle.remove()
    if not all(zero_calls.values()):
        raise RuntimeError("One actual marker projection was bypassed")
    publish(args.output / "zero_parent_gate.json", dict(passed=True, actual_projection_calls=zero_calls,
        metrics={key: float(value) for key, value in zero_metrics.items()}))
    del loss, zero_metrics

    def evaluate(label):
        model.eval()
        rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, original in enumerate(originals):
                for row_index in range(2):
                    _, metrics = model(move(take_row(original, row_index)))
                    rows.append(dict(source=receipt["batches"][index]["sources"][row_index],
                        metrics={key: float(value) for key, value in metrics.items()},
                        marker_metrics=marker_cache_metrics(model.model.ar_helper._last_ce_cache, marker_ids)))
        result = dict(rows=rows, mean_ce=sum(row["metrics"]["ce_loss"] for row in rows) / len(rows),
                      mean_action_ce=sum(row["metrics"]["action_token_ce"] for row in rows) / len(rows))
        publish(args.output / f"{label}_train_diagnostic.json", result)
        return result

    def generate(label):
        model.eval()
        reports = []
        for task, (original, source) in sorted(generation_rows.items()):
            batch, generated = move(original), {}
            original_ar = model.model.inference_ar
            def traced(*a, **kw):
                value = original_ar(*a, **kw)
                generated.setdefault("calls", []).append(dict(ids=value["generated_ids"].cpu().tolist(),
                    max_new_tokens=kw.get("max_new_tokens"), stop_token_ids=kw.get("stop_token_ids")))
                return value
            model.model.inference_ar = traced
            try:
                with torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.manual_seed(17)
                    torch.cuda.manual_seed_all(17)
                    value = model.forward_inference(batch["samples"], batch["pixel_values"],
                                                    action_dim_is_pad=batch["action_dim_is_pad"])
                valid = (~batch["action_is_pad"][:, :16, None] & ~batch["action_dim_is_pad"][:, None, :])
                difference = (value["action"][:, :16] - batch["action"][:, :16])[valid]
                report = dict(valid=True, selected_action_source=value["selected_action_source"],
                    normalized_executed_rmse=float(difference.square().mean().sqrt()),
                    complete_blocks=value["ar_complete_block_receipts"])
            except RuntimeError as exc:
                if not str(exc).startswith("Free AR generation"):
                    raise
                report = dict(valid=False, failure=str(exc))
            finally:
                model.model.inference_ar = original_ar
                publish(args.output / f"{label}_task{task}_raw_generation.json", generated)
            report.update(task=task, source=source, generated=generated)
            reports.append(report)
        publish(args.output / f"{label}_generations.json", reports)
        return reports

    before, generated_before = evaluate("before"), generate("before")
    optimizer_groups = [dict(params=[p for _, p in entries], lr=1e-3 if name == "action_marker_rows" else 1e-5,
                             group_name=name) for name, entries in groups.items()]
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(.9, .95), weight_decay=.03)
    torch.manual_seed(41)
    torch.cuda.manual_seed_all(41)
    model.train()
    updates = []
    with (args.output / "train_metrics.jsonl").open("x") as stream:
        for index in range(MAX_UPDATES):
            batch = move(cpu_batches[index % len(cpu_batches)])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model(batch)
            if not torch.isfinite(loss) or float(metrics["fm_loss"]) != 0.:
                raise RuntimeError("Nonfinite CE or unintended FM supervision")
            loss.backward()
            facts = audit.after_backward()
            if (args.arm == "markers" and (adapter.delta.grad is None
                    or not torch.count_nonzero(adapter.delta.grad))):
                raise RuntimeError("Actual CE did not reach the marker delta")
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient norm")
            optimizer.step()
            if index == 0:
                audit.after_optimizer_step(facts, step=1, effective_lrs=[group["lr"] for group in optimizer.param_groups])
            row = dict(step=index + 1, time=now(), grad_norm=float(norm),
                metrics={key: float(value) for key, value in metrics.items()},
                sources=receipt["batches"][index % len(cpu_batches)]["sources"],
                marker_delta_norms=adapter.delta.detach().norm(dim=1).cpu().tolist())
            updates.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(dict(stage="update", step=index + 1, ce_loss=float(metrics["ce_loss"]))), flush=True)
            del batch, loss, metrics
    optimizer.zero_grad(set_to_none=True)
    audit.require_verified_update()
    audit._assert_frozen_unchanged()
    if (len(optimizer.state) != len(params) or {int(item["step"]) for item in optimizer.state.values()} != {MAX_UPDATES}
            or any(not torch.isfinite(p).all() for p in model.parameters())
            or any(not torch.isfinite(item[key]).all() for item in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq"))):
        raise RuntimeError("Actual Adam count, finiteness, or updates did not match the finite recipe")
    # Small experimental artifact, only meaningful with the SHA-bound parent.
    # Exercise restoration by perturbing and reloading the actual trainables.
    saved = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
             if "lora_" in name or name.startswith("model.action_marker_rows.")}
    artifact = args.output / "trainable_state.pt"
    torch.save(dict(standalone_policy=False, parent_sha256=AR500_SHA, step=MAX_UPDATES,
        arm=args.arm, trainable_state=saved, optimizer=optimizer.state_dict()), artifact)
    loaded = torch.load(artifact, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for parameter in params:
            parameter.add_(.01)
    restored_rows = restore_marker_rows(adapter, loaded["trainable_state"])
    model.load_state_dict({key: value for key, value in loaded["trainable_state"].items() if "lora_" in key}, strict=False)
    optimizer.state.clear()
    optimizer.load_state_dict(loaded["optimizer"])
    if (any(not torch.equal(model.state_dict()[key].detach().cpu(), tensor) for key, tensor in saved.items())
            or len(optimizer.state) != len(params)
            or {int(item["step"]) for item in optimizer.state.values()} != {MAX_UPDATES}):
        raise RuntimeError("Actual trainable/Adam roundtrip failed")
    for current_group, saved_group in zip(optimizer.param_groups, loaded["optimizer"]["param_groups"]):
        if any(current_group[key] != saved_group[key] for key in ("lr", "betas", "weight_decay", "group_name")):
            raise RuntimeError("Restored optimizer group settings changed")
        for parameter, saved_id in zip(current_group["params"], saved_group["params"]):
            current_state, saved_state = optimizer.state[parameter], loaded["optimizer"]["state"][saved_id]
            if (set(current_state) != set(saved_state)
                    or any(not torch.equal(current_state[key].cpu(), value) for key, value in saved_state.items())):
                raise RuntimeError("Restored optimizer moments or step did not match saved tensors")
    assert_marker_bindings(vlm, adapter)
    audit._assert_frozen_unchanged()
    after, generated_after = evaluate("after"), generate("after")
    publish(args.output / "result.json", dict(complete=True, arm=args.arm, actual_updates=MAX_UPDATES,
        actual_train_draws=40, train_cache_only=True, before=before, after=after,
        generations_before=generated_before, generations_after=generated_after,
        actual_adam_states=len(optimizer.state), trainable_roundtrip_passed=True,
        restored_marker_rows=restored_rows, frozen_unchanged=True, parent_sha256=AR500_SHA,
        artifact=str(artifact), artifact_sha256=sha(artifact), standalone_policy=False,
        new_process_resume_tested=False, simulator_controls=0,
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        limitations=["Ten original train rows repeatedly fitted: not original-split generalization or success rate.",
                     "Marker rank under teacher forcing is not actual free-generation correctness.",
                     "No fallback, schema-constrained decoding, CoT or simulator was used."]))
    print(json.dumps(dict(complete=True, actual_updates=MAX_UPDATES,
        result_sha256=sha(args.output / "result.json"), free_complete=sum(row["valid"] for row in generated_after))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
