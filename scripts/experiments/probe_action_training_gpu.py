"""Bounded real-model AR/joint/KI route gate; not a method-effect experiment.

A4 full-weight warm start or explicitly verified native G0.5 initialization,
audited original train inputs, two temporary Adam updates, actual per-loss
gradients and target-free generation. The native probe includes one real row
per task before/after, not a single row repeated five times. No checkpoint is
published. Each route is a separate process/output with explicit identity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import traceback

from action_training_runtime import bootstrap, REPO, INPUT, INPUT_SHA, sha
from probe_ar_execution_codec import publish
from train_fm_method_probe import PARENT, PARENT_SHA, verify_parent
from native_action_initialization import (
    NATIVE_PARENT, NATIVE_PARENT_SHA, NATIVE_CONFIG, NATIVE_CONFIG_SHA, verify_native_parent,
)


def architecture_for(cfg, settings, *, parent=PARENT):
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.tokenizer.vq_config.dropout_noop_parts = False
    arch = cfg.model.model_arch
    arch._target_ = "g05.models.g05.g05_policy_memlite_action.G05PolicyMEMLiteAction"
    arch.action_training = settings.as_dict()
    arch.discrete_action, arch.continuous_action = True, settings.uses_fm
    arch.predict_cot, arch.memlite_train_mode = settings.predicts_cot, "off"
    if settings.predicts_cot:
        if Path(parent) != NATIVE_PARENT:
            raise ValueError("The declared Subtask-CoT arm starts from native G0.5 only")
        arch.input_preprocessor.pred_eov = True
        arch.native_cot_max_new_tokens = 256
    arch.register_memlite_hl_end = Path(parent) != NATIVE_PARENT
    arch.fm.joint_training = settings.route == "joint"
    arch.AT_CONFIG.dropout_noop_parts = False
    arch.language_loss_weight = 1.
    arch.coordination_train.stage = "experimental_" + settings.route
    arch.coordination_train.trainability_profile = "low_ce_fm_lora" if settings.uses_fm else "low_ar_lora"
    arch.coordination_train.expected_trainable_groups = ["action_expert", "vlm_lora"] if settings.uses_fm else ["vlm_lora"]
    cfg.model.pretrained_ckpt, cfg.resume_ckpt = str(parent), None
    return arch


def take_row(batch, index=0):
    import torch
    result = {}
    for key in ("samples", "action", "action_is_pad", "action_dim_is_pad", "pixel_values"):
        value = batch[key]
        if key == "pixel_values" and isinstance(value, dict):
            result[key] = {name: tensor[index:index + 1].clone() for name, tensor in value.items()}
        elif isinstance(value, torch.Tensor):
            result[key] = value[index:index + 1].clone()
        else:
            result[key] = deepcopy(value[index:index + 1])
    return result


def prepare_action_updates(model, device):
    contract = model.configure_coordination_trainability()
    model.apply_fp32_params()
    # The codec is an independent tokenizer object, not an nn.Module child
    # moved by policy.to(). Mirror the real finetune loader's explicit move.
    model.action_tokenizer.to(device)
    return contract


def task_generation_rows(originals, receipt):
    """One actual cached train row per task, with its exact original source."""
    if len(originals) != len(receipt["batches"]):
        raise RuntimeError("Cached batch/source lengths differ")
    selected = {}
    for batch, item in zip(originals, receipt["batches"]):
        if len(batch["samples"]) != len(item["sources"]):
            raise RuntimeError("Cached row/source lengths differ")
        for index, source in enumerate(item["sources"]):
            task = int(source["task"])
            if task not in selected:
                selected[task] = (take_row(batch, index), deepcopy(source))
    if set(selected) != set(range(5)):
        raise RuntimeError("Native generation probe requires actual rows from all five tasks")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=["ar", "joint", "ki"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--initialization", choices=["a4", "native"], default="a4")
    parser.add_argument("--conditioning", choices=["skills", "task", "native_task", "native_subtask_cot"], default="skills")
    args = parser.parse_args()
    if not args.output.is_absolute() or args.gpu not in range(4):
        raise ValueError("Use one explicit robo GPU and new absolute output")
    if args.initialization == "native" and args.route != "ar":
        raise ValueError("The native initialization probe is currently an explicit pure-AR experiment")
    parent_path, parent_sha = ((NATIVE_PARENT, NATIVE_PARENT_SHA) if args.initialization == "native"
                               else (PARENT, PARENT_SHA))
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run from a clean, pinned worktree")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.training.coordination_grad_audit import CoordinationGradientAudit
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    from g05.utils.common.pytorch_utils import dict_apply

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info()
    if free < 40 * 1024**3:
        raise RuntimeError("Single-row engineering gate requires at least 40 GiB free; do not evict other work")
    original_receipt = json.loads((INPUT / "result.json").read_text())
    if (sha(parent_path) != parent_sha or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or original_receipt["status"] != "complete" or not original_receipt["train_only"]
            or original_receipt["processor_config_sha256"] != sha(INPUT / "diagnostic_processor_config.yaml")):
        raise RuntimeError("Declared parent or audited train inputs changed")
    if args.initialization == "native" and sha(NATIVE_CONFIG) != NATIVE_CONFIG_SHA:
        raise RuntimeError("Native checkpoint configuration identity changed")
    settings = ActionTrainingSettings(route=args.route, conditioning=args.conditioning)
    cfg = OmegaConf.load(INPUT / "diagnostic_processor_config.yaml")
    arch = architecture_for(cfg, settings, parent=parent_path)
    identity.update(commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        gpu=args.gpu, parent=str(parent_path), parent_sha256=parent_sha, input_sha256=INPUT_SHA,
        initialization=args.initialization,
        native_config_sha256=NATIVE_CONFIG_SHA if args.initialization == "native" else None,
        initialization_helper_sha256=sha(Path(__file__).with_name("native_action_initialization.py")),
        input_config_sha256=sha(INPUT / "diagnostic_processor_config.yaml"), entry_sha256=sha(Path(__file__)),
        start_time=datetime.now(timezone.utc).isoformat(), settings=settings.as_dict(),
        architecture=OmegaConf.to_container(arch, resolve=True), max_updates=2, single_row_microbatch=True,
        target_free_generations=10 if args.initialization == "native" else 2,
        checkpoint_saved=False, simulator_controls=0, formal_training=False,
        timing_comparison_valid=False, timing_note="May share a GPU with an independent FM job; correctness only.")
    publish(args.output / "manifest.json", identity)
    print(json.dumps(dict(stage="load", route=args.route, parent_sha256=parent_sha)), flush=True)
    model, parent = load_model_from_checkpoint(arch, str(parent_path), device="cuda:0",
                                              eval_mode=False, return_full_checkpoint=True)
    if args.initialization == "native":
        restoration = verify_native_parent(model, parent)
    else:
        verify_parent(model, parent)
        restoration = dict(passed=True, exact_state_entries=1138, exact_lora_entries=192)
    del parent
    gc.collect()
    contract = prepare_action_updates(model, "cuda:0")
    publish(args.output / "restoration.json", dict(**restoration, trainability=contract))
    originals = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    cpu_rows = [take_row(batch) for batch in originals[:2]]
    generation_rows = task_generation_rows(originals, original_receipt) if args.initialization == "native" else {}
    def move(batch):
        return dict_apply(deepcopy(batch), lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x)
    first = move(cpu_rows[0])
    groups = model.coordination_trainable_parameter_groups()
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    audit = CoordinationGradientAudit(model, expected_groups=sorted(groups),
                                     frozen_prefixes=frozen, output_dir=args.output, rank=0)
    torch.cuda.reset_peak_memory_stats()

    def fixed_forward(alter_suffix=False):
        original_encode = model.processor.encode_train
        altered = {}
        def encode(*a, **kw):
            ids, labels, mask, split = original_encode(*a, **kw)
            if alter_suffix:
                ids = ids.clone()
                begin, end = model.action_tokenizer.action_token_begin_idx, model.action_tokenizer.action_token_end_idx
                selected = (ids >= begin) & (ids < end)
                selected[:, :split] = False
                ids[selected] = begin + (ids[selected] - begin + 1) % (end - begin)
                altered["token_count"] = int(selected.sum())
                if not altered["token_count"]:
                    raise RuntimeError("No teacher-action suffix tokens changed in counterfactual")
            return ids, labels, mask, split
        model.processor.encode_train = encode
        try:
            with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(771)
                torch.cuda.manual_seed_all(771)
                loss, metrics = model(first)
            return {k: float(v) for k, v in metrics.items()}, altered
        finally:
            model.processor.encode_train = original_encode

    model.eval()
    before, _ = fixed_forward()
    if settings.uses_fm:
        counterfactual, changed = fixed_forward(True)
        if before["fm_loss"] != counterfactual["fm_loss"]:
            raise RuntimeError("Actual FM loss changed with teacher-action suffix at fixed observation/target/noise")
        publish(args.output / "teacher_suffix_gate.json", dict(passed=True,
            actual_model_forwards=2, fm_loss_bitwise_equal=True, original=before,
            counterfactual=counterfactual, altered_action_tokens=changed["token_count"]))

    # Raw neural CE/FM tensors are captured before the policy detaches its
    # logging metrics. Test each loss separately on the actual parameter graph.
    raw = {}
    hook = model.model.register_forward_hook(lambda _m, _i, output: raw.update(
        {key: output[key] for key in ("ce_loss", "fm_loss")}))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        total, _ = model(first)
    gradient_routes = {}
    for loss_name, value in raw.items():
        gradient_routes[loss_name] = {}
        for group, rows in groups.items():
            derivatives = (torch.autograd.grad(value, [p for _, p in rows], retain_graph=True, allow_unused=True)
                           if value.requires_grad else [None] * len(rows))
            if any(not torch.isfinite(g).all() for g in derivatives if g is not None):
                raise RuntimeError("Nonfinite separate-loss gradient")
            gradient_routes[loss_name][group] = dict(connected=sum(g is not None for g in derivatives),
                nonzero=sum(g is not None and bool(torch.count_nonzero(g)) for g in derivatives))
            del derivatives
    if not gradient_routes["ce_loss"]["vlm_lora"]["nonzero"]:
        raise RuntimeError("Action CE has no real LoRA learning signal")
    if settings.uses_fm:
        if (gradient_routes["ce_loss"]["action_expert"]["connected"]
                or not gradient_routes["fm_loss"]["action_expert"]["nonzero"]):
            raise RuntimeError("CE/FM do not obey action-expert gradient contract")
        vlm_fm = gradient_routes["fm_loss"]["vlm_lora"]
        if args.route == "ki" and vlm_fm["connected"] or args.route == "joint" and not vlm_fm["nonzero"]:
            raise RuntimeError("Actual FM-to-LoRA gradient isolation differs from recipe")
    elif raw["fm_loss"].requires_grad or float(raw["fm_loss"]) != 0.:
        raise RuntimeError("Pure AR unexpectedly executed/train-supervised FM")
    publish(args.output / "gradient_routes.json", dict(passed=True, routes=gradient_routes))
    hook.remove()
    del total, value
    raw.clear()

    def free_generation(label, current=None, source=None):
        current = first if current is None else current
        source = original_receipt["batches"][0]["sources"][0] if source is None else source
        model.eval()
        generated = {}
        original_ar = model.model.inference_ar
        def traced(*a, **kw):
            result = original_ar(*a, **kw)
            generated["ids"] = result["generated_ids"].detach().cpu().tolist()
            generated.setdefault("calls", []).append(dict(ids=generated["ids"],
                max_new_tokens=kw.get("max_new_tokens"), stop_token_ids=kw.get("stop_token_ids")))
            return result
        model.model.inference_ar = traced
        try:
            with torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                result = model.forward_inference(current["samples"], current["pixel_values"],
                                                 action_dim_is_pad=current["action_dim_is_pad"])
            target = current["action"].to(result["action"].device)
            valid = ((~current["action_is_pad"][:, :16])[:, :, None]
                     & (~current["action_dim_is_pad"])[:, None, :]).to(target.device)
            error = (result["action"][:, :16] - target[:, :16])[valid]
            report = dict(valid=True, selected_action_source=result["selected_action_source"],
                normalized_executed_rmse=float(error.square().mean().sqrt()),
                execution_start=result["execution_start"], execution_steps=result["execution_steps"])
            if "cot_text" in result:
                report["cot_text"] = result["cot_text"]
        except RuntimeError as exc:
            if not str(exc).startswith("Free AR generation"):
                raise
            report = dict(valid=False, failure=str(exc))
        finally:
            model.model.inference_ar = original_ar
            publish(args.output / f"{label}_raw_generation.json", generated)
        report.update(generated=generated, source=source)
        publish(args.output / f"{label}_generation.json", report)
        return report

    generated_before = free_generation("before")
    def other_task_generations(label, first_result):
        if not generation_rows:
            return [first_result]
        reports = []
        for task, (original, source) in sorted(generation_rows.items()):
            if source == first_result["source"]:
                reports.append(first_result)
            else:
                reports.append(free_generation(f"{label}_task{task}", move(original), source))
        return reports
    generations_before = other_task_generations("before", generated_before)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(.9, .95), weight_decay=.03)
    model.train()
    updates = []
    for index, original in enumerate(cpu_rows):
        batch = move(original)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite action-training loss")
        loss.backward()
        facts = audit.after_backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite action-training gradient norm")
        optimizer.step()
        if index == 0:
            audit.after_optimizer_step(facts, step=1, effective_lrs=[1e-5])
        updates.append(dict(step=index + 1, grad_norm=float(norm), losses={k: float(v) for k, v in metrics.items()}))
        print(json.dumps(dict(stage="update", **updates[-1])), flush=True)
        del batch, loss, metrics
    optimizer.zero_grad(set_to_none=True)
    audit.require_verified_update()
    audit._assert_frozen_unchanged()
    if (len(optimizer.state) != len(params) or {int(s["step"]) for s in optimizer.state.values()} != {2}
            or any(not torch.isfinite(p).all() for p in model.parameters())
            or any(not torch.isfinite(s[key]).all() for s in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq"))):
        raise RuntimeError("Adam state/actual update count/finiteness check failed")
    generated_after = free_generation("after")
    generations_after = other_task_generations("after", generated_after)
    model.eval()
    after, _ = fixed_forward()
    publish(args.output / "result.json", dict(complete=True, gradient_and_update_gate_passed=True,
        inference_valid_before=generated_before["valid"], inference_valid_after=generated_after["valid"],
        generations_before=generations_before, generations_after=generations_after,
        settings=settings.as_dict(), actual_updates=2, adam_state_count=len(optimizer.state),
        before=before, after=after, updates=updates, gradient_routes=gradient_routes,
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(), parent_sha256=parent_sha,
        initialization=args.initialization,
        checkpoint_saved=False, formal_training=False, simulator_controls=0,
        limitations=["Two temporary updates are not a training-method effect or success-rate experiment.",
                     "Initialization is explicit; this is not the published BEHAVIOR policy or full CoT recipe.",
                     "Train-window generation validity is not success rate; timing may include GPU sharing."]))
    print(json.dumps(dict(complete=True, route=args.route, actual_updates=2,
                         inference_valid_after=generated_after["valid"])), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
