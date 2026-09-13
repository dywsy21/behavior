"""Bounded, real-data AR / joint / KI / paired-FM training experiments.

This is an explicit experimental trainer, not a bypass of the stock trainer's
stage contract. Source, full A4 initialization, train/eval identities, actual
draws, gradient groups, optimizer clocks and checkpoint bytes are audited.
One invocation runs five-update DDP save/read validation and, only on success,
500 fresh updates. No deployment, retry, extra training or SR claim.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from action_training_runtime import bootstrap, REPO, INPUT, sha
from action_training_data import build_original_pipeline, exact_eval_batch, STATS, STATS_SHA, FIXED_SHA
from probe_action_training_gpu import architecture_for, prepare_action_updates
from probe_ar_execution_codec import publish
from train_fm_method_probe import BASE, PARENT, PARENT_SHA, legacy, verify_parent

HERE = Path(__file__).resolve()
ROUTES = ("ar", "joint", "ki", "fm")
RECIPE = dict(seed=41, batch_size=2, accumulation=2, world_size=4,
    # Match the working A3 pipeline's per-rank prefetch concurrency. Exact
    # continuation of stochastic worker state is NOT claimed by saving RNG.
    workers=4, max_updates=500, smoke_updates=5, learning_rate=1e-5,
    warmup_steps=50, lr_min_ratio=.1, betas=[.9, .95], weight_decay=.03,
    max_grad_norm=1., eval_every=100, history=6, prediction=32, execution=[0, 16])


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def code_identity():
    paths = [HERE, Path(__file__).with_name("action_training_runtime.py"),
        Path(__file__).with_name("action_training_data.py"),
        Path(__file__).with_name("probe_ar_execution_codec.py"),
        Path(__file__).with_name("probe_action_training_gpu.py"),
        Path(__file__).with_name("native_action_initialization.py"),
        Path(__file__).with_name("train_fm_method_probe.py")]
    from action_training_runtime import EXTENSIONS
    paths.extend(REPO / relative for relative in EXTENSIONS.values())
    return {str(path.relative_to(REPO)): sha(path) for path in paths}


def validate_spec(spec):
    if (spec["route"] not in ROUTES or spec["recipe"] != RECIPE
            or spec["parent_sha256"] != PARENT_SHA or spec["code"] != code_identity()
            or Path(spec["output"]).parent != BASE):
        raise RuntimeError("Action experiment recipe/source/parent/budget changed")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean immutable action experiment worktree")
    if subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() != spec["commit"]:
        raise RuntimeError("Active action recipe worktree must not be switched")
    if spec["prerequisites"] != prerequisites(spec["route"]):
        raise RuntimeError("Required completed input/gradient evidence changed")


def configure(cfg, route):
    from omegaconf import OmegaConf
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    OmegaConf.set_struct(cfg, False)
    if route == "fm":
        arch = cfg.model.model_arch
        if (arch.discrete_action or not arch.continuous_action or arch.predict_cot
                or "G05PolicyMEMLiteSkillFM" not in arch._target_):
            raise RuntimeError("Paired FM must use the unchanged original SkillFM architecture")
        cfg.model.pretrained_ckpt, cfg.resume_ckpt = str(PARENT), None
    else:
        arch = architecture_for(cfg, ActionTrainingSettings(route=route))
    for key, value in dict(batch_size=2, grad_accumulation_steps=2, num_workers=4,
            learning_rate=1e-5, warmup_steps=50, lr_min_ratio=.1, max_steps=500,
            weight_decay=.03, betas=[.9, .95], max_grad_norm=1.).items():
        cfg.model[key] = value
    cfg.seed = 41
    return arch


def make_optimizer(model):
    import torch
    from g05.utils.training.get_scheduler import get_scheduler
    groups = model.get_optim_param_groups(lr=1e-5, weight_decay=.03,
        apply_decay_on_norm_and_bias=False, backbone_lr_multiplier=1., vision_lr_multiplier=1.)
    actual = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Optimizer must cover every and only declared parameter once")
    optimizer = torch.optim.AdamW(groups, lr=1e-5, betas=(.9, .95))
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=50,
                              num_training_steps=500, lr_min_ratio=.1)
    return optimizer, scheduler


def expected_adam_states(route):
    return 192 if route == "ar" else 504 if route == "fm" else 514


def prerequisites(route):
    input_path, data_path = BASE / "ar_input_gate_v2/result.json", BASE / "ar_loader_gate_v1/result.json"
    inputs, data = read(input_path), read(data_path)
    if not inputs["complete"] or inputs["rendered_views"] != 40 or not data["complete"]:
        raise RuntimeError("Real tokenizer/complete full-data input gates are required")
    name = {"ar": "ar_gpu_gate_v2", "ki": "ki_gpu_gate_v1", "joint": "joint_gpu_gate_v1",
            "fm": "fm_control_v1/gate"}[route]
    path = BASE / name / "result.json"
    gate = read(path)
    if not gate.get("complete", gate.get("passed", False)) or gate["actual_updates"] != 2:
        raise RuntimeError("Real route-specific gradient/update gate is required")
    return {str(p): sha(p) for p in (input_path, data_path, path)}


def dependency_identity(root):
    root = Path(root)
    if root.parent != BASE or root.name != "fm_ae_lr2x_v1":
        raise ValueError("Only the predeclared AE-LR candidate may precede this AR run")
    launch, method = read(root / "launch.json"), read(root / "method_spec.json")
    if method["trial"] != "ae_lr2x" or method["parent_sha256"] != PARENT_SHA or method["max_updates"] != 500:
        raise RuntimeError("FM dependency is not the declared finite paired candidate")
    return dict(root=str(root), supervisor_pid=launch["supervisor_pid"],
                launch_sha256=sha(root / "launch.json"), method_sha256=sha(root / "method_spec.json"))


def process_start_ticks(pid, proc_root=Path("/proc")):
    """Read a live Linux process identity without optional Python pidfd APIs."""
    try:
        raw = (Path(proc_root) / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return None
    fields = raw.rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        return None
    # fields begins at proc(5) field3 (state); starttime is field22.
    return int(fields[19])


def wait_for_dependency(spec, old):
    dependency = spec.get("after_fm")
    if dependency is None:
        return
    root = Path(dependency["root"])
    if dependency_identity(root) != dependency:
        raise RuntimeError("Predecessor launch/recipe identity changed")
    pid = dependency["supervisor_pid"]
    original_ticks = process_start_ticks(pid)
    while original_ticks is not None and process_start_ticks(pid) == original_ticks:
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if process_start_ticks(pid) != original_ticks:
                break
            if (b"supervise" not in argv or str(root / "method_spec.json").encode() not in argv
                    or not any(value.endswith(b"/train_fm_method_probe.py") for value in argv)):
                raise RuntimeError("Predecessor PID is not its recorded FM supervisor")
        except FileNotFoundError:
            if process_start_ticks(pid) == original_ticks:
                raise
            break
        # Read-only waiting: PID + start ticks + argv are checked on each
        # iteration. No process (including a reused PID) is ever signalled.
        old.publish(Path(spec["output"]) / "status.json", dict(state="waiting", time=now(),
            phase="verified_live_dependency", supervisor_pid=os.getpid(), dependency=dependency,
            dependency_start_ticks=original_ticks, policy_updates=0, gpu_allocated=False))
        time.sleep(10)
    state = read(root / "status.json")
    inspection = read(root / "formal_checkpoint_inspection.json")
    if (state["state"] != "complete" or state["verified_optimizer_steps"] != 500
            or not inspection["passed"] or inspection["step"] != 500
            or sha(inspection["checkpoint"]) != inspection["checkpoint_sha256"]):
        raise RuntimeError("Predecessor did not finish and verify; do not start AR or retry it")
    publish(Path(spec["output"]) / "dependency_completed.json", dict(verified=True, time=now(),
        dependency=dependency, inspection_sha256=sha(root / "formal_checkpoint_inspection.json"),
        initializes_from_original_A4_not_dependency=True))


@contextmanager
def original_fm_flags(policy):
    """Explicit prefix-only reference measurement; never used for training."""
    before = policy.discrete_action, policy.continuous_action
    try:
        policy.discrete_action, policy.continuous_action = False, True
        yield
    finally:
        policy.discrete_action, policy.continuous_action = before


def reference_fm(policy, batch):
    import torch
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
    from g05.models.g05.g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
    if torch.is_grad_enabled() or policy.training:
        raise RuntimeError("Reference-only prefix FM may not change the training objective")
    G05PolicyMEMLiteSkillFM._validate_skill_batch(policy, batch["samples"], batch["action"], batch["action_is_pad"])
    with original_fm_flags(policy):
        _, metrics = G05PolicyQwen35.forward_train(policy, batch["samples"], batch["pixel_values"],
            actions=batch["action"], action_pad_masks=batch["action_is_pad"],
            action_dim_is_pad=batch["action_dim_is_pad"])
    return float(metrics["fm_loss"])


def summarize(rows):
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    if set(grouped) != {f"task{i}" for i in range(5)}:
        raise RuntimeError("Diagnostic missed a task")
    per_task = {}
    for task, values in sorted(grouped.items()):
        per_task[task] = {metric: sum(row[metric] for row in values) / len(values)
                         for metric in ("ce_loss", "reference_fm_loss")}
    return dict(per_task=per_task, aggregate={key: sum(v[key] for v in per_task.values()) / 5
                for key in ("ce_loss", "reference_fm_loss")})


def evaluate(model, pipeline, route, step, output, rank, world, *, full):
    import torch
    import torch.distributed as dist
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.training.fixed_diagnostic_evaluator import (
        preserve_fixed_diagnostic_state, replay_official_fm_sampler_seed)
    windows = pipeline.manifest["windows"]
    first_per_task = {}
    for window in windows:
        first_per_task.setdefault(window["task_id"], window["window_id"])
    if not full:
        windows = [w for w in windows if w["window_id"] == first_per_task[w["task_id"]]]
    local = []
    device = next(model.parameters()).device
    with preserve_fixed_diagnostic_state(model):
        for window in windows[rank::world]:
            batch = dict_apply(exact_eval_batch(pipeline, window),
                               lambda t: t.to(device) if isinstance(t, torch.Tensor) else t)
            replay_official_fm_sampler_seed(window["fm_replay_seed"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fm = reference_fm(model, batch)
                replay_official_fm_sampler_seed(window["fm_replay_seed"])
                _, metrics = model(batch)
            row = dict(task_id=window["task_id"], window_id=window["window_id"],
                skill=window["skill"], ce_loss=float(metrics.get("ce_loss", 0.)), reference_fm_loss=fm)
            if not all(torch.isfinite(torch.tensor(row[k])) for k in ("ce_loss", "reference_fm_loss")):
                raise RuntimeError("Nonfinite fixed heldout metric")
            if window["window_id"] == first_per_task[window["task_id"]] and route != "fm":
                replay_official_fm_sampler_seed(17)
                raw = {}
                original = model.model.inference_ar
                def traced(*args, **kwargs):
                    result = original(*args, **kwargs)
                    raw["ids"] = result["generated_ids"].detach().cpu().tolist()
                    return result
                model.model.inference_ar = traced
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model.forward_inference(batch["samples"], batch["pixel_values"],
                            action_dim_is_pad=batch["action_dim_is_pad"])
                    action = prediction["action"].to(device)
                    valid = (~batch["action_is_pad"][:, :16, None]) & (~batch["action_dim_is_pad"][:, None, :])
                    error = (action[:, :16] - batch["action"][:, :16])[valid]
                    row["generation"] = dict(valid=True, source=prediction["selected_action_source"],
                        normalized_executed_rmse=float(error.square().mean().sqrt()), raw=raw)
                except RuntimeError as exc:
                    if not str(exc).startswith("Free AR generation"):
                        raise
                    row["generation"] = dict(valid=False, error=str(exc), raw=raw)
                finally:
                    model.model.inference_ar = original
            local.append(row)
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    rows = sorted([row for subset in gathered for row in subset], key=lambda row: (row["task_id"], row["window_id"]))
    if len(rows) != len(windows) or len({(r["task_id"], r["window_id"]) for r in rows}) != len(windows):
        raise RuntimeError("Incomplete/duplicate distributed diagnostic windows")
    result = dict(complete=True, step=step, windows=len(rows), rows=rows, **summarize(rows),
        fixed_manifest_sha256=FIXED_SHA, route=route, full_heldout=False,
        prefix_only_fm_reference=True, ce_metric_defined=route != "fm", success_rate_claim=False)
    if rank == 0:
        publish(output / f"eval_step_{step}.json", result)
    return result


def capture_rng(pipeline, rank):
    import numpy as np
    import torch
    return dict(rank=rank, python=random.getstate(), numpy=np.random.get_state(),
        cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
        loader=pipeline.loader_generator.get_state())


def assert_saved_state(saved, model, optimizer, scheduler, spec, step, rank_states):
    import torch
    if (saved["step"] != step or saved["action_experiment_spec"] != spec
            or saved["scheduler_state_dict"] != scheduler.state_dict()
            or len(saved["rng_by_rank"]) != 4 or sorted(s["rank"] for s in saved["rng_by_rank"]) != [0, 1, 2, 3]
            or saved["next_microbatch"] != step * 2):
        raise RuntimeError("Saved action training clock/recipe/RNG identity differs")
    expected, actual = model.state_dict(), saved["model_state_dict"]
    if set(expected) != set(actual) or len(actual) != 1138:
        raise RuntimeError("Saved full policy schema differs")
    for name, value in expected.items():
        other = actual[name]
        if value.dtype != other.dtype or not torch.isfinite(other).all() or not torch.equal(value.detach().cpu(), other):
            raise RuntimeError("Checkpoint tensor did not roundtrip: " + name)
    current, states = optimizer.state_dict(), saved["optimizer_state_dict"]
    if current["param_groups"] != states["param_groups"] or set(current["state"]) != set(states["state"]):
        raise RuntimeError("Adam parameter-group identity changed")
    if len(states["state"]) != expected_adam_states(spec["route"]):
        raise RuntimeError("Wrong number of actual Adam states")
    for index, state in states["state"].items():
        if int(state["step"]) != step:
            raise RuntimeError("Wrong saved Adam update count")
        for key, value in state.items():
            if not torch.isfinite(value).all() or not torch.equal(value, current["state"][index][key].cpu()):
                raise RuntimeError("Adam state did not roundtrip")
    for saved_rng, original_rng in zip(saved["rng_by_rank"], rank_states):
        for key in ("cpu", "cuda", "loader"):
            if not torch.equal(saved_rng[key], original_rng[key]):
                raise RuntimeError("Per-rank RNG failed roundtrip")


def train(spec, phase):
    if phase not in ("smoke", "formal"):
        raise ValueError("Unsupported training phase")
    if phase == "formal":
        gate = read(Path(spec["output"]) / "smoke/checkpoint_inspection.json")
        if not gate["passed"] or gate["actual_updates"] != 5:
            raise RuntimeError("Formal phase requires this run's actual DDP saved-state gate")
    identity = bootstrap()
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint, save_training_checkpoint
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.training.coordination_grad_audit import CoordinationGradientAudit
    from g05.utils.training.coordination_receipts import record_coordination_batch_receipt

    register_default_resolvers()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 4 or rank != local_rank:
        raise RuntimeError("This bounded recipe requires one-node four-rank DDP")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local_rank)
    random.seed(41 + rank)
    np.random.seed(41 + rank)
    torch.manual_seed(41 + rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    output = Path(spec["output"]) / phase
    if rank == 0:
        output.mkdir(exist_ok=False)
    dist.barrier()
    config_path = INPUT / "diagnostic_processor_config.yaml"
    if (sha(config_path) != read(INPUT / "result.json")["processor_config_sha256"]
            or sha(PARENT) != PARENT_SHA):
        raise RuntimeError("Original processor recipe or A4 parent differs")
    cfg = OmegaConf.load(config_path)
    arch = configure(cfg, spec["route"])
    pipeline = build_original_pipeline(cfg, rank=rank, world_size=world, workers=RECIPE["workers"])
    identity.update(pipeline=pipeline.identity, config_sha256=sha(config_path), spec=spec)
    publish(output / f"identity_rank{rank}.json", identity)
    if rank == 0:
        OmegaConf.save(cfg, output / "config.yaml")
        publish(output / "train_source_spec.json", pipeline.source_spec)
        shutil.copyfile(STATS, output / "dataset_stats.json")
    print(json.dumps(dict(rank=rank, stage="pipeline_ready", route=spec["route"])), flush=True)
    model, parent = load_model_from_checkpoint(arch, str(PARENT), device=str(device),
                                               eval_mode=False, return_full_checkpoint=True)
    verify_parent(model, parent)
    del parent
    gc.collect()
    contract = prepare_action_updates(model, device)
    publish(output / f"restoration_rank{rank}.json", dict(passed=True, full_parent_entries=1138, contract=contract))
    # Audit ALL frozen parameters, including the action expert on pure AR.
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
    audit = CoordinationGradientAudit(model, expected_groups=contract["expected_trainable_groups"],
        frozen_prefixes=frozen_names, output_dir=output, rank=rank)
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True,
              gradient_as_bucket_view=True, bucket_cap_mb=128)
    audit.after_distributed_initialization()
    optimizer, scheduler = make_optimizer(model)
    total = 5 if phase == "smoke" else 500
    pipeline.sampler.set_epoch(0)
    pipeline.sampler.set_start_batch(0)
    before = evaluate(model, pipeline, spec["route"], 0, output, rank, world, full=phase == "formal")
    # Verify the reference metric against the immutable A4 result, not an
    # untested claim that a new CE/FM path happens to be numerically identical.
    if phase == "formal" and abs(before["aggregate"]["reference_fm_loss"] - .1969439941) > 2e-6:
        raise RuntimeError("A4 prefix-only fixed80 metric differs from the original reference")
    iterator = iter(pipeline.loader)
    params = [p for p in model.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(1, total + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros(3, device=device, dtype=torch.float64)
        for micro in range(2):
            original = next(iterator)
            batch_index = (step - 1) * 2 + micro
            record_coordination_batch_receipt(original, output, rank=rank, step=step,
                batch_index=batch_index, stage="experimental_" + spec["route"],
                source_spec_path=pipeline.source_spec, audit_resolver=pipeline.train_resolver)
            batch = dict_apply(original, lambda t: t.to(device) if isinstance(t, torch.Tensor) else t)
            with (ddp.no_sync() if micro == 0 else nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = ddp(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite real-data training objective")
                if spec["route"] == "ar" and float(metrics["fm_loss"]) != 0.:
                    raise RuntimeError("Pure AR unexpectedly uses FM supervision")
                (loss / 2).backward()
            stats += torch.tensor([float(loss.detach()), float(metrics.get("ce_loss", 0.)),
                                   float(metrics["fm_loss"])], dtype=torch.float64, device=device) / 2
            del batch, original, loss, metrics
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        rates = [float(g["lr"]) for g in optimizer.param_groups]
        facts = audit.after_backward() if step <= 2 else None
        optimizer.step()
        if step <= 2:
            audit.after_optimizer_step(facts, step=step, effective_lrs=rates)
        scheduler.step()
        dist.all_reduce(stats)
        stats /= world
        if rank == 0:
            row = dict(step=step, loss=float(stats[0]), ce_loss=float(stats[1]), fm_loss=float(stats[2]),
                rank0_grad_norm=float(norm), lr=rates, elapsed_seconds=time.monotonic() - started,
                actual_train_rows=step * 16, route=spec["route"])
            with (output / "train_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if step <= 5 or step % 10 == 0:
                print(json.dumps(row), flush=True)
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == total:
            evaluate(model, pipeline, spec["route"], step, output, rank, world, full=phase == "formal")
    audit.require_verified_update()
    audit._assert_frozen_unchanged()
    rng = [None] * world
    dist.all_gather_object(rng, capture_rng(pipeline, rank))
    if rank == 0:
        path = save_training_checkpoint(output, step=total, epoch=0, batch_idx=total * 2 - 1,
            model=model, optimizer=optimizer, scheduler=scheduler, action_experiment_spec=spec,
            model_arch=OmegaConf.to_container(arch, resolve=True), rng_by_rank=rng,
            next_microbatch=total * 2, dataset_stats_sha256=STATS_SHA,
            parameter_group_names=audit.group_name_lists, gradient_evidence_rank0=audit.checkpoint_evidence(),
            resume_scope="Exact saved tensors; stochastic worker/prefetch continuation is not verified")
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert_saved_state(saved, model, optimizer, scheduler, spec, total, rng)
        publish(output / "checkpoint_inspection.json", dict(passed=True, checkpoint=str(path),
            checkpoint_sha256=sha(path), actual_updates=total, actual_adam_states=len(optimizer.state),
            frozen_unchanged=True, full_model_optimizer_rng_roundtrip=True,
            new_process_resume_tested=False, actual_train_rows=total * 16,
            peak_rank0_reserved_bytes=torch.cuda.max_memory_reserved(), success_rate_claim=False))
    dist.barrier()
    dist.destroy_process_group()


def supervise(spec, path):
    old = legacy(dict(output=spec["output"]))
    try:
        # The supervisor/verified wait has no CUDA context. Child environments
        # explicitly select the four GPUs only after the predecessor finishes.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        bootstrap()
        from g05.utils.training.coordination_runtime import CoordinationLock
        wait_for_dependency(spec, old)
        for phase in ("smoke", "formal"):
            if phase == "formal" and not read(Path(spec["output"]) / "smoke/checkpoint_inspection.json")["passed"]:
                raise RuntimeError("Real DDP saved-state inspection is required")
            old.gpu_budget(range(4))
            command = [old.PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                       str(HERE), "train", "--spec", str(path), "--phase", phase]
            with CoordinationLock(purpose="training") as lock:
                def heartbeat(child):
                    lock.heartbeat(metadata=dict(run_dir=str(Path(spec["output"]) / phase), pid=child.pid,
                                                 source_root=str(REPO)))
                    old.publish(Path(spec["output"]) / "status.json", dict(state="running", phase=phase,
                        supervisor_pid=os.getpid(), trainer_pid=child.pid, time=now(), route=spec["route"]))
                old.execute(command, old.environment(), Path(spec["output"]) / f"{phase}.log",
                            1800 if phase == "smoke" else None, heartbeat)
        old.publish(Path(spec["output"]) / "status.json", dict(state="complete", time=now(), route=spec["route"],
            verified_updates=500, success_rate_claim=False, next="Free-generation and matched closed-loop evaluation"))
    except BaseException as exc:
        old.publish(Path(spec["output"]) / "status.json", dict(state="failed", time=now(), error=str(exc),
            automatic_retry=False, prior_checkpoints_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "supervise", "train"))
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--phase", choices=("smoke", "formal"))
    parser.add_argument("--after-fm-run", type=Path)
    args = parser.parse_args()
    if args.mode == "start":
        if args.route is None or args.output is None or args.output.parent != BASE:
            raise ValueError("Select one action route and a new child of the experiment root")
        spec = dict(route=args.route, output=str(args.output), recipe=RECIPE, parent_sha256=PARENT_SHA,
            code=code_identity(), commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            fresh_adam=True, high_unchanged=True, train_split="original950", eval_split="original50",
            native_upstream_ar_replication=False, no_automatic_deployment=True,
            timing_comparable_to_stock_fm=False, prerequisites=prerequisites(args.route),
            after_fm=dependency_identity(args.after_fm_run) if args.after_fm_run else None)
        validate_spec(spec)
        args.output.mkdir(exist_ok=False)
        path = args.output / "method_spec.json"
        publish(path, spec)
        old = legacy(dict(output=str(args.output)))
        with (args.output / "supervisor.log").open("x") as stream:
            child = subprocess.Popen([old.PYTHON, str(HERE), "supervise", "--spec", str(path)],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                env=old.environment(), start_new_session=True)
        result = dict(time=now(), supervisor_pid=child.pid, route=args.route, spec=str(path), spec_sha256=sha(path))
        publish(args.output / "launch.json", result)
        print(json.dumps(result), flush=True)
    else:
        if args.spec is None or not args.spec.is_absolute():
            raise ValueError("Use a pinned absolute method spec")
        spec = read(args.spec)
        validate_spec(spec)
        if args.mode == "supervise":
            supervise(spec, args.spec)
        elif args.phase is None:
            raise ValueError("Explicit smoke/formal phase required")
        else:
            train(spec, args.phase)


if __name__ == "__main__":
    main()
