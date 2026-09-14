"""Eight zero-update forwards locate the one-window initial FM discrepancy.

Compare the original SkillFM call, the auxiliary reference call, and AR's
reference using identical A4 bytes/input/noise. No tolerance changes, training,
data release, free generation, or simulation. Preserve failed run evidence.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

from action_training_runtime import REPO, INPUT, bootstrap, sha
from action_training_data import build_original_pipeline, exact_eval_batch
from probe_ar_execution_codec import publish
from probe_action_training_gpu import prepare_action_updates
from train_action_method_probe import configure, reference_fm, read, now
from train_fm_method_probe import BASE, PARENT, PARENT_SHA, verify_parent

WINDOW = "task4-74-990-2895"


def fingerprint(value):
    """Small tensor identity records, including recurrent-cache content/layout."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()),
                    sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value, np.ndarray):
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    if hasattr(value, "key_cache"):
        return dict(type=type(value).__name__, **{k: fingerprint(getattr(value, k)) for k in
            ("key_cache", "value_cache", "conv_states", "recurrent_states", "split_recurrent_states", "last_linear_layer")})
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Untraced diagnostic input: " + str(type(value)))


@contextmanager
def trace_forward(policy, evidence):
    helper = policy.model.fm_helper
    original = helper.cal_fm_loss
    def traced_loss(*args, **kwargs):
        evidence["fm_loss_inputs"] = fingerprint(dict(args=args, kwargs=kwargs))
        loss = original(*args, **kwargs)
        evidence["fm_loss_result"] = fingerprint(loss.reshape(1))
        return loss
    def before_vlm(_, args, kwargs):
        evidence["model_inputs"] = fingerprint(dict(args=args, kwargs=kwargs))
    def before_action(_, args, kwargs):
        evidence["action_expert_inputs"] = fingerprint(dict(args=args, kwargs=kwargs))
    hooks = [policy.model.register_forward_pre_hook(before_vlm, with_kwargs=True),
             policy.model.action_expert.register_forward_pre_hook(before_action, with_kwargs=True)]
    helper.cal_fm_loss = traced_loss
    try:
        yield
    finally:
        helper.cal_fm_loss = original
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    if args.output != BASE / "fm_initial_reference_probe_v1" or args.gpu != 1:
        raise ValueError("Only the declared single-window/GPU1/one-run probe is allowed")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned worktree")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    identity = bootstrap()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.training.fixed_diagnostic_evaluator import preserve_fixed_diagnostic_state, replay_official_fm_sampler_seed

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Need 40 GiB free; do not evict other work")
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    cfg_path = INPUT / "diagnostic_processor_config.yaml"
    if sha(cfg_path) != read(INPUT / "result.json")["processor_config_sha256"] or sha(PARENT) != PARENT_SHA:
        raise RuntimeError("Original configuration/parent changed")
    previous = {route: BASE / name / "formal/eval_step_0.json" for route, name in
        (("fm", "fm_action_control_v2"), ("ar", "ar_a4_fulltrain_v2"))}
    recorded = {route: next(r for r in read(path)["rows"] if r["window_id"] == WINDOW)
                for route, path in previous.items()}
    args.output.mkdir(exist_ok=False)
    identity.update(start_time=now(), commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), parent_sha256=PARENT_SHA, config_sha256=sha(cfg_path),
        previous={k: dict(path=str(v), sha256=sha(v)) for k, v in previous.items()}, recorded=recorded,
        window_id=WINDOW, gpu=1, forward_budget=8, optimizer_updates=0, simulator_controls=0, no_data_release=True)
    publish(args.output / "manifest.json", identity)
    cfg = OmegaConf.load(cfg_path)
    configure(cfg, "fm")
    pipeline = build_original_pipeline(cfg, rank=0, world_size=1, workers=0)
    window = next(w for w in pipeline.manifest["windows"] if w["window_id"] == WINDOW)
    batch = None
    inputs = []
    for seed in (41, 17, 771):
        random.seed(seed)
        np.random.seed(seed)
        replay_official_fm_sampler_seed(seed)
        observed = exact_eval_batch(pipeline, window)
        inputs.append(dict(seed_before_load=seed, identity=fingerprint(observed)))
        if batch is None:
            batch = observed
    publish(args.output / "input_repeats.json", dict(window=window, repeats=inputs,
        identical=all(x["identity"] == inputs[0]["identity"] for x in inputs)))
    records = []
    for route in ("fm", "ar"):
        cfg = OmegaConf.load(cfg_path)
        arch = configure(cfg, route)
        torch.manual_seed(41)
        model, parent = load_model_from_checkpoint(arch, str(PARENT), device="cuda:0", eval_mode=False, return_full_checkpoint=True)
        verify_parent(model, parent)
        del parent
        gc.collect()
        contract = prepare_action_updates(model, "cuda:0")
        publish(args.output / f"{route}_restoration.json", dict(passed=True, contract=contract,
            model_dtypes={str(dtype): sum(1 for p in model.parameters() if p.dtype == dtype)
                          for dtype in {p.dtype for p in model.parameters()}}))
        model_batch = dict_apply(deepcopy(batch), lambda v: v.to("cuda:0") if isinstance(v, torch.Tensor) else v)
        variants = ("reference", "original_call", "reference_repeat", "toggle_joint") if route == "fm" else (
            "reference", "reference_repeat", "toggle_joint", "ae_requires_grad")
        for variant in variants:
            helper = model.model.fm_helper
            old_joint = helper.joint_training
            flags = [(p, p.requires_grad) for p in model.model.action_expert.parameters()]
            evidence = dict(route=route, variant=variant)
            try:
                if variant == "toggle_joint":
                    helper.joint_training = not old_joint
                if variant == "ae_requires_grad":
                    for parameter, _ in flags:
                        parameter.requires_grad_(True)
                with preserve_fixed_diagnostic_state(model):
                    replay_official_fm_sampler_seed(window["fm_replay_seed"])
                    with trace_forward(model, evidence), torch.autocast("cuda", dtype=torch.bfloat16):
                        if variant == "original_call":
                            _, metrics = model(deepcopy(model_batch))
                            value = float(metrics["fm_loss"])
                        else:
                            value = reference_fm(model, deepcopy(model_batch))
                    evidence.update(value=value, cpu_rng=fingerprint(torch.get_rng_state()), cuda_rng=fingerprint(torch.cuda.get_rng_state()))
            finally:
                helper.joint_training = old_joint
                for parameter, required in flags:
                    parameter.requires_grad_(required)
            if not torch.isfinite(torch.tensor(value)):
                raise RuntimeError("Nonfinite probe")
            records.append(evidence)
            publish(args.output / f"forward_{len(records):02d}.json", evidence)
            print(json.dumps(dict(forward=len(records), route=route, variant=variant, value=value)), flush=True)
        del model_batch, model, contract, helper, flags, parameter
        gc.collect()
        torch.cuda.empty_cache()
    if len(records) != 8:
        raise RuntimeError("Unexpected probe budget")
    publish(args.output / "result.json", dict(complete=True, actual_forwards=8, optimizer_updates=0, simulator_controls=0,
        input_reads_identical=all(x["identity"] == inputs[0]["identity"] for x in inputs),
        rows=[{k: r[k] for k in ("route", "variant", "value")} for r in records],
        reference_check_relaxed=False, training_admissible=False, no_effect_or_success_claim=True))


if __name__ == "__main__":
    main()
