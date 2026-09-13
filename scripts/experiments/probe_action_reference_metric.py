"""Two inference-only forwards verify the new prefix-FM reference on real A4.

Reuse the exact two-row original training input and seed771 from the finished
FM-control GPU gate. No updates, new data, policy deployment or simulation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from probe_action_training_gpu import architecture_for, prepare_action_updates
from probe_ar_execution_codec import publish
from train_action_method_probe import reference_fm, read, now
from train_fm_method_probe import BASE, PARENT, PARENT_SHA, verify_parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.parent != BASE or args.gpu not in range(4):
        raise ValueError("Use a new direct child run and one explicit robo GPU")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned reference-probe worktree")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output.mkdir(exist_ok=False)
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    from g05.utils.training.fixed_diagnostic_evaluator import preserve_fixed_diagnostic_state

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Insufficient free GPU memory; do not stop other jobs")
    config = INPUT / "diagnostic_processor_config.yaml"
    reference_path = BASE / "fm_control_v1/gate/reference_metric_gate.json"
    reference = read(reference_path)
    if (sha(PARENT) != PARENT_SHA or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or sha(config) != read(INPUT / "result.json")["processor_config_sha256"]
            or not reference["passed"] or not reference["eval_loss_and_rng_bitwise_equal"]
            or read(BASE / "fm_control_v1/method_spec.json")["parent_sha256"] != PARENT_SHA):
        raise RuntimeError("Original A4/reference/input identity differs")
    identity.update(start_time=now(), gpu=args.gpu, original_input_sha256=INPUT_SHA,
        parent_sha256=PARENT_SHA, reference_result_sha256=sha(reference_path), seed=771,
        optimizer_updates=0, model_forwards=2, simulator_controls=0,
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)),
        reference_entry_sha256=sha(Path(__file__).with_name("train_action_method_probe.py")))
    publish(args.output / "manifest.json", identity)
    cfg = OmegaConf.load(config)
    arch = architecture_for(cfg, ActionTrainingSettings(route="ar"))
    model, parent = load_model_from_checkpoint(arch, str(PARENT), device="cuda:0",
                                               eval_mode=False, return_full_checkpoint=True)
    verify_parent(model, parent)
    del parent
    gc.collect()
    prepare_action_updates(model, "cuda:0")
    batches = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    batch = dict_apply(deepcopy(batches[0]), lambda t: t.to("cuda:0") if isinstance(t, torch.Tensor) else t)
    values = []
    flags_before = model.discrete_action, model.continuous_action
    torch.manual_seed(41)
    rng_before = torch.get_rng_state(), torch.cuda.get_rng_state()
    for _ in range(2):
        with preserve_fixed_diagnostic_state(model):
            torch.manual_seed(771)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values.append(reference_fm(model, batch))
    if flags_before != (model.discrete_action, model.continuous_action):
        raise RuntimeError("Reference evaluation changed the AR training route")
    if not torch.equal(rng_before[0], torch.get_rng_state()) or not torch.equal(rng_before[1], torch.cuda.get_rng_state()):
        raise RuntimeError("Reference evaluation perturbed training RNG")
    result = dict(complete=True, values=values, original_reference=reference["reference_fm_loss"],
        exactly_equal=all(value == reference["reference_fm_loss"] for value in values),
        route_and_rng_restored=True, actual_model_forwards=2, actual_optimizer_updates=0,
        source=identity, no_effect_or_success_rate_claim=True)
    publish(args.output / "result.json", result)
    if not result["exactly_equal"]:
        raise RuntimeError("New prefix-FM reference differs; inspect evidence before formal training")
    print(result["values"], flush=True)


if __name__ == "__main__":
    main()
