"""Ten target-free schema-constrained AR500 calls; zero updates or simulator.

Five original train windows and five original fixed80 heldout windows remain
separately labelled. Codec syntax is forced, payloads are neural predictions.
This does not turn syntax completeness into a learned success-rate claim.
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
from action_training_data import build_original_pipeline, exact_eval_batch
from probe_action_training_gpu import architecture_for, prepare_action_updates, task_generation_rows
from probe_ar_decode_consistency import verify_probe_checkpoint
from probe_ar_marker_learning import AR500_SHA
from probe_ar_execution_codec import publish
from train_action_method_probe import read, now
from train_fm_method_probe import BASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    if args.output.parent != BASE or args.gpu not in range(4):
        raise ValueError("Use a new direct child run and explicit robo GPU")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned worktree")
    run = BASE / "ar_a4_fulltrain_v2"
    checkpoint = run / "formal/checkpoints/step_500.pt"
    inspection = read(run / "formal/checkpoint_inspection.json")
    if (read(run / "status.json").get("state") != "complete" or not inspection["passed"]
            or inspection["actual_updates"] != 500 or inspection["checkpoint_sha256"] != AR500_SHA
            or inspection["checkpoint"] != str(checkpoint)):
        raise RuntimeError("Require the verified completed original AR500, not a marker artifact")
    args.output.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.models.g05.helpers.action_schema_decoding import constrained_action_schema

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Require 40 GiB headroom; do not evict other work")
    torch.cuda.set_per_process_memory_fraction(.4)
    receipt = read(INPUT / "result.json")
    config = INPUT / "diagnostic_processor_config.yaml"
    if (sha(checkpoint) != AR500_SHA or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or sha(config) != receipt["processor_config_sha256"] or receipt["status"] != "complete"
            or not receipt["train_only"]):
        raise RuntimeError("Parent or original train-cache/config identity changed")
    cfg = OmegaConf.load(config)
    settings = ActionTrainingSettings(route="ar", conditioning="skills")
    arch = architecture_for(cfg, settings)
    pipeline = build_original_pipeline(cfg, rank=0, world_size=1, workers=0)
    identity.update(start_time=now(), gpu=args.gpu, entry_sha256=sha(Path(__file__)),
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        checkpoint=str(checkpoint), checkpoint_sha256=AR500_SHA, input_sha256=INPUT_SHA,
        config_sha256=sha(config), original_pipeline=pipeline.identity, settings=settings.as_dict(),
        architecture=OmegaConf.to_container(arch, resolve=True), policy_seed=17,
        train_windows=5, heldout_windows=5, generations=10, max_tokens_per_generation=96,
        optimizer_updates=0, simulator_controls=0, static_format_forced=True,
        payloads_from_neural_logits=True, marker_adapter=False, gt_actions_in_actor=False,
        skill_source="same_state_supervised_condition_for_offline_diagnostic_not_autonomous_planner",
        timing_comparison_valid=False, success_rate_claim=False)
    publish(args.output / "manifest.json", identity)
    print(json.dumps(dict(stage="load", checkpoint_sha256=AR500_SHA)), flush=True)
    model, payload = load_model_from_checkpoint(arch, str(checkpoint), device="cuda:0",
                                               eval_mode=False, return_full_checkpoint=True)
    restoration = verify_probe_checkpoint(model, payload, "ar500")
    del payload
    gc.collect()
    prepare_action_updates(model, "cuda:0")
    model.eval()
    publish(args.output / "restoration.json", restoration)
    originals = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    candidates = [(f"train_task{task}", original, {**source, "diagnostic_split": "train"})
                  for task, (original, source) in sorted(task_generation_rows(originals, receipt).items())]
    first = {}
    for window in pipeline.manifest["windows"]:
        first.setdefault(window["task_id"], window)
    if set(first) != {f"task{index}" for index in range(5)}:
        raise RuntimeError("Fixed heldout manifest did not cover all five tasks")
    candidates.extend((f"heldout_{task}", exact_eval_batch(pipeline, window), {**window, "diagnostic_split": "heldout"})
                      for task, window in sorted(first.items()))
    if len(candidates) != 10:
        raise RuntimeError("Declared finite generation budget changed")
    outputs = []
    for label, original, source in candidates:
        batch = dict_apply(deepcopy(original), lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x)
        raw = {}
        original_ar = model.model.inference_ar
        def traced(*a, **kw):
            result = original_ar(*a, **kw)
            raw["ids"] = result["generated_ids"].detach().cpu().tolist()
            return result
        model.model.inference_ar = traced
        sampler = None
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                with constrained_action_schema(model) as sampler:
                    prediction = model.forward_inference(batch["samples"], batch["pixel_values"],
                                                         action_dim_is_pad=batch["action_dim_is_pad"])
            if raw.get("ids") != [sampler.generated]:
                raise RuntimeError("Actual generated sequence differs from constrained sampling trace")
            valid = ~batch["action_is_pad"][:, :16, None] & ~batch["action_dim_is_pad"][:, None, :]
            errors = (prediction["action"][:, :16] - batch["action"][:, :16])[valid]
            output = dict(valid=True, selected_action_source=prediction["selected_action_source"],
                normalized_executed_rmse=float(errors.square().mean().sqrt()), valid_scalar_targets=int(valid.sum()),
                generated_action=prediction["action"].detach().cpu().tolist(),
                complete_blocks=prediction["ar_complete_block_receipts"], schema=sampler.require_complete())
        except RuntimeError as exc:
            if not str(exc).startswith("Free AR generation"):
                raise
            output = dict(valid=False, error=str(exc))
        finally:
            model.model.inference_ar = original_ar
            publish(args.output / f"{label}_raw.json", dict(raw=raw, trace=sampler.trace if sampler else []))
        output.update(label=label, source=source, raw=raw, trace=sampler.trace if sampler else [])
        outputs.append(output)
        publish(args.output / f"{label}.json", output)
        print(json.dumps(dict(label=label, valid=output["valid"],
            rmse=output.get("normalized_executed_rmse"))), flush=True)
    publish(args.output / "result.json", dict(complete=True, rows=outputs,
        complete_generations=sum(row["valid"] for row in outputs), actual_generations=len(outputs),
        static_format_forced=True, payloads_from_neural_logits=True, checkpoint_sha256=AR500_SHA,
        optimizer_updates=0, simulator_controls=0, success_rate_claim=False,
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        limitations=["Format completeness is imposed, not learned. No unconstrained-policy success claim.",
                     "Skill conditions are supervised same-state conditions, not an autonomous planner.",
                     "Normalized open-loop error is not raw control error or simulator task success."]))
    print(json.dumps(dict(complete=True, result_sha256=sha(args.output / "result.json"))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
