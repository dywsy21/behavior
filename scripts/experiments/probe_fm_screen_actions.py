"""Thirty actual FM generations compare the completed finite FM screens.

Reuse the ten already audited train/heldout sources, and cached A4 predictions.
Score unweighted executed 0:16 and unexecuted 16:32 separately. No new AR/codec
calls, optimizer updates, simulator actions, teacher actions in the actor, or
claim that action RMSE proves success rate.
"""
from copy import deepcopy
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from action_training_data import build_original_pipeline, exact_eval_batch
from probe_action_training_gpu import prepare_action_updates, task_generation_rows, generation_execution_metrics
from probe_ar_execution_codec import publish
from probe_ar_schema_reference import SCHEMA_RESULT, SCHEMA_SHA, validate_paired_sources
from train_action_method_probe import read, now
from train_fm_method_probe import BASE, PARENT_SHA

SCREEN = {
    "fm_control_v1": "def222a6674e6ac92e6ee982c22836b789240f1542c459d5de2111cd646a244e",
    "fm_beta_stratified_v2": "05ea17bcbbf62437b70e286ce58e6113f20cfe1dd7b79eb56010b73d1f7a6c1e",
    "fm_exec_weight2_v2": "101c4b32d7df3bacd529159474f5c1f1720ade2a93614b5d68ea43de5e7eca2f",
}
A4_RESULT = BASE / "ar_schema_reference_a4_v1/result.json"
A4_SHA = "6fdd96d8738fe8fe13682b7b97a7cc5bc1b1d08c49318da98b3246b1238fe5ab"


def interval_metrics(action, batch, begin, end):
    import torch
    if (type(begin) is not int or type(end) is not int or not 0 <= begin < end <= 32
            or action.shape != batch["action"].shape or action.ndim != 3 or action.shape[1:] != (32, 27)
            or not torch.isfinite(action).all() or not torch.isfinite(batch["action"]).all()
            or batch["action_is_pad"].dtype != torch.bool or batch["action_is_pad"].shape != action.shape[:2]
            or batch["action_dim_is_pad"].dtype != torch.bool or batch["action_dim_is_pad"].shape != (action.shape[0], 27)):
        raise ValueError("Invalid unchanged horizon/target/support for generated-action comparison")
    target = batch["action"].to(action.device).float()
    valid = (~batch["action_is_pad"][:, begin:end, None]).to(action.device) & (
             ~batch["action_dim_is_pad"][:, None, :]).to(action.device)
    error = (action[:, begin:end].float() - target[:, begin:end])[valid]
    count = int(error.numel())
    # Sum in FP64 for downstream pooling; retain individual scalar counts.
    sse = float(error.double().square().sum())
    return dict(defined=count > 0, valid_scalar_targets=count, squared_error_sum=sse,
                normalized_rmse=(sse / count)**.5 if count else None, interval=[begin, end], weighted=False)


def verify_screen_state(model, saved):
    import torch
    actual, expected = model.state_dict(), saved["model_state_dict"]
    if (saved["step"] != 500 or set(actual) != set(expected) or len(actual) != 1138
            or sum("lora_" in name for name in actual) != 192):
        raise RuntimeError("Expected the complete 500-step FM model and 192 LoRA tensors")
    for name, tensor in actual.items():
        a, b = tensor.detach().cpu(), expected[name].cpu()
        if (a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all()
                or not torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))):
            raise RuntimeError("Actual full FM500 state restoration differs: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    if args.output != BASE / "fm_screen_actions_v1" or args.gpu != 1:
        raise ValueError("Only the declared thirty-generation/GPU1 comparison is allowed")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean immutable worktree")
    if sha(A4_RESULT) != A4_SHA or sha(SCHEMA_RESULT) != SCHEMA_SHA:
        raise RuntimeError("Completed original reference identities changed")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.common.pytorch_utils import dict_apply

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Need 40 GiB free; do not stop other work")
    torch.cuda.set_per_process_memory_fraction(.4)
    config = INPUT / "diagnostic_processor_config.yaml"
    input_receipt = read(INPUT / "result.json")
    if sha(config) != input_receipt["processor_config_sha256"] or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA:
        raise RuntimeError("Original processor/train sources changed")
    references, cached_a4 = read(SCHEMA_RESULT), read(A4_RESULT)
    cfg = OmegaConf.load(config)
    cfg.tokenizer.vq_config.dropout_noop_parts = False
    cfg.model.model_arch.AT_CONFIG.dropout_noop_parts = False
    pipeline = build_original_pipeline(cfg, rank=0, world_size=1, workers=0)
    originals = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    candidates = [(f"train_task{task}", batch, {**source, "diagnostic_split": "train"})
                  for task, (batch, source) in sorted(task_generation_rows(originals, input_receipt).items())]
    first = {}
    for window in pipeline.manifest["windows"]:
        first.setdefault(window["task_id"], window)
    candidates.extend((f"heldout_{task}", exact_eval_batch(pipeline, window), {**window, "diagnostic_split": "heldout"})
                      for task, window in sorted(first.items()))
    validate_paired_sources(candidates, references)
    if (not cached_a4["complete"] or cached_a4["fm_generations"] != 10 or len(cached_a4["rows"]) != 10
            or any((label, source) != (r["label"], r["source"])
                   for (label, _, source), r in zip(candidates, cached_a4["rows"]))):
        raise RuntimeError("Original A4 predictions do not use the exact paired ten sources")
    args.output.mkdir(exist_ok=False)
    identity.update(start_time=now(), entry_sha256=sha(Path(__file__)), pipeline=pipeline.identity,
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        screen=SCREEN, config_sha256=sha(config), a4_cached_result_sha256=A4_SHA, policy_seed=17,
        fm_generation_budget=30, a4_new_generations=0, ar_generations=0, codec_calls=0,
        optimizer_updates=0, simulator_controls=0, no_teacher_actions_to_actor=True, new_data_release=False)
    publish(args.output / "manifest.json", identity)
    rows = []
    for name, digest in SCREEN.items():
        root = BASE / name
        inspection, status = read(root / "formal_checkpoint_inspection.json"), read(root / "status.json")
        path = root / "formal/checkpoints/step_500.pt"
        if (status["state"] != "complete" or not inspection["passed"] or inspection["step"] != 500
                or inspection["checkpoint_sha256"] != digest or inspection["parent_sha256"] != PARENT_SHA
                or inspection["actual_adam_states"] != 504 or not inspection["frozen_bitwise_unchanged"]
                or sha(path) != digest):
            raise RuntimeError("Unverified or different completed screen checkpoint")
        arch = deepcopy(cfg.model.model_arch)
        if arch.discrete_action or not arch.continuous_action or arch.predict_cot:
            raise RuntimeError("Comparison must remain original FM inference")
        model, saved = load_model_from_checkpoint(arch, str(path), device="cuda:0", eval_mode=False, return_full_checkpoint=True)
        verify_screen_state(model, saved)
        del saved
        gc.collect()
        prepare_action_updates(model, "cuda:0")
        model.eval()
        publish(args.output / f"{name}_restoration.json", dict(passed=True, checkpoint_sha256=digest,
            complete_state_entries=1138, lora_entries=192, step=500))
        for (label, original, source), prior in zip(candidates, cached_a4["rows"]):
            batch = dict_apply(deepcopy(original), lambda t: t.to("cuda:0") if isinstance(t, torch.Tensor) else t)
            if any("action" in sample for sample in batch["samples"]):
                raise RuntimeError("Teacher action in actor prefix")
            with torch.no_grad(), torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(17)
                state = model.prefill(batch["samples"], batch["pixel_values"])
                prediction = model.generate_action(state, batch["samples"], action_dim_is_pad=batch["action_dim_is_pad"], action_gt=None)
            if prediction["selected_action_source"] != "fm" or "action_ar" in prediction:
                raise RuntimeError("Unexpected non-FM inference branch")
            action = prediction["action"]
            old_action = torch.tensor(prior["a4_fm_generated_action"], dtype=torch.float32, device=action.device)
            old_score = generation_execution_metrics(old_action, batch)
            if abs(old_score["normalized_executed_rmse"] - prior["a4_fm"]["normalized_executed_rmse"]) > 1e-6:
                raise RuntimeError("Cached A4 score does not reproduce against the original target")
            row = dict(run=name, checkpoint_sha256=digest, label=label, source=source,
                executed=interval_metrics(action, batch, 0, 16), unexecuted=interval_metrics(action, batch, 16, 32),
                a4_executed=interval_metrics(old_action, batch, 0, 16), a4_unexecuted=interval_metrics(old_action, batch, 16, 32),
                generated_action=action.detach().cpu().tolist())
            publish(args.output / f"{name}_{label}.json", row)
            rows.append(row)
            print(json.dumps(dict(run=name, label=label, executed_rmse=row["executed"]["normalized_rmse"])), flush=True)
        del batch, action, old_action, state, prediction, model
        gc.collect()
        torch.cuda.empty_cache()
    if len(rows) != 30:
        raise RuntimeError("Unexpected action comparison budget")
    publish(args.output / "result.json", dict(complete=True, fm_generations=30, optimizer_updates=0, simulator_controls=0,
        rows=rows, no_data_release=True, success_rate_claim=False,
        limitations=["Only five original train/five existing heldout diagnostic windows and policy seed17.",
            "Same-state supervised skill conditioning, not an autonomous planner or closed-loop evaluation.",
            "A4 is a cached parent reference; only the three500 arms are matched training-budget candidates."]))


if __name__ == "__main__":
    main()
