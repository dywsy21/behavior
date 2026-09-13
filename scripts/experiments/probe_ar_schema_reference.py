"""Same ten windows: original A4 FM generation and codec reconstruction.

Read the completed constrained-AR outputs, do not generate AR again. All
methods are scored against the same valid 0:16/23D targets. Reconstruction is
an offline codec reference, not an actor result or a theoretical error bound.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from action_training_data import build_original_pipeline, exact_eval_batch
from probe_action_training_gpu import prepare_action_updates, task_generation_rows, generation_execution_metrics
from probe_ar_execution_codec import publish
from train_action_method_probe import read, now
from train_fm_method_probe import BASE, PARENT, PARENT_SHA, verify_parent

SCHEMA_RESULT = BASE / "ar_schema_ar500_v2/result.json"
SCHEMA_SHA = "1c9ad37c043182d71db1fd0ad91f855376eb09daefa4342ec523bab331d9bb9d"


def validate_paired_sources(candidates, reference):
    if (reference.get("complete") is not True or reference.get("actual_generations") != 10
            or reference.get("complete_generations") != 10 or len(candidates) != 10
            or len(reference.get("rows", [])) != 10):
        raise ValueError("Need exactly the completed ten constrained-AR windows")
    expected = [f"{split}_task{task}" for split in ("train", "heldout") for task in range(5)]
    if [label for label, _, _ in candidates] != expected:
        raise ValueError("Declared five-train/five-heldout ordering changed")
    for (label, _, source), row in zip(candidates, reference["rows"]):
        if label != row["label"] or source != row["source"] or row.get("valid") is not True:
            raise ValueError("Exact paired AR/reference source differs; do not substitute windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.parent != BASE or args.gpu not in range(4):
        raise ValueError("Use a new experiment child and explicit robo GPU")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned worktree")
    if sha(SCHEMA_RESULT) != SCHEMA_SHA:
        raise RuntimeError("Original completed schema result changed")
    reference = read(SCHEMA_RESULT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output.mkdir(exist_ok=False)
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.training.ar_training_methods import validate_complete_action_tokens

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Need 40 GiB free headroom; do not evict other work")
    torch.cuda.set_per_process_memory_fraction(.4)
    config = INPUT / "diagnostic_processor_config.yaml"
    input_receipt = read(INPUT / "result.json")
    if (sha(PARENT) != PARENT_SHA or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or sha(config) != input_receipt["processor_config_sha256"]
            or input_receipt["status"] != "complete" or not input_receipt["train_only"]):
        raise RuntimeError("Original A4/train-cache/config identity changed")
    cfg = OmegaConf.load(config)
    arch = cfg.model.model_arch
    if (arch._target_ != "g05.models.g05.g05_policy_memlite_skill_fm.G05PolicyMEMLiteSkillFM"
            or arch.discrete_action or not arch.continuous_action or arch.predict_cot):
        raise RuntimeError("Reference must remain the original SkillFM policy, not an AR/KI proxy")
    # Only the external codec reconstruction uses action tokens. Disable noop
    # omission so this offline reference retains both grippers and lower body.
    cfg.tokenizer.vq_config.dropout_noop_parts = False
    arch.AT_CONFIG.dropout_noop_parts = False
    pipeline = build_original_pipeline(cfg, rank=0, world_size=1, workers=0)
    originals = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    candidates = [(f"train_task{task}", batch, {**source, "diagnostic_split": "train"})
                  for task, (batch, source) in sorted(task_generation_rows(originals, input_receipt).items())]
    first = {}
    for window in pipeline.manifest["windows"]:
        first.setdefault(window["task_id"], window)
    candidates.extend((f"heldout_{task}", exact_eval_batch(pipeline, window), {**window, "diagnostic_split": "heldout"})
                      for task, window in sorted(first.items()))
    validate_paired_sources(candidates, reference)
    identity.update(start_time=now(), gpu=args.gpu, entry_sha256=sha(Path(__file__)),
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        parent_sha256=PARENT_SHA, schema_result_sha256=SCHEMA_SHA, input_sha256=INPUT_SHA,
        config_sha256=sha(config), pipeline=pipeline.identity, policy_seed=17,
        architecture=OmegaConf.to_container(arch, resolve=True), original_fm_policy=True,
        fm_generations=10, ar_generations=0, codec_reconstructions=10, optimizer_updates=0, simulator_controls=0,
        codec_reference_uses_gt_outside_actor=True, gt_actions_in_fm_actor=False,
        same_state_supervised_skill_not_autonomous_planner=True, success_rate_claim=False)
    publish(args.output / "manifest.json", identity)
    model, payload = load_model_from_checkpoint(arch, str(PARENT), device="cuda:0",
                                               eval_mode=False, return_full_checkpoint=True)
    verify_parent(model, payload)
    del payload
    gc.collect()
    prepare_action_updates(model, "cuda:0")
    model.eval()
    publish(args.output / "restoration.json", dict(passed=True, parent_sha256=PARENT_SHA,
        exact_state_entries=1138, exact_lora_entries=192))
    rows = []
    for (label, original, source), ar_row in zip(candidates, reference["rows"]):
        batch = dict_apply(deepcopy(original), lambda t: t.to("cuda:0") if isinstance(t, torch.Tensor) else t)
        if any("action" in sample for sample in batch["samples"]):
            raise RuntimeError("FM actor prefix contains action GT")
        with torch.no_grad(), torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            state = model.prefill(batch["samples"], batch["pixel_values"])
            prediction = model.generate_action(state, batch["samples"],
                action_dim_is_pad=batch["action_dim_is_pad"], action_gt=None)
        if prediction["selected_action_source"] != "fm" or "action_ar" in prediction:
            raise RuntimeError("Original continuous reference unexpectedly used AR")
        fm_metrics = generation_execution_metrics(prediction["action"], batch)
        # Separate offline codec call. No teacher token is passed to prefill,
        # generate_action, or any part of the FM neural sampling above.
        with torch.no_grad():
            codec = model.action_tokenizer
            ids = codec._encode_action_indices(batch["action"],
                encode_kwargs={"action_dim_is_pad": batch["action_dim_is_pad"]})[0]
            decoded = codec.decode_token_ids_to_actions(ids, time_horizon=32, action_dim=27,
                decode_kwargs={"is_action_token_space": True})
            complete = validate_complete_action_tokens(torch.tensor(ids) + codec.action_token_begin_idx, codec)
            if decoded.absent_keys:
                raise RuntimeError("Codec reconstruction reference omitted real controls")
            reconstructed = decoded.action.float().unsqueeze(0)
        codec_metrics = generation_execution_metrics(reconstructed, batch)
        actual_ar = torch.tensor(ar_row["generated_action"], dtype=torch.float32)
        ar_metrics = generation_execution_metrics(actual_ar, batch)
        if ar_metrics != {key: ar_row[key] for key in ar_metrics}:
            raise RuntimeError("Saved AR actions/metrics no longer match the exact targets")
        row = dict(label=label, source=source, ar_schema=ar_metrics, a4_fm=fm_metrics,
            codec_reconstruction=codec_metrics, codec_complete=complete,
            a4_fm_generated_action=prediction["action"].detach().cpu().tolist(),
            codec_reconstructed_action=reconstructed.tolist())
        rows.append(row)
        publish(args.output / f"{label}.json", row)
        print(json.dumps({"label": label, **{name: row[name]["normalized_executed_rmse"]
                           for name in ("ar_schema", "a4_fm", "codec_reconstruction")}}), flush=True)
    publish(args.output / "result.json", dict(complete=True, rows=rows, fm_generations=len(rows), ar_generations=0,
        codec_reconstructions=len(rows), optimizer_updates=0, simulator_controls=0, success_rate_claim=False,
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(), limitations=[
        "Ten fixed diagnostic windows, not the full heldout set or a task success estimate.",
        "AR500 and original A4 FM have different training histories; not a matched-compute algorithm trial.",
        "Single fixed policy seed; same-state supervised skills are not autonomous planner outputs.",
        "Codec reconstruction uses GT offline and is not an actor or a theoretical minimum error."]))
    print(json.dumps(dict(complete=True, result_sha256=sha(args.output / "result.json"))), flush=True)


if __name__ == "__main__":
    main()
