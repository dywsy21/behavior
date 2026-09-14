"""CPU-only group errors from already generated FM actions; zero new inference."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from action_training_data import build_original_pipeline, exact_eval_batch
from probe_action_training_gpu import task_generation_rows
from probe_ar_execution_codec import publish
from probe_fm_screen_actions import SCREEN, A4_RESULT, A4_SHA, interval_metrics
from probe_ar_schema_reference import SCHEMA_RESULT, SCHEMA_SHA, validate_paired_sources
from train_action_method_probe import read, now
from train_fm_method_probe import BASE

SCREEN_RESULT = BASE / "fm_screen_actions_v1/result.json"
SCREEN_SHA = "a9906fa5f4c254a1f5ce7ef0a9ce0456ebf49e8e8d96177f74dda7c51e1adca8"
OUTPUT = BASE / "fm_saved_action_groups_v1"


def action_group_layout(parts, total_dimensions):
    if (not isinstance(parts, dict) or not parts
            or any(not isinstance(k, str) or type(v) is not int or v <= 0 for k, v in parts.items())
            or sum(parts.values()) != total_dimensions):
        raise ValueError("Use the actual ordered parts metadata, not guessed arm/base offsets")
    offset, groups = 0, {}
    for key, width in parts.items():
        groups[key] = (offset, offset + width)
        offset += width
    return groups


def group_scores(action, batch, parts, begin, end):
    import torch
    whole = interval_metrics(action, batch, begin, end)
    groups = action_group_layout(parts, action.shape[-1])
    target = batch["action"].to(action.device).float()
    valid = ((~batch["action_is_pad"][:, begin:end, None]).to(action.device)
             & (~batch["action_dim_is_pad"][:, None, :]).to(action.device))
    delta = action[:, begin:end].float() - target[:, begin:end]
    scores = {}
    for key, (left, right) in groups.items():
        error = delta[..., left:right][valid[..., left:right]]
        count, sse = error.numel(), float(error.double().square().sum())
        dimensions = (~batch["action_dim_is_pad"][:, left:right]).sum(-1).tolist()
        scores[key] = dict(valid_scalar_targets=count, squared_error_sum=sse,
            normalized_rmse=math.sqrt(sse / count) if count else None,
            real_dimensions_per_sample=dimensions, model_interval=[left, right])
    if (sum(item["valid_scalar_targets"] for item in scores.values()) != whole["valid_scalar_targets"]
            or not math.isclose(sum(item["squared_error_sum"] for item in scores.values()),
                                whole["squared_error_sum"], rel_tol=1e-12, abs_tol=1e-12)):
        raise RuntimeError("Group accounting lost or double-counted real action dimensions")
    return dict(interval=[begin, end], whole=whole, groups=scores)


def summarize_groups(rows):
    accumulators = {}
    for row in rows:
        for interval in ("executed", "unexecuted"):
            for group, metric in row[interval]["groups"].items():
                key = (row["run"], row["source"]["diagnostic_split"], interval, group)
                item = accumulators.setdefault(key, dict(run=key[0], split=key[1], interval=key[2], group=group,
                    valid_scalar_targets=0, squared_error_sum=0., windows=0, windows_with_targets=0))
                item["valid_scalar_targets"] += metric["valid_scalar_targets"]
                item["squared_error_sum"] += metric["squared_error_sum"]
                item["windows"] += 1
                item["windows_with_targets"] += metric["valid_scalar_targets"] > 0
    result = []
    for key, item in sorted(accumulators.items()):
        count = item["valid_scalar_targets"]
        item["normalized_rmse"] = math.sqrt(item["squared_error_sum"] / count) if count else None
        total = sum(v["squared_error_sum"] for k, v in accumulators.items() if k[:3] == key[:3])
        item["squared_error_share"] = item["squared_error_sum"] / total if total else None
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output != OUTPUT or OUTPUT.exists():
        raise ValueError("Only the one declared CPU output; do not overwrite or retry")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned source")
    if sha(SCREEN_RESULT) != SCREEN_SHA or sha(A4_RESULT) != A4_SHA or sha(SCHEMA_RESULT) != SCHEMA_SHA:
        raise RuntimeError("Completed prediction identities changed")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("This re-scoring run is CPU-only")
    config = INPUT / "diagnostic_processor_config.yaml"
    input_receipt = read(INPUT / "result.json")
    if sha(config) != input_receipt["processor_config_sha256"] or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA:
        raise RuntimeError("Original input/config identity changed")
    cfg = OmegaConf.load(config)
    parts = OmegaConf.to_container(cfg.model.model_arch.AT_CONFIG.parts_meta, resolve=True)
    other_parts = OmegaConf.to_container(cfg.tokenizer.vq_config.parts_meta, resolve=True)
    if list(parts.items()) != list(other_parts.items()) or sum(parts.values()) != 27:
        raise RuntimeError("Actual model/tokenizer group layout differs")
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
    validate_paired_sources(candidates, read(SCHEMA_RESULT))
    screen, cached = read(SCREEN_RESULT), read(A4_RESULT)
    if (not screen["complete"] or screen["fm_generations"] != 30 or len(screen["rows"]) != 30
            or not cached["complete"] or cached["fm_generations"] != 10 or len(cached["rows"]) != 10):
        raise RuntimeError("Expected the exact existing30 plus cached10 predictions")
    indexed = {(r["run"], r["label"]): r for r in screen["rows"]}
    if len(indexed) != 30:
        raise RuntimeError("Duplicate saved predictions")
    OUTPUT.mkdir(exist_ok=False)
    identity.update(start_time=now(), entry_sha256=sha(Path(__file__)), config_sha256=sha(config),
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        parts_meta=parts, pipeline=pipeline.identity, existing_screen_sha256=SCREEN_SHA,
        cached_a4_sha256=A4_SHA, original_target_windows=10, existing_predictions_scored=40,
        vlm_forwards=0, fm_generations=0, ar_generations=0, codec_calls=0, optimizer_updates=0,
        simulator_controls=0, new_annotation_release=False)
    publish(OUTPUT / "manifest.json", identity)
    rows = []
    for (label, batch, source), prior in zip(candidates, cached["rows"]):
        if (prior["label"], prior["source"]) != (label, source):
            raise RuntimeError("Original A4 pairing changed")
        predictions = [("a4_parent", prior["a4_fm_generated_action"], None)]
        for name, ckpt_sha in SCREEN.items():
            item = indexed[(name, label)]
            if item["source"] != source or item["checkpoint_sha256"] != ckpt_sha:
                raise RuntimeError("Saved screen source/weight pairing changed")
            predictions.append((name, item["generated_action"], item))
        for name, saved_action, saved_metrics in predictions:
            action = torch.tensor(saved_action, dtype=torch.float32)
            row = dict(run=name, label=label, source=source,
                executed=group_scores(action, batch, parts, 0, 16),
                unexecuted=group_scores(action, batch, parts, 16, 32))
            for interval in ("executed", "unexecuted"):
                if saved_metrics is not None:
                    actual, original = row[interval]["whole"], saved_metrics[interval]
                    if (actual["valid_scalar_targets"] != original["valid_scalar_targets"]
                            or not math.isclose(actual["squared_error_sum"], original["squared_error_sum"],
                                                rel_tol=1e-6, abs_tol=1e-7)):
                        raise RuntimeError("CPU re-scoring no longer reproduces saved targets/whole metric")
            if name == "a4_parent" and not math.isclose(row["executed"]["whole"]["normalized_rmse"],
                    prior["a4_fm"]["normalized_executed_rmse"], rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError("CPU re-scoring no longer reproduces cached A4 score")
            publish(OUTPUT / (name + "_" + label + ".json"), row)
            rows.append(row)
    result = dict(complete=True, parts_meta=parts, original_target_windows=10, existing_predictions_scored=len(rows),
        rows=rows, aggregate=summarize_groups(rows), vlm_forwards=0, fm_generations=0, ar_generations=0,
        codec_calls=0, optimizer_updates=0, simulator_controls=0, new_annotation_release=False,
        success_rate_claim=False, limitations=[
            "Group SSE shares depend on group width, normalization and the small sampled target support.",
            "These are normalized action errors, not per-group physical importance or gradient-conflict evidence.",
            "Existing five-train/five-heldout diagnostic windows cannot prove50-task generality or causal gain."])
    publish(OUTPUT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "aggregate"}}), flush=True)


if __name__ == "__main__":
    main()
