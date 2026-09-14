"""Ten free-AR generations from the completed marker500, paired to cached FM.

No training, schema repair, teacher-action encoding, new FM reference, physics,
automatic retries or promotion. Invalid generations remain explicit failures.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from action_training_runtime import bootstrap, EXTENSIONS, INPUT, INPUT_SHA, REPO, sha
from action_training_data import build_original_pipeline, exact_eval_batch
from probe_action_training_gpu import prepare_action_updates, task_generation_rows, generation_execution_metrics
from probe_ar_execution_codec import publish
from probe_ar_schema_reference import SCHEMA_RESULT, SCHEMA_SHA, validate_paired_sources
from probe_fm_screen_actions import A4_RESULT, A4_SHA, interval_metrics
from probe_m04_actions import tensor_fingerprint
from train_action_method_probe import read, now, RECIPE, MARKER_RECIPE, verify_dependency_completed
from train_fm_method_probe import BASE, PARENT_SHA

RUN = "ar_a4_marker_fulltrain_v3"
METHOD_SHA = "9401ca0c526853e756e7f44b5a7fdf6879346cef5dfc96509f84b8d1ea897bd0"
OUTPUT = BASE / "ar_marker_actions_v1"
FM_RESULT = BASE / "m04_actions_fm_v1/result.json"
FM_SHA = "d3502794bb1d3ff7c27d8838c55c331a8faba22b23dd32ff82c4fdf2d87193c7"
MARKER_IDS = (252173, 252174, 252175, 252176, 252177, 252178, 252185, 252186)
EMA_NAME = "g05.utils.training.trainable_parameter_ema"
EMA_LINE = b'    "g05.utils.training.trainable_parameter_ema": "src/g05/utils/training/trainable_parameter_ema.py",\n'


def validate_training(method, inspection):
    if (method.get("route") != "ar" or method.get("initialization") != "a4"
            or method.get("conditioning") != "skills" or method.get("marker_rows") is not True
            or method.get("recipe") != RECIPE or method.get("marker_recipe") != MARKER_RECIPE
            or method.get("parent_sha256") != PARENT_SHA or method.get("train_split") != "original950"
            or method.get("eval_split") != "original50" or not inspection.get("passed")
            or inspection.get("actual_updates") != 500 or inspection.get("actual_train_rows") != 8000
            or inspection.get("actual_adam_states") != 193 or not inspection.get("frozen_unchanged")
            or not inspection.get("full_model_optimizer_rng_roundtrip")):
        raise RuntimeError("Require the exact completed marker500 recipe and full saved-state proof")


def original_runtime_digest(content):
    # Only this unused helper registration was added after marker training.
    # Removing exactly one literal line must reproduce the old ENTIRE file.
    if content.count(EMA_LINE) != 1:
        raise RuntimeError("Expected exactly one unused EMA registration, not a generic source waiver")
    return hashlib.sha256(content.replace(EMA_LINE, b"", 1)).hexdigest()


def validate_sources(method):
    paths = [path for name, path in EXTENSIONS.items() if name != EMA_NAME]
    paths += ["scripts/experiments/action_training_data.py", "scripts/experiments/probe_action_training_gpu.py",
              "scripts/experiments/probe_ar_execution_codec.py"]
    if any(sha(REPO / path) != method["code"].get(path) for path in paths):
        raise RuntimeError("Inference/data code differs from the original marker training")
    runtime = "scripts/experiments/action_training_runtime.py"
    if original_runtime_digest((REPO / runtime).read_bytes()) != method["code"][runtime]:
        raise RuntimeError("Runtime changed beyond the single unused EMA registration")
    return dict(exact_training_files=paths, old_runtime_sha256=method["code"][runtime],
                allowed_difference="one unused EMA module registration; no EMA object or evaluation")


def verify_marker_state(model, saved):
    import torch
    actual, expected = model.state_dict(), saved["model_state_dict"]
    if (saved.get("step") != 500 or set(actual) != set(expected) or len(actual) != 1140
            or sum("lora_" in name for name in actual) != 192):
        raise RuntimeError("Require full1140 state,192 LoRA and two marker entries at500")
    ids = actual.get("model.action_marker_rows.token_ids")
    delta = actual.get("model.action_marker_rows.delta")
    if (ids is None or ids.dtype != torch.long or ids.cpu().tolist() != list(MARKER_IDS)
            or delta is None or delta.shape != (8, 2048) or not torch.count_nonzero(delta)):
        raise RuntimeError("Missing trained eight-row marker adapter or wrong codec identity")
    for name, tensor in actual.items():
        a, b = tensor.detach().cpu(), expected[name].cpu()
        if (a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all()
                or not torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                   b.contiguous().reshape(-1).view(torch.uint8))):
            raise RuntimeError("Full marker state restoration differs: " + name)


@contextmanager
def observe_free_ar(model):
    ar, fm = model.model.inference_ar, model.model.inference_fm
    encode = model.action_tokenizer._encode_action_indices
    record = dict(ar=0, fm=0, teacher_encodes=0, calls=[])

    def observed_ar(*args, **kwargs):
        record["ar"] += 1
        if (record["ar"] != 1 or kwargs.get("max_new_tokens") != 300
                or kwargs.get("logits_processor") is not None or kwargs.get("logits_processors") is not None):
            raise RuntimeError("Require one unchanged bounded free-AR action call, no schema processor")
        result = ar(*args, **kwargs)
        ids = result["generated_ids"].detach().cpu()
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 300:
            raise RuntimeError("Actual AR generation escaped its declared token budget")
        record["calls"].append(dict(ids=ids.tolist(), max_new_tokens=kwargs["max_new_tokens"],
                                    stop_token_ids=kwargs.get("stop_token_ids")))
        return result

    def forbidden_fm(*args, **kwargs):
        record["fm"] += 1
        raise RuntimeError("FM fallback is forbidden in the free-AR comparison")

    def forbidden_encode(*args, **kwargs):
        record["teacher_encodes"] += 1
        raise RuntimeError("Encoding teacher actions is forbidden in this actor")

    model.model.inference_ar, model.model.inference_fm = observed_ar, forbidden_fm
    model.action_tokenizer._encode_action_indices = forbidden_encode
    try:
        yield record
    finally:
        model.model.inference_ar, model.model.inference_fm = ar, fm
        model.action_tokenizer._encode_action_indices = encode


def infer_free_ar(model, batch):
    import torch
    if (model.training or model.predict_cot or not model.discrete_action or model.continuous_action
            or model.action_training.route != "ar" or model.action_training.conditioning != "skills"):
        raise ValueError("Only eval-mode skill-conditioned pure free AR is declared")
    mask = batch["action_dim_is_pad"]
    expected = torch.zeros(1, 27, dtype=torch.bool, device=mask.device)
    expected[:, [7, 8, 17, 18]] = True
    if (mask.dtype != torch.bool or mask.shape != expected.shape or not torch.equal(mask, expected)
            or len(batch["samples"]) != 1 or any(any(key in sample for key in
                ("action", "gt_action", "future_state", "teacher_action")) for sample in batch["samples"])):
        raise ValueError("Invalid original23D metadata or teacher field in actor prefix")
    action, failure = None, None
    with observe_free_ar(model) as calls:
        try:
            result = model.forward_inference(batch["samples"], batch["pixel_values"], action_dim_is_pad=mask)
        except RuntimeError as exc:
            if not str(exc).startswith("Free AR generation"):
                raise
            failure = str(exc)
        else:
            action = result["action"]
            if (result.get("selected_action_source") != "ar" or action.shape != (1, 32, 27)
                    or not torch.isfinite(action).all() or result.get("execution_start") != 0
                    or result.get("execution_steps") != 16 or not result.get("ar_complete_block_receipts")):
                raise RuntimeError("Invalid AR source, full codec validation or execution clock")
    if calls["ar"] != 1 or calls["fm"] or calls["teacher_encodes"] or len(calls["calls"]) != 1:
        raise RuntimeError("Actual free-AR call accounting failed")
    return action, dict(valid=action is not None, failure=failure, calls=calls,
                       selected_action_source="ar" if action is not None else None)


def paired_input_fingerprints(batch, baseline):
    actual = dict(pixel_fingerprints=tensor_fingerprint(batch["pixel_values"]),
        proprio_fingerprints=tensor_fingerprint(batch["samples"][0]["proprio"]),
        mask_fingerprints=tensor_fingerprint({key: batch[key] for key in ("action_is_pad", "action_dim_is_pad")}))
    if any(baseline.get(key) != value for key, value in actual.items()):
        raise RuntimeError("Actual processed inputs differ from cached M04 FM; do not call AR on substitutes")
    return actual


def verify_cached_score(actual, expected):
    # CPU vs GPU FP64 reductions can use different addition orders. Counts,
    # support and weighting are exact; only round-off in the sum is tolerated.
    exact = ("defined", "valid_scalar_targets", "interval", "weighted")
    if any(actual[key] != expected[key] for key in exact):
        raise RuntimeError("Cached score support/definition changed")
    if not math.isclose(actual["squared_error_sum"], expected["squared_error_sum"], rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError("Cached action SSE differs against the actual target")
    a, b = actual["normalized_rmse"], expected["normalized_rmse"]
    if (a is None) != (b is None) or a is not None and not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError("Cached action RMSE differs against the actual target")


def pooled_comparison(rows):
    """Missing AR actions are failures, never zero-error rows or FM-filled actions."""
    result = {}
    for split in ("train", "heldout"):
        selected = [row for row in rows if row["source"]["diagnostic_split"] == split]
        if len(selected) != 5:
            raise ValueError("Require exactly five attempted rows per split")
        valid = [row for row in selected if row["valid"]]
        pools = {}
        for interval in ("executed", "unexecuted"):
            for prefix in ("", "fm_", "a4_"):
                metrics = [row[prefix + interval] for row in valid]
                count = sum(metric["valid_scalar_targets"] for metric in metrics)
                sse = sum(metric["squared_error_sum"] for metric in metrics)
                pools[prefix + interval] = dict(valid_scalar_targets=count, squared_error_sum=sse,
                                               normalized_rmse=(sse / count)**.5 if count else None)
        result[split] = dict(attempts=5, valid=len(valid), invalid=5 - len(valid),
            paired_labels=[row["label"] for row in valid], paired_valid_subset=pools,
            invalid_labels=[row["label"] for row in selected if not row["valid"]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu != 1 or OUTPUT.exists():
        raise ValueError("Only GPU1 and the unique unused marker output; no automatic retry")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned source")
    root = BASE / RUN
    for path, digest in ((root / "method_spec.json", METHOD_SHA), (FM_RESULT, FM_SHA),
                         (A4_RESULT, A4_SHA), (SCHEMA_RESULT, SCHEMA_SHA)):
        if sha(path) != digest:
            raise RuntimeError("Training or cached reference identity changed: " + str(path))
    inspection_path = verify_dependency_completed(root, "action")
    method, inspection = read(root / "method_spec.json"), read(inspection_path)
    validate_training(method, inspection)
    code_receipt = validate_sources(method)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.models.g05.helpers.action_marker_rows import assert_marker_bindings, required_marker_ids

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError("Need40GiB free; do not stop other work")
    torch.cuda.set_per_process_memory_fraction(.4)
    config = INPUT / "diagnostic_processor_config.yaml"
    source_receipt = read(INPUT / "result.json")
    if sha(config) != source_receipt["processor_config_sha256"] or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA:
        raise RuntimeError("Original input/config identity changed")
    cfg = OmegaConf.load(config)
    cfg.tokenizer.vq_config.dropout_noop_parts = False
    cfg.model.model_arch.AT_CONFIG.dropout_noop_parts = False
    pipeline = build_original_pipeline(cfg, rank=0, world_size=1, workers=0)
    originals = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    candidates = [(f"train_task{task}", batch, {**source, "diagnostic_split": "train"})
                  for task, (batch, source) in sorted(task_generation_rows(originals, source_receipt).items())]
    first = {}
    for window in pipeline.manifest["windows"]:
        first.setdefault(window["task_id"], window)
    candidates.extend((f"heldout_{task}", exact_eval_batch(pipeline, window), {**window, "diagnostic_split": "heldout"})
                      for task, window in sorted(first.items()))
    validate_paired_sources(candidates, read(SCHEMA_RESULT))
    fm, a4 = read(FM_RESULT), read(A4_RESULT)
    if (not fm["complete"] or fm["fm_generations"] != 10 or len(fm["rows"]) != 10
            or not a4["complete"] or a4["fm_generations"] != 10 or len(a4["rows"]) != 10):
        raise RuntimeError("Need the exact completed ten-window cached references")
    for (label, batch, source), prior, old in zip(candidates, fm["rows"], a4["rows"]):
        if (label, source) != (prior["label"], prior["source"]) or (label, source) != (old["label"], old["source"]):
            raise RuntimeError("Reference source pairing changed")
        paired_input_fingerprints(batch, prior)
    OUTPUT.mkdir(exist_ok=False)
    identity.update(start_time=now(), entry_sha256=sha(Path(__file__)), pipeline=pipeline.identity,
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        method_sha256=METHOD_SHA, checkpoint_sha256=inspection["checkpoint_sha256"],
        inspection_sha256=sha(inspection_path), fm_cached_result_sha256=FM_SHA, a4_cached_result_sha256=A4_SHA,
        source_comparison=code_receipt, config_sha256=sha(config), policy_seed=17,
        free_ar_generation_budget=10, max_new_tokens=300, new_fm_generations=0,
        optimizer_updates=0, simulator_controls=0, no_teacher_actions_to_actor=True)
    publish(OUTPUT / "manifest.json", identity)
    arch = deepcopy(OmegaConf.load(root / "formal/config.yaml").model.model_arch)
    model, saved = load_model_from_checkpoint(arch, inspection["checkpoint"], device="cuda:0",
        eval_mode=False, return_full_checkpoint=True)
    verify_marker_state(model, saved)
    del saved
    gc.collect()
    prepare_action_updates(model, "cuda:0")
    model.eval()
    assert_marker_bindings(model._marker_vlm(), model.model.action_marker_rows)
    if required_marker_ids(model.action_tokenizer) != MARKER_IDS:
        raise RuntimeError("Actual codec differs from the declared eight markers")
    publish(OUTPUT / "restoration.json", dict(passed=True, full_state_entries=1140, lora_entries=192,
        marker_state_entries=2, marker_ids=list(MARKER_IDS), step=500,
        checkpoint_sha256=inspection["checkpoint_sha256"]))
    rows = []
    for (label, original, source), prior, old in zip(candidates, fm["rows"], a4["rows"]):
        batch = dict_apply(deepcopy(original), lambda t: t.to("cuda:0") if isinstance(t, torch.Tensor) else t)
        fingerprints = paired_input_fingerprints(batch, prior)
        fm_action = torch.tensor(prior["generated_action"], dtype=torch.float32)
        a4_action = torch.tensor(old["a4_fm_generated_action"], dtype=torch.float32)
        cached_scores = {prefix + interval: interval_metrics(pred, batch, begin, end)
            for prefix, pred in (("fm_", fm_action), ("a4_", a4_action))
            for interval, begin, end in (("executed", 0, 16), ("unexecuted", 16, 32))}
        for key, score in cached_scores.items():
            baseline_key = key.removeprefix("fm_") if key.startswith("fm_") else key
            verify_cached_score(score, prior[baseline_key])
        old_score = generation_execution_metrics(a4_action, batch)
        if abs(old_score["normalized_executed_rmse"] - old["a4_fm"]["normalized_executed_rmse"]) > 1e-6:
            raise RuntimeError("A4 cache target mismatch")
        with torch.no_grad(), torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(17)
            action, outcome = infer_free_ar(model, batch)
        row = dict(label=label, source=source, **outcome, **fingerprints, **cached_scores,
            generated_action=action.detach().cpu().tolist() if action is not None else None,
            executed=interval_metrics(action, batch, 0, 16) if action is not None else None,
            unexecuted=interval_metrics(action, batch, 16, 32) if action is not None else None)
        publish(OUTPUT / (label + ".json"), row)
        rows.append(row)
        print(json.dumps(dict(label=label, valid=row["valid"], executed=row["executed"], failure=row["failure"])), flush=True)
    publish(OUTPUT / "result.json", dict(complete=True, run=RUN, rows=rows, summary=pooled_comparison(rows),
        checkpoint_sha256=inspection["checkpoint_sha256"], ar_generations=10, new_fm_generations=0,
        teacher_action_encodes=0, optimizer_updates=0, simulator_controls=0, success_rate_claim=False,
        limitations=["Five train and five repeatedly diagnosed heldout windows, one policy seed; not full SR.",
            "Correct supervised skills condition the actor; no autonomous planner or feedback test.",
            "Valid-only scores are conditional on coverage; invalid AR outputs are not zero-filled.",
            "Marker eager CE plus marker rows is a practical recipe, not an isolated marker-only comparison.",
            "No schema constraints, FM fallback, teacher-action tokenization, or new baseline generation."]))


if __name__ == "__main__":
    main()
