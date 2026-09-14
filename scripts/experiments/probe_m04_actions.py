"""One ten-window continuous-action check per completed M-04 training arm.

Three declared arms, thirty generations total. No AR calls, teacher targets in
the actor, new codec calls, updates, physics, retries or automatic deployment.
The completed M-02 and original A4 generations are not repeated.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

from action_training_runtime import bootstrap, EXTENSIONS, INPUT, INPUT_SHA, REPO, sha
from action_training_data import build_original_pipeline, exact_eval_batch
from probe_action_training_gpu import prepare_action_updates, task_generation_rows, generation_execution_metrics
from probe_ar_execution_codec import publish
from probe_ar_schema_reference import SCHEMA_RESULT, SCHEMA_SHA, validate_paired_sources
from probe_fm_screen_actions import A4_RESULT, A4_SHA, interval_metrics, verify_screen_state
from train_action_method_probe import read, now, RECIPE, expected_adam_states, verify_dependency_completed
from train_fm_method_probe import BASE, PARENT_SHA

ARMS = {
    "fm_action_control_v3": ("fm", "5169c02045ef9efc88039fc5917d8f0c11b4e7cdd003ecdef0ff40c16397fcb8"),
    "joint_a4_fulltrain_v3": ("joint", "4c9098f54af129c4018d176600ed6becc85aa6b8afbb89d4877675062bd4a01d"),
    "ki_a4_fulltrain_v3": ("ki", "8235509ed970b4a999a6cf9c3c038322c123ba96da498467848c025316825a01"),
}


def output_path(name):
    if name not in ARMS:
        raise ValueError("Only the three declared M-04 arms are in this generation budget")
    return BASE / ("m04_actions_" + ARMS[name][0] + "_v1")


def validate_training(name, method, inspection):
    route = ARMS[name][0]
    if (method.get("route") != route or method.get("initialization") != "a4"
            or method.get("conditioning") != "skills" or method.get("marker_rows") is not False
            or method.get("recipe") != RECIPE or method.get("parent_sha256") != PARENT_SHA
            or method.get("train_split") != "original950" or method.get("eval_split") != "original50"
            or not inspection.get("passed") or inspection.get("actual_updates") != 500
            or inspection.get("actual_adam_states") != expected_adam_states(route)
            or inspection.get("actual_train_rows") != 8000 or not inspection.get("frozen_unchanged")
            or not inspection.get("full_model_optimizer_rng_roundtrip")):
        raise RuntimeError("M-04 actions require their exact completed fresh-A4500 recipe and saved-state proof")


def tensor_fingerprint(value):
    """Record actual processed pixels/masks, not only same source/seed labels."""
    import torch
    if isinstance(value, dict):
        return {k: tensor_fingerprint(v) for k, v in sorted(value.items())}
    if not isinstance(value, torch.Tensor):
        raise ValueError("Only actual processed tensors may be fingerprinted")
    tensor = value.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


@contextmanager
def count_continuous_only(model):
    """Catch accidentally using the training-time discrete branch at inference."""
    fm, ar = model.model.inference_fm, model.model.inference_ar
    calls = dict(fm=0, ar=0)

    def observed_fm(*args, **kwargs):
        calls["fm"] += 1
        return fm(*args, **kwargs)

    def forbidden_ar(*args, **kwargs):
        calls["ar"] += 1
        raise RuntimeError("The M-04 deployment comparison must not generate AR actions")

    model.model.inference_fm, model.model.inference_ar = observed_fm, forbidden_ar
    try:
        yield calls
    finally:
        model.model.inference_fm, model.model.inference_ar = fm, ar


def infer_continuous(model, route, batch):
    import torch
    if route not in {"fm", "joint", "ki"} or model.training or model.predict_cot:
        raise ValueError("M-04 comparison is eval-mode skill-conditioned continuous deployment only")
    mask = batch["action_dim_is_pad"]
    expected = torch.zeros(1, 27, dtype=torch.bool, device=mask.device)
    expected[:, [7, 8, 17, 18]] = True
    if (mask.dtype != torch.bool or mask.shape != expected.shape or not torch.equal(mask, expected)
            or len(batch["samples"]) != 1
            or any(any(key in sample for key in ("action", "gt_action", "future_state", "teacher_action"))
                   for sample in batch["samples"])):
        raise ValueError("Invalid original23D metadata or teacher field in actor prefix")
    with count_continuous_only(model) as calls:
        if route == "fm":
            if model.discrete_action or not model.continuous_action:
                raise RuntimeError("Original FM must retain its unchanged policy flags")
            state = model.prefill(batch["samples"], batch["pixel_values"])
            prediction = model.generate_action(state, batch["samples"], action_dim_is_pad=mask, action_gt=None)
        else:
            if (model.action_training.route != route or not model.continuous_action
                    or not model.discrete_action or model.action_training.conditioning != "skills"):
                raise RuntimeError("Loaded joint/KI policy differs from its declared training route")
            prediction = model.forward_inference(batch["samples"], batch["pixel_values"], action_dim_is_pad=mask)
    action = prediction["action"]
    if (calls != dict(fm=1, ar=0) or prediction.get("selected_action_source") != "fm"
            or "action_ar" in prediction or action.shape != (1, 32, 27) or not torch.isfinite(action).all()
            or prediction.get("execution_start", 0) != 0 or prediction.get("execution_steps", 16) != 16):
        raise RuntimeError("Invalid continuous source, call count, horizon or execution clock")
    return action, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=tuple(ARMS))
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = output_path(args.run)
    if args.gpu != 1 or output.exists():
        raise ValueError("GPU1 and a never-before-used output are required; no automatic retry")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned source")
    root, (route, method_sha) = BASE / args.run, ARMS[args.run]
    if sha(root / "method_spec.json") != method_sha or sha(A4_RESULT) != A4_SHA or sha(SCHEMA_RESULT) != SCHEMA_SHA:
        raise RuntimeError("Training or cached-reference identity changed")
    inspection_path = verify_dependency_completed(root, "action")
    method, inspection = read(root / "method_spec.json"), read(inspection_path)
    validate_training(args.run, method, inspection)
    # Inference-relevant extensions are byte-identical to those used to train.
    # The newer trainer file only adds the independent CoT observer/queue gate.
    dependencies = list(EXTENSIONS.values()) + ["scripts/experiments/action_training_runtime.py",
        "scripts/experiments/action_training_data.py", "scripts/experiments/probe_action_training_gpu.py"]
    if any(sha(REPO / path) != method["code"][path] for path in dependencies):
        raise RuntimeError("An inference/data extension changed since this model was trained")
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
        raise RuntimeError("Need40GiB free; do not stop other services/training")
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
    cached = read(A4_RESULT)
    if (len(candidates) != 10 or not cached["complete"] or cached["fm_generations"] != 10
            or len(cached["rows"]) != 10 or any((label, source) != (prior["label"], prior["source"])
                for (label, _, source), prior in zip(candidates, cached["rows"]))):
        raise RuntimeError("Missing exact original ten-source cached A4 pairing")
    output.mkdir(exist_ok=False)
    identity.update(start_time=now(), entry_sha256=sha(Path(__file__)), pipeline=pipeline.identity,
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        training_run=args.run, route=route, method_sha256=method_sha, inspection_sha256=sha(inspection_path),
        checkpoint_sha256=inspection["checkpoint_sha256"], original_a4_cached_result_sha256=A4_SHA,
        config_sha256=sha(config), policy_seed=17, fm_generation_budget=10, ar_generations=0,
        new_a4_generations=0, codec_calls=0, optimizer_updates=0, simulator_controls=0,
        no_teacher_actions_to_actor=True, new_annotation_release=False)
    publish(output / "manifest.json", identity)
    trained_config = OmegaConf.load(root / "formal/config.yaml")
    arch = deepcopy(trained_config.model.model_arch)
    model, saved = load_model_from_checkpoint(arch, inspection["checkpoint"], device="cuda:0",
        eval_mode=False, return_full_checkpoint=True)
    verify_screen_state(model, saved)
    del saved
    gc.collect()
    prepare_action_updates(model, "cuda:0")
    model.eval()
    publish(output / "restoration.json", dict(passed=True, complete_state_entries=1138, lora_entries=192,
        checkpoint_sha256=inspection["checkpoint_sha256"], step=500, route=route))
    rows = []
    for (label, original, source), prior in zip(candidates, cached["rows"]):
        batch = dict_apply(deepcopy(original), lambda t: t.to("cuda:0") if isinstance(t, torch.Tensor) else t)
        with torch.no_grad(), torch.random.fork_rng(devices=[0]), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(17)
            action, calls = infer_continuous(model, route, batch)
        old_action = torch.tensor(prior["a4_fm_generated_action"], dtype=torch.float32, device=action.device)
        old_score = generation_execution_metrics(old_action, batch)
        if abs(old_score["normalized_executed_rmse"] - prior["a4_fm"]["normalized_executed_rmse"]) > 1e-6:
            raise RuntimeError("Cached original A4 score differs against this target/support")
        row = dict(label=label, source=source, route=route, calls=calls,
            executed=interval_metrics(action, batch, 0, 16), unexecuted=interval_metrics(action, batch, 16, 32),
            a4_executed=interval_metrics(old_action, batch, 0, 16), a4_unexecuted=interval_metrics(old_action, batch, 16, 32),
            pixel_fingerprints=tensor_fingerprint(batch["pixel_values"]),
            proprio_fingerprints=tensor_fingerprint(batch["samples"][0]["proprio"]),
            mask_fingerprints=tensor_fingerprint({k: batch[k] for k in ("action_is_pad", "action_dim_is_pad")}),
            generated_action=action.detach().cpu().tolist())
        publish(output / (label + ".json"), row)
        rows.append(row)
        print(json.dumps(dict(run=args.run, label=label, executed_rmse=row["executed"]["normalized_rmse"])), flush=True)
    publish(output / "result.json", dict(complete=True, run=args.run, route=route,
        checkpoint_sha256=inspection["checkpoint_sha256"], fm_generations=len(rows), ar_generations=0,
        codec_calls=0, optimizer_updates=0, simulator_controls=0, no_data_release=True,
        rows=rows, success_rate_claim=False, limitations=[
            "Five original train and five repeatedly used heldout diagnostic windows; policy seed17 only.",
            "Supervised same-state skill conditions, not autonomous high-level decisions or a closed loop.",
            "Original A4 reference is cached; matched new500 candidates must be compared separately.",
            "Same sample sequence does not establish bitwise equal training augmentations or RNG."]))


if __name__ == "__main__":
    main()
