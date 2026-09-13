"""Ten original train states: real public observed-only processing equivalence.

Intercept original dataset processing only to audit its raw observed fields.
The actor adapter receives no action, skill, annotation or dataset locator.
Compare against the same state's original eval-mode tensor pipeline and native
task prefix; this does not claim augmentation or simulator equivalence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from action_training_data import build_original_pipeline
from native_action_initialization import NATIVE_PARENT, verify_native_vocabulary
from probe_action_training_gpu import architecture_for
from probe_ar_execution_codec import publish
from train_action_method_probe import read, now
from train_fm_method_probe import BASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != BASE:
        raise ValueError("Use a new direct child of the experiment root")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned Git source")
    args.output.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
    from g05.utils.training.ar_training_methods import ActionTrainingSettings, native_task_actor_samples
    from g05.utils.training.coordination_receipts import validate_coordination_batch
    from preflight_memlite_skillfm_gpu import _build_cpu_input_preprocessor
    from native_action_observations import NativeTaskObservationProcessor, OBSERVATION_FIELDS

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    register_default_resolvers()
    receipt = read(INPUT / "result.json")
    config = INPUT / "diagnostic_processor_config.yaml"
    if (torch.cuda.is_available() or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or sha(config) != receipt["processor_config_sha256"]
            or receipt["status"] != "complete" or not receipt["train_only"]):
        raise RuntimeError("Original train receipt/CPU/config identity changed")
    cfg = OmegaConf.load(config)
    architecture_for(cfg, ActionTrainingSettings(conditioning="native_task"), parent=NATIVE_PARENT)
    pipeline = build_original_pipeline(cfg, rank=0, world_size=1, workers=0)
    pipeline.train_processor.eval()  # Same-state deterministic comparison, no augmentation claim.
    native = NativeTaskObservationProcessor(pipeline.eval_processor.processors["galaxea_r1pro"])
    tokenizer = _build_cpu_input_preprocessor(cfg)
    proxy = SimpleNamespace(processor=tokenizer, action_tokenizer=tokenizer.action_tokenizer,
        model=SimpleNamespace(ar_helper=SimpleNamespace(set_token_index_ranges=lambda *v: None)))
    G05PolicyQwen35._fix_action_token_offset(proxy)
    verify_native_vocabulary(tokenizer)
    identity.update(start_time=now(), commit=subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), adapter_sha256=sha(REPO / "scripts/experiments/native_action_observations.py"),
        original_input_sha256=INPUT_SHA, config_sha256=sha(config), pipeline=pipeline.identity,
        max_original_train_states=10, optimizer_updates=0, vlm_forwards=0, simulator_controls=0,
        same_state_reference="original processor eval-mode tensors plus native task view",
        actor_requires_planner=False, actor_teacher_fields=False, new_annotation_release=False)
    publish(args.output / "manifest.json", identity)
    sources = [source for batch in receipt["batches"] for source in batch["sources"]]
    if len(sources) != 10 or {source["task"] for source in sources} != set(range(5)):
        raise RuntimeError("Declared original ten-row/five-task input changed")
    rows = []
    for index, source in enumerate(sources):
        captures, originals = [], {}
        for key, processor in pipeline.train_processor.processors.items():
            originals[key] = processor.preprocess
            def capture(raw, *, original=processor.preprocess):
                captures.append(deepcopy(raw))
                return original(raw)
            processor.preprocess = capture
        try:
            reference = pipeline.train[(source["requested_index"], "low")]
        finally:
            for key, original in originals.items():
                pipeline.train_processor.processors[key].preprocess = original
        if len(captures) != 1:
            raise RuntimeError("Expected one exact original raw sample, no fallback/retry")
        batch = pipeline.collate([reference])
        audit = validate_coordination_batch(batch, source_spec_path=pipeline.source_spec,
                                            audit_resolver=pipeline.train_resolver)
        locator = reference["memlite_audit_locator"]
        if (len(audit) != 1 or locator["source_episode_index"] != source["episode"]
                or locator["source_frame_index"] != source["frame"]):
            raise RuntimeError("Original exact train state/source identity changed")
        raw = captures[0]
        observed = {name: deepcopy(raw[name]) for name in OBSERVATION_FIELDS - {"embodiment_type"}}
        observed["embodiment_type"] = "galaxea_r1pro"
        prepared = native.preprocess_for_inference(observed)
        model_batch = native.collate(prepared)
        if model_batch["action_dim_is_pad"].shape != (1, 27) or model_batch["proprio"].shape != (1, 6, 27):
            raise RuntimeError("Actual observed-only collator changed action/history metadata")
        sample = prepared.sample
        if not all(torch.equal(reference[name], sample[name]) for name in
                   ("proprio", "proprio_is_pad", "proprio_dim_is_pad", "action_dim_is_pad")):
            raise RuntimeError("Removing training targets changed normalized state or masks")
        if (list(reference["pixel_values"]) != list(native.pixel_layout) or
                not all(torch.equal(reference["pixel_values"][camera], sample["pixel_values"][camera])
                        for camera in native.pixel_layout)):
            raise RuntimeError("Observed-only public processor changed real camera tensors")
        reference_samples = native_task_actor_samples([reference["samples"]], num_images=18)
        ids, mask = tokenizer.encode_inference(reference_samples, device=torch.device("cpu"), mode="ar")
        actual, actual_mask = tokenizer.encode_inference([sample["samples"]], device=torch.device("cpu"), mode="ar")
        if not torch.equal(ids, actual) or not torch.equal(mask, actual_mask):
            raise RuntimeError("Native observation tokenizer differs from the original same-state native view")
        for name, anchor in prepared.raw_state_anchor.items():
            if not torch.equal(anchor, raw["state"][name][-1:].float().unsqueeze(0)):
                raise RuntimeError("Current raw joint anchor was normalized or shifted")
        # An observed state is not a future action. Check the original inverse
        # machinery on an explicit offline target fixture, never an actor call.
        inverse_fixture = dict(action=reference["action"].unsqueeze(0), selected_action_source="ar",
            ar_complete_block_receipts=[dict(token_count=60, complete_blocks=8)], ar_absent_keys=[set()],
            execution_start=0, execution_steps=16)
        raw_action = native.postprocess_action(inverse_fixture, prepared)
        row = dict(source=source, exact_source_validated=True, tensor_pipeline_equal=True,
            task_prefix_tokens_equal=True, prefix_tokens=int(mask.bool().sum()),
            raw_anchor_equal=True, static_padded_dimensions=native.action_mask.nonzero().flatten().tolist(),
            actor_fields=sorted(sample["samples"]), camera_shapes={key: list(value.shape) for key, value in sample["pixel_values"].items()},
            inverse_fixture_only_not_generated=True, raw_action_shapes={key: list(value.shape)
                for key, value in raw_action.items() if not key.startswith("_")}, execution_start=0)
        rows.append(row)
        publish(args.output / f"row_{index:02d}.json", row)
        print(json.dumps(dict(verified_original_states=len(rows), task=source["task"])), flush=True)
    publish(args.output / "result.json", dict(complete=True, rows=rows, original_train_states=len(rows),
        optimizer_updates=0, vlm_forwards=0, simulator_controls=0, new_annotation_release=False,
        success_rate_claim=False, limitations=["Same-state eval-mode preprocessing, not augmentation equality.",
        "Inverse fixture uses held-out-of-actor expert target only to test plumbing; no generated action quality claim.",
        "History transport must still admit actual observations before this adapter; no simulator/wire calls here."]))
    print(json.dumps(dict(complete=True, result_sha256=sha(args.output / "result.json"))), flush=True)


if __name__ == "__main__":
    main()
