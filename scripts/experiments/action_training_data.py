"""Original five-task train/eval pipeline for the new action-training routes.

Use the real A3 dataset, released labels, episode-balanced sampler, normalizer
and collator. Do not train on the ten-row engineering cache or release eval
rows into a training pool. CPU preflight only reads a few actual loader rows.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from action_training_runtime import bootstrap, REPO, INPUT, sha
from probe_ar_execution_codec import publish

STATS = Path("/mnt/sdc1/robodojo/stats/g05/behavior5_r1pro_trainonly_taskstrat5_stats_v2.json")
STATS_SHA = "846bcbeac181df5555cb5d40d4183d61743a556df5cf8c8a8fb1bc17e2a40b19"
FIXED = Path("/mnt/sdc1/robodojo/datasets/memlite_skill_annotations_task0_4_v6_formal_a_fixed80_eval_v3_20260909/meta/fixed80_fm_diagnostic_manifest.json")
FIXED_SHA = "a365370d81596cb720ba5df38e6935aa1e0a2bdcbf1fbc0c54dfd52bbf45b254"


def build_original_pipeline(cfg, *, rank, world_size, batch_size=2, workers=2):
    import torch
    from omegaconf import OmegaConf
    from g05.utils.data.processor_utils import instantiate_dataset, build_processors
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from g05.utils.data.data_utils import collate_fn_pad_sequences
    from g05.utils.common.task_event_sampler import build_memlite_train_sampler
    from g05.utils.training.coordination_receipts import get_memlite_audit_resolver
    from g05.utils.training.fixed_diagnostic_trainer import _verify_eval_release

    if sha(STATS) != STATS_SHA or sha(FIXED) != FIXED_SHA:
        raise RuntimeError("Original train normalizer or heldout window identity changed")
    sampling = OmegaConf.to_container(cfg.data.memlite_branch_sampling, resolve=True)
    if (sampling.get("strategy") != "coordination_v6" or sampling.get("mode") != "low"
            or sampling.get("leaf_ordering") != "episode_round_robin_v1"):
        raise RuntimeError("Original exact-low, episode-balanced sampling must be preserved")
    train = instantiate_dataset(cfg, is_training_set=True)
    evaluation = instantiate_dataset(cfg, is_training_set=False)
    train_processor, eval_processor = build_processors(cfg), build_processors(cfg)
    eval_processor.eval()
    stats = load_dataset_stats_from_json(STATS)
    train_processor.set_normalizer_from_stats(stats)
    eval_processor.set_normalizer_from_stats(stats)
    train.set_processor(train_processor)
    evaluation.set_processor(eval_processor)
    manifest = json.loads(FIXED.read_text())
    eval_identity, eval_resolver = _verify_eval_release(evaluation, manifest)
    train_resolver = get_memlite_audit_resolver(train)
    identity = train_resolver.identity()
    if (identity["eligible_digest"] != manifest["train_composite_release_identity"]["eligible_digest"]
            or identity["eligible_digest"] == manifest["eval_resolver_identity"]["eligible_digest"]):
        raise RuntimeError("Training resolver does not match the original separate train view")
    sampler = build_memlite_train_sampler(train, sampling_config=sampling, batch_size=batch_size,
        num_replicas=world_size, rank=rank, seed=41, shuffle=True,
        high_fraction=float(sampling.get("high_fraction", .125)), drop_last=bool(sampling.get("drop_last", False)))
    collate = partial(collate_fn_pad_sequences, padding_input_id=train_processor.pad_token_id)
    # A private RNG keeps DataLoader worker initialization separate from model
    # noise and remains explicit in the eventual checkpoint/resume identity.
    generator = torch.Generator().manual_seed(41000 + rank)
    loader = torch.utils.data.DataLoader(train, batch_sampler=sampler, collate_fn=collate,
        num_workers=workers, pin_memory=bool(workers), persistent_workers=bool(workers),
        prefetch_factor=2 if workers else None, generator=generator)
    return SimpleNamespace(train=train, evaluation=evaluation, sampler=sampler, loader=loader,
        train_processor=train_processor, eval_processor=eval_processor, collate=collate,
        source_spec=sampler.receipt_source_spec(), train_resolver=train_resolver,
        eval_resolver=eval_resolver, manifest=manifest, loader_generator=generator,
        identity=dict(normalizer_path=str(STATS), normalizer_sha256=STATS_SHA,
            fixed_window_manifest=str(FIXED), fixed_window_sha256=FIXED_SHA,
            sampling_index_path=sampling["sampling_index_path"], sampling_index_sha256=sha(sampling["sampling_index_path"]),
            train_resolver=identity, eval_release=eval_identity, train_dataset_length=len(train),
            eval_dataset_length=len(evaluation), batch_size=batch_size, rank=rank, world_size=world_size,
            seed=41, loader_generator_seed=41000 + rank, workers=workers))


def exact_eval_batch(pipeline, window):
    from g05.utils.training.fixed_diagnostic_trainer import _exact_outer_sample
    from g05.utils.training.coordination_formal_a import sha256_json
    batch = pipeline.collate([_exact_outer_sample(pipeline.evaluation, window)])
    temporal = batch["action_is_pad"]
    if batch["action"].shape != (1, 32, 27) or temporal.shape != (1, 32):
        raise RuntimeError("Original heldout window shape differs")
    valid_sha = sha256_json(dict(shape=list(temporal[0].shape), valid=(~temporal[0].bool()).tolist()))
    audit = pipeline.eval_resolver.resolve(episode_index=window["episode_index"],
        frame_index=window["frame_index"], actual_branch="low")
    if (valid_sha != window["valid_action_mask_sha256"]
            or audit["source_fingerprint_sha256"] != window["source_fingerprint_sha256"]
            or audit["active_skills_semantic_json"] != window["active_skills_semantic_json"]):
        raise RuntimeError("Heldout source fingerprint/condition/valid-mask mismatch")
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("Use a new absolute output")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned worktree")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.training.coordination_receipts import validate_coordination_batch
    from g05.utils.training.ar_training_methods import action_training_samples, ActionTrainingSettings
    from g05.models.g05.g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    config = INPUT / "diagnostic_processor_config.yaml"
    original = json.loads((INPUT / "result.json").read_text())
    if sha(config) != original["processor_config_sha256"] or original["status"] != "complete":
        raise RuntimeError("Original pipeline configuration identity differs")
    cfg = OmegaConf.load(config)
    pipeline = build_original_pipeline(cfg, rank=0, world_size=1, workers=0)
    identity.update(pipeline=pipeline.identity, config_sha256=sha(config),
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), original_train_microbatch_budget=5, heldout_window_budget=5,
        policy_forwards=0, optimizer_updates=0, simulator_controls=0)
    publish(args.output / "manifest.json", identity)
    publish(args.output / "train_source_spec.json", pipeline.source_spec)
    validator = SimpleNamespace(model_config=cfg.model.model_arch, interface_schema_version=6,
                                _memlite_branch_masks=G05PolicyQwen35._memlite_branch_masks)
    train_rows = []
    iterator = iter(pipeline.loader)
    for index in range(5):
        batch = next(iterator)
        rows = validate_coordination_batch(batch, source_spec_path=pipeline.source_spec,
                                          audit_resolver=pipeline.train_resolver)
        G05PolicyMEMLiteSkillFM._validate_skill_batch(validator, batch["samples"], batch["action"], batch["action_is_pad"])
        prepared = action_training_samples(batch["samples"], batch["action"], batch["action_is_pad"],
                                          batch["action_dim_is_pad"], ActionTrainingSettings())
        if len(rows) != 2 or len(prepared) != 2:
            raise RuntimeError("Expected two real train rows per microbatch")
        train_rows.extend(rows)
        print(json.dumps(dict(train_microbatches=index + 1, verified_train_rows=len(train_rows))), flush=True)
    selected = {}
    for window in pipeline.manifest["windows"]:
        selected.setdefault(window["task_id"], window)
    if set(selected) != {f"task{i}" for i in range(5)}:
        raise RuntimeError("Original fixed evaluation must contain all five tasks")
    for window in selected.values():
        batch = exact_eval_batch(pipeline, window)
        G05PolicyMEMLiteSkillFM._validate_skill_batch(validator, batch["samples"], batch["action"], batch["action_is_pad"])
    publish(args.output / "result.json", dict(complete=True, original_train_rows=10,
        verified_eval_windows=list(selected.values()), train_rows=train_rows, identity=identity,
        policy_forwards=0, optimizer_updates=0, simulator_controls=0, no_dataset_release=True,
        limitations=["Real full-pipeline input gate, not a learning or policy-effect experiment.",
                     "Five heldout windows only verify identity and never enter a training batch."]))
    print(json.dumps(dict(complete=True, original_train_rows=10, verified_eval_windows=5)), flush=True)


if __name__ == "__main__":
    main()
