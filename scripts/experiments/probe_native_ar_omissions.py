"""Read-only CPU audit: native inactive-group omission versus missing controls.

Re-tokenize the ten already audited train rows under both codec dropout rules;
join the two completed native GPU probes by exact source identity. No VLM,
updates, new training release, actor fallback or simulator commands.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


BASE = Path("/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913")
GATES = {
    "skills": ("ar_native_skills_gpu_gate_v2", "4b527cbabe11f4770a32901f5a2146abe1d4666d83ad4a35a2384bf0fd514d6d"),
    "native_task": ("ar_native_task_gpu_gate_v2", "09d6244d99053c8295fad280267044fabc0b8df8709ee33739a7c8e8efb5577c"),
}


def source_key(source):
    return tuple(source[key] for key in ("task", "episode", "frame", "requested_index", "bundle_id"))


def block_layout(ids, *, markers, codebook_size, offset):
    """Strict present-block parsing; absence is reported, never filled.

    markers maps an action-space marker ID to (group, residual level, length).
    Present neural groups must contain every declared residual level. This is
    intentionally stricter than the legacy parser's truncated-block padding.
    """
    if not ids or any(type(value) is not int for value in ids):
        raise ValueError("Expected nonempty integer action-token IDs")
    required = {}
    for marker, (group, level, length) in markers.items():
        if marker < codebook_size or length < 1:
            raise ValueError("Invalid static codec grammar")
        required.setdefault(group, set()).add(level)
    raw, seen, cursor = [value - offset for value in ids], {}, 0
    while cursor < len(raw):
        marker = raw[cursor]
        if marker not in markers or marker in seen:
            raise ValueError("Unknown, duplicate or misplaced block marker")
        group, level, length = markers[marker]
        payload = raw[cursor + 1:cursor + 1 + length]
        if len(payload) != length or any(value < 0 or value >= codebook_size for value in payload):
            raise ValueError("Truncated or malformed block payload")
        seen[marker] = (group, level)
        cursor += length + 1
    present = {}
    for group, level in seen.values():
        present.setdefault(group, set()).add(level)
    incomplete = {group for group, levels in present.items() if levels != required[group]}
    return dict(present=sorted(present), absent=sorted(set(required) - set(present)),
        incomplete_residual_groups=sorted(incomplete), complete_present_blocks=not incomplete,
        token_count=len(ids), block_count=len(seen))


def omission_classification(layout, codec_noop):
    absent, noop = set(layout["absent"]), set(codec_noop)
    if noop - (set(layout["present"]) | absent):
        raise ValueError("Noop audit refers to a different embodiment")
    return dict(codec_noop_omissions=sorted(absent & noop),
        omitted_non_noop_target_groups=sorted(absent - noop),
        incomplete_residual_groups=layout["incomplete_residual_groups"],
        deployable_claim=False, physical_inactivity_proven=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != BASE or args.output.exists():
        raise ValueError("Use a new child of the declared experiment root")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from action_training_runtime import bootstrap, REPO, INPUT, INPUT_SHA, sha
    from probe_ar_execution_codec import publish
    from native_action_initialization import verify_native_vocabulary, NATIVE_PARENT_SHA

    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean, pinned worktree")
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
    from g05.models.g05.g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
    from g05.utils.training.ar_training_methods import ActionTrainingSettings, action_training_samples
    from preflight_memlite_skillfm_gpu import _build_cpu_input_preprocessor
    from types import SimpleNamespace

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(101)
    register_default_resolvers()
    if torch.cuda.is_available():
        raise RuntimeError("This audit must not use CUDA")
    receipt = json.loads((INPUT / "result.json").read_text())
    config = INPUT / "diagnostic_processor_config.yaml"
    if (sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA or receipt["actual_batch_file_sha256"] != INPUT_SHA
            or receipt["status"] != "complete" or not receipt["train_only"] or not receipt["all_five_tasks"]
            or sha(config) != receipt["processor_config_sha256"]):
        raise RuntimeError("Original audited train inputs changed")
    gates = {}
    for conditioning, (name, digest) in GATES.items():
        path = BASE / name / "result.json"
        value = json.loads(path.read_text())
        if (sha(path) != digest or not value["complete"] or value["actual_updates"] != 2
                or value["parent_sha256"] != NATIVE_PARENT_SHA
                or value["settings"]["conditioning"] != conditioning):
            raise RuntimeError("Native free-generation evidence identity changed")
        gates[conditioning] = value
    args.output.mkdir(exist_ok=False)
    cfg = OmegaConf.load(config)
    OmegaConf.set_struct(cfg, False)
    cfg.model.model_arch.register_memlite_hl_end = False
    cfg.model.model_arch.memlite_train_mode = "off"
    cfg.tokenizer.vq_config.dropout_noop_parts = False
    cfg.model.model_arch.AT_CONFIG.dropout_noop_parts = False
    processor = _build_cpu_input_preprocessor(cfg)
    codec = processor.action_tokenizer
    proxy = SimpleNamespace(processor=processor, action_tokenizer=codec,
        model=SimpleNamespace(ar_helper=SimpleNamespace(set_token_index_ranges=lambda *args: None)))
    G05PolicyQwen35._fix_action_token_offset(proxy)
    verify_native_vocabulary(processor)
    codec.action_tokenizer.eval()
    serializer, markers = codec.serializer, {}
    for level in range(serializer.num_residuals):
        for group in serializer.nn_key_names:
            token = f"<{group}_{level}>" if serializer.max_residuals > 1 else f"<{group}>"
            markers[serializer.group_marker_action_indices[token]] = (group, level, serializer.code_len)
    for group in serializer.rule_key_names:
        markers[serializer.group_marker_action_indices[f"<{group}>"]] = (group, 0, serializer.rule_tokens_per_key)
    begin, end = codec.action_token_begin_idx, codec.action_token_end_idx
    grammar = dict(markers=markers, codebook_size=int(codec._codebook_size), offset=begin)
    identity.update(commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        input_sha256=INPUT_SHA, config_sha256=sha(config), entry_sha256=sha(Path(__file__)),
        codec_sha256=sha(cfg.tokenizer.vq_config.ckpt_dir), gates=GATES, grammar=grammar,
        time=datetime.now(timezone.utc).isoformat(), policy_forwards=0, optimizer_updates=0, simulator_controls=0)
    publish(args.output / "manifest.json", identity)
    batches = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    if len(batches) != 5 or len(receipt["batches"]) != 5:
        raise RuntimeError("Expected five original two-row batches")
    validator = SimpleNamespace(model_config=cfg.model.model_arch, interface_schema_version=6,
        _memlite_branch_masks=G05PolicyQwen35._memlite_branch_masks)
    rows = {}
    with torch.inference_mode():
        for batch, entry in zip(batches, receipt["batches"]):
            G05PolicyMEMLiteSkillFM._validate_skill_batch(validator, batch["samples"], batch["action"], batch["action_is_pad"])
            if len(entry["sources"]) != len(batch["samples"]):
                raise RuntimeError("Source and tensor row counts differ")
            prepared = action_training_samples(batch["samples"], batch["action"], batch["action_is_pad"],
                batch["action_dim_is_pad"], ActionTrainingSettings(conditioning="native_task"))
            variants = {}
            try:
                for dropout in (False, True):
                    codec.dropout_noop_parts = dropout
                    ids, _, mask = processor.preprocess(deepcopy(prepared), control_flag="return_suffix",
                        device=torch.device("cpu"), training=False)
                    variants[dropout] = [block_layout(ids[i][mask[i].bool() & (ids[i] >= begin) & (ids[i] < end)].tolist(),
                        **grammar) for i in range(len(prepared))]
            finally:
                codec.dropout_noop_parts = False
            for i, source in enumerate(entry["sources"]):
                key = source_key(source)
                if key in rows or int(batch["sample_meta"][i]["dataset_locator"].rsplit("local_idx=", 1)[-1]) != source["requested_index"]:
                    raise RuntimeError("Duplicate or mismatched original source identity")
                complete, dropped = variants[False][i], variants[True][i]
                if complete["absent"] or not complete["complete_present_blocks"] or not dropped["complete_present_blocks"]:
                    raise RuntimeError("Teacher codec unexpectedly lost or truncated required blocks")
                action, dimpad = batch["action"][i:i+1], batch["action_dim_is_pad"][i:i+1]
                valid = ~batch["action_is_pad"][i]
                noop = codec._derive_noop_keys(action=action, action_dim_is_pad=dimpad, action_op_mask=~dimpad)[0]
                valid_noop = codec._derive_noop_keys(action=action[:, valid], action_dim_is_pad=dimpad, action_op_mask=~dimpad)[0]
                if set(dropped["absent"]) != noop:
                    raise RuntimeError("Actual tokenized omission disagrees with codec's noop rule")
                rows[key] = dict(source=source, full_target=complete, native_dropout_target=dropped,
                    codec_noop_groups=sorted(noop), valid_segment_noop_groups=sorted(valid_noop),
                    valid_target_steps=int(valid.sum()), original_padding_changes_noop_rule=noop != valid_noop)
    if len(rows) != 10 or {key[0] for key in rows} != set(range(5)):
        raise RuntimeError("Audit must cover all ten original rows and five tasks")
    generated = []
    for conditioning, gate in gates.items():
        for stage in ("before", "after"):
            values = gate[f"generations_{stage}"]
            if len(values) != 5 or {v["source"]["task"] for v in values} != set(range(5)):
                raise RuntimeError("Incomplete five-task free-generation evidence")
            for value in values:
                target = rows[source_key(value["source"])]
                raw = value["generated"]["ids"]
                if len(raw) != 1 or not raw[0] or raw[0][-1:] != [91] or any(v < begin or v >= end for v in raw[0][:-1]):
                    raise RuntimeError("Expected existing action-only output ending in the recorded bar token")
                layout = block_layout(raw[0][:-1], **grammar)
                generated.append(dict(conditioning=conditioning, stage=stage, source=value["source"],
                    layout=layout, **omission_classification(layout, target["codec_noop_groups"])))
    publish(args.output / "result.json", dict(complete=True, identity=identity, original_train_rows=10,
        actual_target_views=20, free_generation_records_audited=len(generated), target_rows=list(rows.values()),
        generations=generated, policy_forwards=0, optimizer_updates=0, simulator_controls=0,
        limitations=["Noop means this codec's encoding rule, not confirmed physical inactivity.",
            "Normalized zeros are not a safe absolute-position hold command.",
            "No omitted group is repaired or authorized for deployment by this audit.",
            "Five original train windows are not generalization or success-rate evidence."]))
    print(json.dumps(dict(complete=True, target_views=20, generations=len(generated))), flush=True)


if __name__ == "__main__":
    main()
