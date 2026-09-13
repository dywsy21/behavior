"""Real CPU tokenizer/prefix gate for pure AR and CE+FM/KI, no policy weights.

Ten audited original train rows, four representation/conditioning views.
Proves target placement, no truncation, train/inference prefix equality and
complete codec decoding. It does NOT prove free generation or learning.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from action_training_runtime import bootstrap, REPO, INPUT, INPUT_SHA, sha
from probe_ar_execution_codec import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native", action="store_true", help="Verify the native vocabulary and task-only template")
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("Use a new absolute run directory")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean, pinned Git worktree")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
    from g05.models.g05.g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
    from g05.utils.training.ar_training_methods import (
        ActionTrainingSettings, CANONICAL_PARTS, action_prefix_samples, action_training_samples,
        validate_complete_action_tokens,
    )
    from preflight_memlite_skillfm_gpu import _build_cpu_input_preprocessor

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(101)
    if torch.cuda.is_available():
        raise RuntimeError("CPU gate must not allocate CUDA")
    register_default_resolvers()
    config_path = INPUT / "diagnostic_processor_config.yaml"
    receipt = json.loads((INPUT / "result.json").read_text())
    if (sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or receipt["actual_batch_file_sha256"] != INPUT_SHA
            or receipt["status"] != "complete" or not receipt["train_only"]
            or not receipt["all_five_tasks"] or receipt["processor_config_sha256"] != sha(config_path)):
        raise RuntimeError("Audited original train inputs/config identity differs")
    cfg = OmegaConf.load(config_path)
    if args.native:
        OmegaConf.set_struct(cfg, False)
        cfg.model.model_arch.register_memlite_hl_end = False
        cfg.model.model_arch.memlite_train_mode = "off"
    cfg.tokenizer.vq_config.dropout_noop_parts = False
    cfg.model.model_arch.AT_CONFIG.dropout_noop_parts = False
    processor = _build_cpu_input_preprocessor(cfg)
    tokenizer = processor.action_tokenizer
    offset_updates = []
    proxy = SimpleNamespace(processor=processor, action_tokenizer=tokenizer,
        model=SimpleNamespace(ar_helper=SimpleNamespace(set_token_index_ranges=lambda *v: offset_updates.append(v))))
    # Use the actual Qwen policy hook, not a second guessed offset convention.
    G05PolicyQwen35._fix_action_token_offset(proxy)
    tokenizer.action_tokenizer.eval()
    begin, end = tokenizer.action_token_begin_idx, tokenizer.action_token_end_idx
    if processor.tokenizer.convert_tokens_to_ids(tokenizer.action_tokens[0]) != begin:
        raise RuntimeError("Qwen action token offset remains inconsistent")
    if args.native:
        from native_action_initialization import verify_native_vocabulary
        verify_native_vocabulary(processor)
    architecture = cfg.model.model_arch
    validator = SimpleNamespace(model_config=architecture, interface_schema_version=6,
                                _memlite_branch_masks=G05PolicyQwen35._memlite_branch_masks)
    batches = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    if len(batches) != 5 or len(receipt["batches"]) != 5:
        raise RuntimeError("Expected five original microbatches")
    settings_list = [ActionTrainingSettings(),
        ActionTrainingSettings(codec_mode="prefix16_holdpad32"),
        ActionTrainingSettings(conditioning="task"), ActionTrainingSettings(route="ki")]
    if args.native:
        settings_list = [ActionTrainingSettings(), ActionTrainingSettings(conditioning="native_task")]
    identity.update(commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        input_sha256=INPUT_SHA, config_sha256=sha(config_path), entry_sha256=sha(Path(__file__)),
        codec_sha256=sha(cfg.tokenizer.vq_config.ckpt_dir), start_time=datetime.now(timezone.utc).isoformat(),
        settings=[s.as_dict() for s in settings_list], action_token_range=[begin, end],
        native_vocabulary=args.native, state_token_id=processor.state_token_id,
        eov_token_id=processor.eov_token_id, hl_end_token_id=processor.hl_end_token_id,
        qwen_offset_hook_updates=offset_updates, max_chunk_token_length=int(architecture.max_chunk_token_length),
        dropout_noop_parts=False, train_only=True, policy_forwards=0, optimizer_updates=0, simulator_controls=0)
    publish(args.output / "manifest.json", identity)
    rows = []
    original_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            samples, actions = batch["samples"], batch["action"]
            temporal, dimensions = batch["action_is_pad"], batch["action_dim_is_pad"]
            G05PolicyMEMLiteSkillFM._validate_skill_batch(validator, samples, actions, temporal)
            sources = receipt["batches"][batch_index]["sources"]
            if len(samples) != len(sources):
                raise RuntimeError("Original source row count mismatch")
            for ri, source in enumerate(sources):
                locator = batch["sample_meta"][ri]["dataset_locator"]
                if int(locator.rsplit("local_idx=", 1)[-1]) != source["requested_index"]:
                    raise RuntimeError("Original sample locator differs from source receipt")
            original_count += len(samples)
            for settings in settings_list:
                prepared = action_training_samples(samples, actions, temporal, dimensions, settings)
                inference = action_prefix_samples(samples, settings)
                if any("action" in sample for sample in inference + samples):
                    raise RuntimeError("Target-free samples were mutated or contain action GT")
                device = torch.device("cpu")
                prefix, prefix_labels, prefix_mask = processor.preprocess(
                    deepcopy(prepared), control_flag="return_prefix", device=device, training=False)
                suffix, suffix_labels, suffix_mask = processor.preprocess(
                    deepcopy(prepared), control_flag="return_suffix", device=device, training=False)
                ids, labels, mask, split = processor.encode_train(prepared, device=device, training=False,
                    max_chunk_token_length=int(architecture.max_chunk_token_length))
                observed, observed_mask = processor.encode_inference(inference, device=device, mode="ar")
                if (split != prefix.shape[1] or ids.shape[1] != prefix.shape[1] + suffix.shape[1]
                        or not torch.equal(ids, torch.cat([prefix, suffix], dim=1))
                        or not torch.equal(labels, torch.cat([prefix_labels, suffix_labels], dim=1))
                        or not torch.equal(mask, torch.cat([prefix_mask, suffix_mask], dim=1))):
                    raise RuntimeError("Training truncated or changed the true prefix/target concatenation")
                if (not torch.equal(prefix, observed) or not torch.equal(prefix_mask, observed_mask)
                        or not (labels[:, :split] == -100).all()):
                    raise RuntimeError("Train/inference prefix mismatch or unintended prefix CE supervision")
                changed_actions = actions.clone()
                changed_actions[:, :, ~dimensions[0]] += .25
                alternate = action_training_samples(samples, changed_actions, temporal, dimensions, settings)
                changed_prefix, _, changed_mask = processor.preprocess(
                    alternate, control_flag="return_prefix", device=device, training=False)
                if not torch.equal(changed_prefix, prefix) or not torch.equal(changed_mask, prefix_mask):
                    raise RuntimeError("Counterfactual action answers changed the observation prefix")
                generated_ids = [ids[i, split:][mask[i, split:].bool()] for i in range(len(samples))]
                decoded, decoded_tokens, _, missing = processor.decode_ar(generated_ids,
                    horizon_steps=32, action_dim=27, device=device, action_dim_is_pad=dimensions,
                    frequencies=[s.get("frequency") for s in samples], embodiments=[s.get("embodiment") for s in samples],
                    input_parts_meta_list=[CANONICAL_PARTS] * len(samples), action_only=True)
                for ri, source in enumerate(sources):
                    selected = (ids[ri] >= begin) & (ids[ri] < end) & mask[ri].bool()
                    if (not selected.any() or selected[:split].any()
                            or not torch.equal(labels[ri, selected], ids[ri, selected])
                            or missing[ri] or decoded_tokens[ri].numel() < 2
                            or decoded[ri].shape != (32, 27) or not torch.isfinite(decoded[ri]).all()):
                        raise RuntimeError("Incomplete/corrupt action supervision or missing control groups")
                    # Real token-to-ID conversion must preserve every codec ID;
                    # this catches action-token offset and text roundtrip drift.
                    if not torch.equal(decoded_tokens[ri], ids[ri, selected]):
                        raise RuntimeError("Supervised action token IDs changed during text/codec decoding")
                    block_receipt = validate_complete_action_tokens(decoded_tokens[ri], tokenizer)
                    valid = (~temporal[ri, :16])[:, None] & (~dimensions[ri])[None, :]
                    error = (decoded[ri][:16] - actions[ri, :16])[valid]
                    row = dict(source=source, settings=settings.as_dict(),
                        prefix_length=int(split), sequence_length=ids.shape[1], action_token_count=int(selected.sum()),
                        exact_prefix_match=True, prefix_labels_masked=True, action_counterfactual_prefix_invariant=True,
                        untruncated=True, complete_action_groups=True, exact_token_roundtrip=True,
                        complete_codec_blocks=block_receipt,
                        valid_execution_steps=int((~temporal[ri, :16]).sum()),
                        normalized_executed_rmse=float(error.square().mean().sqrt()))
                    rows.append(row)
                    publish(args.output / f"row_{len(rows):02d}.json", row)
                print(json.dumps(dict(batch=batch_index, settings=settings.as_dict(), passed_rows=len(rows))), flush=True)
    if original_count != 10 or len(rows) != 10 * len(settings_list) or {r["source"]["task"] for r in rows} != set(range(5)):
        raise RuntimeError("Expected ten original train rows/five tasks and all declared tokenization views")
    publish(args.output / "result.json", dict(complete=True, original_train_rows=10, rendered_views=len(rows),
        identity=identity, rows=rows, policy_forwards=0, optimizer_updates=0, simulator_controls=0,
        limitations=["Actual tokenizer and codec only, no learned/free AR generation.",
                     "Original train engineering cache, not generalization or success rate.",
                     "Task-only is an explicit conditioning ablation, not upstream full-CoT replication."]))
    print(json.dumps(dict(complete=True, original_train_rows=10, rendered_views=len(rows))), flush=True)


if __name__ == "__main__":
    main()
