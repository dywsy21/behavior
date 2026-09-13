"""Ten original train rows: native Subtask-CoT rendering and leakage gate.

No new annotations: the output string copies the previously reviewed same-state
skill text. It is not an upstream full-CoT dataset replication. Zero VLM/update/
physics. Native actor observations must be invariant to any teacher answers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from action_training_runtime import bootstrap, INPUT, INPUT_SHA, REPO, sha
from native_action_initialization import verify_native_vocabulary
from probe_action_training_gpu import architecture_for
from native_action_initialization import NATIVE_PARENT
from probe_ar_execution_codec import publish
from train_fm_method_probe import BASE
from train_action_method_probe import read, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != BASE:
        raise ValueError("Use a new direct child run")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean pinned worktree")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.output.mkdir(exist_ok=False)
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
    from g05.models.g05.g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
    from g05.utils.training.ar_training_methods import (
        ActionTrainingSettings, action_prefix_samples, action_training_samples,
        validate_complete_action_tokens, CANONICAL_PARTS)
    from preflight_memlite_skillfm_gpu import _build_cpu_input_preprocessor

    register_default_resolvers()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(101)
    config = INPUT / "diagnostic_processor_config.yaml"
    receipt = read(INPUT / "result.json")
    if (torch.cuda.is_available() or sha(INPUT / "actual_cpu_batches.pt") != INPUT_SHA
            or receipt["processor_config_sha256"] != sha(config)
            or receipt["status"] != "complete" or not receipt["train_only"]):
        raise RuntimeError("Original train/CPU identity changed")
    cfg = OmegaConf.load(config)
    settings = ActionTrainingSettings(conditioning="native_subtask_cot")
    arch = architecture_for(cfg, settings, parent=NATIVE_PARENT)
    processor = _build_cpu_input_preprocessor(cfg)
    proxy = SimpleNamespace(processor=processor, action_tokenizer=processor.action_tokenizer,
        model=SimpleNamespace(ar_helper=SimpleNamespace(set_token_index_ranges=lambda *v: None)))
    G05PolicyQwen35._fix_action_token_offset(proxy)
    verify_native_vocabulary(processor)
    tokenizer = processor.action_tokenizer
    tokenizer.action_tokenizer.eval()
    begin, end = tokenizer.action_token_begin_idx, tokenizer.action_token_end_idx
    validator = SimpleNamespace(model_config=arch, interface_schema_version=6,
                                _memlite_branch_masks=G05PolicyQwen35._memlite_branch_masks)
    identity.update(start_time=now(), entry_sha256=sha(Path(__file__)), original_input_sha256=INPUT_SHA,
        input_config_sha256=sha(config), settings=settings.as_dict(), original_train_rows=10,
        native_vocab=True, target_source="exact reviewed same-state active_skills_text, output only",
        commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        optimizer_updates=0, vlm_forwards=0, simulator_controls=0, new_annotation_release=False)
    publish(args.output / "manifest.json", identity)
    rows = []
    batches = torch.load(INPUT / "actual_cpu_batches.pt", map_location="cpu", weights_only=False)
    if len(batches) != 5 or len(receipt["batches"]) != 5:
        raise RuntimeError("Original five-batch cache changed")
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            samples, actions, temporal, dimensions = (batch[key] for key in
                ("samples", "action", "action_is_pad", "action_dim_is_pad"))
            G05PolicyMEMLiteSkillFM._validate_skill_batch(validator, samples, actions, temporal)
            prepared = action_training_samples(samples, actions, temporal, dimensions, settings)
            observed = action_prefix_samples(samples, settings)
            ids, labels, mask, split = processor.encode_train(prepared, device=torch.device("cpu"),
                training=False, max_chunk_token_length=int(arch.max_chunk_token_length))
            prefix, prefix_mask = processor.encode_inference(observed, device=torch.device("cpu"), mode="ar")
            context = prefix.shape[1]
            if (not torch.equal(ids[:, :context], prefix) or not torch.equal(mask[:, :context], prefix_mask)
                    or not (labels[:, :context] == -100).all() or not context < split < ids.shape[1]):
                raise RuntimeError("Native CoT observation/teacher boundary mismatch")
            p, pl, pm = processor.preprocess(deepcopy(prepared), device=torch.device("cpu"),
                control_flag="return_prefix", training=False)
            s, sl, sm = processor.preprocess(deepcopy(prepared), device=torch.device("cpu"),
                control_flag="return_suffix", training=False)
            if (not torch.equal(ids, torch.cat([p, s], 1)) or not torch.equal(labels, torch.cat([pl, sl], 1))
                    or not torch.equal(mask, torch.cat([pm, sm], 1))):
                raise RuntimeError("Subtask/action training sequence was truncated or changed")
            alternate = deepcopy(prepared)
            for row in alternate:
                row["atomic_task"] = "Subtask: DIAGNOSTIC_COUNTERFACTUAL_NOT_AN_ANNOTATION"
                row["action"]["value"] += .25
            changed, changed_mask = processor.encode_inference(alternate, device=torch.device("cpu"), mode="ar")
            if not torch.equal(changed, prefix) or not torch.equal(changed_mask, prefix_mask):
                raise RuntimeError("Teacher subtask or future actions entered the observation prefix")
            selected = (ids >= begin) & (ids < end) & mask.bool()
            decoded, decoded_ids, _, absent = processor.decode_ar(
                [ids[i, selected[i]] for i in range(len(samples))], horizon_steps=32, action_dim=27,
                device=torch.device("cpu"), action_dim_is_pad=dimensions,
                frequencies=[row.get("frequency") for row in prepared], embodiments=[row["embodiment"] for row in prepared],
                input_parts_meta_list=[CANONICAL_PARTS] * len(samples), action_only=True)
            for i, source in enumerate(receipt["batches"][batch_index]["sources"]):
                if (int(batch["sample_meta"][i]["dataset_locator"].rsplit("local_idx=", 1)[-1]) != source["requested_index"]
                        or prepared[i]["atomic_task"] != "Subtask: " + samples[i]["active_skills_text"]):
                    raise RuntimeError("CoT target is not its original reviewed same-state text")
                eov = (ids[i] == processor.eov_token_id) & mask[i].bool()
                if (int(eov.sum()) != 1 or not torch.equal(labels[i, eov], ids[i, eov])
                        or not (labels[i, context:split] != -100).any()
                        or selected[i, :split].any() or not torch.equal(labels[i, selected[i]], ids[i, selected[i]])
                        or absent[i] or not torch.equal(decoded_ids[i], ids[i, selected[i]])
                        or decoded[i].shape != (32, 27) or not torch.isfinite(decoded[i]).all()):
                    raise RuntimeError("CoT/EOV/action target masks or complete 23D codec supervision are invalid")
                complete = validate_complete_action_tokens(decoded_ids[i], tokenizer)
                row = dict(source=source, cot_target=prepared[i]["atomic_task"], observation_tokens=context,
                    eov_prefix_length=split, total_tokens=ids.shape[1], complete_codec_blocks=complete,
                    supervised_cot_tokens=int((labels[i, context:split] != -100).sum()),
                    original_skill_text_exact=True, target_free_prefix_exact=True,
                    teacher_counterfactual_prefix_invariant=True, eov_supervised=True, untruncated=True)
                rows.append(row)
                publish(args.output / f"row_{len(rows):02d}.json", row)
    if len(rows) != 10 or {row["source"]["task"] for row in rows} != set(range(5)):
        raise RuntimeError("CoT gate did not cover all original ten train rows/five tasks")
    publish(args.output / "result.json", dict(complete=True, identity=identity, rows=rows, rendered_views=10,
        no_free_generation_or_learning_claim=True, optimizer_updates=0, vlm_forwards=0, simulator_controls=0))


if __name__ == "__main__":
    main()
