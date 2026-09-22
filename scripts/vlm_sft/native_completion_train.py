"""One independently authorized warmstarted120 run, including its two gate updates."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from common import sha, write_json
from native_dataset import checked_images
from native_completion_modeling import encode, collate, load_model, supervised_loss, eos_id, check_mask
from native_completion_protocol import VERSION, query_row
from native_completion_training_data import load_reviewed, BalancedSampler, DATA_SHA, STATES_SHA, REVIEW_SHA
from native_completion_runtime import (CONFIG, validate_config, clean_source, initial_checkpoint,
    require_authorization, CompletionStorage, BASE, base_identity, INITIAL_ADAPTER_SHA,
    INITIAL_RESULT_SHA, EXECUTOR_DIGEST, CORE_COMMIT, execution_metadata, CARRY_PROFILE)


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "motion-data", "data-review", "initial-training", "config", "authorization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); validate_config(cfg)
    code = clean_source(); auth = json.loads(args.authorization.read_text())
    require_authorization(auth, stage="training", code=code, config_sha=sha(args.config), output=args.output)
    examples, data = load_reviewed(args.data, args.motion_data, args.data_review)
    initial, _ = initial_checkpoint(args.initial_training)
    storage = CompletionStorage(auth, args.output); storage.create(); base_files = base_identity()
    args.output.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    write_json(args.output/"launch.json", {"pid": os.getpid(), "code_commit": code, "protocol": VERSION,
        "authorization_sha256": sha(args.authorization), "config_sha256": sha(args.config),
        "started_unix": time.time(), "status": "STARTING_NOT_COMPLETE"})
    try:
        run_training(args, cfg, code, examples, data, initial, storage, base_files, start)
    except BaseException as exc:
        if not (args.output/"failure.json").exists():
            write_json(args.output/"failure.json", {"error": repr(exc), "optimizer_updates": 0,
                "phase": "BEFORE_OPTIMIZER_LOOP", "wall_seconds": time.monotonic()-start,
                "no_resume_or_retry_authorized": True})
        raise


def run_training(args, cfg, code, examples, data, initial, storage, base_files, start):
    def budget():
        if time.monotonic()-start >= cfg["max_wall_seconds"]: raise TimeoutError("No implicit continuation beyond2700s")
        storage.check()
    import torch, transformers, peft
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(41); np.random.seed(41); random.seed(41)
    budget(); model, processor = load_model(BASE, adapter=initial, train=True, cfg=cfg); budget()
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if (len(names) != 372 or sum(p.numel() for p in params) != 16819200 or
            any("lora_" not in n or "language_model" not in n for n in names)):
        raise RuntimeError("Only language LoRA may be trainable")
    identity = {"protocol": VERSION, "code_commit": code, "config": cfg, "config_sha256": sha(args.config),
        **data, "initial_adapter_sha256": INITIAL_ADAPTER_SHA, "initial_result_sha256": INITIAL_RESULT_SHA,
        "warmstart_loaded": True, "base_model": BASE, **base_files, "physical_gpu": 3,
        "executor_digest": EXECUTOR_DIGEST, "carry_duration_core_commit": CORE_COMMIT, **execution_metadata(CARRY_PROFILE),
        "authorization_sha256": sha(args.authorization), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "trainable_parameters": sum(p.numel() for p in params), "trainable_parameter_names": names,
        "storage": storage.check(), "eos": eos_id(processor), "started_unix": time.time()}
    write_json(args.output/"identity.json", identity)
    cache = {}
    for example in examples:
        budget(); encoded = encode(processor, example["query"], checked_images(example["state"]), supervised=True)
        check_mask(processor, example["query"], encoded); cache[example["id"]] = encoded
    def batch(selected):
        return {k: v.to("cuda") for k, v in collate([cache[r["id"]] for r in selected], processor.tokenizer.pad_token_id).items()}
    # Compare native/custom CE for each category, including real two-class EOS
    # masks. Mixed queries are not concatenated into the old motion prompt.
    gates = []; model.eval()
    for category in ("motion", "CONTINUE", "REQUEST_VERIFY"):
        ordered = sorted([e for e in examples if e["category"] == category], key=lambda e: cache[e["id"]]["input_ids"].shape[1])
        selected = [ordered[0], ordered[-1]]; gb = batch(selected); budget()
        with torch.no_grad():
            native = model(**gb, use_cache=False).loss; custom = supervised_loss(model, gb)
        if not torch.isfinite(native) or not torch.allclose(native, custom, rtol=1e-5, atol=1e-4):
            raise RuntimeError("Real native/custom response CE disagreement")
        gates.append({"category": category, "native": float(native), "custom": float(custom),
            "ids": [e["id"] for e in selected], "left_padding": bool((gb["attention_mask"] == 0).any())})
        del gb, native, custom
    write_json(args.output/"native_loss_gate.json", {"passed": True, "all73_response_only_EOS": True,
        "categories": gates, "rtol": 1e-5, "atol": 1e-4})
    model.train(); before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], betas=(.9, .95), weight_decay=.01)
    sampler = BalancedSampler(examples); checkpoints = []; steps = 0
    def save(step):
        budget()
        from native_storage import tree_bytes, NVME
        bound = sum(p.numel()*max(4, p.element_size()) for p in params)+2*1024**2
        from native_reference_profile import ROOT
        if (tree_bytes(args.output, NVME.stat().st_dev)+bound >= cfg["max_artifact_MiB"]*1024**2 or
                tree_bytes(ROOT, NVME.stat().st_dev)+bound >= 6*1024**3):
            raise RuntimeError("Insufficient checkpoint capacity BEFORE writing")
        folder = args.output/f"adapter_{step:04d}"; model.save_pretrained(folder, safe_serialization=True); budget()
        checkpoints.append({"step": step, "path": str(folder), "adapter_sha256": sha(folder/"adapter_model.safetensors"),
            "adapter_config_sha256": sha(folder/"adapter_config.json")})
        write_json(args.output/"checkpoints.json", checkpoints); return folder
    try:
        with (args.output/"steps.jsonl").open("x", buffering=1) as ledger:
            for step in range(1, 121):
                budget(); optimizer.zero_grad(set_to_none=True); losses = []; drawn = []; started = time.monotonic()
                for selected in sampler.update():
                    budget(); loss = supervised_loss(model, batch(selected))
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite response loss")
                    (loss/4).backward(); losses.append(float(loss.detach()))
                    drawn.extend({"id": e["id"], "category": e["category"], "group": e["group"]} for e in selected)
                grad = torch.nn.utils.clip_grad_norm_(params, 1.)
                if not torch.isfinite(grad) or float(grad) <= 0: raise RuntimeError("Nonfinite/zero LoRA gradient")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen vision/projector/base received a gradient")
                optimizer.step(); steps = step
                ledger.write(json.dumps({"step": step, "loss": float(np.mean(losses)), "micro_losses": losses,
                    "motion_loss": float(np.mean(losses[:2])), "status_loss": float(np.mean(losses[2:])),
                    "gradient_norm": float(grad), "draws": drawn, "elapsed": time.monotonic()-start,
                    "seconds": time.monotonic()-started}, allow_nan=False)+"\n")
                if step <= 2 or step % 10 == 0: print(json.dumps({"step": step, "loss": float(np.mean(losses)), "elapsed": time.monotonic()-start}), flush=True)
                if step == 2:
                    changed = sum(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.requires_grad)
                    if not changed: raise RuntimeError("Warmstarted adapter did not update")
                    del before
                    folder = save(step); model.eval(); restored, _ = load_model(BASE, adapter=folder)
                    errors = []
                    for category in ("motion", "CONTINUE", "REQUEST_VERIFY"):
                        example = next(e for e in examples if e["category"] == category)
                        row = query_row(example["query"]["actor"], example["query"]["query_kind"])
                        x = encode(processor, row, checked_images(example["state"]), supervised=False)
                        inputs = {k: v.to("cuda") for k, v in x.items()}; budget()
                        with torch.no_grad():
                            first = model(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
                            second = restored(**inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
                        if not torch.allclose(first, second, rtol=1e-5, atol=1e-4): raise RuntimeError("Save/reload logits differ")
                        errors.append({"category": category, "max_logit_error": float((first-second).abs().max())})
                        del first, second, inputs, x
                    write_json(args.output/"restore_gate.json", {"passed": True, "changed_tensors": changed,
                        "queries": errors, "rtol": 1e-5, "atol": 1e-4, "updates_included_in_total": 2})
                    del restored; torch.cuda.empty_cache(); model.train(); budget()
            save(120); budget()
        write_json(args.output/"result.json", {"status": "COMPLETE", "optimizer_updates": steps,
            "wall_seconds": time.monotonic()-start, "protocol": VERSION, "dataset_sha256": DATA_SHA,
            "identity_sha256": sha(args.output/"identity.json"), "checkpoints": checkpoints,
            "initial_adapter_sha256": INITIAL_ADAPTER_SHA, "policy_effect_evaluated": False})
    except BaseException as exc:
        write_json(args.output/"failure.json", {"error": repr(exc), "optimizer_updates": steps,
            "wall_seconds": time.monotonic()-start, "no_resume_or_retry_authorized": True}); raise


if __name__ == "__main__": main()
