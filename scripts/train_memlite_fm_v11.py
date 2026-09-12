"""Gated, durable four-GPU FM adapter smoke/train launcher; no baseline campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from memlite_fm_v11_common import environment, sha, source_hashes, TASK5_RUN, HIGH_RUN, RECOVERY, SIDECAR
from train_memlite_v10 import verify_sources


def command(mode):
    smoke = mode == "smoke"
    steps = 5 if smoke else 5000
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        "scripts/finetune.py", "task=r1pro_memlite_fm_v11",
        # Hydra's saved YAML retains interpolations. Persist the actual
        # provenance/serving paths, not ephemeral launcher environment values.
        f"model.pretrained_ckpt={TASK5_RUN / 'checkpoints/step_5000.pt'}",
        f"memlite_fm.controller_init={TASK5_RUN / 'checkpoints/step_5000.pt'}",
        f"memlite_fm.high_checkpoint={HIGH_RUN / 'checkpoints/step_5000.pt'}",
        f"memlite_recovery.manifest={RECOVERY / 'recovery_manifest.json'}",
        f"data.memlite_branch_sampling.sampling_index_path={RECOVERY / 'motion_recovery_index_v2.json'}",
        f"data.embodiment_datasets.galaxea_r1pro.memlite_sidecar={SIDECAR}",
        f"tokenizer.vq_config.ckpt_dir={TASK5_RUN / 'action_tokenizer.pt'}",
        "model.model_arch.attn_implementation=sdpa", f"model.max_steps={steps}", "model.max_epochs=null",
        "resume_ckpt=null", "model.batch_size=8", "model.grad_accumulation_steps=1", "batch_size_val=8",
        "model.num_workers=8", "model.prefetch_factor=2", "model.learning_rate=0.0001",
        f"model.warmup_steps={1 if smoke else 100}", "model.lr_min_ratio=0.1",
        f"checkpointing_steps={5 if smoke else 500}", f"eval_steps={1 if smoke else 100}",
        f"logger.log_steps={1 if smoke else 10}", "logger.mode=offline", "seed=17"]


def verify_gates(data_dir, model_receipt, review_path, intent_path, repo):
    manifest = data_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    model = json.loads(model_receipt.read_text())
    review = json.loads(review_path.read_text())
    intent = json.loads(intent_path.read_text())
    if not intent.get("passed") or intent.get("sidecar_sha256") != sha(SIDECAR) or intent.get("recovery_manifest_sha256") != sha(RECOVERY / "recovery_manifest.json"):
        raise ValueError("Full intent text-context audit is missing or stale")
    expected = sha(manifest)
    if not data.get("passed") or not data.get("source_rank_disjoint") or data.get("steps_audited") != 5000:
        raise ValueError("Complete 5000-step source/split/sampling audit required")
    if not model.get("passed") or model.get("data_manifest_sha256") != expected:
        raise ValueError("Real model gate and exact source manifest disagree")
    if not model.get("frozen_parameters_unchanged_after_updates") or model.get("post_smoke_disabled_max_abs_diff") != 0:
        raise ValueError("Preserved FM controller parity failed")
    actual = {k: v for k, v in source_hashes(repo).items() if k.startswith(("src/", "configs/"))}
    if model.get("model_sources") != actual:
        raise ValueError("Model/config changed after real model gate; repeat it")
    if review.get("status") != "approved_by_primary_manual_inspection" or review.get("manifest_sha256") != expected:
        raise ValueError("Primary manual inspection of this exact FM view is required")
    by_id = {r["id"]: r for r in data["rows"]}
    notes = review.get("reviewed_samples", [])
    ids = [r["id"] for r in notes]
    if len(ids) < 60 or len(ids) != len(set(ids)) or any(i not in by_id for i in ids):
        raise ValueError("Need at least 60 distinct documented real sample inspections")
    if any(not str(r.get("note", "")).strip() for r in notes):
        raise ValueError("Every inspected sample needs an actual review note")
    if {by_id[i]["task_id"] for i in ids} != set(range(5)) or {by_id[i]["split"] for i in ids} != {"train", "eval"}:
        raise ValueError("Manual review must cover all tasks and both splits")
    for row in data["rows"]:
        if sha(data_dir / row["file"]) != row["sample_sha256"]:
            raise ValueError("Reviewed FM sample changed")
    hashes = verify_sources(RECOVERY / "recovery_manifest.json", RECOVERY / "motion_recovery_index_v2.json")
    hashes.update(data_gate=expected, model_gate=sha(model_receipt), manual_review=sha(review_path), intent_gate=sha(intent_path),
                  fm_init=sha(TASK5_RUN / "checkpoints/step_5000.pt"),
                  frozen_high=sha(HIGH_RUN / "checkpoints/step_5000.pt"))
    return hashes


def compare_delivery(delivered, planned, resolve):
    """Check actual loader identities against exact sampler draws and source rows."""
    if len(delivered) != len(planned):
        raise ValueError("Delivered source receipt count differs from planned optimizer updates")
    for step, (receipt, indices) in enumerate(zip(delivered, planned)):
        if receipt["step"] != step+1 or receipt["batch_index"] != step:
            raise ValueError("Delivered optimizer step/batch cursor mismatch")
        samples = receipt["samples"]
        if [s["requested_index"] for s in samples] != indices:
            raise ValueError("Actual loader draws differ from the audited sampler schedule")
        for sample, index in zip(samples, indices):
            kind, raw = resolve(index)
            if sample["branch"] != "low" or sample["source_kind"] != kind or sample["actual_raw_index"] != raw or not sample.get("locator"):
                raise ValueError("Loader silently substituted or misidentified a source row")


def verify_delivered_sources(run_dir, step, data_dir):
    from memlite_fm_v11_common import config
    from g05.utils.data.processor_utils import build_processors, instantiate_dataset
    from g05.data.memlite_recovery_dataset import RecoveryAugmentedDataset
    from g05.utils.common.task_event_sampler import build_memlite_train_sampler
    from omegaconf import OmegaConf
    cfg = config()
    base = instantiate_dataset(cfg, is_training_set=True)
    data = RecoveryAugmentedDataset(base, RECOVERY / "recovery_manifest.json", build_processors(cfg))
    def resolve(index):
        if index >= data.base_length:
            return "recovery", index
        child, local = base._resolve_index(index)
        return "original", int(base.datasets[child]._map_active_index(local))
    preflight = json.loads((data_dir / "manifest.json").read_text())
    owned = []
    for rank in range(4):
        sampler = build_memlite_train_sampler(data,
            sampling_config=OmegaConf.to_container(cfg.data.memlite_branch_sampling, resolve=True),
            batch_size=8, num_replicas=4, rank=rank, seed=17, high_fraction=0.)
        planned = list(itertools.islice(sampler, step))
        if planned[:5] != preflight["first_five_batches"][rank]:
            raise ValueError("Current sampler prefix differs from the reviewed 5000-step plan")
        delivered = [json.loads(line) for line in (run_dir/f"sample_receipts_rank{rank}.jsonl").read_text().splitlines()]
        compare_delivery(delivered, planned, resolve)
        owned.append({i for batch in planned for i in batch})
    if any(not owned[a].isdisjoint(owned[b]) for a in range(4) for b in range(a)):
        raise ValueError("Delivered DDP source ownership overlaps")
    return {"steps_per_rank": step, "samples": step*4*8, "exact_draws_and_raw_sources": True, "rank_disjoint": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("smoke", "train"))
    ap.add_argument("--data-preflight", type=Path, required=True)
    ap.add_argument("--model-preflight", type=Path, required=True)
    ap.add_argument("--manual-review", type=Path, required=True)
    ap.add_argument("--intent-preflight", type=Path, required=True)
    ap.add_argument("--smoke-receipt", type=Path)
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--control", type=Path)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    control_root = Path("/mnt/sdc1/robodojo/behavior_dev")
    control = args.control or Path(tempfile.mkdtemp(prefix="memlite_fm_v11_control.", dir=control_root))
    if control.resolve().parent != control_root.resolve() or not control.name.startswith("memlite_fm_v11_control."):
        raise ValueError("Invalid explicit run-control directory")
    if args.detach:
        child = [sys.executable, __file__, args.mode, "--data-preflight", str(args.data_preflight.resolve()),
                 "--model-preflight", str(args.model_preflight.resolve()), "--manual-review", str(args.manual_review.resolve()),
                 "--intent-preflight", str(args.intent_preflight.resolve()), "--control", str(control)]
        if args.smoke_receipt:
            child.extend(["--smoke-receipt", str(args.smoke_receipt.resolve())])
        with (control / "supervisor.log").open("x") as log:
            process = subprocess.Popen(child, cwd=repo, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"control_dir": str(control), "supervisor_pid": process.pid}), flush=True)
        return
    receipt = {"mode": args.mode, "control_dir": str(control), "launcher_pid": os.getpid(),
               "state": "verifying", "exit_code": None, "verified_checkpoint_step": None}
    def publish():
        temp = control / "receipt.tmp"
        temp.write_text(json.dumps(receipt, indent=2)+"\n")
        temp.replace(control / "receipt.json")
    publish()
    try:
        lock = open(control_root / "memlite_fm_v11_training.lock", "a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = verify_gates(args.data_preflight, args.model_preflight, args.manual_review, args.intent_preflight, repo)
        sources = source_hashes(repo)
        if args.mode == "train":
            if not args.smoke_receipt:
                raise ValueError("Formal training requires a completed 4-GPU optimizer smoke")
            smoke = json.loads(args.smoke_receipt.read_text())
            if smoke.get("exit_code") != 0 or smoke.get("verified_checkpoint_step") != 5:
                raise ValueError("Smoke optimizer/checkpoint did not finish")
            if smoke.get("hashes") != hashes or smoke.get("source_hashes") != sources:
                raise ValueError("Source or data changed since smoke; rerun smoke")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        usage = [list(map(int, line.split(","))) for line in gpu.strip().splitlines()]
        if len(usage) != 4 or any(memory > 1000 or utilization > 0 for memory, utilization in usage):
            raise RuntimeError("All four GPUs must be idle; no existing job is stopped")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"behavior5_memlite_fm_v11_{args.mode}_{stamp}"
        env = {**os.environ, **environment(), "EXP_NAME": name, "CUDA_VISIBLE_DEVICES": "0,1,2,3",
               "WANDB_MODE": "offline", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
               "PYTHONUNBUFFERED": "1", "DRY_RUN": "0", "PYTHONPATH": str(repo / "src") + ":" + str(repo)}
        for key in ("MAX_EMBODIMENTS", "MAX_DATASETS", "OVERRIDE_DATASET"):
            env.pop(key, None)
        run_dir = Path(env["G05_OUTPUT_DIR"]) / "r1pro_memlite_fm_v11" / name
        if run_dir.exists():
            raise FileExistsError(run_dir)
        receipt.update(state="starting", argv=command(args.mode), run_dir=str(run_dir),
                       hashes=hashes, source_hashes=sources, fresh_optimizer_scheduler=True,
                       trainable_component="fm_intent_adapter", high_level_frozen=True)
        with (control / "train.log").open("x") as log:
            process = subprocess.Popen(receipt["argv"], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt.update(state="running", torchrun_pid=process.pid)
            publish()
            receipt["exit_code"] = process.wait()
        if receipt["exit_code"] == 0:
            import torch
            step = 5 if args.mode == "smoke" else 5000
            checkpoint = run_dir / "checkpoints" / f"step_{step}.pt"
            state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
            if state["step"] != step or not state.get("optimizer_state_dict") or not state.get("scheduler_state_dict"):
                raise RuntimeError("Checkpoint does not prove the requested optimizer updates")
            if (run_dir / "last.pt").resolve() != checkpoint.resolve():
                raise RuntimeError("last.pt must reference the finalized checkpoint")
            audit = json.loads((run_dir / "parameter_audit.json").read_text())
            if audit["trainable_parameter_numel_after_freeze"] >= 10_000_000:
                raise RuntimeError("Unexpected trainable backbone; adapter-only run must be small")
            model_gate = json.loads(args.model_preflight.read_text())
            if audit["trainable_parameter_numel_after_freeze"] != model_gate["trainable_numel"]:
                raise RuntimeError("Optimizer trainable count differs from verified model")
            receipt.update(delivery_audit=verify_delivered_sources(run_dir, step, args.data_preflight))
            receipt.update(state="complete", verified_checkpoint_step=step, checkpoint=str(checkpoint))
        else:
            receipt["state"] = "failed"
    except Exception as error:
        receipt.update(state="failed", exit_code=1, error=f"{type(error).__name__}: {error}")
        publish()
        raise
    publish()
    print(json.dumps({key: receipt.get(key) for key in ("state", "exit_code", "verified_checkpoint_step", "run_dir", "control_dir")}), flush=True)
    sys.exit(receipt["exit_code"])


if __name__ == "__main__":
    main()
