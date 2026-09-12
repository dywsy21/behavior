"""Audited real 4-GPU smoke/train launcher. Does not launch canceled baselines."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


INIT_RUN = Path("/mnt/sdc1/robodojo/outputs/g05/r1pro_memlite_ar_v9/behavior5_memlite_ar_v9_train_20260906T031051Z")
SIDECAR = Path("/mnt/sdc1/robodojo/datasets/memlite_annotations_task0_4_causal_v9_final_20260906/meta/memlite_annotations.parquet")
SIDECAR_SHA256 = "a5f1026110cac3ed57682d8177d2dabcb2f30d6572791ebb27c2800100f5c5d2"
STATS = Path("/mnt/sdc1/robodojo/stats/g05/behavior5_r1pro_trainonly_taskstrat5_stats_v2.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def command(repo, mode):
    smoke = mode == "smoke"
    steps = 5 if smoke else 5000
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "scripts/finetune.py",
        "task=r1pro_memlite_ar_v10", f"data.embodiment_datasets.galaxea_r1pro.memlite_sidecar={SIDECAR}",
        f"tokenizer.vq_config.ckpt_dir={INIT_RUN / 'action_tokenizer.pt'}",
        f"model.max_steps={steps}", "model.max_epochs=null", "resume_ckpt=null",
        "model.batch_size=8", "model.grad_accumulation_steps=1", "model.num_workers=8", "model.prefetch_factor=2",
        "batch_size_val=8", "model.learning_rate=0.00002", f"model.warmup_steps={1 if smoke else 100}",
        "model.lr_min_ratio=0.1", f"checkpointing_steps={5 if smoke else 500}", f"eval_steps={1 if smoke else 100}",
        "logger.log_steps=1" if smoke else "logger.log_steps=10", "logger.mode=offline", "seed=17"]


def verify_manual_payload(data):
    reviewed = dict(data)
    expected = reviewed.pop("review_payload_sha256")
    review = reviewed.pop("manual_review")
    reviewed["ready_for_training"] = False
    actual = hashlib.sha256(json.dumps(reviewed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != expected or review.get("payload_sha256") != expected:
        raise ValueError("Manual review no longer matches the exact recovery payload")


def verify_sources(manifest_path, sampling_path):
    data = json.loads(manifest_path.read_text())
    index = json.loads(sampling_path.read_text())
    if not data.get("ready_for_training") or not index.get("ready_for_training"):
        raise ValueError("Recovery data / motion index are not approved for training")
    if data.get("manual_review", {}).get("status") != "approved_by_primary_manual_inspection":
        raise ValueError("Primary manual review is required")
    verify_manual_payload(data)
    if index["view_identity"]["recovery"] != {"manifest": str(manifest_path), "sha256": sha(manifest_path), "split": "train"}:
        raise ValueError("Training sampler does not match the approved recovery manifest")
    if sha(SIDECAR) != SIDECAR_SHA256:
        raise ValueError("Original reviewed sidecar changed")
    for trajectory in data["trajectories"]:
        root = Path(trajectory["path"])
        if sha(trajectory["execution_source"]) != trajectory["execution_source_sha256"]:
            raise ValueError("Collector execution source changed after review")
        for relative, expected in trajectory["hashes"].items():
            if sha(root / relative) != expected:
                raise ValueError(f"Reviewed recovery content changed: {root / relative}")
    return {"recovery_manifest_sha256": sha(manifest_path), "sampling_index_sha256": sha(sampling_path),
            "sidecar_sha256": SIDECAR_SHA256, "train_only_stats_sha256": sha(STATS),
            "original_codec_sha256": sha(INIT_RUN / "action_tokenizer.pt")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("smoke", "train"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sampling-index", type=Path, required=True)
    ap.add_argument("--smoke-receipt", type=Path)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    manifest = args.manifest.resolve(strict=True)
    sampling = args.sampling_index.resolve(strict=True)
    hashes = verify_sources(manifest, sampling)
    source_hashes = {str(p.relative_to(repo)): sha(p) for directory in ("src", "configs", "scripts")
                     for p in sorted((repo / directory).rglob("*")) if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}}
    if args.mode == "train":
        if not args.smoke_receipt:
            raise ValueError("Formal training requires a completed real optimizer smoke receipt")
        smoke = json.loads(args.smoke_receipt.read_text())
        if smoke.get("exit_code") != 0 or smoke.get("verified_checkpoint_step") != 5:
            raise ValueError("Smoke did not complete five real optimizer updates")
        if smoke["hashes"] != hashes or smoke["source_hashes"] != source_hashes:
            raise ValueError("Code/data changed after smoke; run a new smoke on the actual recipe")
    init = INIT_RUN / "checkpoints/step_5000.pt"
    if not init.is_file():
        raise FileNotFoundError(init)
    lock = open("/mnt/sdc1/robodojo/behavior_dev/memlite_v10_training.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    usage = [list(map(int, line.split(","))) for line in gpu.strip().splitlines()]
    if len(usage) != 4 or any(memory > 1000 or utilization > 0 for memory, utilization in usage):
        raise RuntimeError("All four GPUs must be idle before launching; no existing jobs are stopped")
    control = Path(tempfile.mkdtemp(prefix="memlite_v10_train_control.", dir="/mnt/sdc1/robodojo/behavior_dev"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"behavior5_memlite_ar_v10_{args.mode}_{stamp}"
    env = {**os.environ, "EXP_NAME": name, "G05_OUTPUT_DIR": "/mnt/sdc1/robodojo/outputs/g05",
        "MEMLITE_INIT_CKPT": str(init), "MEMLITE_TASK_EVENT_INDEX": str(sampling),
        "MEMLITE_RECOVERY_MANIFEST": str(manifest), "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "WANDB_MODE": "offline", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1", "DRY_RUN": "0", "PYTHONPATH": str(repo / "src") + ":" + str(repo)}
    for debug_override in ("MAX_EMBODIMENTS", "MAX_DATASETS", "OVERRIDE_DATASET"):
        env.pop(debug_override, None)
    run_dir = Path(env["G05_OUTPUT_DIR"]) / "r1pro_memlite_ar_v10" / name
    if run_dir.exists():
        raise FileExistsError(run_dir)
    receipt = {"mode": args.mode, "launcher_pid": os.getpid(), "run_dir": str(run_dir), "control_dir": str(control),
        "argv": command(repo, args.mode), "hashes": hashes, "source_hashes": source_hashes,
        "init_checkpoint": str(init), "fresh_optimizer_scheduler": True, "dry_run": False,
        "exit_code": None, "verified_checkpoint_step": None}
    def publish():
        temp = control / "receipt.tmp"
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        temp.replace(control / "receipt.json")
    with (control / "train.log").open("wb") as log:
        process = subprocess.Popen(receipt["argv"], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["torchrun_pid"] = process.pid
        publish()
        print(json.dumps({"control_dir": str(control), "run_dir": str(run_dir), "torchrun_pid": process.pid}), flush=True)
        receipt["exit_code"] = process.wait()
    if receipt["exit_code"] == 0:
        try:
            import torch
            expected_step = 5 if args.mode == "smoke" else 5000
            checkpoint = run_dir / "checkpoints" / f"step_{expected_step}.pt"
            state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
            if state["step"] != expected_step or not state.get("optimizer_state_dict") or not state.get("scheduler_state_dict"):
                raise RuntimeError("Checkpoint does not prove the completed optimizer state")
            if (run_dir / "last.pt").resolve() != checkpoint.resolve():
                raise RuntimeError("last.pt does not point to the finalized checkpoint")
            receipt["verified_checkpoint_step"] = expected_step
            receipt["checkpoint"] = str(checkpoint)
        except Exception as error:
            receipt["exit_code"] = 1
            receipt["verification_error"] = f"{type(error).__name__}: {error}"
    publish()
    print(json.dumps({k: receipt[k] for k in ("mode", "exit_code", "verified_checkpoint_step", "run_dir", "control_dir")}), flush=True)
    sys.exit(receipt["exit_code"])


if __name__ == "__main__":
    main()
