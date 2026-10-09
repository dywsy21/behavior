"""CPU-only pinned parent/data/environment audit on the shared A800 disk.

No installs, optimizer, simulator, GPU context, credential contents, or edits
to existing runs. A fresh output directory is required for every invocation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from recovery_corpus import file_sha, group_key  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = args.root.resolve(strict=True)
    source = json.loads(args.source_manifest.read_text())
    release = root / "datasets/memlite-stage1-20260930-v4"
    accepted = json.loads((release / "acceptance.json").read_text())
    if (accepted["status"] != "ACCEPTED" or not all(accepted["gates"].values())
            or file_sha(release / "manifest.json") != accepted["manifest_sha256"]):
        raise ValueError("Original expert release no longer matches acceptance")
    protected, seen, split_counts = set(), {}, Counter()
    with (release / "episodes.jsonl").open() as stream:
        for line in stream:
            ep = json.loads(line)
            split = ep["split"]
            if split not in {"train", "eval"}:
                raise ValueError("Unexpected source split")
            key = group_key(ep["task_name"], ep["row"]["task_instance_id"])
            if key in seen and seen[key] != split:
                raise ValueError("Original task-instance spans expert splits")
            seen[key] = split
            split_counts[split] += 1
            if split == "eval":
                protected.add(key)
    for group in source["groups"]:
        for task in group["tasks"]:
            frozen = {group_key(task["task"], i) for i in task["sft_heldout_excluded"]}
            actual = {key for key in protected if key.rsplit(":", 1)[0] == group_key(task["task"], 0).rsplit(":", 1)[0]}
            if not actual <= frozen:
                raise ValueError("RL manifest missed an original expert eval instance")
            train = {group_key(task["task"], i) for i in task["train_instances"]}
            if train & protected:
                raise ValueError("RL TRAIN contains an expert heldout instance")
    protected_receipt = dict(schema="protected_source_groups_v1", groups=sorted(protected),
        episodes_sha256=file_sha(release / "episodes.jsonl"), manifest_sha256=accepted["manifest_sha256"],
        source_manifest_sha256=file_sha(args.source_manifest), split_counts=dict(split_counts))
    (args.output / "protected-groups.json").write_text(json.dumps(protected_receipt, indent=2) + "\n")
    parents = {}
    for branch, step in (("high", 48045), ("low", 98414)):
        directory = root / f"runs/memlite_stage1_{branch}_100task_v1/checkpoints"
        latest = json.loads((directory / "latest.json").read_text())
        path = directory / latest["path"]
        expected = source["model"]["checkpoints"][branch + "/" + path.name]["sha256"]
        actual = file_sha(path)
        if actual != expected or actual != latest["sha256"] or latest["state"]["step"] != step:
            raise ValueError("Stage1 parent checkpoint mismatch: " + branch)
        parents[branch] = dict(path=str(path), bytes=path.stat().st_size, sha256=actual, step=step)
        print(json.dumps(dict(verified_parent=branch, **parents[branch])), flush=True)
    stats = root / "manifests/memlite-stage1-v4-action-bounds/stats.json"
    stats_sha = file_sha(stats)
    if stats_sha != accepted["stats_sha256"]:
        raise ValueError("Inherited TRAIN-only action bounds mismatch")
    import torch
    import g05
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.dram,ecc.errors.uncorrected.aggregate.dram", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True).stdout
    result = dict(schema="recovery_a800_preflight_v1", hostname=socket.gethostname(),
        unix_time=time.time(), seconds=time.monotonic()-started, parents=parents,
        original_release=str(release), acceptance_sha256=file_sha(release / "acceptance.json"),
        protected_receipt_sha256=file_sha(args.output / "protected-groups.json"),
        stats_path=str(stats), stats_sha256=stats_sha, python=sys.executable,
        torch_version=torch.__version__, cuda_build=torch.version.cuda, g05_import=str(g05.__file__),
        cuda_context_created=torch.cuda.is_initialized(), free_bytes=shutil.disk_usage(root).free,
        gpu_inventory=gpu, existing_compute_processes=processes,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        wandb_key_mode=oct((root / "secrets/stage1-wandb.key").stat().st_mode & 0o777),
        training_authorized_by_receipt=False, semantic_training_data_ready=False)
    if result["cuda_context_created"]:
        raise RuntimeError("CPU audit unexpectedly initialized a CUDA context")
    with (args.output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
